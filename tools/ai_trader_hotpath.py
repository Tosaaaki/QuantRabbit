#!/usr/bin/env python3
"""Bounded preflight and acceptance handoff for the AI trader hot path.

This command stops before model invocation.  For a live profile it starts an
independent, fail-closed acceptor before returning the model-neutral prepared
run, so broker acceptance does not depend on the model task reaching a later
shell step.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from quant_rabbit.ai_trading_runtime import (
    AIRuntimeError,
    PreparedRun,
    finish_hotpath_lease_if_owned,
    prepare_run,
)
from quant_rabbit.policy_snapshot import (
    PolicyBinding,
    PolicySnapshotError,
    load_and_verify_policy_snapshot,
)
from quant_rabbit.runtime_capacity import (
    CapacityPolicy,
    CapacityStatus,
    RootQuota,
    evaluate_capacity,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ai_trading_runtime.json"
MAX_OUTPUT_BYTES = 16 * 1024
MAX_BLOCKERS = 32
MAX_TEXT_CHARS = 512
HASH_CHUNK_BYTES = 1024 * 1024
ACCEPTOR_HANDSHAKE_SECONDS = 2.0


@dataclass(frozen=True)
class HotPathOptions:
    config_path: Path
    profile: str
    repo_root: Path
    state_root: Path | None
    policy_snapshot_path: Path
    project_key: str
    broker_account_id: str
    environment: str
    revocation_epoch: int
    required_source_pages: tuple[str, ...]
    lock_path: Path
    capacity_filesystem: Path | None
    low_free_bytes: int
    high_free_bytes: int
    state_quota_pressure_bytes: int
    state_quota_block_bytes: int
    auto_accept: bool = False
    acceptor_poll_seconds: float = 0.1


def run_hotpath(
    options: HotPathOptions,
    *,
    now: datetime | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run one preflight cycle and return ``(exit_code, compact_payload)``."""

    try:
        with _singleflight(options.lock_path):
            return _run_locked(options, now=_utc_now(now))
    except BlockingIOError:
        return 75, {"status": "LOCKED", "code": "HOTPATH_ALREADY_RUNNING"}
    except OSError as exc:
        return 2, _compact_payload(
            {
                "status": "BLOCKED_LOCK",
                "code": "HOTPATH_LOCK_UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run_locked(options: HotPathOptions, *, now: datetime) -> tuple[int, dict[str, Any]]:
    try:
        policy = load_and_verify_policy_snapshot(
            options.policy_snapshot_path,
            binding=PolicyBinding(
                project_key=_required_text(options.project_key, "project_key"),
                broker_account_id=_required_text(options.broker_account_id, "broker_account_id"),
                environment=_required_text(options.environment, "environment"),
                revocation_epoch=_nonnegative_int(options.revocation_epoch, "revocation_epoch"),
            ),
            now=now,
            required_source_pages=options.required_source_pages,
        )
    except (PolicySnapshotError, ValueError) as exc:
        code = getattr(exc, "code", "POLICY_BINDING_INVALID")
        return 2, _compact_payload(
            {"status": "BLOCKED_POLICY", "code": code, "error": str(exc)}
        )

    try:
        config = _load_json_object(options.config_path, "runtime config")
        state_root = _resolve_state_root(config, options.state_root)
        state_root.mkdir(parents=True, exist_ok=True)
        profile_config = _profile_runtime_config(config, options.profile)
        live_profile = str(profile_config.get("sink") or "") == "live_gateway"
        if live_profile and not options.auto_accept:
            return 2, {
                "status": "BLOCKED_ACCEPTOR",
                "code": "LIVE_AUTO_ACCEPT_REQUIRED",
                "profile": options.profile,
            }
        active_lease = _read_active_lease(state_root / "hotpath_lease.json", now=now)
        if active_lease is not None:
            return 75, {
                "status": "LOCKED",
                "code": "HOTPATH_ACTIVE_LEASE",
                "run_id": active_lease["run_id"],
                "expires_at_utc": active_lease["expires_at_utc"],
            }
        capacity = evaluate_capacity(
            CapacityPolicy(
                filesystem_path=options.capacity_filesystem or state_root,
                low_free_bytes=_nonnegative_int(options.low_free_bytes, "low_free_bytes"),
                high_free_bytes=_positive_int(options.high_free_bytes, "high_free_bytes"),
                root_quotas=(
                    RootQuota(
                        name="ai_trader_state",
                        path=state_root,
                        pressure_bytes=_nonnegative_int(
                            options.state_quota_pressure_bytes,
                            "state_quota_pressure_bytes",
                        ),
                        block_bytes=_positive_int(
                            options.state_quota_block_bytes,
                            "state_quota_block_bytes",
                        ),
                    ),
                ),
            )
        )
    except (AIRuntimeError, OSError, TypeError, ValueError) as exc:
        return 2, _compact_payload(
            {
                "status": "BLOCKED_CAPACITY",
                "code": "CAPACITY_PREFLIGHT_INVALID",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    if capacity.status is CapacityStatus.BLOCK:
        return 2, _compact_payload(
            {
                "status": "BLOCKED_CAPACITY",
                "code": "RUNTIME_CAPACITY_BLOCK",
                "capacity": capacity.status.value,
                "free_bytes": capacity.free_bytes,
                "issues": list(capacity.issues),
            }
        )

    try:
        input_digest = _input_digest(
            config=config,
            profile=options.profile,
            repo_root=options.repo_root,
            state_root=state_root,
            policy=policy,
            now=now,
        )
        digest_path = state_root / "hotpath_input.json"
        if _read_trusted_digest(digest_path) == input_digest:
            return 0, {
                "status": "NO_UPDATE",
                "profile": options.profile,
                "input_digest": input_digest,
                "capacity": capacity.status.value,
            }

        prepared = prepare_run(
            config_path=options.config_path,
            profile=options.profile,
            repo_root=options.repo_root,
            state_root=state_root,
            now=now,
        )
        lease = (
            _write_active_lease(
                state_root / "hotpath_lease.json",
                run_id=prepared.run_id,
                input_digest=input_digest,
                now=now,
                ttl_seconds=_positive_int(
                    profile_config.get("decision_max_age_seconds"),
                    "decision_max_age_seconds",
                ),
            )
            if prepared.ready
            else None
        )
        acceptor = None
        if prepared.ready and live_profile:
            try:
                acceptor = _launch_acceptor(
                    options,
                    prepared=prepared,
                    state_root=state_root,
                )
            except (AIRuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
                finish_hotpath_lease_if_owned(
                    state_root,
                    run_id=prepared.run_id,
                    status="FAILED:ACCEPTOR_START_FAILED",
                    now=now,
                )
                return 2, _compact_payload(
                    {
                        "status": "BLOCKED_ACCEPTOR",
                        "code": "ACCEPTOR_START_FAILED",
                        "run_id": prepared.run_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        _write_digest(digest_path, input_digest)
    except (AIRuntimeError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return 2, _compact_payload(
            {
                "status": "BLOCKED_PREPARE",
                "code": getattr(exc, "code", "HOTPATH_PREPARE_FAILED"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    payload = _prepared_payload(
        prepared,
        capacity=capacity.status.value,
        input_digest=input_digest,
        lease=lease,
        acceptor=acceptor,
    )
    return (0 if prepared.ready else 2), _compact_payload(payload)


def _prepared_payload(
    prepared: PreparedRun,
    *,
    capacity: str,
    input_digest: str,
    lease: Mapping[str, Any] | None,
    acceptor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": "READY" if prepared.ready else "BLOCKED_INPUTS",
        "run_id": prepared.run_id,
        "profile": prepared.profile,
        "kind": prepared.kind,
        "manifest_path": str(prepared.manifest_path),
        "candidate_path": str(prepared.candidate_path),
        "blockers": list(prepared.blockers),
        "input_digest": input_digest,
        "capacity": capacity,
        "lease_expires_at_utc": None if lease is None else lease["expires_at_utc"],
        "acceptor": None if acceptor is None else dict(acceptor),
    }


def _read_active_lease(path: Path, *, now: datetime) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise OSError("hot-path lease path is unsafe")
    payload = _load_json_object(path, "hot-path lease")
    material = {key: value for key, value in payload.items() if key != "lease_sha256"}
    if payload.get("lease_sha256") != _canonical_digest(material):
        raise OSError("hot-path lease is tampered")
    if payload.get("status") != "ACTIVE":
        return None
    expires = datetime.fromisoformat(str(payload.get("expires_at_utc") or "").replace("Z", "+00:00"))
    if expires.tzinfo is None:
        raise OSError("hot-path lease expiry is invalid")
    return payload if now < expires.astimezone(timezone.utc) else None


def _write_active_lease(
    path: Path,
    *,
    run_id: str,
    input_digest: str,
    now: datetime,
    ttl_seconds: int,
) -> dict[str, Any]:
    material = {
        "schema_version": 1,
        "status": "ACTIVE",
        "run_id": run_id,
        "input_digest": input_digest,
        "issued_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }
    payload = {**material, "lease_sha256": _canonical_digest(material)}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return payload


def _launch_acceptor(
    options: HotPathOptions,
    *,
    prepared: PreparedRun,
    state_root: Path,
) -> dict[str, Any]:
    manifest = _load_json_object(prepared.manifest_path, "run manifest")
    initial_sha256 = _required_text(
        manifest.get("candidate_template_sha256"),
        "candidate_template_sha256",
    )
    status_path = Path(_required_text(manifest.get("acceptor_status_path"), "acceptor_status_path"))
    expected_status_path = (state_root / "runs" / prepared.run_id / "acceptor_status.json").resolve()
    if status_path.resolve() != expected_status_path:
        raise AIRuntimeError("ACCEPTOR_SCOPE_MISMATCH", "acceptor status path is outside its run")

    command = [
        sys.executable,
        str(options.repo_root / "tools" / "ai_decision_acceptor.py"),
        "--config",
        str(options.config_path),
        "--manifest",
        str(prepared.manifest_path),
        "--candidate",
        str(prepared.candidate_path),
        "--repo-root",
        str(options.repo_root),
        "--state-root",
        str(state_root),
        "--initial-candidate-sha256",
        initial_sha256,
        "--poll-interval-seconds",
        str(options.acceptor_poll_seconds),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(options.repo_root / "src"),
            "QR_AI_PROJECT_KEY": options.project_key,
            "QR_AI_BROKER_ACCOUNT_ID": options.broker_account_id,
            "QR_AI_ENVIRONMENT": options.environment,
            "QR_AI_POLICY_REVOCATION_EPOCH": str(options.revocation_epoch),
            "QR_AI_REQUIRED_POLICY_SOURCE_PAGES": ",".join(options.required_source_pages),
            "QR_AI_ORDER_AUTHORITY": "LIVE",
            "QR_LIVE_ENABLED": "1",
        }
    )
    log_path = expected_status_path.with_name("acceptor.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=options.repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    deadline = time.monotonic() + ACCEPTOR_HANDSHAKE_SECONDS
    while time.monotonic() < deadline:
        if status_path.is_file() and not status_path.is_symlink():
            status = _load_json_object(status_path, "acceptor status")
            if status.get("run_id") == prepared.run_id and status.get("status") == "WAITING_FOR_CANDIDATE":
                return {
                    "pid": process.pid,
                    "status": "WAITING_FOR_CANDIDATE",
                    "status_path": str(status_path),
                    "log_path": str(log_path),
                }
        return_code = process.poll()
        if return_code is not None:
            raise AIRuntimeError(
                "ACCEPTOR_START_FAILED",
                f"acceptor exited before handshake with status {return_code}",
            )
        time.sleep(0.02)

    process.terminate()
    raise AIRuntimeError("ACCEPTOR_HANDSHAKE_TIMEOUT", "acceptor did not confirm readiness")


def _input_digest(
    *,
    config: Mapping[str, Any],
    profile: str,
    repo_root: Path,
    state_root: Path,
    policy: Mapping[str, Any],
    now: datetime,
) -> str:
    profiles = config.get("profiles")
    profile_config = profiles.get(profile) if isinstance(profiles, Mapping) else None
    if not isinstance(profile_config, Mapping):
        raise AIRuntimeError("PROFILE_NOT_FOUND", f"unknown profile: {profile}")
    workers = profile_config.get("workers")
    if not isinstance(workers, Mapping) or not workers:
        raise AIRuntimeError("PROFILE_INVALID", f"profile {profile!r} has no workers")

    sources: list[dict[str, Any]] = []
    for worker, configured_sources in sorted(workers.items(), key=lambda item: str(item[0])):
        if not isinstance(worker, str) or not worker.strip() or not isinstance(configured_sources, list):
            raise AIRuntimeError("PROFILE_INVALID", "workers must map names to source lists")
        for source in configured_sources:
            if not isinstance(source, Mapping):
                raise AIRuntimeError("PROFILE_INVALID", "worker source must be an object")
            raw_path = _required_text(source.get("path"), "source.path")
            max_age = _positive_int(source.get("max_age_seconds"), f"{raw_path}.max_age_seconds")
            path = _source_path(raw_path, repo_root=repo_root, state_root=state_root)
            descriptor: dict[str, Any] = {
                "worker": worker,
                "path": raw_path,
                "required": source.get("required") is True,
                "max_age_seconds": max_age,
                "status": "MISSING",
                "sha256": None,
                "size": None,
                "mtime_ns": None,
            }
            if path.is_file() and not path.is_symlink():
                stat = path.stat()
                modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                descriptor.update(
                    {
                        "status": "READY" if max(0.0, (now - modified).total_seconds()) <= max_age else "STALE",
                        "sha256": _sha256_file(path),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
            sources.append(descriptor)

    material = {
        "profile": profile,
        "profile_config": profile_config,
        "policy_snapshot_sha256": policy.get("snapshot_sha256"),
        "sources": sources,
    }
    return _canonical_digest(material)


def _source_path(raw_path: str, *, repo_root: Path, state_root: Path) -> Path:
    if raw_path.startswith("@state/"):
        return state_root / raw_path.removeprefix("@state/")
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else repo_root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_trusted_digest(path: Path) -> str | None:
    if path.is_symlink():
        raise OSError("hot-path digest path is a symlink")
    if not path.exists():
        return None
    if not path.is_file():
        raise OSError("hot-path digest path is not a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OSError("hot-path digest receipt is malformed") from exc
    if not isinstance(payload, Mapping):
        raise OSError("hot-path digest receipt must be an object")
    digest = payload.get("input_digest")
    receipt = payload.get("receipt_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or not isinstance(receipt, str):
        raise OSError("hot-path digest receipt is invalid")
    expected = _canonical_digest({"input_digest": digest, "schema_version": 1})
    if receipt != expected:
        raise OSError("hot-path digest receipt is tampered")
    return digest


def _write_digest(path: Path, input_digest: str) -> None:
    if path.is_symlink():
        raise OSError("hot-path digest path is a symlink")
    payload = {"input_digest": input_digest, "schema_version": 1}
    payload["receipt_sha256"] = _canonical_digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextmanager
def _singleflight(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise BlockingIOError from exc
            raise
        yield
    finally:
        os.close(descriptor)


def _compact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    compact = {str(key): value for key, value in payload.items()}
    for key in ("error", "manifest_path", "candidate_path"):
        if isinstance(compact.get(key), str):
            compact[key] = compact[key][:MAX_TEXT_CHARS]
    for key in ("blockers", "issues"):
        value = compact.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            compact[key] = [str(item)[:MAX_TEXT_CHARS] for item in value[:MAX_BLOCKERS]]
            if len(value) > MAX_BLOCKERS:
                compact[f"{key}_omitted"] = len(value) - MAX_BLOCKERS
    encoded = _encode_payload(compact)
    if len(encoded) < MAX_OUTPUT_BYTES:
        return compact
    return {
        "status": str(compact.get("status", "BLOCKED_OUTPUT"))[:64],
        "code": "OUTPUT_TRUNCATED",
        "profile": str(compact.get("profile", ""))[:128],
        "run_id": str(compact.get("run_id", ""))[:256],
        "detail_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _encode_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AIRuntimeError("JSON_READ_FAILED", f"unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AIRuntimeError("JSON_READ_FAILED", f"{label} must be a JSON object")
    return value


def _resolve_state_root(config: Mapping[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser()
    environment_override = os.environ.get("QR_AI_TRADER_STATE_ROOT")
    raw = environment_override or str(config.get("state_root") or "")
    if not raw.strip():
        raise AIRuntimeError("CONFIG_INVALID", "state_root is required")
    return Path(raw).expanduser()


def _profile_runtime_config(config: Mapping[str, Any], profile: str) -> dict[str, Any]:
    profiles = config.get("profiles")
    value = profiles.get(profile) if isinstance(profiles, Mapping) else None
    if not isinstance(value, Mapping):
        raise AIRuntimeError("PROFILE_NOT_FOUND", f"unknown profile: {profile}")
    return dict(value)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(timezone.utc)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded offline preflight for the AI trader hot path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", default="intraday")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument(
        "--policy-snapshot",
        type=Path,
        default=os.environ.get("QR_AI_POLICY_SNAPSHOT"),
        required=os.environ.get("QR_AI_POLICY_SNAPSHOT") is None,
    )
    parser.add_argument("--project-key", default=os.environ.get("QR_AI_PROJECT_KEY"), required=os.environ.get("QR_AI_PROJECT_KEY") is None)
    parser.add_argument("--broker-account-id", default=os.environ.get("QR_AI_BROKER_ACCOUNT_ID"), required=os.environ.get("QR_AI_BROKER_ACCOUNT_ID") is None)
    parser.add_argument("--environment", default=os.environ.get("QR_AI_ENVIRONMENT"), required=os.environ.get("QR_AI_ENVIRONMENT") is None)
    parser.add_argument(
        "--revocation-epoch",
        type=int,
        default=os.environ.get("QR_AI_POLICY_REVOCATION_EPOCH"),
        required=os.environ.get("QR_AI_POLICY_REVOCATION_EPOCH") is None,
    )
    parser.add_argument("--required-source-page", action="append", default=[])
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=Path(os.environ.get("QR_AI_HOTPATH_LOCK", Path(tempfile.gettempdir()) / "quant-rabbit-ai-trader-hotpath.lock")),
    )
    parser.add_argument("--capacity-filesystem", type=Path)
    parser.add_argument("--low-free-bytes", type=int, default=_env_int("QR_AI_LOW_FREE_BYTES", 2 * 1024**3))
    parser.add_argument("--high-free-bytes", type=int, default=_env_int("QR_AI_HIGH_FREE_BYTES", 5 * 1024**3))
    parser.add_argument("--state-quota-pressure-bytes", type=int, default=_env_int("QR_AI_STATE_PRESSURE_BYTES", 256 * 1024**2))
    parser.add_argument("--state-quota-block-bytes", type=int, default=_env_int("QR_AI_STATE_BLOCK_BYTES", 512 * 1024**2))
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="start and handshake an independent acceptor (required for live profiles)",
    )
    parser.add_argument(
        "--acceptor-poll-seconds",
        type=float,
        default=0.1,
        help="candidate polling interval for the independent acceptor",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    options = HotPathOptions(
        config_path=args.config,
        profile=args.profile,
        repo_root=args.repo_root,
        state_root=args.state_root,
        policy_snapshot_path=args.policy_snapshot,
        project_key=args.project_key,
        broker_account_id=args.broker_account_id,
        environment=args.environment,
        revocation_epoch=args.revocation_epoch,
        required_source_pages=tuple(args.required_source_page),
        lock_path=args.lock_path,
        capacity_filesystem=args.capacity_filesystem,
        low_free_bytes=args.low_free_bytes,
        high_free_bytes=args.high_free_bytes,
        state_quota_pressure_bytes=args.state_quota_pressure_bytes,
        state_quota_block_bytes=args.state_quota_block_bytes,
        auto_accept=args.auto_accept,
        acceptor_poll_seconds=args.acceptor_poll_seconds,
    )
    code, payload = run_hotpath(options)
    encoded = _encode_payload(_compact_payload(payload))
    if len(encoded) >= MAX_OUTPUT_BYTES:
        encoded = _encode_payload({"status": "BLOCKED_OUTPUT", "code": "OUTPUT_LIMIT_EXCEEDED"})
        code = 2
    print(encoded.decode("utf-8"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

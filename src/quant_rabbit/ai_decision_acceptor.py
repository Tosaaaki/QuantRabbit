from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from quant_rabbit.ai_trading_runtime import (
    AIRuntimeError,
    accept_run,
    finish_hotpath_lease_if_owned,
)


ACCEPTOR_SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"ACCEPTED", "REJECTED", "EXPIRED_NO_DECISION", "FAILED"})


def monitor_candidate(
    *,
    config_path: Path,
    manifest_path: Path,
    candidate_path: Path,
    repo_root: Path,
    state_root: Path,
    initial_candidate_sha256: str,
    poll_interval_seconds: float = 0.1,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Accept the first stable, non-template candidate and always record an outcome.

    This process is deliberately independent of the model task that writes the
    candidate.  Once a changed candidate becomes stable it is attempted exactly
    once; a potentially ambiguous gateway outcome is never retried.
    """

    if poll_interval_seconds < 0.02 or poll_interval_seconds > 1.0:
        raise ValueError("poll_interval_seconds must be between 0.02 and 1.0")
    if len(initial_candidate_sha256) != 64:
        raise ValueError("initial_candidate_sha256 must be a SHA-256 digest")

    clock = now_fn or (lambda: datetime.now(timezone.utc))
    manifest = _load_json_object(manifest_path, "run manifest")
    run_id = _required_text(manifest, "run_id")
    run_dir = (state_root / "runs" / run_id).resolve()
    expected_manifest = run_dir / "manifest.json"
    expected_candidate = run_dir / "candidate.json"
    expected_status = run_dir / "acceptor_status.json"
    if manifest_path.resolve() != expected_manifest or candidate_path.resolve() != expected_candidate:
        raise AIRuntimeError("ACCEPTOR_SCOPE_MISMATCH", "acceptor paths are outside the configured run")
    if Path(_required_text(manifest, "acceptor_status_path")).resolve() != expected_status:
        raise AIRuntimeError("ACCEPTOR_SCOPE_MISMATCH", "manifest acceptor status path is invalid")
    if manifest.get("candidate_template_sha256") != initial_candidate_sha256:
        raise AIRuntimeError("ACCEPTOR_BINDING_MISMATCH", "initial candidate digest differs from manifest")

    prepared_at = _parse_utc(manifest.get("prepared_at_utc"), "prepared_at_utc")
    max_age = _positive_int(manifest.get("decision_max_age_seconds"), "decision_max_age_seconds")
    deadline = prepared_at + timedelta(seconds=max_age)
    sink = str(manifest.get("sink") or "")
    waiting = {
        "schema_version": ACCEPTOR_SCHEMA_VERSION,
        "status": "WAITING_FOR_CANDIDATE",
        "run_id": run_id,
        "pid": os.getpid(),
        "started_at_utc": _utc_now(clock()).isoformat(),
        "deadline_utc": deadline.isoformat(),
        "initial_candidate_sha256": initial_candidate_sha256,
        "attempt_count": 0,
    }
    _atomic_write_json(expected_status, waiting)

    stable_digest: str | None = None
    stable_since = 0.0
    stability_seconds = max(0.1, poll_interval_seconds * 2)
    candidate: dict[str, Any] | None = None
    candidate_sha256: str | None = None
    parse_error: str | None = None

    while True:
        current = _utc_now(clock())
        if current >= deadline:
            outcome = {
                **waiting,
                "status": "EXPIRED_NO_DECISION",
                "completed_at_utc": current.isoformat(),
                "code": "CANDIDATE_NOT_AUTHORED_BEFORE_DEADLINE",
                "candidate_sha256": candidate_sha256,
                "error": parse_error,
                "broker_outcome_unknown": False,
            }
            _atomic_write_json(expected_status, outcome)
            finish_hotpath_lease_if_owned(
                state_root,
                run_id=run_id,
                status="EXPIRED_NO_DECISION",
                now=current,
            )
            return outcome

        raw: bytes | None = None
        try:
            if candidate_path.is_symlink() or not candidate_path.is_file():
                raise OSError("candidate path is not a safe regular file")
            raw = candidate_path.read_bytes()
            raw_digest = hashlib.sha256(raw).hexdigest()
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("candidate must be a JSON object")
            parsed_digest = _canonical_digest(value)
            if parsed_digest == initial_candidate_sha256:
                stable_digest = None
                parse_error = None
            else:
                candidate_sha256 = parsed_digest
                parse_error = None
                if stable_digest != raw_digest:
                    stable_digest = raw_digest
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= stability_seconds:
                    candidate = value
                    break
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raw_digest = hashlib.sha256(raw or b"").hexdigest()
            parse_error = f"{type(exc).__name__}: {exc}"
            candidate_sha256 = raw_digest
            if stable_digest != raw_digest:
                stable_digest = raw_digest
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= stability_seconds:
                outcome = {
                    **waiting,
                    "status": "REJECTED",
                    "completed_at_utc": current.isoformat(),
                    "code": "CANDIDATE_JSON_INVALID",
                    "candidate_sha256": raw_digest,
                    "error": parse_error,
                    "broker_outcome_unknown": False,
                }
                _atomic_write_json(expected_status, outcome)
                finish_hotpath_lease_if_owned(
                    state_root,
                    run_id=run_id,
                    status="REJECTED:CANDIDATE_JSON_INVALID",
                    now=current,
                )
                return outcome
        sleep_fn(poll_interval_seconds)

    detected_at = _utc_now(clock())
    try:
        decided_at = _parse_utc(candidate.get("decided_at_utc"), "decided_at_utc")
        decision_to_accept: float | None = max(0.0, (detected_at - decided_at).total_seconds())
    except AIRuntimeError:
        decision_to_accept = None
    accept_slo = float(manifest.get("candidate_accept_slo_seconds") or 2.0)
    accepting = {
        **waiting,
        "status": "ACCEPTING",
        "candidate_sha256": candidate_sha256,
        "candidate_detected_at_utc": detected_at.isoformat(),
        "decision_to_accept_seconds": (
            None if decision_to_accept is None else round(decision_to_accept, 6)
        ),
        "candidate_accept_slo_seconds": accept_slo,
        "slo_met": decision_to_accept is not None and decision_to_accept <= accept_slo,
        "attempt_count": 1,
    }
    _atomic_write_json(expected_status, accepting)

    try:
        accepted = accept_run(
            config_path=config_path,
            manifest_path=manifest_path,
            candidate_path=candidate_path,
            candidate_payload=candidate,
            repo_root=repo_root,
            state_root=state_root,
            now=detected_at,
        )
    except AIRuntimeError as exc:
        outcome = {
            **accepting,
            "status": "REJECTED",
            "completed_at_utc": _utc_now(clock()).isoformat(),
            "code": exc.code,
            "error": str(exc),
            "broker_outcome_unknown": exc.code == "LIVE_GATEWAY_REJECTED",
        }
        _atomic_write_json(expected_status, outcome)
        finish_hotpath_lease_if_owned(
            state_root,
            run_id=run_id,
            status=f"REJECTED:{exc.code}",
            now=_utc_now(clock()),
        )
        return outcome
    except Exception as exc:
        action = str(candidate.get("action") or "").upper()
        outcome = {
            **accepting,
            "status": "FAILED",
            "completed_at_utc": _utc_now(clock()).isoformat(),
            "code": "ACCEPTOR_UNEXPECTED_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "broker_outcome_unknown": sink == "live_gateway" and action in {"ENTER", "EXIT"},
        }
        _atomic_write_json(expected_status, outcome)
        finish_hotpath_lease_if_owned(
            state_root,
            run_id=run_id,
            status="FAILED:ACCEPTOR_UNEXPECTED_FAILURE",
            now=_utc_now(clock()),
        )
        return outcome

    outcome = {
        **accepting,
        "status": "ACCEPTED",
        "completed_at_utc": _utc_now(clock()).isoformat(),
        "accepted_status": accepted.status,
        "receipt_path": str(accepted.receipt_path),
        "broker_outcome_unknown": False,
    }
    _atomic_write_json(expected_status, outcome)
    return outcome


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AIRuntimeError("JSON_READ_FAILED", f"unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AIRuntimeError("JSON_READ_FAILED", f"{label} must be a JSON object")
    return value


def _required_text(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise AIRuntimeError("FIELD_REQUIRED", f"{field} must be a non-empty string")
    return item.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AIRuntimeError("MANIFEST_INVALID", f"{label} must be a positive integer")
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise AIRuntimeError("TIMESTAMP_INVALID", f"{label} is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AIRuntimeError("TIMESTAMP_INVALID", f"invalid {label}: {value}") from exc
    if parsed.tzinfo is None:
        raise AIRuntimeError("TIMESTAMP_INVALID", f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise AIRuntimeError("TIMESTAMP_INVALID", "acceptor clock must include a timezone")
    return value.astimezone(timezone.utc)


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise OSError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass

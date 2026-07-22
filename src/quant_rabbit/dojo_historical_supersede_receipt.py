"""Append-only custody receipt for superseding a historical generation.

This module never repairs, resumes, completes, or otherwise mutates the old
run.  It validates the old plan, schedule, and execution state with the
existing long-horizon validators, proves that no runner owns the old lock,
and seals the exact control-plane JSON inventory before naming a new sealed
generation.  The resulting receipt is research custody only; it grants no
broker, promotion, live, or old-claim resume authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Final

from quant_rabbit.dojo_long_horizon_execution import (
    long_horizon_execution_status,
)
from quant_rabbit.dojo_long_horizon_plan import (
    canonical_sha256,
    validate_long_horizon_train_plan,
)
from quant_rabbit.dojo_long_horizon_schedule import (
    validate_long_horizon_stream_schedule,
)


CONTRACT: Final = "QR_DOJO_HISTORICAL_GENERATION_SUPERSEDE_RECEIPT_V2"
SCHEMA_VERSION: Final = 2
LEGACY_V1_CONTRACT: Final = "QR_DOJO_HISTORICAL_GENERATION_SUPERSEDE_RECEIPT_V1"
LINEAGE_REGISTRY_DIRECTORY: Final = ".dojo-generation-successors-v2"
MAX_JSON_BYTES: Final = 16 * 1024 * 1024
MAX_STATE_FILES: Final = 4096
MAX_STATE_BYTES: Final = 512 * 1024 * 1024
MAX_RECEIPTS: Final = 1024
MAX_PENDING_ANCHORS: Final = 32
HASH_CHUNK_BYTES: Final = 1024 * 1024

_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GENERATION_LABEL_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,62}\Z")
_JOB_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CLAIM_DIRECTORY_RE: Final = re.compile(r"attempt-[0-9]{4}-[0-9a-f]{64}\Z")
_CLAIM_FILE_RE: Final = re.compile(r"attempt-[0-9]{4}\.json\Z")
_TERMINAL_FILE_RE: Final = re.compile(
    r"attempt-[0-9]{4}-[0-9a-f]{64}-[0-9a-f]{64}\.json\Z"
)
_GENERATION_MANIFEST_CONTRACT: Final = "QR_DOJO_HISTORICAL_TRAIN_GENERATION_MANIFEST_V1"
_RUN_CONTROL_CONTRACT: Final = "QR_DOJO_G2_HISTORICAL_RUN_CONTROL_V1"
_IMPLEMENTATION_MANIFEST_CONTRACT: Final = "QR_DOJO_IMPLEMENTATION_DIGEST_MANIFEST_V1"
_CARRY_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_CARRY_SLOT_V1"
_REDUCER_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_REDUCER_HANDOFF_V1"
_AUTHORITY: Final = {
    "automatic_deployment_allowed": False,
    "broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
    "promotion_eligible": False,
}
_TRANSITION: Final = {
    "evidence_preservation_scope": (
        "OLD_PLAN_SCHEDULE_AND_EXECUTION_STATE_INVENTORY_ONLY"
    ),
    "old_claim_completed_by_receipt": False,
    "old_execution_state_preserved": True,
    "old_non_state_evidence_preservation_asserted": False,
    "old_resume_authorized": False,
    "old_root_deletion_authorized": False,
    "reason": "PREDECESSOR_GENERATION_SUPERSEDED_WITHOUT_STATE_MUTATION",
}

_RECEIPT_KEYS: Final = {
    "contract",
    "schema_version",
    "transition_identity_sha256",
    "predecessor_state_sha256",
    "old_generation",
    "new_generation",
    "transition",
    "authority",
    "receipt_sha256",
}


class DojoHistoricalSupersedeReceiptError(ValueError):
    """An old run cannot be safely superseded by the named generation."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "value is not canonical JSON"
        ) from exc


def _strict_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoHistoricalSupersedeReceiptError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _stable_regular_bytes(
    path: Path, *, field: str, maximum_bytes: int = MAX_JSON_BYTES
) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalSupersedeReceiptError(f"{field} is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or not 0 < before.st_size <= maximum_bytes
    ):
        raise DojoHistoricalSupersedeReceiptError(
            f"{field} must be a bounded nonempty regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            raw = handle.read(maximum_bytes + 1)
            after_open = os.fstat(handle.fileno())
    except OSError as exc:
        raise DojoHistoricalSupersedeReceiptError(f"cannot read {field}") from exc
    try:
        after_path = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalSupersedeReceiptError(
            f"{field} changed while read"
        ) from exc
    identities = {
        (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        for row in (before, opened, after_open, after_path)
    }
    if len(identities) != 1 or len(raw) != before.st_size:
        raise DojoHistoricalSupersedeReceiptError(f"{field} changed while read")
    return raw


def _parse_json(raw: bytes, *, field: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise DojoHistoricalSupersedeReceiptError(
            f"{field} contains non-finite JSON: {token}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoHistoricalSupersedeReceiptError(
                    f"{field} contains duplicate key: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoHistoricalSupersedeReceiptError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoHistoricalSupersedeReceiptError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise DojoHistoricalSupersedeReceiptError(f"{field} must be one JSON object")
    canonical = _canonical_bytes(value)
    if raw not in {canonical, canonical + b"\n"}:
        raise DojoHistoricalSupersedeReceiptError(f"{field} is not canonical JSON")
    return value


def read_bounded_json_artifact(path: Path, *, field: str) -> dict[str, Any]:
    """Read one stable, canonical, bounded JSON object without mutation."""

    return _parse_json(_stable_regular_bytes(Path(path), field=field), field=field)


def _safe_root(path: Path, *, field: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise DojoHistoricalSupersedeReceiptError(f"{field} must be absolute")
    try:
        state = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalSupersedeReceiptError(f"{field} is unavailable") from exc
    if candidate.is_symlink() or not stat.S_ISDIR(state.st_mode):
        raise DojoHistoricalSupersedeReceiptError(f"{field} must be a real directory")
    return candidate.resolve(strict=True)


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise DojoHistoricalSupersedeReceiptError(f"{field} is not a path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DojoHistoricalSupersedeReceiptError(f"{field} is unsafe")
    return pure.as_posix()


def _state_inventory(
    state_root: Path, *, schedule: Mapping[str, Any]
) -> dict[str, Any]:
    root = _safe_root(state_root, field="old execution-state root")
    if {path.name for path in root.iterdir()} != {
        "execution-manifest.json",
        "claims",
        "cells",
        "terminals",
        "carry",
        "reducers",
    }:
        raise DojoHistoricalSupersedeReceiptError(
            "old execution-state top-level tree is not exact"
        )
    jobs = {job["job_sha256"]: job for job in schedule["jobs"]}
    coordinate_ids = {
        job_sha: {coordinate["coordinate_id"] for coordinate in job["coordinates"]}
        for job_sha, job in jobs.items()
    }
    carry_slots = {
        slot
        for job in schedule["jobs"]
        for coordinate in job["coordinates"]
        for slot in (
            coordinate.get("predecessor_state_slot_id"),
            coordinate.get("carry_out_state_slot_id"),
        )
        if isinstance(slot, str)
    }
    rows: list[dict[str, Any]] = []
    directories: list[str] = []
    parsed: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        directory_path = Path(directory)
        directory_state = directory_path.stat(follow_symlinks=False)
        if directory_path.is_symlink() or not stat.S_ISDIR(directory_state.st_mode):
            raise DojoHistoricalSupersedeReceiptError(
                "old execution-state contains an unsafe directory"
            )
        for name in sorted(directory_names):
            child = directory_path / name
            child_state = child.stat(follow_symlinks=False)
            if child.is_symlink() or not stat.S_ISDIR(child_state.st_mode):
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state contains a symlink or special directory"
                )
            directories.append(child.relative_to(root).as_posix())
        for name in sorted(file_names):
            child = directory_path / name
            if child.suffix != ".json":
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state contains a non-JSON artifact"
                )
            relative = _safe_relative(
                child.relative_to(root).as_posix(), field="state inventory path"
            )
            raw = _stable_regular_bytes(child, field=f"state artifact {relative}")
            parsed[relative] = _parse_json(raw, field=f"state artifact {relative}")
            total_bytes += len(raw)
            if total_bytes > MAX_STATE_BYTES:
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state inventory exceeds its byte bound"
                )
            rows.append(
                {
                    "relative_path": relative,
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            if len(rows) > MAX_STATE_FILES:
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state inventory exceeds its file bound"
                )
    rows.sort(key=lambda row: row["relative_path"])
    directories.sort()
    if not rows:
        raise DojoHistoricalSupersedeReceiptError(
            "old execution-state inventory is empty"
        )
    claims: dict[tuple[str, int], str] = {}
    terminals: set[tuple[str, str]] = set()
    for relative, value in parsed.items():
        parts = PurePosixPath(relative).parts
        if len(parts) == 3 and parts[0] == "claims":
            if parts[1] not in jobs or _CLAIM_FILE_RE.fullmatch(parts[2]) is None:
                raise DojoHistoricalSupersedeReceiptError(
                    f"old execution-state contains an unknown artifact: {relative}"
                )
            attempt = int(parts[2][8:12])
            claim_sha = value.get("claim_sha256")
            if (
                value.get("job_sha256") != parts[1]
                or value.get("attempt_ordinal") != attempt
                or not isinstance(claim_sha, str)
                or _SHA_RE.fullmatch(claim_sha) is None
                or (parts[1], attempt) in claims
            ):
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state claim path binding is invalid"
                )
            claims[(parts[1], attempt)] = claim_sha
        elif len(parts) == 3 and parts[0] == "terminals":
            match = _TERMINAL_FILE_RE.fullmatch(parts[2])
            claim = value.get("claim")
            if match is None or parts[1] not in jobs or not isinstance(claim, Mapping):
                raise DojoHistoricalSupersedeReceiptError(
                    f"old execution-state contains an unknown artifact: {relative}"
                )
            filename_parts = parts[2][:-5].split("-")
            attempt = int(filename_parts[1])
            claim_sha = filename_parts[2]
            terminal_sha = filename_parts[3]
            if (
                claim.get("claim_sha256") != claim_sha
                or claim.get("attempt_ordinal") != attempt
                or value.get("job_sha256") != parts[1]
                or value.get("terminal_sha256") != terminal_sha
                or claims.get((parts[1], attempt)) != claim_sha
            ):
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state terminal path binding is invalid"
                )
            terminals.add((parts[1], terminal_sha))

    allowed_directories = {
        "claims",
        "cells",
        "terminals",
        "carry",
        "reducers",
    }
    allowed_directories.update(
        f"{section}/{job_sha}"
        for section in ("claims", "cells", "terminals", "reducers")
        for job_sha in jobs
    )
    allowed_directories.update(
        f"cells/{job_sha}/attempt-{attempt:04d}-{claim_sha}"
        for (job_sha, attempt), claim_sha in claims.items()
    )
    unknown_directories = sorted(set(directories) - allowed_directories)
    if unknown_directories:
        raise DojoHistoricalSupersedeReceiptError(
            "old execution-state contains an unknown directory: "
            f"{unknown_directories[0]}"
        )

    for relative, value in parsed.items():
        parts = PurePosixPath(relative).parts
        if parts == ("execution-manifest.json",):
            continue
        if len(parts) == 3 and parts[0] in {"claims", "terminals"}:
            continue
        if len(parts) == 4 and parts[0] == "cells":
            match = _CLAIM_DIRECTORY_RE.fullmatch(parts[2])
            coordinate_id = parts[3][:-5] if parts[3].endswith(".json") else ""
            if match is not None:
                directory_parts = parts[2].split("-")
                attempt = int(directory_parts[1])
                claim_sha = directory_parts[2]
            else:
                attempt = -1
                claim_sha = ""
            if (
                parts[1] not in jobs
                or coordinate_id not in coordinate_ids[parts[1]]
                or claims.get((parts[1], attempt)) != claim_sha
                or value.get("job_sha256") != parts[1]
                or value.get("claim_sha256") != claim_sha
                or value.get("coordinate_id") != coordinate_id
            ):
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state cell path binding is invalid"
                )
            continue
        if len(parts) == 3 and parts[0] == "reducers":
            terminal_sha = parts[2][:-5] if parts[2].endswith(".json") else ""
            body = {
                key: item
                for key, item in value.items()
                if key != "reducer_handoff_sha256"
            }
            if (
                parts[1] not in jobs
                or (parts[1], terminal_sha) not in terminals
                or value.get("contract") != _REDUCER_CONTRACT
                or value.get("job_sha256") != parts[1]
                or value.get("terminal_sha256") != terminal_sha
                or value.get("reducer_handoff_sha256") != canonical_sha256(body)
            ):
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state reducer path binding is invalid"
                )
            continue
        if len(parts) == 2 and parts[0] == "carry":
            slot = parts[1][:-5] if parts[1].endswith(".json") else ""
            body = {
                key: item
                for key, item in value.items()
                if key != "carry_receipt_sha256"
            }
            if (
                slot not in carry_slots
                or value.get("contract") != _CARRY_CONTRACT
                or value.get("state_slot_id") != slot
                or (
                    value.get("producer_job_sha256"),
                    value.get("producer_terminal_sha256"),
                )
                not in terminals
                or value.get("carry_receipt_sha256") != canonical_sha256(body)
            ):
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state carry path binding is invalid"
                )
            continue
        raise DojoHistoricalSupersedeReceiptError(
            f"old execution-state contains an unknown artifact: {relative}"
        )

    body = {
        "root_relative_path": "execution-state",
        "directory_count": len(directories),
        "directories": directories,
        "file_count": len(rows),
        "total_size_bytes": total_bytes,
        "files": rows,
    }
    return {**body, "inventory_sha256": canonical_sha256(body)}


def _validated_artifacts(
    plan: Mapping[str, Any], schedule: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        validated_plan = validate_long_horizon_train_plan(plan)
        validated_schedule = validate_long_horizon_stream_schedule(
            schedule, plan=validated_plan
        )
    except (TypeError, ValueError) as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "long-horizon plan or schedule validation failed"
        ) from exc
    return validated_plan, validated_schedule


def _validated_new_generation(
    new_root: Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    str,
]:
    root = _safe_root(new_root, field="new historical root")
    plan = read_bounded_json_artifact(root / "plan.json", field="new plan")
    schedule = read_bounded_json_artifact(root / "schedule.json", field="new schedule")
    validated_plan, validated_schedule = _validated_artifacts(plan, schedule)
    manifest = read_bounded_json_artifact(
        root / "control-manifest.json", field="new control manifest"
    )
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    label = manifest.get("generation")
    if (
        manifest.get("contract") != _GENERATION_MANIFEST_CONTRACT
        or manifest.get("schema_version") != 1
        or manifest.get("manifest_sha256") != canonical_sha256(body)
        or manifest.get("plan_sha256") != validated_plan["plan_sha256"]
        or manifest.get("schedule_sha256") != validated_schedule["schedule_sha256"]
        or any(manifest.get(key) != value for key, value in _AUTHORITY.items())
        or not isinstance(label, str)
        or _GENERATION_LABEL_RE.fullmatch(label) is None
    ):
        raise DojoHistoricalSupersedeReceiptError(
            "new control manifest or artifact binding is invalid"
        )
    sealed_rows = manifest.get("sealed_input_artifacts")
    if not isinstance(sealed_rows, list) or manifest.get(
        "sealed_input_artifacts_sha256"
    ) != canonical_sha256(sealed_rows):
        raise DojoHistoricalSupersedeReceiptError(
            "new sealed input inventory is invalid"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in sealed_rows:
        if not isinstance(row, Mapping):
            raise DojoHistoricalSupersedeReceiptError("new sealed input row is invalid")
        artifact_id = row.get("artifact_id")
        if not isinstance(artifact_id, str) or artifact_id in by_id:
            raise DojoHistoricalSupersedeReceiptError(
                "new sealed input artifact ids are invalid"
            )
        by_id[artifact_id] = row
    required_paths = {
        "RUN_CONTROL": "sealed-inputs/run-control.json",
        "IMPLEMENTATION_MANIFEST": "sealed-inputs/implementation-manifest.json",
    }
    bound_values: dict[str, dict[str, Any]] = {}
    file_digests: dict[str, str] = {}
    for artifact_id, expected_relative in required_paths.items():
        row = by_id.get(artifact_id)
        if row is None or set(row) != {
            "artifact_id",
            "relative_path",
            "file_size_bytes",
            "file_sha256",
        }:
            raise DojoHistoricalSupersedeReceiptError(
                f"new sealed {artifact_id.lower()} binding is invalid"
            )
        relative = _safe_relative(
            row["relative_path"], field=f"new sealed {artifact_id.lower()} path"
        )
        expected_size = row["file_size_bytes"]
        expected_sha = _strict_sha(
            row["file_sha256"], field=f"new sealed {artifact_id.lower()} SHA-256"
        )
        if (
            relative != expected_relative
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 1
        ):
            raise DojoHistoricalSupersedeReceiptError(
                f"new sealed {artifact_id.lower()} metadata is invalid"
            )
        raw = _stable_regular_bytes(
            root / relative, field=f"new sealed {artifact_id.lower()}"
        )
        if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha:
            raise DojoHistoricalSupersedeReceiptError(
                f"new sealed {artifact_id.lower()} bytes drifted"
            )
        bound_values[artifact_id] = _parse_json(
            raw, field=f"new sealed {artifact_id.lower()}"
        )
        file_digests[f"{artifact_id.lower()}_file_sha256"] = expected_sha

    run_control = bound_values["RUN_CONTROL"]
    run_execution = run_control.get("execution")
    run_fixed = run_control.get("fixed_inputs")
    run_authority = run_control.get("authority")
    run_control_source_sha = _strict_sha(
        manifest.get("run_control_sha256"),
        field="new source run-control SHA-256",
    )
    if (
        run_control.get("contract") != _RUN_CONTROL_CONTRACT
        or run_control.get("schema_version") != 1
        or not isinstance(run_execution, Mapping)
        or not isinstance(run_fixed, Mapping)
        or not isinstance(run_authority, Mapping)
        or run_fixed.get("generation") != label
        or run_execution.get("output_root") != os.fspath(root)
        or run_authority.get("broker_mutation_allowed") is not False
        or run_authority.get("live_permission") is not False
        or run_authority.get("order_authority") != "NONE"
        or run_authority.get("historical_replay_process_start_allowed") is not True
        or run_authority.get("research_filesystem_write_allowed") is not True
    ):
        raise DojoHistoricalSupersedeReceiptError(
            "new sealed run-control does not bind the new generation"
        )

    implementation = bound_values["IMPLEMENTATION_MANIFEST"]
    implementation_body = {
        key: value
        for key, value in implementation.items()
        if key != "implementation_manifest_sha256"
    }
    implementation_digests = implementation.get("implementation_digests")
    plan_binding = validated_plan.get("implementation_binding")
    if (
        implementation.get("contract") != _IMPLEMENTATION_MANIFEST_CONTRACT
        or implementation.get("schema_version") != 1
        or not isinstance(implementation_digests, Mapping)
        or not implementation_digests
        or not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and _SHA_RE.fullmatch(value) is not None
            for key, value in implementation_digests.items()
        )
        or implementation.get("implementation_digests_sha256")
        != canonical_sha256(implementation_digests)
        or implementation.get("implementation_manifest_sha256")
        != canonical_sha256(implementation_body)
        or any(implementation.get(key) != value for key, value in _AUTHORITY.items())
        or not isinstance(plan_binding, Mapping)
        or plan_binding.get("digests") != implementation_digests
    ):
        raise DojoHistoricalSupersedeReceiptError(
            "new implementation manifest or plan binding is invalid"
        )
    bindings = {
        "new_root": os.fspath(root),
        "control_manifest_sha256": manifest["manifest_sha256"],
        "run_control_source_sha256": run_control_source_sha,
        "run_control_file_sha256": file_digests["run_control_file_sha256"],
        "implementation_file_sha256": file_digests[
            "implementation_manifest_file_sha256"
        ],
        "implementation_manifest_sha256": implementation[
            "implementation_manifest_sha256"
        ],
        "implementation_digests_sha256": implementation[
            "implementation_digests_sha256"
        ],
    }
    identity = canonical_sha256(
        {
            "generation": label,
            "plan_sha256": validated_plan["plan_sha256"],
            "schedule_sha256": validated_schedule["schedule_sha256"],
            **bindings,
        }
    )
    return (
        root,
        validated_plan,
        validated_schedule,
        manifest,
        bindings,
        f"{label}:{identity}",
    )


@contextmanager
def _unowned_old_lock(old_root: Path) -> Iterator[int]:
    lock_path = old_root / ".historical-train.lock"
    try:
        before = lock_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "old historical lock is unavailable"
        ) from exc
    if lock_path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise DojoHistoricalSupersedeReceiptError(
            "old historical lock must be an existing regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "cannot open old historical lock"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DojoHistoricalSupersedeReceiptError(
                "old historical run still owns its lock"
            ) from exc
        yield descriptor
        after = lock_path.stat(follow_symlinks=False)
        identities = {
            (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
            for row in (before, opened, after)
        }
        if len(identities) != 1:
            raise DojoHistoricalSupersedeReceiptError(
                "old historical lock changed during supersede verification"
            )
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _old_snapshot(old_root: Path) -> dict[str, Any]:
    plan = read_bounded_json_artifact(old_root / "plan.json", field="old plan")
    schedule = read_bounded_json_artifact(
        old_root / "schedule.json", field="old schedule"
    )
    validated_plan, validated_schedule = _validated_artifacts(plan, schedule)
    try:
        status = long_horizon_execution_status(
            old_root / "execution-state",
            schedule=validated_schedule,
            plan=validated_plan,
        )
    except (TypeError, ValueError) as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "old execution status validation failed"
        ) from exc
    active = status.get("active_job_count")
    if (
        isinstance(active, bool)
        or not isinstance(active, int)
        or active < 0
        or active > 1
    ):
        raise DojoHistoricalSupersedeReceiptError(
            "old execution state must have at most one active claim"
        )
    return {
        "root": os.fspath(old_root),
        "plan_sha256": validated_plan["plan_sha256"],
        "schedule_sha256": validated_schedule["schedule_sha256"],
        "execution_status": status,
        "active_job_count": active,
        "execution_state_inventory": _state_inventory(
            old_root / "execution-state", schedule=validated_schedule
        ),
    }


def _stable_old_snapshot(old_root: Path) -> dict[str, Any]:
    first = _old_snapshot(old_root)
    second = _old_snapshot(old_root)
    if first != second:
        raise DojoHistoricalSupersedeReceiptError(
            "old execution state drifted during supersede verification"
        )
    return first


def _identity_body(
    *,
    old_snapshot: Mapping[str, Any],
    new_generation_id: str,
    new_plan: Mapping[str, Any],
    new_schedule: Mapping[str, Any],
    new_control_manifest: Mapping[str, Any],
    new_generation_bindings: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "old_root": old_snapshot["root"],
        "old_plan_sha256": old_snapshot["plan_sha256"],
        "old_schedule_sha256": old_snapshot["schedule_sha256"],
        "new_generation_id": new_generation_id,
        "new_plan_sha256": new_plan["plan_sha256"],
        "new_schedule_sha256": new_schedule["schedule_sha256"],
        "new_control_manifest_sha256": new_control_manifest["manifest_sha256"],
        "new_generation_bindings": dict(new_generation_bindings),
    }


def _build_receipt(
    *,
    old_snapshot: Mapping[str, Any],
    new_generation_id: str,
    new_plan: Mapping[str, Any],
    new_schedule: Mapping[str, Any],
    new_control_manifest: Mapping[str, Any],
    new_generation_bindings: Mapping[str, str],
) -> dict[str, Any]:
    identity = _identity_body(
        old_snapshot=old_snapshot,
        new_generation_id=new_generation_id,
        new_plan=new_plan,
        new_schedule=new_schedule,
        new_control_manifest=new_control_manifest,
        new_generation_bindings=new_generation_bindings,
    )
    predecessor_state_sha256 = canonical_sha256(old_snapshot)
    body = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "transition_identity_sha256": canonical_sha256(identity),
        "predecessor_state_sha256": predecessor_state_sha256,
        "old_generation": dict(old_snapshot),
        "new_generation": {
            "generation_id": new_generation_id,
            "plan_sha256": new_plan["plan_sha256"],
            "schedule_sha256": new_schedule["schedule_sha256"],
            **dict(new_generation_bindings),
        },
        "transition": dict(_TRANSITION),
        "authority": dict(_AUTHORITY),
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _validate_receipt_against_snapshot(
    receipt: Mapping[str, Any],
    *,
    old_snapshot: Mapping[str, Any],
    new_generation_id: str,
    new_plan: Mapping[str, Any],
    new_schedule: Mapping[str, Any],
    new_control_manifest: Mapping[str, Any],
    new_generation_bindings: Mapping[str, str],
) -> dict[str, Any]:
    expected_keys = {
        "contract",
        "schema_version",
        "transition_identity_sha256",
        "predecessor_state_sha256",
        "old_generation",
        "new_generation",
        "transition",
        "authority",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt schema is not exact"
        )
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("contract") != CONTRACT
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("authority") != _AUTHORITY
        or receipt.get("transition") != _TRANSITION
        or receipt.get("receipt_sha256") != canonical_sha256(body)
    ):
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt contract or seal is invalid"
        )
    expected = _build_receipt(
        old_snapshot=old_snapshot,
        new_generation_id=new_generation_id,
        new_plan=new_plan,
        new_schedule=new_schedule,
        new_control_manifest=new_control_manifest,
        new_generation_bindings=new_generation_bindings,
    )
    if dict(receipt) != expected:
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt no longer matches old or new generation evidence"
        )
    return expected


def _receipt_directory(new_root: Path, *, create: bool) -> Path:
    candidate = new_root / "transition-receipts"
    if not candidate.exists():
        if not create:
            raise DojoHistoricalSupersedeReceiptError(
                "receipt directory is unavailable"
            )
        parent = _safe_root(candidate.parent, field="receipt parent")
        try:
            candidate.mkdir(mode=0o700)
        except OSError as exc:
            raise DojoHistoricalSupersedeReceiptError(
                "cannot create append-only receipt directory"
            ) from exc
        if candidate.parent.resolve(strict=True) != parent:
            raise DojoHistoricalSupersedeReceiptError(
                "receipt parent changed while creating store"
            )
    root = _safe_root(candidate, field="receipt directory")
    return root


def _receipt_filename(receipt: Mapping[str, Any]) -> str:
    identity = _strict_sha(
        receipt.get("transition_identity_sha256"),
        field="transition identity SHA-256",
    )
    digest = _strict_sha(receipt.get("receipt_sha256"), field="receipt SHA-256")
    return f"supersede-{identity}-{digest}.json"


def _pending_path(path: Path) -> Path:
    return path.parent / f".{path.name}.pending"


def _pending_paths(path: Path) -> Iterator[Path]:
    yield _pending_path(path)
    for ordinal in range(1, MAX_PENDING_ANCHORS):
        yield path.parent / f".{path.name}.pending-{ordinal:04d}"


def _pending_destination_name(name: str) -> str | None:
    match = re.fullmatch(r"\.(.+\.json)\.pending(?:-[0-9]{4})?", name)
    if match is None:
        return None
    return match.group(1)


def _validate_stored_receipt_envelope(
    value: Mapping[str, Any], *, allow_legacy_v1: bool, field: str
) -> int:
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if set(value) != _RECEIPT_KEYS:
        raise DojoHistoricalSupersedeReceiptError(f"{field} has an invalid shape")
    _strict_sha(
        value.get("transition_identity_sha256"),
        field=f"{field} transition identity SHA-256",
    )
    _strict_sha(
        value.get("predecessor_state_sha256"),
        field=f"{field} predecessor state SHA-256",
    )
    if value.get("receipt_sha256") != canonical_sha256(body):
        raise DojoHistoricalSupersedeReceiptError(f"{field} seal is invalid")
    if (
        value.get("contract") == CONTRACT
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("authority") == _AUTHORITY
        and value.get("transition") == _TRANSITION
    ):
        return SCHEMA_VERSION
    if (
        allow_legacy_v1
        and value.get("contract") == LEGACY_V1_CONTRACT
        and value.get("schema_version") == 1
    ):
        # V1 is recognized only to cross the migration boundary.  It is never
        # returned as, or used to establish, V2 successor custody.
        return 1
    raise DojoHistoricalSupersedeReceiptError(f"{field} contract is invalid")


def _local_v2_receipts(
    receipt_dir: Path, *, recoverable_receipt: Mapping[str, Any] | None = None
) -> list[tuple[Path, dict[str, Any]]]:
    entries = sorted(receipt_dir.iterdir())
    if len(entries) > MAX_RECEIPTS:
        raise DojoHistoricalSupersedeReceiptError(
            "append-only supersede receipt store exceeds its file bound"
        )
    recoverable_destination_name = (
        _receipt_filename(recoverable_receipt)
        if recoverable_receipt is not None
        else None
    )
    receipts: list[tuple[Path, dict[str, Any]]] = []
    pending: list[tuple[Path, dict[str, Any], str]] = []
    for path in entries:
        destination_name = _pending_destination_name(path.name)
        if destination_name is not None:
            try:
                state = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise DojoHistoricalSupersedeReceiptError(
                    "stored supersede receipt pending is unavailable"
                ) from exc
            if path.is_symlink() or not stat.S_ISREG(state.st_mode):
                raise DojoHistoricalSupersedeReceiptError(
                    "append-only supersede receipt store contains an unknown entry"
                )
            if destination_name == recoverable_destination_name:
                # All anchors for the caller's exact final are deferred to the
                # hard-link publisher.  It preserves invalid/partial anchors
                # and selects the next deterministic slot without unlinking.
                continue
            try:
                value = read_bounded_json_artifact(
                    path, field="stored supersede receipt pending"
                )
            except DojoHistoricalSupersedeReceiptError:
                # A crash during anchor preparation is non-authoritative and
                # append-only.  Keep it, but never accept it as V2 evidence.
                continue
            version = _validate_stored_receipt_envelope(
                value,
                allow_legacy_v1=False,
                field="stored supersede receipt pending",
            )
            if version != SCHEMA_VERSION or destination_name != _receipt_filename(
                value
            ):
                raise DojoHistoricalSupersedeReceiptError(
                    "append-only supersede receipt store contains an invalid pending"
                )
            pending.append((path, value, destination_name))
            continue
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise DojoHistoricalSupersedeReceiptError(
                "append-only supersede receipt store contains an unknown entry"
            )
        value = read_bounded_json_artifact(path, field="stored supersede receipt")
        version = _validate_stored_receipt_envelope(
            value,
            allow_legacy_v1=True,
            field="stored supersede receipt",
        )
        if path.name != _receipt_filename(value):
            raise DojoHistoricalSupersedeReceiptError(
                "append-only supersede receipt store contains an invalid receipt"
            )
        if version == SCHEMA_VERSION:
            receipts.append((path, value))
    final_by_name = {path.name: (path, value) for path, value in receipts}
    linked_finals: set[str] = set()
    for anchor_path, value, destination_name in pending:
        final = final_by_name.get(destination_name)
        if final is None or final[1] != value:
            raise DojoHistoricalSupersedeReceiptError(
                "append-only supersede receipt store contains an orphan pending"
            )
        final_path, _ = final
        try:
            anchor_state = anchor_path.stat(follow_symlinks=False)
            final_state = final_path.stat(follow_symlinks=False)
        except OSError as exc:
            raise DojoHistoricalSupersedeReceiptError(
                "append-only supersede receipt anchor is unavailable"
            ) from exc
        if _entry_identity(anchor_state) == _entry_identity(final_state):
            linked_finals.add(destination_name)
    for final_name in final_by_name:
        if (
            final_name != recoverable_destination_name
            and final_name not in linked_finals
        ):
            raise DojoHistoricalSupersedeReceiptError(
                "append-only supersede receipt final lacks its durable anchor"
            )
    return receipts


def _receipt_files_for_predecessor(
    receipt_dir: Path,
    *,
    predecessor_state_sha256: str,
    recoverable_receipt: Mapping[str, Any] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    predecessor = _strict_sha(
        predecessor_state_sha256, field="predecessor state SHA-256"
    )
    matches = [
        (path, value)
        for path, value in _local_v2_receipts(
            receipt_dir, recoverable_receipt=recoverable_receipt
        )
        if value.get("predecessor_state_sha256") == predecessor
    ]
    if len(matches) > 1:
        raise DojoHistoricalSupersedeReceiptError(
            "multiple V2 supersede receipts name the same predecessor state"
        )
    return matches


def _lineage_registry_directory(old_root: Path, *, create: bool) -> Path:
    parent = _safe_root(old_root.parent, field="old historical parent")
    candidate = parent / LINEAGE_REGISTRY_DIRECTORY
    if not candidate.exists():
        if not create:
            raise DojoHistoricalSupersedeReceiptError(
                "successor lineage registry is unavailable"
            )
        try:
            candidate.mkdir(mode=0o700)
        except OSError as exc:
            raise DojoHistoricalSupersedeReceiptError(
                "cannot create successor lineage registry"
            ) from exc
        if candidate.parent.resolve(strict=True) != parent:
            raise DojoHistoricalSupersedeReceiptError(
                "successor lineage registry parent changed"
            )
    return _safe_root(candidate, field="successor lineage registry")


def _lineage_receipt_filename(receipt: Mapping[str, Any]) -> str:
    predecessor = _strict_sha(
        receipt.get("predecessor_state_sha256"),
        field="predecessor state SHA-256",
    )
    return f"predecessor-{predecessor}.json"


def _validated_lineage_entries(
    registry: Path, *, recoverable_receipt: Mapping[str, Any] | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    entries = sorted(registry.iterdir())
    if len(entries) > MAX_RECEIPTS:
        raise DojoHistoricalSupersedeReceiptError(
            "successor lineage registry exceeds its file bound"
        )
    result: dict[str, dict[str, Any]] = {}
    result_paths: dict[str, Path] = {}
    pending: list[tuple[Path, str, dict[str, Any]]] = []
    recoverable_destination_name = (
        _lineage_receipt_filename(recoverable_receipt)
        if recoverable_receipt is not None
        else None
    )
    for path in entries:
        destination_name = _pending_destination_name(path.name)
        if destination_name is not None:
            try:
                state = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise DojoHistoricalSupersedeReceiptError(
                    "successor lineage pending is unavailable"
                ) from exc
            if path.is_symlink() or not stat.S_ISREG(state.st_mode):
                raise DojoHistoricalSupersedeReceiptError(
                    "successor lineage registry contains an unknown entry"
                )
            if destination_name == recoverable_destination_name:
                continue
            try:
                value = read_bounded_json_artifact(
                    path, field="successor lineage pending"
                )
            except DojoHistoricalSupersedeReceiptError:
                continue
            _validate_stored_receipt_envelope(
                value,
                allow_legacy_v1=False,
                field="successor lineage pending",
            )
            predecessor = str(value["predecessor_state_sha256"])
            if destination_name != _lineage_receipt_filename(value):
                raise DojoHistoricalSupersedeReceiptError(
                    "successor lineage registry contains an invalid pending"
                )
            pending.append((path, predecessor, value))
            continue
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise DojoHistoricalSupersedeReceiptError(
                "successor lineage registry contains an unknown entry"
            )
        value = read_bounded_json_artifact(path, field="successor lineage receipt")
        _validate_stored_receipt_envelope(
            value,
            allow_legacy_v1=False,
            field="successor lineage receipt",
        )
        predecessor = value.get("predecessor_state_sha256")
        if (
            not isinstance(predecessor, str)
            or _SHA_RE.fullmatch(predecessor) is None
            or path.name != _lineage_receipt_filename(value)
            or predecessor in result
        ):
            raise DojoHistoricalSupersedeReceiptError(
                "successor lineage registry contains an invalid receipt"
            )
        result[predecessor] = value
        result_paths[predecessor] = path
    linked: dict[str, dict[str, Any]] = {}
    for anchor_path, predecessor, value in pending:
        if result.get(predecessor) != value:
            raise DojoHistoricalSupersedeReceiptError(
                "successor lineage registry contains an orphan pending"
            )
        try:
            anchor_state = anchor_path.stat(follow_symlinks=False)
            final_state = result_paths[predecessor].stat(follow_symlinks=False)
        except OSError as exc:
            raise DojoHistoricalSupersedeReceiptError(
                "successor lineage anchor is unavailable"
            ) from exc
        if _entry_identity(anchor_state) == _entry_identity(final_state):
            linked[predecessor] = value
    recoverable_predecessor = (
        str(recoverable_receipt["predecessor_state_sha256"])
        if recoverable_receipt is not None
        else None
    )
    for predecessor in result:
        if predecessor != recoverable_predecessor and predecessor not in linked:
            raise DojoHistoricalSupersedeReceiptError(
                "successor lineage final lacks its durable anchor"
            )
    return result, linked


def _reserve_or_verify_unique_successor(
    *, old_root: Path, receipt: Mapping[str, Any], create: bool
) -> None:
    registry = _lineage_registry_directory(old_root, create=create)
    entries, _ = _validated_lineage_entries(
        registry, recoverable_receipt=receipt if create else None
    )
    predecessor = str(receipt["predecessor_state_sha256"])
    destination = registry / _lineage_receipt_filename(receipt)
    existing = entries.get(predecessor)
    if existing is not None:
        if existing != dict(receipt):
            raise DojoHistoricalSupersedeReceiptError(
                "predecessor state already names a different successor"
            )
        if create:
            _write_exclusive(destination, receipt)
        return
    if not create:
        raise DojoHistoricalSupersedeReceiptError(
            "successor lineage receipt is unavailable"
        )
    _write_exclusive(destination, receipt)


def _entry_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


@contextmanager
def _stable_directory_descriptor(path: Path) -> Iterator[tuple[int, tuple[int, int]]]:
    descriptor: int | None = None
    try:
        before = path.stat(follow_symlinks=False)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt parent directory is unavailable"
        ) from exc
    if descriptor is None:
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt parent directory descriptor is unavailable"
        )
    identity = _entry_identity(opened)
    if (
        not stat.S_ISDIR(before.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or _entry_identity(before) != identity
        or _entry_identity(after) != identity
    ):
        os.close(descriptor)
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt parent directory changed while opened"
        )
    try:
        yield descriptor, identity
    finally:
        exit_error: DojoHistoricalSupersedeReceiptError | None = None
        try:
            opened_after = os.fstat(descriptor)
            path_after = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened_after.st_mode)
                or not stat.S_ISDIR(path_after.st_mode)
                or _entry_identity(opened_after) != identity
                or _entry_identity(path_after) != identity
            ):
                exit_error = DojoHistoricalSupersedeReceiptError(
                    "supersede receipt parent directory changed while in use"
                )
        except OSError as exc:
            exit_error = DojoHistoricalSupersedeReceiptError(
                "supersede receipt parent directory changed while in use"
            )
            exit_error.__cause__ = exc
        try:
            os.close(descriptor)
        except OSError as exc:
            if exit_error is None:
                exit_error = DojoHistoricalSupersedeReceiptError(
                    "supersede receipt parent directory cannot be closed"
                )
                exit_error.__cause__ = exc
        if exit_error is not None:
            raise exit_error


def _atomic_link_no_replace(
    source_path: Path, destination_path: Path, *, directory_fd: int
) -> bool:
    if source_path.parent != destination_path.parent:
        raise DojoHistoricalSupersedeReceiptError(
            "atomic receipt publication requires one directory"
        )
    try:
        os.link(
            source_path.name,
            destination_path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        return False
    except OSError as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "atomic no-replace receipt hard-link failed"
        ) from exc
    return True


def _stable_exact_anchor(
    path: Path, *, payload: bytes, directory_fd: int
) -> os.stat_result | None:
    try:
        before = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before.st_mode):
        return None
    try:
        observed = _stable_regular_bytes(path, field="supersede receipt anchor")
        after = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except (FileNotFoundError, DojoHistoricalSupersedeReceiptError):
        return None
    if observed != payload or _entry_identity(before) != _entry_identity(after):
        return None
    return after


def _anchor_inventory(
    final_path: Path, *, payload: bytes, directory_fd: int
) -> tuple[list[tuple[Path, os.stat_result]], list[Path]]:
    exact: list[tuple[Path, os.stat_result]] = []
    available: list[Path] = []
    for anchor in _pending_paths(final_path):
        try:
            state = os.stat(
                anchor.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            available.append(anchor)
            continue
        if not stat.S_ISREG(state.st_mode):
            continue
        verified = _stable_exact_anchor(
            anchor, payload=payload, directory_fd=directory_fd
        )
        if verified is not None:
            exact.append((anchor, verified))
    return exact, available


def _create_durable_anchor(
    final_path: Path, *, payload: bytes, directory_fd: int
) -> tuple[Path, os.stat_result]:
    _, available = _anchor_inventory(
        final_path, payload=payload, directory_fd=directory_fd
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for anchor in available:
        try:
            descriptor = os.open(anchor.name, flags, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            continue
        except OSError as exc:
            raise DojoHistoricalSupersedeReceiptError(
                "supersede receipt anchor cannot be created"
            ) from exc
        created = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                if handle.write(payload) != len(payload):
                    raise DojoHistoricalSupersedeReceiptError(
                        "supersede receipt anchor write was incomplete"
                    )
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            # Never unlink a partially written anchor.  A retry uses the next
            # deterministic slot, so a concurrently substituted file cannot
            # be mistaken for cleanup authority.
            raise
        verified = _stable_exact_anchor(
            anchor, payload=payload, directory_fd=directory_fd
        )
        if verified is None or _entry_identity(verified) != _entry_identity(created):
            raise DojoHistoricalSupersedeReceiptError(
                "supersede receipt anchor was replaced after write"
            )
        os.fsync(directory_fd)
        return anchor, verified
    raise DojoHistoricalSupersedeReceiptError(
        "supersede receipt anchor capacity is exhausted"
    )


def _verify_final_anchor(
    *,
    final_path: Path,
    anchor_path: Path,
    payload: bytes,
    directory_fd: int,
) -> os.stat_result:
    if _stable_regular_bytes(final_path, field="supersede receipt final") != payload:
        raise DojoHistoricalSupersedeReceiptError(
            "published supersede receipt bytes drifted"
        )
    anchor = _stable_exact_anchor(
        anchor_path, payload=payload, directory_fd=directory_fd
    )
    try:
        final = os.stat(
            final_path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "published supersede receipt is unavailable"
        ) from exc
    if anchor is None or _entry_identity(anchor) != _entry_identity(final):
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt final is not bound to its durable anchor"
        )
    return final


def _write_exclusive(path: Path, receipt: Mapping[str, Any]) -> None:
    directory = _safe_root(path.parent, field="supersede receipt parent")
    payload = _canonical_bytes(receipt) + b"\n"
    with _stable_directory_descriptor(directory) as (directory_fd, _):
        try:
            current = _stable_regular_bytes(path, field="supersede receipt final")
        except DojoHistoricalSupersedeReceiptError:
            try:
                os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                current = None
            else:
                raise
        if current is not None:
            if current != payload:
                raise DojoHistoricalSupersedeReceiptError(
                    "content-addressed supersede receipt already drifted"
                )
            final_state = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            exact, available = _anchor_inventory(
                path, payload=payload, directory_fd=directory_fd
            )
            for anchor_path, anchor_state in exact:
                if _entry_identity(anchor_state) == _entry_identity(final_state):
                    _verify_final_anchor(
                        final_path=path,
                        anchor_path=anchor_path,
                        payload=payload,
                        directory_fd=directory_fd,
                    )
                    return
            if not available:
                raise DojoHistoricalSupersedeReceiptError(
                    "supersede receipt anchor capacity is exhausted"
                )
            anchor_path = available[0]
            if not _atomic_link_no_replace(
                path, anchor_path, directory_fd=directory_fd
            ):
                raise DojoHistoricalSupersedeReceiptError(
                    "supersede receipt anchor slot changed concurrently"
                )
            os.fsync(directory_fd)
            _verify_final_anchor(
                final_path=path,
                anchor_path=anchor_path,
                payload=payload,
                directory_fd=directory_fd,
            )
            return

        exact, _ = _anchor_inventory(path, payload=payload, directory_fd=directory_fd)
        if exact:
            anchor_path, _ = exact[0]
        else:
            anchor_path, _ = _create_durable_anchor(
                path, payload=payload, directory_fd=directory_fd
            )
        if not _atomic_link_no_replace(anchor_path, path, directory_fd=directory_fd):
            raise DojoHistoricalSupersedeReceiptError(
                "content-addressed supersede receipt appeared concurrently"
            )
        os.fsync(directory_fd)
        _verify_final_anchor(
            final_path=path,
            anchor_path=anchor_path,
            payload=payload,
            directory_fd=directory_fd,
        )


def _reverify_canonical_published_receipt(
    *, old_root: Path, new_root: Path, receipt: Mapping[str, Any]
) -> None:
    predecessor = _strict_sha(
        receipt.get("predecessor_state_sha256"),
        field="published predecessor state SHA-256",
    )
    store = _receipt_directory(new_root, create=False)
    local_matches = _receipt_files_for_predecessor(
        store,
        predecessor_state_sha256=predecessor,
    )
    if (
        len(local_matches) != 1
        or local_matches[0][0].name != _receipt_filename(receipt)
        or local_matches[0][1] != dict(receipt)
    ):
        raise DojoHistoricalSupersedeReceiptError(
            "canonical supersede receipt publication could not be reverified"
        )
    registry = _lineage_registry_directory(old_root, create=False)
    lineage_entries, linked_entries = _validated_lineage_entries(registry)
    if lineage_entries.get(predecessor) != dict(receipt) or linked_entries.get(
        predecessor
    ) != dict(receipt):
        raise DojoHistoricalSupersedeReceiptError(
            "canonical successor lineage publication could not be reverified"
        )


def create_historical_supersede_receipt(
    *,
    old_root: Path,
    new_root: Path,
) -> dict[str, Any]:
    """Create or idempotently re-open one append-only supersede receipt."""

    old = _safe_root(old_root, field="old historical root")
    (
        new,
        validated_new_plan,
        validated_new_schedule,
        new_control_manifest,
        new_generation_bindings,
        generation_id,
    ) = _validated_new_generation(new_root)
    if new == old or old in new.parents or new in old.parents:
        raise DojoHistoricalSupersedeReceiptError(
            "old and new historical roots must be isolated"
        )
    store = _receipt_directory(new, create=True)
    with _unowned_old_lock(old) as old_lock_descriptor:
        before_lock_identity = _assert_old_lock_descriptor(old, old_lock_descriptor)
        snapshot = _stable_old_snapshot(old)
        expected = _build_receipt(
            old_snapshot=snapshot,
            new_generation_id=generation_id,
            new_plan=validated_new_plan,
            new_schedule=validated_new_schedule,
            new_control_manifest=new_control_manifest,
            new_generation_bindings=new_generation_bindings,
        )
        matches = _receipt_files_for_predecessor(
            store,
            predecessor_state_sha256=expected["predecessor_state_sha256"],
            recoverable_receipt=expected,
        )
        if matches:
            path, existing = matches[0]
            if path.name != _receipt_filename(existing):
                raise DojoHistoricalSupersedeReceiptError(
                    "stored supersede receipt filename is not content addressed"
                )
            verified = _validate_receipt_against_snapshot(
                existing,
                old_snapshot=snapshot,
                new_generation_id=generation_id,
                new_plan=validated_new_plan,
                new_schedule=validated_new_schedule,
                new_control_manifest=new_control_manifest,
                new_generation_bindings=new_generation_bindings,
            )
            _reserve_or_verify_unique_successor(
                old_root=old, receipt=existing, create=True
            )
            _write_exclusive(store / _receipt_filename(existing), existing)
            _reverify_canonical_published_receipt(
                old_root=old,
                new_root=new,
                receipt=existing,
            )
            if (
                _assert_old_lock_descriptor(old, old_lock_descriptor)
                != before_lock_identity
            ):
                raise DojoHistoricalSupersedeReceiptError(
                    "old historical lock changed during supersede creation"
                )
            return verified
        _reserve_or_verify_unique_successor(old_root=old, receipt=expected, create=True)
        _write_exclusive(store / _receipt_filename(expected), expected)
        _reverify_canonical_published_receipt(
            old_root=old,
            new_root=new,
            receipt=expected,
        )
        if (
            _assert_old_lock_descriptor(old, old_lock_descriptor)
            != before_lock_identity
        ):
            raise DojoHistoricalSupersedeReceiptError(
                "old historical lock changed during supersede creation"
            )
        return expected


def _lock_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _assert_old_lock_descriptor(
    old_root: Path, descriptor: int
) -> tuple[int, int, int, int, int]:
    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 0
    ):
        raise DojoHistoricalSupersedeReceiptError("old lock descriptor is invalid")
    lock_path = old_root / ".historical-train.lock"
    try:
        path_state = lock_path.stat(follow_symlinks=False)
        descriptor_state = os.fstat(descriptor)
    except OSError as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "old lock descriptor is unavailable"
        ) from exc
    if (
        lock_path.is_symlink()
        or not stat.S_ISREG(path_state.st_mode)
        or (path_state.st_dev, path_state.st_ino)
        != (descriptor_state.st_dev, descriptor_state.st_ino)
    ):
        raise DojoHistoricalSupersedeReceiptError(
            "old lock descriptor does not bind the configured lock"
        )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "old lock descriptor does not own the generation lock"
        ) from exc
    try:
        after_path = lock_path.stat(follow_symlinks=False)
        after_descriptor = os.fstat(descriptor)
    except OSError as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "old lock changed while its descriptor was verified"
        ) from exc
    identities = {
        _lock_identity(value)
        for value in (path_state, descriptor_state, after_path, after_descriptor)
    }
    if len(identities) != 1:
        raise DojoHistoricalSupersedeReceiptError(
            "old lock path or descriptor changed while verified"
        )
    return identities.pop()


def _verify_for_roots(
    receipt: Mapping[str, Any], *, old: Path, new: Path
) -> dict[str, Any]:
    (
        _,
        validated_new_plan,
        validated_new_schedule,
        new_control_manifest,
        new_generation_bindings,
        generation_id,
    ) = _validated_new_generation(new)
    snapshot = _stable_old_snapshot(old)
    verified = _validate_receipt_against_snapshot(
        receipt,
        old_snapshot=snapshot,
        new_generation_id=generation_id,
        new_plan=validated_new_plan,
        new_schedule=validated_new_schedule,
        new_control_manifest=new_control_manifest,
        new_generation_bindings=new_generation_bindings,
    )
    _reserve_or_verify_unique_successor(old_root=old, receipt=verified, create=False)
    return verified


def verify_historical_supersede_receipt(
    receipt: Mapping[str, Any],
    *,
    old_root: Path,
    new_root: Path,
) -> dict[str, Any]:
    """Purely verify one receipt against unchanged old and new artifacts."""

    old = _safe_root(old_root, field="old historical root")
    new = _safe_root(new_root, field="new historical root")
    with _unowned_old_lock(old):
        return _verify_for_roots(receipt, old=old, new=new)


def verify_historical_supersede_receipt_locked(
    receipt: Mapping[str, Any],
    *,
    old_root: Path,
    new_root: Path,
    old_lock_descriptor: int,
) -> dict[str, Any]:
    """Verify while the caller already holds the exact old generation lock."""

    old = _safe_root(old_root, field="old historical root")
    new = _safe_root(new_root, field="new historical root")
    before = _assert_old_lock_descriptor(old, old_lock_descriptor)
    verified = _verify_for_roots(receipt, old=old, new=new)
    after = _assert_old_lock_descriptor(old, old_lock_descriptor)
    if before != after:
        raise DojoHistoricalSupersedeReceiptError(
            "old historical lock changed during locked receipt verification"
        )
    return verified


def verify_historical_supersede_receipt_store_locked(
    *,
    old_root: Path,
    new_root: Path,
    old_lock_descriptor: int,
) -> dict[str, Any]:
    """Verify the unique predecessor receipt while its exact old lock is held."""

    old = _safe_root(old_root, field="old historical root")
    new = _safe_root(new_root, field="new historical root")
    before = _assert_old_lock_descriptor(old, old_lock_descriptor)
    store = _receipt_directory(new, create=False)
    candidates = []
    for _, receipt in _local_v2_receipts(store):
        old_generation = receipt.get("old_generation")
        if isinstance(old_generation, Mapping) and old_generation.get(
            "root"
        ) == os.fspath(old):
            candidates.append(receipt)
    if len(candidates) != 1:
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt store does not contain one exact predecessor"
        )
    verified = _verify_for_roots(candidates[0], old=old, new=new)
    matches = _receipt_files_for_predecessor(
        store,
        predecessor_state_sha256=verified["predecessor_state_sha256"],
    )
    if len(matches) != 1:
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt store predecessor is ambiguous"
        )
    after = _assert_old_lock_descriptor(old, old_lock_descriptor)
    if before != after:
        raise DojoHistoricalSupersedeReceiptError(
            "old historical lock changed during locked store verification"
        )
    return verified


def verify_historical_supersede_receipt_file(
    receipt_path: Path,
    *,
    old_root: Path,
    new_root: Path,
) -> dict[str, Any]:
    """Verify a receipt and enforce uniqueness in its append-only store."""

    path = Path(receipt_path)
    receipt = read_bounded_json_artifact(path, field="supersede receipt")
    old = _safe_root(old_root, field="old historical root")
    new = _safe_root(new_root, field="new historical root")
    store = _receipt_directory(new, create=False)
    if path.parent.resolve(strict=True) != store or path.name != _receipt_filename(
        receipt
    ):
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt path is not content addressed"
        )
    matches = _receipt_files_for_predecessor(
        store,
        predecessor_state_sha256=receipt.get("predecessor_state_sha256"),
    )
    if len(matches) != 1 or matches[0][0].name != path.name:
        raise DojoHistoricalSupersedeReceiptError(
            "supersede receipt store identity is ambiguous"
        )
    return verify_historical_supersede_receipt(
        receipt,
        old_root=old,
        new_root=new,
    )


__all__ = [
    "CONTRACT",
    "DojoHistoricalSupersedeReceiptError",
    "create_historical_supersede_receipt",
    "read_bounded_json_artifact",
    "verify_historical_supersede_receipt",
    "verify_historical_supersede_receipt_file",
    "verify_historical_supersede_receipt_locked",
    "verify_historical_supersede_receipt_store_locked",
]

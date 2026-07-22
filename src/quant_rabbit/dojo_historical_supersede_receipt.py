"""Append-only custody receipt for superseding an orphaned historical run.

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


CONTRACT: Final = "QR_DOJO_HISTORICAL_GENERATION_SUPERSEDE_RECEIPT_V1"
SCHEMA_VERSION: Final = 1
MAX_JSON_BYTES: Final = 16 * 1024 * 1024
MAX_STATE_FILES: Final = 4096
MAX_STATE_BYTES: Final = 512 * 1024 * 1024
MAX_RECEIPTS: Final = 1024
HASH_CHUNK_BYTES: Final = 1024 * 1024

_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GENERATION_LABEL_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,62}\Z")
_JOB_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CLAIM_DIRECTORY_RE: Final = re.compile(r"attempt-[0-9]{4}-[0-9a-f]{64}\Z")
_CLAIM_FILE_RE: Final = re.compile(r"attempt-[0-9]{4}\.json\Z")
_TERMINAL_FILE_RE: Final = re.compile(
    r"attempt-[0-9]{4}-[0-9a-f]{64}-[0-9a-f]{64}\.json\Z"
)
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
    "reason": "ORPHANED_ACTIVE_CLAIM_SUPERSEDED_WITHOUT_STATE_MUTATION",
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


def _known_state_path(relative: str, *, schedule: Mapping[str, Any]) -> bool:
    parts = PurePosixPath(relative).parts
    jobs = {job["job_sha256"]: job for job in schedule["jobs"]}
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
    if parts == ("execution-manifest.json",):
        return True
    if len(parts) == 3 and parts[0] == "claims":
        return parts[1] in jobs and _CLAIM_FILE_RE.fullmatch(parts[2]) is not None
    if len(parts) == 4 and parts[0] == "cells":
        job = jobs.get(parts[1])
        return (
            job is not None
            and _CLAIM_DIRECTORY_RE.fullmatch(parts[2]) is not None
            and parts[3].endswith(".json")
            and parts[3][:-5]
            in {coordinate["coordinate_id"] for coordinate in job["coordinates"]}
        )
    if len(parts) == 3 and parts[0] == "terminals":
        return parts[1] in jobs and _TERMINAL_FILE_RE.fullmatch(parts[2]) is not None
    if len(parts) == 3 and parts[0] == "reducers":
        return (
            parts[1] in jobs
            and parts[2].endswith(".json")
            and _JOB_SHA_RE.fullmatch(parts[2][:-5]) is not None
        )
    if len(parts) == 2 and parts[0] == "carry":
        return parts[1].endswith(".json") and parts[1][:-5] in carry_slots
    return False


def _known_state_directory(relative: str, *, schedule: Mapping[str, Any]) -> bool:
    parts = PurePosixPath(relative).parts
    jobs = {job["job_sha256"] for job in schedule["jobs"]}
    if len(parts) == 1:
        return parts[0] in {"claims", "cells", "terminals", "carry", "reducers"}
    if len(parts) == 2 and parts[0] in {
        "claims",
        "cells",
        "terminals",
        "reducers",
    }:
        return parts[1] in jobs
    if len(parts) == 3 and parts[0] == "cells":
        return parts[1] in jobs and _CLAIM_DIRECTORY_RE.fullmatch(parts[2]) is not None
    return False


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
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
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
            relative_directory = child.relative_to(root).as_posix()
            if not _known_state_directory(relative_directory, schedule=schedule):
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state contains an unknown directory"
                )
        for name in sorted(file_names):
            child = directory_path / name
            if child.suffix != ".json":
                raise DojoHistoricalSupersedeReceiptError(
                    "old execution-state contains a non-JSON artifact"
                )
            relative = _safe_relative(
                child.relative_to(root).as_posix(), field="state inventory path"
            )
            if not _known_state_path(relative, schedule=schedule):
                raise DojoHistoricalSupersedeReceiptError(
                    f"old execution-state contains an unknown artifact: {relative}"
                )
            raw = _stable_regular_bytes(child, field=f"state artifact {relative}")
            _parse_json(raw, field=f"state artifact {relative}")
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
    if not rows:
        raise DojoHistoricalSupersedeReceiptError(
            "old execution-state inventory is empty"
        )
    body = {
        "root_relative_path": "execution-state",
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
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], str]:
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
        manifest.get("contract") != "QR_DOJO_HISTORICAL_TRAIN_GENERATION_MANIFEST_V1"
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
    identity = canonical_sha256(
        {
            "generation": label,
            "plan_sha256": validated_plan["plan_sha256"],
            "schedule_sha256": validated_schedule["schedule_sha256"],
            "control_manifest_sha256": manifest["manifest_sha256"],
        }
    )
    return (
        root,
        validated_plan,
        validated_schedule,
        manifest,
        f"{label}:{identity}",
    )


@contextmanager
def _unowned_old_lock(old_root: Path) -> Iterator[None]:
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
        yield
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
    if isinstance(active, bool) or not isinstance(active, int) or active != 1:
        raise DojoHistoricalSupersedeReceiptError(
            "old execution state must have exactly one orphaned active claim"
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
) -> dict[str, Any]:
    return {
        "old_root": old_snapshot["root"],
        "old_plan_sha256": old_snapshot["plan_sha256"],
        "old_schedule_sha256": old_snapshot["schedule_sha256"],
        "new_generation_id": new_generation_id,
        "new_plan_sha256": new_plan["plan_sha256"],
        "new_schedule_sha256": new_schedule["schedule_sha256"],
        "new_control_manifest_sha256": new_control_manifest["manifest_sha256"],
    }


def _build_receipt(
    *,
    old_snapshot: Mapping[str, Any],
    new_generation_id: str,
    new_plan: Mapping[str, Any],
    new_schedule: Mapping[str, Any],
    new_control_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _identity_body(
        old_snapshot=old_snapshot,
        new_generation_id=new_generation_id,
        new_plan=new_plan,
        new_schedule=new_schedule,
        new_control_manifest=new_control_manifest,
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
            "control_manifest_sha256": new_control_manifest["manifest_sha256"],
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


def _receipt_files_for_predecessor(
    receipt_dir: Path, *, predecessor_state_sha256: str
) -> list[tuple[Path, dict[str, Any]]]:
    predecessor = _strict_sha(
        predecessor_state_sha256, field="predecessor state SHA-256"
    )
    entries = sorted(receipt_dir.iterdir())
    if len(entries) > MAX_RECEIPTS:
        raise DojoHistoricalSupersedeReceiptError(
            "append-only supersede receipt store exceeds its file bound"
        )
    matches = []
    for path in entries:
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise DojoHistoricalSupersedeReceiptError(
                "append-only supersede receipt store contains an unknown entry"
            )
        value = read_bounded_json_artifact(path, field="stored supersede receipt")
        body = {key: item for key, item in value.items() if key != "receipt_sha256"}
        if (
            value.get("contract") != CONTRACT
            or value.get("schema_version") != SCHEMA_VERSION
            or value.get("receipt_sha256") != canonical_sha256(body)
            or path.name != _receipt_filename(value)
        ):
            raise DojoHistoricalSupersedeReceiptError(
                "append-only supersede receipt store contains an invalid receipt"
            )
        if value.get("predecessor_state_sha256") == predecessor:
            matches.append((path, value))
    if len(matches) > 1:
        raise DojoHistoricalSupersedeReceiptError(
            "multiple supersede receipts name the same predecessor state"
        )
    return matches


def _receipt_filename(receipt: Mapping[str, Any]) -> str:
    identity = _strict_sha(
        receipt.get("transition_identity_sha256"),
        field="transition identity SHA-256",
    )
    digest = _strict_sha(receipt.get("receipt_sha256"), field="receipt SHA-256")
    return f"supersede-{identity}-{digest}.json"


def _write_exclusive(path: Path, receipt: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(receipt) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            if handle.write(payload) != len(payload):
                raise DojoHistoricalSupersedeReceiptError(
                    "supersede receipt write was incomplete"
                )
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as exc:
        raise DojoHistoricalSupersedeReceiptError(
            "content-addressed supersede receipt already exists"
        ) from exc


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
        generation_id,
    ) = _validated_new_generation(new_root)
    if new == old or old in new.parents or new in old.parents:
        raise DojoHistoricalSupersedeReceiptError(
            "old and new historical roots must be isolated"
        )
    store = _receipt_directory(new, create=True)
    with _unowned_old_lock(old):
        snapshot = _stable_old_snapshot(old)
        expected = _build_receipt(
            old_snapshot=snapshot,
            new_generation_id=generation_id,
            new_plan=validated_new_plan,
            new_schedule=validated_new_schedule,
            new_control_manifest=new_control_manifest,
        )
        matches = _receipt_files_for_predecessor(
            store,
            predecessor_state_sha256=expected["predecessor_state_sha256"],
        )
        if matches:
            path, existing = matches[0]
            if path.name != _receipt_filename(existing):
                raise DojoHistoricalSupersedeReceiptError(
                    "stored supersede receipt filename is not content addressed"
                )
            return _validate_receipt_against_snapshot(
                existing,
                old_snapshot=snapshot,
                new_generation_id=generation_id,
                new_plan=validated_new_plan,
                new_schedule=validated_new_schedule,
                new_control_manifest=new_control_manifest,
            )
        _write_exclusive(store / _receipt_filename(expected), expected)
        return expected


def _assert_old_lock_descriptor(old_root: Path, descriptor: int) -> None:
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


def _verify_for_roots(
    receipt: Mapping[str, Any], *, old: Path, new: Path
) -> dict[str, Any]:
    (
        _,
        validated_new_plan,
        validated_new_schedule,
        new_control_manifest,
        generation_id,
    ) = _validated_new_generation(new)
    snapshot = _stable_old_snapshot(old)
    return _validate_receipt_against_snapshot(
        receipt,
        old_snapshot=snapshot,
        new_generation_id=generation_id,
        new_plan=validated_new_plan,
        new_schedule=validated_new_schedule,
        new_control_manifest=new_control_manifest,
    )


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
    _assert_old_lock_descriptor(old, old_lock_descriptor)
    return _verify_for_roots(receipt, old=old, new=new)


def verify_historical_supersede_receipt_store_locked(
    *,
    old_root: Path,
    new_root: Path,
    old_lock_descriptor: int,
) -> dict[str, Any]:
    """Verify the unique predecessor receipt while its exact old lock is held."""

    old = _safe_root(old_root, field="old historical root")
    new = _safe_root(new_root, field="new historical root")
    _assert_old_lock_descriptor(old, old_lock_descriptor)
    store = _receipt_directory(new, create=False)
    candidates = []
    for path in sorted(store.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise DojoHistoricalSupersedeReceiptError(
                "append-only supersede receipt store contains an unknown entry"
            )
        receipt = read_bounded_json_artifact(path, field="stored supersede receipt")
        if path.name != _receipt_filename(receipt):
            raise DojoHistoricalSupersedeReceiptError(
                "stored supersede receipt filename is not content addressed"
            )
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

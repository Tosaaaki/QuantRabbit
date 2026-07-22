"""Fail-closed reclaim for the legacy per-cell Drive archive workflow.

The legacy trainer archived one coordinate at a time with
``dojo_drive_archive``.  A cell archive contains three shared terminal JSON
artifacts plus that coordinate's main and LOPO session directories.  This
module accepts only externally-authored ``REMOTE_VERIFIED`` receipts, rebuilds
their exact local plan/finalization/archive lineage, and permits unlink of the
three known raw files inside remotely verified, uniquely-owned session
directories.  Cells without a remote receipt are enumerated as an excluded
set and never enter the unlink plan.

Verification is read-only.  Reclaim first publishes an append-only plan under
an exclusive run lock; a crash can therefore resume only the same sealed
target set.  The old archive receipts remain immutable and continue to state
``source_deleted=false``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final

import quant_rabbit.dojo_drive_archive as drive_archive


REMOTE_RECEIPT_CONTRACT: Final = "QR_DOJO_DRIVE_ARCHIVE_REMOTE_VERIFIED_RECEIPT_V1"
LEGACY_REMOTE_RECEIPT_CONTRACT: Final = (
    "QR_DOJO_DRIVE_ARCHIVE_REMOTE_VERIFIED_RECEIPT_LEGACY_V0"
)
RECLAIM_PLAN_CONTRACT: Final = "QR_DOJO_LEGACY_CELL_RAW_RECLAIM_PLAN_V1"
RECLAIM_RECEIPT_CONTRACT: Final = "QR_DOJO_LEGACY_CELL_RAW_RECLAIM_V1"
SCHEMA_VERSION: Final = 1
MAX_JSON_BYTES: Final = 256 * 1024 * 1024
MAX_REMOTE_RECEIPTS: Final = 128
HASH_CHUNK_BYTES: Final = 1024 * 1024
RAW_FILENAMES: Final = frozenset({"broker_snapshot.json", "ledger.jsonl", "state.json"})
SHARED_FILENAMES: Final = frozenset({"cells.json", "evaluation.json", "run.json"})

_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_MD5_RE: Final = re.compile(r"[0-9a-f]{32}\Z")
_DRIVE_ID_RE: Final = re.compile(r"[A-Za-z0-9_-]{8,256}\Z")
_REMOTE_BASE_KEYS: Final = {
    "archive_local_md5",
    "archive_local_path",
    "archive_local_size_bytes",
    "archive_name",
    "archive_remote_md5",
    "archive_remote_size_bytes",
    "archive_sha256",
    "broker_mutation_allowed",
    "checked_at_utc",
    "classification",
    "content_tree_sha256",
    "coordinate",
    "drive_file_id",
    "drive_file_name",
    "drive_modified_time",
    "drive_parent_id",
    "drive_parent_name",
    "evaluation_artifact_sha256",
    "evaluation_sha256",
    "finalization_sha256",
    "live_permission",
    "local_payload_verified",
    "order_authority",
    "plan_sha256",
    "promotion_eligible",
    "proof_eligible",
    "receipt_sha256",
    "remote_verified",
    "source_deleted",
    "source_run_artifact_sha256",
    "source_run_dir",
    "source_run_sha256",
    "status",
    "study_sha256",
}
_REMOTE_V1_KEYS: Final = _REMOTE_BASE_KEYS | {"contract", "schema_version"}
_COORDINATE_KEYS: Final = {"candidate_id", "intrabar", "cost_arm"}
_COORDINATE_FIELDS: Final = ("candidate_id", "intrabar", "cost_arm")
_TARGET_KEYS: Final = {
    "coordinate_id",
    "path",
    "raw_kind",
    "sha256",
    "size_bytes",
}
_VERIFIED_CELL_KEYS: Final = {
    "archive_md5",
    "archive_path",
    "archive_sha256",
    "archive_size_bytes",
    "content_tree_sha256",
    "coordinate_id",
    "drive_file_id",
    "drive_parent_id",
    "finalization_sha256",
    "plan_path",
    "plan_sha256",
    "receipt_contract",
    "receipt_path",
    "receipt_sha256",
    "session_prefixes",
}
_EXCLUDED_FILE_KEYS: Final = {"coordinate_id", "path", "size_bytes"}
_RECLAIM_PLAN_BODY_KEYS: Final = {
    "archive_root",
    "broker_mutation_allowed",
    "cells_artifact_sha256",
    "contract",
    "evaluation_artifact_sha256",
    "evaluation_sha256",
    "expected_drive_parent_id",
    "full_verified_archive_inventory_rehashed",
    "historical_train_is_proof",
    "live_permission",
    "order_authority",
    "promotion_eligible",
    "proof_eligible",
    "reclaim_mode",
    "remote_receipt_set_sha256",
    "remote_receipts_dir",
    "remote_unverified_cells_excluded",
    "schema_version",
    "shared_terminal_files",
    "source_deleted",
    "source_deletion_allowed",
    "source_run_artifact_sha256",
    "source_run_root",
    "source_run_sha256",
    "study_sha256",
    "target_bytes",
    "target_count",
    "targets",
    "unverified_cell_count",
    "unverified_coordinate_ids",
    "unverified_raw_file_count",
    "unverified_raw_files",
    "verified_cell_count",
    "verified_cells",
}
_RECLAIM_PLAN_KEYS: Final = _RECLAIM_PLAN_BODY_KEYS | {"reclaim_plan_sha256"}
_RECLAIM_RECEIPT_BODY_KEYS: Final = {
    "broker_mutation_allowed",
    "completed_at_utc",
    "contract",
    "deleted_file_count",
    "deleted_files",
    "free_disk_bytes_after",
    "free_disk_bytes_before",
    "historical_train_is_proof",
    "live_permission",
    "order_authority",
    "promotion_eligible",
    "proof_eligible",
    "reclaim_plan_sha256",
    "reclaimed_allocated_bytes_observed",
    "reclaimed_logical_bytes",
    "remote_receipt_set_sha256",
    "remote_unverified_cells_excluded",
    "restore_requires_verified_cell_archives",
    "schema_version",
    "source_run_sha256",
    "status",
    "unverified_cell_count",
    "verified_cell_count",
}
_RECLAIM_RECEIPT_KEYS: Final = _RECLAIM_RECEIPT_BODY_KEYS | {"reclaim_receipt_sha256"}
_AUTHORITY: Final = {
    "historical_train_is_proof": False,
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}


class DojoLegacyCellRawReclaimError(ValueError):
    """Legacy raw evidence is not safely reclaimable."""


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
        raise DojoLegacyCellRawReclaimError("value is not strict JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact(
    value: Any, keys: set[str] | frozenset[str], field: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise DojoLegacyCellRawReclaimError(f"{field} schema is invalid")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoLegacyCellRawReclaimError(f"{field} is not a SHA-256")
    return value


def _md5(value: Any, field: str) -> str:
    if not isinstance(value, str) or _MD5_RE.fullmatch(value) is None:
        raise DojoLegacyCellRawReclaimError(f"{field} is not an MD5")
    return value


def _drive_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _DRIVE_ID_RE.fullmatch(value) is None:
        raise DojoLegacyCellRawReclaimError(f"{field} is not a Drive id")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DojoLegacyCellRawReclaimError(f"{field} must be positive")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DojoLegacyCellRawReclaimError(f"{field} must be nonnegative")
    return value


def _utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise DojoLegacyCellRawReclaimError(f"{field} is invalid")
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoLegacyCellRawReclaimError(f"{field} is not an instant") from exc
    if instant.tzinfo is None or instant.utcoffset() != timezone.utc.utcoffset(instant):
        raise DojoLegacyCellRawReclaimError(f"{field} must be UTC-aware")
    return instant


def _safe_relative(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DojoLegacyCellRawReclaimError("relative path is unsafe")
    return pure.as_posix()


def _safe_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    safe = _safe_relative(relative)
    path = root / PurePosixPath(safe)
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as exc:
        raise DojoLegacyCellRawReclaimError(f"path is unavailable: {relative}") from exc
    if resolved == root or root not in resolved.parents:
        raise DojoLegacyCellRawReclaimError(f"path escapes run root: {relative}")
    return path


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = drive_archive._load_json(path, field=field)
    except drive_archive.DojoDriveArchiveError as exc:
        raise DojoLegacyCellRawReclaimError(str(exc)) from exc
    if not isinstance(value, dict):
        raise DojoLegacyCellRawReclaimError(f"{field} must be an object")
    return value


def _hash_file(path: Path) -> tuple[str, str, int, int]:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise DojoLegacyCellRawReclaimError(f"not a regular file: {path}")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        sha256 = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        size = 0
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened_before = os.fstat(handle.fileno())
            while chunk := handle.read(HASH_CHUNK_BYTES):
                sha256.update(chunk)
                md5.update(chunk)
                size += len(chunk)
            opened_after = os.fstat(handle.fileno())
        after = path.stat(follow_symlinks=False)
    except DojoLegacyCellRawReclaimError:
        raise
    except OSError as exc:
        raise DojoLegacyCellRawReclaimError(f"cannot hash file: {path}") from exc

    def identity(row: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )

    if (
        identity(before) != identity(opened_before)
        or identity(opened_before) != identity(opened_after)
        or identity(opened_after) != identity(after)
        or size != before.st_size
    ):
        raise DojoLegacyCellRawReclaimError(f"file changed while hashed: {path}")
    return sha256.hexdigest(), md5.hexdigest(), size, before.st_blocks * 512


def _coordinate_id(value: Mapping[str, Any]) -> str:
    coordinate = _exact(value, _COORDINATE_KEYS, "coordinate")
    parts = []
    for field in _COORDINATE_FIELDS:
        item = coordinate[field]
        if not isinstance(item, str) or not item or "|" in item:
            raise DojoLegacyCellRawReclaimError(f"coordinate {field} is invalid")
        parts.append(item)
    coordinate_id = "|".join(parts)
    if drive_archive._CHUNK_ID_RE.fullmatch(coordinate_id) is None:
        raise DojoLegacyCellRawReclaimError("coordinate id is unsafe")
    return coordinate_id


def _run_coordinate_id(value: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for field in _COORDINATE_FIELDS:
        item = value.get(field)
        if not isinstance(item, str) or not item or "|" in item:
            raise DojoLegacyCellRawReclaimError(f"run coordinate {field} is invalid")
        parts.append(item)
    coordinate_id = "|".join(parts)
    if drive_archive._CHUNK_ID_RE.fullmatch(coordinate_id) is None:
        raise DojoLegacyCellRawReclaimError("run coordinate id is unsafe")
    return coordinate_id


def _relative_from_declared(root: Path, value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DojoLegacyCellRawReclaimError(f"{field} is invalid")
    candidate = Path(value)
    absolute = Path(
        os.path.abspath(candidate if candidate.is_absolute() else root / candidate)
    )
    try:
        return _safe_relative(absolute.relative_to(root).as_posix())
    except ValueError as exc:
        raise DojoLegacyCellRawReclaimError(f"{field} escapes run root") from exc


def _session_owners(
    root: Path, run: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    coordinates: dict[str, Mapping[str, Any]] = {}
    owners: dict[str, str] = {}
    raw_coordinates = run.get("coordinates")
    if not isinstance(raw_coordinates, list):
        raise DojoLegacyCellRawReclaimError("run coordinates are invalid")
    for index, raw in enumerate(raw_coordinates):
        if not isinstance(raw, Mapping):
            raise DojoLegacyCellRawReclaimError("run coordinate is invalid")
        coordinate_id = _run_coordinate_id(raw)
        if coordinate_id in coordinates:
            raise DojoLegacyCellRawReclaimError("duplicate run coordinate")
        coordinates[coordinate_id] = raw
        declared = [("main_session_dir", raw.get("main_session_dir"))]
        lopo = raw.get("lopo")
        if not isinstance(lopo, list):
            raise DojoLegacyCellRawReclaimError("coordinate LOPO is invalid")
        declared.extend(
            (f"lopo[{lopo_index}].session_dir", row.get("session_dir"))
            for lopo_index, row in enumerate(lopo)
            if isinstance(row, Mapping)
        )
        if len(declared) != 1 + len(lopo):
            raise DojoLegacyCellRawReclaimError("coordinate LOPO row is invalid")
        for field, value in declared:
            relative = _relative_from_declared(
                root, value, field=f"coordinates[{index}].{field}"
            )
            if not PurePosixPath(relative).parts[0] == "sessions":
                raise DojoLegacyCellRawReclaimError(
                    "legacy raw session is outside sessions/"
                )
            if relative in owners:
                raise DojoLegacyCellRawReclaimError(
                    "a legacy session is shared across cells"
                )
            owners[relative] = coordinate_id
    return coordinates, owners


def _verify_empty_runtime_inbox(path: Path) -> None:
    state = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(state.st_mode) or path.is_symlink():
        raise DojoLegacyCellRawReclaimError(
            "legacy session inbox is not a real directory"
        )
    inbox_children = sorted(path.iterdir(), key=lambda child: child.name)
    if len(inbox_children) != 1 or inbox_children[0].name != "processed":
        raise DojoLegacyCellRawReclaimError("legacy session inbox shape is invalid")
    processed = inbox_children[0]
    processed_state = processed.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(processed_state.st_mode)
        or processed.is_symlink()
        or any(processed.iterdir())
    ):
        raise DojoLegacyCellRawReclaimError(
            "legacy session processed inbox is not an empty real directory"
        )


def _files_under(root: Path, prefix: str) -> list[str]:
    directory = _safe_path(root, prefix)
    try:
        state = directory.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoLegacyCellRawReclaimError("session directory is unavailable") from exc
    if not stat.S_ISDIR(state.st_mode):
        raise DojoLegacyCellRawReclaimError("session path is not a directory")
    rows: list[str] = []
    for item in sorted(directory.iterdir(), key=lambda path: path.name):
        item_state = item.stat(follow_symlinks=False)
        if item.name == "inbox":
            _verify_empty_runtime_inbox(item)
            continue
        if not stat.S_ISREG(item_state.st_mode) or item.is_symlink():
            raise DojoLegacyCellRawReclaimError(
                "legacy session contains a non-regular direct child"
            )
        if item.name not in RAW_FILENAMES:
            raise DojoLegacyCellRawReclaimError(
                f"legacy session file is outside the raw allowlist: {item.name}"
            )
        rows.append(item.relative_to(root).as_posix())
    if set(Path(row).name for row in rows) != RAW_FILENAMES:
        raise DojoLegacyCellRawReclaimError(
            "legacy session does not contain the exact raw allowlist"
        )
    return rows


def _coordinate_prefixes(
    root: Path,
    coordinate_id: str,
    coordinate: Mapping[str, Any],
    owners: Mapping[str, str],
) -> list[str]:
    prefixes = [
        _relative_from_declared(
            root, coordinate.get("main_session_dir"), field="main_session_dir"
        )
    ]
    lopo = coordinate.get("lopo")
    if not isinstance(lopo, list):
        raise DojoLegacyCellRawReclaimError("coordinate LOPO is invalid")
    prefixes.extend(
        _relative_from_declared(root, row.get("session_dir"), field="LOPO session")
        for row in lopo
        if isinstance(row, Mapping)
    )
    if (
        len(prefixes) != 1 + len(lopo)
        or len(set(prefixes)) != len(prefixes)
        or any(owners.get(prefix) != coordinate_id for prefix in prefixes)
    ):
        raise DojoLegacyCellRawReclaimError(
            "cell sessions are not uniquely owned by its coordinate"
        )
    return sorted(prefixes)


def _archived_rows_for_prefixes(
    *,
    plan: Mapping[str, Any],
    coordinate_id: str,
    prefixes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_paths = {
        PurePosixPath(prefix, filename).as_posix()
        for prefix in prefixes
        for filename in RAW_FILENAMES
    }
    targets: list[dict[str, Any]] = []
    shared: list[dict[str, Any]] = []
    for raw in plan["files"]:
        row = dict(raw)
        path = row["path"]
        if path in expected_paths:
            targets.append(
                {
                    **row,
                    "coordinate_id": coordinate_id,
                    "raw_kind": "SESSION_RAW",
                }
            )
        elif path in SHARED_FILENAMES:
            shared.append(row)
        else:
            raise DojoLegacyCellRawReclaimError(
                f"cell archive contains a non-allowlisted file: {path}"
            )
    if {row["path"] for row in targets} != expected_paths or {
        row["path"] for row in shared
    } != SHARED_FILENAMES:
        raise DojoLegacyCellRawReclaimError(
            "cell archive does not exactly cover shared plus cell raw files"
        )
    return sorted(targets, key=lambda row: row["path"]), sorted(
        shared, key=lambda row: row["path"]
    )


def _archive_plan_path(
    archive_root: Path, coordinate_id: str, plan_sha256: str
) -> Path:
    return archive_root / "plans" / f"cell-{coordinate_id}-{plan_sha256}.json"


def _archive_receipt_path(
    archive_root: Path, coordinate_id: str, plan_sha256: str
) -> Path:
    return archive_root / "receipts" / f"cell-{coordinate_id}-{plan_sha256}.json"


def _static_archive_bundle(
    *,
    archive_root: Path,
    coordinate_id: str,
    plan_sha256: str,
    zstd_bin: str,
    fresh_source_verification: bool,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    plan_path = _archive_plan_path(archive_root, coordinate_id, plan_sha256)
    if fresh_source_verification:
        try:
            drive_archive.verify_finalized_archive(
                plan_path=plan_path, zstd_bin=zstd_bin
            )
        except drive_archive.DojoDriveArchiveError as exc:
            raise DojoLegacyCellRawReclaimError(str(exc)) from exc
    plan = _load_json(plan_path, field="archive plan")
    _exact(plan, drive_archive._PLAN_KEYS, "archive plan")
    try:
        drive_archive._sealed_body(plan, "plan_sha256", "archive plan")
    except drive_archive.DojoDriveArchiveError as exc:
        raise DojoLegacyCellRawReclaimError(str(exc)) from exc
    if (
        plan.get("contract") != drive_archive.PLAN_CONTRACT
        or plan.get("schema_version") != 1
        or plan.get("chunk_kind") != "cell"
        or plan.get("chunk_id") != coordinate_id
        or plan.get("plan_sha256") != plan_sha256
        or plan.get("source_deletion_allowed") is not False
        or plan.get("source_deleted") is not False
        or plan.get("proof_eligible") is not False
        or plan.get("promotion_eligible") is not False
        or plan.get("live_permission") is not False
        or plan.get("order_authority") != "NONE"
        or plan.get("broker_mutation_allowed") is not False
        or plan.get("remote_verification")
        != {
            "status": "NOT_REQUESTED",
            "remote_verified": False,
            "metadata_receipt_sha256": None,
        }
    ):
        raise DojoLegacyCellRawReclaimError("archive plan boundary is invalid")
    files = plan.get("files")
    if (
        not isinstance(files, list)
        or not files
        or files != sorted(files, key=lambda row: row.get("path", ""))
        or plan.get("file_count") != len(files)
        or plan.get("content_tree_sha256") != drive_archive.canonical_sha256(files)
        or plan.get("total_source_bytes")
        != sum(
            row.get("size_bytes", -1) if isinstance(row, Mapping) else -1
            for row in files
        )
    ):
        raise DojoLegacyCellRawReclaimError("archive plan inventory is invalid")
    for raw in files:
        row = _exact(raw, drive_archive._PLAN_FILE_KEYS, "archive inventory row")
        _safe_relative(str(row.get("path", "")))
        _nonnegative_int(row.get("size_bytes"), "archive inventory size")
        _sha(row.get("sha256"), "archive inventory SHA-256")

    receipt_path = _archive_receipt_path(archive_root, coordinate_id, plan_sha256)
    receipt = _load_json(receipt_path, field="archive finalization")
    _exact(receipt, drive_archive._FINALIZATION_KEYS, "archive finalization")
    try:
        drive_archive._sealed_body(
            receipt, "finalization_sha256", "archive finalization"
        )
    except drive_archive.DojoDriveArchiveError as exc:
        raise DojoLegacyCellRawReclaimError(str(exc)) from exc
    archive_path = (
        archive_root / "archives" / f"cell-{coordinate_id}-{plan_sha256}.tar.zst"
    )
    if (
        receipt.get("contract") != drive_archive.FINALIZATION_CONTRACT
        or receipt.get("schema_version") != 1
        or not drive_archive._same_unicode_canonical_file(
            receipt.get("plan_path", ""), plan_path.resolve(strict=True)
        )
        or receipt.get("plan_sha256") != plan_sha256
        or receipt.get("content_tree_sha256") != plan["content_tree_sha256"]
        or receipt.get("chunk_kind") != "cell"
        or receipt.get("chunk_id") != coordinate_id
        or not drive_archive._same_unicode_canonical_file(
            receipt.get("archive_path", ""), archive_path
        )
        or receipt.get("file_count") != plan["file_count"]
        or receipt.get("total_source_bytes") != plan["total_source_bytes"]
        or receipt.get("local_payload_verified") is not True
        or receipt.get("atomic_publish_complete") is not True
        or receipt.get("source_deletion_allowed") is not False
        or receipt.get("source_deleted") is not False
        or receipt.get("remote_verification") != plan["remote_verification"]
        or receipt.get("proof_eligible") is not False
        or receipt.get("promotion_eligible") is not False
        or receipt.get("live_permission") is not False
        or receipt.get("order_authority") != "NONE"
        or receipt.get("broker_mutation_allowed") is not False
    ):
        raise DojoLegacyCellRawReclaimError("archive finalization lineage is invalid")
    sha256, _, size, _ = _hash_file(archive_path)
    if sha256 != receipt.get("archive_sha256") or size != receipt.get(
        "archive_size_bytes"
    ):
        raise DojoLegacyCellRawReclaimError("archive bytes drifted")
    if not fresh_source_verification:
        try:
            drive_archive._verify_archive_payload(archive_path, plan, zstd_bin=zstd_bin)
        except drive_archive.DojoDriveArchiveError as exc:
            raise DojoLegacyCellRawReclaimError(str(exc)) from exc
    return plan, receipt, archive_path


def _validate_remote_receipt(
    *,
    receipt_path: Path,
    root: Path,
    run: Mapping[str, Any],
    archive_root: Path,
    expected_drive_parent_id: str,
    plan: Mapping[str, Any],
    finalization: Mapping[str, Any],
    archive_path: Path,
) -> dict[str, Any]:
    remote = _load_json(receipt_path, field="remote verification receipt")
    keys = set(remote)
    if keys == _REMOTE_V1_KEYS:
        if (
            remote.get("contract") != REMOTE_RECEIPT_CONTRACT
            or remote.get("schema_version") != 1
        ):
            raise DojoLegacyCellRawReclaimError(
                "remote receipt contract/version is invalid"
            )
        receipt_contract = REMOTE_RECEIPT_CONTRACT
    elif keys == _REMOTE_BASE_KEYS:
        receipt_contract = LEGACY_REMOTE_RECEIPT_CONTRACT
    else:
        raise DojoLegacyCellRawReclaimError("remote receipt schema is invalid")
    body = {key: value for key, value in remote.items() if key != "receipt_sha256"}
    receipt_sha = _sha(remote.get("receipt_sha256"), "remote receipt SHA-256")
    coordinate_id = _coordinate_id(remote.get("coordinate"))
    expected_name = f"REMOTE_VERIFIED__cell-{coordinate_id}__{receipt_sha}.json"
    checked = _utc(remote.get("checked_at_utc"), "remote checked_at_utc")
    modified = _utc(remote.get("drive_modified_time"), "Drive modified time")
    archive_sha, archive_md5, archive_size, _ = _hash_file(archive_path)
    run_sha, _, _, _ = _hash_file(root / "run.json")
    evaluation_sha, _, _, _ = _hash_file(root / "evaluation.json")
    source_run_dir = Path(str(remote.get("source_run_dir", "")))
    local_archive = Path(str(remote.get("archive_local_path", "")))
    if (
        receipt_path.name != expected_name
        or receipt_sha != _sha256(body)
        or remote.get("status") != "REMOTE_VERIFIED"
        or remote.get("remote_verified") is not True
        or remote.get("local_payload_verified") is not True
        or remote.get("source_deleted") is not False
        or remote.get("proof_eligible") is not False
        or remote.get("promotion_eligible") is not False
        or remote.get("live_permission") is not False
        or remote.get("order_authority") != "NONE"
        or remote.get("broker_mutation_allowed") is not False
        or remote.get("classification") != run.get("classification")
        or remote.get("source_run_sha256") != run.get("run_sha256")
        or remote.get("source_run_artifact_sha256") != run_sha
        or remote.get("study_sha256") != run.get("study_sha256")
        or remote.get("evaluation_sha256") != run.get("evaluation_sha256")
        or remote.get("evaluation_artifact_sha256") != evaluation_sha
        or remote.get("plan_sha256") != plan.get("plan_sha256")
        or remote.get("content_tree_sha256") != plan.get("content_tree_sha256")
        or remote.get("finalization_sha256") != finalization.get("finalization_sha256")
        or remote.get("archive_sha256") != archive_sha
        or remote.get("archive_name") != archive_path.name
        or remote.get("drive_file_name") != archive_path.name
        or remote.get("archive_local_size_bytes") != archive_size
        or remote.get("archive_remote_size_bytes") != archive_size
        or remote.get("archive_local_md5") != archive_md5
        or remote.get("archive_remote_md5") != archive_md5
        or _md5(remote.get("archive_local_md5"), "archive local MD5") != archive_md5
        or _md5(remote.get("archive_remote_md5"), "archive remote MD5") != archive_md5
        or _drive_id(remote.get("drive_file_id"), "Drive file id")
        != remote.get("drive_file_id")
        or _drive_id(remote.get("drive_parent_id"), "Drive parent id")
        != expected_drive_parent_id
        or remote.get("drive_parent_name") != "archives"
        or modified > checked
    ):
        raise DojoLegacyCellRawReclaimError(
            "remote receipt Drive metadata or lineage is invalid"
        )
    try:
        if source_run_dir.resolve(strict=True) != root:
            raise DojoLegacyCellRawReclaimError(
                "remote receipt names another source run"
            )
        if not drive_archive._same_unicode_canonical_file(
            local_archive, archive_path.resolve(strict=True)
        ):
            raise DojoLegacyCellRawReclaimError(
                "remote receipt names another local archive"
            )
    except OSError as exc:
        raise DojoLegacyCellRawReclaimError(
            "remote receipt local path is unavailable"
        ) from exc
    return {
        "receipt_contract": receipt_contract,
        "receipt_path": os.fspath(receipt_path),
        "receipt_sha256": receipt_sha,
        "coordinate_id": coordinate_id,
        "drive_file_id": remote["drive_file_id"],
        "drive_parent_id": remote["drive_parent_id"],
        "archive_md5": archive_md5,
    }


def _target_rows(
    *,
    root: Path,
    coordinate_id: str,
    coordinate: Mapping[str, Any],
    owners: Mapping[str, str],
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    prefixes = _coordinate_prefixes(root, coordinate_id, coordinate, owners)
    target_paths = {path for prefix in prefixes for path in _files_under(root, prefix)}
    targets, shared = _archived_rows_for_prefixes(
        plan=plan, coordinate_id=coordinate_id, prefixes=prefixes
    )
    if {row["path"] for row in targets} != target_paths:
        raise DojoLegacyCellRawReclaimError(
            "current cell raw differs from the archived allowlist"
        )
    return targets, shared, prefixes


def _discover_remote_receipts(path: Path) -> list[Path]:
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise DojoLegacyCellRawReclaimError(
            "remote receipt directory is unavailable"
        ) from exc
    if not root.is_dir() or root.is_symlink():
        raise DojoLegacyCellRawReclaimError(
            "remote receipt directory must be a real directory"
        )
    rows = sorted(root.glob("REMOTE_VERIFIED__cell-*.json"))
    if not rows or len(rows) > MAX_REMOTE_RECEIPTS:
        raise DojoLegacyCellRawReclaimError(
            "remote receipt count is outside the bounded range"
        )
    all_entries = sorted(root.iterdir(), key=lambda item: item.name)
    if all_entries != rows or any(
        item.is_symlink() or not item.is_file() for item in rows
    ):
        raise DojoLegacyCellRawReclaimError(
            "remote receipt directory contains foreign or unsafe entries"
        )
    return rows


def _fresh_plan_body(
    *,
    source_run: Path,
    archive_root: Path,
    remote_receipts_dir: Path,
    expected_drive_parent_id: str,
    zstd_bin: str,
) -> dict[str, Any]:
    try:
        root = source_run.resolve(strict=True)
        destination = archive_root.resolve(strict=True)
        remote_root = remote_receipts_dir.resolve(strict=True)
        run = drive_archive.validate_terminal_run(root)
    except (OSError, drive_archive.DojoDriveArchiveError) as exc:
        raise DojoLegacyCellRawReclaimError(str(exc)) from exc
    parent_id = _drive_id(expected_drive_parent_id, "expected Drive parent id")
    coordinates, owners = _session_owners(root, run)
    verified_cells: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    shared_by_path: dict[str, dict[str, Any]] = {}
    verified_ids: set[str] = set()
    target_paths: set[str] = set()
    for remote_path in _discover_remote_receipts(remote_root):
        preliminary = _load_json(remote_path, field="remote verification receipt")
        coordinate_id = _coordinate_id(preliminary.get("coordinate"))
        if coordinate_id not in coordinates or coordinate_id in verified_ids:
            raise DojoLegacyCellRawReclaimError(
                "remote receipt coordinate is duplicate or outside the run"
            )
        plan_sha = _sha(preliminary.get("plan_sha256"), "archive plan SHA-256")
        plan, finalization, archive_path = _static_archive_bundle(
            archive_root=destination,
            coordinate_id=coordinate_id,
            plan_sha256=plan_sha,
            zstd_bin=zstd_bin,
            fresh_source_verification=True,
        )
        remote = _validate_remote_receipt(
            receipt_path=remote_path,
            root=root,
            run=run,
            archive_root=destination,
            expected_drive_parent_id=parent_id,
            plan=plan,
            finalization=finalization,
            archive_path=archive_path,
        )
        cell_targets, shared, prefixes = _target_rows(
            root=root,
            coordinate_id=coordinate_id,
            coordinate=coordinates[coordinate_id],
            owners=owners,
            plan=plan,
        )
        if target_paths & {row["path"] for row in cell_targets}:
            raise DojoLegacyCellRawReclaimError(
                "remotely verified cell target sets overlap"
            )
        target_paths.update(row["path"] for row in cell_targets)
        targets.extend(cell_targets)
        for row in shared:
            prior = shared_by_path.setdefault(row["path"], row)
            if prior != row:
                raise DojoLegacyCellRawReclaimError(
                    "cell archives disagree on shared terminal bytes"
                )
        verified_cells.append(
            {
                "coordinate_id": coordinate_id,
                "session_prefixes": prefixes,
                "plan_path": os.fspath(
                    _archive_plan_path(destination, coordinate_id, plan_sha)
                ),
                "plan_sha256": plan_sha,
                "content_tree_sha256": plan["content_tree_sha256"],
                "finalization_sha256": finalization["finalization_sha256"],
                "archive_path": os.fspath(archive_path),
                "archive_sha256": finalization["archive_sha256"],
                "archive_size_bytes": finalization["archive_size_bytes"],
                **remote,
            }
        )
        verified_ids.add(coordinate_id)

    excluded_ids = sorted(set(coordinates) - verified_ids)
    excluded_files = []
    for prefix, owner in sorted(owners.items()):
        if owner not in excluded_ids:
            continue
        for relative in _files_under(root, prefix):
            path = _safe_path(root, relative)
            state = path.stat(follow_symlinks=False)
            excluded_files.append(
                {
                    "coordinate_id": owner,
                    "path": relative,
                    "size_bytes": state.st_size,
                }
            )
    targets.sort(key=lambda row: row["path"])
    verified_cells.sort(key=lambda row: row["coordinate_id"])
    excluded_files.sort(key=lambda row: row["path"])
    run_file_sha, _, _, _ = _hash_file(root / "run.json")
    evaluation_file_sha, _, _, _ = _hash_file(root / "evaluation.json")
    cells_file_sha, _, _, _ = _hash_file(root / "cells.json")
    remote_set_sha = _sha256([row["receipt_sha256"] for row in verified_cells])
    return {
        "contract": RECLAIM_PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "source_run_root": os.fspath(root),
        "archive_root": os.fspath(destination),
        "remote_receipts_dir": os.fspath(remote_root),
        "expected_drive_parent_id": parent_id,
        "source_run_sha256": run["run_sha256"],
        "source_run_artifact_sha256": run_file_sha,
        "study_sha256": run["study_sha256"],
        "evaluation_sha256": run["evaluation_sha256"],
        "evaluation_artifact_sha256": evaluation_file_sha,
        "cells_artifact_sha256": cells_file_sha,
        "remote_receipt_set_sha256": remote_set_sha,
        "verified_cell_count": len(verified_cells),
        "verified_cells": verified_cells,
        "unverified_cell_count": len(excluded_ids),
        "unverified_coordinate_ids": excluded_ids,
        "unverified_raw_file_count": len(excluded_files),
        "unverified_raw_files": excluded_files,
        "target_count": len(targets),
        "target_bytes": sum(row["size_bytes"] for row in targets),
        "targets": targets,
        "shared_terminal_files": [
            shared_by_path[path] for path in sorted(shared_by_path)
        ],
        "reclaim_mode": "UNLINK_REMOTE_VERIFIED_CELL_SESSION_RAW_ONLY",
        "full_verified_archive_inventory_rehashed": True,
        "remote_unverified_cells_excluded": True,
        "source_deletion_allowed": True,
        "source_deleted": False,
        **_AUTHORITY,
    }


def _verify_row(root: Path, row: Mapping[str, Any], *, missing_allowed: bool) -> bool:
    path = _safe_path(root, str(row.get("path", "")), must_exist=False)
    try:
        state = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        if missing_allowed:
            return False
        raise DojoLegacyCellRawReclaimError(
            f"required file is missing: {row.get('path')}"
        ) from None
    if not stat.S_ISREG(state.st_mode):
        raise DojoLegacyCellRawReclaimError("reclaim inventory file is not regular")
    sha256, _, size, _ = _hash_file(path)
    if sha256 != row.get("sha256") or size != row.get("size_bytes"):
        raise DojoLegacyCellRawReclaimError(
            f"reclaim inventory bytes drifted: {row.get('path')}"
        )
    return True


def _validate_plan_resume(
    *,
    plan: Mapping[str, Any],
    source_run: Path,
    archive_root: Path,
    remote_receipts_dir: Path,
    expected_drive_parent_id: str,
    zstd_bin: str,
) -> tuple[Path, dict[str, Any]]:
    _exact(plan, _RECLAIM_PLAN_KEYS, "legacy reclaim plan")
    body = {key: value for key, value in plan.items() if key != "reclaim_plan_sha256"}
    root = source_run.resolve(strict=True)
    destination = archive_root.resolve(strict=True)
    remote_root = remote_receipts_dir.resolve(strict=True)
    parent_id = _drive_id(expected_drive_parent_id, "expected Drive parent id")
    if (
        plan.get("contract") != RECLAIM_PLAN_CONTRACT
        or plan.get("schema_version") != SCHEMA_VERSION
        or _sha(plan.get("reclaim_plan_sha256"), "reclaim plan SHA-256")
        != _sha256(body)
        or plan.get("source_run_root") != os.fspath(root)
        or plan.get("archive_root") != os.fspath(destination)
        or plan.get("remote_receipts_dir") != os.fspath(remote_root)
        or plan.get("expected_drive_parent_id") != parent_id
        or plan.get("reclaim_mode") != "UNLINK_REMOTE_VERIFIED_CELL_SESSION_RAW_ONLY"
        or plan.get("full_verified_archive_inventory_rehashed") is not True
        or plan.get("source_deletion_allowed") is not True
        or plan.get("source_deleted") is not False
        or plan.get("remote_unverified_cells_excluded") is not True
        or any(plan.get(key) != value for key, value in _AUTHORITY.items())
    ):
        raise DojoLegacyCellRawReclaimError("existing reclaim plan is invalid")
    run = _load_json(root / "run.json", field="run receipt")
    _exact(run, drive_archive._RUN_KEYS, "run receipt")
    run_body = {key: value for key, value in run.items() if key != "run_sha256"}
    if (
        run.get("contract") != drive_archive.RUN_CONTRACT
        or run.get("schema_version") != 1
        or run.get("status") != "COMPLETE"
        or run.get("run_sha256") != plan.get("source_run_sha256")
        or run.get("run_sha256") != drive_archive.canonical_sha256(run_body)
        or run.get("study_sha256") != plan.get("study_sha256")
        or run.get("evaluation_sha256") != plan.get("evaluation_sha256")
        or run.get("proof_eligible") is not False
        or run.get("promotion_eligible") is not False
        or run.get("live_permission") is not False
        or run.get("order_authority") != "NONE"
        or run.get("broker_mutation_allowed") is not False
    ):
        raise DojoLegacyCellRawReclaimError("run lineage drifted after planning")
    run_sha, _, _, _ = _hash_file(root / "run.json")
    evaluation_sha, _, _, _ = _hash_file(root / "evaluation.json")
    cells_sha, _, _, _ = _hash_file(root / "cells.json")
    if (
        run_sha != plan.get("source_run_artifact_sha256")
        or evaluation_sha != plan.get("evaluation_artifact_sha256")
        or cells_sha != plan.get("cells_artifact_sha256")
    ):
        raise DojoLegacyCellRawReclaimError("shared terminal artifacts drifted")
    coordinates, owners = _session_owners(root, run)
    verified_cells = plan.get("verified_cells")
    if (
        not isinstance(verified_cells, list)
        or not verified_cells
        or plan.get("verified_cell_count") != len(verified_cells)
        or verified_cells
        != sorted(verified_cells, key=lambda row: row.get("coordinate_id", ""))
    ):
        raise DojoLegacyCellRawReclaimError("verified cell plan is invalid")
    receipt_paths = {os.fspath(path) for path in _discover_remote_receipts(remote_root)}
    observed_receipts: set[str] = set()
    verified_ids: set[str] = set()
    expected_targets: list[dict[str, Any]] = []
    expected_shared: dict[str, dict[str, Any]] = {}
    expected_target_paths: set[str] = set()
    for raw_cell in verified_cells:
        cell = _exact(raw_cell, _VERIFIED_CELL_KEYS, "verified cell row")
        coordinate_id = str(cell.get("coordinate_id", ""))
        if coordinate_id not in coordinates or coordinate_id in verified_ids:
            raise DojoLegacyCellRawReclaimError(
                "planned coordinate is duplicate or left the run"
            )
        verified_ids.add(coordinate_id)
        plan_sha = _sha(cell.get("plan_sha256"), "planned archive SHA")
        archive_plan, finalization, archive_path = _static_archive_bundle(
            archive_root=destination,
            coordinate_id=coordinate_id,
            plan_sha256=plan_sha,
            zstd_bin=zstd_bin,
            fresh_source_verification=False,
        )
        prefixes = _coordinate_prefixes(
            root, coordinate_id, coordinates[coordinate_id], owners
        )
        cell_targets, shared = _archived_rows_for_prefixes(
            plan=archive_plan,
            coordinate_id=coordinate_id,
            prefixes=prefixes,
        )
        cell_target_paths = {row["path"] for row in cell_targets}
        if expected_target_paths & cell_target_paths:
            raise DojoLegacyCellRawReclaimError("planned verified cell targets overlap")
        expected_target_paths.update(cell_target_paths)
        expected_targets.extend(cell_targets)
        for row in shared:
            prior = expected_shared.setdefault(row["path"], row)
            if prior != row:
                raise DojoLegacyCellRawReclaimError(
                    "planned archives disagree on shared terminal bytes"
                )
        try:
            remote_path = Path(str(cell.get("receipt_path", ""))).resolve(strict=True)
        except OSError as exc:
            raise DojoLegacyCellRawReclaimError(
                "planned remote receipt is unavailable"
            ) from exc
        if remote_root not in remote_path.parents:
            raise DojoLegacyCellRawReclaimError(
                "planned remote receipt escaped its directory"
            )
        observed_receipts.add(os.fspath(remote_path))
        remote = _validate_remote_receipt(
            receipt_path=remote_path,
            root=root,
            run=run,
            archive_root=destination,
            expected_drive_parent_id=parent_id,
            plan=archive_plan,
            finalization=finalization,
            archive_path=archive_path,
        )
        expected_cell = {
            "coordinate_id": coordinate_id,
            "session_prefixes": prefixes,
            "plan_path": os.fspath(
                _archive_plan_path(destination, coordinate_id, plan_sha)
            ),
            "plan_sha256": plan_sha,
            "content_tree_sha256": archive_plan["content_tree_sha256"],
            "finalization_sha256": finalization["finalization_sha256"],
            "archive_path": os.fspath(archive_path),
            "archive_sha256": finalization["archive_sha256"],
            "archive_size_bytes": finalization["archive_size_bytes"],
            **remote,
        }
        if dict(cell) != expected_cell:
            raise DojoLegacyCellRawReclaimError("planned archive lineage drifted")
    if observed_receipts != receipt_paths:
        raise DojoLegacyCellRawReclaimError(
            "remote receipt set changed after reclaim planning"
        )
    if plan.get("remote_receipt_set_sha256") != _sha256(
        [row["receipt_sha256"] for row in verified_cells]
    ):
        raise DojoLegacyCellRawReclaimError("remote receipt set seal is invalid")
    targets = plan.get("targets")
    expected_targets.sort(key=lambda row: row["path"])
    if (
        not isinstance(targets, list)
        or targets != expected_targets
        or targets != sorted(targets, key=lambda row: row.get("path", ""))
        or plan.get("target_count") != len(targets)
        or plan.get("target_bytes")
        != sum(row["size_bytes"] for row in expected_targets)
    ):
        raise DojoLegacyCellRawReclaimError("reclaim target plan is invalid")
    for raw in targets:
        row = _exact(raw, _TARGET_KEYS, "reclaim target row")
        _safe_relative(str(row["path"]))
        if (
            row["coordinate_id"] not in verified_ids
            or row["raw_kind"] != "SESSION_RAW"
            or Path(row["path"]).name not in RAW_FILENAMES
        ):
            raise DojoLegacyCellRawReclaimError("reclaim target row is invalid")
        _nonnegative_int(row["size_bytes"], "reclaim target size")
        _sha(row["sha256"], "reclaim target SHA-256")
    allowed_paths = {row["path"] for row in targets}
    for cell in verified_cells:
        for prefix in cell["session_prefixes"]:
            directory = _safe_path(root, prefix)
            observed: set[str] = set()
            for item in directory.iterdir():
                if item.name == "inbox":
                    _verify_empty_runtime_inbox(item)
                    continue
                state = item.stat(follow_symlinks=False)
                if item.is_symlink() or not stat.S_ISREG(state.st_mode):
                    raise DojoLegacyCellRawReclaimError(
                        "unsafe entry appeared in a verified cell session"
                    )
                observed.add(item.relative_to(root).as_posix())
            expected = {
                path
                for path in allowed_paths
                if PurePosixPath(prefix) in PurePosixPath(path).parents
            }
            if not observed.issubset(expected):
                raise DojoLegacyCellRawReclaimError(
                    "unmanifested file appeared in a verified cell session"
                )
    for row in targets:
        _verify_row(root, row, missing_allowed=True)
    shared = plan.get("shared_terminal_files")
    expected_shared_rows = [expected_shared[path] for path in sorted(expected_shared)]
    if shared != expected_shared_rows:
        raise DojoLegacyCellRawReclaimError("shared terminal inventory is invalid")
    for raw in shared:
        row = _exact(raw, drive_archive._PLAN_FILE_KEYS, "shared terminal row")
        if row["path"] not in SHARED_FILENAMES:
            raise DojoLegacyCellRawReclaimError(
                "non-shared file entered the shared terminal inventory"
            )
        _verify_row(root, row, missing_allowed=False)
    excluded_ids = sorted(set(coordinates) - verified_ids)
    if plan.get("unverified_coordinate_ids") != excluded_ids or plan.get(
        "unverified_cell_count"
    ) != len(excluded_ids):
        raise DojoLegacyCellRawReclaimError("unverified cell set is invalid")
    excluded = plan.get("unverified_raw_files")
    expected_excluded: list[dict[str, Any]] = []
    for prefix, owner in sorted(owners.items()):
        if owner not in excluded_ids:
            continue
        for relative in _files_under(root, prefix):
            state = _safe_path(root, relative).stat(follow_symlinks=False)
            expected_excluded.append(
                {
                    "coordinate_id": owner,
                    "path": relative,
                    "size_bytes": state.st_size,
                }
            )
    expected_excluded.sort(key=lambda row: row["path"])
    if (
        not isinstance(excluded, list)
        or excluded != expected_excluded
        or plan.get("unverified_raw_file_count") != len(excluded)
    ):
        raise DojoLegacyCellRawReclaimError("unverified cell exclusion is invalid")
    for raw in excluded:
        row = _exact(raw, _EXCLUDED_FILE_KEYS, "unverified raw row")
        path = _safe_path(root, str(row["path"]))
        state = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(state.st_mode) or state.st_size != row["size_bytes"]:
            raise DojoLegacyCellRawReclaimError(
                "an unverified cell raw file changed or disappeared"
            )
    return root, dict(plan)


def verify_legacy_cell_raw_reclaim(
    *,
    source_run: Path,
    archive_root: Path,
    remote_receipts_dir: Path,
    expected_drive_parent_id: str,
    zstd_bin: str = "zstd",
) -> dict[str, Any]:
    """Read-only run-level eligibility check; no plans or files are changed."""

    body = _fresh_plan_body(
        source_run=Path(source_run),
        archive_root=Path(archive_root),
        remote_receipts_dir=Path(remote_receipts_dir),
        expected_drive_parent_id=expected_drive_parent_id,
        zstd_bin=zstd_bin,
    )
    return {
        "status": "LEGACY_CELL_RAW_RECLAIM_VERIFIED_NOT_EXECUTED",
        "plan": {**body, "reclaim_plan_sha256": _sha256(body)},
        **_AUTHORITY,
    }


def _write_once(path: Path, value: Mapping[str, Any], *, field: str) -> None:
    payload = _canonical_bytes(value) + b"\n"
    if path.exists() or path.is_symlink():
        try:
            current = path.read_bytes()
        except OSError as exc:
            raise DojoLegacyCellRawReclaimError(f"{field} is unavailable") from exc
        if current != payload:
            raise DojoLegacyCellRawReclaimError(f"{field} already drifted")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise DojoLegacyCellRawReclaimError(f"{field} concurrently drifted")
        drive_archive._fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def reclaim_legacy_cell_raw(
    *,
    source_run: Path,
    archive_root: Path,
    remote_receipts_dir: Path,
    expected_drive_parent_id: str,
    zstd_bin: str = "zstd",
) -> dict[str, Any]:
    """Reject V1 reclaim because its remote receipts are self-attested JSON.

    V1 verification remains available as diagnostic migration input.  It can
    never authorize another unlink: the receipt contract has no authenticated
    issuer and cannot prove that its Drive metadata came from Google Drive.
    """

    del (
        source_run,
        archive_root,
        remote_receipts_dir,
        expected_drive_parent_id,
        zstd_bin,
    )
    raise DojoLegacyCellRawReclaimError("V1_REMOTE_RECEIPTS_UNSIGNED_RECLAIM_DISABLED")


__all__ = [
    "DojoLegacyCellRawReclaimError",
    "LEGACY_REMOTE_RECEIPT_CONTRACT",
    "RECLAIM_PLAN_CONTRACT",
    "RECLAIM_RECEIPT_CONTRACT",
    "REMOTE_RECEIPT_CONTRACT",
    "reclaim_legacy_cell_raw",
    "verify_legacy_cell_raw_reclaim",
]

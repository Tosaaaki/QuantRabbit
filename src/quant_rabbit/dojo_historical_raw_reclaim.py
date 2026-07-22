"""Fail-closed raw-space reclamation for archived historical DOJO jobs.

This module deliberately cannot create Drive verification evidence.  It only
accepts one content-addressed receipt emitted after an authenticated external
Google Drive raw readback, revalidates the local archive lineage and every
current source byte, and then removes the two explicitly reclaimable raw
classes: the authenticated source slice and ``*.economic.jsonl`` transcripts.

The local archive receipt remains immutable and continues to say
``remote_verified=false``.  Remote verification and reclamation are separate,
append-only receipts so neither step can rewrite an earlier claim.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final, Iterator

from quant_rabbit.dojo_historical_job_archive import (
    SUCCESS_BUNDLE_KIND,
    DojoHistoricalJobArchiveError,
    _bundle_kind,
    _canonical_bytes,
    _canonical_sha256,
    _collect_files,
    _completion,
    _fsync_directory,
    _hash_file,
    _identity,
    _read_json,
    _safe_relative,
    _stable_regular_bytes,
    _validate_receipt,
    _verify_archive,
    _verify_remote_readback_objects,
    verify_existing_historical_job_archive,
)


REMOTE_READBACK_RECEIPT_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_JOB_DRIVE_REMOTE_READBACK_V1"
)
RECLAIM_PLAN_CONTRACT: Final = "QR_DOJO_HISTORICAL_JOB_RAW_RECLAIM_PLAN_V1"
RECLAIM_RECEIPT_CONTRACT: Final = "QR_DOJO_HISTORICAL_JOB_RAW_RECLAIM_V1"
RECLAIM_VERIFICATION_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_JOB_RAW_RECLAIM_VERIFICATION_V1"
)
SCHEMA_VERSION: Final = 1
MAX_JSON_BYTES: Final = 256 * 1024 * 1024
HASH_CHUNK_BYTES: Final = 1024 * 1024
_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_MD5_RE: Final = re.compile(r"[0-9a-f]{32}\Z")
_DRIVE_ID_RE: Final = re.compile(r"[A-Za-z0-9_-]{8,256}\Z")
_DECIMAL_RE: Final = re.compile(r"(?:0|[1-9][0-9]{0,31})\Z")
_LOCAL_ARCHIVE_RECEIPT_KEYS: Final = {
    "contract",
    "schema_version",
    "job_sha256",
    "completion_sha256",
    "bundle_kind",
    "manifest_sha256",
    "archive_path",
    "archive_sha256",
    "archive_size_bytes",
    "file_count",
    "total_source_bytes",
    "local_payload_verified",
    "remote_readback_objects",
    "source_deletion_allowed",
    "source_deleted",
    "remote_verification",
    "historical_train_is_proof",
    "promotion_eligible",
    "live_permission",
    "order_authority",
    "broker_mutation_allowed",
    "receipt_sha256",
}
_ARCHIVE_MANIFEST_KEYS: Final = {
    "contract",
    "schema_version",
    "job_sha256",
    "completion_sha256",
    "bundle_kind",
    "file_count",
    "total_source_bytes",
    "files",
    "historical_train_is_proof",
    "promotion_eligible",
    "live_permission",
    "order_authority",
    "broker_mutation_allowed",
    "manifest_sha256",
}
_REMOTE_RECEIPT_KEYS: Final = {
    "contract",
    "schema_version",
    "status",
    "provider",
    "verification_method",
    "job_sha256",
    "completion_sha256",
    "bundle_kind",
    "manifest_sha256",
    "local_archive_receipt_sha256",
    "archive_sha256",
    "archive_size_bytes",
    "object_set_sha256",
    "object_count",
    "expected_drive_parent_id",
    "drive_parent",
    "readback_at_utc",
    "objects",
    "download_bytes_match_local_objects",
    "drive_metadata_revision_bound",
    "drive_parents_bound",
    "drive_trashed_false",
    "external_readback_attested",
    "remote_verified",
    "raw_reclaim_eligible",
    "historical_train_is_proof",
    "promotion_eligible",
    "live_permission",
    "order_authority",
    "broker_mutation_allowed",
    "remote_receipt_sha256",
}
_DRIVE_PARENT_KEYS: Final = {
    "drive_folder_id",
    "drive_folder_parent_id",
    "drive_folder_name",
    "mime_type",
    "trashed",
}
_REMOTE_OBJECT_KEYS: Final = {
    "index",
    "offset_bytes",
    "relative_path",
    "size_bytes",
    "sha256",
    "metadata_before",
    "metadata_after",
    "downloaded",
}
_DRIVE_METADATA_KEYS: Final = {
    "drive_file_id",
    "drive_parent_id",
    "drive_file_name",
    "mime_type",
    "content_size_bytes",
    "md5_checksum",
    "modified_time",
    "version",
    "head_revision_id",
    "trashed",
}
_DOWNLOAD_KEYS: Final = {
    "content_size_bytes",
    "sha256",
    "md5_checksum",
}
_RECLAIM_PLAN_KEYS: Final = {
    "contract",
    "schema_version",
    "job_sha256",
    "completion_sha256",
    "bundle_kind",
    "manifest_sha256",
    "local_archive_receipt_sha256",
    "remote_receipt_sha256",
    "object_set_sha256",
    "archive_sha256",
    "archive_size_bytes",
    "reclaim_mode",
    "target_count",
    "target_bytes",
    "targets",
    "retained_file_count",
    "retained_bytes",
    "full_source_inventory_verified",
    "remote_raw_readback_verified",
    "historical_train_is_proof",
    "promotion_eligible",
    "live_permission",
    "order_authority",
    "broker_mutation_allowed",
    "plan_sha256",
}
_RECLAIM_RECEIPT_KEYS: Final = {
    "contract",
    "schema_version",
    "status",
    "job_sha256",
    "completion_sha256",
    "bundle_kind",
    "manifest_sha256",
    "local_archive_receipt_sha256",
    "remote_receipt_sha256",
    "reclaim_plan_sha256",
    "completed_at_utc",
    "deleted_file_count",
    "deleted_files",
    "reclaimed_logical_bytes",
    "reclaimed_allocated_bytes_observed",
    "free_disk_bytes_before",
    "free_disk_bytes_after",
    "retained_file_count",
    "retained_bytes",
    "restore_requires_verified_archive",
    "historical_train_is_proof",
    "promotion_eligible",
    "live_permission",
    "order_authority",
    "broker_mutation_allowed",
    "reclaim_receipt_sha256",
}
_AUTHORITY: Final = {
    "historical_train_is_proof": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}


class DojoHistoricalRawReclaimError(ValueError):
    """Raw evidence cannot be reclaimed without the complete remote lineage."""


@dataclass(frozen=True)
class _PreparedReclaim:
    run_root: Path
    archive_root: Path
    archive_receipt_path: Path
    remote_receipt_path: Path
    local_receipt: dict[str, Any]
    remote_receipt: dict[str, Any]
    manifest: dict[str, Any]
    targets: tuple[dict[str, Any], ...]
    retained: tuple[dict[str, Any], ...]


def _exact(value: Any, keys: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise DojoHistoricalRawReclaimError(f"{field} schema is invalid")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoHistoricalRawReclaimError(f"{field} is not a SHA-256")
    return value


def _md5(value: Any, field: str) -> str:
    if not isinstance(value, str) or _MD5_RE.fullmatch(value) is None:
        raise DojoHistoricalRawReclaimError(f"{field} is not an MD5")
    return value


def _drive_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _DRIVE_ID_RE.fullmatch(value) is None:
        raise DojoHistoricalRawReclaimError(f"{field} is not a Drive id")
    return value


def _bounded_text(value: Any, field: str, *, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise DojoHistoricalRawReclaimError(f"{field} is invalid")
    return value


def _filename(value: Any, field: str) -> str:
    text = _bounded_text(value, field, maximum=255)
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise DojoHistoricalRawReclaimError(f"{field} is not a filename")
    return text


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DojoHistoricalRawReclaimError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DojoHistoricalRawReclaimError(f"{field} must be a nonnegative integer")
    return value


def _utc(value: Any, field: str) -> datetime:
    text = _bounded_text(value, field, maximum=64)
    try:
        instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoHistoricalRawReclaimError(f"{field} is not an instant") from exc
    if instant.tzinfo is None or instant.utcoffset() != timezone.utc.utcoffset(instant):
        raise DojoHistoricalRawReclaimError(f"{field} must be UTC-aware")
    return instant


def _read_canonical_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        raw = _stable_regular_bytes(path, field=field, maximum=MAX_JSON_BYTES)
        value = _read_json(path, field=field)
    except DojoHistoricalJobArchiveError as exc:
        raise DojoHistoricalRawReclaimError(str(exc)) from exc
    if raw != _canonical_bytes(value) + b"\n":
        raise DojoHistoricalRawReclaimError(f"{field} is not canonical JSON")
    return value


def _hashes_file(path: Path) -> tuple[str, str, int, int]:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise DojoHistoricalRawReclaimError(
                f"reclaim input is not a regular file: {path}"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        sha256 = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        size = 0
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            while chunk := handle.read(HASH_CHUNK_BYTES):
                sha256.update(chunk)
                md5.update(chunk)
                size += len(chunk)
            opened = os.fstat(handle.fileno())
        after = path.stat(follow_symlinks=False)
    except DojoHistoricalRawReclaimError:
        raise
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            f"reclaim input is unavailable: {path}"
        ) from exc
    if (
        _identity(before) != _identity(opened)
        or _identity(opened) != _identity(after)
        or size != before.st_size
    ):
        raise DojoHistoricalRawReclaimError(
            f"reclaim input changed while hashed: {path}"
        )
    return sha256.hexdigest(), md5.hexdigest(), size, before.st_blocks * 512


def _safe_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    try:
        safe = _safe_relative(relative)
    except DojoHistoricalJobArchiveError as exc:
        raise DojoHistoricalRawReclaimError(str(exc)) from exc
    path = root / PurePosixPath(safe)
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            f"reclaim inventory path is unavailable: {relative}"
        ) from exc
    if resolved == root or root not in resolved.parents:
        raise DojoHistoricalRawReclaimError(
            f"reclaim inventory escapes its run root: {relative}"
        )
    return path


def _validate_local_archive(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    require_full_inventory: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    receipt_path = archive_receipt_path.resolve(strict=True)
    if receipt_path.parent.name != "receipts":
        raise DojoHistoricalRawReclaimError(
            "local archive receipt is outside its receipts directory"
        )
    archive_root = receipt_path.parent.parent.resolve(strict=True)
    local = _read_canonical_json(receipt_path, field="local archive receipt")
    _exact(local, _LOCAL_ARCHIVE_RECEIPT_KEYS, "local archive receipt")
    if archive_root == run_root or run_root in archive_root.parents:
        raise DojoHistoricalRawReclaimError(
            "local archive root is not independent from the run root"
        )
    job_sha256 = _sha(local.get("job_sha256"), "local receipt job SHA-256")
    try:
        completion = _completion(run_root, job_sha256)
        bundle_kind = _bundle_kind(completion)
        if bundle_kind != SUCCESS_BUNDLE_KIND:
            raise DojoHistoricalRawReclaimError(
                "raw reclaim supports only SUCCESS_ECONOMIC archives"
            )
        archive_path, manifest_sha256 = _validate_receipt(
            local,
            job_sha256=job_sha256,
            completion_sha256=completion["completion_sha256"],
            bundle_kind=bundle_kind,
            archives=archive_root / "archives",
        )
    except DojoHistoricalJobArchiveError as exc:
        raise DojoHistoricalRawReclaimError(str(exc)) from exc
    expected_receipt = (
        archive_root / "receipts" / f"job-{job_sha256}-{manifest_sha256}.json"
    )
    if receipt_path != expected_receipt:
        raise DojoHistoricalRawReclaimError(
            "local archive receipt filename does not match its lineage"
        )
    if require_full_inventory:
        try:
            public_verified = verify_existing_historical_job_archive(
                run_root=run_root,
                job_sha256=job_sha256,
                archive_root=archive_root,
            )
        except DojoHistoricalJobArchiveError as exc:
            raise DojoHistoricalRawReclaimError(str(exc)) from exc
        if public_verified != local:
            raise DojoHistoricalRawReclaimError(
                "public archive verifier returned another receipt"
            )
    try:
        archive_sha256, archive_size = _hash_file(archive_path)
        if (
            archive_sha256 != local["archive_sha256"]
            or archive_size != local["archive_size_bytes"]
        ):
            raise DojoHistoricalRawReclaimError("local archive bytes drifted")
        stem = f"job-{job_sha256}-{manifest_sha256}"
        _verify_remote_readback_objects(
            local["remote_readback_objects"],
            destination=archive_root,
            archive_path=archive_path,
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size,
            stem=stem,
        )
        manifest = _verify_archive(
            archive_path,
            zstd_bin="zstd",
            expected_job_sha256=job_sha256,
            expected_completion_sha256=completion["completion_sha256"],
            expected_bundle_kind=bundle_kind,
        )
    except DojoHistoricalJobArchiveError as exc:
        raise DojoHistoricalRawReclaimError(str(exc)) from exc
    if (
        set(manifest) != _ARCHIVE_MANIFEST_KEYS
        or isinstance(manifest.get("schema_version"), bool)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("bundle_kind") != SUCCESS_BUNDLE_KIND
        or manifest["manifest_sha256"] != manifest_sha256
        or manifest["file_count"] != local["file_count"]
        or manifest["total_source_bytes"] != local["total_source_bytes"]
    ):
        raise DojoHistoricalRawReclaimError(
            "local archive receipt and embedded manifest diverge"
        )
    return archive_root, local, manifest


def _validate_drive_parent(value: Any, *, expected_parent_id: str) -> None:
    row = _exact(value, _DRIVE_PARENT_KEYS, "Drive parent")
    if (
        _drive_id(row["drive_folder_id"], "Drive folder id") != expected_parent_id
        or _drive_id(row["drive_folder_parent_id"], "Drive folder parent id")
        == expected_parent_id
        or not _filename(row["drive_folder_name"], "Drive folder name")
        or row["mime_type"] != "application/vnd.google-apps.folder"
        or row["trashed"] is not False
    ):
        raise DojoHistoricalRawReclaimError("Drive parent metadata is invalid")


def _validate_metadata(
    value: Any,
    *,
    expected_parent_id: str,
    expected_name: str,
    expected_size: int,
    expected_md5: str,
    readback_at: datetime,
) -> dict[str, Any]:
    row = _exact(value, _DRIVE_METADATA_KEYS, "Drive object metadata")
    mime = _bounded_text(row["mime_type"], "Drive object MIME", maximum=255)
    modified = _utc(row["modified_time"], "Drive object modified_time")
    version = _bounded_text(row["version"], "Drive object version", maximum=32)
    if _DECIMAL_RE.fullmatch(version) is None:
        raise DojoHistoricalRawReclaimError("Drive object version is invalid")
    if (
        _drive_id(row["drive_parent_id"], "Drive object parent id")
        != expected_parent_id
        or _filename(row["drive_file_name"], "Drive object filename") != expected_name
        or mime == "application/vnd.google-apps.shortcut"
        or _positive_integer(row["content_size_bytes"], "Drive metadata content size")
        != expected_size
        or _md5(row["md5_checksum"], "Drive metadata MD5") != expected_md5
        or modified > readback_at
        or not _drive_id(row["drive_file_id"], "Drive object id")
        or not _drive_id(row["head_revision_id"], "Drive head revision id")
        or row["trashed"] is not False
    ):
        raise DojoHistoricalRawReclaimError("Drive object metadata is invalid")
    return dict(row)


def _validate_remote_receipt(
    *,
    remote_receipt_path: Path,
    archive_root: Path,
    local: Mapping[str, Any],
    expected_drive_parent_id: str,
) -> dict[str, Any]:
    parent_id = _drive_id(expected_drive_parent_id, "expected Drive parent id")
    expected_remote_root = archive_root / "remote-receipts"
    try:
        receipt_path = remote_receipt_path.resolve(strict=True)
        remote_root = expected_remote_root.resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            "remote readback receipt is unavailable"
        ) from exc
    if receipt_path.parent != remote_root:
        raise DojoHistoricalRawReclaimError(
            "remote readback receipt is outside its append-only directory"
        )
    remote = _read_canonical_json(receipt_path, field="remote readback receipt")
    row = _exact(remote, _REMOTE_RECEIPT_KEYS, "remote readback receipt")
    body = {key: value for key, value in row.items() if key != "remote_receipt_sha256"}
    receipt_sha256 = _sha(
        row["remote_receipt_sha256"], "remote readback receipt SHA-256"
    )
    expected_name = (
        f"remote-job-{local['job_sha256']}-{local['manifest_sha256']}-"
        f"{receipt_sha256}.json"
    )
    if receipt_path.name != expected_name:
        raise DojoHistoricalRawReclaimError(
            "remote readback receipt filename is not content-addressed"
        )
    object_set = local["remote_readback_objects"]
    objects = row["objects"]
    readback_at = _utc(row["readback_at_utc"], "Drive readback_at_utc")
    if (
        row["contract"] != REMOTE_READBACK_RECEIPT_CONTRACT
        or isinstance(row["schema_version"], bool)
        or row["schema_version"] != SCHEMA_VERSION
        or row["status"] != "REMOTE_VERIFIED"
        or row["provider"] != "GOOGLE_DRIVE"
        or row["verification_method"] != "AUTHENTICATED_EXTERNAL_RAW_READBACK"
        or row["job_sha256"] != local["job_sha256"]
        or row["completion_sha256"] != local["completion_sha256"]
        or row["bundle_kind"] != SUCCESS_BUNDLE_KIND
        or row["bundle_kind"] != local["bundle_kind"]
        or row["manifest_sha256"] != local["manifest_sha256"]
        or row["local_archive_receipt_sha256"] != local["receipt_sha256"]
        or row["archive_sha256"] != local["archive_sha256"]
        or isinstance(row["archive_size_bytes"], bool)
        or row["archive_size_bytes"] != local["archive_size_bytes"]
        or row["object_set_sha256"] != object_set["object_set_sha256"]
        or isinstance(row["object_count"], bool)
        or row["object_count"] != object_set["object_count"]
        or row["expected_drive_parent_id"] != parent_id
        or not isinstance(objects, list)
        or len(objects) != object_set["object_count"]
        or row["download_bytes_match_local_objects"] is not True
        or row["drive_metadata_revision_bound"] is not True
        or row["drive_parents_bound"] is not True
        or row["drive_trashed_false"] is not True
        or row["external_readback_attested"] is not True
        or row["remote_verified"] is not True
        or row["raw_reclaim_eligible"] is not True
        or row["historical_train_is_proof"] is not False
        or row["promotion_eligible"] is not False
        or row["live_permission"] is not False
        or row["order_authority"] != "NONE"
        or row["broker_mutation_allowed"] is not False
        or receipt_sha256 != _canonical_sha256(body)
    ):
        raise DojoHistoricalRawReclaimError(
            "remote readback receipt lineage or authority is invalid"
        )
    _validate_drive_parent(row["drive_parent"], expected_parent_id=parent_id)
    seen_file_ids: set[str] = set()
    seen_revisions: set[tuple[str, str]] = set()
    total = 0
    for index, (observed, expected) in enumerate(
        zip(objects, object_set["objects"], strict=True)
    ):
        item = _exact(observed, _REMOTE_OBJECT_KEYS, "remote readback object")
        relative = _safe_relative(expected["relative_path"])
        local_path = archive_root / relative
        local_sha, local_md5, local_size, _ = _hashes_file(local_path)
        if (
            isinstance(item["index"], bool)
            or item["index"] != index
            or isinstance(item["offset_bytes"], bool)
            or item["offset_bytes"] != total
            or item["relative_path"] != relative
            or isinstance(item["size_bytes"], bool)
            or item["size_bytes"] != expected["size_bytes"]
            or item["sha256"] != expected["sha256"]
            or local_sha != expected["sha256"]
            or local_size != expected["size_bytes"]
        ):
            raise DojoHistoricalRawReclaimError(
                "remote readback object does not bind the local object set"
            )
        metadata_before = _validate_metadata(
            item["metadata_before"],
            expected_parent_id=parent_id,
            expected_name=Path(relative).name,
            expected_size=local_size,
            expected_md5=local_md5,
            readback_at=readback_at,
        )
        metadata_after = _validate_metadata(
            item["metadata_after"],
            expected_parent_id=parent_id,
            expected_name=Path(relative).name,
            expected_size=local_size,
            expected_md5=local_md5,
            readback_at=readback_at,
        )
        downloaded = _exact(item["downloaded"], _DOWNLOAD_KEYS, "downloaded object")
        if (
            metadata_before != metadata_after
            or _positive_integer(
                downloaded["content_size_bytes"], "downloaded content size"
            )
            != local_size
            or _sha(downloaded["sha256"], "downloaded SHA-256") != local_sha
            or _md5(downloaded["md5_checksum"], "downloaded MD5") != local_md5
        ):
            raise DojoHistoricalRawReclaimError(
                "Drive raw readback differs from the local object"
            )
        file_id = metadata_after["drive_file_id"]
        revision = (file_id, metadata_after["version"])
        if file_id in seen_file_ids or revision in seen_revisions:
            raise DojoHistoricalRawReclaimError(
                "Drive raw readback reuses an object or revision"
            )
        seen_file_ids.add(file_id)
        seen_revisions.add(revision)
        total += local_size
    if total != local["archive_size_bytes"]:
        raise DojoHistoricalRawReclaimError(
            "Drive raw readback object size denominator is incomplete"
        )
    return dict(remote)


def _reclaimable_rows(
    *, run_root: Path, job_sha256: str, manifest: Mapping[str, Any]
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    source_receipt = _read_canonical_json(
        run_root / "jobs" / job_sha256 / "source-slice-receipt.json",
        field="source slice receipt",
    )
    relative = source_receipt.get("relative_path")
    if not isinstance(relative, str):
        raise DojoHistoricalRawReclaimError("source receipt path is invalid")
    source_path = f"source-slices/{_safe_relative(relative)}"
    economic_prefix = f"jobs/{job_sha256}/economic-evidence/"
    targets = []
    retained = []
    source_count = 0
    transcript_count = 0
    for raw in manifest["files"]:
        row = dict(raw)
        path = row["path"]
        if path == source_path:
            row["raw_kind"] = "AUTHENTICATED_SOURCE_SLICE"
            targets.append(row)
            source_count += 1
        elif path.startswith(economic_prefix) and path.endswith(".economic.jsonl"):
            row["raw_kind"] = "ECONOMIC_TRANSCRIPT"
            targets.append(row)
            transcript_count += 1
        else:
            if path.startswith(economic_prefix) and path.endswith(".jsonl"):
                raise DojoHistoricalRawReclaimError(
                    "economic JSONL is outside the reclaim filename allowlist"
                )
            retained.append(row)
    if source_count != 1 or transcript_count < 1:
        raise DojoHistoricalRawReclaimError(
            "reclaim inventory lacks one source slice or economic transcripts"
        )
    targets.sort(key=lambda row: row["path"])
    retained.sort(key=lambda row: row["path"])
    return tuple(targets), tuple(retained)


def _verify_rows(
    root: Path,
    rows: tuple[dict[str, Any], ...],
    *,
    missing_allowed: bool,
) -> tuple[set[str], int]:
    missing: set[str] = set()
    allocated = 0
    for row in rows:
        path = _safe_path(root, row["path"], must_exist=False)
        try:
            state = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            if missing_allowed:
                missing.add(row["path"])
                continue
            raise DojoHistoricalRawReclaimError(
                f"reclaim inventory file is missing: {row['path']}"
            ) from None
        except OSError as exc:
            raise DojoHistoricalRawReclaimError(
                f"reclaim inventory file is unavailable: {row['path']}"
            ) from exc
        if not stat.S_ISREG(state.st_mode):
            raise DojoHistoricalRawReclaimError(
                f"reclaim inventory file is not regular: {row['path']}"
            )
        sha256, _, size, blocks = _hashes_file(path)
        if sha256 != row["sha256"] or size != row["size_bytes"]:
            raise DojoHistoricalRawReclaimError(
                f"reclaim inventory bytes drifted: {row['path']}"
            )
        allocated += blocks
    return missing, allocated


def _verify_no_unmanifested_job_files(
    *, root: Path, job_sha256: str, manifest_paths: set[str]
) -> None:
    trees = [
        root / "jobs" / job_sha256,
        *[
            root / "execution-state" / section / job_sha256
            for section in ("claims", "reducers", "terminals", "cells")
        ],
    ]
    for tree in trees:
        try:
            tree_state = tree.stat(follow_symlinks=False)
        except OSError as exc:
            raise DojoHistoricalRawReclaimError(
                "reclaim job evidence tree is unavailable"
            ) from exc
        if not stat.S_ISDIR(tree_state.st_mode):
            raise DojoHistoricalRawReclaimError(
                "reclaim job evidence tree is not a directory"
            )
        for path in tree.rglob("*"):
            try:
                state = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise DojoHistoricalRawReclaimError(
                    "reclaim job evidence changed while enumerated"
                ) from exc
            if stat.S_ISDIR(state.st_mode):
                continue
            if not stat.S_ISREG(state.st_mode):
                raise DojoHistoricalRawReclaimError(
                    "reclaim job evidence contains a non-regular file"
                )
            relative = path.relative_to(root).as_posix()
            if relative not in manifest_paths:
                raise DojoHistoricalRawReclaimError(
                    f"unmanifested job evidence appeared: {relative}"
                )


def _prepare_reclaim(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    remote_receipt_path: Path,
    expected_drive_parent_id: str,
    permit_missing_targets: bool,
) -> _PreparedReclaim:
    try:
        root = run_root.resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError("run root is unavailable") from exc
    if not root.is_dir():
        raise DojoHistoricalRawReclaimError("run root is not a directory")
    archive_root, local, manifest = _validate_local_archive(
        run_root=root,
        archive_receipt_path=archive_receipt_path,
        require_full_inventory=not permit_missing_targets,
    )
    remote = _validate_remote_receipt(
        remote_receipt_path=remote_receipt_path,
        archive_root=archive_root,
        local=local,
        expected_drive_parent_id=expected_drive_parent_id,
    )
    targets, retained = _reclaimable_rows(
        run_root=root,
        job_sha256=local["job_sha256"],
        manifest=manifest,
    )
    manifest_paths = {row["path"] for row in manifest["files"]}
    _verify_no_unmanifested_job_files(
        root=root,
        job_sha256=local["job_sha256"],
        manifest_paths=manifest_paths,
    )
    if not permit_missing_targets:
        try:
            current_paths = _collect_files(
                root,
                local["job_sha256"],
                bundle_kind=SUCCESS_BUNDLE_KIND,
            )
        except DojoHistoricalJobArchiveError as exc:
            raise DojoHistoricalRawReclaimError(str(exc)) from exc
        ordered_manifest_paths = [row["path"] for row in manifest["files"]]
        if current_paths != ordered_manifest_paths:
            raise DojoHistoricalRawReclaimError(
                "current source-job inventory differs from the archive manifest"
            )
    missing, _ = _verify_rows(
        root,
        targets,
        missing_allowed=permit_missing_targets,
    )
    if missing and not permit_missing_targets:
        raise DojoHistoricalRawReclaimError("raw target disappeared before planning")
    _verify_rows(root, retained, missing_allowed=False)
    return _PreparedReclaim(
        run_root=root,
        archive_root=archive_root,
        archive_receipt_path=archive_receipt_path.resolve(strict=True),
        remote_receipt_path=remote_receipt_path.resolve(strict=True),
        local_receipt=local,
        remote_receipt=remote,
        manifest=manifest,
        targets=targets,
        retained=retained,
    )


def _plan_body(prepared: _PreparedReclaim) -> dict[str, Any]:
    return {
        "contract": RECLAIM_PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "job_sha256": prepared.local_receipt["job_sha256"],
        "completion_sha256": prepared.local_receipt["completion_sha256"],
        "bundle_kind": SUCCESS_BUNDLE_KIND,
        "manifest_sha256": prepared.local_receipt["manifest_sha256"],
        "local_archive_receipt_sha256": prepared.local_receipt["receipt_sha256"],
        "remote_receipt_sha256": prepared.remote_receipt["remote_receipt_sha256"],
        "object_set_sha256": prepared.local_receipt["remote_readback_objects"][
            "object_set_sha256"
        ],
        "archive_sha256": prepared.local_receipt["archive_sha256"],
        "archive_size_bytes": prepared.local_receipt["archive_size_bytes"],
        "reclaim_mode": "UNLINK_EXACT_ALLOWLISTED_RAW",
        "target_count": len(prepared.targets),
        "target_bytes": sum(row["size_bytes"] for row in prepared.targets),
        "targets": list(prepared.targets),
        "retained_file_count": len(prepared.retained),
        "retained_bytes": sum(row["size_bytes"] for row in prepared.retained),
        "full_source_inventory_verified": True,
        "remote_raw_readback_verified": True,
        **_AUTHORITY,
    }


def _write_once(path: Path, value: Mapping[str, Any], *, field: str) -> None:
    payload = _canonical_bytes(value) + b"\n"
    try:
        current = _stable_regular_bytes(path, field=field, maximum=MAX_JSON_BYTES)
    except DojoHistoricalJobArchiveError:
        try:
            existing = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            raise DojoHistoricalRawReclaimError(f"{field} is unavailable") from exc
        if existing is not None:
            raise DojoHistoricalRawReclaimError(f"{field} is invalid") from None
    else:
        if current != payload:
            raise DojoHistoricalRawReclaimError(f"{field} already drifted")
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
            current = _stable_regular_bytes(path, field=field, maximum=MAX_JSON_BYTES)
            if current != payload:
                raise DojoHistoricalRawReclaimError(f"{field} concurrently drifted")
        _fsync_directory(path.parent)
    except DojoHistoricalJobArchiveError as exc:
        raise DojoHistoricalRawReclaimError(str(exc)) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _exclusive_lock(path: Path, *, field: str) -> Iterator[None]:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode):
            raise DojoHistoricalRawReclaimError(f"{field} is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise DojoHistoricalRawReclaimError(f"{field} is already held") from exc
    except DojoHistoricalRawReclaimError:
        raise
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(f"{field} is unavailable") from exc
    try:
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _unlink_target(root: Path, row: Mapping[str, Any]) -> int:
    path = _safe_path(root, row["path"], must_exist=False)
    try:
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            f"raw target parent is unavailable: {row['path']}"
        ) from exc
    try:
        try:
            state = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return 0
        if not stat.S_ISREG(state.st_mode):
            raise DojoHistoricalRawReclaimError(
                f"raw target is not regular: {row['path']}"
            )
        sha256, _, size, allocated = _hashes_file(path)
        observed = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            _identity(state) != _identity(observed)
            or sha256 != row["sha256"]
            or size != row["size_bytes"]
        ):
            raise DojoHistoricalRawReclaimError(
                f"raw target changed before unlink: {row['path']}"
            )
        os.unlink(path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        return allocated
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            f"raw target unlink failed: {row['path']}"
        ) from exc
    finally:
        os.close(parent_descriptor)


def _reclaim_paths(root: Path, job_sha256: str) -> tuple[Path, str]:
    receipts = root / "reclaim-receipts"
    receipts.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        state = receipts.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            "reclaim receipt directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(state.st_mode):
        raise DojoHistoricalRawReclaimError(
            "reclaim receipt directory is not a real directory"
        )
    _fsync_directory(receipts.parent)
    return receipts, f"job-{job_sha256}"


def verify_historical_job_raw_reclaim(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    remote_receipt_path: Path,
    expected_drive_parent_id: str,
) -> dict[str, Any]:
    """Read-only eligibility check; it never writes a receipt or removes data."""

    prepared = _prepare_reclaim(
        run_root=Path(run_root),
        archive_receipt_path=Path(archive_receipt_path),
        remote_receipt_path=Path(remote_receipt_path),
        expected_drive_parent_id=expected_drive_parent_id,
        permit_missing_targets=False,
    )
    body = _plan_body(prepared)
    return {
        "status": "RAW_RECLAIM_VERIFIED_NOT_EXECUTED",
        "plan": {**body, "plan_sha256": _canonical_sha256(body)},
        **_AUTHORITY,
    }


def _validate_plan(path: Path, expected_body: Mapping[str, Any]) -> dict[str, Any]:
    plan = _read_canonical_json(path, field="raw reclaim plan")
    expected = {**expected_body, "plan_sha256": _canonical_sha256(expected_body)}
    if plan != expected:
        raise DojoHistoricalRawReclaimError(
            "existing raw reclaim plan does not match current lineage"
        )
    return plan


def _validate_plan_seal(path: Path, *, expected_job_sha256: str) -> dict[str, Any]:
    plan = _read_canonical_json(path, field="raw reclaim plan")
    _exact(plan, _RECLAIM_PLAN_KEYS, "raw reclaim plan")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    targets = plan.get("targets")
    if (
        plan.get("contract") != RECLAIM_PLAN_CONTRACT
        or isinstance(plan.get("schema_version"), bool)
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("job_sha256") != expected_job_sha256
        or plan.get("bundle_kind") != SUCCESS_BUNDLE_KIND
        or _sha(plan.get("completion_sha256"), "plan completion SHA-256")
        != plan.get("completion_sha256")
        or _sha(plan.get("manifest_sha256"), "plan manifest SHA-256")
        != plan.get("manifest_sha256")
        or _sha(
            plan.get("local_archive_receipt_sha256"),
            "plan local archive receipt SHA-256",
        )
        != plan.get("local_archive_receipt_sha256")
        or _sha(plan.get("remote_receipt_sha256"), "plan remote receipt SHA-256")
        != plan.get("remote_receipt_sha256")
        or _sha(plan.get("object_set_sha256"), "plan object-set SHA-256")
        != plan.get("object_set_sha256")
        or _sha(plan.get("archive_sha256"), "plan archive SHA-256")
        != plan.get("archive_sha256")
        or _positive_integer(plan.get("archive_size_bytes"), "plan archive size")
        != plan.get("archive_size_bytes")
        or plan.get("reclaim_mode") != "UNLINK_EXACT_ALLOWLISTED_RAW"
        or not isinstance(targets, list)
        or not targets
        or _positive_integer(plan.get("target_count"), "plan target count")
        != len(targets)
        or _positive_integer(plan.get("target_bytes"), "plan target bytes")
        != plan.get("target_bytes")
        or _positive_integer(
            plan.get("retained_file_count"), "plan retained file count"
        )
        != plan.get("retained_file_count")
        or _positive_integer(plan.get("retained_bytes"), "plan retained bytes")
        != plan.get("retained_bytes")
        or plan.get("full_source_inventory_verified") is not True
        or plan.get("remote_raw_readback_verified") is not True
        or plan.get("historical_train_is_proof") is not False
        or plan.get("promotion_eligible") is not False
        or plan.get("live_permission") is not False
        or plan.get("order_authority") != "NONE"
        or plan.get("broker_mutation_allowed") is not False
        or plan.get("plan_sha256") != _canonical_sha256(body)
    ):
        raise DojoHistoricalRawReclaimError("raw reclaim plan seal is invalid")
    return plan


def _validate_reclaim_receipt(
    path: Path,
    *,
    plan: Mapping[str, Any],
    prepared: _PreparedReclaim,
) -> dict[str, Any]:
    receipt = _read_canonical_json(path, field="raw reclaim receipt")
    _exact(receipt, _RECLAIM_RECEIPT_KEYS, "raw reclaim receipt")
    body = {
        key: value for key, value in receipt.items() if key != "reclaim_receipt_sha256"
    }
    if (
        receipt.get("contract") != RECLAIM_RECEIPT_CONTRACT
        or isinstance(receipt.get("schema_version"), bool)
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "RAW_RECLAIMED"
        or receipt.get("job_sha256") != plan["job_sha256"]
        or receipt.get("completion_sha256") != plan["completion_sha256"]
        or receipt.get("bundle_kind") != SUCCESS_BUNDLE_KIND
        or receipt.get("bundle_kind") != plan["bundle_kind"]
        or receipt.get("reclaim_plan_sha256") != plan["plan_sha256"]
        or receipt.get("local_archive_receipt_sha256")
        != plan["local_archive_receipt_sha256"]
        or receipt.get("remote_receipt_sha256") != plan["remote_receipt_sha256"]
        or receipt.get("manifest_sha256") != plan["manifest_sha256"]
        or receipt.get("deleted_files") != plan["targets"]
        or receipt.get("deleted_file_count") != plan["target_count"]
        or isinstance(receipt.get("deleted_file_count"), bool)
        or receipt.get("reclaimed_logical_bytes") != plan["target_bytes"]
        or isinstance(receipt.get("reclaimed_logical_bytes"), bool)
        or _nonnegative_integer(
            receipt.get("reclaimed_allocated_bytes_observed"),
            "reclaimed allocated bytes",
        )
        != receipt.get("reclaimed_allocated_bytes_observed")
        or _positive_integer(
            receipt.get("free_disk_bytes_before"), "free disk before reclaim"
        )
        != receipt.get("free_disk_bytes_before")
        or _positive_integer(
            receipt.get("free_disk_bytes_after"), "free disk after reclaim"
        )
        != receipt.get("free_disk_bytes_after")
        or receipt.get("retained_file_count") != plan["retained_file_count"]
        or isinstance(receipt.get("retained_file_count"), bool)
        or receipt.get("retained_bytes") != plan["retained_bytes"]
        or isinstance(receipt.get("retained_bytes"), bool)
        or receipt.get("restore_requires_verified_archive") is not True
        or receipt.get("historical_train_is_proof") is not False
        or receipt.get("promotion_eligible") is not False
        or receipt.get("live_permission") is not False
        or receipt.get("order_authority") != "NONE"
        or receipt.get("broker_mutation_allowed") is not False
        or receipt.get("reclaim_receipt_sha256") != _canonical_sha256(body)
    ):
        raise DojoHistoricalRawReclaimError("raw reclaim receipt is invalid")
    _utc(receipt["completed_at_utc"], "raw reclaim completed_at_utc")
    expected_name = (
        f"reclaim-{plan['job_sha256']}-{receipt['reclaim_receipt_sha256']}.json"
    )
    if path.name != expected_name:
        raise DojoHistoricalRawReclaimError(
            "raw reclaim receipt filename is not content-addressed"
        )
    missing, _ = _verify_rows(prepared.run_root, prepared.targets, missing_allowed=True)
    if missing != {row["path"] for row in prepared.targets}:
        raise DojoHistoricalRawReclaimError(
            "raw reclaim receipt exists while a target still exists"
        )
    _verify_rows(prepared.run_root, prepared.retained, missing_allowed=False)
    return receipt


def reclaim_historical_job_raw(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    remote_receipt_path: Path,
    expected_drive_parent_id: str,
) -> dict[str, Any]:
    """Remove only raw transcripts/source after complete local+remote proof."""

    root = Path(run_root).resolve(strict=True)
    run_lock = root / ".historical-train.lock"
    with _exclusive_lock(run_lock, field="historical train run lock"):
        preliminary_archive = _read_canonical_json(
            Path(archive_receipt_path), field="local archive receipt"
        )
        job_sha256 = _sha(
            preliminary_archive.get("job_sha256"), "local receipt job SHA-256"
        )
        receipts, prefix = _reclaim_paths(root, job_sha256)
        with _exclusive_lock(
            receipts / f".{prefix}.lock", field="raw reclaim job lock"
        ):
            existing_receipts = sorted(receipts.glob(f"reclaim-{job_sha256}-*.json"))
            existing_plans = sorted(receipts.glob(f"plan-{prefix}-*.json"))
            permit_missing = bool(existing_plans or existing_receipts)
            prepared = _prepare_reclaim(
                run_root=root,
                archive_receipt_path=Path(archive_receipt_path),
                remote_receipt_path=Path(remote_receipt_path),
                expected_drive_parent_id=expected_drive_parent_id,
                permit_missing_targets=permit_missing,
            )
            plan_body = _plan_body(prepared)
            plan = {**plan_body, "plan_sha256": _canonical_sha256(plan_body)}
            plan_path = receipts / (
                f"plan-{prefix}-{prepared.remote_receipt['remote_receipt_sha256']}.json"
            )
            if existing_plans:
                if existing_plans != [plan_path]:
                    raise DojoHistoricalRawReclaimError(
                        "multiple or foreign raw reclaim plans name this job"
                    )
                plan = _validate_plan(plan_path, plan_body)
            else:
                if permit_missing:
                    raise DojoHistoricalRawReclaimError(
                        "raw target is missing without an append-only reclaim plan"
                    )
                _write_once(plan_path, plan, field="raw reclaim plan")

            if existing_receipts:
                if len(existing_receipts) != 1:
                    raise DojoHistoricalRawReclaimError(
                        "multiple raw reclaim receipts name this job"
                    )
                return _validate_reclaim_receipt(
                    existing_receipts[0], plan=plan, prepared=prepared
                )

            free_before = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
            reclaimed_allocated = 0
            for row in prepared.targets:
                reclaimed_allocated += _unlink_target(root, row)
            missing, _ = _verify_rows(root, prepared.targets, missing_allowed=True)
            if missing != {row["path"] for row in prepared.targets}:
                raise DojoHistoricalRawReclaimError(
                    "not every planned raw target was reclaimed"
                )
            _verify_rows(root, prepared.retained, missing_allowed=False)
            free_after = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
            completed_at = datetime.now(timezone.utc).isoformat()
            receipt_body = {
                "contract": RECLAIM_RECEIPT_CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "status": "RAW_RECLAIMED",
                "job_sha256": plan["job_sha256"],
                "completion_sha256": plan["completion_sha256"],
                "bundle_kind": SUCCESS_BUNDLE_KIND,
                "manifest_sha256": plan["manifest_sha256"],
                "local_archive_receipt_sha256": plan["local_archive_receipt_sha256"],
                "remote_receipt_sha256": plan["remote_receipt_sha256"],
                "reclaim_plan_sha256": plan["plan_sha256"],
                "completed_at_utc": completed_at,
                "deleted_file_count": plan["target_count"],
                "deleted_files": plan["targets"],
                "reclaimed_logical_bytes": plan["target_bytes"],
                "reclaimed_allocated_bytes_observed": reclaimed_allocated,
                "free_disk_bytes_before": free_before,
                "free_disk_bytes_after": free_after,
                "retained_file_count": plan["retained_file_count"],
                "retained_bytes": plan["retained_bytes"],
                "restore_requires_verified_archive": True,
                **_AUTHORITY,
            }
            receipt = {
                **receipt_body,
                "reclaim_receipt_sha256": _canonical_sha256(receipt_body),
            }
            receipt_path = receipts / (
                f"reclaim-{job_sha256}-{receipt['reclaim_receipt_sha256']}.json"
            )
            _write_once(receipt_path, receipt, field="raw reclaim receipt")
            return receipt


def verify_existing_historical_job_raw_reclaim(
    *,
    run_root: Path,
    archive_root: Path,
    job_sha256: str,
    expected_drive_parent_id: str,
) -> dict[str, Any]:
    """Deep-verify one already reclaimed job without requiring deleted raw."""

    job = _sha(job_sha256, "job SHA-256")
    try:
        root = Path(run_root).resolve(strict=True)
        destination = Path(archive_root).resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError("post-reclaim root is unavailable") from exc
    if not root.is_dir() or not destination.is_dir():
        raise DojoHistoricalRawReclaimError(
            "post-reclaim roots must be real directories"
        )
    reclaim_root = root / "reclaim-receipts"
    try:
        reclaim_state = reclaim_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            "reclaim receipt directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(reclaim_state.st_mode):
        raise DojoHistoricalRawReclaimError(
            "reclaim receipt directory is not a real directory"
        )
    plan_paths = sorted(reclaim_root.glob(f"plan-job-{job}-*.json"))
    reclaim_paths = sorted(reclaim_root.glob(f"reclaim-{job}-*.json"))
    if len(plan_paths) != 1 or len(reclaim_paths) != 1:
        raise DojoHistoricalRawReclaimError(
            "post-reclaim verification requires exact-one plan and receipt"
        )
    plan = _validate_plan_seal(plan_paths[0], expected_job_sha256=job)
    expected_plan_path = reclaim_root / (
        f"plan-job-{job}-{plan['remote_receipt_sha256']}.json"
    )
    if plan_paths[0] != expected_plan_path:
        raise DojoHistoricalRawReclaimError(
            "raw reclaim plan filename does not match its lineage"
        )

    local_paths = sorted((destination / "receipts").glob(f"job-{job}-*.json"))
    if len(local_paths) != 1:
        raise DojoHistoricalRawReclaimError(
            "post-reclaim verification requires exact-one local archive receipt"
        )
    local = _read_canonical_json(local_paths[0], field="local archive receipt")
    if (
        local.get("receipt_sha256") != plan["local_archive_receipt_sha256"]
        or local.get("manifest_sha256") != plan["manifest_sha256"]
        or local.get("bundle_kind") != SUCCESS_BUNDLE_KIND
    ):
        raise DojoHistoricalRawReclaimError(
            "reclaim plan resolves another local archive receipt"
        )

    remote_root = destination / "remote-receipts"
    remote_paths = sorted(
        remote_root.glob(f"remote-job-{job}-{plan['manifest_sha256']}-*.json")
    )
    expected_remote_path = remote_root / (
        f"remote-job-{job}-{plan['manifest_sha256']}-"
        f"{plan['remote_receipt_sha256']}.json"
    )
    if remote_paths != [expected_remote_path]:
        raise DojoHistoricalRawReclaimError(
            "post-reclaim verification cannot uniquely resolve remote receipt"
        )

    prepared = _prepare_reclaim(
        run_root=root,
        archive_receipt_path=local_paths[0],
        remote_receipt_path=expected_remote_path,
        expected_drive_parent_id=expected_drive_parent_id,
        permit_missing_targets=True,
    )
    expected_plan_body = _plan_body(prepared)
    plan = _validate_plan(plan_paths[0], expected_plan_body)
    missing, _ = _verify_rows(root, prepared.targets, missing_allowed=True)
    expected_missing = {row["path"] for row in prepared.targets}
    if missing != expected_missing:
        raise DojoHistoricalRawReclaimError(
            "post-reclaim verification found a raw target still present"
        )
    _verify_rows(root, prepared.retained, missing_allowed=False)
    receipt = _validate_reclaim_receipt(
        reclaim_paths[0],
        plan=plan,
        prepared=prepared,
    )
    verification_body = {
        "contract": RECLAIM_VERIFICATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "LOCALLY_ARCHIVED_AND_RAW_RECLAIMED",
        "job_sha256": job,
        "bundle_kind": SUCCESS_BUNDLE_KIND,
        "manifest_sha256": plan["manifest_sha256"],
        "local_archive_receipt_sha256": plan["local_archive_receipt_sha256"],
        "remote_receipt_sha256": plan["remote_receipt_sha256"],
        "reclaim_plan_sha256": plan["plan_sha256"],
        "reclaim_receipt_sha256": receipt["reclaim_receipt_sha256"],
        "archive_and_parts_verified": True,
        "remote_raw_readback_verified": True,
        "retained_bytes_verified": True,
        "all_raw_targets_missing": True,
        "raw_target_count": plan["target_count"],
        "reclaimed_logical_bytes": plan["target_bytes"],
        **_AUTHORITY,
    }
    return {
        **verification_body,
        "verification_sha256": _canonical_sha256(verification_body),
    }


__all__ = [
    "REMOTE_READBACK_RECEIPT_CONTRACT",
    "RECLAIM_PLAN_CONTRACT",
    "RECLAIM_RECEIPT_CONTRACT",
    "RECLAIM_VERIFICATION_CONTRACT",
    "DojoHistoricalRawReclaimError",
    "reclaim_historical_job_raw",
    "verify_existing_historical_job_raw_reclaim",
    "verify_historical_job_raw_reclaim",
]

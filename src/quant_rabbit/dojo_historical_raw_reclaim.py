"""Fail-closed raw-space reclamation for archived historical DOJO jobs.

This module never calls Google Drive.  It can seal a content-addressed receipt
only from an authenticated external raw-readback packet whose downloaded bytes,
before/after Drive revision metadata, parent, and exact local object set all
revalidate.  Reclamation then accepts that receipt, revalidates the local
archive lineage and every current source byte, and removes only the two
explicitly reclaimable raw classes: the authenticated source slice and
``*.economic.jsonl`` transcripts.

The local archive receipt remains immutable and continues to say
``remote_verified=false``.  Remote verification and reclamation are separate,
append-only receipts so neither step can rewrite an earlier claim.

Raw-file retirement and DriveFS cache eviction are separate operations.  This
module can affect only exact allowlisted files under ``run_root``; it never
deletes archive objects and never requests eviction of DriveFS local cache.

Ed25519 verification uses the optional ``cryptography`` package.  Its absence
is a hard authorization failure: there is no unsigned fallback or alternate
algorithm.  Only a pre-observation public-key seal is accepted; private-key
creation, loading, and persistence are intentionally outside this module.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import errno
import fcntl
import hashlib
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final, Iterator

from quant_rabbit.dojo_historical_job_archive import (
    SUCCESS_BUNDLE_KIND,
    DojoHistoricalJobArchiveError,
    _bundle_kind,
    _atomic_rename_at_no_replace,
    _canonical_bytes,
    _canonical_sha256,
    _collect_files,
    _completion,
    _fsync_directory,
    _hash_file,
    _identity,
    _read_json,
    _retirement_anchor_name,
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
REMOTE_READBACK_EVIDENCE_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_JOB_DRIVE_RAW_READBACK_EVIDENCE_V1"
)
REMOTE_READBACK_CANDIDATE_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_JOB_DRIVE_READBACK_CANDIDATE_V1"
)
REMOTE_READBACK_SIGNED_ATTESTATION_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_JOB_DRIVE_SIGNED_ATTESTATION_V2"
)
REMOTE_READBACK_ATTESTATION_BODY_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_JOB_DRIVE_ATTESTATION_BODY_V2"
)
ATTESTATION_PUBLIC_KEY_SEAL_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_JOB_DRIVE_ATTESTATION_PUBLIC_KEY_SEAL_V2"
)
RECLAIM_PLAN_CONTRACT: Final = "QR_DOJO_HISTORICAL_JOB_RAW_RECLAIM_PLAN_V1"
RECLAIM_PLAN_CONTRACT_V2: Final = "QR_DOJO_HISTORICAL_JOB_RAW_RECLAIM_PLAN_V2"
RECLAIM_RECEIPT_CONTRACT: Final = "QR_DOJO_HISTORICAL_JOB_RAW_RECLAIM_V1"
RESTORE_RECEIPT_CONTRACT: Final = "QR_DOJO_HISTORICAL_JOB_RAW_RESTORE_V1"
RECLAIM_VERIFICATION_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_JOB_RAW_RECLAIM_VERIFICATION_V1"
)
RAW_STORAGE_BOUNDARY_CONTRACT: Final = "QR_DOJO_HISTORICAL_RAW_STORAGE_BOUNDARY_V1"
LEGACY_RECLAIM_MODE: Final = "UNLINK_EXACT_ALLOWLISTED_RAW"
RECLAIM_MODE: Final = "ATOMIC_RETIRE_THEN_FD_TRUNCATE_EXACT_ALLOWLISTED_RAW"
SCHEMA_VERSION: Final = 1
MAX_JSON_BYTES: Final = 256 * 1024 * 1024
HASH_CHUNK_BYTES: Final = 1024 * 1024
MAX_ATTESTATION_TTL_SECONDS: Final = 15 * 60
MAX_CLOCK_SKEW_SECONDS: Final = 60
_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_MD5_RE: Final = re.compile(r"[0-9a-f]{32}\Z")
_HEX_KEY_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_DRIVE_ID_RE: Final = re.compile(r"[A-Za-z0-9_-]{8,256}\Z")
_DRIVE_REVISION_ID_RE: Final = re.compile(r"[A-Za-z0-9_+/=-]{1,256}\Z")
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
_REMOTE_EVIDENCE_PACKET_KEYS: Final = {
    "contract",
    "schema_version",
    "provider",
    "verification_method",
    "job_sha256",
    "completion_sha256",
    "manifest_sha256",
    "local_archive_receipt_sha256",
    "object_set_sha256",
    "object_count",
    "expected_drive_parent_id",
    "drive_parent",
    "readback_at_utc",
    "objects",
    "external_readback_attested",
    "source_deletion_requested",
}
_REMOTE_EVIDENCE_OBJECT_KEYS: Final = {
    "index",
    "relative_path",
    "readback_at_utc",
    "metadata_before",
    "downloaded_local_path",
    "metadata_after",
}
_REMOTE_CANDIDATE_KEYS: Final = {
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
    "drive_metadata_revision_claims_well_formed",
    "external_readback_attested",
    "remote_verified",
    "raw_reclaim_eligible",
    "trusted_provider_attestation_present",
    "historical_train_is_proof",
    "promotion_eligible",
    "live_permission",
    "order_authority",
    "broker_mutation_allowed",
    "candidate_sha256",
}
_SIGNED_ATTESTATION_KEYS: Final = {
    "contract",
    "schema_version",
    "algorithm",
    "public_key_sha256",
    "body",
    "signature_base64",
    "remote_receipt_sha256",
}
_ATTESTATION_PUBLIC_KEY_SEAL_KEYS: Final = {
    "contract",
    "schema_version",
    "status",
    "job_sha256",
    "manifest_sha256",
    "local_archive_receipt_sha256",
    "expected_drive_parent_id",
    "algorithm",
    "public_key_hex",
    "public_key_sha256",
    "enrolled_at_utc",
    "private_key_material_accepted",
    "historical_train_is_proof",
    "promotion_eligible",
    "live_permission",
    "order_authority",
    "broker_mutation_allowed",
    "authority_seal_sha256",
}
_SIGNED_ATTESTATION_BODY_KEYS: Final = {
    "contract",
    "schema_version",
    "attestation_id",
    "provider",
    "verification_method",
    "issued_at_utc",
    "expires_at_utc",
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
    "files_get_before_after_verified",
    "revisions_list_head_present_unique",
    "independent_revision_readback_verified",
    "exact_revision_bytes_hashed",
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
}
_SIGNED_ATTESTATION_OBJECT_KEYS: Final = _REMOTE_OBJECT_KEYS | {
    "listed_revision_ids",
    "drivefs_file_id_xattr",
    "drivefs_md5_field48",
    "drivefs_version_field57",
    "drivefs_current_revision_id_field78",
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
_RECLAIM_PLAN_V1_KEYS: Final = set(_RECLAIM_PLAN_KEYS)
_RECLAIM_PLAN_KEYS = _RECLAIM_PLAN_KEYS | {
    "attestation_id",
    "attestation_public_key_sha256",
    "attestation_public_key_hex",
    "attestation_authority_seal_sha256",
    "attestation_authority_enrolled_at_utc",
    "attestation_issued_at_utc",
    "attestation_expires_at_utc",
    "zstd_executable_path",
    "zstd_executable_sha256",
    "zstd_executable_size_bytes",
    "zstd_version",
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
_RESTORE_RECEIPT_KEYS: Final = {
    "contract",
    "schema_version",
    "status",
    "job_sha256",
    "completion_sha256",
    "bundle_kind",
    "manifest_sha256",
    "local_archive_receipt_sha256",
    "archive_sha256",
    "reclaim_plan_sha256",
    "reclaim_receipt_sha256",
    "restored_at_utc",
    "restored_file_count",
    "restored_files",
    "restored_logical_bytes",
    "published_file_count",
    "preexisting_matching_file_count",
    "local_archive_deep_verified",
    "retained_bytes_verified",
    "all_raw_targets_present",
    "remote_receipt_trusted",
    "historical_train_is_proof",
    "promotion_eligible",
    "live_permission",
    "order_authority",
    "broker_mutation_allowed",
    "restore_receipt_sha256",
}
_AUTHORITY: Final = {
    "historical_train_is_proof": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}
_RAW_STORAGE_BOUNDARY_BODY: Final = {
    "contract": RAW_STORAGE_BOUNDARY_CONTRACT,
    "schema_version": SCHEMA_VERSION,
    "raw_reclaim_scope": "EXACT_ALLOWLISTED_RUN_ROOT_FILES_ONLY",
    "raw_reclaim_mechanism": RECLAIM_MODE,
    "filesystem_unlink_used": False,
    "archive_object_deletion_allowed": False,
    "drivefs_cache_eviction_included": False,
    "drivefs_cache_eviction_requires_separate_explicit_operation": True,
    "drivefs_cache_eviction_implemented": False,
    **_AUTHORITY,
}
RAW_STORAGE_BOUNDARY: Final = {
    **_RAW_STORAGE_BOUNDARY_BODY,
    "contract_sha256": _canonical_sha256(_RAW_STORAGE_BOUNDARY_BODY),
}


class DojoHistoricalRawReclaimError(ValueError):
    """Raw evidence cannot be reclaimed without the complete remote lineage."""


def historical_raw_storage_boundary() -> dict[str, Any]:
    """Return the sealed storage-effect boundary for reclaim and restore."""

    return dict(RAW_STORAGE_BOUNDARY)


@dataclass(frozen=True)
class _PreparedReclaim:
    run_root: Path
    archive_root: Path
    archive_receipt_path: Path
    remote_receipt_path: Path
    local_receipt: dict[str, Any]
    remote_receipt: dict[str, Any]
    attestation_authority_seal: dict[str, Any]
    zstd_seal: dict[str, Any]
    manifest: dict[str, Any]
    targets: tuple[dict[str, Any], ...]
    retained: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _PreparedRestore:
    run_root: Path
    archive_root: Path
    archive_path: Path
    archive_receipt_path: Path
    reclaim_plan_path: Path
    reclaim_receipt_path: Path
    local_receipt: dict[str, Any]
    manifest: dict[str, Any]
    reclaim_plan: dict[str, Any]
    reclaim_receipt: dict[str, Any]
    targets: tuple[dict[str, Any], ...]
    retained: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _OpenedLock:
    """One flock whose opened inode must remain the named path while held."""

    path: Path
    descriptor: int
    parent_descriptor: int
    identity: tuple[int, int, int]
    parent_identity: tuple[int, int, int]
    field: str

    def assert_stable(self) -> None:
        try:
            opened = os.fstat(self.descriptor)
            anchored = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            named = self.path.stat(follow_symlinks=False)
            opened_parent = os.fstat(self.parent_descriptor)
            named_parent = self.path.parent.stat(follow_symlinks=False)
        except OSError as exc:
            raise DojoHistoricalRawReclaimError(
                f"{self.field} path identity changed while held"
            ) from exc
        if (
            not stat.S_ISREG(opened.st_mode)
            or self.identity != _inode_identity(opened)
            or self.identity != _inode_identity(anchored)
            or self.identity != _inode_identity(named)
            or not stat.S_ISDIR(opened_parent.st_mode)
            or self.parent_identity != _inode_identity(opened_parent)
            or self.parent_identity != _inode_identity(named_parent)
        ):
            raise DojoHistoricalRawReclaimError(
                f"{self.field} path identity changed while held"
            )


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


def _drive_revision_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _DRIVE_REVISION_ID_RE.fullmatch(value) is None:
        raise DojoHistoricalRawReclaimError(f"{field} is not a Drive revision id")
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


def _normalize_public_key(value: str | None) -> dict[str, Any]:
    if value is None:
        raise DojoHistoricalRawReclaimError(
            "trusted attestation public key is required"
        )
    normalized = value.strip().lower()
    if _HEX_KEY_RE.fullmatch(normalized) is None:
        raise DojoHistoricalRawReclaimError(
            "trusted attestation public key must be 32-byte lowercase hex"
        )
    raw = bytes.fromhex(normalized)
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        Ed25519PublicKey.from_public_bytes(raw)
    except (ImportError, ValueError) as exc:
        raise DojoHistoricalRawReclaimError(
            "Ed25519 verifier is unavailable or the public key is invalid"
        ) from exc
    return {
        "algorithm": "ED25519",
        "status": "SEALED_OPERATOR_PUBLIC_KEY",
        "public_key_hex": normalized,
        "public_key_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _verify_ed25519_signature(
    *, public_key_hex: str, signature_base64: Any, body: Mapping[str, Any]
) -> None:
    if not isinstance(signature_base64, str) or not signature_base64:
        raise DojoHistoricalRawReclaimError("trusted provider signature is missing")
    try:
        signature = base64.b64decode(signature_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise DojoHistoricalRawReclaimError(
            "trusted provider signature is not canonical base64"
        ) from exc
    if (
        len(signature) != 64
        or base64.b64encode(signature).decode("ascii") != signature_base64
    ):
        raise DojoHistoricalRawReclaimError(
            "trusted provider signature has an invalid Ed25519 encoding"
        )
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            signature,
            _canonical_bytes(body),
        )
    except InvalidSignature as exc:
        raise DojoHistoricalRawReclaimError(
            "trusted provider attestation signature is invalid"
        ) from exc
    except (ImportError, ValueError) as exc:
        raise DojoHistoricalRawReclaimError(
            "Ed25519 verifier is unavailable or the public key is invalid"
        ) from exc


def _utc_now() -> datetime:
    """Private test seam; public callers cannot choose authority enrollment time."""

    return datetime.now(timezone.utc)


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


def _strict_regular_path(value: Path | str, *, field: str) -> Path:
    """Resolve one explicit local input while rejecting every symlink component."""

    text = _bounded_text(str(value), field, maximum=4096)
    path = Path(text)
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise DojoHistoricalRawReclaimError(f"{field} must be an absolute safe path")
    cursor = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            cursor /= part
            state = cursor.lstat()
            if stat.S_ISLNK(state.st_mode):
                raise DojoHistoricalRawReclaimError(
                    f"{field} contains a symlink component"
                )
        final = path.stat(follow_symlinks=False)
    except DojoHistoricalRawReclaimError:
        raise
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(f"{field} is unavailable") from exc
    if not stat.S_ISREG(final.st_mode):
        raise DojoHistoricalRawReclaimError(f"{field} is not a regular file")
    return path.resolve(strict=True)


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


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _inode_identity(value: os.stat_result) -> tuple[int, int, int]:
    """Identity fields that remain stable while an inode is written or linked."""

    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _seal_zstd_executable(zstd_bin: str) -> dict[str, Any]:
    """Seal one absolute executable by stable bytes and bounded version output."""

    requested = Path(zstd_bin)
    if not requested.is_absolute():
        raise DojoHistoricalRawReclaimError("zstd executable must be an absolute path")
    try:
        executable = requested.resolve(strict=True)
        before = executable.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError("zstd executable is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size < 1
        or before.st_size > 512 * 1024**2
        or not os.access(executable, os.X_OK)
    ):
        raise DojoHistoricalRawReclaimError(
            "zstd path does not resolve to a bounded regular executable"
        )
    sha256, _, size, _ = _hashes_file(executable)
    try:
        completed = subprocess.run(
            [os.fspath(executable), "--version"],
            check=False,
            capture_output=True,
            timeout=10,
        )
        after = executable.stat(follow_symlinks=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DojoHistoricalRawReclaimError(
            "zstd executable cannot be version-sealed"
        ) from exc
    if _identity(before) != _identity(after) or completed.returncode != 0:
        raise DojoHistoricalRawReclaimError("zstd executable changed while sealed")
    try:
        version = (completed.stdout + completed.stderr).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise DojoHistoricalRawReclaimError("zstd version output is not UTF-8") from exc
    if not version or len(version.encode("utf-8")) > 4096:
        raise DojoHistoricalRawReclaimError("zstd version output is invalid")
    return {
        "zstd_executable_path": os.fspath(executable),
        "zstd_executable_sha256": sha256,
        "zstd_executable_size_bytes": size,
        "zstd_version": version,
    }


def _validated_restore_zstd(zstd_bin: str) -> str:
    return str(_seal_zstd_executable(zstd_bin)["zstd_executable_path"])


def _atomic_rename_between_at_no_replace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> bool:
    """Atomically move one leaf between anchored directories without replacement."""

    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError as exc:
            raise DojoHistoricalRawReclaimError(
                "atomic restore no-replace rename is unsupported"
            ) from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_directory_fd,
            source,
            destination_directory_fd,
            destination,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise DojoHistoricalRawReclaimError(
                "atomic restore no-replace rename is unsupported"
            ) from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_directory_fd,
            source,
            destination_directory_fd,
            destination,
            0x00000001,
        )
    else:
        raise DojoHistoricalRawReclaimError(
            "atomic restore no-replace rename is unsupported"
        )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        return False
    if error_number == errno.ENOENT:
        raise FileNotFoundError(error_number, os.strerror(error_number), source_name)
    if error_number in {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
        errno.EXDEV,
    }:
        raise DojoHistoricalRawReclaimError(
            "atomic restore no-replace rename is unsupported by the filesystem"
        )
    raise DojoHistoricalRawReclaimError(
        f"atomic restore no-replace rename failed: {os.strerror(error_number)}"
    )


def _open_anchored_parent(
    root: Path,
    relative: str,
    *,
    create_parents: bool,
) -> tuple[int, str, str]:
    """Open a lexical child parent by openat without following any component."""

    try:
        safe = _safe_relative(relative)
    except DojoHistoricalJobArchiveError as exc:
        raise DojoHistoricalRawReclaimError(str(exc)) from exc
    parts = PurePosixPath(safe).parts
    if not parts:
        raise DojoHistoricalRawReclaimError("anchored path is empty")
    try:
        descriptor = os.open(root, _directory_open_flags())
    except OSError as exc:
        raise DojoHistoricalRawReclaimError("anchored run root is unavailable") from exc
    try:
        for component in parts[:-1]:
            if create_parents:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise DojoHistoricalRawReclaimError(
                        f"restore parent cannot be created: {safe}"
                    ) from exc
            try:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise DojoHistoricalRawReclaimError(
                    f"anchored parent is unavailable: {safe}"
                ) from exc
            state = os.fstat(child)
            if not stat.S_ISDIR(state.st_mode):
                os.close(child)
                raise DojoHistoricalRawReclaimError(
                    f"anchored parent is not a directory: {safe}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1], safe
    except Exception:
        os.close(descriptor)
        raise


def _hash_open_regular(descriptor: int, *, field: str) -> tuple[str, int, int]:
    """Hash one already-open regular file while checking its stable identity."""

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DojoHistoricalRawReclaimError(f"{field} is not a regular file")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    except DojoHistoricalRawReclaimError:
        raise
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(f"{field} cannot be hashed") from exc
    if _identity(before) != _identity(after) or size != before.st_size:
        raise DojoHistoricalRawReclaimError(f"{field} changed while hashed")
    return digest.hexdigest(), size, before.st_blocks * 512


def _hash_anchored_file(
    root: Path,
    relative: str,
    *,
    missing_allowed: bool,
) -> tuple[str, int, int] | None:
    """Hash one root-anchored file and bind the opened fd to its dirent."""

    parent, name, safe = _open_anchored_parent(
        root,
        relative,
        create_parents=False,
    )
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent,
            )
        except FileNotFoundError:
            if missing_allowed:
                return None
            raise DojoHistoricalRawReclaimError(
                f"anchored file is missing: {safe}"
            ) from None
        except OSError as exc:
            raise DojoHistoricalRawReclaimError(
                f"anchored file is unavailable: {safe}"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            sha256, size, allocated = _hash_open_regular(
                descriptor,
                field=f"anchored file {safe}",
            )
            observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if _identity(opened) != _identity(observed):
                raise DojoHistoricalRawReclaimError(
                    f"anchored file dirent changed while hashed: {safe}"
                )
            return sha256, size, allocated
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _validate_local_archive(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    require_full_inventory: bool,
    zstd_bin: str,
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
                zstd_bin=zstd_bin,
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
            zstd_bin=zstd_bin,
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
        or not _drive_revision_id(row["head_revision_id"], "Drive head revision id")
        or row["trashed"] is not False
    ):
        raise DojoHistoricalRawReclaimError("Drive object metadata is invalid")
    return dict(row)


def _attestation_authority_namespace_paths(
    *,
    archive_root: Path,
    job_sha256: str,
    require_exact_one: bool,
) -> list[Path]:
    """Find every seal that claims one job, including non-canonical filenames."""

    job = _sha(job_sha256, "attestation authority job SHA-256")
    authority_root = archive_root / "remote-authorities"
    try:
        root_state = authority_root.stat(follow_symlinks=False)
    except FileNotFoundError:
        paths: list[Path] = []
        if require_exact_one:
            raise DojoHistoricalRawReclaimError(
                "attestation authority namespace does not contain exact-one seal"
            ) from None
        return paths
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            "attestation authority namespace is unavailable"
        ) from exc
    if not stat.S_ISDIR(root_state.st_mode) or authority_root.is_symlink():
        raise DojoHistoricalRawReclaimError("attestation authority namespace is unsafe")

    canonical_prefix = f"key-job-{job}-"
    matches: list[Path] = []
    try:
        with os.scandir(authority_root) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            "attestation authority namespace cannot be enumerated"
        ) from exc
    for entry in entries:
        name_claims_job = entry.name.startswith(canonical_prefix)
        if not entry.name.endswith(".json"):
            if name_claims_job:
                raise DojoHistoricalRawReclaimError(
                    "attestation authority namespace contains a malformed job seal"
                )
            continue
        path = authority_root / entry.name
        try:
            state = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise DojoHistoricalRawReclaimError(
                "attestation authority namespace changed while enumerated"
            ) from exc
        if not stat.S_ISREG(state.st_mode):
            raise DojoHistoricalRawReclaimError(
                "attestation authority namespace contains a non-regular JSON entry"
            )
        try:
            candidate = _read_canonical_json(
                path,
                field="attestation authority namespace entry",
            )
        except DojoHistoricalRawReclaimError:
            if name_claims_job:
                raise
            continue
        if name_claims_job or candidate.get("job_sha256") == job:
            matches.append(path.resolve(strict=True))
    if len(matches) > 1:
        raise DojoHistoricalRawReclaimError(
            "multiple attestation public-key seals or forks name this job"
        )
    if require_exact_one and len(matches) != 1:
        raise DojoHistoricalRawReclaimError(
            "attestation authority namespace does not contain exact-one seal"
        )
    return matches


def _validate_attestation_public_key_seal(
    *,
    path: Path,
    archive_root: Path,
    local: Mapping[str, Any],
    expected_drive_parent_id: str,
) -> dict[str, Any]:
    namespace_paths = _attestation_authority_namespace_paths(
        archive_root=archive_root,
        job_sha256=str(local["job_sha256"]),
        require_exact_one=True,
    )
    seal_path = _strict_regular_path(path, field="attestation public-key seal")
    if namespace_paths != [seal_path]:
        raise DojoHistoricalRawReclaimError(
            "explicit attestation public-key seal differs from the job authority"
        )
    seal = _read_canonical_json(seal_path, field="attestation public-key seal")
    _exact(
        seal,
        _ATTESTATION_PUBLIC_KEY_SEAL_KEYS,
        "attestation public-key seal",
    )
    authority = _normalize_public_key(seal.get("public_key_hex"))
    body = {key: value for key, value in seal.items() if key != "authority_seal_sha256"}
    if (
        seal.get("contract") != ATTESTATION_PUBLIC_KEY_SEAL_CONTRACT
        or seal.get("schema_version") != 2
        or seal.get("status") != "OPERATOR_PUBLIC_KEY_ENROLLED_BEFORE_READBACK"
        or seal.get("job_sha256") != local["job_sha256"]
        or seal.get("manifest_sha256") != local["manifest_sha256"]
        or seal.get("local_archive_receipt_sha256") != local["receipt_sha256"]
        or seal.get("expected_drive_parent_id") != expected_drive_parent_id
        or seal.get("algorithm") != "ED25519"
        or seal.get("public_key_sha256") != authority["public_key_sha256"]
        or seal.get("private_key_material_accepted") is not False
        or any(seal.get(key) != value for key, value in _AUTHORITY.items())
        or seal.get("authority_seal_sha256") != _canonical_sha256(body)
    ):
        raise DojoHistoricalRawReclaimError("attestation public-key seal is invalid")
    _utc(seal.get("enrolled_at_utc"), "attestation key enrolled_at_utc")
    expected = (
        archive_root
        / "remote-authorities"
        / (
            f"key-job-{local['job_sha256']}-{local['manifest_sha256']}-"
            f"{seal['authority_seal_sha256']}.json"
        )
    ).resolve(strict=False)
    if seal_path != expected:
        raise DojoHistoricalRawReclaimError(
            "attestation public-key seal path is not canonical"
        )
    return seal


def enroll_historical_job_attestation_public_key(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    expected_drive_parent_id: str,
    attestation_public_key_hex: str,
    zstd_bin: str,
) -> dict[str, Any]:
    """Enroll one public key before external Drive observation.

    Only a 32-byte public key is accepted.  There is deliberately no private
    key parameter, loader, generator, or persistence surface in this module.
    """

    root = Path(run_root).resolve(strict=True)
    local_path = _strict_regular_path(
        archive_receipt_path,
        field="local archive receipt path",
    )
    archive_root, local, _ = _validate_local_archive(
        run_root=root,
        archive_receipt_path=local_path,
        require_full_inventory=True,
        zstd_bin=str(_seal_zstd_executable(zstd_bin)["zstd_executable_path"]),
    )
    parent_id = _drive_id(
        expected_drive_parent_id,
        "operator expected Drive parent id",
    )
    authority = _normalize_public_key(attestation_public_key_hex)
    enrolled = _utc_now()
    if enrolled.tzinfo is None or enrolled.utcoffset() != timedelta(0):
        raise DojoHistoricalRawReclaimError("key enrollment time must be UTC")
    body = {
        "contract": ATTESTATION_PUBLIC_KEY_SEAL_CONTRACT,
        "schema_version": 2,
        "status": "OPERATOR_PUBLIC_KEY_ENROLLED_BEFORE_READBACK",
        "job_sha256": local["job_sha256"],
        "manifest_sha256": local["manifest_sha256"],
        "local_archive_receipt_sha256": local["receipt_sha256"],
        "expected_drive_parent_id": parent_id,
        "algorithm": "ED25519",
        "public_key_hex": authority["public_key_hex"],
        "public_key_sha256": authority["public_key_sha256"],
        "enrolled_at_utc": enrolled.isoformat(),
        "private_key_material_accepted": False,
        **_AUTHORITY,
    }
    seal = {**body, "authority_seal_sha256": _canonical_sha256(body)}
    authority_root = archive_root / "remote-authorities"
    authority_root.mkdir(mode=0o700, exist_ok=True)
    if authority_root.is_symlink() or not stat.S_ISDIR(
        authority_root.stat(follow_symlinks=False).st_mode
    ):
        raise DojoHistoricalRawReclaimError("attestation authority directory is unsafe")
    prefix = f"key-job-{local['job_sha256']}-{local['manifest_sha256']}"
    output = authority_root / f"{prefix}-{seal['authority_seal_sha256']}.json"
    with _exclusive_lock(
        authority_root / f".{prefix}.lock",
        field="attestation public-key enrollment lock",
    ) as enrollment_lock:
        existing = _attestation_authority_namespace_paths(
            archive_root=archive_root,
            job_sha256=local["job_sha256"],
            require_exact_one=False,
        )
        if existing and existing != [output]:
            raise DojoHistoricalRawReclaimError(
                "another attestation public key is already enrolled for this job"
            )
        enrollment_lock.assert_stable()
        _write_once(output, seal, field="attestation public-key seal")
        enrollment_lock.assert_stable()
        if _attestation_authority_namespace_paths(
            archive_root=archive_root,
            job_sha256=local["job_sha256"],
            require_exact_one=True,
        ) != [output]:
            raise DojoHistoricalRawReclaimError(
                "attestation public-key enrollment became ambiguous"
            )
        return _validate_attestation_public_key_seal(
            path=output,
            archive_root=archive_root,
            local=local,
            expected_drive_parent_id=parent_id,
        )


def _validate_remote_receipt(
    *,
    remote_receipt_path: Path,
    archive_root: Path,
    local: Mapping[str, Any],
    expected_drive_parent_id: str,
    attestation_authority_seal_path: Path | None,
    now_utc: datetime | None = None,
    allow_expired: bool = False,
    require_canonical_path: bool = True,
) -> dict[str, Any]:
    if attestation_authority_seal_path is None:
        raise DojoHistoricalRawReclaimError(
            "pre-enrolled attestation public-key seal is required"
        )
    authority_seal = _validate_attestation_public_key_seal(
        path=Path(attestation_authority_seal_path),
        archive_root=archive_root,
        local=local,
        expected_drive_parent_id=expected_drive_parent_id,
    )
    authority = _normalize_public_key(authority_seal["public_key_hex"])
    path = _strict_regular_path(
        remote_receipt_path,
        field="signed Drive attestation receipt",
    )
    receipt = _read_canonical_json(path, field="signed Drive attestation receipt")
    _exact(receipt, _SIGNED_ATTESTATION_KEYS, "signed Drive attestation receipt")
    body = _exact(
        receipt.get("body"),
        _SIGNED_ATTESTATION_BODY_KEYS,
        "signed Drive attestation body",
    )
    signed_receipt_body = {
        key: value for key, value in receipt.items() if key != "remote_receipt_sha256"
    }
    receipt_sha = _sha(
        receipt.get("remote_receipt_sha256"),
        "signed Drive attestation receipt SHA-256",
    )
    if (
        receipt.get("contract") != REMOTE_READBACK_SIGNED_ATTESTATION_CONTRACT
        or receipt.get("schema_version") != 2
        or receipt.get("algorithm") != "ED25519"
        or receipt.get("public_key_sha256") != authority["public_key_sha256"]
        or receipt_sha != _canonical_sha256(signed_receipt_body)
    ):
        raise DojoHistoricalRawReclaimError(
            "signed Drive attestation receipt seal is invalid"
        )
    _verify_ed25519_signature(
        public_key_hex=authority["public_key_hex"],
        signature_base64=receipt.get("signature_base64"),
        body=body,
    )

    issued = _utc(body.get("issued_at_utc"), "attestation issued_at_utc")
    expires = _utc(body.get("expires_at_utc"), "attestation expires_at_utc")
    readback_at = _utc(body.get("readback_at_utc"), "attestation readback_at_utc")
    observed_now = now_utc or datetime.now(timezone.utc)
    if observed_now.tzinfo is None or observed_now.utcoffset() != timedelta(0):
        raise DojoHistoricalRawReclaimError("attestation verifier clock must be UTC")
    if (
        expires <= issued
        or expires - issued > timedelta(seconds=MAX_ATTESTATION_TTL_SECONDS)
        or readback_at > issued
        or readback_at
        < _utc(
            authority_seal["enrolled_at_utc"],
            "attestation key enrolled_at_utc",
        )
        or issued - readback_at > timedelta(seconds=MAX_ATTESTATION_TTL_SECONDS)
        or observed_now < issued - timedelta(seconds=MAX_CLOCK_SKEW_SECONDS)
        or (not allow_expired and observed_now > expires)
    ):
        raise DojoHistoricalRawReclaimError(
            "signed Drive attestation is stale or outside its short TTL"
        )

    parent_id = _drive_id(
        body.get("expected_drive_parent_id"),
        "attestation expected Drive parent id",
    )
    expected_parent = _drive_id(
        expected_drive_parent_id,
        "operator expected Drive parent id",
    )
    if parent_id != expected_parent:
        raise DojoHistoricalRawReclaimError(
            "signed Drive attestation names another Drive parent"
        )
    _validate_drive_parent(body.get("drive_parent"), expected_parent_id=parent_id)
    expected_objects = local["remote_readback_objects"]
    if (
        body.get("contract") != REMOTE_READBACK_ATTESTATION_BODY_CONTRACT
        or body.get("schema_version") != 2
        or _sha(body.get("attestation_id"), "Drive attestation id")
        != body.get("attestation_id")
        or body.get("provider") != "GOOGLE_DRIVE"
        or body.get("verification_method")
        != "GOOGLE_DRIVE_V3_FILES_GET_REVISIONS_LIST_AND_INDEPENDENT_REVISION_READBACK"
        or body.get("job_sha256") != local["job_sha256"]
        or body.get("completion_sha256") != local["completion_sha256"]
        or body.get("bundle_kind") != SUCCESS_BUNDLE_KIND
        or body.get("manifest_sha256") != local["manifest_sha256"]
        or body.get("local_archive_receipt_sha256") != local["receipt_sha256"]
        or body.get("archive_sha256") != local["archive_sha256"]
        or body.get("archive_size_bytes") != local["archive_size_bytes"]
        or isinstance(body.get("archive_size_bytes"), bool)
        or body.get("object_set_sha256") != expected_objects["object_set_sha256"]
        or body.get("object_count") != expected_objects["object_count"]
        or isinstance(body.get("object_count"), bool)
        or any(
            body.get(field) is not True
            for field in (
                "files_get_before_after_verified",
                "revisions_list_head_present_unique",
                "independent_revision_readback_verified",
                "exact_revision_bytes_hashed",
                "download_bytes_match_local_objects",
                "drive_metadata_revision_bound",
                "drive_parents_bound",
                "drive_trashed_false",
                "external_readback_attested",
                "remote_verified",
                "raw_reclaim_eligible",
            )
        )
        or body.get("historical_train_is_proof") is not False
        or body.get("promotion_eligible") is not False
        or body.get("live_permission") is not False
        or body.get("order_authority") != "NONE"
        or body.get("broker_mutation_allowed") is not False
    ):
        raise DojoHistoricalRawReclaimError(
            "signed Drive attestation lineage or authority is invalid"
        )

    raw_objects = body.get("objects")
    if (
        not isinstance(raw_objects, list)
        or len(raw_objects) != expected_objects["object_count"]
    ):
        raise DojoHistoricalRawReclaimError(
            "signed Drive attestation object denominator is incomplete"
        )
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    seen_revisions: set[tuple[str, str, str]] = set()
    total = 0
    for index, (raw, expected) in enumerate(
        zip(raw_objects, expected_objects["objects"], strict=True)
    ):
        item = _exact(
            raw,
            _SIGNED_ATTESTATION_OBJECT_KEYS,
            "signed Drive attestation object",
        )
        downloaded = _exact(
            item.get("downloaded"),
            _DOWNLOAD_KEYS,
            "signed Drive downloaded revision",
        )
        expected_md5 = _md5(
            downloaded.get("md5_checksum"),
            "signed Drive downloaded revision MD5",
        )
        local_object_path = _strict_regular_path(
            archive_root / expected["relative_path"],
            field="local archive readback object",
        )
        local_sha, local_md5, local_size, _ = _hashes_file(local_object_path)
        before = _validate_metadata(
            item.get("metadata_before"),
            expected_parent_id=parent_id,
            expected_name=Path(expected["relative_path"]).name,
            expected_size=expected["size_bytes"],
            expected_md5=expected_md5,
            readback_at=readback_at,
        )
        after = _validate_metadata(
            item.get("metadata_after"),
            expected_parent_id=parent_id,
            expected_name=Path(expected["relative_path"]).name,
            expected_size=expected["size_bytes"],
            expected_md5=expected_md5,
            readback_at=readback_at,
        )
        listed = item.get("listed_revision_ids")
        if (
            before != after
            or not isinstance(listed, list)
            or not listed
            or len(listed) > 100_000
            or len(listed) != len(set(listed))
            or any(
                _drive_revision_id(value, "listed Drive revision id") != value
                for value in listed
            )
            or listed.count(after["head_revision_id"]) != 1
            or item.get("drivefs_file_id_xattr") != after["drive_file_id"]
            or item.get("drivefs_md5_field48") != expected_md5
            or item.get("drivefs_version_field57") != after["version"]
            or item.get("drivefs_current_revision_id_field78")
            != after["head_revision_id"]
            or item.get("index") != index
            or isinstance(item.get("index"), bool)
            or item.get("index") != expected["index"]
            or item.get("offset_bytes") != expected["offset_bytes"]
            or isinstance(item.get("offset_bytes"), bool)
            or item.get("relative_path") != expected["relative_path"]
            or item.get("size_bytes") != expected["size_bytes"]
            or isinstance(item.get("size_bytes"), bool)
            or item.get("sha256") != expected["sha256"]
            or downloaded.get("content_size_bytes") != expected["size_bytes"]
            or isinstance(downloaded.get("content_size_bytes"), bool)
            or downloaded.get("sha256") != expected["sha256"]
            or local_sha != expected["sha256"]
            or local_size != expected["size_bytes"]
            or local_md5 != expected_md5
        ):
            raise DojoHistoricalRawReclaimError(
                "signed Drive attestation object or revision list drifted"
            )
        identity = (
            after["drive_file_id"],
            after["version"],
            after["head_revision_id"],
        )
        if (
            after["drive_file_id"] in seen_ids
            or after["drive_file_name"] in seen_names
            or identity in seen_revisions
        ):
            raise DojoHistoricalRawReclaimError(
                "signed Drive attestation reuses a file or revision"
            )
        seen_ids.add(after["drive_file_id"])
        seen_names.add(after["drive_file_name"])
        seen_revisions.add(identity)
        total += expected["size_bytes"]
    if total != local["archive_size_bytes"]:
        raise DojoHistoricalRawReclaimError(
            "signed Drive attestation byte denominator is incomplete"
        )

    if require_canonical_path:
        expected_path = (
            archive_root
            / "remote-receipts"
            / (
                f"signed-job-{local['job_sha256']}-{local['manifest_sha256']}-"
                f"{receipt_sha}.json"
            )
        ).resolve(strict=False)
        if path != expected_path:
            raise DojoHistoricalRawReclaimError(
                "signed Drive attestation receipt path is not canonical"
            )
    return receipt


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
        observed = _hash_anchored_file(
            root,
            row["path"],
            missing_allowed=missing_allowed,
        )
        if observed is None:
            missing.add(row["path"])
            continue
        sha256, size, blocks = observed
        if sha256 != row["sha256"] or size != row["size_bytes"]:
            raise DojoHistoricalRawReclaimError(
                f"reclaim inventory bytes drifted: {row['path']}"
            )
        allocated += blocks
    return missing, allocated


def _verify_no_unmanifested_job_files(
    *,
    root: Path,
    job_sha256: str,
    manifest_paths: set[str],
    allowed_retirement_anchors: set[str] | None = None,
) -> None:
    allowed_anchors = allowed_retirement_anchors or set()
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
            if relative not in manifest_paths and relative not in allowed_anchors:
                raise DojoHistoricalRawReclaimError(
                    f"unmanifested job evidence appeared: {relative}"
                )


def _prepare_reclaim(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    remote_receipt_path: Path,
    expected_drive_parent_id: str,
    attestation_authority_seal_path: Path | None,
    permit_missing_targets: bool,
    zstd_bin: str,
    now_utc: datetime | None = None,
) -> _PreparedReclaim:
    zstd_seal = _seal_zstd_executable(zstd_bin)
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
        zstd_bin=str(zstd_seal["zstd_executable_path"]),
    )
    if attestation_authority_seal_path is None:
        raise DojoHistoricalRawReclaimError(
            "pre-enrolled attestation public-key seal is required"
        )
    authority_seal = _validate_attestation_public_key_seal(
        path=Path(attestation_authority_seal_path),
        archive_root=archive_root,
        local=local,
        expected_drive_parent_id=expected_drive_parent_id,
    )
    remote = _validate_remote_receipt(
        remote_receipt_path=remote_receipt_path,
        archive_root=archive_root,
        local=local,
        expected_drive_parent_id=expected_drive_parent_id,
        attestation_authority_seal_path=attestation_authority_seal_path,
        now_utc=now_utc,
        allow_expired=permit_missing_targets,
    )
    targets, retained = _reclaimable_rows(
        run_root=root,
        job_sha256=local["job_sha256"],
        manifest=manifest,
    )
    manifest_paths = {row["path"] for row in manifest["files"]}
    allowed_retirement_anchors = {
        (
            PurePosixPath(row["path"]).parent / _raw_retirement_anchor_name(row)
        ).as_posix()
        for row in targets
    }
    _verify_no_unmanifested_job_files(
        root=root,
        job_sha256=local["job_sha256"],
        manifest_paths=manifest_paths,
        allowed_retirement_anchors=(
            allowed_retirement_anchors if permit_missing_targets else None
        ),
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
    if permit_missing_targets:
        for row in targets:
            if row["path"] in missing:
                _verify_raw_retirement_anchor(root, row)
    _verify_rows(root, retained, missing_allowed=False)
    return _PreparedReclaim(
        run_root=root,
        archive_root=archive_root,
        archive_receipt_path=archive_receipt_path.resolve(strict=True),
        remote_receipt_path=remote_receipt_path.resolve(strict=True),
        local_receipt=local,
        remote_receipt=remote,
        attestation_authority_seal=authority_seal,
        zstd_seal=zstd_seal,
        manifest=manifest,
        targets=targets,
        retained=retained,
    )


def _plan_body(prepared: _PreparedReclaim) -> dict[str, Any]:
    attestation = prepared.remote_receipt["body"]
    return {
        "contract": RECLAIM_PLAN_CONTRACT_V2,
        "schema_version": 2,
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
        "reclaim_mode": RECLAIM_MODE,
        "target_count": len(prepared.targets),
        "target_bytes": sum(row["size_bytes"] for row in prepared.targets),
        "targets": list(prepared.targets),
        "retained_file_count": len(prepared.retained),
        "retained_bytes": sum(row["size_bytes"] for row in prepared.retained),
        "full_source_inventory_verified": True,
        "remote_raw_readback_verified": True,
        "attestation_id": attestation["attestation_id"],
        "attestation_public_key_sha256": prepared.remote_receipt["public_key_sha256"],
        "attestation_public_key_hex": prepared.attestation_authority_seal[
            "public_key_hex"
        ],
        "attestation_authority_seal_sha256": prepared.attestation_authority_seal[
            "authority_seal_sha256"
        ],
        "attestation_authority_enrolled_at_utc": prepared.attestation_authority_seal[
            "enrolled_at_utc"
        ],
        "attestation_issued_at_utc": attestation["issued_at_utc"],
        "attestation_expires_at_utc": attestation["expires_at_utc"],
        **prepared.zstd_seal,
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
        directory = os.open(path.parent, _directory_open_flags())
        try:
            published = _atomic_rename_at_no_replace(
                directory,
                temporary.name,
                path.name,
            )
        finally:
            os.close(directory)
        if not published:
            current = _stable_regular_bytes(path, field=field, maximum=MAX_JSON_BYTES)
            if current != payload:
                raise DojoHistoricalRawReclaimError(f"{field} concurrently drifted")
        _fsync_directory(path.parent)
    except DojoHistoricalJobArchiveError as exc:
        raise DojoHistoricalRawReclaimError(str(exc)) from exc
    finally:
        _remove_anchored_verified_file(
            temporary.parent,
            temporary.name,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size_bytes=len(payload),
        )


def _write_once_anchored(
    root: Path,
    relative: str,
    value: Mapping[str, Any],
    *,
    field: str,
) -> None:
    """Atomically rename one immutable JSON file under an anchored directory fd."""

    payload = _canonical_bytes(value) + b"\n"
    parent, leaf, safe = _open_anchored_parent(
        root,
        relative,
        create_parents=False,
    )

    def read_existing() -> bytes | None:
        try:
            descriptor = os.open(
                leaf,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise DojoHistoricalRawReclaimError(
                f"{field} cannot be opened safely: {safe}"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON_BYTES:
                raise DojoHistoricalRawReclaimError(f"{field} is invalid")
            chunks: list[bytes] = []
            size = 0
            while chunk := os.read(
                descriptor, min(HASH_CHUNK_BYTES, MAX_JSON_BYTES + 1)
            ):
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_JSON_BYTES:
                    raise DojoHistoricalRawReclaimError(f"{field} is oversized")
            after = os.fstat(descriptor)
            observed = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if (
                _identity(before) != _identity(after)
                or _identity(after) != _identity(observed)
                or size != before.st_size
            ):
                raise DojoHistoricalRawReclaimError(f"{field} changed while read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    temporary_leaf: str | None = None
    temporary_relative: str | None = None
    try:
        current = read_existing()
        if current is not None:
            if current != payload:
                raise DojoHistoricalRawReclaimError(f"{field} already drifted")
            return
        temporary_leaf = f".{leaf}.{os.getpid()}.{os.urandom(12).hex()}.tmp"
        relative_parent = PurePosixPath(safe).parent
        temporary_relative = (
            temporary_leaf
            if relative_parent == PurePosixPath(".")
            else (relative_parent / temporary_leaf).as_posix()
        )
        descriptor = os.open(
            temporary_leaf,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise DojoHistoricalRawReclaimError(
                        f"{field} temporary write made no progress"
                    )
                remaining = remaining[written:]
            os.fsync(descriptor)
            if os.fstat(descriptor).st_size != len(payload):
                raise DojoHistoricalRawReclaimError(f"{field} temporary size changed")
        finally:
            os.close(descriptor)
        published = _atomic_rename_at_no_replace(
            parent,
            temporary_leaf,
            leaf,
        )
        if not published:
            current = read_existing()
            if current != payload:
                raise DojoHistoricalRawReclaimError(f"{field} concurrently drifted")
        else:
            current = read_existing()
            if current != payload:
                raise DojoHistoricalRawReclaimError(f"{field} publish drifted")
        os.fsync(parent)
    except DojoHistoricalRawReclaimError:
        raise
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(f"{field} write failed") from exc
    finally:
        if temporary_relative is not None:
            _remove_anchored_verified_file(
                root,
                temporary_relative,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size_bytes=len(payload),
            )
        os.close(parent)


def _open_absolute_parent(path: Path, *, field: str) -> tuple[int, str]:
    """Open one absolute lexical parent without following a symlink component."""

    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise DojoHistoricalRawReclaimError(f"{field} path is unsafe")
    try:
        descriptor = os.open(Path(path.anchor), _directory_open_flags())
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(f"{field} parent is unavailable") from exc
    try:
        for component in path.parts[1:-1]:
            if component in {"", ".", ".."}:
                raise DojoHistoricalRawReclaimError(f"{field} path is unsafe")
            try:
                child = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise DojoHistoricalRawReclaimError(
                    f"{field} parent is unavailable"
                ) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor, path.name
    except Exception:
        os.close(descriptor)
        raise


def _sealed_run_control_global_heavy_lock(root: Path) -> Path | None:
    """Read the optional global lease binding from the sealed run control."""

    manifest_path = root / "control-manifest.json"
    control_path = root / "sealed-inputs" / "run-control.json"
    manifest = _read_canonical_json(
        manifest_path,
        field="generation control manifest for global heavy lease",
    )
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != _canonical_sha256(manifest_body):
        raise DojoHistoricalRawReclaimError(
            "generation control manifest global lease binding is unsealed"
        )
    rows = manifest.get("sealed_input_artifacts")
    if not isinstance(rows, list) or manifest.get(
        "sealed_input_artifacts_sha256"
    ) != _canonical_sha256(rows):
        raise DojoHistoricalRawReclaimError(
            "generation sealed input inventory global lease binding is invalid"
        )
    run_control_rows = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("artifact_id") == "RUN_CONTROL"
    ]
    if len(run_control_rows) != 1:
        raise DojoHistoricalRawReclaimError(
            "generation has no unique sealed run-control global lease binding"
        )
    row = _exact(
        run_control_rows[0],
        {"artifact_id", "relative_path", "file_sha256", "file_size_bytes"},
        "sealed run-control inventory row",
    )
    if row.get("relative_path") != "sealed-inputs/run-control.json":
        raise DojoHistoricalRawReclaimError(
            "sealed run-control global lease path is noncanonical"
        )
    raw = _stable_regular_bytes(
        control_path,
        field="sealed run control for global heavy lease",
        maximum=MAX_JSON_BYTES,
    )
    if _sha(
        row.get("file_sha256"), "sealed run-control file SHA-256"
    ) != hashlib.sha256(raw).hexdigest() or _positive_integer(
        row.get("file_size_bytes"), "sealed run-control file size"
    ) != len(raw):
        raise DojoHistoricalRawReclaimError(
            "sealed run-control bytes do not match the generation manifest"
        )
    control = _read_canonical_json(
        control_path,
        field="sealed run control for global heavy lease",
    )
    execution = control.get("execution")
    if execution is None:
        return None
    if not isinstance(execution, Mapping):
        raise DojoHistoricalRawReclaimError(
            "sealed run-control execution binding is invalid"
        )
    value = execution.get("global_heavy_lock_path")
    if value is None:
        return None
    if not isinstance(value, str):
        raise DojoHistoricalRawReclaimError("sealed global heavy lock path is invalid")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise DojoHistoricalRawReclaimError("sealed global heavy lock path is unsafe")
    return path


def historical_global_heavy_lock_path(*, run_root: Path) -> Path | None:
    """Return the machine-wide lease path sealed into one generation."""

    root = Path(run_root).resolve(strict=True)
    return _sealed_run_control_global_heavy_lock(root)


def _bound_global_heavy_lock_path(
    *, root: Path, requested_path: Path | None
) -> Path | None:
    """Validate an explicit lease against the sealed control when available."""

    if requested_path is None:
        # The train controller already owns this lease when it calls custody
        # verification.  Absence therefore preserves that nested call path.
        return None
    path = Path(requested_path)
    if (
        not path.is_absolute()
        or path.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise DojoHistoricalRawReclaimError(
            "global heavy operation lease path is unsafe"
        )
    sealed_path = _sealed_run_control_global_heavy_lock(root)
    if sealed_path is not None and path != sealed_path:
        raise DojoHistoricalRawReclaimError(
            "explicit global heavy lease differs from the sealed run control"
        )
    return path


@contextmanager
def _exclusive_lock(path: Path, *, field: str) -> Iterator[_OpenedLock]:
    lock_path = Path(path)
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor, leaf = _open_absolute_parent(lock_path, field=field)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            leaf,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode):
            raise DojoHistoricalRawReclaimError(f"{field} is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
        raise DojoHistoricalRawReclaimError(f"{field} is already held") from exc
    except DojoHistoricalRawReclaimError:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
        raise DojoHistoricalRawReclaimError(f"{field} is unavailable") from exc
    assert descriptor is not None
    guard = _OpenedLock(
        path=lock_path,
        descriptor=descriptor,
        parent_descriptor=parent_descriptor,
        identity=_inode_identity(state),
        parent_identity=_inode_identity(os.fstat(parent_descriptor)),
        field=field,
    )
    try:
        guard.assert_stable()
        yield guard
    finally:
        try:
            guard.assert_stable()
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
                os.close(parent_descriptor)


@contextmanager
def _optional_global_heavy_lease(
    *, root: Path, requested_path: Path | None
) -> Iterator[_OpenedLock | None]:
    path = _bound_global_heavy_lock_path(root=root, requested_path=requested_path)
    if path is None:
        yield None
        return
    with _exclusive_lock(path, field="global heavy operation lease") as guard:
        guard.assert_stable()
        yield guard
        guard.assert_stable()


def _ensure_remote_candidate_directory(archive_root: Path) -> Path:
    remote_root = archive_root / "remote-candidates"
    try:
        remote_root.mkdir(mode=0o700, exist_ok=True)
        state = remote_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            "remote candidate directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(state.st_mode) or remote_root.is_symlink():
        raise DojoHistoricalRawReclaimError(
            "remote candidate directory is not a real directory"
        )
    try:
        _fsync_directory(archive_root)
    except DojoHistoricalJobArchiveError as exc:
        raise DojoHistoricalRawReclaimError(str(exc)) from exc
    return remote_root


def _remote_candidate_from_evidence(
    *,
    run_root: Path,
    archive_root: Path,
    local: Mapping[str, Any],
    packet: Any,
    expected_drive_parent_id: str,
) -> dict[str, Any]:
    row = _exact(packet, _REMOTE_EVIDENCE_PACKET_KEYS, "remote evidence packet")
    parent_id = _drive_id(expected_drive_parent_id, "expected Drive parent id")
    object_set = local["remote_readback_objects"]
    objects = row["objects"]
    readback_at_text = _bounded_text(
        row["readback_at_utc"], "Drive readback_at_utc", maximum=64
    )
    readback_at = _utc(readback_at_text, "Drive readback_at_utc")
    if (
        row["contract"] != REMOTE_READBACK_EVIDENCE_CONTRACT
        or isinstance(row["schema_version"], bool)
        or row["schema_version"] != SCHEMA_VERSION
        or row["provider"] != "GOOGLE_DRIVE"
        or row["verification_method"] != "AUTHENTICATED_EXTERNAL_RAW_READBACK"
        or row["job_sha256"] != local["job_sha256"]
        or row["completion_sha256"] != local["completion_sha256"]
        or row["manifest_sha256"] != local["manifest_sha256"]
        or row["local_archive_receipt_sha256"] != local["receipt_sha256"]
        or row["object_set_sha256"] != object_set["object_set_sha256"]
        or isinstance(row["object_count"], bool)
        or row["object_count"] != object_set["object_count"]
        or row["expected_drive_parent_id"] != parent_id
        or not isinstance(objects, list)
        or len(objects) != object_set["object_count"]
        or row["external_readback_attested"] is not True
        or row["source_deletion_requested"] is not False
    ):
        raise DojoHistoricalRawReclaimError(
            "remote evidence packet lineage or authority is invalid"
        )
    _validate_drive_parent(row["drive_parent"], expected_parent_id=parent_id)

    verified_objects: list[dict[str, Any]] = []
    seen_file_ids: set[str] = set()
    seen_revisions: set[tuple[str, str, str]] = set()
    seen_head_revision_ids: set[str] = set()
    seen_download_files: set[tuple[int, int]] = set()
    total = 0
    for index, (observed, expected) in enumerate(
        zip(objects, object_set["objects"], strict=True)
    ):
        item = _exact(observed, _REMOTE_EVIDENCE_OBJECT_KEYS, "remote evidence object")
        relative = _safe_relative(expected["relative_path"])
        if (
            isinstance(item["index"], bool)
            or item["index"] != index
            or item["relative_path"] != relative
            or item["readback_at_utc"] != readback_at_text
            or expected["index"] != index
            or expected["offset_bytes"] != total
        ):
            raise DojoHistoricalRawReclaimError(
                "remote evidence object does not bind the local object set"
            )
        local_path = _safe_path(archive_root, relative)
        local_sha, local_md5, local_size, _ = _hashes_file(local_path)
        if local_sha != expected["sha256"] or local_size != expected["size_bytes"]:
            raise DojoHistoricalRawReclaimError(
                "local readback object differs from its sealed object set"
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
        if metadata_before != metadata_after:
            raise DojoHistoricalRawReclaimError(
                "Drive metadata or revision drifted during raw readback"
            )
        downloaded_path = _strict_regular_path(
            item["downloaded_local_path"], field="downloaded Drive object"
        )
        if (
            downloaded_path == local_path
            or archive_root in downloaded_path.parents
            or run_root in downloaded_path.parents
        ):
            raise DojoHistoricalRawReclaimError(
                "downloaded Drive object is not independent from local evidence"
            )
        downloaded_state = downloaded_path.stat(follow_symlinks=False)
        downloaded_identity = (downloaded_state.st_dev, downloaded_state.st_ino)
        if downloaded_identity in seen_download_files:
            raise DojoHistoricalRawReclaimError(
                "remote evidence reuses one downloaded object"
            )
        download_sha, download_md5, download_size, _ = _hashes_file(downloaded_path)
        if (
            download_sha != local_sha
            or download_md5 != local_md5
            or download_size != local_size
        ):
            raise DojoHistoricalRawReclaimError(
                "downloaded Drive bytes differ from the local object"
            )
        file_id = metadata_after["drive_file_id"]
        head_revision_id = metadata_after["head_revision_id"]
        revision = (file_id, metadata_after["version"], head_revision_id)
        if (
            file_id in seen_file_ids
            or revision in seen_revisions
            or head_revision_id in seen_head_revision_ids
        ):
            raise DojoHistoricalRawReclaimError(
                "remote evidence reuses a Drive object or revision"
            )
        seen_file_ids.add(file_id)
        seen_revisions.add(revision)
        seen_head_revision_ids.add(head_revision_id)
        seen_download_files.add(downloaded_identity)
        verified_objects.append(
            {
                "index": index,
                "offset_bytes": total,
                "relative_path": relative,
                "size_bytes": local_size,
                "sha256": local_sha,
                "metadata_before": metadata_before,
                "metadata_after": metadata_after,
                "downloaded": {
                    "content_size_bytes": download_size,
                    "sha256": download_sha,
                    "md5_checksum": download_md5,
                },
            }
        )
        total += local_size
    if total != local["archive_size_bytes"]:
        raise DojoHistoricalRawReclaimError(
            "remote evidence object size denominator is incomplete"
        )
    body = {
        "contract": REMOTE_READBACK_CANDIDATE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "CANDIDATE_ONLY",
        "provider": "GOOGLE_DRIVE",
        "verification_method": "UNTRUSTED_LOCAL_EVIDENCE_PACKET",
        "job_sha256": local["job_sha256"],
        "completion_sha256": local["completion_sha256"],
        "bundle_kind": SUCCESS_BUNDLE_KIND,
        "manifest_sha256": local["manifest_sha256"],
        "local_archive_receipt_sha256": local["receipt_sha256"],
        "archive_sha256": local["archive_sha256"],
        "archive_size_bytes": local["archive_size_bytes"],
        "object_set_sha256": object_set["object_set_sha256"],
        "object_count": object_set["object_count"],
        "expected_drive_parent_id": parent_id,
        "drive_parent": dict(row["drive_parent"]),
        "readback_at_utc": readback_at_text,
        "objects": verified_objects,
        "download_bytes_match_local_objects": True,
        "drive_metadata_revision_claims_well_formed": True,
        "external_readback_attested": False,
        "remote_verified": False,
        "raw_reclaim_eligible": False,
        "trusted_provider_attestation_present": False,
        **_AUTHORITY,
    }
    return {**body, "candidate_sha256": _canonical_sha256(body)}


def create_historical_job_remote_readback_receipt(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    evidence_packet_path: Path,
    expected_drive_parent_id: str,
    zstd_bin: str,
) -> dict[str, Any]:
    """Seal a non-authoritative candidate; it can never authorize raw removal."""

    try:
        root = Path(run_root).resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError("run root is unavailable") from exc
    if not root.is_dir():
        raise DojoHistoricalRawReclaimError("run root is not a directory")
    receipt_path = _strict_regular_path(
        archive_receipt_path, field="local archive receipt path"
    )
    packet_path = _strict_regular_path(
        evidence_packet_path, field="remote evidence packet path"
    )
    archive_root, local, _ = _validate_local_archive(
        run_root=root,
        archive_receipt_path=receipt_path,
        require_full_inventory=True,
        zstd_bin=str(_seal_zstd_executable(zstd_bin)["zstd_executable_path"]),
    )
    try:
        packet = _read_json(packet_path, field="remote evidence packet")
    except DojoHistoricalJobArchiveError as exc:
        raise DojoHistoricalRawReclaimError(str(exc)) from exc
    candidate = _remote_candidate_from_evidence(
        run_root=root,
        archive_root=archive_root,
        local=local,
        packet=packet,
        expected_drive_parent_id=expected_drive_parent_id,
    )
    remote_root = _ensure_remote_candidate_directory(archive_root)
    prefix = f"candidate-job-{local['job_sha256']}-{local['manifest_sha256']}"
    lock_path = remote_root / f".{prefix}.lock"
    with _exclusive_lock(
        lock_path,
        field="remote candidate seal lock",
    ) as candidate_lock:
        existing = sorted(remote_root.glob(f"{prefix}-*.json"))
        if len(existing) > 1:
            raise DojoHistoricalRawReclaimError(
                "multiple remote readback candidates name one job"
            )
        if existing:
            sealed = _read_canonical_json(
                existing[0],
                field="remote readback candidate",
            )
            if sealed != candidate:
                raise DojoHistoricalRawReclaimError(
                    "existing remote readback candidate drifted"
                )
            return sealed
        output = remote_root / f"{prefix}-{candidate['candidate_sha256']}.json"
        candidate_lock.assert_stable()
        _write_once(output, candidate, field="remote readback candidate")
        candidate_lock.assert_stable()
        sealed = _read_canonical_json(
            output,
            field="remote readback candidate",
        )
        if set(sealed) != _REMOTE_CANDIDATE_KEYS or sealed != candidate:
            raise DojoHistoricalRawReclaimError(
                "sealed remote readback candidate changed during verification"
            )
        if len(list(remote_root.glob(f"{prefix}-*.json"))) != 1:
            raise DojoHistoricalRawReclaimError(
                "remote readback candidate set became ambiguous"
            )
        return sealed


def publish_historical_job_signed_remote_readback_receipt(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    signed_attestation_path: Path,
    expected_drive_parent_id: str,
    attestation_authority_seal_path: Path,
    zstd_bin: str,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Validate and append one provider-signed short-lived Drive receipt.

    The signer remains external.  This process receives only the public key and
    never creates, reads, or persists private-key material.
    """

    try:
        root = Path(run_root).resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError("run root is unavailable") from exc
    local_path = _strict_regular_path(
        archive_receipt_path,
        field="local archive receipt path",
    )
    packet_path = _strict_regular_path(
        signed_attestation_path,
        field="external signed Drive attestation path",
    )
    archive_root, local, _ = _validate_local_archive(
        run_root=root,
        archive_receipt_path=local_path,
        require_full_inventory=True,
        zstd_bin=str(_seal_zstd_executable(zstd_bin)["zstd_executable_path"]),
    )
    receipt = _validate_remote_receipt(
        remote_receipt_path=packet_path,
        archive_root=archive_root,
        local=local,
        expected_drive_parent_id=expected_drive_parent_id,
        attestation_authority_seal_path=attestation_authority_seal_path,
        now_utc=now_utc,
        require_canonical_path=False,
    )
    remote_root = archive_root / "remote-receipts"
    try:
        remote_root.mkdir(mode=0o700, exist_ok=True)
        remote_state = remote_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            "signed Drive receipt directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(remote_state.st_mode) or remote_root.is_symlink():
        raise DojoHistoricalRawReclaimError(
            "signed Drive receipt directory is not a real directory"
        )
    prefix = f"signed-job-{local['job_sha256']}-{local['manifest_sha256']}"
    output = remote_root / f"{prefix}-{receipt['remote_receipt_sha256']}.json"
    with _exclusive_lock(
        remote_root / f".{prefix}.lock",
        field="signed Drive receipt publication lock",
    ) as publication_lock:
        publication_lock.assert_stable()
        _validate_attestation_public_key_seal(
            path=attestation_authority_seal_path,
            archive_root=archive_root,
            local=local,
            expected_drive_parent_id=expected_drive_parent_id,
        )
        same_lineage_paths = sorted(remote_root.glob(f"{prefix}-*.json"))
        if len(same_lineage_paths) > 1:
            raise DojoHistoricalRawReclaimError(
                "multiple signed Drive attestations exist for this job and manifest"
            )
        if same_lineage_paths:
            existing_path = same_lineage_paths[0]
            existing = _read_canonical_json(
                existing_path,
                field="existing signed Drive receipt for job and manifest",
            )
            if existing_path == output and existing == receipt:
                return existing
            existing_body = existing.get("body")
            if (
                isinstance(existing_body, Mapping)
                and existing_body.get("attestation_id")
                == receipt["body"]["attestation_id"]
            ):
                raise DojoHistoricalRawReclaimError(
                    "signed Drive attestation id was replayed"
                )
            raise DojoHistoricalRawReclaimError(
                "another signed Drive attestation already exists for this job and manifest"
            )
        attestation_id = receipt["body"]["attestation_id"]
        for existing_path in sorted(remote_root.glob("signed-job-*.json")):
            existing = _read_canonical_json(
                existing_path,
                field="existing signed Drive receipt",
            )
            existing_body = existing.get("body")
            if (
                isinstance(existing_body, Mapping)
                and existing_body.get("attestation_id") == attestation_id
            ):
                if existing_path != output or existing != receipt:
                    raise DojoHistoricalRawReclaimError(
                        "signed Drive attestation id was replayed"
                    )
                return existing
        publication_lock.assert_stable()
        _validate_attestation_public_key_seal(
            path=attestation_authority_seal_path,
            archive_root=archive_root,
            local=local,
            expected_drive_parent_id=expected_drive_parent_id,
        )
        publication_lock.assert_stable()
        if list(remote_root.glob(f"{prefix}-*.json")):
            raise DojoHistoricalRawReclaimError(
                "another signed Drive attestation appeared for this job and manifest"
            )
        _write_once(output, receipt, field="signed Drive attestation receipt")
        publication_lock.assert_stable()
        published = _validate_remote_receipt(
            remote_receipt_path=output,
            archive_root=archive_root,
            local=local,
            expected_drive_parent_id=expected_drive_parent_id,
            attestation_authority_seal_path=attestation_authority_seal_path,
            now_utc=now_utc,
        )
        if sorted(remote_root.glob(f"{prefix}-*.json")) != [output]:
            raise DojoHistoricalRawReclaimError(
                "signed Drive attestation lineage is no longer exact-one"
            )
        return published


def _raw_retirement_anchor_name(row: Mapping[str, Any]) -> str:
    path_sha = hashlib.sha256(str(row["path"]).encode("utf-8")).hexdigest()[:32]
    return f".retired-{path_sha}-{str(row['sha256'])[:32]}.anchor"


def _verify_raw_retirement_anchor(root: Path, row: Mapping[str, Any]) -> None:
    parent, leaf, _ = _open_anchored_parent(
        root,
        str(row["path"]),
        create_parents=False,
    )
    try:
        try:
            os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise DojoHistoricalRawReclaimError("retired raw path still exists")
        anchor = os.stat(
            _raw_retirement_anchor_name(row),
            dir_fd=parent,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(anchor.st_mode) or anchor.st_size != 0:
            raise DojoHistoricalRawReclaimError("raw retirement anchor is invalid")
    except FileNotFoundError as exc:
        raise DojoHistoricalRawReclaimError(
            "planned raw is missing without its zero-byte retirement anchor"
        ) from exc
    finally:
        os.close(parent)


def _archive_raw_retirement_anchor_after_restore(
    prepared: _PreparedRestore,
    row: Mapping[str, Any],
) -> None:
    source_parent, _, _ = _open_anchored_parent(
        prepared.run_root,
        str(row["path"]),
        create_parents=False,
    )
    run_name = _filename(prepared.run_root.name, "restore run directory name")
    audit_relative = (
        f".dojo-raw-retirement-audit/{run_name}/"
        f"{prepared.local_receipt['job_sha256']}-"
        f"{prepared.reclaim_plan['plan_sha256']}/.probe"
    )
    audit_parent, _, _ = _open_anchored_parent(
        prepared.run_root.parent,
        audit_relative,
        create_parents=True,
    )
    anchor_name = _raw_retirement_anchor_name(row)
    try:
        try:
            source = os.stat(
                anchor_name,
                dir_fd=source_parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            try:
                destination = os.stat(
                    anchor_name,
                    dir_fd=audit_parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if prepared.reclaim_plan["contract"] == RECLAIM_PLAN_CONTRACT:
                    return
                raise DojoHistoricalRawReclaimError(
                    "signed-V2 restore lacks its retirement audit anchor"
                ) from None
            if not stat.S_ISREG(destination.st_mode) or destination.st_size != 0:
                raise DojoHistoricalRawReclaimError(
                    "archived raw retirement anchor is invalid"
                )
            return
        if not stat.S_ISREG(source.st_mode) or source.st_size != 0:
            raise DojoHistoricalRawReclaimError(
                "raw retirement anchor changed before archival"
            )
        published = _atomic_rename_between_at_no_replace(
            source_parent,
            anchor_name,
            audit_parent,
            anchor_name,
        )
        if not published:
            raise DojoHistoricalRawReclaimError("raw retirement audit anchor collided")
        os.fsync(source_parent)
        os.fsync(audit_parent)
        destination = os.stat(
            anchor_name,
            dir_fd=audit_parent,
            follow_symlinks=False,
        )
        if _inode_identity(destination) != _inode_identity(source):
            raise DojoHistoricalRawReclaimError(
                "raw retirement anchor changed while archived"
            )
    finally:
        os.close(audit_parent)
        os.close(source_parent)


def _retire_target(root: Path, row: Mapping[str, Any]) -> int:
    return _remove_anchored_verified_file(
        root,
        str(row["path"]),
        expected_sha256=str(row["sha256"]),
        expected_size_bytes=int(row["size_bytes"]),
        retirement_anchor_name=_raw_retirement_anchor_name(row),
    )


def _reclaim_paths(root: Path, job_sha256: str) -> tuple[Path, str]:
    receipts = root / "reclaim-v2-receipts"
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
    zstd_bin: str,
    attestation_authority_seal_path: Path | None = None,
    global_heavy_lock_path: Path | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Read-only eligibility check; it never writes a receipt or removes data."""

    root = Path(run_root).resolve(strict=True)
    with _optional_global_heavy_lease(
        root=root,
        requested_path=global_heavy_lock_path,
    ) as global_lock_guard:
        if global_lock_guard is not None:
            global_lock_guard.assert_stable()
        prepared = _prepare_reclaim(
            run_root=root,
            archive_receipt_path=Path(archive_receipt_path),
            remote_receipt_path=Path(remote_receipt_path),
            expected_drive_parent_id=expected_drive_parent_id,
            attestation_authority_seal_path=attestation_authority_seal_path,
            permit_missing_targets=False,
            zstd_bin=zstd_bin,
            now_utc=now_utc,
        )
        if global_lock_guard is not None:
            global_lock_guard.assert_stable()
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
    is_v2 = plan.get("contract") == RECLAIM_PLAN_CONTRACT_V2
    _exact(
        plan,
        _RECLAIM_PLAN_KEYS if is_v2 else _RECLAIM_PLAN_V1_KEYS,
        "raw reclaim plan",
    )
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    targets = plan.get("targets")
    if (
        plan.get("contract")
        != (RECLAIM_PLAN_CONTRACT_V2 if is_v2 else RECLAIM_PLAN_CONTRACT)
        or isinstance(plan.get("schema_version"), bool)
        or plan.get("schema_version") != (2 if is_v2 else SCHEMA_VERSION)
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
        or plan.get("reclaim_mode") != (RECLAIM_MODE if is_v2 else LEGACY_RECLAIM_MODE)
        or not isinstance(targets, list)
        or not targets
        or _positive_integer(plan.get("target_count"), "plan target count")
        != len(targets)
        or _positive_integer(plan.get("target_bytes"), "plan target bytes")
        != plan.get("target_bytes")
        or plan.get("target_bytes")
        != sum(
            _positive_integer(row.get("size_bytes"), "plan target row size")
            for row in targets
            if isinstance(row, Mapping)
        )
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
    if len(targets) != sum(isinstance(row, Mapping) for row in targets):
        raise DojoHistoricalRawReclaimError("raw reclaim target row is invalid")
    target_paths = [row.get("path") for row in targets]
    if len(target_paths) != len(set(target_paths)):
        raise DojoHistoricalRawReclaimError("raw reclaim target path is duplicated")
    if is_v2:
        authority = _normalize_public_key(plan.get("attestation_public_key_hex"))
        zstd_path = plan.get("zstd_executable_path")
        if not isinstance(zstd_path, str) or not Path(zstd_path).is_absolute():
            raise DojoHistoricalRawReclaimError("raw reclaim plan zstd path is invalid")
        zstd_version = _bounded_text(
            plan.get("zstd_version"),
            "plan zstd version",
            maximum=4096,
        )
        issued = _utc(
            plan.get("attestation_issued_at_utc"),
            "plan attestation issued_at_utc",
        )
        expires = _utc(
            plan.get("attestation_expires_at_utc"),
            "plan attestation expires_at_utc",
        )
        enrolled = _utc(
            plan.get("attestation_authority_enrolled_at_utc"),
            "plan attestation authority enrolled_at_utc",
        )
        if (
            _sha(plan.get("attestation_id"), "plan attestation id")
            != plan.get("attestation_id")
            or plan.get("attestation_public_key_sha256")
            != authority["public_key_sha256"]
            or _sha(
                plan.get("attestation_authority_seal_sha256"),
                "plan attestation authority seal SHA-256",
            )
            != plan.get("attestation_authority_seal_sha256")
            or enrolled > issued
            or expires <= issued
            or expires - issued > timedelta(seconds=MAX_ATTESTATION_TTL_SECONDS)
            or _sha(
                plan.get("zstd_executable_sha256"),
                "plan zstd executable SHA-256",
            )
            != plan.get("zstd_executable_sha256")
            or _positive_integer(
                plan.get("zstd_executable_size_bytes"),
                "plan zstd executable size",
            )
            != plan.get("zstd_executable_size_bytes")
            or not zstd_version
        ):
            raise DojoHistoricalRawReclaimError(
                "raw reclaim plan attestation authority is invalid"
            )
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
    zstd_bin: str,
    attestation_authority_seal_path: Path | None = None,
    confirmed_plan_sha256: str | None = None,
    confirmed_target_count: int | None = None,
    confirmed_target_bytes: int | None = None,
    global_heavy_lock_path: Path | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Remove only raw transcripts/source after complete local+remote proof."""

    root = Path(run_root).resolve(strict=True)
    run_lock = root / ".historical-train.lock"
    with ExitStack() as lock_stack:
        run_lock_guard = lock_stack.enter_context(
            _exclusive_lock(
                run_lock,
                field="historical train run lock",
            )
        )
        global_lock_guard = lock_stack.enter_context(
            _optional_global_heavy_lease(
                root=root,
                requested_path=global_heavy_lock_path,
            )
        )
        preliminary_archive = _read_canonical_json(
            Path(archive_receipt_path), field="local archive receipt"
        )
        job_sha256 = _sha(
            preliminary_archive.get("job_sha256"), "local receipt job SHA-256"
        )
        receipts, prefix = _reclaim_paths(root, job_sha256)
        with _exclusive_lock(
            receipts / f".{prefix}.lock", field="raw reclaim job lock"
        ) as job_lock_guard:

            def assert_reclaim_locks() -> None:
                run_lock_guard.assert_stable()
                if global_lock_guard is not None:
                    global_lock_guard.assert_stable()
                job_lock_guard.assert_stable()

            assert_reclaim_locks()
            existing_receipts = sorted(receipts.glob(f"reclaim-{job_sha256}-*.json"))
            existing_plans = sorted(receipts.glob(f"plan-{prefix}-*.json"))
            if len(existing_plans) > 1 or len(existing_receipts) > 1:
                raise DojoHistoricalRawReclaimError(
                    "multiple raw reclaim lineage artifacts name this job"
                )
            if existing_receipts and not existing_plans:
                raise DojoHistoricalRawReclaimError(
                    "raw reclaim receipt exists without its append-only plan"
                )
            resume_plan = (
                _validate_plan_seal(
                    existing_plans[0],
                    expected_job_sha256=job_sha256,
                )
                if existing_plans
                else None
            )
            if resume_plan is not None and (
                resume_plan["contract"] != RECLAIM_PLAN_CONTRACT_V2
                or resume_plan["remote_receipt_sha256"]
                not in Path(remote_receipt_path).name
                or confirmed_plan_sha256 != resume_plan["plan_sha256"]
                or confirmed_target_count != resume_plan["target_count"]
                or confirmed_target_bytes != resume_plan["target_bytes"]
            ):
                raise DojoHistoricalRawReclaimError(
                    "existing plan cannot authorize an expired reclaim resume"
                )
            permit_missing = resume_plan is not None
            prepared = _prepare_reclaim(
                run_root=root,
                archive_receipt_path=Path(archive_receipt_path),
                remote_receipt_path=Path(remote_receipt_path),
                expected_drive_parent_id=expected_drive_parent_id,
                attestation_authority_seal_path=attestation_authority_seal_path,
                permit_missing_targets=permit_missing,
                zstd_bin=zstd_bin,
                now_utc=now_utc,
            )
            assert_reclaim_locks()
            plan_body = _plan_body(prepared)
            plan = {**plan_body, "plan_sha256": _canonical_sha256(plan_body)}
            if (
                confirmed_plan_sha256 != plan["plan_sha256"]
                or confirmed_target_count != plan["target_count"]
                or confirmed_target_bytes != plan["target_bytes"]
            ):
                raise DojoHistoricalRawReclaimError(
                    "exact plan SHA/count/bytes confirmation does not match"
                )
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
                assert_reclaim_locks()
                _write_once(plan_path, plan, field="raw reclaim plan")
                assert_reclaim_locks()

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
                assert_reclaim_locks()
                reclaimed_allocated += _retire_target(root, row)
                assert_reclaim_locks()
            missing, _ = _verify_rows(root, prepared.targets, missing_allowed=True)
            if missing != {row["path"] for row in prepared.targets}:
                raise DojoHistoricalRawReclaimError(
                    "not every planned raw target was reclaimed"
                )
            for row in prepared.targets:
                _verify_raw_retirement_anchor(root, row)
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
            assert_reclaim_locks()
            _write_once(receipt_path, receipt, field="raw reclaim receipt")
            assert_reclaim_locks()
            return receipt


def _validate_reclaim_receipt_seal(
    path: Path,
    *,
    plan: Mapping[str, Any],
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
        or receipt.get("manifest_sha256") != plan["manifest_sha256"]
        or receipt.get("local_archive_receipt_sha256")
        != plan["local_archive_receipt_sha256"]
        or receipt.get("remote_receipt_sha256") != plan["remote_receipt_sha256"]
        or receipt.get("reclaim_plan_sha256") != plan["plan_sha256"]
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
        raise DojoHistoricalRawReclaimError("raw reclaim receipt seal is invalid")
    _utc(receipt["completed_at_utc"], "raw reclaim completed_at_utc")
    expected_name = (
        f"reclaim-{plan['job_sha256']}-{receipt['reclaim_receipt_sha256']}.json"
    )
    if path.name != expected_name:
        raise DojoHistoricalRawReclaimError(
            "raw reclaim receipt filename is not content-addressed"
        )
    return receipt


def _validate_restore_attestation_authority(
    *,
    archive_root: Path,
    local: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    """Bind a signed-V2 restore to the exact-one enrolled job authority."""

    if plan["contract"] != RECLAIM_PLAN_CONTRACT_V2:
        return
    paths = _attestation_authority_namespace_paths(
        archive_root=archive_root,
        job_sha256=str(local["job_sha256"]),
        require_exact_one=True,
    )
    expected = (
        archive_root
        / "remote-authorities"
        / (
            f"key-job-{local['job_sha256']}-{local['manifest_sha256']}-"
            f"{plan['attestation_authority_seal_sha256']}.json"
        )
    )
    if paths != [expected]:
        raise DojoHistoricalRawReclaimError(
            "restore attestation authority differs from its reclaim plan"
        )
    raw_seal = _read_canonical_json(
        expected,
        field="restore attestation public-key seal",
    )
    parent_id = _drive_id(
        raw_seal.get("expected_drive_parent_id"),
        "restore attestation authority Drive parent id",
    )
    seal = _validate_attestation_public_key_seal(
        path=expected,
        archive_root=archive_root,
        local=local,
        expected_drive_parent_id=parent_id,
    )
    if (
        seal["authority_seal_sha256"] != plan["attestation_authority_seal_sha256"]
        or seal["public_key_sha256"] != plan["attestation_public_key_sha256"]
        or seal["public_key_hex"] != plan["attestation_public_key_hex"]
        or seal["enrolled_at_utc"] != plan["attestation_authority_enrolled_at_utc"]
    ):
        raise DojoHistoricalRawReclaimError(
            "restore attestation authority differs from its reclaim plan"
        )


def _prepare_restore(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    reclaim_plan_path: Path,
    reclaim_receipt_path: Path,
    zstd_bin: str,
) -> _PreparedRestore:
    zstd_seal = _seal_zstd_executable(zstd_bin)
    validated_zstd = str(zstd_seal["zstd_executable_path"])
    try:
        root = Path(run_root).resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError("restore run root is unavailable") from exc
    if not root.is_dir():
        raise DojoHistoricalRawReclaimError("restore run root is not a directory")
    archive_receipt = _strict_regular_path(
        archive_receipt_path,
        field="restore local archive receipt",
    )
    plan_path = _strict_regular_path(reclaim_plan_path, field="restore reclaim plan")
    receipt_path = _strict_regular_path(
        reclaim_receipt_path,
        field="restore reclaim receipt",
    )
    allowed_reclaim_roots = {
        (root / "reclaim-receipts").resolve(strict=False),
        (root / "reclaim-v2-receipts").resolve(strict=False),
    }
    if (
        plan_path.parent not in allowed_reclaim_roots
        or receipt_path.parent != plan_path.parent
    ):
        raise DojoHistoricalRawReclaimError(
            "restore reclaim lineage is outside the run reclaim-receipts directory"
        )
    archive_root, local, manifest = _validate_local_archive(
        run_root=root,
        archive_receipt_path=archive_receipt,
        require_full_inventory=False,
        zstd_bin=validated_zstd,
    )
    job_sha256 = local["job_sha256"]
    targets, retained = _reclaimable_rows(
        run_root=root,
        job_sha256=job_sha256,
        manifest=manifest,
    )
    plan = _validate_plan_seal(plan_path, expected_job_sha256=job_sha256)
    if plan["contract"] == RECLAIM_PLAN_CONTRACT_V2 and any(
        plan.get(key) != value for key, value in zstd_seal.items()
    ):
        raise DojoHistoricalRawReclaimError(
            "restore zstd executable differs from the sealed reclaim plan"
        )
    expected_reclaim_root = (
        root
        / (
            "reclaim-v2-receipts"
            if plan["contract"] == RECLAIM_PLAN_CONTRACT_V2
            else "reclaim-receipts"
        )
    ).resolve(strict=True)
    if plan_path.parent != expected_reclaim_root:
        raise DojoHistoricalRawReclaimError(
            "restore reclaim lineage uses the wrong generation directory"
        )
    expected_plan_path = expected_reclaim_root / (
        f"plan-job-{job_sha256}-{plan['remote_receipt_sha256']}.json"
    )
    if plan_path != expected_plan_path:
        raise DojoHistoricalRawReclaimError(
            "restore reclaim plan filename does not match its lineage"
        )
    if (
        plan["completion_sha256"] != local["completion_sha256"]
        or plan["manifest_sha256"] != local["manifest_sha256"]
        or plan["local_archive_receipt_sha256"] != local["receipt_sha256"]
        or plan["object_set_sha256"]
        != local["remote_readback_objects"]["object_set_sha256"]
        or plan["archive_sha256"] != local["archive_sha256"]
        or plan["archive_size_bytes"] != local["archive_size_bytes"]
        or plan["targets"] != list(targets)
        or plan["target_count"] != len(targets)
        or plan["target_bytes"] != sum(row["size_bytes"] for row in targets)
        or plan["retained_file_count"] != len(retained)
        or plan["retained_bytes"] != sum(row["size_bytes"] for row in retained)
    ):
        raise DojoHistoricalRawReclaimError(
            "restore reclaim plan differs from the deep local archive manifest"
        )
    _validate_restore_attestation_authority(
        archive_root=archive_root,
        local=local,
        plan=plan,
    )
    reclaim = _validate_reclaim_receipt_seal(receipt_path, plan=plan)
    _verify_rows(root, targets, missing_allowed=True)
    _verify_rows(root, retained, missing_allowed=False)
    archive_path = _strict_regular_path(
        local["archive_path"],
        field="restore archive path",
    )
    if archive_path.parent != (archive_root / "archives").resolve(strict=True):
        raise DojoHistoricalRawReclaimError(
            "restore archive is outside its append-only archive directory"
        )
    return _PreparedRestore(
        run_root=root,
        archive_root=archive_root,
        archive_path=archive_path,
        archive_receipt_path=archive_receipt,
        reclaim_plan_path=plan_path,
        reclaim_receipt_path=receipt_path,
        local_receipt=local,
        manifest=manifest,
        reclaim_plan=plan,
        reclaim_receipt=reclaim,
        targets=targets,
        retained=retained,
    )


def _restore_staging_root(prepared: _PreparedRestore) -> Path:
    run_name = _filename(prepared.run_root.name, "restore run directory name")
    relative = (
        f".dojo-raw-restore-staging/{run_name}/"
        f"{prepared.local_receipt['job_sha256']}-{prepared.reclaim_plan['plan_sha256']}/.probe"
    )
    descriptor, _, _ = _open_anchored_parent(
        prepared.run_root.parent,
        relative,
        create_parents=True,
    )
    try:
        state = os.fstat(descriptor)
        if not stat.S_ISDIR(state.st_mode):
            raise DojoHistoricalRawReclaimError(
                "restore staging root is not a directory"
            )
    finally:
        os.close(descriptor)
    return (
        prepared.run_root.parent
        / ".dojo-raw-restore-staging"
        / run_name
        / f"{prepared.local_receipt['job_sha256']}-{prepared.reclaim_plan['plan_sha256']}"
    )


def _stage_name(index: int, row: Mapping[str, Any]) -> str:
    return f"target-{index:05d}-{row['sha256']}.raw"


def _remove_anchored_verified_file(
    root: Path,
    name: str,
    *,
    expected_identity: tuple[int, int, int] | None = None,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    retirement_anchor_name: str | None = None,
) -> int:
    """Retire one verified inode, truncate it by fd, and retain its zero anchor."""

    parent, leaf, _ = _open_anchored_parent(
        root,
        name,
        create_parents=False,
    )
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                leaf,
                os.O_RDWR
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent,
            )
        except FileNotFoundError:
            if retirement_anchor_name is not None:
                try:
                    anchored = os.stat(
                        retirement_anchor_name,
                        dir_fd=parent,
                        follow_symlinks=False,
                    )
                except FileNotFoundError as exc:
                    raise DojoHistoricalRawReclaimError(
                        "planned raw is missing without its retirement anchor"
                    ) from exc
                if not stat.S_ISREG(anchored.st_mode) or anchored.st_size != 0:
                    raise DojoHistoricalRawReclaimError(
                        "existing raw retirement anchor is invalid"
                    )
            return 0
        except OSError as exc:
            raise DojoHistoricalRawReclaimError(
                "verified cleanup target cannot be opened"
            ) from exc
        opened = os.fstat(descriptor)
        allocated = opened.st_blocks * 512
        if not stat.S_ISREG(opened.st_mode):
            raise DojoHistoricalRawReclaimError(
                "verified cleanup target is not regular"
            )
        if (
            expected_identity is not None
            and _inode_identity(opened) != expected_identity
        ):
            raise DojoHistoricalRawReclaimError("verified cleanup target inode changed")
        if expected_sha256 is not None or expected_size_bytes is not None:
            if expected_sha256 is None or expected_size_bytes is None:
                raise DojoHistoricalRawReclaimError(
                    "verified cleanup byte contract is incomplete"
                )
            sha256, size, _ = _hash_open_regular(
                descriptor,
                field="verified cleanup target",
            )
            if sha256 != expected_sha256 or size != expected_size_bytes:
                raise DojoHistoricalRawReclaimError(
                    "verified cleanup target bytes changed"
                )
        observed = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if _identity(opened) != _identity(observed):
            raise DojoHistoricalRawReclaimError(
                "verified cleanup dirent changed before retirement"
            )

        anchor_name: str | None = None
        attempts = 1 if retirement_anchor_name is not None else 4
        for _ in range(attempts):
            candidate = retirement_anchor_name or _retirement_anchor_name(
                Path(leaf), device=opened.st_dev, inode=opened.st_ino
            )
            try:
                published = _atomic_rename_at_no_replace(
                    parent,
                    leaf,
                    candidate,
                )
            except FileNotFoundError:
                return 0
            except DojoHistoricalJobArchiveError as exc:
                raise DojoHistoricalRawReclaimError(str(exc)) from exc
            if published:
                anchor_name = candidate
                break
        if anchor_name is None:
            raise DojoHistoricalRawReclaimError(
                "verified cleanup retirement anchor collided"
            )
        os.fsync(parent)
        anchored = os.stat(
            anchor_name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(anchored.st_mode) or _inode_identity(
            anchored
        ) != _inode_identity(opened):
            # A last-moment replacement was retired. Its payload remains
            # untouched under the durable anchor; only the opened inode can
            # ever be truncated below.
            os.fsync(parent)
            raise DojoHistoricalRawReclaimError(
                "verified cleanup replacement was retired and preserved"
            )
        before_release = os.fstat(descriptor)
        if before_release.st_nlink != 1:
            raise DojoHistoricalRawReclaimError(
                "verified cleanup target has an unexpected hard link"
            )
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
        released = os.fstat(descriptor)
        anchor_after = os.stat(
            anchor_name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (
            _inode_identity(released) != _inode_identity(anchor_after)
            or released.st_size != 0
            or anchor_after.st_size != 0
        ):
            raise DojoHistoricalRawReclaimError(
                "verified cleanup retirement anchor did not truncate"
            )
        os.fsync(parent)
        return allocated
    except DojoHistoricalRawReclaimError:
        raise
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            "verified cleanup retirement failed"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _open_stage_output(
    *,
    stage_root: Path,
    name: str,
    row: Mapping[str, Any],
) -> tuple[int | None, tuple[int, int, int] | None]:
    """Reuse one exact stage file, or replace only invalid private scratch bytes."""

    parent, leaf, _ = _open_anchored_parent(
        stage_root,
        name,
        create_parents=False,
    )
    try:
        try:
            current = os.open(
                leaf,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent,
            )
        except FileNotFoundError:
            current = None
        except OSError as exc:
            raise DojoHistoricalRawReclaimError(
                "restore staging file cannot be opened safely"
            ) from exc
        remove_identity: tuple[int, int, int] | None = None
        if current is not None:
            try:
                opened = os.fstat(current)
                sha256, size, _ = _hash_open_regular(
                    current,
                    field="restore staging file",
                )
                observed = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                if _identity(opened) != _identity(observed):
                    raise DojoHistoricalRawReclaimError(
                        "restore staging dirent changed while verified"
                    )
                if sha256 == row["sha256"] and size == row["size_bytes"]:
                    return None, None
                remove_identity = _inode_identity(opened)
            finally:
                os.close(current)
        if remove_identity is not None:
            _remove_anchored_verified_file(
                stage_root,
                name,
                expected_identity=remove_identity,
            )
        try:
            descriptor = os.open(
                leaf,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent,
            )
        except OSError as exc:
            raise DojoHistoricalRawReclaimError(
                "restore staging file cannot be created atomically"
            ) from exc
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            os.close(descriptor)
            raise DojoHistoricalRawReclaimError(
                "restore staging output is not a regular file"
            )
        return descriptor, _inode_identity(opened)
    finally:
        os.close(parent)


def _stream_member(
    *,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    row: Mapping[str, Any],
    output_descriptor: int | None,
) -> None:
    handle = archive.extractfile(member)
    if handle is None:
        raise DojoHistoricalRawReclaimError("restore archive member is unreadable")
    digest = hashlib.sha256()
    size = 0
    with handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
            if output_descriptor is not None:
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(output_descriptor, remaining)
                    if written <= 0:
                        raise DojoHistoricalRawReclaimError(
                            "restore staging write made no progress"
                        )
                    remaining = remaining[written:]
    if size != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
        raise DojoHistoricalRawReclaimError(
            f"restore archive payload differs from manifest: {row['path']}"
        )


def _extract_restore_targets(
    prepared: _PreparedRestore,
    *,
    zstd_bin: str,
) -> tuple[tuple[dict[str, Any], str], ...]:
    """Stream exact manifest members into private content-addressed staging."""

    executable = _validated_restore_zstd(zstd_bin)
    stage_root = _restore_staging_root(prepared)
    target_indices = {
        row["path"]: (index, row) for index, row in enumerate(prepared.targets)
    }
    staged: list[tuple[dict[str, Any], str]] = []
    try:
        archive_before = prepared.archive_path.stat(follow_symlinks=False)
        archive_descriptor = os.open(
            prepared.archive_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        archive_opened = os.fstat(archive_descriptor)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            "restore archive cannot be opened safely"
        ) from exc
    if not stat.S_ISREG(archive_opened.st_mode) or _identity(
        archive_before
    ) != _identity(archive_opened):
        os.close(archive_descriptor)
        raise DojoHistoricalRawReclaimError("restore archive changed while opened")

    tar_error: Exception | None = None
    process: subprocess.Popen[bytes] | None = None
    with tempfile.TemporaryFile() as process_error:
        try:
            process = subprocess.Popen(
                [executable, "-q", "-d", "-c"],
                stdin=archive_descriptor,
                stdout=subprocess.PIPE,
                stderr=process_error,
                close_fds=True,
            )
            if process.stdout is None:
                raise DojoHistoricalRawReclaimError(
                    "zstd restore output is unavailable"
                )
            with tarfile.open(
                fileobj=process.stdout,
                mode="r|",
                encoding="utf-8",
                errors="strict",
            ) as archive:
                iterator = iter(archive)
                first = next(iterator, None)
                manifest_payload = _canonical_bytes(prepared.manifest)
                if (
                    first is None
                    or not first.isfile()
                    or first.name != "MANIFEST.json"
                    or first.size != len(manifest_payload)
                ):
                    raise DojoHistoricalRawReclaimError(
                        "restore archive manifest member is invalid"
                    )
                manifest_handle = archive.extractfile(first)
                if manifest_handle is None:
                    raise DojoHistoricalRawReclaimError(
                        "restore archive manifest is unreadable"
                    )
                with manifest_handle:
                    observed_manifest = manifest_handle.read(len(manifest_payload) + 1)
                if observed_manifest != manifest_payload:
                    raise DojoHistoricalRawReclaimError(
                        "restore archive embeds another manifest"
                    )

                for manifest_row in prepared.manifest["files"]:
                    row = dict(manifest_row)
                    member = next(iterator, None)
                    if (
                        member is None
                        or not member.isfile()
                        or member.name != f"payload/{row['path']}"
                        or member.size != row["size_bytes"]
                    ):
                        raise DojoHistoricalRawReclaimError(
                            "restore archive member header differs from manifest"
                        )
                    target = target_indices.get(row["path"])
                    if target is None:
                        _stream_member(
                            archive=archive,
                            member=member,
                            row=row,
                            output_descriptor=None,
                        )
                        continue
                    target_index, target_row = target
                    name = _stage_name(target_index, target_row)
                    descriptor, created_identity = _open_stage_output(
                        stage_root=stage_root,
                        name=name,
                        row=target_row,
                    )
                    try:
                        _stream_member(
                            archive=archive,
                            member=member,
                            row=target_row,
                            output_descriptor=descriptor,
                        )
                        if descriptor is not None:
                            os.fsync(descriptor)
                            written = os.fstat(descriptor)
                            if (
                                created_identity != _inode_identity(written)
                                or written.st_size != target_row["size_bytes"]
                            ):
                                raise DojoHistoricalRawReclaimError(
                                    "restore staging output changed while written"
                                )
                    except Exception:
                        if descriptor is not None:
                            os.close(descriptor)
                            descriptor = None
                            _remove_anchored_verified_file(
                                stage_root,
                                name,
                                expected_identity=created_identity,
                            )
                        raise
                    finally:
                        if descriptor is not None:
                            os.close(descriptor)
                    observed = _hash_anchored_file(
                        stage_root,
                        name,
                        missing_allowed=False,
                    )
                    assert observed is not None
                    if (
                        observed[0] != target_row["sha256"]
                        or observed[1] != target_row["size_bytes"]
                    ):
                        raise DojoHistoricalRawReclaimError(
                            "restore staging output failed final verification"
                        )
                    staged.append((target_row, name))
                if next(iterator, None) is not None:
                    raise DojoHistoricalRawReclaimError(
                        "restore archive contains an unmanifested member"
                    )
        except Exception as exc:
            tar_error = exc
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
        return_code = process.wait() if process is not None else -1
        process_error.seek(0)
        diagnostic = process_error.read(4096).decode("utf-8", "replace").strip()

    try:
        archive_opened_after = os.fstat(archive_descriptor)
        archive_after = prepared.archive_path.stat(follow_symlinks=False)
    except OSError as exc:
        os.close(archive_descriptor)
        raise DojoHistoricalRawReclaimError(
            "restore archive disappeared while extracted"
        ) from exc
    os.close(archive_descriptor)
    if _identity(archive_before) != _identity(archive_opened_after) or _identity(
        archive_opened_after
    ) != _identity(archive_after):
        raise DojoHistoricalRawReclaimError("restore archive changed while extracted")
    if tar_error is not None:
        if isinstance(tar_error, DojoHistoricalRawReclaimError):
            message = str(tar_error)
        else:
            message = f"restore archive tar stream is invalid: {tar_error}"
        raise DojoHistoricalRawReclaimError(
            f"{message}; zstd return code {return_code}; "
            f"zstd stderr: {diagnostic or '<empty>'}"
        ) from tar_error
    if return_code != 0:
        raise DojoHistoricalRawReclaimError(
            f"zstd restore failed with return code {return_code}: "
            f"{diagnostic or '<empty>'}"
        )
    archive_sha256, archive_size = _hash_file(prepared.archive_path)
    if (
        archive_sha256 != prepared.local_receipt["archive_sha256"]
        or archive_size != prepared.local_receipt["archive_size_bytes"]
        or len(staged) != len(prepared.targets)
    ):
        raise DojoHistoricalRawReclaimError(
            "restore archive or staged target denominator changed"
        )
    return tuple(staged)


def _publish_restore_target(
    *,
    run_root: Path,
    row: Mapping[str, Any],
    stage_root: Path,
    stage_name: str,
) -> bool:
    """Atomically publish a staged inode without replacing a destination."""

    stage_parent, stage_leaf, _ = _open_anchored_parent(
        stage_root,
        stage_name,
        create_parents=False,
    )
    destination_parent, destination_leaf, safe = _open_anchored_parent(
        run_root,
        str(row["path"]),
        create_parents=True,
    )
    stage_descriptor: int | None = None
    try:
        stage_descriptor = os.open(
            stage_leaf,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=stage_parent,
        )
        stage_opened = os.fstat(stage_descriptor)
        stage_sha256, stage_size, _ = _hash_open_regular(
            stage_descriptor,
            field=f"restore stage {stage_leaf}",
        )
        stage_observed = os.stat(
            stage_leaf,
            dir_fd=stage_parent,
            follow_symlinks=False,
        )
        if (
            _identity(stage_opened) != _identity(stage_observed)
            or stage_sha256 != row["sha256"]
            or stage_size != row["size_bytes"]
        ):
            raise DojoHistoricalRawReclaimError(
                f"restore stage changed before publish: {safe}"
            )

        if _verify_restore_destination(
            parent=destination_parent,
            leaf=destination_leaf,
            row=row,
            safe=safe,
        ):
            return False

        try:
            published = _atomic_rename_between_at_no_replace(
                stage_parent,
                stage_leaf,
                destination_parent,
                destination_leaf,
            )
        except FileNotFoundError as exc:
            raise DojoHistoricalRawReclaimError(
                f"restore stage disappeared before publish: {safe}"
            ) from exc
        if not published:
            if not _verify_restore_destination(
                parent=destination_parent,
                leaf=destination_leaf,
                row=row,
                safe=safe,
            ):
                raise DojoHistoricalRawReclaimError(
                    f"restore destination raced with publish: {safe}"
                ) from None
            return False
        destination_observed = os.stat(
            destination_leaf,
            dir_fd=destination_parent,
            follow_symlinks=False,
        )
        stage_after = os.fstat(stage_descriptor)
        if _inode_identity(stage_opened) != _inode_identity(
            stage_after
        ) or _inode_identity(stage_after) != _inode_identity(destination_observed):
            raise DojoHistoricalRawReclaimError(
                f"restore destination identity mismatch: {safe}"
            )
        try:
            os.stat(stage_leaf, dir_fd=stage_parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise DojoHistoricalRawReclaimError(
                f"restore stage name reappeared after publish: {safe}"
            )
        os.fsync(stage_parent)
        os.fsync(destination_parent)
        return True
    except DojoHistoricalRawReclaimError:
        raise
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            f"restore target verification failed: {safe}"
        ) from exc
    finally:
        if stage_descriptor is not None:
            os.close(stage_descriptor)
        os.close(stage_parent)
        os.close(destination_parent)


def _verify_restore_destination(
    *,
    parent: int,
    leaf: str,
    row: Mapping[str, Any],
    safe: str,
) -> bool:
    """Return whether an exact existing destination is present; never follow it."""

    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            f"restore destination cannot be opened safely: {safe}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        sha256, size, _ = _hash_open_regular(
            descriptor,
            field=f"restore destination {safe}",
        )
        observed = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if (
            _identity(opened) != _identity(observed)
            or sha256 != row["sha256"]
            or size != row["size_bytes"]
        ):
            raise DojoHistoricalRawReclaimError(
                f"restore destination already exists with other bytes: {safe}"
            )
        return True
    finally:
        os.close(descriptor)


def _restore_receipt_root(
    run_root: Path,
    *,
    plan: Mapping[str, Any],
) -> Path:
    directory_name = (
        "restore-v2-receipts"
        if plan["contract"] == RECLAIM_PLAN_CONTRACT_V2
        else "restore-receipts"
    )
    descriptor, _, _ = _open_anchored_parent(
        run_root,
        f"{directory_name}/.probe",
        create_parents=True,
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise DojoHistoricalRawReclaimError(
                "restore receipt directory is not a real directory"
            )
    finally:
        os.close(descriptor)
    return run_root / directory_name


def _restore_receipt_body(
    prepared: _PreparedRestore,
    *,
    published_count: int,
    preexisting_count: int,
) -> dict[str, Any]:
    return {
        "contract": RESTORE_RECEIPT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "RAW_RESTORED",
        "job_sha256": prepared.local_receipt["job_sha256"],
        "completion_sha256": prepared.local_receipt["completion_sha256"],
        "bundle_kind": SUCCESS_BUNDLE_KIND,
        "manifest_sha256": prepared.local_receipt["manifest_sha256"],
        "local_archive_receipt_sha256": prepared.local_receipt["receipt_sha256"],
        "archive_sha256": prepared.local_receipt["archive_sha256"],
        "reclaim_plan_sha256": prepared.reclaim_plan["plan_sha256"],
        "reclaim_receipt_sha256": prepared.reclaim_receipt["reclaim_receipt_sha256"],
        "restored_at_utc": datetime.now(timezone.utc).isoformat(),
        "restored_file_count": len(prepared.targets),
        "restored_files": list(prepared.targets),
        "restored_logical_bytes": sum(row["size_bytes"] for row in prepared.targets),
        "published_file_count": published_count,
        "preexisting_matching_file_count": preexisting_count,
        "local_archive_deep_verified": True,
        "retained_bytes_verified": True,
        "all_raw_targets_present": True,
        "remote_receipt_trusted": False,
        **_AUTHORITY,
    }


def _validate_restore_receipt(
    path: Path,
    *,
    prepared: _PreparedRestore,
) -> dict[str, Any]:
    receipt = _read_canonical_json(path, field="raw restore receipt")
    _exact(receipt, _RESTORE_RECEIPT_KEYS, "raw restore receipt")
    body = {
        key: value for key, value in receipt.items() if key != "restore_receipt_sha256"
    }
    target_count = len(prepared.targets)
    if (
        receipt.get("contract") != RESTORE_RECEIPT_CONTRACT
        or isinstance(receipt.get("schema_version"), bool)
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "RAW_RESTORED"
        or receipt.get("job_sha256") != prepared.local_receipt["job_sha256"]
        or receipt.get("completion_sha256")
        != prepared.local_receipt["completion_sha256"]
        or receipt.get("bundle_kind") != SUCCESS_BUNDLE_KIND
        or receipt.get("manifest_sha256") != prepared.local_receipt["manifest_sha256"]
        or receipt.get("local_archive_receipt_sha256")
        != prepared.local_receipt["receipt_sha256"]
        or receipt.get("archive_sha256") != prepared.local_receipt["archive_sha256"]
        or receipt.get("reclaim_plan_sha256") != prepared.reclaim_plan["plan_sha256"]
        or receipt.get("reclaim_receipt_sha256")
        != prepared.reclaim_receipt["reclaim_receipt_sha256"]
        or receipt.get("restored_file_count") != target_count
        or isinstance(receipt.get("restored_file_count"), bool)
        or receipt.get("restored_files") != list(prepared.targets)
        or receipt.get("restored_logical_bytes")
        != sum(row["size_bytes"] for row in prepared.targets)
        or isinstance(receipt.get("restored_logical_bytes"), bool)
        or _nonnegative_integer(
            receipt.get("published_file_count"), "restore published file count"
        )
        != receipt.get("published_file_count")
        or _nonnegative_integer(
            receipt.get("preexisting_matching_file_count"),
            "restore preexisting file count",
        )
        != receipt.get("preexisting_matching_file_count")
        or receipt.get("published_file_count")
        + receipt.get("preexisting_matching_file_count")
        != target_count
        or receipt.get("local_archive_deep_verified") is not True
        or receipt.get("retained_bytes_verified") is not True
        or receipt.get("all_raw_targets_present") is not True
        or receipt.get("remote_receipt_trusted") is not False
        or receipt.get("historical_train_is_proof") is not False
        or receipt.get("promotion_eligible") is not False
        or receipt.get("live_permission") is not False
        or receipt.get("order_authority") != "NONE"
        or receipt.get("broker_mutation_allowed") is not False
        or receipt.get("restore_receipt_sha256") != _canonical_sha256(body)
    ):
        raise DojoHistoricalRawReclaimError("raw restore receipt is invalid")
    _utc(receipt["restored_at_utc"], "raw restore restored_at_utc")
    expected_name = (
        f"restore-{receipt['job_sha256']}-{receipt['restore_receipt_sha256']}.json"
    )
    if path.name != expected_name:
        raise DojoHistoricalRawReclaimError(
            "raw restore receipt filename is not content-addressed"
        )
    missing, _ = _verify_rows(
        prepared.run_root,
        prepared.targets,
        missing_allowed=True,
    )
    if missing:
        raise DojoHistoricalRawReclaimError(
            "raw restore receipt exists while a target is missing"
        )
    _verify_rows(prepared.run_root, prepared.retained, missing_allowed=False)
    return receipt


def _cleanup_restore_staging(prepared: _PreparedRestore) -> None:
    stage_root = _restore_staging_root(prepared)
    for index, row in enumerate(prepared.targets):
        name = _stage_name(index, row)
        observed = _hash_anchored_file(
            stage_root,
            name,
            missing_allowed=True,
        )
        if observed is None:
            continue
        if observed[0] != row["sha256"] or observed[1] != row["size_bytes"]:
            raise DojoHistoricalRawReclaimError(
                "restore staging drifted before cleanup"
            )
        _remove_anchored_verified_file(
            stage_root,
            name,
            expected_sha256=row["sha256"],
            expected_size_bytes=row["size_bytes"],
        )


def restore_historical_job_raw(
    *,
    run_root: Path,
    archive_receipt_path: Path,
    reclaim_plan_path: Path,
    reclaim_receipt_path: Path,
    zstd_bin: str,
) -> dict[str, Any]:
    """Restore exact historical raw targets from one deep-verified local archive."""

    validated_zstd = _validated_restore_zstd(zstd_bin)
    try:
        root = Path(run_root).resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError("restore run root is unavailable") from exc
    with _exclusive_lock(
        root / ".historical-train.lock",
        field="historical train run lock",
    ) as run_lock_guard:
        prepared = _prepare_restore(
            run_root=root,
            archive_receipt_path=Path(archive_receipt_path),
            reclaim_plan_path=Path(reclaim_plan_path),
            reclaim_receipt_path=Path(reclaim_receipt_path),
            zstd_bin=validated_zstd,
        )
        receipt_root = _restore_receipt_root(
            root,
            plan=prepared.reclaim_plan,
        )
        job_sha256 = prepared.local_receipt["job_sha256"]
        with _exclusive_lock(
            receipt_root / f".job-{job_sha256}.lock",
            field="raw restore job lock",
        ) as job_lock_guard:

            def assert_restore_locks() -> None:
                run_lock_guard.assert_stable()
                job_lock_guard.assert_stable()

            assert_restore_locks()
            existing = sorted(receipt_root.glob(f"restore-{job_sha256}-*.json"))
            if len(existing) > 1:
                raise DojoHistoricalRawReclaimError(
                    "multiple raw restore receipts name one job"
                )
            if existing:
                receipt = _validate_restore_receipt(
                    existing[0],
                    prepared=prepared,
                )
                assert_restore_locks()
                _cleanup_restore_staging(prepared)
                assert_restore_locks()
                return receipt

            assert_restore_locks()
            staged = _extract_restore_targets(prepared, zstd_bin=validated_zstd)
            assert_restore_locks()
            stage_root = _restore_staging_root(prepared)
            published_count = 0
            for row, stage_name in staged:
                assert_restore_locks()
                if _publish_restore_target(
                    run_root=root,
                    row=row,
                    stage_root=stage_root,
                    stage_name=stage_name,
                ):
                    published_count += 1
                assert_restore_locks()
            missing, _ = _verify_rows(root, prepared.targets, missing_allowed=True)
            if missing:
                raise DojoHistoricalRawReclaimError(
                    "restore did not publish every planned target"
                )
            _verify_rows(root, prepared.retained, missing_allowed=False)
            for row in prepared.targets:
                assert_restore_locks()
                _archive_raw_retirement_anchor_after_restore(prepared, row)
                assert_restore_locks()
            try:
                deep = verify_existing_historical_job_archive(
                    run_root=root,
                    job_sha256=job_sha256,
                    archive_root=prepared.archive_root,
                    zstd_bin=validated_zstd,
                )
            except DojoHistoricalJobArchiveError as exc:
                raise DojoHistoricalRawReclaimError(str(exc)) from exc
            if deep != prepared.local_receipt:
                raise DojoHistoricalRawReclaimError(
                    "deep archive verification returned another receipt"
                )
            body = _restore_receipt_body(
                prepared,
                published_count=published_count,
                preexisting_count=len(prepared.targets) - published_count,
            )
            receipt = {
                **body,
                "restore_receipt_sha256": _canonical_sha256(body),
            }
            receipt_path = receipt_root / (
                f"restore-{job_sha256}-{receipt['restore_receipt_sha256']}.json"
            )
            assert_restore_locks()
            _write_once_anchored(
                receipt_root,
                receipt_path.name,
                receipt,
                field="raw restore receipt",
            )
            assert_restore_locks()
            verified = _validate_restore_receipt(
                receipt_path,
                prepared=prepared,
            )
            assert_restore_locks()
            _cleanup_restore_staging(prepared)
            assert_restore_locks()
            return verified


def verify_existing_historical_job_raw_restore(
    *,
    run_root: Path,
    archive_root: Path,
    job_sha256: str,
    zstd_bin: str,
) -> dict[str, Any]:
    """Deep-verify one restored job and its exact append-only lineage."""

    validated_zstd = _validated_restore_zstd(zstd_bin)
    job = _sha(job_sha256, "job SHA-256")
    try:
        root = Path(run_root).resolve(strict=True)
        destination = Path(archive_root).resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError("post-restore root is unavailable") from exc
    if not root.is_dir() or not destination.is_dir():
        raise DojoHistoricalRawReclaimError(
            "post-restore roots must be real directories"
        )
    v2_restore_root = root / "restore-v2-receipts"
    use_v2 = bool(
        v2_restore_root.is_dir() and list(v2_restore_root.glob(f"restore-{job}-*.json"))
    )
    reclaim_root = root / ("reclaim-v2-receipts" if use_v2 else "reclaim-receipts")
    restore_root = v2_restore_root if use_v2 else root / "restore-receipts"
    try:
        reclaim_state = reclaim_root.stat(follow_symlinks=False)
        restore_state = restore_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError(
            "restore lineage directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(reclaim_state.st_mode) or not stat.S_ISDIR(
        restore_state.st_mode
    ):
        raise DojoHistoricalRawReclaimError(
            "restore lineage directories must be real directories"
        )
    plan_paths = sorted(reclaim_root.glob(f"plan-job-{job}-*.json"))
    reclaim_paths = sorted(reclaim_root.glob(f"reclaim-{job}-*.json"))
    restore_paths = sorted(restore_root.glob(f"restore-{job}-*.json"))
    if len(plan_paths) != 1 or len(reclaim_paths) != 1 or len(restore_paths) != 1:
        raise DojoHistoricalRawReclaimError(
            "post-restore verification requires exact-one plan, reclaim, and restore receipt"
        )
    local_paths = sorted((destination / "receipts").glob(f"job-{job}-*.json"))
    if len(local_paths) != 1:
        raise DojoHistoricalRawReclaimError(
            "post-restore verification requires exact-one local archive receipt"
        )
    prepared = _prepare_restore(
        run_root=root,
        archive_receipt_path=local_paths[0],
        reclaim_plan_path=plan_paths[0],
        reclaim_receipt_path=reclaim_paths[0],
        zstd_bin=validated_zstd,
    )
    receipt = _validate_restore_receipt(
        restore_paths[0],
        prepared=prepared,
    )
    try:
        deep = verify_existing_historical_job_archive(
            run_root=root,
            job_sha256=job,
            archive_root=destination,
            zstd_bin=validated_zstd,
        )
    except DojoHistoricalJobArchiveError as exc:
        raise DojoHistoricalRawReclaimError(str(exc)) from exc
    if deep != prepared.local_receipt:
        raise DojoHistoricalRawReclaimError(
            "post-restore archive verifier returned another receipt"
        )
    verification_body = {
        "contract": RECLAIM_VERIFICATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "LOCALLY_ARCHIVED_AND_RAW_RESTORED",
        "job_sha256": job,
        "bundle_kind": SUCCESS_BUNDLE_KIND,
        "manifest_sha256": prepared.local_receipt["manifest_sha256"],
        "local_archive_receipt_sha256": prepared.local_receipt["receipt_sha256"],
        "reclaim_plan_sha256": prepared.reclaim_plan["plan_sha256"],
        "reclaim_receipt_sha256": prepared.reclaim_receipt["reclaim_receipt_sha256"],
        "restore_receipt_sha256": receipt["restore_receipt_sha256"],
        "archive_and_parts_verified": True,
        "remote_raw_readback_verified": False,
        "remote_receipt_trusted": False,
        "retained_bytes_verified": True,
        "all_raw_targets_missing": False,
        "all_raw_targets_present": True,
        "raw_reclaimed": False,
        "raw_restored": True,
        "raw_target_count": len(prepared.targets),
        "restored_logical_bytes": sum(row["size_bytes"] for row in prepared.targets),
        "zstd_executable_path": prepared.reclaim_plan.get("zstd_executable_path"),
        "zstd_executable_sha256": prepared.reclaim_plan.get("zstd_executable_sha256"),
        "zstd_executable_size_bytes": prepared.reclaim_plan.get(
            "zstd_executable_size_bytes"
        ),
        "zstd_version": prepared.reclaim_plan.get("zstd_version"),
        **_AUTHORITY,
    }
    return {
        **verification_body,
        "verification_sha256": _canonical_sha256(verification_body),
    }


def verify_existing_historical_job_raw_reclaim(
    *,
    run_root: Path,
    archive_root: Path,
    job_sha256: str,
    zstd_bin: str,
    expected_drive_parent_id: str | None = None,
) -> dict[str, Any]:
    """Verify restored state or one signed-V2 reclaimed-only lineage."""

    job = _sha(job_sha256, "job SHA-256")
    try:
        root = Path(run_root).resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalRawReclaimError("post-reclaim root is unavailable") from exc
    v2_reclaim_root = root / "reclaim-v2-receipts"
    v2_plans = (
        sorted(v2_reclaim_root.glob(f"plan-job-{job}-*.json"))
        if v2_reclaim_root.is_dir()
        else []
    )
    v2_receipts = (
        sorted(v2_reclaim_root.glob(f"reclaim-{job}-*.json"))
        if v2_reclaim_root.is_dir()
        else []
    )
    v2_restore_root = root / "restore-v2-receipts"
    v2_restore_paths = (
        sorted(v2_restore_root.glob(f"restore-{job}-*.json"))
        if v2_restore_root.is_dir()
        else []
    )
    if v2_restore_paths:
        return verify_existing_historical_job_raw_restore(
            run_root=root,
            archive_root=archive_root,
            job_sha256=job,
            zstd_bin=zstd_bin,
        )
    restore_root = root / "restore-receipts"
    restore_paths = (
        sorted(restore_root.glob(f"restore-{job}-*.json"))
        if restore_root.is_dir()
        else []
    )
    if not v2_plans and not v2_receipts and restore_paths:
        return verify_existing_historical_job_raw_restore(
            run_root=root,
            archive_root=archive_root,
            job_sha256=job,
            zstd_bin=zstd_bin,
        )
    if expected_drive_parent_id is None:
        raise DojoHistoricalRawReclaimError(
            "reclaimed-only verification requires the expected Drive parent"
        )
    destination = Path(archive_root).resolve(strict=True)
    reclaim_root = v2_reclaim_root
    plans = v2_plans
    receipts = v2_receipts
    local_paths = sorted((destination / "receipts").glob(f"job-{job}-*.json"))
    if len(plans) != 1 or len(receipts) != 1 or len(local_paths) != 1:
        raise DojoHistoricalRawReclaimError(
            "post-reclaim verification requires exact-one plan, receipt, and local archive"
        )
    plan = _validate_plan_seal(plans[0], expected_job_sha256=job)
    if plan["contract"] != RECLAIM_PLAN_CONTRACT_V2:
        raise DojoHistoricalRawReclaimError(
            "unsigned legacy reclaim lineage is permanently non-authoritative"
        )
    expected_plan_path = reclaim_root / (
        f"plan-job-{job}-{plan['remote_receipt_sha256']}.json"
    )
    if plans[0] != expected_plan_path:
        raise DojoHistoricalRawReclaimError(
            "raw reclaim plan filename does not match its lineage"
        )
    local = _read_canonical_json(local_paths[0], field="local archive receipt")
    if (
        local.get("receipt_sha256") != plan["local_archive_receipt_sha256"]
        or local.get("manifest_sha256") != plan["manifest_sha256"]
    ):
        raise DojoHistoricalRawReclaimError(
            "raw reclaim plan resolves another local archive"
        )
    authority_path = (
        destination
        / "remote-authorities"
        / (
            f"key-job-{job}-{plan['manifest_sha256']}-"
            f"{plan['attestation_authority_seal_sha256']}.json"
        )
    )
    remote_path = (
        destination
        / "remote-receipts"
        / (
            f"signed-job-{job}-{plan['manifest_sha256']}-"
            f"{plan['remote_receipt_sha256']}.json"
        )
    )
    prepared = _prepare_reclaim(
        run_root=root,
        archive_receipt_path=local_paths[0],
        remote_receipt_path=remote_path,
        expected_drive_parent_id=expected_drive_parent_id,
        attestation_authority_seal_path=authority_path,
        permit_missing_targets=True,
        zstd_bin=zstd_bin,
    )
    plan = _validate_plan(plans[0], _plan_body(prepared))
    receipt = _validate_reclaim_receipt(
        receipts[0],
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
        "remote_receipt_trusted": True,
        "retained_bytes_verified": True,
        "zero_byte_retirement_anchors_verified": True,
        "all_raw_targets_missing": True,
        "all_raw_targets_present": False,
        "raw_reclaimed": True,
        "raw_restored": False,
        "raw_target_count": plan["target_count"],
        "reclaimed_logical_bytes": plan["target_bytes"],
        "zstd_executable_path": plan["zstd_executable_path"],
        "zstd_executable_sha256": plan["zstd_executable_sha256"],
        "zstd_executable_size_bytes": plan["zstd_executable_size_bytes"],
        "zstd_version": plan["zstd_version"],
        **_AUTHORITY,
    }
    return {
        **verification_body,
        "verification_sha256": _canonical_sha256(verification_body),
    }


__all__ = [
    "REMOTE_READBACK_EVIDENCE_CONTRACT",
    "REMOTE_READBACK_CANDIDATE_CONTRACT",
    "REMOTE_READBACK_RECEIPT_CONTRACT",
    "REMOTE_READBACK_SIGNED_ATTESTATION_CONTRACT",
    "REMOTE_READBACK_ATTESTATION_BODY_CONTRACT",
    "ATTESTATION_PUBLIC_KEY_SEAL_CONTRACT",
    "RECLAIM_PLAN_CONTRACT",
    "RECLAIM_PLAN_CONTRACT_V2",
    "RECLAIM_RECEIPT_CONTRACT",
    "RESTORE_RECEIPT_CONTRACT",
    "RECLAIM_VERIFICATION_CONTRACT",
    "RAW_STORAGE_BOUNDARY_CONTRACT",
    "RAW_STORAGE_BOUNDARY",
    "LEGACY_RECLAIM_MODE",
    "RECLAIM_MODE",
    "DojoHistoricalRawReclaimError",
    "create_historical_job_remote_readback_receipt",
    "enroll_historical_job_attestation_public_key",
    "historical_raw_storage_boundary",
    "historical_global_heavy_lock_path",
    "publish_historical_job_signed_remote_readback_receipt",
    "reclaim_historical_job_raw",
    "restore_historical_job_raw",
    "verify_existing_historical_job_raw_reclaim",
    "verify_existing_historical_job_raw_restore",
    "verify_historical_job_raw_reclaim",
]

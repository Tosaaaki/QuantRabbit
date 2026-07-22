"""Authenticated, append-only second-wave reclaim for legacy DOJO cells.

V1 remote receipts were self-attested JSON.  They remain migration lineage,
but can never authorize this module to unlink data.  V2 binds every archive
(including the already reclaimed first wave) to a short-lived Ed25519-signed
Drive attestation.  The public key must first be enrolled in a separate
append-only pre-observation seal, which is then bound into the immutable V2
plan; this module never creates, loads, or stores a private key.

Planning and verification are non-destructive.  Reclaim requires exact plan
SHA/count/bytes confirmations and a complete signed attestation set.  Restore
does not require remote authority: it re-verifies the sealed local archives,
then publishes raw members atomically without overwriting an existing path.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tarfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Final, Iterator

import quant_rabbit.dojo_drive_archive as drive_archive
import quant_rabbit.dojo_legacy_cell_raw_reclaim as legacy


PLAN_CONTRACT: Final = "QR_DOJO_LEGACY_CELL_RAW_RECLAIM_PLAN_V2"
AUTHORIZATION_CONTRACT: Final = "QR_DOJO_LEGACY_CELL_RAW_RECLAIM_AUTHORIZATION_V2"
RECLAIM_RECEIPT_CONTRACT: Final = "QR_DOJO_LEGACY_CELL_RAW_RECLAIM_V2"
RESTORE_RECEIPT_CONTRACT: Final = "QR_DOJO_LEGACY_CELL_RAW_RESTORE_V2"
ATTESTATION_CONTRACT: Final = "QR_DOJO_DRIVE_SIGNED_ATTESTATION_V2"
ATTESTATION_BODY_CONTRACT: Final = "QR_DOJO_DRIVE_ATTESTATION_BODY_V2"
ATTESTATION_KEY_ENROLLMENT_CONTRACT: Final = (
    "QR_DOJO_LEGACY_CELL_RAW_ATTESTATION_KEY_ENROLLMENT_V2"
)
SCHEMA_VERSION: Final = 2
GENERATION: Final = 2
HASH_CHUNK_BYTES: Final = 1024 * 1024

# Fifteen minutes bounds the gap between a read-only Drive observation and
# destructive use.  It is an engineering freshness window, not a market rule.
MAX_ATTESTATION_TTL_SECONDS: Final = 15 * 60
# One minute tolerates small clock skew without accepting future observations.
MAX_CLOCK_SKEW_SECONDS: Final = 60

_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_MD5_RE: Final = re.compile(r"[0-9a-f]{32}\Z")
_HEX_KEY_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_DRIVE_ID_RE: Final = re.compile(r"[A-Za-z0-9_-]{8,256}\Z")
_VERSION_RE: Final = re.compile(r"[1-9][0-9]{0,31}\Z")
_REVISION_RE: Final = re.compile(r"[A-Za-z0-9_+/=-]{8,512}\Z")
_AUTHORITY: Final = {
    "historical_train_is_proof": False,
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}
_LOCK_HELD_TOKEN: Final = object()


class DojoLegacyCellRawReclaimV2Error(ValueError):
    """V2 reclaim or restore evidence is incomplete or unsafe."""


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
        raise DojoLegacyCellRawReclaimV2Error("value is not strict JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoLegacyCellRawReclaimV2Error(f"{field} is not a SHA-256")
    return value


def _require_md5(value: Any, field: str) -> str:
    if not isinstance(value, str) or _MD5_RE.fullmatch(value) is None:
        raise DojoLegacyCellRawReclaimV2Error(f"{field} is not an MD5")
    return value


def _require_drive_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _DRIVE_ID_RE.fullmatch(value) is None:
        raise DojoLegacyCellRawReclaimV2Error(f"{field} is not a Drive id")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DojoLegacyCellRawReclaimV2Error(f"{field} must be nonnegative")
    return value


def _require_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise DojoLegacyCellRawReclaimV2Error(f"{field} is invalid")
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoLegacyCellRawReclaimV2Error(f"{field} is not an instant") from exc
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise DojoLegacyCellRawReclaimV2Error(f"{field} must be UTC-aware")
    return instant


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise DojoLegacyCellRawReclaimV2Error("relative path is invalid")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise DojoLegacyCellRawReclaimV2Error("relative path is unsafe")
    return pure.as_posix()


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = drive_archive._load_json(path, field=field)
    except drive_archive.DojoDriveArchiveError as exc:
        raise DojoLegacyCellRawReclaimV2Error(str(exc)) from exc
    if not isinstance(value, dict):
        raise DojoLegacyCellRawReclaimV2Error(f"{field} must be an object")
    return value


def _file_hashes(path: Path) -> tuple[str, str, int, int]:
    try:
        sha256, md5, size, allocated = legacy._hash_file(path)
    except legacy.DojoLegacyCellRawReclaimError as exc:
        raise DojoLegacyCellRawReclaimV2Error(str(exc)) from exc
    return sha256, md5, size, allocated


def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _seal_zstd(zstd_bin: str) -> dict[str, Any]:
    invocation = Path(zstd_bin)
    if not invocation.is_absolute():
        raise DojoLegacyCellRawReclaimV2Error("zstd path must be absolute")
    try:
        resolved = invocation.resolve(strict=True)
        state = resolved.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoLegacyCellRawReclaimV2Error("zstd is unavailable") from exc
    if not stat.S_ISREG(state.st_mode) or not os.access(resolved, os.X_OK):
        raise DojoLegacyCellRawReclaimV2Error("zstd is not an executable regular file")
    digest, _, size, _ = _file_hashes(resolved)
    try:
        completed = subprocess.run(
            [os.fspath(invocation), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DojoLegacyCellRawReclaimV2Error("zstd version check failed") from exc
    version = completed.stdout.strip()
    if not version or len(version) > 512:
        raise DojoLegacyCellRawReclaimV2Error("zstd version is invalid")
    return {
        "invocation_path": os.fspath(invocation),
        "resolved_path": os.fspath(resolved),
        "sha256": digest,
        "size_bytes": size,
        "version": version,
    }


def _normalize_public_key(value: str | None) -> dict[str, Any]:
    if value is None:
        return {
            "algorithm": "ED25519",
            "status": "UNCONFIGURED",
            "public_key_hex": None,
            "public_key_sha256": None,
        }
    normalized = value.strip().lower()
    if _HEX_KEY_RE.fullmatch(normalized) is None:
        raise DojoLegacyCellRawReclaimV2Error(
            "attestation public key must be 32-byte lowercase hex"
        )
    raw = bytes.fromhex(normalized)
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        Ed25519PublicKey.from_public_bytes(raw)
    except (ImportError, ValueError) as exc:
        raise DojoLegacyCellRawReclaimV2Error(
            "Ed25519 verifier is unavailable or the public key is invalid"
        ) from exc
    return {
        "algorithm": "ED25519",
        "status": "SEALED_OPERATOR_PUBLIC_KEY",
        "public_key_hex": normalized,
        "public_key_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _unconfigured_authority() -> dict[str, Any]:
    return {
        "algorithm": "ED25519",
        "status": "UNCONFIGURED",
        "public_key_hex": None,
        "public_key_sha256": None,
        "enrollment_seal_path": None,
        "enrollment_seal_sha256": None,
        "enrollment_created_at_utc": None,
    }


def _validate_authority_seal(*, source_run: Path, seal_path: Path) -> dict[str, Any]:
    root = Path(source_run).resolve(strict=True)
    path = Path(seal_path).resolve(strict=True)
    authority_dir = root / "legacy-cell-reclaim-v2" / "attestation-authority"
    authority_fd = os.open(authority_dir, _directory_flags())
    try:
        seals = _matching_regular_names(authority_fd, "key-")
        if len(seals) != 1:
            raise DojoLegacyCellRawReclaimV2Error(
                "V2 key enrollment authority is not exact-one"
            )
        seal = _load_canonical_json_at(
            authority_fd,
            seals[0],
            field="V2 attestation key enrollment seal",
        )
    finally:
        os.close(authority_fd)
    expected_keys = {
        "algorithm",
        "authority_seal_sha256",
        "broker_mutation_allowed",
        "contract",
        "created_at_utc",
        "historical_train_is_proof",
        "live_permission",
        "order_authority",
        "private_key_material_accepted",
        "promotion_eligible",
        "proof_eligible",
        "public_key_hex",
        "public_key_sha256",
        "schema_version",
        "source_run_root",
        "source_run_sha256",
        "status",
    }
    if set(seal) != expected_keys:
        raise DojoLegacyCellRawReclaimV2Error("V2 key enrollment schema is invalid")
    normalized = _normalize_public_key(seal.get("public_key_hex"))
    body = {key: value for key, value in seal.items() if key != "authority_seal_sha256"}
    digest = _require_sha(
        seal.get("authority_seal_sha256"), "V2 key enrollment seal SHA"
    )
    run = _load_json(root / "run.json", field="terminal run")
    run_body = {key: value for key, value in run.items() if key != "run_sha256"}
    if (
        seal.get("contract") != ATTESTATION_KEY_ENROLLMENT_CONTRACT
        or seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("status") != "OPERATOR_PUBLIC_KEY_ENROLLED_BEFORE_OBSERVATION"
        or seal.get("source_run_root") != os.fspath(root)
        or seal.get("source_run_sha256") != run.get("run_sha256")
        or run.get("run_sha256") != drive_archive.canonical_sha256(run_body)
        or seal.get("algorithm") != "ED25519"
        or seal.get("public_key_sha256") != normalized["public_key_sha256"]
        or seal.get("private_key_material_accepted") is not False
        or digest != _sha256(body)
        or any(seal.get(key) != value for key, value in _AUTHORITY.items())
    ):
        raise DojoLegacyCellRawReclaimV2Error("V2 key enrollment seal is invalid")
    created = _require_utc(seal.get("created_at_utc"), "key enrollment creation time")
    expected = authority_dir / f"key-{digest}.json"
    if path != expected:
        raise DojoLegacyCellRawReclaimV2Error(
            "V2 key enrollment seal path is not canonical"
        )
    if seals != [path.name]:
        raise DojoLegacyCellRawReclaimV2Error(
            "V2 key enrollment authority is not exact-one"
        )
    return {
        "algorithm": "ED25519",
        "status": "SEALED_OPERATOR_PUBLIC_KEY",
        "public_key_hex": normalized["public_key_hex"],
        "public_key_sha256": normalized["public_key_sha256"],
        "enrollment_seal_path": os.fspath(path),
        "enrollment_seal_sha256": digest,
        "enrollment_created_at_utc": created.isoformat(),
    }


def _validate_prior_receipt(
    *, receipt_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, field="V1 reclaim receipt")
    if set(receipt) != set(legacy._RECLAIM_RECEIPT_KEYS):
        raise DojoLegacyCellRawReclaimV2Error("V1 reclaim receipt schema is invalid")
    body = {
        key: value for key, value in receipt.items() if key != "reclaim_receipt_sha256"
    }
    digest = _require_sha(receipt.get("reclaim_receipt_sha256"), "V1 receipt SHA")
    if (
        receipt_path.name != f"reclaim-{digest}.json"
        or digest != _sha256(body)
        or receipt.get("contract") != legacy.RECLAIM_RECEIPT_CONTRACT
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "LEGACY_CELL_RAW_RECLAIMED"
        or receipt.get("reclaim_plan_sha256") != plan.get("reclaim_plan_sha256")
        or receipt.get("source_run_sha256") != plan.get("source_run_sha256")
        or receipt.get("remote_receipt_set_sha256")
        != plan.get("remote_receipt_set_sha256")
        or receipt.get("verified_cell_count") != plan.get("verified_cell_count")
        or receipt.get("unverified_cell_count") != plan.get("unverified_cell_count")
        or receipt.get("deleted_file_count") != plan.get("target_count")
        or receipt.get("deleted_files") != plan.get("targets")
        or receipt.get("reclaimed_logical_bytes") != plan.get("target_bytes")
        or receipt.get("remote_unverified_cells_excluded") is not True
        or receipt.get("restore_requires_verified_cell_archives") is not True
        or any(receipt.get(key) != value for key, value in _AUTHORITY.items())
    ):
        raise DojoLegacyCellRawReclaimV2Error("V1 reclaim receipt lineage is invalid")
    _require_utc(receipt.get("completed_at_utc"), "V1 receipt completion")
    return receipt


def _prior_lineage(
    *,
    source_run: Path,
    archive_root: Path,
    prior_plan_path: Path,
    prior_receipt_path: Path,
    prior_remote_receipts_dir: Path,
    expected_drive_parent_id: str,
    zstd_bin: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    prior_plan = _load_json(prior_plan_path, field="V1 reclaim plan")
    try:
        root, validated = legacy._validate_plan_resume(
            plan=prior_plan,
            source_run=source_run,
            archive_root=archive_root,
            remote_receipts_dir=prior_remote_receipts_dir,
            expected_drive_parent_id=expected_drive_parent_id,
            zstd_bin=zstd_bin,
        )
    except (legacy.DojoLegacyCellRawReclaimError, OSError) as exc:
        raise DojoLegacyCellRawReclaimV2Error(str(exc)) from exc
    prior_receipt = _validate_prior_receipt(
        receipt_path=prior_receipt_path.resolve(strict=True), plan=validated
    )
    for row in validated["targets"]:
        try:
            present = legacy._verify_row(root, row, missing_allowed=True)
        except legacy.DojoLegacyCellRawReclaimError as exc:
            raise DojoLegacyCellRawReclaimV2Error(str(exc)) from exc
        if present:
            raise DojoLegacyCellRawReclaimV2Error(
                "a V1 receipt target unexpectedly exists"
            )
    run = _load_json(root / "run.json", field="terminal run")
    return root, validated, prior_receipt, run


def _archive_plan_for_coordinate(archive_root: Path, coordinate_id: str) -> Path:
    plan_dir = archive_root / "plans"
    rows = sorted(plan_dir.glob(f"cell-{coordinate_id}-*.json"))
    if len(rows) != 1 or rows[0].is_symlink() or not rows[0].is_file():
        raise DojoLegacyCellRawReclaimV2Error(
            f"coordinate must have exactly one real archive plan: {coordinate_id}"
        )
    return rows[0]


def _detached_cell(
    *,
    root: Path,
    archive_root: Path,
    coordinate_id: str,
    coordinate: Mapping[str, Any],
    owners: Mapping[str, str],
    zstd_bin: str,
    require_source: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    plan_path = _archive_plan_for_coordinate(archive_root, coordinate_id)
    raw_plan = _load_json(plan_path, field="cell archive plan")
    plan_sha = _require_sha(raw_plan.get("plan_sha256"), "cell plan SHA")
    try:
        plan, finalization, archive_path = legacy._static_archive_bundle(
            archive_root=archive_root,
            coordinate_id=coordinate_id,
            plan_sha256=plan_sha,
            zstd_bin=zstd_bin,
            fresh_source_verification=False,
        )
        prefixes = legacy._coordinate_prefixes(root, coordinate_id, coordinate, owners)
        targets, shared = legacy._archived_rows_for_prefixes(
            plan=plan,
            coordinate_id=coordinate_id,
            prefixes=prefixes,
        )
        if require_source:
            observed, observed_shared, observed_prefixes = legacy._target_rows(
                root=root,
                coordinate_id=coordinate_id,
                coordinate=coordinate,
                owners=owners,
                plan=plan,
            )
            if (
                observed != targets
                or observed_shared != shared
                or observed_prefixes != prefixes
            ):
                raise DojoLegacyCellRawReclaimV2Error(
                    "current raw does not match its detached archive"
                )
            for row in targets:
                legacy._verify_row(root, row, missing_allowed=False)
    except legacy.DojoLegacyCellRawReclaimError as exc:
        raise DojoLegacyCellRawReclaimV2Error(str(exc)) from exc
    archive_sha, archive_md5, archive_size, _ = _file_hashes(archive_path)
    if (
        archive_sha != finalization["archive_sha256"]
        or archive_size != finalization["archive_size_bytes"]
    ):
        raise DojoLegacyCellRawReclaimV2Error("detached archive identity drifted")
    return (
        {
            "coordinate_id": coordinate_id,
            "plan_path": os.fspath(plan_path),
            "plan_sha256": plan_sha,
            "content_tree_sha256": plan["content_tree_sha256"],
            "finalization_sha256": finalization["finalization_sha256"],
            "archive_path": os.fspath(archive_path),
            "archive_sha256": archive_sha,
            "archive_md5": archive_md5,
            "archive_size_bytes": archive_size,
            "target_paths": [row["path"] for row in targets],
        },
        targets,
        shared,
    )


def build_v2_candidate_plan(
    *,
    source_run: Path,
    archive_root: Path,
    prior_plan_path: Path,
    prior_receipt_path: Path,
    prior_remote_receipts_dir: Path,
    expected_drive_parent_id: str,
    zstd_bin: str,
    attestation_authority_seal_path: Path | None = None,
) -> dict[str, Any]:
    """Build a read-only generation-2 candidate from the exact V1 lineage."""

    zstd = _seal_zstd(zstd_bin)
    root, prior, prior_receipt, run = _prior_lineage(
        source_run=Path(source_run),
        archive_root=Path(archive_root),
        prior_plan_path=Path(prior_plan_path),
        prior_receipt_path=Path(prior_receipt_path),
        prior_remote_receipts_dir=Path(prior_remote_receipts_dir),
        expected_drive_parent_id=expected_drive_parent_id,
        zstd_bin=zstd["invocation_path"],
    )
    destination = Path(archive_root).resolve(strict=True)
    try:
        coordinates, owners = legacy._session_owners(root, run)
    except legacy.DojoLegacyCellRawReclaimError as exc:
        raise DojoLegacyCellRawReclaimV2Error(str(exc)) from exc

    prior_ids = {row["coordinate_id"] for row in prior["verified_cells"]}
    new_ids = set(prior["unverified_coordinate_ids"])
    if (
        not prior_ids
        or not new_ids
        or prior_ids & new_ids
        or prior_ids | new_ids != set(coordinates)
    ):
        raise DojoLegacyCellRawReclaimV2Error(
            "V1 verified/unverified cells do not partition the terminal run"
        )

    prior_targets = sorted(prior["targets"], key=lambda row: row["path"])
    prior_target_paths = {row["path"] for row in prior_targets}
    cells: list[dict[str, Any]] = []
    new_targets: list[dict[str, Any]] = []
    cumulative_shared: dict[str, dict[str, Any]] = {}
    for coordinate_id in sorted(coordinates):
        cell, targets, shared = _detached_cell(
            root=root,
            archive_root=destination,
            coordinate_id=coordinate_id,
            coordinate=coordinates[coordinate_id],
            owners=owners,
            zstd_bin=zstd["invocation_path"],
            require_source=coordinate_id in new_ids,
        )
        expected_paths = {row["path"] for row in targets}
        if coordinate_id in prior_ids:
            planned = {
                row["path"]
                for row in prior_targets
                if row["coordinate_id"] == coordinate_id
            }
            if expected_paths != planned:
                raise DojoLegacyCellRawReclaimV2Error(
                    "V1 deleted target set differs from its detached archive"
                )
            cell["wave"] = "PRIOR_RECLAIMED_V1"
        else:
            if prior_target_paths & expected_paths:
                raise DojoLegacyCellRawReclaimV2Error(
                    "generation-2 targets overlap V1 targets"
                )
            new_targets.extend(targets)
            cell["wave"] = "GENERATION_2_PENDING"
        for row in shared:
            previous = cumulative_shared.setdefault(row["path"], row)
            if previous != row:
                raise DojoLegacyCellRawReclaimV2Error(
                    "cell archives disagree on shared terminal bytes"
                )
        cells.append(cell)

    new_targets.sort(key=lambda row: row["path"])
    cumulative_targets = sorted(
        prior_targets + new_targets, key=lambda row: row["path"]
    )
    cumulative_paths = [row["path"] for row in cumulative_targets]
    if len(cumulative_paths) != len(set(cumulative_paths)):
        raise DojoLegacyCellRawReclaimV2Error("cumulative raw target paths overlap")
    if len(new_targets) != prior.get("unverified_raw_file_count"):
        raise DojoLegacyCellRawReclaimV2Error(
            "generation-2 raw count differs from the V1 excluded set"
        )
    if {row["coordinate_id"] for row in new_targets} != new_ids:
        raise DojoLegacyCellRawReclaimV2Error(
            "generation-2 coordinates differ from the V1 excluded set"
        )
    authority = (
        _unconfigured_authority()
        if attestation_authority_seal_path is None
        else _validate_authority_seal(
            source_run=root, seal_path=Path(attestation_authority_seal_path)
        )
    )
    body = {
        "contract": PLAN_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "generation": GENERATION,
        "status": "OPERATOR_CANDIDATE_SIGNED_ATTESTATIONS_REQUIRED",
        "source_run_root": os.fspath(root),
        "archive_root": os.fspath(destination),
        "expected_drive_parent_id": _require_drive_id(
            expected_drive_parent_id, "expected Drive parent id"
        ),
        "source_run_sha256": prior["source_run_sha256"],
        "source_run_artifact_sha256": prior["source_run_artifact_sha256"],
        "evaluation_artifact_sha256": prior["evaluation_artifact_sha256"],
        "cells_artifact_sha256": prior["cells_artifact_sha256"],
        "predecessor": {
            "plan_path": os.fspath(Path(prior_plan_path).resolve(strict=True)),
            "plan_sha256": prior["reclaim_plan_sha256"],
            "receipt_path": os.fspath(Path(prior_receipt_path).resolve(strict=True)),
            "receipt_sha256": prior_receipt["reclaim_receipt_sha256"],
            "deleted_target_set_sha256": _sha256(prior_targets),
            "deleted_target_count": len(prior_targets),
            "deleted_target_bytes": sum(row["size_bytes"] for row in prior_targets),
            "verified_cell_ids": sorted(prior_ids),
        },
        "generation_2_cell_ids": sorted(new_ids),
        "generation_2_target_set_sha256": _sha256(new_targets),
        "generation_2_target_count": len(new_targets),
        "generation_2_target_bytes": sum(row["size_bytes"] for row in new_targets),
        "generation_2_targets": new_targets,
        "cumulative_cell_count": len(cells),
        "cumulative_target_set_sha256": _sha256(cumulative_targets),
        "cumulative_target_count": len(cumulative_targets),
        "cumulative_target_bytes": sum(row["size_bytes"] for row in cumulative_targets),
        "cumulative_targets": cumulative_targets,
        "cells": sorted(cells, key=lambda row: row["coordinate_id"]),
        "shared_terminal_files": [
            cumulative_shared[path] for path in sorted(cumulative_shared)
        ],
        "zstd": zstd,
        "attestation_authority": authority,
        "required_attestation_count": len(cells),
        "attestation_scope": "ALL_CUMULATIVE_CELL_ARCHIVES",
        "source_deletion_allowed": False,
        "source_deleted": False,
        "restore_supported": True,
        **_AUTHORITY,
    }
    return {**body, "reclaim_plan_sha256": _sha256(body)}


def publish_v2_plan(plan: Mapping[str, Any]) -> Path:
    """Publish one immutable generation-2 plan after public-key enrollment."""

    validated = _validate_plan_shape(plan)
    authority = validated["attestation_authority"]
    if authority["status"] != "SEALED_OPERATOR_PUBLIC_KEY":
        raise DojoLegacyCellRawReclaimV2Error(
            "SIGNED_ATTESTATION_PUBLIC_KEY_NOT_CONFIGURED"
        )
    root = Path(validated["source_run_root"]).resolve(strict=True)
    generation_dir = root / "legacy-cell-reclaim-v2" / "generation-000002"
    path = generation_dir / f"plan-{validated['reclaim_plan_sha256']}.json"
    with _locked_root(root) as (root_fd, lock_fd):
        _assert_lock_identity(root, root_fd, lock_fd)
        generation_fd = _directory_at(
            root_fd,
            "legacy-cell-reclaim-v2/generation-000002",
            create=True,
        )
        try:
            plans = _matching_regular_names(generation_fd, "plan-")
            if plans and plans != [path.name]:
                raise DojoLegacyCellRawReclaimV2Error("another V2 plan already exists")
            _write_once_at(
                generation_fd,
                path.name,
                validated,
                field="V2 reclaim plan",
                before_publish=lambda: _assert_lock_identity(root, root_fd, lock_fd),
            )
            _assert_unique_plan_at(generation_fd, validated)
        finally:
            os.close(generation_fd)
    return path


def _load_canonical_json_at(
    directory_fd: int, leaf: str, *, field: str
) -> dict[str, Any]:
    descriptor = os.open(
        leaf,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        payload = bytearray()
        while len(payload) <= 16 * 1024 * 1024:
            chunk = os.read(descriptor, HASH_CHUNK_BYTES)
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or _stable_identity(before) != _stable_identity(after)
        or len(payload) > 16 * 1024 * 1024
    ):
        raise DojoLegacyCellRawReclaimV2Error(f"{field} is unstable or oversized")
    try:
        value = json.loads(
            bytes(payload).decode("utf-8"),
            object_pairs_hook=drive_archive._reject_duplicates,
            parse_constant=drive_archive._reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DojoLegacyCellRawReclaimV2Error(f"{field} is invalid") from exc
    if not isinstance(value, dict) or bytes(payload) != _canonical_bytes(value) + b"\n":
        raise DojoLegacyCellRawReclaimV2Error(f"{field} is not canonical JSON")
    return value


def _assert_unique_plan_at(generation_fd: int, plan: Mapping[str, Any]) -> None:
    expected = f"plan-{plan['reclaim_plan_sha256']}.json"
    plans = _matching_regular_names(generation_fd, "plan-")
    if plans != [expected]:
        raise DojoLegacyCellRawReclaimV2Error(
            "generation-2 must contain exactly one bound plan"
        )
    observed = _load_canonical_json_at(
        generation_fd, expected, field="published V2 reclaim plan"
    )
    if observed != dict(plan):
        raise DojoLegacyCellRawReclaimV2Error("published V2 plan bytes drifted")


def _assert_unique_published_plan(
    plan: Mapping[str, Any], *, explicit_path: Path | None = None
) -> None:
    root = Path(plan["source_run_root"]).resolve(strict=True)
    expected = (
        root
        / "legacy-cell-reclaim-v2"
        / "generation-000002"
        / f"plan-{plan['reclaim_plan_sha256']}.json"
    )
    if (
        explicit_path is not None
        and Path(explicit_path).resolve(strict=True) != expected
    ):
        raise DojoLegacyCellRawReclaimV2Error("V2 plan path is not canonical")
    with _locked_root(root) as (root_fd, lock_fd):
        _assert_lock_identity(root, root_fd, lock_fd)
        generation_fd = _directory_at(
            root_fd,
            "legacy-cell-reclaim-v2/generation-000002",
            create=False,
        )
        try:
            _assert_unique_plan_at(generation_fd, plan)
        finally:
            os.close(generation_fd)


def _validate_plan_shape(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise DojoLegacyCellRawReclaimV2Error("V2 plan must be an object")
    result = dict(plan)
    digest = _require_sha(result.get("reclaim_plan_sha256"), "V2 plan SHA")
    body = {key: value for key, value in result.items() if key != "reclaim_plan_sha256"}
    if (
        result.get("contract") != PLAN_CONTRACT
        or result.get("schema_version") != SCHEMA_VERSION
        or result.get("generation") != GENERATION
        or result.get("status") != "OPERATOR_CANDIDATE_SIGNED_ATTESTATIONS_REQUIRED"
        or digest != _sha256(body)
        or result.get("source_deletion_allowed") is not False
        or result.get("source_deleted") is not False
        or result.get("restore_supported") is not True
        or any(result.get(key) != value for key, value in _AUTHORITY.items())
    ):
        raise DojoLegacyCellRawReclaimV2Error("V2 plan boundary is invalid")
    authority = result.get("attestation_authority")
    if not isinstance(authority, Mapping):
        raise DojoLegacyCellRawReclaimV2Error("V2 attestation authority is invalid")
    if authority.get("status") == "UNCONFIGURED":
        normalized = _unconfigured_authority()
    else:
        seal_path = authority.get("enrollment_seal_path")
        if not isinstance(seal_path, str):
            raise DojoLegacyCellRawReclaimV2Error(
                "V2 attestation authority seal path is missing"
            )
        normalized = _validate_authority_seal(
            source_run=Path(str(result.get("source_run_root", ""))),
            seal_path=Path(seal_path),
        )
    if dict(authority) != normalized:
        raise DojoLegacyCellRawReclaimV2Error("V2 public-key seal is invalid")
    targets = result.get("generation_2_targets")
    cumulative = result.get("cumulative_targets")
    cells = result.get("cells")
    predecessor = result.get("predecessor")
    generation_ids = result.get("generation_2_cell_ids")
    if (
        not isinstance(targets, list)
        or not targets
        or not isinstance(cumulative, list)
        or not cumulative
        or not isinstance(cells, list)
        or not cells
        or any(not isinstance(row, Mapping) for row in targets)
        or any(not isinstance(row, Mapping) for row in cumulative)
        or any(not isinstance(row, Mapping) for row in cells)
        or not isinstance(predecessor, Mapping)
        or not isinstance(generation_ids, list)
        or not generation_ids
        or result.get("generation_2_target_count") != len(targets)
        or result.get("generation_2_target_bytes")
        != sum(
            _require_nonnegative_int(row.get("size_bytes"), "target size")
            for row in targets
        )
        or result.get("generation_2_target_set_sha256") != _sha256(targets)
        or result.get("cumulative_target_count") != len(cumulative)
        or result.get("cumulative_target_bytes")
        != sum(
            _require_nonnegative_int(row.get("size_bytes"), "cumulative size")
            for row in cumulative
        )
        or result.get("cumulative_target_set_sha256") != _sha256(cumulative)
        or result.get("cumulative_cell_count") != len(cells)
        or result.get("required_attestation_count") != len(cells)
    ):
        raise DojoLegacyCellRawReclaimV2Error("V2 target summary is invalid")
    if (
        targets != sorted(targets, key=lambda row: row.get("path", ""))
        or cumulative != sorted(cumulative, key=lambda row: row.get("path", ""))
        or len({row.get("path") for row in cumulative}) != len(cumulative)
        or not all(row in cumulative for row in targets)
        or predecessor.get("deleted_target_count") != len(cumulative) - len(targets)
        or predecessor.get("deleted_target_bytes")
        != sum(row["size_bytes"] for row in cumulative if row not in targets)
        or predecessor.get("deleted_target_set_sha256")
        != _sha256([row for row in cumulative if row not in targets])
        or generation_ids != sorted(set(generation_ids))
        or result.get("attestation_scope") != "ALL_CUMULATIVE_CELL_ARCHIVES"
    ):
        raise DojoLegacyCellRawReclaimV2Error("V2 wave partition is invalid")
    for row in cumulative:
        if not isinstance(row, Mapping):
            raise DojoLegacyCellRawReclaimV2Error("V2 target row is invalid")
        if set(row) != set(legacy._TARGET_KEYS):
            raise DojoLegacyCellRawReclaimV2Error("V2 target schema is invalid")
        _safe_relative(row.get("path"))
        _require_sha(row.get("sha256"), "V2 target SHA")
        if Path(str(row.get("path"))).name not in legacy.RAW_FILENAMES:
            raise DojoLegacyCellRawReclaimV2Error("V2 target is outside allowlist")
    cell_rows = _attestation_cells(result)
    prior_ids = predecessor.get("verified_cell_ids")
    if (
        not isinstance(prior_ids, list)
        or prior_ids != sorted(set(prior_ids))
        or set(prior_ids) & set(generation_ids)
        or set(prior_ids) | set(generation_ids) != set(cell_rows)
    ):
        raise DojoLegacyCellRawReclaimV2Error("V2 cell partition is invalid")
    archive_root = Path(str(result.get("archive_root", "")))
    if not archive_root.is_absolute():
        raise DojoLegacyCellRawReclaimV2Error("V2 archive root is not absolute")
    for coordinate_id, cell in cell_rows.items():
        expected_wave = (
            "GENERATION_2_PENDING"
            if coordinate_id in generation_ids
            else "PRIOR_RECLAIMED_V1"
        )
        archive_path = Path(str(cell.get("archive_path", "")))
        plan_path = Path(str(cell.get("plan_path", "")))
        coordinate_targets = [
            row["path"] for row in cumulative if row["coordinate_id"] == coordinate_id
        ]
        if (
            cell.get("wave") != expected_wave
            or cell.get("target_paths") != coordinate_targets
            or archive_path.parent != archive_root / "archives"
            or plan_path.parent != archive_root / "plans"
            or archive_path.name
            != f"cell-{coordinate_id}-{cell.get('plan_sha256')}.tar.zst"
            or plan_path.name != f"cell-{coordinate_id}-{cell.get('plan_sha256')}.json"
        ):
            raise DojoLegacyCellRawReclaimV2Error("V2 cell archive mapping is invalid")
        _require_sha(cell.get("plan_sha256"), "V2 cell plan SHA")
        _require_sha(cell.get("content_tree_sha256"), "V2 content tree SHA")
        _require_sha(cell.get("finalization_sha256"), "V2 finalization SHA")
        _require_sha(cell.get("archive_sha256"), "V2 archive SHA")
        _require_md5(cell.get("archive_md5"), "V2 archive MD5")
        _require_nonnegative_int(cell.get("archive_size_bytes"), "V2 archive size")
    _require_drive_id(result.get("expected_drive_parent_id"), "Drive parent id")
    _require_sha(predecessor.get("plan_sha256"), "predecessor plan SHA")
    _require_sha(predecessor.get("receipt_sha256"), "predecessor receipt SHA")
    zstd = result.get("zstd")
    if not isinstance(zstd, Mapping) or dict(zstd) != _seal_zstd(
        str(zstd.get("invocation_path", ""))
    ):
        raise DojoLegacyCellRawReclaimV2Error("V2 zstd seal drifted")
    return result


def load_v2_plan(path: Path) -> dict[str, Any]:
    plan_path = Path(path).resolve(strict=True)
    plan = _validate_plan_shape(_load_json(plan_path, field="V2 reclaim plan"))
    root = Path(plan["source_run_root"]).resolve(strict=True)
    expected = (
        root
        / "legacy-cell-reclaim-v2"
        / "generation-000002"
        / f"plan-{plan['reclaim_plan_sha256']}.json"
    )
    if plan_path != expected:
        raise DojoLegacyCellRawReclaimV2Error("V2 plan path is not canonical")
    _assert_unique_published_plan(plan, explicit_path=plan_path)
    return plan


_ATTESTATION_BODY_KEYS: Final = {
    "archive_md5",
    "archive_name",
    "archive_sha256",
    "archive_size_bytes",
    "attestation_method",
    "contract",
    "coordinate_id",
    "drivefs_file_id_xattr",
    "drivefs_md5_field48",
    "drivefs_revision_id_field78",
    "drivefs_version_field57",
    "expires_at_utc",
    "files_get_after",
    "files_get_before",
    "independent_readback",
    "issuer",
    "observed_at_utc",
    "plan_sha256",
    "revisions_list",
    "schema_version",
}
_FILES_GET_METADATA_KEYS: Final = {
    "content_size_bytes",
    "drive_file_id",
    "drive_file_name",
    "drive_parent_ids",
    "head_revision_id",
    "md5_checksum",
    "mime_type",
    "modified_time",
    "trashed",
    "version",
}
_REVISIONS_LIST_KEYS: Final = {
    "current_revision_id",
    "drive_file_id",
    "listed_revision_ids",
}
_INDEPENDENT_READBACK_KEYS: Final = {
    "content_size_bytes",
    "drive_file_id",
    "md5_checksum",
    "requested_revision_id",
    "resolved_revision_id",
    "sha256",
}
_ATTESTATION_OBSERVATION_KEYS: Final = {
    "attestation_method",
    "coordinate_id",
    "drivefs_file_id_xattr",
    "drivefs_md5_field48",
    "drivefs_revision_id_field78",
    "drivefs_version_field57",
    "expires_at_utc",
    "files_get_after",
    "files_get_before",
    "independent_readback",
    "issuer",
    "observed_at_utc",
    "revisions_list",
}
_ATTESTATION_KEYS: Final = {
    "attestation_sha256",
    "body",
    "contract",
    "schema_version",
    "signature_algorithm",
    "signature_base64",
    "signing_public_key_sha256",
}


def _exact_mapping(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise DojoLegacyCellRawReclaimV2Error(f"{field} schema is invalid")
    return dict(value)


def _require_revision_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise DojoLegacyCellRawReclaimV2Error(f"{field} is not a Drive revision id")
    return value


def _attestation_cells(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for raw in plan["cells"]:
        if not isinstance(raw, Mapping):
            raise DojoLegacyCellRawReclaimV2Error("V2 cell row is invalid")
        coordinate_id = raw.get("coordinate_id")
        if not isinstance(coordinate_id, str) or coordinate_id in rows:
            raise DojoLegacyCellRawReclaimV2Error("V2 cell identity is invalid")
        rows[coordinate_id] = raw
    return rows


def _validate_attestation_body(
    *,
    body: Mapping[str, Any],
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    now: datetime,
    enforce_freshness: bool,
) -> dict[str, Any]:
    normalized = _exact_mapping(body, _ATTESTATION_BODY_KEYS, "signed attestation body")
    observed = _require_utc(
        normalized.get("observed_at_utc"), "attestation observed_at"
    )
    enrollment_created = _require_utc(
        plan["attestation_authority"].get("enrollment_created_at_utc"),
        "attestation key enrollment creation time",
    )
    if observed < enrollment_created:
        raise DojoLegacyCellRawReclaimV2Error(
            "signed attestation predates public-key enrollment"
        )
    expires = _require_utc(normalized.get("expires_at_utc"), "attestation expires_at")
    if (
        expires <= observed
        or expires - observed > timedelta(seconds=MAX_ATTESTATION_TTL_SECONDS)
        or observed > now + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS)
        or (enforce_freshness and expires <= now)
    ):
        raise DojoLegacyCellRawReclaimV2Error("signed attestation is stale")

    before = _exact_mapping(
        normalized.get("files_get_before"),
        _FILES_GET_METADATA_KEYS,
        "files.get before metadata",
    )
    after = _exact_mapping(
        normalized.get("files_get_after"),
        _FILES_GET_METADATA_KEYS,
        "files.get after metadata",
    )
    revisions = _exact_mapping(
        normalized.get("revisions_list"),
        _REVISIONS_LIST_KEYS,
        "revisions.list evidence",
    )
    readback = _exact_mapping(
        normalized.get("independent_readback"),
        _INDEPENDENT_READBACK_KEYS,
        "independent revision readback evidence",
    )

    expected_name = Path(cell["archive_path"]).name
    before_file_id = _require_drive_id(
        before.get("drive_file_id"), "files.get before Drive file id"
    )
    before_revision = _require_revision_id(
        before.get("head_revision_id"), "files.get before head revision"
    )
    before_version = before.get("version")
    before_modified = _require_utc(
        before.get("modified_time"), "files.get before modified time"
    )
    before_size = _require_nonnegative_int(
        before.get("content_size_bytes"), "files.get before content size"
    )
    before_md5 = _require_md5(before.get("md5_checksum"), "files.get before MD5")
    parent_ids = before.get("drive_parent_ids")
    mime_type = before.get("mime_type")
    if (
        before != after
        or before.get("drive_file_name") != expected_name
        or not isinstance(parent_ids, list)
        or parent_ids != [plan["expected_drive_parent_id"]]
        or before_size != cell["archive_size_bytes"]
        or before_md5 != cell["archive_md5"]
        or before.get("trashed") is not False
        or not isinstance(mime_type, str)
        or not mime_type
        or len(mime_type) > 256
        or not isinstance(before_version, str)
        or _VERSION_RE.fullmatch(before_version) is None
        or before_modified > observed
    ):
        raise DojoLegacyCellRawReclaimV2Error(
            "files.get before/after metadata is invalid or changed"
        )

    revisions_file_id = _require_drive_id(
        revisions.get("drive_file_id"), "revisions.list Drive file id"
    )
    current_revision = _require_revision_id(
        revisions.get("current_revision_id"), "revisions.list current revision"
    )
    listed = revisions.get("listed_revision_ids")
    if not isinstance(listed, list) or not listed or len(listed) > 100_000:
        raise DojoLegacyCellRawReclaimV2Error("revisions.list revision set is invalid")
    normalized_listed = [
        _require_revision_id(value, "revisions.list listed revision")
        for value in listed
    ]
    if (
        len(normalized_listed) != len(set(normalized_listed))
        or normalized_listed.count(before_revision) != 1
        or current_revision != before_revision
        or revisions_file_id != before_file_id
    ):
        raise DojoLegacyCellRawReclaimV2Error(
            "revisions.list does not uniquely bind the current head revision"
        )

    readback_file_id = _require_drive_id(
        readback.get("drive_file_id"), "independent readback Drive file id"
    )
    requested_revision = _require_revision_id(
        readback.get("requested_revision_id"),
        "independent readback requested revision",
    )
    resolved_revision = _require_revision_id(
        readback.get("resolved_revision_id"),
        "independent readback resolved revision",
    )
    readback_size = _require_nonnegative_int(
        readback.get("content_size_bytes"), "independent readback content size"
    )
    readback_md5 = _require_md5(
        readback.get("md5_checksum"), "independent readback MD5"
    )
    readback_sha = _require_sha(readback.get("sha256"), "independent readback SHA")
    if (
        readback_file_id != before_file_id
        or requested_revision != before_revision
        or resolved_revision != before_revision
        or readback_size != cell["archive_size_bytes"]
        or readback_md5 != cell["archive_md5"]
        or readback_sha != cell["archive_sha256"]
    ):
        raise DojoLegacyCellRawReclaimV2Error(
            "independent readback is not bound to the current Drive revision"
        )

    method = normalized.get("attestation_method")
    issuer = normalized.get("issuer")
    drivefs_file_id = _require_drive_id(
        normalized.get("drivefs_file_id_xattr"), "DriveFS file id xattr"
    )
    drivefs_md5 = _require_md5(
        normalized.get("drivefs_md5_field48"), "DriveFS MD5 field 48"
    )
    drivefs_revision = _require_revision_id(
        normalized.get("drivefs_revision_id_field78"),
        "DriveFS current revision field 78",
    )
    drivefs_version = normalized.get("drivefs_version_field57")
    if method not in {
        "DRIVE_API_DOWNLOAD_SHA256",
        "DRIVEFS_EVICTED_REMOTE_REDOWNLOAD_SHA256_WITH_DRIVE_API_REVISION",
    }:
        raise DojoLegacyCellRawReclaimV2Error(
            "only structured files.get/revisions.list/current-revision readback "
            "evidence can authorize deletion"
        )
    if (
        normalized.get("contract") != ATTESTATION_BODY_CONTRACT
        or normalized.get("schema_version") != SCHEMA_VERSION
        or normalized.get("plan_sha256") != plan["reclaim_plan_sha256"]
        or normalized.get("coordinate_id") != cell["coordinate_id"]
        or normalized.get("archive_name") != expected_name
        or normalized.get("archive_sha256") != cell["archive_sha256"]
        or normalized.get("archive_md5") != cell["archive_md5"]
        or normalized.get("archive_size_bytes") != cell["archive_size_bytes"]
        or isinstance(normalized.get("archive_size_bytes"), bool)
        or drivefs_file_id != before_file_id
        or drivefs_md5 != before_md5
        or drivefs_revision != before_revision
        or drivefs_version != before_version
        or not isinstance(drivefs_version, str)
        or _VERSION_RE.fullmatch(drivefs_version) is None
        or not isinstance(issuer, str)
        or not issuer
        or len(issuer) > 256
    ):
        raise DojoLegacyCellRawReclaimV2Error(
            "signed Drive attestation lineage or DriveFS binding is invalid"
        )
    return normalized


def build_v2_attestation_body_candidate(
    *,
    plan: Mapping[str, Any],
    observation: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build and validate one unsigned body for an external Ed25519 signer."""

    validated = _validate_plan_shape(plan)
    _assert_unique_published_plan(validated)
    normalized_observation = _exact_mapping(
        observation,
        _ATTESTATION_OBSERVATION_KEYS,
        "attestation observation packet",
    )
    coordinate_id = normalized_observation.get("coordinate_id")
    cells = _attestation_cells(validated)
    if not isinstance(coordinate_id, str) or coordinate_id not in cells:
        raise DojoLegacyCellRawReclaimV2Error(
            "attestation observation coordinate is not in the plan"
        )
    cell = cells[coordinate_id]
    body = {
        "contract": ATTESTATION_BODY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": validated["reclaim_plan_sha256"],
        "coordinate_id": coordinate_id,
        "archive_name": Path(cell["archive_path"]).name,
        "archive_sha256": cell["archive_sha256"],
        "archive_md5": cell["archive_md5"],
        "archive_size_bytes": cell["archive_size_bytes"],
        **normalized_observation,
    }
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise DojoLegacyCellRawReclaimV2Error("builder time must be UTC-aware")
    normalized_body = _validate_attestation_body(
        body=body,
        plan=validated,
        cell=cell,
        now=instant,
        enforce_freshness=True,
    )
    return {
        "status": "OPERATOR_CANDIDATE_ATTESTATION_BODY_NOT_SIGNED",
        "body": normalized_body,
        "canonical_body_sha256": _sha256(normalized_body),
        "source_deletion_allowed": False,
        **_AUTHORITY,
    }


def _verify_attestation(
    *,
    envelope: Mapping[str, Any],
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    now: datetime,
    enforce_freshness: bool,
) -> dict[str, Any]:
    if set(envelope) != _ATTESTATION_KEYS:
        raise DojoLegacyCellRawReclaimV2Error("signed attestation schema is invalid")
    body = envelope.get("body")
    if not isinstance(body, Mapping) or set(body) != _ATTESTATION_BODY_KEYS:
        raise DojoLegacyCellRawReclaimV2Error(
            "signed attestation body schema is invalid"
        )
    normalized_body = dict(body)
    authority = plan["attestation_authority"]
    public_key_hex = authority.get("public_key_hex")
    if authority.get("status") != "SEALED_OPERATOR_PUBLIC_KEY" or not isinstance(
        public_key_hex, str
    ):
        raise DojoLegacyCellRawReclaimV2Error(
            "SIGNED_ATTESTATION_PUBLIC_KEY_NOT_CONFIGURED"
        )
    digest_body = {
        key: value for key, value in envelope.items() if key != "attestation_sha256"
    }
    if (
        envelope.get("contract") != ATTESTATION_CONTRACT
        or envelope.get("schema_version") != SCHEMA_VERSION
        or envelope.get("signature_algorithm") != "ED25519"
        or envelope.get("signing_public_key_sha256")
        != authority.get("public_key_sha256")
        or _require_sha(envelope.get("attestation_sha256"), "signed attestation SHA")
        != _sha256(digest_body)
    ):
        raise DojoLegacyCellRawReclaimV2Error("signed attestation envelope is invalid")
    signature_text = envelope.get("signature_base64")
    if not isinstance(signature_text, str) or len(signature_text) > 128:
        raise DojoLegacyCellRawReclaimV2Error("attestation signature is invalid")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise DojoLegacyCellRawReclaimV2Error(
            "Ed25519 verifier is unavailable"
        ) from exc
    try:
        signature = base64.b64decode(signature_text, validate=True)
        if (
            len(signature) != 64
            or base64.b64encode(signature).decode("ascii") != signature_text
        ):
            raise ValueError
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            signature, _canonical_bytes(normalized_body)
        )
    except (binascii.Error, InvalidSignature, ValueError) as exc:
        raise DojoLegacyCellRawReclaimV2Error(
            "attestation Ed25519 signature is invalid"
        ) from exc

    _validate_attestation_body(
        body=normalized_body,
        plan=plan,
        cell=cell,
        now=now,
        enforce_freshness=enforce_freshness,
    )
    return dict(envelope)


def verify_signed_attestations(
    *,
    plan: Mapping[str, Any],
    attestations_dir: Path,
    now: datetime | None = None,
    enforce_freshness: bool = True,
    _lock_held_token: object | None = None,
) -> list[dict[str, Any]]:
    """Verify one authenticated Drive observation for every sealed archive."""

    validated = _validate_plan_shape(plan)
    if _lock_held_token is not _LOCK_HELD_TOKEN:
        _assert_unique_published_plan(validated)
    directory = Path(attestations_dir).resolve(strict=True)
    if directory.is_symlink() or not directory.is_dir():
        raise DojoLegacyCellRawReclaimV2Error("attestation directory is unsafe")
    paths = sorted(directory.iterdir(), key=lambda path: path.name)
    if len(paths) != validated["required_attestation_count"] or any(
        path.is_symlink() or not path.is_file() or path.suffix != ".json"
        for path in paths
    ):
        raise DojoLegacyCellRawReclaimV2Error(
            "attestation directory is not the exact required set"
        )
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise DojoLegacyCellRawReclaimV2Error("verification time must be UTC-aware")
    cells = _attestation_cells(validated)
    observed: dict[str, dict[str, Any]] = {}
    file_ids: set[str] = set()
    for path in paths:
        envelope = _load_json(path, field="signed Drive attestation")
        body = envelope.get("body")
        coordinate_id = body.get("coordinate_id") if isinstance(body, Mapping) else None
        if not isinstance(coordinate_id, str) or coordinate_id not in cells:
            raise DojoLegacyCellRawReclaimV2Error(
                "attestation coordinate is not in the plan"
            )
        if coordinate_id in observed:
            raise DojoLegacyCellRawReclaimV2Error(
                "attestation coordinate is duplicated"
            )
        verified = _verify_attestation(
            envelope=envelope,
            plan=validated,
            cell=cells[coordinate_id],
            now=instant,
            enforce_freshness=enforce_freshness,
        )
        file_id = verified["body"]["files_get_before"]["drive_file_id"]
        if file_id in file_ids:
            raise DojoLegacyCellRawReclaimV2Error("Drive file id is duplicated")
        file_ids.add(file_id)
        observed[coordinate_id] = verified
    if set(observed) != set(cells):
        raise DojoLegacyCellRawReclaimV2Error("attestation set is incomplete")
    return [observed[key] for key in sorted(observed)]


def _generation_directory(plan: Mapping[str, Any]) -> Path:
    return (
        Path(plan["source_run_root"]).resolve(strict=True)
        / "legacy-cell-reclaim-v2"
        / "generation-000002"
    )


def _authorization_from_attestations(
    *, plan: Mapping[str, Any], attestations: Sequence[Mapping[str, Any]], now: datetime
) -> dict[str, Any]:
    expires = min(
        _require_utc(row["body"]["expires_at_utc"], "attestation expiry")
        for row in attestations
    )
    body = {
        "contract": AUTHORIZATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "SIGNED_REMOTE_SET_AUTHORIZED",
        "plan_sha256": plan["reclaim_plan_sha256"],
        "authorized_at_utc": now.isoformat(),
        "valid_until_utc": expires.isoformat(),
        "attestation_set_sha256": _sha256(list(attestations)),
        "attestations": list(attestations),
        "target_count": plan["generation_2_target_count"],
        "target_bytes": plan["generation_2_target_bytes"],
        "authority": "SIGNED_ED25519_ATTESTATION_SET",
        **_AUTHORITY,
    }
    return {**body, "authorization_sha256": _sha256(body)}


def _validate_authorization(
    *, plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    authorization = dict(value)
    digest = _require_sha(
        authorization.get("authorization_sha256"), "authorization SHA"
    )
    body = {
        key: item
        for key, item in authorization.items()
        if key != "authorization_sha256"
    }
    attestations = authorization.get("attestations")
    authorized_at = _require_utc(
        authorization.get("authorized_at_utc"), "authorization time"
    )
    if (
        authorization.get("contract") != AUTHORIZATION_CONTRACT
        or authorization.get("schema_version") != SCHEMA_VERSION
        or authorization.get("status") != "SIGNED_REMOTE_SET_AUTHORIZED"
        or digest != _sha256(body)
        or authorization.get("plan_sha256") != plan["reclaim_plan_sha256"]
        or authorization.get("target_count") != plan["generation_2_target_count"]
        or authorization.get("target_bytes") != plan["generation_2_target_bytes"]
        or authorization.get("authority") != "SIGNED_ED25519_ATTESTATION_SET"
        or not isinstance(attestations, list)
        or authorization.get("attestation_set_sha256") != _sha256(attestations)
        or any(authorization.get(key) != item for key, item in _AUTHORITY.items())
    ):
        raise DojoLegacyCellRawReclaimV2Error("reclaim authorization is invalid")
    cells = _attestation_cells(plan)
    seen: set[str] = set()
    expires: list[datetime] = []
    for envelope in attestations:
        if not isinstance(envelope, Mapping):
            raise DojoLegacyCellRawReclaimV2Error(
                "authorization attestation is invalid"
            )
        raw_body = envelope.get("body")
        coordinate = (
            raw_body.get("coordinate_id") if isinstance(raw_body, Mapping) else None
        )
        if (
            not isinstance(coordinate, str)
            or coordinate not in cells
            or coordinate in seen
        ):
            raise DojoLegacyCellRawReclaimV2Error(
                "authorization attestation set is invalid"
            )
        _verify_attestation(
            envelope=envelope,
            plan=plan,
            cell=cells[coordinate],
            now=authorized_at,
            enforce_freshness=True,
        )
        expires.append(_require_utc(raw_body["expires_at_utc"], "attestation expiry"))
        seen.add(coordinate)
    if (
        seen != set(cells)
        or authorization.get("valid_until_utc") != min(expires).isoformat()
    ):
        raise DojoLegacyCellRawReclaimV2Error("authorization coverage is invalid")
    return authorization


def _load_or_create_authorization(
    *, plan: Mapping[str, Any], attestations_dir: Path
) -> tuple[dict[str, Any], Path]:
    directory = _generation_directory(plan)
    existing = sorted(directory.glob("authorization-*.json"))
    if len(existing) > 1:
        raise DojoLegacyCellRawReclaimV2Error("multiple authorizations exist")
    if existing:
        value = _validate_authorization(
            plan=plan, value=_load_json(existing[0], field="reclaim authorization")
        )
        if existing[0].name != f"authorization-{value['authorization_sha256']}.json":
            raise DojoLegacyCellRawReclaimV2Error("authorization path is invalid")
        return value, existing[0]
    now = datetime.now(timezone.utc)
    attestations = verify_signed_attestations(
        plan=plan,
        attestations_dir=attestations_dir,
        now=now,
        _lock_held_token=_LOCK_HELD_TOKEN,
    )
    value = _authorization_from_attestations(
        plan=plan, attestations=attestations, now=now
    )
    path = directory / f"authorization-{value['authorization_sha256']}.json"
    _write_once_v2(path, value, field="V2 reclaim authorization")
    return value, path


def _atomic_rename_at_no_replace(
    directory_fd: int, source_name: str, destination_name: str
) -> bool:
    """Rename leaves in one open directory without replacing an anchor."""

    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError as exc:
            raise DojoLegacyCellRawReclaimV2Error(
                "atomic no-replace rename is unsupported"
            ) from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(directory_fd, source, directory_fd, destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise DojoLegacyCellRawReclaimV2Error(
                "atomic no-replace rename is unsupported"
            ) from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(directory_fd, source, directory_fd, destination, 0x00000001)
    else:
        raise DojoLegacyCellRawReclaimV2Error("atomic no-replace rename is unsupported")
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
        raise DojoLegacyCellRawReclaimV2Error(
            "atomic no-replace rename is unsupported by this filesystem"
        )
    raise DojoLegacyCellRawReclaimV2Error(
        f"atomic no-replace rename failed: {os.strerror(error_number)}"
    )


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_parent_at(
    root_fd: int, relative: str, *, create: bool = False
) -> tuple[int, str]:
    parts = _safe_relative(relative).split("/")
    descriptor = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            created = False
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.fsync(descriptor)
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            if created:
                os.fsync(child)
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except OSError:
        os.close(descriptor)
        raise


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), size


def _hash_descriptor_sha_md5(descriptor: int) -> tuple[str, str, int]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, HASH_CHUNK_BYTES):
        sha256.update(chunk)
        md5.update(chunk)
        size += len(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return sha256.hexdigest(), md5.hexdigest(), size


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise DojoLegacyCellRawReclaimV2Error("restore write made no progress")
        view = view[written:]


def _existing_payload_is_exact(
    directory_fd: int, leaf: str, payload: bytes, *, field: str
) -> bool:
    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except FileNotFoundError:
        return False
    try:
        state = os.fstat(descriptor)
        observed = bytearray()
        while len(observed) <= len(payload):
            chunk = os.read(descriptor, min(HASH_CHUNK_BYTES, len(payload) + 1))
            if not chunk:
                break
            observed.extend(chunk)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(state.st_mode) or bytes(observed) != payload:
        raise DojoLegacyCellRawReclaimV2Error(f"{field} already drifted")
    return True


def _write_once_at(
    directory_fd: int,
    leaf: str,
    value: Mapping[str, Any],
    *,
    field: str,
    before_publish: Callable[[], None] | None = None,
) -> None:
    payload = _canonical_bytes(value) + b"\n"
    descriptor: int | None = None
    temporary = f".{leaf}.publish-{secrets.token_hex(12)}.tmp"
    published = False
    try:
        if _existing_payload_is_exact(directory_fd, leaf, payload, field=field):
            return
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        named = os.stat(temporary, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stable_identity(opened) != _stable_identity(named)
        ):
            raise DojoLegacyCellRawReclaimV2Error(
                f"{field} temporary changed before publication"
            )
        if before_publish is not None:
            before_publish()
        if not _atomic_rename_at_no_replace(directory_fd, temporary, leaf):
            _existing_payload_is_exact(directory_fd, leaf, payload, field=field)
            return
        os.fsync(directory_fd)
        published_named = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        published_opened = os.fstat(descriptor)
        if _stable_identity(published_named) != _stable_identity(published_opened):
            raise DojoLegacyCellRawReclaimV2Error(
                f"{field} temporary was replaced during publication"
            )
        published = True
    finally:
        if descriptor is not None:
            if not published:
                try:
                    named = os.stat(
                        temporary, dir_fd=directory_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    pass
                else:
                    opened = os.fstat(descriptor)
                    if _stable_identity(named) == _stable_identity(opened):
                        if before_publish is not None:
                            before_publish()
                        os.ftruncate(descriptor, 0)
                        os.fsync(descriptor)
                        os.fsync(directory_fd)
            os.close(descriptor)


def _write_once_v2(path: Path, value: Mapping[str, Any], *, field: str) -> None:
    """Publish canonical JSON atomically without ever unlinking a path name."""

    directory_fd = os.open(path.parent, _directory_flags())
    try:
        _write_once_at(directory_fd, path.name, value, field=field)
    finally:
        os.close(directory_fd)


def _directory_at(root_fd: int, relative: str, *, create: bool) -> int:
    descriptor, _ = _open_parent_at(
        root_fd, f"{_safe_relative(relative)}/.directory-sentinel", create=create
    )
    return descriptor


def _matching_regular_names(directory_fd: int, prefix: str) -> list[str]:
    names = sorted(
        name
        for name in os.listdir(directory_fd)
        if name.startswith(prefix) and name.endswith(".json")
    )
    for name in names:
        state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(state.st_mode):
            raise DojoLegacyCellRawReclaimV2Error(
                "append-only lineage contains a non-regular artifact"
            )
    return names


def _assert_lock_identity(root: Path, root_fd: int, lock_fd: int) -> None:
    if _stable_identity(os.fstat(root_fd)) != _stable_identity(
        os.stat(root, follow_symlinks=False)
    ):
        raise DojoLegacyCellRawReclaimV2Error("locked root pathname was replaced")
    locked_name = os.stat(
        ".dojo-legacy-cell-raw-reclaim-v2.lock",
        dir_fd=root_fd,
        follow_symlinks=False,
    )
    if _stable_identity(os.fstat(lock_fd)) != _stable_identity(locked_name):
        raise DojoLegacyCellRawReclaimV2Error("V2 run lock pathname was replaced")


@contextmanager
def _locked_root(root: Path) -> Iterator[tuple[int, int]]:
    root_fd: int | None = None
    lock_fd: int | None = None
    try:
        root_fd = os.open(root, _directory_flags())
        before = os.fstat(root_fd)
        observed = os.stat(root, follow_symlinks=False)
        if _stable_identity(before) != _stable_identity(observed):
            raise DojoLegacyCellRawReclaimV2Error("source root changed while opened")
        lock_fd = os.open(
            ".dojo-legacy-cell-raw-reclaim-v2.lock",
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_fd,
        )
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise DojoLegacyCellRawReclaimV2Error("V2 run lock is not regular")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked_name = os.stat(
            ".dojo-legacy-cell-raw-reclaim-v2.lock",
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        if _stable_identity(os.fstat(lock_fd)) != _stable_identity(locked_name):
            raise DojoLegacyCellRawReclaimV2Error("V2 run lock was replaced")
    except BlockingIOError as exc:
        if lock_fd is not None:
            os.close(lock_fd)
        if root_fd is not None:
            os.close(root_fd)
        raise DojoLegacyCellRawReclaimV2Error("V2 reclaim run lock is held") from exc
    except OSError as exc:
        if lock_fd is not None:
            os.close(lock_fd)
        if root_fd is not None:
            os.close(root_fd)
        raise DojoLegacyCellRawReclaimV2Error("V2 reclaim lock failed") from exc
    except Exception:
        if lock_fd is not None:
            os.close(lock_fd)
        if root_fd is not None:
            os.close(root_fd)
        raise
    try:
        yield root_fd, lock_fd
        _assert_lock_identity(root, root_fd, lock_fd)
    finally:
        if lock_fd is not None:
            try:
                if root_fd is not None:
                    _assert_lock_identity(root, root_fd, lock_fd)
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        if root_fd is not None:
            os.close(root_fd)


@contextmanager
def _locked_roots_canonical(
    roots: Sequence[Path],
) -> Iterator[dict[Path, tuple[int, int]]]:
    normalized = sorted(
        {Path(root).resolve(strict=True) for root in roots}, key=os.fspath
    )
    with ExitStack() as stack:
        locked = {root: stack.enter_context(_locked_root(root)) for root in normalized}
        yield locked


def enroll_v2_attestation_public_key(
    *,
    source_run: Path,
    attestation_public_key_hex: str,
    expected_public_key_sha256: str,
) -> dict[str, Any]:
    """Enroll exactly one public key before any signed Drive observation."""

    root = Path(source_run).resolve(strict=True)
    normalized = _normalize_public_key(attestation_public_key_hex)
    fingerprint = _require_sha(
        expected_public_key_sha256, "confirmed public-key fingerprint"
    )
    if fingerprint != normalized["public_key_sha256"]:
        raise DojoLegacyCellRawReclaimV2Error(
            "public-key fingerprint confirmation differs"
        )
    run = _load_json(root / "run.json", field="terminal run")
    run_sha = _require_sha(run.get("run_sha256"), "source run SHA")
    seal: dict[str, Any] | None = None
    with _locked_root(root) as (root_fd, lock_fd):
        _assert_lock_identity(root, root_fd, lock_fd)
        authority_fd = _directory_at(
            root_fd,
            "legacy-cell-reclaim-v2/attestation-authority",
            create=True,
        )
        try:
            existing = _matching_regular_names(authority_fd, "key-")
            if existing:
                if len(existing) != 1:
                    raise DojoLegacyCellRawReclaimV2Error(
                        "V2 key enrollment authority is not exact-one"
                    )
                existing_path = (
                    root
                    / "legacy-cell-reclaim-v2"
                    / "attestation-authority"
                    / existing[0]
                )
                authority = _validate_authority_seal(
                    source_run=root, seal_path=existing_path
                )
                if authority["public_key_sha256"] != fingerprint:
                    raise DojoLegacyCellRawReclaimV2Error(
                        "another public key is already enrolled for this run"
                    )
                return authority
            body = {
                "contract": ATTESTATION_KEY_ENROLLMENT_CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "status": "OPERATOR_PUBLIC_KEY_ENROLLED_BEFORE_OBSERVATION",
                "source_run_root": os.fspath(root),
                "source_run_sha256": run_sha,
                "algorithm": "ED25519",
                "public_key_hex": normalized["public_key_hex"],
                "public_key_sha256": fingerprint,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "private_key_material_accepted": False,
                **_AUTHORITY,
            }
            seal = {**body, "authority_seal_sha256": _sha256(body)}
            _write_once_at(
                authority_fd,
                f"key-{seal['authority_seal_sha256']}.json",
                seal,
                field="V2 attestation key enrollment seal",
                before_publish=lambda: _assert_lock_identity(root, root_fd, lock_fd),
            )
            os.fsync(authority_fd)
        finally:
            os.close(authority_fd)
    if seal is None:
        raise DojoLegacyCellRawReclaimV2Error("V2 key enrollment did not publish")
    return _validate_authority_seal(
        source_run=root,
        seal_path=(
            root
            / "legacy-cell-reclaim-v2"
            / "attestation-authority"
            / f"key-{seal['authority_seal_sha256']}.json"
        ),
    )


def _anchor_name(row: Mapping[str, Any]) -> str:
    path_sha = hashlib.sha256(str(row["path"]).encode("utf-8")).hexdigest()[:20]
    return f".dojo-retired-v2-{path_sha}-{row['sha256'][:20]}.anchor"


def _retire_target(
    root_fd: int,
    row: Mapping[str, Any],
    before_irreversible: Callable[[], None],
) -> int:
    """Retire exactly one bound inode and retain a zero-byte audit anchor."""

    parent_fd, leaf = _open_parent_at(root_fd, str(row["path"]))
    descriptor: int | None = None
    anchor = _anchor_name(row)
    try:
        try:
            descriptor = os.open(
                leaf,
                os.O_RDWR
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            try:
                anchored = os.stat(anchor, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise DojoLegacyCellRawReclaimV2Error(
                    "planned raw is missing without a retirement anchor"
                ) from exc
            if not stat.S_ISREG(anchored.st_mode):
                raise DojoLegacyCellRawReclaimV2Error(
                    "incomplete retirement anchor requires manual recovery"
                )
            if anchored.st_size == 0:
                return 0
            descriptor = os.open(
                anchor,
                os.O_RDWR
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
            opened_anchor = os.fstat(descriptor)
            digest, size = _hash_descriptor(descriptor)
            anchored_after = os.stat(anchor, dir_fd=parent_fd, follow_symlinks=False)
            if (
                _stable_identity(opened_anchor) != _stable_identity(anchored)
                or _stable_identity(opened_anchor) != _stable_identity(anchored_after)
                or opened_anchor.st_nlink != 1
                or digest != row["sha256"]
                or size != row["size_bytes"]
            ):
                raise DojoLegacyCellRawReclaimV2Error(
                    "incomplete retirement anchor differs from its sealed raw"
                )
            allocated = opened_anchor.st_blocks * 512
            before_irreversible()
            os.ftruncate(descriptor, 0)
            os.fsync(descriptor)
            os.fsync(parent_fd)
            return allocated
        opened = os.fstat(descriptor)
        named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        digest, size = _hash_descriptor(descriptor)
        after = os.fstat(descriptor)
        named_after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stable_identity(opened) != _stable_identity(named)
            or _stable_identity(opened) != _stable_identity(after)
            or _stable_identity(after) != _stable_identity(named_after)
            or digest != row["sha256"]
            or size != row["size_bytes"]
        ):
            raise DojoLegacyCellRawReclaimV2Error(
                "raw target changed before retirement"
            )
        allocated = opened.st_blocks * 512
        before_irreversible()
        if not _atomic_rename_at_no_replace(parent_fd, leaf, anchor):
            raise DojoLegacyCellRawReclaimV2Error("retirement anchor collided")
        os.fsync(parent_fd)
        anchored = os.stat(anchor, dir_fd=parent_fd, follow_symlinks=False)
        bound_after_rename = os.fstat(descriptor)
        if _stable_identity(anchored) != _stable_identity(bound_after_rename):
            # A last-moment replacement may have been moved, but is retained
            # intact under the anchor and is never truncated here.
            raise DojoLegacyCellRawReclaimV2Error(
                "raw target was replaced during retirement"
            )
        if bound_after_rename.st_nlink != 1:
            raise DojoLegacyCellRawReclaimV2Error(
                "raw target has an unexpected hard link"
            )
        before_irreversible()
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
        os.fsync(parent_fd)
        return allocated
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _verify_missing_with_anchor(root_fd: int, row: Mapping[str, Any]) -> None:
    parent_fd, leaf = _open_parent_at(root_fd, str(row["path"]))
    try:
        try:
            os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise DojoLegacyCellRawReclaimV2Error("retired raw path still exists")
        anchor = os.stat(_anchor_name(row), dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(anchor.st_mode) or anchor.st_size != 0:
            raise DojoLegacyCellRawReclaimV2Error("retirement anchor is invalid")
    finally:
        os.close(parent_fd)


def reclaim_generation_2_raw(
    *,
    plan_path: Path,
    attestations_dir: Path,
    expected_plan_sha256: str,
    expected_target_count: int,
    expected_target_bytes: int,
) -> dict[str, Any]:
    """Release generation-2 raw only after exact signed remote authority."""

    plan = load_v2_plan(plan_path)
    if (
        expected_plan_sha256 != plan["reclaim_plan_sha256"]
        or expected_target_count != plan["generation_2_target_count"]
        or expected_target_bytes != plan["generation_2_target_bytes"]
    ):
        raise DojoLegacyCellRawReclaimV2Error(
            "exact plan SHA/count/bytes confirmation does not match"
        )
    root = Path(plan["source_run_root"]).resolve(strict=True)
    with _locked_root(root) as (root_fd, lock_fd):
        _assert_lock_identity(root, root_fd, lock_fd)
        generation_fd = _directory_at(
            root_fd,
            "legacy-cell-reclaim-v2/generation-000002",
            create=False,
        )
        try:
            _assert_unique_plan_at(generation_fd, plan)
        finally:
            os.close(generation_fd)
        receipts = sorted(_generation_directory(plan).glob("reclaim-*.json"))
        if len(receipts) > 1:
            raise DojoLegacyCellRawReclaimV2Error("multiple V2 receipts exist")
        if receipts:
            receipt = _load_json(receipts[0], field="V2 reclaim receipt")
            receipt_body = {
                key: value
                for key, value in receipt.items()
                if key != "reclaim_receipt_sha256"
            }
            if (
                receipt.get("contract") != RECLAIM_RECEIPT_CONTRACT
                or receipt.get("schema_version") != SCHEMA_VERSION
                or receipt.get("status") != "GENERATION_2_RAW_RECLAIMED"
                or receipt.get("plan_sha256") != plan["reclaim_plan_sha256"]
                or receipt.get("deleted_files") != plan["generation_2_targets"]
                or receipt.get("deleted_file_count")
                != plan["generation_2_target_count"]
                or receipt.get("reclaimed_logical_bytes")
                != plan["generation_2_target_bytes"]
                or receipt.get("reclaim_receipt_sha256") != _sha256(receipt_body)
                or receipts[0].name
                != f"reclaim-{receipt.get('reclaim_receipt_sha256')}.json"
                or any(receipt.get(key) != value for key, value in _AUTHORITY.items())
            ):
                raise DojoLegacyCellRawReclaimV2Error("V2 reclaim receipt is invalid")
            for row in plan["generation_2_targets"]:
                _verify_missing_with_anchor(root_fd, row)
            return receipt
        authorization, authorization_path = _load_or_create_authorization(
            plan=plan, attestations_dir=attestations_dir
        )
        for row in plan["cumulative_targets"]:
            if row not in plan["generation_2_targets"]:
                try:
                    legacy._verify_row(root, row, missing_allowed=True)
                except legacy.DojoLegacyCellRawReclaimError as exc:
                    raise DojoLegacyCellRawReclaimV2Error(str(exc)) from exc
                if (root / row["path"]).exists():
                    raise DojoLegacyCellRawReclaimV2Error(
                        "V1 predecessor target unexpectedly exists"
                    )
        for row in plan["shared_terminal_files"]:
            try:
                legacy._verify_row(root, row, missing_allowed=False)
            except legacy.DojoLegacyCellRawReclaimError as exc:
                raise DojoLegacyCellRawReclaimV2Error(str(exc)) from exc
        free_before = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
        allocated = 0
        for row in plan["generation_2_targets"]:
            _assert_lock_identity(root, root_fd, lock_fd)
            allocated += _retire_target(
                root_fd,
                row,
                lambda: _assert_lock_identity(root, root_fd, lock_fd),
            )
        for row in plan["generation_2_targets"]:
            _verify_missing_with_anchor(root_fd, row)
        for row in plan["shared_terminal_files"]:
            try:
                legacy._verify_row(root, row, missing_allowed=False)
            except legacy.DojoLegacyCellRawReclaimError as exc:
                raise DojoLegacyCellRawReclaimV2Error(str(exc)) from exc
        free_after = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
        receipt_body = {
            "contract": RECLAIM_RECEIPT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "status": "GENERATION_2_RAW_RECLAIMED",
            "plan_sha256": plan["reclaim_plan_sha256"],
            "authorization_path": os.fspath(authorization_path),
            "authorization_sha256": authorization["authorization_sha256"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "deleted_file_count": plan["generation_2_target_count"],
            "deleted_files": plan["generation_2_targets"],
            "reclaimed_logical_bytes": plan["generation_2_target_bytes"],
            "reclaimed_allocated_bytes_observed": allocated,
            "free_disk_bytes_before": free_before,
            "free_disk_bytes_after": free_after,
            "zero_byte_retirement_anchors_retained": True,
            **_AUTHORITY,
        }
        receipt = {
            **receipt_body,
            "reclaim_receipt_sha256": _sha256(receipt_body),
        }
        receipt_path = _generation_directory(plan) / (
            f"reclaim-{receipt['reclaim_receipt_sha256']}.json"
        )
        _assert_lock_identity(root, root_fd, lock_fd)
        _write_once_v2(receipt_path, receipt, field="V2 reclaim receipt")
        _assert_lock_identity(root, root_fd, lock_fd)
        return receipt


def _restore_member(
    *,
    root_fd: int,
    row: Mapping[str, Any],
    source: BinaryIO,
    before_publish: Callable[[], None],
) -> str:
    parent_fd, leaf = _open_parent_at(root_fd, str(row["path"]), create=True)
    temporary = f".{leaf}.restore-{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    published = False
    try:
        try:
            existing_fd = os.open(
                leaf,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            existing_fd = None
        if existing_fd is not None:
            try:
                state = os.fstat(existing_fd)
                digest, size = _hash_descriptor(existing_fd)
            finally:
                os.close(existing_fd)
            if (
                not stat.S_ISREG(state.st_mode)
                or digest != row["sha256"]
                or size != row["size_bytes"]
            ):
                raise DojoLegacyCellRawReclaimV2Error(
                    "restore destination exists with different bytes"
                )
            while source.read(HASH_CHUNK_BYTES):
                pass
            return "REUSED_EXACT"
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        digest = hashlib.sha256()
        size = 0
        while chunk := source.read(HASH_CHUNK_BYTES):
            _write_all(descriptor, chunk)
            digest.update(chunk)
            size += len(chunk)
        os.fsync(descriptor)
        if digest.hexdigest() != row["sha256"] or size != row["size_bytes"]:
            raise DojoLegacyCellRawReclaimV2Error(
                "restored member differs from the sealed inventory"
            )
        opened = os.fstat(descriptor)
        named = os.stat(temporary, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stable_identity(opened) != _stable_identity(named)
        ):
            raise DojoLegacyCellRawReclaimV2Error(
                "restore temporary changed before publication"
            )
        before_publish()
        if not _atomic_rename_at_no_replace(parent_fd, temporary, leaf):
            raise DojoLegacyCellRawReclaimV2Error(
                "restore destination appeared concurrently"
            )
        os.fsync(parent_fd)
        published_named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        published_opened = os.fstat(descriptor)
        if _stable_identity(published_named) != _stable_identity(published_opened):
            # A last-moment replacement is retained intact at the destination.
            # No name-based cleanup is safe after this point.
            raise DojoLegacyCellRawReclaimV2Error(
                "restore temporary was replaced during publication"
            )
        published = True
        return "RESTORED"
    finally:
        if descriptor is not None:
            if not published:
                try:
                    named = os.stat(temporary, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    opened = os.fstat(descriptor)
                    if _stable_identity(named) == _stable_identity(opened):
                        before_publish()
                        os.ftruncate(descriptor, 0)
                        os.fsync(descriptor)
                        os.fsync(parent_fd)
            os.close(descriptor)
        os.close(parent_fd)


def _restore_archive_targets(
    *,
    root_fd: int,
    cell: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    zstd_path: str,
    before_publish: Callable[[], None],
) -> tuple[int, int]:
    archive_path = Path(cell["archive_path"])
    archive_fd: int | None = None
    try:
        before = archive_path.stat(follow_symlinks=False)
        archive_fd = os.open(
            archive_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(archive_fd)
        archive_sha, archive_md5, archive_size = _hash_descriptor_sha_md5(archive_fd)
    except OSError as exc:
        if archive_fd is not None:
            os.close(archive_fd)
        raise DojoLegacyCellRawReclaimV2Error("restore archive is unavailable") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or _stable_identity(before) != _stable_identity(opened)
        or archive_sha != cell["archive_sha256"]
        or archive_md5 != cell["archive_md5"]
        or archive_size != cell["archive_size_bytes"]
    ):
        os.close(archive_fd)
        raise DojoLegacyCellRawReclaimV2Error("restore archive identity drifted")
    expected = {f"run/{row['path']}": row for row in rows}
    observed: set[str] = set()
    restored = 0
    reused = 0
    try:
        process = subprocess.Popen(
            [zstd_path, "-q", "-d", "-c"],
            stdin=archive_fd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        os.close(archive_fd)
        raise DojoLegacyCellRawReclaimV2Error("zstd restore could not start") from exc
    try:
        if process.stdout is None:
            raise DojoLegacyCellRawReclaimV2Error("zstd stdout was not created")
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if member.name not in expected:
                    continue
                if member.name in observed or not member.isfile():
                    raise DojoLegacyCellRawReclaimV2Error(
                        "restore archive member is duplicated or unsafe"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise DojoLegacyCellRawReclaimV2Error(
                        "restore archive member cannot be read"
                    )
                before_publish()
                status = _restore_member(
                    root_fd=root_fd,
                    row=expected[member.name],
                    source=extracted,
                    before_publish=before_publish,
                )
                restored += status == "RESTORED"
                reused += status == "REUSED_EXACT"
                observed.add(member.name)
        return_code = process.wait()
        error = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr
            else ""
        )
        if return_code != 0:
            raise DojoLegacyCellRawReclaimV2Error(
                f"zstd restore failed: {error[:1000]}"
            )
        after = os.fstat(archive_fd)
        named_after = archive_path.stat(follow_symlinks=False)
        if _stable_identity(opened) != _stable_identity(after) or _stable_identity(
            after
        ) != _stable_identity(named_after):
            raise DojoLegacyCellRawReclaimV2Error(
                "restore archive changed while streamed"
            )
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait()
        os.close(archive_fd)
        raise
    if observed != set(expected):
        os.close(archive_fd)
        raise DojoLegacyCellRawReclaimV2Error("restore archive members are incomplete")
    os.close(archive_fd)
    return restored, reused


def restore_raw_from_v2_plan(
    *,
    plan_path: Path,
    destination: Path,
    scope: str,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Restore sealed raw atomically, never replacing a destination path."""

    plan = load_v2_plan(plan_path)
    if expected_plan_sha256 != plan["reclaim_plan_sha256"]:
        raise DojoLegacyCellRawReclaimV2Error("restore plan SHA confirmation differs")
    if scope == "prior":
        rows = [
            row
            for row in plan["cumulative_targets"]
            if row not in plan["generation_2_targets"]
        ]
    elif scope == "generation2":
        rows = list(plan["generation_2_targets"])
    elif scope == "all":
        rows = list(plan["cumulative_targets"])
    else:
        raise DojoLegacyCellRawReclaimV2Error("restore scope is invalid")
    destination_path = Path(destination)
    if destination_path.is_symlink():
        raise DojoLegacyCellRawReclaimV2Error("restore destination is a symlink")
    destination_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination_state = destination_path.lstat()
    if not stat.S_ISDIR(destination_state.st_mode):
        raise DojoLegacyCellRawReclaimV2Error("restore destination is not a directory")
    root = destination_path.resolve(strict=True)
    if not root.is_dir():
        raise DojoLegacyCellRawReclaimV2Error("restore destination is unsafe")
    cells = _attestation_cells(plan)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        coordinate = row.get("coordinate_id")
        if coordinate not in cells:
            raise DojoLegacyCellRawReclaimV2Error("restore target owner is invalid")
        grouped[str(coordinate)].append(row)
    restored = 0
    reused = 0
    source_root = Path(plan["source_run_root"]).resolve(strict=True)
    with _locked_roots_canonical([source_root, root]) as locked:
        source_fd, source_lock_fd = locked[source_root]
        root_fd, root_lock_fd = locked[root]

        def assert_dual_locks() -> None:
            _assert_lock_identity(source_root, source_fd, source_lock_fd)
            _assert_lock_identity(root, root_fd, root_lock_fd)

        assert_dual_locks()
        generation_fd = _directory_at(
            source_fd,
            "legacy-cell-reclaim-v2/generation-000002",
            create=False,
        )
        try:
            _assert_unique_plan_at(generation_fd, plan)
        finally:
            os.close(generation_fd)
        for coordinate in sorted(grouped):
            added, retained = _restore_archive_targets(
                root_fd=root_fd,
                cell=cells[coordinate],
                rows=grouped[coordinate],
                zstd_path=plan["zstd"]["invocation_path"],
                before_publish=assert_dual_locks,
            )
            restored += added
            reused += retained
        for row in rows:
            parent_fd, leaf = _open_parent_at(root_fd, str(row["path"]))
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    leaf,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                state = os.fstat(descriptor)
                digest, size = _hash_descriptor(descriptor)
                if (
                    not stat.S_ISREG(state.st_mode)
                    or digest != row["sha256"]
                    or size != row["size_bytes"]
                ):
                    raise DojoLegacyCellRawReclaimV2Error(
                        "restored roundtrip verification failed"
                    )
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                os.close(parent_fd)
        assert_dual_locks()
        receipt_body = {
            "contract": RESTORE_RECEIPT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "status": "RAW_RESTORE_VERIFIED",
            "plan_sha256": plan["reclaim_plan_sha256"],
            "scope": scope,
            "destination_root": os.fspath(root),
            "target_set_sha256": _sha256(rows),
            "target_count": len(rows),
            "target_bytes": sum(row["size_bytes"] for row in rows),
            "restored_file_count": restored,
            "reused_exact_file_count": reused,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "atomic_no_overwrite": True,
            "roundtrip_verified": True,
            **_AUTHORITY,
        }
        receipt = {**receipt_body, "restore_receipt_sha256": _sha256(receipt_body)}
        receipt_fd = _directory_at(
            root_fd, ".dojo-legacy-restore-receipts", create=True
        )
        try:
            _write_once_at(
                receipt_fd,
                f"restore-{receipt['restore_receipt_sha256']}.json",
                receipt,
                field="V2 restore receipt",
            )
        finally:
            os.close(receipt_fd)
        return receipt


__all__ = [
    "ATTESTATION_BODY_CONTRACT",
    "ATTESTATION_CONTRACT",
    "ATTESTATION_KEY_ENROLLMENT_CONTRACT",
    "DojoLegacyCellRawReclaimV2Error",
    "build_v2_candidate_plan",
    "enroll_v2_attestation_public_key",
    "load_v2_plan",
    "publish_v2_plan",
    "reclaim_generation_2_raw",
    "restore_raw_from_v2_plan",
    "verify_signed_attestations",
]

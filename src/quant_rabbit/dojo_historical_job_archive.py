"""Deterministic local archive for one terminal historical DOJO job.

The archive is a storage handoff, not evidence promotion.  It captures the
fixed generation artifacts, the job-local economic evidence, the exact source
slice and the matching execution-state records in one verified ``tar.zst``.
Raw evidence is never deleted here; a separately verified remote receipt is
required before any later reclamation policy may do that.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Final, Iterator

import fcntl


ARCHIVE_RECEIPT_CONTRACT: Final = "QR_DOJO_HISTORICAL_JOB_ARCHIVE_V1"
ARCHIVE_MANIFEST_CONTRACT: Final = "QR_DOJO_HISTORICAL_JOB_ARCHIVE_MANIFEST_V1"
ARCHIVE_READBACK_CONTRACT: Final = "QR_DOJO_ARCHIVE_READBACK_OBJECT_SET_V1"
ARCHIVE_SOURCE_INSPECTION_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_JOB_ARCHIVE_SOURCE_INSPECTION_V1"
)
ARCHIVE_RECOVERY_CAPACITY_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_JOB_ARCHIVE_RECOVERY_CAPACITY_V1"
)
JOB_COMPLETION_CONTRACT: Final = "QR_DOJO_HISTORICAL_TRAIN_JOB_COMPLETION_V1"
SOURCE_FAILURE_CONTRACT: Final = "QR_DOJO_HISTORICAL_SOURCE_FAILURE_V1"
ECONOMIC_JOB_RESULT_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_ECONOMIC_JOB_RESULT_V1"
SOURCE_SLICE_RECEIPT_CONTRACT: Final = "QR_DOJO_SPARSE_MONTH_SOURCE_SLICE_V2"
IMPLEMENTATION_MANIFEST_CONTRACT: Final = "QR_DOJO_IMPLEMENTATION_DIGEST_MANIFEST_V1"
SUCCESS_BUNDLE_KIND: Final = "SUCCESS_ECONOMIC"
FAILED_SOURCE_BUNDLE_KIND: Final = "FAILED_SOURCE"
MAX_JSON_BYTES: Final = 256 * 1024 * 1024
MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024
MAX_FILES: Final = 100_000
HASH_CHUNK_BYTES: Final = 1024 * 1024
ARCHIVE_PART_BYTES: Final = 64 * 1024 * 1024
MAX_ARCHIVE_PARTS: Final = 100_000
_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_FIXED_ARTIFACTS: Final = (
    "candidate-proposals.json",
    "control-manifest.json",
    "plan.json",
    "resource-policy.json",
    "schedule.json",
    "tuned-runtime-seal.json",
    "worker-catalog.json",
    "execution-state/execution-manifest.json",
)
_SEALED_INPUT_IDS: Final = (
    "IMPLEMENTATION_MANIFEST",
    "RUN_CONTROL",
    "SOURCE_MANIFEST",
    "STRATEGY_REGISTRY",
)
_BOUND_ARTIFACT_PATHS: Final = {
    "plan": "plan.json",
    "proposals": "candidate-proposals.json",
    "resource_policy": "resource-policy.json",
    "runtime_seal": "tuned-runtime-seal.json",
    "schedule": "schedule.json",
    "worker_catalog": "worker-catalog.json",
}


class DojoHistoricalJobArchiveError(ValueError):
    """A terminal job cannot be archived without weakening provenance."""


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
        raise DojoHistoricalJobArchiveError("value is not canonical JSON") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _decode_json_object(raw: bytes, *, field: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise DojoHistoricalJobArchiveError(
            f"{field} contains non-finite JSON: {token}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoHistoricalJobArchiveError(
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
    except DojoHistoricalJobArchiveError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoHistoricalJobArchiveError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise DojoHistoricalJobArchiveError(f"{field} must be a JSON object")
    return value


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Strict metadata identity retained for raw-reclaim compatibility."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _content_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    """Content-bearing identity; DriveFS timestamp refreshes are metadata-only."""

    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode), value.st_size


def _lock_inode_identity(value: os.stat_result) -> tuple[int, int, int]:
    """Stable lock authority fields; timestamps are not lock identity."""

    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _read_open_bounded_pass(handle: BinaryIO, *, maximum: int) -> bytes:
    return handle.read(maximum + 1)


def _hash_open_pass(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := handle.read(HASH_CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _stable_regular_bytes(path: Path, *, field: str, maximum: int) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise DojoHistoricalJobArchiveError(
                f"{field} must be a bounded nonempty regular file"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            raw = _read_open_bounded_pass(handle, maximum=maximum)
            opened_after_first = os.fstat(handle.fileno())
            handle.seek(0)
            confirmed_raw = _read_open_bounded_pass(handle, maximum=maximum)
            opened_after_second = os.fstat(handle.fileno())
        after = path.stat(follow_symlinks=False)
    except DojoHistoricalJobArchiveError:
        raise
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(f"{field} is unavailable") from exc
    if (
        _content_identity(before) != _content_identity(opened_after_first)
        or _content_identity(opened_after_first)
        != _content_identity(opened_after_second)
        or _content_identity(opened_after_second) != _content_identity(after)
        or len(raw) != before.st_size
        or raw != confirmed_raw
    ):
        raise DojoHistoricalJobArchiveError(f"{field} changed while read")
    return raw


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    return _decode_json_object(
        _stable_regular_bytes(path, field=field, maximum=MAX_JSON_BYTES),
        field=field,
    )


def _hash_file_with_identity(path: Path) -> tuple[str, int, int, int]:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise DojoHistoricalJobArchiveError(
                f"archive source is not regular: {path}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            digest, size = _hash_open_pass(handle)
            opened_after_first = os.fstat(handle.fileno())
            handle.seek(0)
            confirmed_digest, confirmed_size = _hash_open_pass(handle)
            opened_after_second = os.fstat(handle.fileno())
        after = path.stat(follow_symlinks=False)
    except DojoHistoricalJobArchiveError:
        raise
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            f"archive source is unavailable: {path}"
        ) from exc
    if (
        _content_identity(before) != _content_identity(opened_after_first)
        or _content_identity(opened_after_first)
        != _content_identity(opened_after_second)
        or _content_identity(opened_after_second) != _content_identity(after)
        or size != before.st_size
        or confirmed_size != size
        or confirmed_digest != digest
    ):
        raise DojoHistoricalJobArchiveError(
            f"archive source changed while hashing: {path}"
        )
    return digest, size, after.st_dev, after.st_ino


def _hash_file(path: Path) -> tuple[str, int]:
    sha256, size, _, _ = _hash_file_with_identity(path)
    return sha256, size


def _safe_relative(value: str) -> str:
    if not value or "\\" in value or any(ord(character) < 32 for character in value):
        raise DojoHistoricalJobArchiveError("archive member path is unsafe")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != value
    ):
        raise DojoHistoricalJobArchiveError("archive member path is unsafe")
    return pure.as_posix()


def _regular_source_path(run_root: Path, relative: str) -> Path:
    safe = _safe_relative(relative)
    current = run_root
    try:
        root_state = current.stat(follow_symlinks=False)
        if not stat.S_ISDIR(root_state.st_mode):
            raise DojoHistoricalJobArchiveError("run root must be a real directory")
        parts = PurePosixPath(safe).parts
        for index, part in enumerate(parts):
            current = current / part
            state = current.stat(follow_symlinks=False)
            if stat.S_ISLNK(state.st_mode):
                raise DojoHistoricalJobArchiveError(
                    f"archive source path contains a symlink: {safe}"
                )
            if index < len(parts) - 1 and not stat.S_ISDIR(state.st_mode):
                raise DojoHistoricalJobArchiveError(
                    f"archive source parent is not a directory: {safe}"
                )
            if index == len(parts) - 1 and not stat.S_ISREG(state.st_mode):
                raise DojoHistoricalJobArchiveError(
                    f"archive source is not regular: {safe}"
                )
    except DojoHistoricalJobArchiveError:
        raise
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            f"archive source is unavailable: {safe}"
        ) from exc
    return current


def _completion(run_root: Path, job_sha256: str) -> dict[str, Any]:
    completion = _read_json(
        run_root / "jobs" / job_sha256 / "completion.json",
        field="job completion",
    )
    body = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    if (
        completion.get("contract") != JOB_COMPLETION_CONTRACT
        or completion.get("job_sha256") != job_sha256
        or completion.get("completion_sha256") != _canonical_sha256(body)
        or completion.get("automatic_deployment_allowed") is not False
        or completion.get("broker_mutation_allowed") is not False
        or completion.get("live_permission") is not False
        or completion.get("order_authority") != "NONE"
        or completion.get("promotion_eligible") is not False
    ):
        raise DojoHistoricalJobArchiveError("job completion seal is invalid")
    return completion


def _bundle_kind(completion: Mapping[str, Any]) -> str:
    status = completion.get("job_status")
    economic_sha256 = completion.get("economic_job_result_sha256")
    if status == "FAILED_SOURCE":
        new_failures = completion.get("new_source_failure_coordinate_count")
        predecessor_failures = completion.get("predecessor_failure_coordinate_count")
        complete_count = completion.get("complete_coordinate_count")
        failed_count = completion.get("failed_coordinate_count")
        coordinate_count = completion.get("coordinate_result_count")
        claim_sha256 = completion.get("claim_sha256")
        if (
            economic_sha256 is not None
            or not isinstance(claim_sha256, str)
            or _SHA_RE.fullmatch(claim_sha256) is None
            or isinstance(new_failures, bool)
            or not isinstance(new_failures, int)
            or new_failures < 1
            or isinstance(predecessor_failures, bool)
            or not isinstance(predecessor_failures, int)
            or predecessor_failures < 0
            or isinstance(complete_count, bool)
            or not isinstance(complete_count, int)
            or complete_count < 0
            or isinstance(failed_count, bool)
            or not isinstance(failed_count, int)
            or failed_count != new_failures + predecessor_failures
            or isinstance(coordinate_count, bool)
            or not isinstance(coordinate_count, int)
            or coordinate_count != complete_count + failed_count
        ):
            raise DojoHistoricalJobArchiveError(
                "FAILED_SOURCE completion has an invalid fixed denominator"
            )
        return FAILED_SOURCE_BUNDLE_KIND
    if (
        status not in {"COMPLETE", "INCOMPLETE_FAILED"}
        or not isinstance(economic_sha256, str)
        or _SHA_RE.fullmatch(economic_sha256) is None
    ):
        raise DojoHistoricalJobArchiveError(
            "terminal completion is neither economic success nor source failure"
        )
    return SUCCESS_BUNDLE_KIND


def _safe_authority(value: Mapping[str, Any], *, field: str) -> None:
    if (
        value.get("automatic_deployment_allowed") is not False
        or value.get("broker_mutation_allowed") is not False
        or value.get("live_permission") is not False
        or value.get("order_authority") != "NONE"
        or value.get("promotion_eligible") is not False
    ):
        raise DojoHistoricalJobArchiveError(f"{field} grants unsafe authority")


def _validate_fixed_artifact_bindings(
    *, run_root: Path, control_manifest: Mapping[str, Any]
) -> None:
    bindings = control_manifest.get("artifact_sha256")
    if not isinstance(bindings, Mapping) or set(bindings) != set(_BOUND_ARTIFACT_PATHS):
        raise DojoHistoricalJobArchiveError(
            "control manifest artifact bindings are incomplete"
        )
    for artifact_id, relative in _BOUND_ARTIFACT_PATHS.items():
        expected = bindings.get(artifact_id)
        if not isinstance(expected, str) or _SHA_RE.fullmatch(expected) is None:
            raise DojoHistoricalJobArchiveError(
                "control manifest artifact binding is invalid"
            )
        observed, _ = _hash_file(_regular_source_path(run_root, relative))
        if observed != expected:
            raise DojoHistoricalJobArchiveError(f"prepared {artifact_id} bytes drifted")


def _validate_sealed_inputs(run_root: Path) -> list[str]:
    control_manifest = _read_json(
        run_root / "control-manifest.json", field="control manifest"
    )
    manifest_body = {
        key: value
        for key, value in control_manifest.items()
        if key != "manifest_sha256"
    }
    rows = control_manifest.get("sealed_input_artifacts")
    if (
        control_manifest.get("contract")
        != "QR_DOJO_HISTORICAL_TRAIN_GENERATION_MANIFEST_V1"
        or control_manifest.get("schema_version") != 1
        or control_manifest.get("manifest_sha256") != _canonical_sha256(manifest_body)
        or not isinstance(rows, list)
        or control_manifest.get("sealed_input_artifacts_sha256")
        != _canonical_sha256(rows)
    ):
        raise DojoHistoricalJobArchiveError("control manifest seal is invalid")
    _safe_authority(control_manifest, field="control manifest")
    _validate_fixed_artifact_bindings(
        run_root=run_root, control_manifest=control_manifest
    )
    if [row.get("artifact_id") for row in rows if isinstance(row, Mapping)] != list(
        _SEALED_INPUT_IDS
    ) or len(rows) != len(_SEALED_INPUT_IDS):
        raise DojoHistoricalJobArchiveError(
            "sealed input inventory has the wrong artifact set or order"
        )

    parsed: dict[str, dict[str, Any]] = {}
    paths: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "artifact_id",
            "relative_path",
            "file_sha256",
            "file_size_bytes",
        }:
            raise DojoHistoricalJobArchiveError("sealed input row is malformed")
        artifact_id = row.get("artifact_id")
        relative = row.get("relative_path")
        expected_sha256 = row.get("file_sha256")
        expected_size = row.get("file_size_bytes")
        if (
            artifact_id not in _SEALED_INPUT_IDS
            or not isinstance(relative, str)
            or not isinstance(expected_sha256, str)
            or _SHA_RE.fullmatch(expected_sha256) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or not 0 < expected_size <= MAX_JSON_BYTES
        ):
            raise DojoHistoricalJobArchiveError("sealed input row is invalid")
        safe_relative = _safe_relative(relative)
        if not safe_relative.startswith("sealed-inputs/") or not safe_relative.endswith(
            ".json"
        ):
            raise DojoHistoricalJobArchiveError(
                "sealed input is outside the sealed-inputs directory"
            )
        sealed_path = _regular_source_path(run_root, safe_relative)
        observed_sha256, observed_size = _hash_file(sealed_path)
        if observed_sha256 != expected_sha256 or observed_size != expected_size:
            raise DojoHistoricalJobArchiveError("sealed input bytes drifted")
        parsed[artifact_id] = _read_json(
            sealed_path, field=f"sealed input {artifact_id}"
        )
        paths.append(safe_relative)
    if len(set(paths)) != len(paths):
        raise DojoHistoricalJobArchiveError("sealed input paths are not unique")

    run_control = parsed["RUN_CONTROL"]
    authority = run_control.get("authority")
    if (
        run_control.get("contract") != "QR_DOJO_G2_HISTORICAL_RUN_CONTROL_V1"
        or run_control.get("schema_version") != 1
        or not isinstance(authority, Mapping)
        or authority.get("historical_replay_process_start_allowed") is not True
        or authority.get("research_filesystem_write_allowed") is not True
        or authority.get("broker_mutation_allowed") is not False
        or authority.get("live_permission") is not False
        or authority.get("order_authority") != "NONE"
    ):
        raise DojoHistoricalJobArchiveError("sealed run control is invalid")
    raw_control_sha256 = control_manifest.get("run_control_sha256")
    if (
        not isinstance(raw_control_sha256, str)
        or _SHA_RE.fullmatch(raw_control_sha256) is None
    ):
        raise DojoHistoricalJobArchiveError(
            "control manifest raw run-control binding is invalid"
        )

    source_manifest = parsed["SOURCE_MANIFEST"]
    source_body = {
        key: value
        for key, value in source_manifest.items()
        if key != "source_manifest_sha256"
    }
    source_sha256 = source_manifest.get("source_manifest_sha256")
    if (
        not isinstance(source_sha256, str)
        or _SHA_RE.fullmatch(source_sha256) is None
        or source_sha256 != _canonical_sha256(source_body)
        or source_sha256 != control_manifest.get("source_manifest_sha256")
    ):
        raise DojoHistoricalJobArchiveError("sealed source manifest is invalid")
    source_authority = source_manifest.get("authority")
    if not isinstance(source_authority, Mapping):
        raise DojoHistoricalJobArchiveError(
            "sealed source manifest authority is missing"
        )
    _safe_authority(source_authority, field="sealed source manifest")

    registry = parsed["STRATEGY_REGISTRY"]
    registry_body = {
        key: value for key, value in registry.items() if key != "artifact_sha256"
    }
    registry_sha256 = registry.get("artifact_sha256")
    if (
        not isinstance(registry_sha256, str)
        or _SHA_RE.fullmatch(registry_sha256) is None
        or registry_sha256 != _canonical_sha256(registry_body)
        or registry_sha256 != control_manifest.get("registry_artifact_sha256")
    ):
        raise DojoHistoricalJobArchiveError("sealed strategy registry is invalid")
    registry_authority = registry.get("authority")
    if not isinstance(registry_authority, Mapping):
        raise DojoHistoricalJobArchiveError(
            "sealed strategy registry authority is missing"
        )
    _safe_authority(registry_authority, field="sealed strategy registry")

    implementation = parsed["IMPLEMENTATION_MANIFEST"]
    implementation_body = {
        key: value
        for key, value in implementation.items()
        if key != "implementation_manifest_sha256"
    }
    digests = implementation.get("implementation_digests")
    if (
        implementation.get("contract") != IMPLEMENTATION_MANIFEST_CONTRACT
        or implementation.get("schema_version") != 1
        or not isinstance(digests, Mapping)
        or not digests
        or not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and _SHA_RE.fullmatch(value) is not None
            for key, value in digests.items()
        )
        or implementation.get("implementation_digests_sha256")
        != _canonical_sha256(digests)
        or implementation.get("implementation_manifest_sha256")
        != _canonical_sha256(implementation_body)
    ):
        raise DojoHistoricalJobArchiveError("sealed implementation manifest is invalid")
    _safe_authority(implementation, field="sealed implementation manifest")
    plan = _read_json(run_root / "plan.json", field="plan")
    binding = plan.get("implementation_binding")
    if not isinstance(binding, Mapping) or binding.get("digests") != digests:
        raise DojoHistoricalJobArchiveError(
            "sealed implementation digests do not match the plan"
        )
    return paths


def _collect_regular_tree(
    *, run_root: Path, tree_root: Path, paths: set[str], field: str
) -> int:
    try:
        root_state = tree_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(f"{field} is unavailable") from exc
    if not stat.S_ISDIR(root_state.st_mode):
        raise DojoHistoricalJobArchiveError(f"{field} must be a directory")
    count = 0
    for path in tree_root.rglob("*"):
        try:
            state = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                f"{field} changed while enumerated"
            ) from exc
        if stat.S_ISLNK(state.st_mode):
            raise DojoHistoricalJobArchiveError(f"{field} contains a symlink")
        if stat.S_ISDIR(state.st_mode):
            continue
        if not stat.S_ISREG(state.st_mode):
            raise DojoHistoricalJobArchiveError(f"{field} contains a non-regular file")
        paths.add(path.relative_to(run_root).as_posix())
        count += 1
        if len(paths) > MAX_FILES:
            raise DojoHistoricalJobArchiveError("archive file count is outside bounds")
    return count


def _entry_exists(path: Path) -> bool:
    try:
        path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            f"archive source cannot be inspected: {path}"
        ) from exc
    return True


def _collect_files(
    run_root: Path,
    job_sha256: str,
    *,
    bundle_kind: str,
) -> list[str]:
    paths = set(_FIXED_ARTIFACTS)
    paths.update(_validate_sealed_inputs(run_root))
    transition_root = run_root / "transition-receipts"
    if _entry_exists(transition_root):
        if (
            _collect_regular_tree(
                run_root=run_root,
                tree_root=transition_root,
                paths=paths,
                field="generation transition evidence",
            )
            == 0
        ):
            raise DojoHistoricalJobArchiveError(
                "generation transition evidence is empty"
            )
    job_root = run_root / "jobs" / job_sha256
    if (
        _collect_regular_tree(
            run_root=run_root,
            tree_root=job_root,
            paths=paths,
            field="job evidence",
        )
        == 0
    ):
        raise DojoHistoricalJobArchiveError("job evidence is empty")
    result_path = run_root / "job-results" / f"{job_sha256}.json"
    failure_path = job_root / "source-failure.json"
    source_receipt_path = job_root / "source-slice-receipt.json"
    if bundle_kind == SUCCESS_BUNDLE_KIND:
        if _entry_exists(failure_path):
            raise DojoHistoricalJobArchiveError(
                "economic success bundle contains source-failure evidence"
            )
        try:
            result_state = result_path.stat(follow_symlinks=False)
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                "economic job result is unavailable"
            ) from exc
        if not stat.S_ISREG(result_state.st_mode):
            raise DojoHistoricalJobArchiveError(
                "economic job result must be a regular file"
            )
        paths.add(result_path.relative_to(run_root).as_posix())
    elif bundle_kind == FAILED_SOURCE_BUNDLE_KIND:
        if (
            _entry_exists(result_path)
            or _entry_exists(source_receipt_path)
            or _entry_exists(job_root / "economic-evidence")
        ):
            raise DojoHistoricalJobArchiveError(
                "FAILED_SOURCE bundle mixes economic success artifacts"
            )
        if not _entry_exists(failure_path):
            raise DojoHistoricalJobArchiveError("FAILED_SOURCE evidence is unavailable")
    else:
        raise DojoHistoricalJobArchiveError("archive bundle kind is invalid")
    for section in ("claims", "reducers", "terminals", "cells"):
        section_root = run_root / "execution-state" / section / job_sha256
        if (
            _collect_regular_tree(
                run_root=run_root,
                tree_root=section_root,
                paths=paths,
                field=f"execution {section} evidence",
            )
            == 0
        ):
            raise DojoHistoricalJobArchiveError(
                f"execution {section} evidence is empty"
            )
    if bundle_kind == SUCCESS_BUNDLE_KIND:
        receipt = _read_json(source_receipt_path, field="source receipt")
        receipt_body = {
            key: value
            for key, value in receipt.items()
            if key != "source_slice_receipt_sha256"
        }
        if (
            receipt.get("contract") != SOURCE_SLICE_RECEIPT_CONTRACT
            or receipt.get("schema_version") != 2
            or receipt.get("job_sha256") != job_sha256
            or receipt.get("source_slice_receipt_sha256")
            != _canonical_sha256(receipt_body)
        ):
            raise DojoHistoricalJobArchiveError("source receipt names another job")
        source_relative = receipt.get("relative_path")
        if not isinstance(source_relative, str):
            raise DojoHistoricalJobArchiveError("source receipt path is invalid")
        paths.add(f"source-slices/{_safe_relative(source_relative)}")
    ordered = sorted(_safe_relative(path) for path in paths)
    if not ordered or len(ordered) > MAX_FILES:
        raise DojoHistoricalJobArchiveError("archive file count is outside bounds")
    return ordered


def _inventory(
    run_root: Path,
    job_sha256: str,
    *,
    bundle_kind: str,
) -> tuple[list[dict[str, Any]], int]:
    rows = []
    total = 0
    for relative in _collect_files(
        run_root,
        job_sha256,
        bundle_kind=bundle_kind,
    ):
        path = _regular_source_path(run_root, relative)
        digest, size = _hash_file(path)
        rows.append({"path": relative, "size_bytes": size, "sha256": digest})
        total += size
    return rows, total


def _validate_inventory_links(
    *,
    run_root: Path,
    job_sha256: str,
    completion: Mapping[str, Any],
    rows: list[dict[str, Any]],
    bundle_kind: str,
) -> None:
    by_path = {row["path"]: row for row in rows}
    if bundle_kind == FAILED_SOURCE_BUNDLE_KIND:
        failure_relative = f"jobs/{job_sha256}/source-failure.json"
        failure = _read_json(run_root / failure_relative, field="source failure")
        failure_body = {
            key: value
            for key, value in failure.items()
            if key != "failure_evidence_sha256"
        }
        if (
            failure.get("contract") != SOURCE_FAILURE_CONTRACT
            or failure.get("schema_version") != 1
            or failure.get("job_sha256") != job_sha256
            or failure.get("claim_sha256") != completion.get("claim_sha256")
            or failure.get("failure_evidence_sha256") != _canonical_sha256(failure_body)
            or failure_relative not in by_path
        ):
            raise DojoHistoricalJobArchiveError(
                "source failure evidence does not match completion"
            )
        _safe_authority(failure, field="source failure evidence")
        return
    if bundle_kind != SUCCESS_BUNDLE_KIND:
        raise DojoHistoricalJobArchiveError("archive bundle kind is invalid")
    result_relative = f"job-results/{job_sha256}.json"
    result = _read_json(run_root / result_relative, field="economic job result")
    result_body = {
        key: value
        for key, value in result.items()
        if key != "economic_job_result_sha256"
    }
    if (
        result.get("contract") != ECONOMIC_JOB_RESULT_CONTRACT
        or result.get("schema_version") != 1
        or result.get("job_sha256") != job_sha256
        or result.get("job_status") != completion.get("job_status")
        or result.get("economic_job_result_sha256") != _canonical_sha256(result_body)
        or result.get("economic_job_result_sha256")
        != completion.get("economic_job_result_sha256")
        or result_relative not in by_path
    ):
        raise DojoHistoricalJobArchiveError(
            "economic job result does not match completion"
        )
    source_receipt = _read_json(
        run_root / "jobs" / job_sha256 / "source-slice-receipt.json",
        field="source receipt",
    )
    relative = source_receipt.get("relative_path")
    if not isinstance(relative, str):
        raise DojoHistoricalJobArchiveError("source receipt path is invalid")
    source_relative = f"source-slices/{_safe_relative(relative)}"
    source_row = by_path.get(source_relative)
    if (
        source_row is None
        or source_receipt.get("file_sha256") != source_row["sha256"]
        or source_receipt.get("file_size_bytes") != source_row["size_bytes"]
    ):
        raise DojoHistoricalJobArchiveError(
            "source slice bytes do not match their receipt"
        )


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o444
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


class _VerifiedSource:
    """Stream one inventory member while proving it stayed the same file."""

    def __init__(self, path: Path, *, expected_size: int, expected_sha256: str) -> None:
        self._path = path
        self._expected_size = expected_size
        self._expected_sha256 = expected_sha256
        try:
            self._before = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(self._before.st_mode)
                or self._before.st_size != expected_size
            ):
                raise DojoHistoricalJobArchiveError(
                    f"archive source drifted before streaming: {path}"
                )
            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
            self._handle: BinaryIO = os.fdopen(descriptor, "rb", closefd=True)
            self._opened = os.fstat(self._handle.fileno())
        except DojoHistoricalJobArchiveError:
            raise
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                f"archive source is unavailable: {path}"
            ) from exc
        if _content_identity(self._before) != _content_identity(self._opened):
            self._handle.close()
            raise DojoHistoricalJobArchiveError(
                f"archive source changed while opened: {path}"
            )
        self._digest = hashlib.sha256()
        self._size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        self._digest.update(chunk)
        self._size += len(chunk)
        return chunk

    def verify(self) -> None:
        try:
            opened_after = os.fstat(self._handle.fileno())
            after = self._path.stat(follow_symlinks=False)
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                f"archive source disappeared while streaming: {self._path}"
            ) from exc
        if (
            _content_identity(self._before) != _content_identity(opened_after)
            or _content_identity(opened_after) != _content_identity(after)
            or self._size != self._expected_size
            or self._digest.hexdigest() != self._expected_sha256
        ):
            raise DojoHistoricalJobArchiveError(
                f"archive source changed while streaming: {self._path}"
            )

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> _VerifiedSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _write_archive(
    *,
    output: BinaryIO,
    run_root: Path,
    manifest: Mapping[str, Any],
    zstd_bin: str,
) -> None:
    resolved = shutil.which(zstd_bin)
    if resolved is None:
        raise DojoHistoricalJobArchiveError("zstd executable is unavailable")
    manifest_payload = _canonical_bytes(manifest)
    if len(manifest_payload) > MAX_MANIFEST_BYTES:
        raise DojoHistoricalJobArchiveError("archive manifest exceeds its byte bound")
    with tempfile.TemporaryFile() as process_error:
        process = subprocess.Popen(
            [resolved, "-q", "-3", "-T1", "-c"],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=process_error,
        )
        if process.stdin is None:
            process.kill()
            process.wait()
            raise DojoHistoricalJobArchiveError("zstd stdin is unavailable")
        try:
            with tarfile.open(
                fileobj=process.stdin,
                mode="w|",
                format=tarfile.PAX_FORMAT,
                encoding="utf-8",
                errors="strict",
            ) as archive:
                archive.addfile(
                    _tar_info("MANIFEST.json", len(manifest_payload)),
                    _BytesReader(manifest_payload),
                )
                for row in manifest["files"]:
                    with _VerifiedSource(
                        _regular_source_path(run_root, row["path"]),
                        expected_size=row["size_bytes"],
                        expected_sha256=row["sha256"],
                    ) as source:
                        archive.addfile(
                            _tar_info(f"payload/{row['path']}", row["size_bytes"]),
                            source,
                        )
                        source.verify()
        except Exception as exc:
            if not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            if isinstance(exc, DojoHistoricalJobArchiveError):
                raise
            raise DojoHistoricalJobArchiveError(
                "tar stream construction failed"
            ) from exc
        finally:
            if not process.stdin.closed:
                process.stdin.close()
        return_code = process.wait()
        process_error.seek(0)
        error = process_error.read(4096)
        if return_code != 0:
            raise DojoHistoricalJobArchiveError(
                f"zstd failed: {error.decode('utf-8', 'replace')[:1000]}"
            )
    output.flush()
    os.fsync(output.fileno())


class _BytesReader:
    def __init__(self, value: bytes) -> None:
        self._value = value
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._value) - self._offset
        result = self._value[self._offset : self._offset + size]
        self._offset += len(result)
        return result


def _validate_manifest(
    value: Mapping[str, Any],
    *,
    expected_job_sha256: str | None = None,
    expected_completion_sha256: str | None = None,
    expected_bundle_kind: str | None = None,
) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    files = value.get("files")
    if (
        value.get("contract") != ARCHIVE_MANIFEST_CONTRACT
        or isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != 1
        or not isinstance(files, list)
        or not 0 < len(files) <= MAX_FILES
        or value.get("file_count") != len(files)
        or value.get("bundle_kind")
        not in {SUCCESS_BUNDLE_KIND, FAILED_SOURCE_BUNDLE_KIND}
        or value.get("manifest_sha256") != _canonical_sha256(body)
        or value.get("historical_train_is_proof") is not False
        or value.get("promotion_eligible") is not False
        or value.get("live_permission") is not False
        or value.get("order_authority") != "NONE"
        or value.get("broker_mutation_allowed") is not False
    ):
        raise DojoHistoricalJobArchiveError("archive manifest seal is invalid")
    job_sha256 = value.get("job_sha256")
    completion_sha256 = value.get("completion_sha256")
    if (
        not isinstance(job_sha256, str)
        or _SHA_RE.fullmatch(job_sha256) is None
        or not isinstance(completion_sha256, str)
        or _SHA_RE.fullmatch(completion_sha256) is None
        or (expected_job_sha256 is not None and job_sha256 != expected_job_sha256)
        or (
            expected_completion_sha256 is not None
            and completion_sha256 != expected_completion_sha256
        )
        or (
            expected_bundle_kind is not None
            and value.get("bundle_kind") != expected_bundle_kind
        )
    ):
        raise DojoHistoricalJobArchiveError("archive manifest identity is invalid")
    normalized: list[dict[str, Any]] = []
    total = 0
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise DojoHistoricalJobArchiveError(
                "archive manifest file row is malformed"
            )
        path = row.get("path")
        size = row.get("size_bytes")
        sha256 = row.get("sha256")
        if (
            not isinstance(path, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or _SHA_RE.fullmatch(sha256) is None
        ):
            raise DojoHistoricalJobArchiveError("archive manifest file row is invalid")
        normalized.append(
            {"path": _safe_relative(path), "size_bytes": size, "sha256": sha256}
        )
        total += size
    if (
        normalized != sorted(normalized, key=lambda row: row["path"])
        or len({row["path"] for row in normalized}) != len(normalized)
        or isinstance(value.get("total_source_bytes"), bool)
        or value.get("total_source_bytes") != total
    ):
        raise DojoHistoricalJobArchiveError(
            "archive manifest inventory is not canonical"
        )
    return dict(value)


def _build_archive_manifest(
    *,
    job_sha256: str,
    completion_sha256: str,
    bundle_kind: str,
    files: list[dict[str, Any]],
    total_source_bytes: int,
) -> dict[str, Any]:
    body = {
        "contract": ARCHIVE_MANIFEST_CONTRACT,
        "schema_version": 1,
        "job_sha256": job_sha256,
        "completion_sha256": completion_sha256,
        "bundle_kind": bundle_kind,
        "file_count": len(files),
        "total_source_bytes": total_source_bytes,
        "files": files,
        "historical_train_is_proof": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return _validate_manifest(
        {**body, "manifest_sha256": _canonical_sha256(body)},
        expected_job_sha256=job_sha256,
        expected_completion_sha256=completion_sha256,
        expected_bundle_kind=bundle_kind,
    )


def _read_member_bytes(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    expected_name: str,
    expected_size: int,
) -> bytes:
    if (
        not member.isfile()
        or member.name != expected_name
        or member.size != expected_size
    ):
        raise DojoHistoricalJobArchiveError("archive member header is invalid")
    handle = archive.extractfile(member)
    if handle is None:
        raise DojoHistoricalJobArchiveError("archive member is unreadable")
    payload = handle.read(expected_size + 1)
    if len(payload) != expected_size:
        raise DojoHistoricalJobArchiveError("archive member size drifted")
    return payload


def _verify_archive(
    archive_path: Path,
    *,
    zstd_bin: str,
    manifest: Mapping[str, Any] | None = None,
    expected_job_sha256: str | None = None,
    expected_completion_sha256: str | None = None,
    expected_bundle_kind: str | None = None,
) -> dict[str, Any]:
    resolved = shutil.which(zstd_bin)
    if resolved is None:
        raise DojoHistoricalJobArchiveError("zstd executable is unavailable")
    supplied = (
        _validate_manifest(
            manifest,
            expected_job_sha256=expected_job_sha256,
            expected_completion_sha256=expected_completion_sha256,
            expected_bundle_kind=expected_bundle_kind,
        )
        if manifest is not None
        else None
    )
    try:
        archive_before = archive_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(archive_before.st_mode):
            raise DojoHistoricalJobArchiveError("archive must be a regular file")
        archive_descriptor = os.open(
            archive_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        archive_opened = os.fstat(archive_descriptor)
    except DojoHistoricalJobArchiveError:
        raise
    except OSError as exc:
        raise DojoHistoricalJobArchiveError("archive is unavailable") from exc
    if _content_identity(archive_before) != _content_identity(archive_opened):
        os.close(archive_descriptor)
        raise DojoHistoricalJobArchiveError("archive changed while opened")
    verified: dict[str, Any] | None = None
    tar_error: Exception | None = None
    with tempfile.TemporaryFile() as process_error:
        try:
            process = subprocess.Popen(
                [resolved, "-q", "-d", "-c"],
                stdin=archive_descriptor,
                stdout=subprocess.PIPE,
                stderr=process_error,
            )
        except OSError as exc:
            os.close(archive_descriptor)
            raise DojoHistoricalJobArchiveError(
                "zstd verification process could not start"
            ) from exc
        if process.stdout is None:
            process.kill()
            process.wait()
            os.close(archive_descriptor)
            raise DojoHistoricalJobArchiveError("zstd output is unavailable")
        try:
            with tarfile.open(
                fileobj=process.stdout,
                mode="r|",
                encoding="utf-8",
                errors="strict",
            ) as archive:
                iterator = iter(archive)
                first = next(iterator, None)
                if first is None or first.size > MAX_MANIFEST_BYTES:
                    raise DojoHistoricalJobArchiveError(
                        "archive manifest member is missing or oversized"
                    )
                manifest_payload = _read_member_bytes(
                    archive,
                    first,
                    expected_name="MANIFEST.json",
                    expected_size=first.size,
                )
                parsed = _decode_json_object(manifest_payload, field="archive manifest")
                if manifest_payload != _canonical_bytes(parsed):
                    raise DojoHistoricalJobArchiveError(
                        "archive manifest is not canonical JSON"
                    )
                verified = _validate_manifest(
                    parsed,
                    expected_job_sha256=expected_job_sha256,
                    expected_completion_sha256=expected_completion_sha256,
                    expected_bundle_kind=expected_bundle_kind,
                )
                if supplied is not None and verified != supplied:
                    raise DojoHistoricalJobArchiveError(
                        "archive embeds another manifest"
                    )
                for row in verified["files"]:
                    member = next(iterator, None)
                    if member is None:
                        raise DojoHistoricalJobArchiveError(
                            "archive payload ended before its inventory"
                        )
                    expected_name = f"payload/{row['path']}"
                    if (
                        not member.isfile()
                        or member.name != expected_name
                        or member.size != row["size_bytes"]
                    ):
                        raise DojoHistoricalJobArchiveError(
                            "archive payload member header mismatch"
                        )
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise DojoHistoricalJobArchiveError(
                            "archive payload member is unreadable"
                        )
                    digest = hashlib.sha256()
                    size = 0
                    while chunk := handle.read(HASH_CHUNK_BYTES):
                        digest.update(chunk)
                        size += len(chunk)
                    if size != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
                        raise DojoHistoricalJobArchiveError(
                            "archive payload inventory mismatch"
                        )
                if next(iterator, None) is not None:
                    raise DojoHistoricalJobArchiveError(
                        "archive contains an unmanifested member"
                    )
        except Exception as exc:
            tar_error = exc
        finally:
            if not process.stdout.closed:
                process.stdout.close()
        return_code = process.wait()
        try:
            archive_opened_after = os.fstat(archive_descriptor)
            archive_after = archive_path.stat(follow_symlinks=False)
        except OSError as exc:
            os.close(archive_descriptor)
            raise DojoHistoricalJobArchiveError(
                "archive disappeared while verified"
            ) from exc
        os.close(archive_descriptor)
        if _content_identity(archive_before) != _content_identity(
            archive_opened_after
        ) or _content_identity(archive_opened_after) != _content_identity(
            archive_after
        ):
            raise DojoHistoricalJobArchiveError("archive changed while verified")
        process_error.seek(0)
        error = process_error.read(4096)
        diagnostic = error.decode("utf-8", "replace")[:1000].strip()
        if tar_error is not None:
            if isinstance(tar_error, DojoHistoricalJobArchiveError):
                message = str(tar_error)
            else:
                message = f"archive tar stream is invalid: {tar_error}"
            raise DojoHistoricalJobArchiveError(
                f"{message}; zstd return code {return_code}; "
                f"zstd stderr: {diagnostic or '<empty>'}"
            ) from tar_error
        if return_code != 0:
            raise DojoHistoricalJobArchiveError(
                f"zstd verification failed with return code {return_code}: "
                f"{diagnostic or '<empty>'}"
            )
    if verified is None:
        raise DojoHistoricalJobArchiveError("archive manifest was not verified")
    return verified


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        state = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            f"archive directory is unavailable: {path}"
        ) from exc
    if not stat.S_ISDIR(state.st_mode):
        raise DojoHistoricalJobArchiveError(
            f"archive directory is not a real directory: {path}"
        )


def _copy_regular_file_to_output(
    *,
    source_path: Path,
    output: BinaryIO,
    expected_sha256: str,
    expected_size_bytes: int,
    field: str,
) -> None:
    """Copy one stable regular file and fsync the receiving file descriptor."""

    try:
        before = source_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise DojoHistoricalJobArchiveError(f"{field} source is not regular")
        descriptor = os.open(
            source_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
    except DojoHistoricalJobArchiveError:
        raise
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(f"{field} source is unavailable") from exc
    if _content_identity(before) != _content_identity(opened):
        os.close(descriptor)
        raise DojoHistoricalJobArchiveError(f"{field} source changed while opened")
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            while chunk := source.read(HASH_CHUNK_BYTES):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            opened_after = os.fstat(source.fileno())
        after = source_path.stat(follow_symlinks=False)
        output.flush()
        os.fsync(output.fileno())
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(f"{field} copy failed") from exc
    if (
        _content_identity(before) != _content_identity(opened_after)
        or _content_identity(opened_after) != _content_identity(after)
        or size != expected_size_bytes
        or digest.hexdigest() != expected_sha256
    ):
        raise DojoHistoricalJobArchiveError(f"{field} source drifted while copied")


def _retirement_anchor_name(path: Path, *, device: int, inode: int) -> str:
    leaf_digest = hashlib.sha256(os.fsencode(path.name)).hexdigest()[:16]
    return (
        f".retired-{leaf_digest}-{device:x}-{inode:x}-" f"{secrets.token_hex(8)}.anchor"
    )


def _quarantine_pending_without_release(path: Path, *, field: str) -> None:
    """Remove a name from active use without deleting or truncating its inode."""

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(path.parent, directory_flags)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            f"{field} pending parent is unavailable"
        ) from exc
    try:
        try:
            current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                f"{field} pending is unavailable"
            ) from exc
        anchor_name = _retirement_anchor_name(
            path,
            device=current.st_dev,
            inode=current.st_ino,
        )
        try:
            published = _atomic_rename_at_no_replace(
                directory_fd,
                path.name,
                anchor_name,
            )
        except FileNotFoundError:
            return
        if not published:
            raise DojoHistoricalJobArchiveError(f"{field} retirement anchor collided")
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _unlink_if_same_inode(path: Path, *, device: int, inode: int) -> None:
    """Retire and release only the caller-bound inode, never a replacement."""

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(path.parent, directory_flags)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            f"incomplete archive output parent is unavailable: {path.parent}"
        ) from exc
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                path.name,
                os.O_RDWR
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
            opened = os.fstat(descriptor)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                f"incomplete archive output could not be opened: {path}"
            ) from exc
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != device
            or opened.st_ino != inode
        ):
            raise DojoHistoricalJobArchiveError(
                f"incomplete archive output was replaced concurrently: {path}"
            )
        anchor_name = _retirement_anchor_name(path, device=device, inode=inode)
        try:
            published = _atomic_rename_at_no_replace(
                directory_fd,
                path.name,
                anchor_name,
            )
        except FileNotFoundError:
            return
        if not published:
            raise DojoHistoricalJobArchiveError(
                "incomplete archive retirement anchor collided"
            )
        os.fsync(directory_fd)
        anchored = os.stat(
            anchor_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(anchored.st_mode)
            or anchored.st_dev != device
            or anchored.st_ino != inode
        ):
            # The atomic rename captured a last-moment replacement.  Keep all
            # of its bytes under the durable anchor and never unlink/truncate it.
            os.fsync(directory_fd)
            raise DojoHistoricalJobArchiveError(
                f"incomplete archive output was replaced concurrently: {path}"
            )
        before_release = os.fstat(descriptor)
        if before_release.st_nlink != 1:
            raise DojoHistoricalJobArchiveError(
                f"incomplete archive output has an unexpected hard link: {path}"
            )
        # Truncation is descriptor-bound: even if the anchor is replaced now,
        # no same-name replacement can be deleted or modified.  The zero-byte
        # anchor is retained as durable audit evidence and consumes no payload
        # blocks.
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _remove_exact_pending(
    path: Path,
    *,
    field: str,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> None:
    if (expected_sha256 is None) != (expected_size_bytes is None):
        raise DojoHistoricalJobArchiveError(
            f"{field} pending cleanup expectation is incomplete"
        )
    if expected_sha256 is None:
        _quarantine_pending_without_release(path, field=field)
        return
    try:
        path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(f"{field} pending is unavailable") from exc
    try:
        observed_sha256, observed_size, device, inode = _hash_file_with_identity(path)
    except DojoHistoricalJobArchiveError:
        try:
            path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return
        raise
    if observed_sha256 != expected_sha256 or observed_size != expected_size_bytes:
        _quarantine_pending_without_release(path, field=field)
        return
    _unlink_if_same_inode(path, device=device, inode=inode)


def _pending_hash_if_exists(
    path: Path,
    *,
    field: str,
) -> tuple[str, int] | None:
    try:
        state = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(f"{field} pending is unavailable") from exc
    if not stat.S_ISREG(state.st_mode):
        raise DojoHistoricalJobArchiveError(f"{field} pending is not regular")
    return _hash_file(path)


def _atomic_rename_at_no_replace(
    directory_fd: int,
    source_name: str,
    destination_name: str,
) -> bool:
    """Atomically rename two leaves bound to one already-open directory."""

    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError as exc:
            raise DojoHistoricalJobArchiveError(
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
            raise DojoHistoricalJobArchiveError(
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
        raise DojoHistoricalJobArchiveError("atomic no-replace rename is unsupported")
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
        raise DojoHistoricalJobArchiveError(
            "atomic no-replace rename is unsupported by the archive filesystem"
        )
    raise DojoHistoricalJobArchiveError(
        f"atomic no-replace rename failed: {os.strerror(error_number)}"
    )


def _atomic_rename_no_replace(source_path: Path, destination_path: Path) -> bool:
    """Atomically publish within one directory, or fail closed if unsupported."""

    if source_path.parent != destination_path.parent:
        raise DojoHistoricalJobArchiveError(
            "atomic archive publication requires one directory"
        )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(source_path.parent, directory_flags)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            "atomic archive publication directory is unavailable"
        ) from exc
    try:
        return _atomic_rename_at_no_replace(
            directory_fd,
            source_path.name,
            destination_path.name,
        )
    finally:
        os.close(directory_fd)


def _prepare_checked_copy_pending(
    *,
    source_path: Path,
    pending_path: Path,
    expected_sha256: str,
    expected_size_bytes: int,
    field: str,
    mutation_guard: Callable[[], None] | None = None,
) -> Path:
    observed = _pending_hash_if_exists(pending_path, field=field)
    if observed is not None:
        pending_sha256, pending_size = observed
        if pending_sha256 == expected_sha256 and pending_size == expected_size_bytes:
            return pending_path
        if mutation_guard is not None:
            mutation_guard()
        _remove_exact_pending(pending_path, field=field)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if mutation_guard is not None:
        mutation_guard()
    try:
        descriptor = os.open(pending_path, flags, 0o600)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            f"{field} pending cannot be created"
        ) from exc
    created = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            _copy_regular_file_to_output(
                source_path=source_path,
                output=output,
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
                field=field,
            )
        pending_sha256, pending_size = _hash_file(pending_path)
        if pending_sha256 != expected_sha256 or pending_size != expected_size_bytes:
            raise DojoHistoricalJobArchiveError(f"{field} pending bytes drifted")
        _fsync_directory(pending_path.parent)
    except Exception:
        _unlink_if_same_inode(
            pending_path,
            device=created.st_dev,
            inode=created.st_ino,
        )
        raise
    return pending_path


def _publish_prepared_pending(
    *,
    pending_path: Path,
    destination_path: Path,
    expected_sha256: str,
    expected_size_bytes: int,
    field: str,
    mutation_guard: Callable[[], None] | None = None,
) -> bool:
    try:
        pending_state = pending_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(f"{field} pending is unavailable") from exc
    if (
        not stat.S_ISREG(pending_state.st_mode)
        or pending_state.st_size != expected_size_bytes
    ):
        raise DojoHistoricalJobArchiveError(f"{field} pending is not verified")
    if mutation_guard is not None:
        mutation_guard()
    published = _atomic_rename_no_replace(pending_path, destination_path)
    if not published:
        observed_sha256, observed_size = _hash_file(destination_path)
        if observed_sha256 != expected_sha256 or observed_size != expected_size_bytes:
            raise DojoHistoricalJobArchiveError(f"existing {field} drifted")
        if mutation_guard is not None:
            mutation_guard()
        _remove_exact_pending(
            pending_path,
            field=field,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
        return False
    _fsync_directory(destination_path.parent)
    try:
        published_state = destination_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            f"published {field} is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(published_state.st_mode)
        or published_state.st_dev != pending_state.st_dev
        or published_state.st_ino != pending_state.st_ino
        or published_state.st_size != expected_size_bytes
    ):
        raise DojoHistoricalJobArchiveError(f"published {field} bytes drifted")
    return True


def _prepare_slice_pending(
    *,
    source_path: Path,
    offset_bytes: int,
    pending_path: Path,
    expected_sha256: str,
    expected_size_bytes: int,
) -> Path:
    observed = _pending_hash_if_exists(
        pending_path,
        field="remote readback part",
    )
    if observed is not None:
        pending_sha256, pending_size = observed
        if pending_sha256 == expected_sha256 and pending_size == expected_size_bytes:
            return pending_path
        _remove_exact_pending(pending_path, field="remote readback part")
    try:
        before = source_path.stat(follow_symlinks=False)
        source_descriptor = os.open(
            source_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(source_descriptor)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            "archive source is unavailable while preparing readback part"
        ) from exc
    if not stat.S_ISREG(before.st_mode) or _content_identity(
        before
    ) != _content_identity(opened):
        os.close(source_descriptor)
        raise DojoHistoricalJobArchiveError(
            "archive source drifted before preparing readback part"
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        pending_descriptor = os.open(pending_path, flags, 0o600)
    except OSError as exc:
        os.close(source_descriptor)
        raise DojoHistoricalJobArchiveError(
            "remote readback part pending cannot be created"
        ) from exc
    digest = hashlib.sha256()
    copied = 0
    try:
        with os.fdopen(source_descriptor, "rb", closefd=True) as source, os.fdopen(
            pending_descriptor, "wb", closefd=True
        ) as output:
            source.seek(offset_bytes)
            while copied < expected_size_bytes:
                chunk = source.read(min(HASH_CHUNK_BYTES, expected_size_bytes - copied))
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            opened_after = os.fstat(source.fileno())
            output.flush()
            os.fsync(output.fileno())
        after = source_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            "remote readback part pending copy failed"
        ) from exc
    if (
        _content_identity(before) != _content_identity(opened_after)
        or _content_identity(opened_after) != _content_identity(after)
        or copied != expected_size_bytes
        or digest.hexdigest() != expected_sha256
    ):
        raise DojoHistoricalJobArchiveError(
            "remote readback part pending source drifted"
        )
    pending_sha256, pending_size = _hash_file(pending_path)
    if pending_sha256 != expected_sha256 or pending_size != expected_size_bytes:
        raise DojoHistoricalJobArchiveError(
            "remote readback part pending bytes drifted"
        )
    _fsync_directory(pending_path.parent)
    return pending_path


def _build_remote_readback_objects(
    *,
    destination: Path,
    archive_path: Path,
    archive_sha256: str,
    archive_size_bytes: int,
    stem: str,
    source_archive_path: Path | None = None,
    mutation_guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    readback_root = destination / "readback-objects"
    if mutation_guard is not None:
        mutation_guard()
    _ensure_directory(readback_root)
    source_path = source_archive_path or archive_path
    try:
        before = source_path.stat(follow_symlinks=False)
        descriptor = os.open(
            source_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            "archive is unavailable while split for remote readback"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or _content_identity(before) != _content_identity(opened)
        or before.st_size != archive_size_bytes
    ):
        os.close(descriptor)
        raise DojoHistoricalJobArchiveError(
            "archive drifted before remote readback split"
        )
    objects: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    combined_size = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            while combined_size < archive_size_bytes:
                if len(objects) >= MAX_ARCHIVE_PARTS:
                    raise DojoHistoricalJobArchiveError(
                        "archive remote readback part count exceeds its bound"
                    )
                index = len(objects)
                offset = combined_size
                part_digest = hashlib.sha256()
                part_size = 0
                while part_size < ARCHIVE_PART_BYTES:
                    chunk = source.read(
                        min(HASH_CHUNK_BYTES, ARCHIVE_PART_BYTES - part_size)
                    )
                    if not chunk:
                        break
                    part_digest.update(chunk)
                    combined.update(chunk)
                    part_size += len(chunk)
                    combined_size += len(chunk)
                if part_size <= 0:
                    raise DojoHistoricalJobArchiveError(
                        "archive ended before its declared size"
                    )
                digest = part_digest.hexdigest()
                final = readback_root / (f"{stem}-part-{index:05d}-{digest}.bin")
                pending = readback_root / f".{stem}-part-{index:05d}.pending"
                try:
                    final_sha256, final_size = _hash_file(final)
                except DojoHistoricalJobArchiveError:
                    try:
                        final.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        final_exists = False
                    else:
                        raise
                else:
                    final_exists = True
                    if final_sha256 != digest or final_size != part_size:
                        raise DojoHistoricalJobArchiveError(
                            "existing remote readback part drifted"
                        )
                if mutation_guard is not None:
                    mutation_guard()
                if final_exists:
                    _remove_exact_pending(
                        pending,
                        field="remote readback part",
                        expected_sha256=digest,
                        expected_size_bytes=part_size,
                    )
                else:
                    _prepare_slice_pending(
                        source_path=source_path,
                        offset_bytes=offset,
                        pending_path=pending,
                        expected_sha256=digest,
                        expected_size_bytes=part_size,
                    )
                    _publish_prepared_pending(
                        pending_path=pending,
                        destination_path=final,
                        expected_sha256=digest,
                        expected_size_bytes=part_size,
                        field="remote readback part",
                        mutation_guard=mutation_guard,
                    )
                objects.append(
                    {
                        "index": index,
                        "offset_bytes": offset,
                        "relative_path": final.relative_to(destination).as_posix(),
                        "size_bytes": part_size,
                        "sha256": digest,
                    }
                )
            opened_after = os.fstat(source.fileno())
        after = source_path.stat(follow_symlinks=False)
    except DojoHistoricalJobArchiveError:
        raise
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            "archive changed during remote readback split"
        ) from exc
    if (
        _content_identity(before) != _content_identity(opened_after)
        or _content_identity(opened_after) != _content_identity(after)
        or combined_size != archive_size_bytes
        or combined.hexdigest() != archive_sha256
    ):
        raise DojoHistoricalJobArchiveError(
            "remote readback parts do not reconstruct the archive"
        )
    _fsync_directory(readback_root)
    expected_names = {Path(row["relative_path"]).name for row in objects}
    existing_names = {path.name for path in readback_root.glob(f"{stem}-part-*.bin")}
    if existing_names != expected_names:
        raise DojoHistoricalJobArchiveError(
            "remote readback object directory contains a same-job orphan"
        )
    body = {
        "contract": ARCHIVE_READBACK_CONTRACT,
        "schema_version": 1,
        "object_size_limit_bytes": ARCHIVE_PART_BYTES,
        "object_count": len(objects),
        "total_size_bytes": archive_size_bytes,
        "concatenated_sha256": archive_sha256,
        "objects": objects,
    }
    return {**body, "object_set_sha256": _canonical_sha256(body)}


def _verify_remote_readback_objects(
    value: Any,
    *,
    destination: Path,
    archive_path: Path,
    archive_sha256: str,
    archive_size_bytes: int,
    stem: str,
) -> None:
    if not isinstance(value, Mapping):
        raise DojoHistoricalJobArchiveError("remote readback object set is missing")
    body = {key: item for key, item in value.items() if key != "object_set_sha256"}
    objects = value.get("objects")
    limit = value.get("object_size_limit_bytes")
    if (
        value.get("contract") != ARCHIVE_READBACK_CONTRACT
        or isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != 1
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 0 < limit <= ARCHIVE_PART_BYTES
        or not isinstance(objects, list)
        or not 0 < len(objects) <= MAX_ARCHIVE_PARTS
        or isinstance(value.get("object_count"), bool)
        or value.get("object_count") != len(objects)
        or isinstance(value.get("total_size_bytes"), bool)
        or value.get("total_size_bytes") != archive_size_bytes
        or not isinstance(value.get("concatenated_sha256"), str)
        or value.get("concatenated_sha256") != archive_sha256
        or value.get("object_set_sha256") != _canonical_sha256(body)
    ):
        raise DojoHistoricalJobArchiveError("remote readback object set is invalid")
    total = 0
    combined = hashlib.sha256()
    layout: str | None = None
    for index, row in enumerate(objects):
        if not isinstance(row, Mapping) or set(row) != {
            "index",
            "offset_bytes",
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise DojoHistoricalJobArchiveError(
                "remote readback object row is malformed"
            )
        relative = row.get("relative_path")
        size = row.get("size_bytes")
        sha256 = row.get("sha256")
        if (
            isinstance(row.get("index"), bool)
            or row.get("index") != index
            or isinstance(row.get("offset_bytes"), bool)
            or row.get("offset_bytes") != total
            or not isinstance(relative, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= limit
            or not isinstance(sha256, str)
            or _SHA_RE.fullmatch(sha256) is None
        ):
            raise DojoHistoricalJobArchiveError("remote readback object row is invalid")
        safe_relative = _safe_relative(relative)
        expected_small = archive_path.relative_to(destination).as_posix()
        expected_current = f"readback-objects/{stem}-part-{index:05d}-{sha256}.bin"
        expected_legacy = f"parts/{stem}/part-{index:05d}-{sha256}.bin"
        is_current = safe_relative == expected_current
        is_legacy = safe_relative == (
            expected_small if len(objects) == 1 else expected_legacy
        )
        if not is_current and not is_legacy:
            raise DojoHistoricalJobArchiveError(
                "remote readback object path is outside its part set"
            )
        row_layout = "CURRENT" if is_current else "LEGACY"
        if layout is None:
            layout = row_layout
        elif layout != row_layout:
            raise DojoHistoricalJobArchiveError(
                "remote readback object layouts are mixed"
            )
        path = destination / safe_relative
        try:
            before = path.stat(follow_symlinks=False)
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            digest = hashlib.sha256()
            observed_size = 0
            with os.fdopen(descriptor, "rb", closefd=True) as source:
                while chunk := source.read(HASH_CHUNK_BYTES):
                    digest.update(chunk)
                    combined.update(chunk)
                    observed_size += len(chunk)
                opened_after = os.fstat(source.fileno())
            after = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                "remote readback object is unavailable"
            ) from exc
        if (
            not stat.S_ISREG(before.st_mode)
            or _content_identity(before) != _content_identity(opened)
            or _content_identity(opened) != _content_identity(opened_after)
            or _content_identity(opened_after) != _content_identity(after)
            or observed_size != size
            or digest.hexdigest() != sha256
        ):
            raise DojoHistoricalJobArchiveError("remote readback object bytes drifted")
        total += observed_size
    if total != archive_size_bytes or combined.hexdigest() != archive_sha256:
        raise DojoHistoricalJobArchiveError(
            "remote readback objects do not reconstruct the archive"
        )


def _write_once(
    path: Path,
    value: Mapping[str, Any],
    *,
    mutation_guard: Callable[[], None] | None = None,
) -> None:
    payload = _canonical_bytes(value) + b"\n"
    try:
        current = _stable_regular_bytes(
            path, field="archive receipt", maximum=MAX_JSON_BYTES
        )
    except DojoHistoricalJobArchiveError:
        try:
            exists = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            exists = None
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                "archive receipt is unavailable"
            ) from exc
        if exists is not None:
            raise
    else:
        if current != payload:
            raise DojoHistoricalJobArchiveError("archive receipt already drifted")
        return
    pending = path.parent / f".{path.name}.pending"
    try:
        pending_state = pending.stat(follow_symlinks=False)
    except FileNotFoundError:
        pending_state = None
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            "archive receipt pending is unavailable"
        ) from exc
    else:
        if not stat.S_ISREG(pending_state.st_mode):
            raise DojoHistoricalJobArchiveError(
                "archive receipt pending is not regular"
            )
        if not 0 < pending_state.st_size <= MAX_JSON_BYTES:
            if mutation_guard is not None:
                mutation_guard()
            _remove_exact_pending(pending, field="archive receipt")
        else:
            current_pending = _stable_regular_bytes(
                pending,
                field="archive receipt pending",
                maximum=MAX_JSON_BYTES,
            )
            if current_pending != payload:
                if mutation_guard is not None:
                    mutation_guard()
                _remove_exact_pending(pending, field="archive receipt")
            else:
                if mutation_guard is not None:
                    mutation_guard()
                _publish_prepared_pending(
                    pending_path=pending,
                    destination_path=path,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    expected_size_bytes=len(payload),
                    field="archive receipt",
                    mutation_guard=mutation_guard,
                )
                current = _stable_regular_bytes(
                    path, field="archive receipt", maximum=MAX_JSON_BYTES
                )
                if current != payload:
                    raise DojoHistoricalJobArchiveError(
                        "archive receipt concurrently drifted"
                    )
                return
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if mutation_guard is not None:
        mutation_guard()
    try:
        descriptor = os.open(pending, flags, 0o600)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            "archive receipt pending cannot be created"
        ) from exc
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    pending_payload = _stable_regular_bytes(
        pending,
        field="archive receipt pending",
        maximum=MAX_JSON_BYTES,
    )
    if pending_payload != payload:
        raise DojoHistoricalJobArchiveError("archive receipt pending bytes drifted")
    _fsync_directory(path.parent)
    if mutation_guard is not None:
        mutation_guard()
    _publish_prepared_pending(
        pending_path=pending,
        destination_path=path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
        field="archive receipt",
        mutation_guard=mutation_guard,
    )
    current = _stable_regular_bytes(
        path, field="archive receipt", maximum=MAX_JSON_BYTES
    )
    if current != payload:
        raise DojoHistoricalJobArchiveError("archive receipt concurrently drifted")


@contextmanager
def _job_archive_lock(
    receipts: Path,
    job_sha256: str,
) -> Iterator[Callable[[], None]]:
    try:
        canonical_receipts = receipts.resolve(strict=True)
        named_directory = canonical_receipts.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            "archive lock directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(named_directory.st_mode):
        raise DojoHistoricalJobArchiveError(
            "archive lock directory must be a real directory"
        )
    expected_directory_identity = (
        named_directory.st_dev,
        named_directory.st_ino,
    )
    lock_name = f".job-{job_sha256}.lock"
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(canonical_receipts, directory_flags)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            "archive lock directory is unavailable"
        ) from exc
    descriptor: int | None = None
    locked = False
    try:

        def assert_directory_current() -> None:
            try:
                opened_directory = os.fstat(directory_fd)
                current_named_directory = canonical_receipts.stat(follow_symlinks=False)
            except OSError as exc:
                raise DojoHistoricalJobArchiveError(
                    "archive lock directory pathname is unavailable"
                ) from exc
            if (
                not stat.S_ISDIR(opened_directory.st_mode)
                or not stat.S_ISDIR(current_named_directory.st_mode)
                or opened_directory.st_nlink < 1
                or current_named_directory.st_nlink < 1
                or (
                    opened_directory.st_dev,
                    opened_directory.st_ino,
                )
                != expected_directory_identity
                or (
                    current_named_directory.st_dev,
                    current_named_directory.st_ino,
                )
                != expected_directory_identity
            ):
                raise DojoHistoricalJobArchiveError(
                    "archive lock directory was replaced concurrently"
                )

        # Opening through the canonical pathname is not sufficient: an
        # attacker can rename that directory and recreate the same path before
        # the per-job lock is opened.  Bind both the descriptor and the named
        # pathname to the directory inode captured above.
        assert_directory_current()
        try:
            for attempt in range(2):
                try:
                    descriptor = os.open(
                        lock_name,
                        flags,
                        0o600,
                        dir_fd=directory_fd,
                    )
                except FileNotFoundError:
                    if attempt == 0:
                        continue
                    raise
                else:
                    break
            if descriptor is None:
                raise OSError("archive lock descriptor was not opened")
            state = os.fstat(descriptor)
        except OSError as exc:
            raise DojoHistoricalJobArchiveError("archive lock is unavailable") from exc
        if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
            raise DojoHistoricalJobArchiveError(
                "archive lock must be a singly linked regular file"
            )
        expected_identity = _lock_inode_identity(state)

        def assert_current() -> None:
            assert_directory_current()
            try:
                opened = os.fstat(descriptor)
                named = os.stat(
                    lock_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise DojoHistoricalJobArchiveError(
                    "archive lock pathname is unavailable"
                ) from exc
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(named.st_mode)
                or opened.st_nlink != 1
                or named.st_nlink != 1
                or _lock_inode_identity(opened) != expected_identity
                or _lock_inode_identity(named) != expected_identity
            ):
                raise DojoHistoricalJobArchiveError(
                    "archive lock pathname was replaced concurrently"
                )

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DojoHistoricalJobArchiveError("archive lock is already held") from exc
        locked = True
        assert_current()
        try:
            yield assert_current
        except BaseException as active_error:
            try:
                assert_current()
            except DojoHistoricalJobArchiveError as lock_error:
                add_note = getattr(active_error, "add_note", None)
                if callable(add_note):
                    add_note(f"archive lock release check failed: {lock_error}")
            raise
        else:
            assert_current()
    finally:
        try:
            if descriptor is not None and locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory_fd)


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    job_sha256: str,
    completion_sha256: str,
    bundle_kind: str,
    archives: Path,
) -> tuple[Path, str]:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    manifest_sha256 = receipt.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or _SHA_RE.fullmatch(manifest_sha256) is None
    ):
        raise DojoHistoricalJobArchiveError("archive receipt manifest SHA is invalid")
    expected_archive = archives / f"job-{job_sha256}-{manifest_sha256}.tar.zst"
    archive_path = receipt.get("archive_path")
    remote = receipt.get("remote_verification")
    if (
        receipt.get("contract") != ARCHIVE_RECEIPT_CONTRACT
        or isinstance(receipt.get("schema_version"), bool)
        or receipt.get("schema_version") != 1
        or receipt.get("job_sha256") != job_sha256
        or receipt.get("completion_sha256") != completion_sha256
        or receipt.get("bundle_kind") != bundle_kind
        or receipt.get("receipt_sha256") != _canonical_sha256(body)
        or not isinstance(archive_path, str)
        or Path(archive_path) != expected_archive
        or not isinstance(receipt.get("archive_sha256"), str)
        or _SHA_RE.fullmatch(receipt["archive_sha256"]) is None
        or isinstance(receipt.get("archive_size_bytes"), bool)
        or not isinstance(receipt.get("archive_size_bytes"), int)
        or receipt["archive_size_bytes"] <= 0
        or isinstance(receipt.get("file_count"), bool)
        or not isinstance(receipt.get("file_count"), int)
        or receipt["file_count"] <= 0
        or isinstance(receipt.get("total_source_bytes"), bool)
        or not isinstance(receipt.get("total_source_bytes"), int)
        or receipt["total_source_bytes"] <= 0
        or receipt.get("local_payload_verified") is not True
        or not isinstance(receipt.get("remote_readback_objects"), Mapping)
        or receipt.get("source_deletion_allowed") is not False
        or receipt.get("source_deleted") is not False
        or remote
        != {
            "status": "LOCAL_DRIVE_SYNC_PENDING",
            "remote_verified": False,
            "metadata_receipt_sha256": None,
        }
        or receipt.get("historical_train_is_proof") is not False
        or receipt.get("promotion_eligible") is not False
        or receipt.get("live_permission") is not False
        or receipt.get("order_authority") != "NONE"
        or receipt.get("broker_mutation_allowed") is not False
    ):
        raise DojoHistoricalJobArchiveError("existing archive receipt is invalid")
    return expected_archive, manifest_sha256


def _require_real_directory(path: Path, *, field: str) -> None:
    try:
        state = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(f"{field} is unavailable") from exc
    if not stat.S_ISDIR(state.st_mode):
        raise DojoHistoricalJobArchiveError(f"{field} must be a real directory")


def inspect_historical_job_archive_source(
    *,
    run_root: Path,
    job_sha256: str,
) -> dict[str, Any]:
    """Return the exact prospective archive binding without creating anything."""

    if _SHA_RE.fullmatch(job_sha256) is None:
        raise DojoHistoricalJobArchiveError("job SHA-256 is invalid")
    try:
        root = Path(run_root).resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError("run root is unavailable") from exc
    _require_real_directory(root, field="run root")
    completion = _completion(root, job_sha256)
    bundle_kind = _bundle_kind(completion)
    files, total = _inventory(
        root,
        job_sha256,
        bundle_kind=bundle_kind,
    )
    _validate_inventory_links(
        run_root=root,
        job_sha256=job_sha256,
        completion=completion,
        rows=files,
        bundle_kind=bundle_kind,
    )
    repeated_files, repeated_total = _inventory(
        root,
        job_sha256,
        bundle_kind=bundle_kind,
    )
    if repeated_files != files or repeated_total != total:
        raise DojoHistoricalJobArchiveError(
            "archive source inventory changed while inspected"
        )
    manifest = _build_archive_manifest(
        job_sha256=job_sha256,
        completion_sha256=completion["completion_sha256"],
        bundle_kind=bundle_kind,
        files=files,
        total_source_bytes=total,
    )
    body = {
        "contract": ARCHIVE_SOURCE_INSPECTION_CONTRACT,
        "schema_version": 1,
        "job_sha256": job_sha256,
        "completion_sha256": completion["completion_sha256"],
        "bundle_kind": bundle_kind,
        "file_count": len(files),
        "total_source_bytes": total,
        "source_inventory_sha256": _canonical_sha256(files),
        "manifest_sha256": manifest["manifest_sha256"],
        "historical_train_is_proof": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "inspection_sha256": _canonical_sha256(body)}


def _archive_capacity_upper_bound(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Bound tar/PAX/zstd bytes without assuming any compression benefit."""

    manifest_bytes = len(_canonical_bytes(manifest))
    members = [("MANIFEST.json", manifest_bytes)] + [
        (f"payload/{row['path']}", int(row["size_bytes"])) for row in manifest["files"]
    ]

    def padded(value: int) -> int:
        return (
            (value + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
        ) * tarfile.BLOCKSIZE

    # Reserve a bounded PAX extended-header record for every member even when
    # tarfile would use only the normal 512-byte header.  The path validator
    # bounds member names well below this per-member allowance.
    pax_overhead_per_member = 64 * 1024
    tar_bytes = tarfile.RECORDSIZE + sum(
        tarfile.BLOCKSIZE + padded(size) + pax_overhead_per_member
        for _, size in members
    )
    # zstd incompressible framing overhead is much smaller than this 1% +
    # 128KiB reserve.  Keep both terms so small and multi-gigabyte archives are
    # admitted without relying on compression.
    zstd_framing_bytes = max(128 * 1024, (tar_bytes + 99) // 100)
    return {
        "manifest_payload_bytes": manifest_bytes,
        "tar_pax_upper_bound_bytes": tar_bytes,
        "zstd_framing_upper_bound_bytes": zstd_framing_bytes,
        "archive_upper_bound_bytes": tar_bytes + zstd_framing_bytes,
    }


def inspect_historical_job_archive_recovery_capacity(
    *,
    run_root: Path,
    job_sha256: str,
    archive_root: Path,
    zstd_bin: str = "zstd",
    local_staging_root: Path | None = None,
) -> dict[str, Any]:
    """Deep-validate crash artifacts and return remaining allocation deltas."""

    if _SHA_RE.fullmatch(job_sha256) is None:
        raise DojoHistoricalJobArchiveError("job SHA-256 is invalid")
    root = Path(run_root).resolve(strict=True)
    destination = Path(archive_root).expanduser()
    if not destination.is_absolute():
        raise DojoHistoricalJobArchiveError("archive root must be absolute")
    destination = destination.resolve(strict=False)
    if destination == root or root in destination.parents:
        raise DojoHistoricalJobArchiveError("archive root must be outside the run root")
    completion = _completion(root, job_sha256)
    bundle_kind = _bundle_kind(completion)
    files, total = _inventory(root, job_sha256, bundle_kind=bundle_kind)
    _validate_inventory_links(
        run_root=root,
        job_sha256=job_sha256,
        completion=completion,
        rows=files,
        bundle_kind=bundle_kind,
    )
    manifest = _build_archive_manifest(
        job_sha256=job_sha256,
        completion_sha256=completion["completion_sha256"],
        bundle_kind=bundle_kind,
        files=files,
        total_source_bytes=total,
    )
    bounds = _archive_capacity_upper_bound(manifest)
    stem = f"job-{job_sha256}-{manifest['manifest_sha256']}"
    archives = destination / "archives"
    archive_path = archives / f"{stem}.tar.zst"
    drive_pending = archives / f".{stem}.tar.zst.pending"
    staging_parent = _resolve_local_staging_parent(
        local_staging_root,
        run_root=root,
        archive_root=destination,
    )
    local_pending = staging_parent / f".{stem}.tar.zst.local-pending"

    def verified_archive(path: Path, *, final: bool) -> tuple[str, int] | None:
        try:
            path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        try:
            _verify_archive(
                path,
                manifest=manifest,
                zstd_bin=zstd_bin,
                expected_job_sha256=job_sha256,
                expected_completion_sha256=completion["completion_sha256"],
                expected_bundle_kind=bundle_kind,
            )
            observed = _hash_file(path)
        except DojoHistoricalJobArchiveError:
            if final:
                raise
            return None
        if observed[1] > bounds["archive_upper_bound_bytes"]:
            raise DojoHistoricalJobArchiveError(
                "validated archive exceeds its conservative capacity bound"
            )
        return observed

    final_archive = verified_archive(archive_path, final=True)
    local_archive = verified_archive(local_pending, final=False)
    drive_archive = verified_archive(drive_pending, final=False)
    if final_archive is not None:
        archive_basis = final_archive[1]
        remaining_local = 0
        remaining_final = 0
        source_path = archive_path
        source_binding = final_archive
    elif local_archive is not None:
        archive_basis = local_archive[1]
        remaining_local = 0
        drive_reusable = drive_archive == local_archive
        remaining_final = 0 if drive_reusable else archive_basis
        source_path = local_pending
        source_binding = local_archive
    else:
        archive_basis = bounds["archive_upper_bound_bytes"]
        remaining_local = archive_basis
        remaining_final = archive_basis
        source_path = None
        source_binding = None

    validated_readback_final = 0
    validated_readback_pending = 0
    if source_path is not None and source_binding is not None:
        expected_final_names: set[str] = set()
        with _VerifiedSource(
            source_path,
            expected_size=source_binding[1],
            expected_sha256=source_binding[0],
        ) as source:
            index = 0
            while True:
                part_digest = hashlib.sha256()
                part_size = 0
                while part_size < ARCHIVE_PART_BYTES:
                    chunk = source.read(
                        min(HASH_CHUNK_BYTES, ARCHIVE_PART_BYTES - part_size)
                    )
                    if not chunk:
                        break
                    part_digest.update(chunk)
                    part_size += len(chunk)
                if part_size == 0:
                    break
                digest = part_digest.hexdigest()
                final_name = f"{stem}-part-{index:05d}-{digest}.bin"
                expected_final_names.add(final_name)
                final_part = destination / "readback-objects" / final_name
                pending_part = (
                    destination
                    / "readback-objects"
                    / f".{stem}-part-{index:05d}.pending"
                )
                expected = (digest, part_size)
                if final_part.exists():
                    if _hash_file(final_part) != expected:
                        raise DojoHistoricalJobArchiveError(
                            "existing readback part is not reusable"
                        )
                    validated_readback_final += part_size
                elif pending_part.exists():
                    try:
                        reusable_pending = _hash_file(pending_part) == expected
                    except DojoHistoricalJobArchiveError:
                        reusable_pending = False
                    if reusable_pending:
                        validated_readback_pending += part_size
                index += 1
            source.verify()
        readback_root = destination / "readback-objects"
        existing_final_names = (
            {path.name for path in readback_root.glob(f"{stem}-part-*.bin")}
            if readback_root.is_dir()
            else set()
        )
        if not existing_final_names <= expected_final_names:
            raise DojoHistoricalJobArchiveError(
                "readback recovery contains an unexpected final part"
            )
    validated_readback = validated_readback_final + validated_readback_pending
    remaining_readback = max(0, archive_basis - validated_readback)
    remaining_part_temp = min(ARCHIVE_PART_BYTES, remaining_readback)
    body = {
        "contract": ARCHIVE_RECOVERY_CAPACITY_CONTRACT,
        "schema_version": 1,
        "job_sha256": job_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "total_source_bytes": total,
        **bounds,
        "archive_allocation_basis_bytes": archive_basis,
        "validated_local_archive_pending_bytes": (
            local_archive[1] if local_archive is not None else 0
        ),
        "validated_archive_final_bytes": (
            final_archive[1] if final_archive is not None else 0
        ),
        "validated_archive_pending_bytes": (
            drive_archive[1]
            if local_archive is not None and drive_archive == local_archive
            else 0
        ),
        "validated_readback_final_bytes": validated_readback_final,
        "validated_readback_pending_bytes": validated_readback_pending,
        "remaining_local_staging_bytes": remaining_local,
        "remaining_archive_final_bytes": remaining_final,
        "remaining_archive_readback_bytes": remaining_readback,
        "remaining_archive_part_temp_bytes": remaining_part_temp,
        "remaining_archive_filesystem_bytes": (
            remaining_final + remaining_readback + remaining_part_temp
        ),
        "compression_ratio_assumed": False,
        "historical_train_is_proof": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "capacity_inspection_sha256": _canonical_sha256(body)}


def verify_existing_historical_job_archive(
    *,
    run_root: Path,
    job_sha256: str,
    archive_root: Path,
    zstd_bin: str = "zstd",
) -> dict[str, Any]:
    """Deep-verify one existing archive without creating files or directories."""

    if _SHA_RE.fullmatch(job_sha256) is None:
        raise DojoHistoricalJobArchiveError("job SHA-256 is invalid")
    try:
        root = Path(run_root).resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError("run root is unavailable") from exc
    _require_real_directory(root, field="run root")
    destination_input = Path(archive_root).expanduser()
    if not destination_input.is_absolute():
        raise DojoHistoricalJobArchiveError("archive root must be absolute")
    try:
        destination = destination_input.resolve(strict=True)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError("archive root is unavailable") from exc
    _require_real_directory(destination, field="archive root")
    if destination == root or root in destination.parents:
        raise DojoHistoricalJobArchiveError("archive root must be outside the run root")
    archives = destination / "archives"
    receipts = destination / "receipts"
    _require_real_directory(archives, field="archive directory")
    _require_real_directory(receipts, field="receipt directory")

    completion = _completion(root, job_sha256)
    bundle_kind = _bundle_kind(completion)
    existing_receipts = sorted(receipts.glob(f"job-{job_sha256}-*.json"))
    if len(existing_receipts) != 1:
        raise DojoHistoricalJobArchiveError(
            "exactly one archive receipt must name the job"
        )
    receipt = _read_json(existing_receipts[0], field="archive receipt")
    archive_path, manifest_sha256 = _validate_receipt(
        receipt,
        job_sha256=job_sha256,
        completion_sha256=completion["completion_sha256"],
        bundle_kind=bundle_kind,
        archives=archives,
    )
    expected_receipt = receipts / f"job-{job_sha256}-{manifest_sha256}.json"
    if existing_receipts[0] != expected_receipt:
        raise DojoHistoricalJobArchiveError(
            "archive receipt filename does not match its manifest"
        )
    archive_sha256, archive_size = _hash_file(archive_path)
    if (
        archive_sha256 != receipt["archive_sha256"]
        or archive_size != receipt["archive_size_bytes"]
    ):
        raise DojoHistoricalJobArchiveError("existing archive bytes drifted")
    stem = f"job-{job_sha256}-{manifest_sha256}"
    _verify_remote_readback_objects(
        receipt["remote_readback_objects"],
        destination=destination,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size,
        stem=stem,
    )
    verified = _verify_archive(
        archive_path,
        zstd_bin=zstd_bin,
        expected_job_sha256=job_sha256,
        expected_completion_sha256=completion["completion_sha256"],
        expected_bundle_kind=bundle_kind,
    )
    current_files, current_total = _inventory(
        root,
        job_sha256,
        bundle_kind=bundle_kind,
    )
    _validate_inventory_links(
        run_root=root,
        job_sha256=job_sha256,
        completion=completion,
        rows=current_files,
        bundle_kind=bundle_kind,
    )
    if (
        verified["manifest_sha256"] != manifest_sha256
        or verified["bundle_kind"] != receipt["bundle_kind"]
        or verified["file_count"] != receipt["file_count"]
        or verified["total_source_bytes"] != receipt["total_source_bytes"]
        or verified["files"] != current_files
        or verified["total_source_bytes"] != current_total
    ):
        raise DojoHistoricalJobArchiveError(
            "existing archive receipt does not match its full inventory"
        )
    return dict(receipt)


def _resolve_local_staging_parent(
    value: Path | None,
    *,
    run_root: Path,
    archive_root: Path,
) -> Path:
    def uses_file_provider(path: Path) -> bool:
        candidates = (path, *path.parents)
        if any(
            candidate.name.casefold()
            in {"cloudstorage", "file provider storage", ".fileprovider"}
            for candidate in candidates
        ):
            return True
        for candidate in candidates:
            try:
                attributes = os.listxattr(candidate, follow_symlinks=False)
            except (AttributeError, OSError):
                continue
            if any("fileprovider" in attribute.casefold() for attribute in attributes):
                return True
        return False

    if value is None:
        parent = Path(tempfile.gettempdir()).resolve(strict=True)
    else:
        parent = Path(value).expanduser()
        if not parent.is_absolute():
            raise DojoHistoricalJobArchiveError("local staging root must be absolute")
        if uses_file_provider(parent.resolve(strict=False)):
            raise DojoHistoricalJobArchiveError(
                "local staging root must not use CloudStorage or File Provider"
            )
        _ensure_directory(parent)
        parent = parent.resolve(strict=True)
    if uses_file_provider(parent):
        raise DojoHistoricalJobArchiveError(
            "local staging root must not use CloudStorage or File Provider"
        )
    if parent == archive_root or archive_root in parent.parents:
        raise DojoHistoricalJobArchiveError(
            "local staging root must be outside the archive root"
        )
    if parent == run_root or run_root in parent.parents:
        raise DojoHistoricalJobArchiveError(
            "local staging root must be outside the run root"
        )
    return parent


def _prepare_local_archive_pending(
    *,
    pending_path: Path,
    run_root: Path,
    manifest: Mapping[str, Any],
    zstd_bin: str,
    job_sha256: str,
    completion_sha256: str,
    bundle_kind: str,
    mutation_guard: Callable[[], None] | None = None,
) -> tuple[str, int]:
    if shutil.which(zstd_bin) is None:
        raise DojoHistoricalJobArchiveError("zstd executable is unavailable")
    try:
        pending_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        try:
            _verify_archive(
                pending_path,
                manifest=manifest,
                zstd_bin=zstd_bin,
                expected_job_sha256=job_sha256,
                expected_completion_sha256=completion_sha256,
                expected_bundle_kind=bundle_kind,
            )
            return _hash_file(pending_path)
        except DojoHistoricalJobArchiveError:
            if mutation_guard is not None:
                mutation_guard()
            _remove_exact_pending(pending_path, field="local archive")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if mutation_guard is not None:
        mutation_guard()
    try:
        descriptor = os.open(pending_path, flags, 0o600)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            "local archive pending cannot be created"
        ) from exc
    with os.fdopen(descriptor, "w+b", closefd=True) as output:
        _write_archive(
            output=output,
            run_root=run_root,
            manifest=manifest,
            zstd_bin=zstd_bin,
        )
    _verify_archive(
        pending_path,
        manifest=manifest,
        zstd_bin=zstd_bin,
        expected_job_sha256=job_sha256,
        expected_completion_sha256=completion_sha256,
        expected_bundle_kind=bundle_kind,
    )
    _fsync_directory(pending_path.parent)
    return _hash_file(pending_path)


def archive_completed_historical_job(
    *,
    run_root: Path,
    job_sha256: str,
    archive_root: Path,
    zstd_bin: str = "zstd",
    local_staging_root: Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Archive and re-read one sealed terminal job; never delete its source."""

    if _SHA_RE.fullmatch(job_sha256) is None:
        raise DojoHistoricalJobArchiveError("job SHA-256 is invalid")
    root = Path(run_root).resolve(strict=True)
    if not root.is_dir():
        raise DojoHistoricalJobArchiveError("run root must be a directory")
    destination = Path(archive_root).expanduser()
    if not destination.is_absolute():
        raise DojoHistoricalJobArchiveError("archive root must be absolute")
    unresolved_destination = destination.resolve(strict=False)
    if unresolved_destination == root or root in unresolved_destination.parents:
        raise DojoHistoricalJobArchiveError("archive root must be outside the run root")
    _ensure_directory(destination)
    destination = destination.resolve(strict=True)
    if destination == root or root in destination.parents:
        raise DojoHistoricalJobArchiveError("archive root must be outside the run root")
    archives = destination / "archives"
    receipts = destination / "receipts"
    _ensure_directory(archives)
    _ensure_directory(receipts)
    with _job_archive_lock(receipts, job_sha256) as archive_lock_guard:
        completion = _completion(root, job_sha256)
        bundle_kind = _bundle_kind(completion)
        existing_receipts = sorted(receipts.glob(f"job-{job_sha256}-*.json"))
        if existing_receipts:
            verified_receipt = verify_existing_historical_job_archive(
                run_root=root,
                job_sha256=job_sha256,
                archive_root=destination,
                zstd_bin=zstd_bin,
            )
            existing_stem = f"job-{job_sha256}-{verified_receipt['manifest_sha256']}"
            staging_parent = _resolve_local_staging_parent(
                local_staging_root,
                run_root=root,
                archive_root=destination,
            )
            archive_lock_guard()
            _remove_exact_pending(
                staging_parent / f".{existing_stem}.tar.zst.local-pending",
                field="local archive",
                expected_sha256=verified_receipt["archive_sha256"],
                expected_size_bytes=verified_receipt["archive_size_bytes"],
            )
            archive_lock_guard()
            _remove_exact_pending(
                archives / f".{existing_stem}.tar.zst.pending",
                field="Drive archive",
                expected_sha256=verified_receipt["archive_sha256"],
                expected_size_bytes=verified_receipt["archive_size_bytes"],
            )
            receipt_payload = _canonical_bytes(verified_receipt) + b"\n"
            archive_lock_guard()
            _remove_exact_pending(
                receipts / f".{existing_stem}.json.pending",
                field="archive receipt",
                expected_sha256=hashlib.sha256(receipt_payload).hexdigest(),
                expected_size_bytes=len(receipt_payload),
            )
            for row in verified_receipt["remote_readback_objects"]["objects"]:
                archive_lock_guard()
                _remove_exact_pending(
                    destination
                    / "readback-objects"
                    / f".{existing_stem}-part-{row['index']:05d}.pending",
                    field="remote readback part",
                    expected_sha256=row["sha256"],
                    expected_size_bytes=row["size_bytes"],
                )
            return verified_receipt

        files, total = _inventory(
            root,
            job_sha256,
            bundle_kind=bundle_kind,
        )
        _validate_inventory_links(
            run_root=root,
            job_sha256=job_sha256,
            completion=completion,
            rows=files,
            bundle_kind=bundle_kind,
        )
        manifest = _build_archive_manifest(
            job_sha256=job_sha256,
            completion_sha256=completion["completion_sha256"],
            bundle_kind=bundle_kind,
            files=files,
            total_source_bytes=total,
        )
        if expected_manifest_sha256 is not None and (
            _SHA_RE.fullmatch(expected_manifest_sha256) is None
            or manifest["manifest_sha256"] != expected_manifest_sha256
        ):
            raise DojoHistoricalJobArchiveError(
                "archive source changed after capacity inspection"
            )
        # Capacity inspection and inventory hashing can be long.  Revalidate
        # the named lock before entering the first mutation phase.
        archive_lock_guard()
        stem = f"job-{job_sha256}-{manifest['manifest_sha256']}"
        archive_path = archives / f"{stem}.tar.zst"
        drive_pending = archives / f".{stem}.tar.zst.pending"
        staging_parent = _resolve_local_staging_parent(
            local_staging_root,
            run_root=root,
            archive_root=destination,
        )
        local_pending = staging_parent / f".{stem}.tar.zst.local-pending"
        source_for_readback = archive_path
        try:
            archive_state = archive_path.stat(follow_symlinks=False)
        except FileNotFoundError:
            archive_state = None
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                "archive destination is unavailable"
            ) from exc
        if archive_state is not None:
            if not stat.S_ISREG(archive_state.st_mode):
                raise DojoHistoricalJobArchiveError(
                    "archive destination is not a regular file"
                )
            _verify_archive(
                archive_path,
                manifest=manifest,
                zstd_bin=zstd_bin,
                expected_job_sha256=job_sha256,
                expected_completion_sha256=completion["completion_sha256"],
                expected_bundle_kind=bundle_kind,
            )
            archive_sha, archive_size = _hash_file(archive_path)
            archive_lock_guard()
            _remove_exact_pending(
                drive_pending,
                field="Drive archive",
                expected_sha256=archive_sha,
                expected_size_bytes=archive_size,
            )
            try:
                local_pending.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                try:
                    _verify_archive(
                        local_pending,
                        manifest=manifest,
                        zstd_bin=zstd_bin,
                        expected_job_sha256=job_sha256,
                        expected_completion_sha256=completion["completion_sha256"],
                        expected_bundle_kind=bundle_kind,
                    )
                    local_sha, local_size = _hash_file(local_pending)
                except DojoHistoricalJobArchiveError:
                    archive_lock_guard()
                    _remove_exact_pending(local_pending, field="local archive")
                else:
                    if local_sha == archive_sha and local_size == archive_size:
                        source_for_readback = local_pending
                    else:
                        archive_lock_guard()
                        _remove_exact_pending(
                            local_pending,
                            field="local archive",
                            expected_sha256=local_sha,
                            expected_size_bytes=local_size,
                        )
        else:
            archive_lock_guard()
            local_archive_sha, local_archive_size = _prepare_local_archive_pending(
                pending_path=local_pending,
                run_root=root,
                manifest=manifest,
                zstd_bin=zstd_bin,
                job_sha256=job_sha256,
                completion_sha256=completion["completion_sha256"],
                bundle_kind=bundle_kind,
                mutation_guard=archive_lock_guard,
            )
            archive_lock_guard()
            _prepare_checked_copy_pending(
                source_path=local_pending,
                pending_path=drive_pending,
                expected_sha256=local_archive_sha,
                expected_size_bytes=local_archive_size,
                field="Drive archive",
                mutation_guard=archive_lock_guard,
            )
            try:
                _verify_archive(
                    drive_pending,
                    manifest=manifest,
                    zstd_bin=zstd_bin,
                    expected_job_sha256=job_sha256,
                    expected_completion_sha256=completion["completion_sha256"],
                    expected_bundle_kind=bundle_kind,
                )
                drive_archive_sha, drive_archive_size = _hash_file(drive_pending)
            except DojoHistoricalJobArchiveError:
                # The checked copy above already proved the pending object's
                # stable bytes.  A later deep-read failure can be a transient
                # File Provider/materialization error, so retain that object
                # unless a second stable hash proves that its bytes drifted.
                # An unavailable or identity-unstable re-read is indeterminate
                # and must not trigger an expensive full upload on retry.
                try:
                    pending_sha256, pending_size = _hash_file(drive_pending)
                except DojoHistoricalJobArchiveError:
                    pass
                else:
                    if (
                        pending_sha256 != local_archive_sha
                        or pending_size != local_archive_size
                    ):
                        archive_lock_guard()
                        _remove_exact_pending(drive_pending, field="Drive archive")
                raise
            if (
                drive_archive_sha != local_archive_sha
                or drive_archive_size != local_archive_size
            ):
                archive_lock_guard()
                _remove_exact_pending(drive_pending, field="Drive archive")
                raise DojoHistoricalJobArchiveError(
                    "Drive archive pending differs from local staging"
                )
            archive_lock_guard()
            _publish_prepared_pending(
                pending_path=drive_pending,
                destination_path=archive_path,
                expected_sha256=local_archive_sha,
                expected_size_bytes=local_archive_size,
                field="archive",
                mutation_guard=archive_lock_guard,
            )
            archive_sha = local_archive_sha
            archive_size = local_archive_size
            source_for_readback = local_pending
        archive_lock_guard()
        remote_readback_objects = _build_remote_readback_objects(
            destination=destination,
            archive_path=archive_path,
            archive_sha256=archive_sha,
            archive_size_bytes=archive_size,
            stem=stem,
            source_archive_path=source_for_readback,
            mutation_guard=archive_lock_guard,
        )
        _verify_remote_readback_objects(
            remote_readback_objects,
            destination=destination,
            archive_path=archive_path,
            archive_sha256=archive_sha,
            archive_size_bytes=archive_size,
            stem=stem,
        )
        receipt_body = {
            "contract": ARCHIVE_RECEIPT_CONTRACT,
            "schema_version": 1,
            "job_sha256": job_sha256,
            "completion_sha256": completion["completion_sha256"],
            "bundle_kind": bundle_kind,
            "manifest_sha256": manifest["manifest_sha256"],
            "archive_path": os.fspath(archive_path),
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_size,
            "file_count": len(files),
            "total_source_bytes": total,
            "local_payload_verified": True,
            "remote_readback_objects": remote_readback_objects,
            "source_deletion_allowed": False,
            "source_deleted": False,
            "remote_verification": {
                "status": "LOCAL_DRIVE_SYNC_PENDING",
                "remote_verified": False,
                "metadata_receipt_sha256": None,
            },
            "historical_train_is_proof": False,
            "promotion_eligible": False,
            "live_permission": False,
            "order_authority": "NONE",
            "broker_mutation_allowed": False,
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": _canonical_sha256(receipt_body),
        }
        # The Drive final and every readback object have been independently
        # reconstructed above, so the local staging copy is no longer needed.
        # Remove it before the receipt becomes the settled lifecycle marker;
        # otherwise a crash immediately after atomic receipt publication leaves
        # a full-size pending that archive-next will never revisit.
        archive_lock_guard()
        _remove_exact_pending(
            local_pending,
            field="local archive",
            expected_sha256=archive_sha,
            expected_size_bytes=archive_size,
        )
        archive_lock_guard()
        _write_once(
            receipts / f"{stem}.json",
            receipt,
            mutation_guard=archive_lock_guard,
        )
        archive_lock_guard()
        return receipt


__all__ = [
    "ARCHIVE_PART_BYTES",
    "ARCHIVE_RECOVERY_CAPACITY_CONTRACT",
    "ARCHIVE_RECEIPT_CONTRACT",
    "ARCHIVE_SOURCE_INSPECTION_CONTRACT",
    "DojoHistoricalJobArchiveError",
    "archive_completed_historical_job",
    "inspect_historical_job_archive_source",
    "inspect_historical_job_archive_recovery_capacity",
    "verify_existing_historical_job_archive",
]

"""Deterministic local archive for one terminal historical DOJO job.

The archive is a storage handoff, not evidence promotion.  It captures the
fixed generation artifacts, the job-local economic evidence, the exact source
slice and the matching execution-state records in one verified ``tar.zst``.
Raw evidence is never deleted here; a separately verified remote receipt is
required before any later reclamation policy may do that.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Final, Iterator

import fcntl


ARCHIVE_RECEIPT_CONTRACT: Final = "QR_DOJO_HISTORICAL_JOB_ARCHIVE_V1"
ARCHIVE_MANIFEST_CONTRACT: Final = "QR_DOJO_HISTORICAL_JOB_ARCHIVE_MANIFEST_V1"
ARCHIVE_READBACK_CONTRACT: Final = "QR_DOJO_ARCHIVE_READBACK_OBJECT_SET_V1"
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
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


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
            raw = handle.read(maximum + 1)
            opened = os.fstat(handle.fileno())
        after = path.stat(follow_symlinks=False)
    except DojoHistoricalJobArchiveError:
        raise
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(f"{field} is unavailable") from exc
    if (
        _identity(before) != _identity(opened)
        or _identity(opened) != _identity(after)
        or len(raw) != before.st_size
    ):
        raise DojoHistoricalJobArchiveError(f"{field} changed while read")
    return raw


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    return _decode_json_object(
        _stable_regular_bytes(path, field=field, maximum=MAX_JSON_BYTES),
        field=field,
    )


def _hash_file(path: Path) -> tuple[str, int]:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise DojoHistoricalJobArchiveError(
                f"archive source is not regular: {path}"
            )
        digest = hashlib.sha256()
        size = 0
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
                size += len(chunk)
            opened = os.fstat(handle.fileno())
        after = path.stat(follow_symlinks=False)
    except DojoHistoricalJobArchiveError:
        raise
    except OSError as exc:
        raise DojoHistoricalJobArchiveError(
            f"archive source is unavailable: {path}"
        ) from exc
    if (
        _identity(before) != _identity(opened)
        or _identity(opened) != _identity(after)
        or size != before.st_size
    ):
        raise DojoHistoricalJobArchiveError(
            f"archive source changed while hashing: {path}"
        )
    return digest.hexdigest(), size


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
        if _identity(self._before) != _identity(self._opened):
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
            _identity(self._before) != _identity(opened_after)
            or _identity(opened_after) != _identity(after)
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
    if _identity(archive_before) != _identity(archive_opened):
        os.close(archive_descriptor)
        raise DojoHistoricalJobArchiveError("archive changed while opened")
    verified: dict[str, Any] | None = None
    with tempfile.TemporaryFile() as process_error:
        try:
            process = subprocess.Popen(
                [resolved, "-q", "-d", "-c", f"/dev/fd/{archive_descriptor}"],
                stdout=subprocess.PIPE,
                stderr=process_error,
                pass_fds=(archive_descriptor,),
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
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            os.close(archive_descriptor)
            if isinstance(exc, DojoHistoricalJobArchiveError):
                raise
            raise DojoHistoricalJobArchiveError(
                "archive tar stream is invalid"
            ) from exc
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
        if _identity(archive_before) != _identity(archive_opened_after) or _identity(
            archive_opened_after
        ) != _identity(archive_after):
            raise DojoHistoricalJobArchiveError("archive changed while verified")
        process_error.seek(0)
        error = process_error.read(4096)
        if return_code != 0:
            raise DojoHistoricalJobArchiveError(
                f"zstd verification failed: {error.decode('utf-8', 'replace')[:1000]}"
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


def _build_remote_readback_objects(
    *,
    destination: Path,
    archive_path: Path,
    archive_sha256: str,
    archive_size_bytes: int,
    stem: str,
) -> dict[str, Any]:
    if archive_size_bytes <= ARCHIVE_PART_BYTES:
        objects = [
            {
                "index": 0,
                "offset_bytes": 0,
                "relative_path": archive_path.relative_to(destination).as_posix(),
                "size_bytes": archive_size_bytes,
                "sha256": archive_sha256,
            }
        ]
    else:
        parts_root = destination / "parts"
        _ensure_directory(parts_root)
        part_set_root = parts_root / stem
        _ensure_directory(part_set_root)
        try:
            before = archive_path.stat(follow_symlinks=False)
            descriptor = os.open(
                archive_path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                "archive is unavailable while split for remote readback"
            ) from exc
        if (
            not stat.S_ISREG(before.st_mode)
            or _identity(before) != _identity(opened)
            or before.st_size != archive_size_bytes
        ):
            os.close(descriptor)
            raise DojoHistoricalJobArchiveError(
                "archive drifted before remote readback split"
            )
        objects = []
        combined = hashlib.sha256()
        combined_size = 0
        try:
            with os.fdopen(descriptor, "rb", closefd=True) as source:
                while combined_size < archive_size_bytes:
                    if len(objects) >= MAX_ARCHIVE_PARTS:
                        raise DojoHistoricalJobArchiveError(
                            "archive remote readback part count exceeds its bound"
                        )
                    part_descriptor, part_name = tempfile.mkstemp(
                        prefix=f".part-{len(objects):05d}.",
                        suffix=".tmp",
                        dir=part_set_root,
                    )
                    temporary = Path(part_name)
                    part_digest = hashlib.sha256()
                    part_size = 0
                    try:
                        with os.fdopen(part_descriptor, "wb", closefd=True) as part:
                            while part_size < ARCHIVE_PART_BYTES:
                                chunk = source.read(
                                    min(
                                        HASH_CHUNK_BYTES,
                                        ARCHIVE_PART_BYTES - part_size,
                                    )
                                )
                                if not chunk:
                                    break
                                part.write(chunk)
                                part_digest.update(chunk)
                                combined.update(chunk)
                                part_size += len(chunk)
                                combined_size += len(chunk)
                            part.flush()
                            os.fsync(part.fileno())
                        if part_size <= 0:
                            raise DojoHistoricalJobArchiveError(
                                "archive ended before its declared size"
                            )
                        digest = part_digest.hexdigest()
                        final = part_set_root / (
                            f"part-{len(objects):05d}-{digest}.bin"
                        )
                        try:
                            os.link(temporary, final)
                        except FileExistsError:
                            existing_sha, existing_size = _hash_file(final)
                            if existing_sha != digest or existing_size != part_size:
                                raise DojoHistoricalJobArchiveError(
                                    "existing remote readback part drifted"
                                )
                        objects.append(
                            {
                                "index": len(objects),
                                "offset_bytes": combined_size - part_size,
                                "relative_path": final.relative_to(
                                    destination
                                ).as_posix(),
                                "size_bytes": part_size,
                                "sha256": digest,
                            }
                        )
                    finally:
                        try:
                            temporary.unlink()
                        except FileNotFoundError:
                            pass
                opened_after = os.fstat(source.fileno())
            after = archive_path.stat(follow_symlinks=False)
        except DojoHistoricalJobArchiveError:
            raise
        except OSError as exc:
            raise DojoHistoricalJobArchiveError(
                "archive changed during remote readback split"
            ) from exc
        if (
            _identity(before) != _identity(opened_after)
            or _identity(opened_after) != _identity(after)
            or combined_size != archive_size_bytes
            or combined.hexdigest() != archive_sha256
        ):
            raise DojoHistoricalJobArchiveError(
                "remote readback parts do not reconstruct the archive"
            )
        _fsync_directory(part_set_root)
        expected_names = {Path(row["relative_path"]).name for row in objects}
        existing_names = {
            path.name
            for path in part_set_root.iterdir()
            if not path.name.startswith(".")
        }
        if existing_names != expected_names:
            raise DojoHistoricalJobArchiveError(
                "remote readback part directory contains an orphan"
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
        expected_prefix = f"parts/{stem}/part-{index:05d}-{sha256}.bin"
        if (len(objects) == 1 and safe_relative != expected_small) or (
            len(objects) > 1 and safe_relative != expected_prefix
        ):
            raise DojoHistoricalJobArchiveError(
                "remote readback object path is outside its part set"
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
            or _identity(before) != _identity(opened)
            or _identity(opened) != _identity(opened_after)
            or _identity(opened_after) != _identity(after)
            or observed_size != size
            or digest.hexdigest() != sha256
        ):
            raise DojoHistoricalJobArchiveError("remote readback object bytes drifted")
        total += observed_size
    if total != archive_size_bytes or combined.hexdigest() != archive_sha256:
        raise DojoHistoricalJobArchiveError(
            "remote readback objects do not reconstruct the archive"
        )


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
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
            current = _stable_regular_bytes(
                path, field="archive receipt", maximum=MAX_JSON_BYTES
            )
            if current != payload:
                raise DojoHistoricalJobArchiveError(
                    "archive receipt concurrently drifted"
                )
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _job_archive_lock(receipts: Path, job_sha256: str) -> Iterator[None]:
    lock_path = receipts / f".job-{job_sha256}.lock"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise DojoHistoricalJobArchiveError("archive lock is unavailable") from exc
    try:
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode):
            raise DojoHistoricalJobArchiveError("archive lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


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


def archive_completed_historical_job(
    *,
    run_root: Path,
    job_sha256: str,
    archive_root: Path,
    zstd_bin: str = "zstd",
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
    with _job_archive_lock(receipts, job_sha256):
        completion = _completion(root, job_sha256)
        bundle_kind = _bundle_kind(completion)
        existing_receipts = sorted(receipts.glob(f"job-{job_sha256}-*.json"))
        if existing_receipts:
            return verify_existing_historical_job_archive(
                run_root=root,
                job_sha256=job_sha256,
                archive_root=destination,
                zstd_bin=zstd_bin,
            )

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
        manifest_body = {
            "contract": ARCHIVE_MANIFEST_CONTRACT,
            "schema_version": 1,
            "job_sha256": job_sha256,
            "completion_sha256": completion["completion_sha256"],
            "bundle_kind": bundle_kind,
            "file_count": len(files),
            "total_source_bytes": total,
            "files": files,
            "historical_train_is_proof": False,
            "promotion_eligible": False,
            "live_permission": False,
            "order_authority": "NONE",
            "broker_mutation_allowed": False,
        }
        manifest = _validate_manifest(
            {
                **manifest_body,
                "manifest_sha256": _canonical_sha256(manifest_body),
            },
            expected_job_sha256=job_sha256,
            expected_completion_sha256=completion["completion_sha256"],
            expected_bundle_kind=bundle_kind,
        )
        stem = f"job-{job_sha256}-{manifest['manifest_sha256']}"
        archive_path = archives / f"{stem}.tar.zst"
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
        else:
            descriptor, part_name = tempfile.mkstemp(
                prefix=f".{stem}.", suffix=".tar.zst.part", dir=archives
            )
            part_path = Path(part_name)
            try:
                with os.fdopen(descriptor, "w+b", closefd=True) as output:
                    _write_archive(
                        output=output,
                        run_root=root,
                        manifest=manifest,
                        zstd_bin=zstd_bin,
                    )
                _verify_archive(
                    part_path,
                    manifest=manifest,
                    zstd_bin=zstd_bin,
                    expected_job_sha256=job_sha256,
                    expected_completion_sha256=completion["completion_sha256"],
                    expected_bundle_kind=bundle_kind,
                )
                try:
                    os.link(part_path, archive_path)
                except FileExistsError:
                    _verify_archive(
                        archive_path,
                        manifest=manifest,
                        zstd_bin=zstd_bin,
                        expected_job_sha256=job_sha256,
                        expected_completion_sha256=completion["completion_sha256"],
                        expected_bundle_kind=bundle_kind,
                    )
                _fsync_directory(archives)
            finally:
                try:
                    part_path.unlink()
                except FileNotFoundError:
                    pass
        archive_sha, archive_size = _hash_file(archive_path)
        remote_readback_objects = _build_remote_readback_objects(
            destination=destination,
            archive_path=archive_path,
            archive_sha256=archive_sha,
            archive_size_bytes=archive_size,
            stem=stem,
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
        _write_once(receipts / f"{stem}.json", receipt)
        return receipt


__all__ = [
    "ARCHIVE_RECEIPT_CONTRACT",
    "DojoHistoricalJobArchiveError",
    "archive_completed_historical_job",
    "verify_existing_historical_job_archive",
]

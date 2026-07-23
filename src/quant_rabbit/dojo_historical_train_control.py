"""Reviewed, research-only controller for continuous DOJO historical TRAIN.

The controller closes the gap between the sealed long-horizon primitives and
an actual replay process.  It accepts one repository-owned run-control policy,
materializes one observed-only month slice, executes one immutable job, then
publishes the complete coordinate denominator and carry state.  It has no
broker client, credential input, live permission, promotion path, or model
client.

Generation mutation is deliberately outside this module.  Economic results
become trainer input only after both intrabar paths for one sealed M5 review
block are terminal.  The explicit train months and non-overlapping review
blocks live in the run control rather than in controller code.  A trainer
response must create a new immutable generation; it cannot modify the running
G2 schedule.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_bot_trainer import (
    PROPOSAL_CONTRACT,
    seal_candidate_proposal,
)
from quant_rabbit.dojo_historical_job_archive import (
    ARCHIVE_PART_BYTES,
    ARCHIVE_RECOVERY_CAPACITY_CONTRACT,
    ARCHIVE_RECEIPT_CONTRACT,
    DojoHistoricalJobArchiveError,
    archive_completed_historical_job,
    inspect_historical_job_archive_recovery_capacity,
    verify_existing_historical_job_archive,
)
from quant_rabbit.dojo_historical_raw_reclaim import (
    ATTESTATION_PUBLIC_KEY_SEAL_CONTRACT,
    RECLAIM_RECEIPT_CONTRACT,
    REMOTE_READBACK_ATTESTATION_BODY_CONTRACT,
    REMOTE_READBACK_RECEIPT_CONTRACT,
    REMOTE_READBACK_SIGNED_ATTESTATION_CONTRACT,
    RESTORE_RECEIPT_CONTRACT,
    DojoHistoricalRawReclaimError,
    verify_existing_historical_job_raw_reclaim,
    verify_existing_historical_job_raw_restore,
    verify_historical_job_raw_reclaim,
)
from quant_rabbit.dojo_historical_supersede_receipt import (
    CONTRACT as SUPERSEDE_RECEIPT_CONTRACT,
    DojoHistoricalSupersedeReceiptError,
    verify_historical_supersede_receipt_file,
    verify_historical_supersede_receipt_store_locked,
)
from quant_rabbit.dojo_long_horizon_economic_runner import (
    DojoLongHorizonEconomicRunnerError,
    ECONOMIC_JOB_RESULT_CONTRACT,
    build_sparse_month_source_slice_receipt,
    run_long_horizon_economic_job,
    validate_sparse_month_source_slice_receipt,
)
from quant_rabbit.dojo_long_horizon_execution import (
    CELL_CONTRACT,
    build_long_horizon_coordinate_result,
    claim_next_long_horizon_job,
    initialize_long_horizon_execution_state,
    long_horizon_execution_status,
    record_long_horizon_coordinate_results,
    seal_long_horizon_attempt,
)
from quant_rabbit.dojo_long_horizon_plan import (
    IMPLEMENTATION_DIGEST_KEYS,
    M5_MONTHS,
    RAPID_2025H1_PROFILE,
    build_long_horizon_train_plan,
    canonical_sha256,
    validate_long_horizon_train_plan,
)
from quant_rabbit.dojo_training_rooms import (
    ROOM_TAXONOMY_V2,
    TAXONOMY_V2_REVISION,
)
from quant_rabbit.dojo_long_horizon_schedule import (
    build_long_horizon_stream_schedule,
    validate_long_horizon_stream_schedule,
)
from quant_rabbit.dojo_long_horizon_source_manifest import (
    long_horizon_plan_digest_inputs,
    verify_long_horizon_source_manifest_seal,
)
from quant_rabbit.dojo_portfolio_replay_reducer import seal_portfolio_policy
from quant_rabbit.dojo_sparse_replay import SparseReplayError
from quant_rabbit.dojo_tuned_strategy_runtime import (
    build_tuned_strategy_runtime_factory,
    build_tuned_strategy_runtime_seal,
    verify_tuned_strategy_runtime_seal,
)


CONTROL_CONTRACT: Final = "QR_DOJO_G2_HISTORICAL_RUN_CONTROL_V1"
MANIFEST_CONTRACT: Final = "QR_DOJO_HISTORICAL_TRAIN_GENERATION_MANIFEST_V1"
MILESTONE_CONTRACT: Final = "QR_DOJO_HISTORICAL_TRAIN_MILESTONE_V1"
JOB_COMPLETION_CONTRACT: Final = "QR_DOJO_HISTORICAL_TRAIN_JOB_COMPLETION_V1"
CAPACITY_BASELINE_CONTRACT: Final = "QR_DOJO_HISTORICAL_CAPACITY_BASELINE_V1"
ROOM_STUDY_PROFILE_POLICY_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_ROOM_STUDY_PROFILE_POLICY_V1"
)
SCHEMA_VERSION: Final = 1
GENERATION_ORDINAL: Final = 2
RUNNER_ID: Final = "dojo-g2-historical-train-v1"
MAX_JSON_BYTES: Final = 256 * 1024 * 1024
NON_JPY_STRESS_FINANCING_JPY_PER_UNIT_DAY: Final = 0.02

_AUTHORITY: Final = {
    "automatic_deployment_allowed": False,
    "broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
    "promotion_eligible": False,
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
_LIFECYCLE_BARRIER_POLICY: Final = {
    "archive_before_next_claim": True,
    "signed_remote_attestation_before_next_claim": True,
    "exact_v2_raw_reclaim_before_next_claim": True,
    "max_unreclaimed_terminal_jobs": 1,
    "external_attestor_unavailable_action": "WAIT_WITHOUT_CLAIM",
    "capacity_model": "ONE_WORKING_SET_PLUS_PER_FILESYSTEM_FLOOR",
}
_ROOM_STUDY_PROFILE_POLICY_KEYS: Final = {
    "contract",
    "schema_version",
    "profile_type",
    "train_month_count",
    "train_months_contiguous",
    "review_cadence_months",
    "review_blocks",
    "complete_review_blocks_required",
    "non_overlapping_review_blocks_required",
    "partial_review_block_allowed",
}
_ROOM_REVIEW_BLOCK_KEYS: Final = {"review_block_id", "train_months"}
_MONTH_RE: Final = re.compile(r"[0-9]{4}-(?:0[1-9]|1[0-2])\Z")
_SUPERSEDE_PENDING_ANCHOR_RE: Final = re.compile(
    r"\.supersede-[0-9a-f]{64}-[0-9a-f]{64}\.json" r"\.pending(?:-[0-9]{4})?\Z"
)
_LOCK_BINDINGS: dict[int, tuple[int, Path, str, tuple[int, int], tuple[int, int]]] = {}

_ARTIFACT_NAMES: Final = {
    "control_manifest": "control-manifest.json",
    "plan": "plan.json",
    "proposals": "candidate-proposals.json",
    "resource_policy": "resource-policy.json",
    "runtime_seal": "tuned-runtime-seal.json",
    "schedule": "schedule.json",
    "worker_catalog": "worker-catalog.json",
}


class DojoHistoricalTrainControlError(ValueError):
    """A sealed input, resource gate, or immutable job transition is unsafe."""


def _g2_room_bindings() -> list[dict[str, str]]:
    rows = [
        {"room_id": room.room_id, "family_id": room.strategy_family}
        for room in ROOM_TAXONOMY_V2
        if room.room_id.startswith("room-g2-")
    ]
    rows.sort(key=lambda row: row["room_id"])
    if len(rows) < 2 or len({row["family_id"] for row in rows}) != len(rows):
        raise DojoHistoricalTrainControlError(
            "G2 room taxonomy must bind unique rooms to unique families"
        )
    return rows


def _configured_room_bindings(fixed: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = fixed.get("room_bindings")
    if not isinstance(raw, list) or len(raw) < 2:
        raise DojoHistoricalTrainControlError(
            "room generation requires at least two configured room bindings"
        )
    rows = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != {"room_id", "family_id"}:
            raise DojoHistoricalTrainControlError(
                f"room_bindings[{index}] schema is not exact"
            )
        row = {"room_id": item["room_id"], "family_id": item["family_id"]}
        if not all(isinstance(value, str) and value for value in row.values()):
            raise DojoHistoricalTrainControlError(
                f"room_bindings[{index}] identity is invalid"
            )
        rows.append(row)
    if rows != sorted(rows, key=lambda row: row["room_id"]):
        raise DojoHistoricalTrainControlError("room bindings must be sorted")
    if len({row["room_id"] for row in rows}) != len(rows) or len(
        {row["family_id"] for row in rows}
    ) != len(rows):
        raise DojoHistoricalTrainControlError(
            "room bindings must be a room-to-family bijection"
        )
    return rows


def _room_binding_sha256(
    room_bindings: Sequence[Mapping[str, str]] | None = None,
    *,
    taxonomy_revision: str = TAXONOMY_V2_REVISION,
) -> str:
    return canonical_sha256(
        {
            "taxonomy_revision": taxonomy_revision,
            "room_family_bindings": list(room_bindings or _g2_room_bindings()),
        }
    )


def _is_room_generation(fixed: Mapping[str, Any]) -> bool:
    generation = fixed.get("generation")
    study_profile = fixed.get("study_profile")
    return (
        isinstance(generation, str)
        and bool(generation)
        and generation != "G2"
        and isinstance(study_profile, str)
        and bool(study_profile)
    )


def _month_ordinal(value: Any, *, field: str) -> int:
    if not isinstance(value, str) or _MONTH_RE.fullmatch(value) is None:
        raise DojoHistoricalTrainControlError(f"{field} is not a canonical month")
    year, month = (int(part) for part in value.split("-"))
    return year * 12 + month - 1


def _validate_room_study_profile_policy(
    value: Any,
    *,
    train_months: Any,
    trainer_milestones: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one sealed, fully explicit room study/review partition."""

    if not isinstance(value, Mapping) or set(value) != _ROOM_STUDY_PROFILE_POLICY_KEYS:
        raise DojoHistoricalTrainControlError(
            "room study profile policy schema is not exact"
        )
    if (
        value.get("contract") != ROOM_STUDY_PROFILE_POLICY_CONTRACT
        or value.get("schema_version") != SCHEMA_VERSION
        or isinstance(value.get("schema_version"), bool)
        or value.get("profile_type") != "EXPLICIT_CONTIGUOUS_M5_ISOLATED_ROOM_TRAIN"
        or value.get("train_months_contiguous") is not True
        or value.get("complete_review_blocks_required") is not True
        or value.get("non_overlapping_review_blocks_required") is not True
        or value.get("partial_review_block_allowed") is not False
    ):
        raise DojoHistoricalTrainControlError(
            "room study profile policy boundary drifted"
        )
    if not isinstance(train_months, list) or not train_months:
        raise DojoHistoricalTrainControlError("room train months are missing")
    if len(train_months) > len(M5_MONTHS):
        raise DojoHistoricalTrainControlError("room train window is unbounded")
    ordinals = [
        _month_ordinal(month, field=f"train_months[{index}]")
        for index, month in enumerate(train_months)
    ]
    if len(set(train_months)) != len(train_months):
        raise DojoHistoricalTrainControlError("room train months contain duplicates")
    if ordinals != list(range(ordinals[0], ordinals[0] + len(ordinals))):
        raise DojoHistoricalTrainControlError(
            "room train months must be one contiguous calendar window"
        )
    if any(month not in M5_MONTHS for month in train_months):
        raise DojoHistoricalTrainControlError(
            "room train months exceed sealed M5 source coverage"
        )
    month_count = value.get("train_month_count")
    cadence = value.get("review_cadence_months")
    if (
        isinstance(month_count, bool)
        or not isinstance(month_count, int)
        or month_count != len(train_months)
        or isinstance(cadence, bool)
        or not isinstance(cadence, int)
        or cadence < 1
        or cadence > month_count
        or month_count % cadence != 0
    ):
        raise DojoHistoricalTrainControlError(
            "room train length and review cadence do not form complete blocks"
        )
    if trainer_milestones.get("m5_completed_months_per_review") != cadence:
        raise DojoHistoricalTrainControlError(
            "trainer milestone cadence differs from room study profile"
        )
    if trainer_milestones.get("non_overlapping_review_blocks_required") is not True:
        raise DojoHistoricalTrainControlError(
            "trainer milestone does not require non-overlapping review blocks"
        )
    raw_blocks = value.get("review_blocks")
    if not isinstance(raw_blocks, list) or len(raw_blocks) != month_count // cadence:
        raise DojoHistoricalTrainControlError(
            "room review block count differs from its sealed cadence"
        )
    blocks: list[dict[str, Any]] = []
    flattened: list[str] = []
    for index, raw in enumerate(raw_blocks, start=1):
        if not isinstance(raw, Mapping) or set(raw) != _ROOM_REVIEW_BLOCK_KEYS:
            raise DojoHistoricalTrainControlError(
                f"room review_blocks[{index - 1}] schema is not exact"
            )
        expected_id = f"review-block-{index:04d}"
        months = raw.get("train_months")
        if raw.get("review_block_id") != expected_id or not isinstance(months, list):
            raise DojoHistoricalTrainControlError(
                f"room review block {index} identity is invalid"
            )
        if len(months) != cadence:
            raise DojoHistoricalTrainControlError(
                f"room review block {index} does not match cadence"
            )
        flattened.extend(months)
        blocks.append({"review_block_id": expected_id, "train_months": list(months)})
    if len(set(flattened)) != len(flattened):
        raise DojoHistoricalTrainControlError("room review blocks overlap")
    if flattened != train_months:
        raise DojoHistoricalTrainControlError(
            "room review blocks do not exactly partition train months"
        )
    return {
        key: ([dict(row) for row in blocks] if key == "review_blocks" else value[key])
        for key in sorted(_ROOM_STUDY_PROFILE_POLICY_KEYS)
    }


def _generation_ordinal(fixed: Mapping[str, Any]) -> int:
    value = fixed.get("strategy_generation_ordinal", GENERATION_ORDINAL)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 10_000
    ):
        raise DojoHistoricalTrainControlError("strategy generation ordinal is invalid")
    return value


def _runner_id(fixed: Mapping[str, Any]) -> str:
    if not _is_room_generation(fixed):
        return RUNNER_ID
    return (
        "dojo-room-slot-01-"
        + canonical_sha256(
            {
                "generation": fixed["generation"],
                "registry_artifact_sha256": fixed["registry_artifact_sha256"],
                "room_family_bindings_sha256": fixed["room_family_bindings_sha256"],
            }
        )[:16]
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoHistoricalTrainControlError(
            "artifact is not strict canonical JSON"
        ) from exc


def _stable_regular_bytes(
    path: Path, *, field: str, maximum: int = MAX_JSON_BYTES
) -> bytes:
    candidate = Path(path)
    try:
        before = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalTrainControlError(f"{field} is unavailable") from exc
    if (
        candidate.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or not 0 < before.st_size <= maximum
    ):
        raise DojoHistoricalTrainControlError(
            f"{field} must be a bounded nonempty regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        raw = handle.read(maximum + 1)
        opened = os.fstat(handle.fileno())
    current = candidate.stat(follow_symlinks=False)
    identities = {
        (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        for row in (before, opened, current)
    }
    if len(identities) != 1 or len(raw) != before.st_size:
        raise DojoHistoricalTrainControlError(f"{field} changed while read")
    return raw


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    raw = _stable_regular_bytes(path, field=field)

    def reject_constant(token: str) -> None:
        raise DojoHistoricalTrainControlError(
            f"{field} contains non-finite JSON: {token}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DojoHistoricalTrainControlError(
                    f"{field} contains duplicate key: {key}"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoHistoricalTrainControlError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoHistoricalTrainControlError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise DojoHistoricalTrainControlError(f"{field} must be a JSON object")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _stable_regular_bytes(path, field="immutable artifact") != payload:
            raise DojoHistoricalTrainControlError(
                f"immutable artifact already exists with different bytes: {path}"
            )
        return
    temporary = path.with_name(
        f".{path.name}.{hashlib.sha256(payload).hexdigest()}.{os.getpid()}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            if handle.write(payload) != len(payload):
                raise DojoHistoricalTrainControlError(
                    "immutable artifact write was incomplete"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _file_inventory(
    repo_root: Path, relative_paths: Sequence[str]
) -> list[dict[str, Any]]:
    rows = []
    for relative in relative_paths:
        path = repo_root / relative
        state = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(state.st_mode) or state.st_size <= 0:
            raise DojoHistoricalTrainControlError(
                f"implementation dependency is invalid: {relative}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {"relative_path": relative, "size_bytes": state.st_size, "sha256": digest}
        )
    return rows


def _file_binding(repo_root: Path, relative_paths: Sequence[str]) -> str:
    return canonical_sha256(_file_inventory(repo_root, relative_paths))


def _verified_capacity_baseline_artifact(
    *, repo_root: Path, baseline: Mapping[str, Any]
) -> dict[str, Any]:
    artifact_path_value = baseline.get("baseline_artifact_path")
    configured_artifact_sha = baseline.get("baseline_artifact_sha256")
    if not isinstance(artifact_path_value, str) or not artifact_path_value:
        raise DojoHistoricalTrainControlError(
            "room capacity baseline artifact path is invalid"
        )
    artifact_path = Path(artifact_path_value)
    if not artifact_path.is_absolute():
        artifact_path = repo_root / artifact_path
    raw = _stable_regular_bytes(artifact_path, field="capacity baseline artifact")
    if (
        not _lower_hex_sha(configured_artifact_sha)
        or hashlib.sha256(raw).hexdigest() != configured_artifact_sha
    ):
        raise DojoHistoricalTrainControlError(
            "capacity baseline artifact bytes differ from the sealed input"
        )
    artifact = _read_json(artifact_path, field="capacity baseline artifact")
    expected_keys = {
        "contract",
        "schema_version",
        "baseline_id",
        "job_sha256",
        "coordinate_count",
        "measured_raw_bytes",
        "measured_source_bytes",
        "completion_path",
        "completion_file_sha256",
        "completion_sha256",
        "archive_receipt_path",
        "archive_receipt_file_sha256",
        "archive_receipt_sha256",
        "source_slice_receipt_path",
        "source_slice_receipt_file_sha256",
        "source_slice_receipt_sha256",
        "artifact_sha256",
    }
    artifact_body = {
        key: value for key, value in artifact.items() if key != "artifact_sha256"
    }
    integer_keys = (
        "coordinate_count",
        "measured_raw_bytes",
        "measured_source_bytes",
    )
    if (
        set(artifact) != expected_keys
        or artifact.get("contract") != CAPACITY_BASELINE_CONTRACT
        or artifact.get("schema_version") != SCHEMA_VERSION
        or isinstance(artifact.get("schema_version"), bool)
        or artifact.get("baseline_id") != baseline.get("baseline_id")
        or not _lower_hex_sha(artifact.get("job_sha256"))
        or any(
            isinstance(artifact.get(key), bool)
            or not isinstance(artifact.get(key), int)
            or artifact[key] < 1
            for key in integer_keys
        )
        or artifact["measured_source_bytes"] >= artifact["measured_raw_bytes"]
        or artifact.get("artifact_sha256") != canonical_sha256(artifact_body)
    ):
        raise DojoHistoricalTrainControlError(
            "capacity baseline artifact schema or seal is invalid"
        )

    linked: dict[str, tuple[dict[str, Any], str]] = {}
    for prefix in ("completion", "archive_receipt", "source_slice_receipt"):
        path_value = artifact.get(f"{prefix}_path")
        expected_file_sha = artifact.get(f"{prefix}_file_sha256")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise DojoHistoricalTrainControlError(
                "capacity baseline evidence path is invalid"
            )
        evidence_path = Path(path_value)
        evidence_raw = _stable_regular_bytes(
            evidence_path, field=f"capacity baseline {prefix}"
        )
        observed_file_sha = hashlib.sha256(evidence_raw).hexdigest()
        if (
            not _lower_hex_sha(expected_file_sha)
            or observed_file_sha != expected_file_sha
        ):
            raise DojoHistoricalTrainControlError(
                "capacity baseline evidence bytes differ from the artifact"
            )
        linked[prefix] = (
            _read_json(evidence_path, field=f"capacity baseline {prefix}"),
            observed_file_sha,
        )

    completion = linked["completion"][0]
    completion_body = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    archive_receipt = linked["archive_receipt"][0]
    archive_body = {
        key: value for key, value in archive_receipt.items() if key != "receipt_sha256"
    }
    source_receipt = linked["source_slice_receipt"][0]
    source_body = {
        key: value
        for key, value in source_receipt.items()
        if key != "source_slice_receipt_sha256"
    }
    job_sha = artifact["job_sha256"]
    coordinate_count = artifact["coordinate_count"]
    if (
        completion.get("contract") != JOB_COMPLETION_CONTRACT
        or completion.get("job_sha256") != job_sha
        or completion.get("job_status") != "COMPLETE"
        or completion.get("complete_coordinate_count") != coordinate_count
        or completion.get("coordinate_result_count") != coordinate_count
        or completion.get("failed_coordinate_count") != 0
        or completion.get("completion_sha256") != canonical_sha256(completion_body)
        or completion.get("completion_sha256") != artifact["completion_sha256"]
        or archive_receipt.get("contract") != ARCHIVE_RECEIPT_CONTRACT
        or archive_receipt.get("job_sha256") != job_sha
        or archive_receipt.get("completion_sha256") != completion["completion_sha256"]
        or archive_receipt.get("local_payload_verified") is not True
        or archive_receipt.get("receipt_sha256") != canonical_sha256(archive_body)
        or archive_receipt.get("receipt_sha256") != artifact["archive_receipt_sha256"]
        or archive_receipt.get("total_source_bytes") != artifact["measured_raw_bytes"]
        or source_receipt.get("job_sha256") != job_sha
        or source_receipt.get("file_size_bytes") != artifact["measured_source_bytes"]
        or source_receipt.get("source_slice_receipt_sha256")
        != canonical_sha256(source_body)
        or source_receipt.get("source_slice_receipt_sha256")
        != artifact["source_slice_receipt_sha256"]
    ):
        raise DojoHistoricalTrainControlError(
            "capacity baseline does not reproduce verified r7 evidence"
        )
    if (
        baseline.get("coordinate_count") != coordinate_count
        or baseline.get("raw_bytes_per_job") != artifact["measured_raw_bytes"]
        or baseline.get("source_bytes_per_job") != artifact["measured_source_bytes"]
    ):
        raise DojoHistoricalTrainControlError(
            "capacity baseline claims differ from their sealed artifact"
        )
    return artifact


def _custody_control_plane_binding(repo_root: Path) -> dict[str, Any]:
    relative_paths = (
        "src/quant_rabbit/dojo_historical_train_control.py",
        "src/quant_rabbit/dojo_historical_job_archive.py",
        "src/quant_rabbit/dojo_historical_raw_reclaim.py",
        "src/quant_rabbit/dojo_historical_supersede_receipt.py",
        "scripts/run-dojo-historical-train-control.py",
        "scripts/run-dojo-historical-raw-reclaim.py",
        "scripts/run-dojo-historical-supersede-receipt.py",
    )
    inventory = _file_inventory(repo_root, relative_paths)
    body = {
        "compatibility_contract": "QR_DOJO_CUSTODY_CONTROL_PLANE_COMPATIBILITY_V1",
        "inventory": inventory,
        "inventory_sha256": canonical_sha256(inventory),
    }
    return {**body, "binding_sha256": canonical_sha256(body)}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _archive_runtime_binding(control: Mapping[str, Any]) -> dict[str, Any] | None:
    """Revalidate the sealed local staging and zstd runtime for new rooms."""

    fixed = control["fixed_inputs"]
    execution = control["execution"]
    required = _is_room_generation(fixed) and _generation_ordinal(fixed) >= 3
    staging_value = execution.get("archive_local_staging_root")
    zstd_value = execution.get("archive_zstd_executable")
    if not required:
        return None
    if not isinstance(staging_value, str) or not isinstance(zstd_value, str):
        raise DojoHistoricalTrainControlError(
            "new room generations require local archive staging and zstd"
        )
    staging_configured = Path(staging_value)
    zstd_configured = Path(zstd_value)
    try:
        staging = staging_configured.resolve(strict=True)
        zstd = zstd_configured.resolve(strict=True)
        staging_state = staging.stat(follow_symlinks=False)
        zstd_before = zstd.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalTrainControlError(
            "sealed archive runtime is unavailable"
        ) from exc
    if staging != staging_configured or not stat.S_ISDIR(staging_state.st_mode):
        raise DojoHistoricalTrainControlError(
            "archive local staging root must be an existing resolved directory"
        )
    if "CloudStorage" in staging.parts:
        raise DojoHistoricalTrainControlError(
            "archive local staging root cannot use a CloudStorage filesystem"
        )
    output_root = Path(execution["output_root"]).resolve(strict=False)
    archive_root_value = execution.get("archive_root")
    archive_root = (
        Path(archive_root_value).resolve(strict=False)
        if isinstance(archive_root_value, str)
        else None
    )
    if _is_relative_to(staging, output_root) or (
        archive_root is not None and _is_relative_to(staging, archive_root)
    ):
        raise DojoHistoricalTrainControlError(
            "archive local staging root must be outside run and archive roots"
        )
    if (
        zstd != zstd_configured
        or not stat.S_ISREG(zstd_before.st_mode)
        or zstd_before.st_size < 1
        or zstd_before.st_size > 512 * 1024**2
        or not os.access(zstd, os.X_OK)
    ):
        raise DojoHistoricalTrainControlError(
            "archive zstd executable must be a resolved bounded executable"
        )
    try:
        descriptor = os.open(
            zstd,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            zstd_bytes = handle.read(512 * 1024**2 + 1)
            zstd_opened = os.fstat(handle.fileno())
        version_process = subprocess.run(
            [os.fspath(zstd), "--version"],
            check=False,
            capture_output=True,
            timeout=10,
        )
        zstd_after = zstd.stat(follow_symlinks=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DojoHistoricalTrainControlError(
            "archive zstd executable cannot be revalidated"
        ) from exc
    if (
        len(
            {
                (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
                for row in (zstd_before, zstd_opened, zstd_after)
            }
        )
        != 1
        or len(zstd_bytes) != zstd_before.st_size
        or version_process.returncode != 0
    ):
        raise DojoHistoricalTrainControlError(
            "archive zstd executable changed during verification"
        )
    try:
        version = (
            (version_process.stdout + version_process.stderr)
            .decode("utf-8", errors="strict")
            .strip()
        )
    except UnicodeDecodeError as exc:
        raise DojoHistoricalTrainControlError(
            "archive zstd version output is not UTF-8"
        ) from exc
    if not version or len(version.encode("utf-8")) > 4096:
        raise DojoHistoricalTrainControlError("archive zstd version output is invalid")
    observed = {
        "local_staging_root": os.fspath(staging),
        "local_staging_device": int(staging_state.st_dev),
        "zstd_executable": os.fspath(zstd),
        "zstd_sha256": hashlib.sha256(zstd_bytes).hexdigest(),
        "zstd_size_bytes": len(zstd_bytes),
        "zstd_version": version,
    }
    configured = {
        "zstd_sha256": execution.get("archive_zstd_sha256"),
        "zstd_size_bytes": execution.get("archive_zstd_size_bytes"),
        "zstd_version": execution.get("archive_zstd_version"),
    }
    if (
        not isinstance(configured["zstd_sha256"], str)
        or len(configured["zstd_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in configured["zstd_sha256"]
        )
        or isinstance(configured["zstd_size_bytes"], bool)
        or not isinstance(configured["zstd_size_bytes"], int)
        or configured["zstd_size_bytes"] < 1
        or not isinstance(configured["zstd_version"], str)
        or not configured["zstd_version"]
    ):
        raise DojoHistoricalTrainControlError("archive zstd sealed metadata is invalid")
    if configured != {
        key.removeprefix("archive_"): value
        for key, value in {
            "archive_zstd_sha256": observed["zstd_sha256"],
            "archive_zstd_size_bytes": observed["zstd_size_bytes"],
            "archive_zstd_version": observed["zstd_version"],
        }.items()
    }:
        raise DojoHistoricalTrainControlError(
            "archive zstd executable differs from its sealed metadata"
        )
    return observed


def _verified_control(
    run_control_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    control = _read_json(run_control_path, field="run control")
    if (
        control.get("contract") != CONTROL_CONTRACT
        or control.get("schema_version") != 1
    ):
        raise DojoHistoricalTrainControlError("run-control contract/version drifted")
    authority = control.get("authority")
    if not isinstance(authority, Mapping) or (
        authority.get("historical_replay_process_start_allowed") is not True
        or authority.get("research_filesystem_write_allowed") is not True
        or authority.get("broker_mutation_allowed") is not False
        or authority.get("live_permission") is not False
        or authority.get("order_authority") != "NONE"
    ):
        raise DojoHistoricalTrainControlError("run-control authority is unsafe")
    fixed = control.get("fixed_inputs")
    execution = control.get("execution")
    milestone = control.get("trainer_milestones")
    if not all(isinstance(row, Mapping) for row in (fixed, execution, milestone)):
        raise DojoHistoricalTrainControlError("run-control sections are malformed")
    legacy_generation = (
        fixed.get("generation") == "G2"
        and fixed.get("study_profile") == RAPID_2025H1_PROFILE
    )
    room_generation = _is_room_generation(fixed)
    room_study_profile_policy: dict[str, Any] | None = None
    if room_generation:
        configured_rooms = _configured_room_bindings(fixed)
        taxonomy_revision = fixed.get("room_taxonomy_revision")
        if (
            not isinstance(taxonomy_revision, str)
            or not taxonomy_revision
            or fixed.get("room_family_bindings_sha256")
            != _room_binding_sha256(
                configured_rooms, taxonomy_revision=taxonomy_revision
            )
        ):
            raise DojoHistoricalTrainControlError(
                "run-control G2 room taxonomy binding drifted"
            )
        ordinal = _generation_ordinal(fixed)
        train_months = fixed.get("train_months")
        if ordinal >= 3:
            room_study_profile_policy = _validate_room_study_profile_policy(
                fixed.get("study_profile_policy"),
                train_months=train_months,
                trainer_milestones=milestone,
            )
        elif fixed.get("study_profile_policy") is not None:
            raise DojoHistoricalTrainControlError(
                "legacy room generation cannot acquire a later study profile policy"
            )
        elif not isinstance(train_months, list) or not train_months:
            raise DojoHistoricalTrainControlError(
                "legacy room run-control train window is missing"
            )
        _cost_profiles(control)
    legacy_milestone_boundary = (
        milestone.get("m5_completed_months_per_review") == 6
        and milestone.get("non_overlapping_six_month_blocks_required") is True
    )
    room_milestone_boundary = (
        room_study_profile_policy is not None
        and milestone.get("m5_completed_months_per_review")
        == room_study_profile_policy["review_cadence_months"]
        and milestone.get("non_overlapping_review_blocks_required") is True
    )
    if (
        not (legacy_generation or room_generation)
        or execution.get("evidence_tier") != "WORN_HISTORICAL_TRAIN_ONLY"
        or execution.get("max_parallel_jobs") != 1
        or execution.get("max_jobs_per_invocation") != 1
        or not (
            room_milestone_boundary
            if room_generation and _generation_ordinal(fixed) >= 3
            else legacy_milestone_boundary
        )
        or milestone.get("minimum_completed_intrabar_paths_per_month") != 2
        or milestone.get("rapid_train_evaluation_mode") != "INDEPENDENT_MONTH"
        or milestone.get("continuous_account_role")
        != "SEPARATE_LONG_PROFILE_NOT_RAPID_TUNING_GATE"
        or milestone.get("partial_month_tuning_allowed") is not False
        or milestone.get("parameter_change_applies_only_to_new_generation") is not True
        or milestone.get("target_multiple_may_backsolve_sizing") is not False
    ):
        raise DojoHistoricalTrainControlError("run-control fixed boundary drifted")
    for key in ("registry_path", "source_manifest_path"):
        relative = fixed.get(key)
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise DojoHistoricalTrainControlError(f"run-control {key} is unsafe")
        if not (repo_root / relative).is_file():
            raise DojoHistoricalTrainControlError(f"run-control {key} is missing")
    output_root = execution.get("output_root")
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        raise DojoHistoricalTrainControlError("run-control output root is invalid")
    archive_root = execution.get("archive_root")
    if room_generation and (
        not isinstance(archive_root, str) or not Path(archive_root).is_absolute()
    ):
        raise DojoHistoricalTrainControlError(
            "room run-control archive root is invalid"
        )
    archive_local_staging_root = execution.get("archive_local_staging_root")
    if archive_local_staging_root is not None and (
        not isinstance(archive_local_staging_root, str)
        or not Path(archive_local_staging_root).is_absolute()
    ):
        raise DojoHistoricalTrainControlError(
            "run-control archive local staging root is invalid"
        )
    archive_zstd_executable = execution.get("archive_zstd_executable")
    if archive_zstd_executable is not None and (
        not isinstance(archive_zstd_executable, str)
        or not Path(archive_zstd_executable).is_absolute()
    ):
        raise DojoHistoricalTrainControlError(
            "run-control archive zstd executable is invalid"
        )
    if archive_zstd_executable is not None:
        try:
            resolved_zstd = Path(archive_zstd_executable).resolve(strict=True)
            zstd_state = resolved_zstd.stat(follow_symlinks=False)
        except OSError as exc:
            raise DojoHistoricalTrainControlError(
                "run-control archive zstd executable is unavailable"
            ) from exc
        if not stat.S_ISREG(zstd_state.st_mode) or not os.access(
            resolved_zstd, os.X_OK
        ):
            raise DojoHistoricalTrainControlError(
                "run-control archive zstd executable is not executable"
            )
    if room_generation:
        resource_bounds = {
            "bootstrap_job_working_set_bytes": 1024**3,
            "max_rss_bytes": 1024**3,
            "max_open_files": 128,
            "max_checkpoint_bytes": 1024**2,
            "max_terminal_bytes": 1024**2,
        }
        for key, minimum in resource_bounds.items():
            value = execution.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise DojoHistoricalTrainControlError(
                    f"room run-control {key} is invalid"
                )
        for key in (
            "archive_drive_root_id",
            "archive_drive_readback_parent_id",
            "archive_drive_remote_receipts_parent_id",
        ):
            value = execution.get(key)
            if (
                not isinstance(value, str)
                or not 8 <= len(value) <= 256
                or any(
                    character
                    not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
                    for character in value
                )
            ):
                raise DojoHistoricalTrainControlError(
                    f"room run-control {key} is invalid"
                )
        global_lock_path = execution.get("global_heavy_lock_path")
        if (
            not isinstance(global_lock_path, str)
            or not Path(global_lock_path).is_absolute()
        ):
            raise DojoHistoricalTrainControlError(
                "room run-control global heavy lock is invalid"
            )
        conflicting_roots = execution.get("conflicting_execution_roots")
        if not isinstance(conflicting_roots, list) or not all(
            isinstance(value, str) and Path(value).is_absolute()
            for value in conflicting_roots
        ):
            raise DojoHistoricalTrainControlError(
                "room run-control conflicting execution roots are invalid"
            )
        conflicting_locks = execution.get("conflicting_run_lock_paths")
        if (
            not isinstance(conflicting_locks, list)
            or len(conflicting_locks) != len(conflicting_roots)
            or not all(
                isinstance(value, str) and Path(value).is_absolute()
                for value in conflicting_locks
            )
            or any(
                Path(lock) != Path(root) / ".historical-train.lock"
                for root, lock in zip(conflicting_roots, conflicting_locks, strict=True)
            )
        ):
            raise DojoHistoricalTrainControlError(
                "room run-control conflicting locks must map one-to-one to roots"
            )
        baseline = execution.get("capacity_baseline")
        required_baseline_keys = {
            "baseline_id",
            "coordinate_count",
            "raw_bytes_per_job",
            "source_bytes_per_job",
        }
        if _generation_ordinal(fixed) >= 3:
            required_baseline_keys |= {
                "baseline_artifact_path",
                "baseline_artifact_sha256",
            }
        if not isinstance(baseline, Mapping) or set(baseline) != required_baseline_keys:
            raise DojoHistoricalTrainControlError(
                "room run-control capacity baseline schema is invalid"
            )
        if not isinstance(baseline["baseline_id"], str) or not baseline["baseline_id"]:
            raise DojoHistoricalTrainControlError(
                "room run-control capacity baseline id is invalid"
            )
        for key in ("coordinate_count", "raw_bytes_per_job", "source_bytes_per_job"):
            value = baseline[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DojoHistoricalTrainControlError(
                    f"room run-control capacity baseline {key} is invalid"
                )
        if baseline["source_bytes_per_job"] >= baseline["raw_bytes_per_job"]:
            raise DojoHistoricalTrainControlError(
                "room run-control capacity baseline source bytes are invalid"
            )
        if _generation_ordinal(fixed) >= 3:
            _verified_capacity_baseline_artifact(
                repo_root=repo_root,
                baseline=baseline,
            )
            if execution.get("lifecycle_barrier_policy") != _LIFECYCLE_BARRIER_POLICY:
                raise DojoHistoricalTrainControlError(
                    "new room lifecycle barrier policy is invalid"
                )
        if _generation_ordinal(fixed) >= 3:
            if (
                execution.get("allowed_command")
                != "scripts/run-dojo-historical-train-control.py step"
            ):
                raise DojoHistoricalTrainControlError(
                    "new room generation heartbeat command must be exact step"
                )
            _archive_runtime_binding(control)
    minimum_free = execution.get("minimum_free_disk_bytes")
    if (
        isinstance(minimum_free, bool)
        or not isinstance(minimum_free, int)
        or minimum_free < 20 * 1024**3
    ):
        raise DojoHistoricalTrainControlError("run-control disk floor is too low")
    return control


def _registry_artifact(registry: Mapping[str, Any]) -> str:
    artifact = registry.get("artifact_sha256")
    if not isinstance(artifact, str) or len(artifact) != 64:
        raise DojoHistoricalTrainControlError("G2 registry artifact SHA is invalid")
    body = {key: value for key, value in registry.items() if key != "artifact_sha256"}
    if canonical_sha256(body) != artifact:
        raise DojoHistoricalTrainControlError("G2 registry artifact seal drifted")
    return artifact


def _candidate_proposals(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    workers = registry.get("workers")
    if not isinstance(workers, list) or len(workers) < 2:
        raise DojoHistoricalTrainControlError("G2 registry has no strategy cohort")
    proposals = []
    for worker in workers:
        if not isinstance(worker, Mapping):
            raise DojoHistoricalTrainControlError("G2 worker row is malformed")
        proposal = seal_candidate_proposal(
            {
                "contract": PROPOSAL_CONTRACT,
                "schema_version": 1,
                "candidate_id": worker.get("worker_id"),
                "family": worker.get("family"),
                "hypothesis": (
                    f"G2 fixed {worker.get('family')} family under the "
                    "generation-sealed allocator envelope."
                ),
                "config": worker.get("config"),
                "risk_increase": False,
            }
        )
        if proposal["config_sha256"] != worker.get("config_sha256") or proposal[
            "catalog_sha256"
        ] != worker.get("catalog_sha256"):
            raise DojoHistoricalTrainControlError(
                "G2 worker config/catalog binding drifted"
            )
        proposals.append(proposal)
    proposals.sort(key=lambda row: row["candidate_id"])
    return proposals


def _cost_profiles(
    control: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    defaults = {
        "BASE": {
            "financing_policy": "ZERO_BASE_DIAGNOSTIC",
            "jpy_quote_financing_jpy_per_unit_day": 0.0,
            "non_jpy_quote_financing_jpy_per_unit_day": 0.0,
            "slippage_pips_per_adverse_fill": 0.0,
        },
        "STRESS": {
            "financing_policy": "DECLARED_FIXED_JPY_CONSERVATIVE_PROXY",
            "jpy_quote_financing_jpy_per_unit_day": 0.008,
            "non_jpy_quote_financing_jpy_per_unit_day": (
                NON_JPY_STRESS_FINANCING_JPY_PER_UNIT_DAY
            ),
            "slippage_pips_per_adverse_fill": 0.3,
        },
    }
    raw = (
        control.get("fixed_inputs", {}).get("cost_profiles")
        if isinstance(control, Mapping)
        and isinstance(control.get("fixed_inputs"), Mapping)
        else None
    )
    if raw is None:
        return defaults
    keys = {
        "financing_policy",
        "jpy_quote_financing_jpy_per_unit_day",
        "non_jpy_quote_financing_jpy_per_unit_day",
        "slippage_pips_per_adverse_fill",
    }
    if not isinstance(raw, Mapping) or set(raw) != {"BASE", "STRESS"}:
        raise DojoHistoricalTrainControlError(
            "cost profiles must contain exact BASE and STRESS arms"
        )
    result = {}
    for scenario in ("BASE", "STRESS"):
        row = raw[scenario]
        if not isinstance(row, Mapping) or set(row) != keys:
            raise DojoHistoricalTrainControlError(
                f"{scenario} cost profile schema is not exact"
            )
        if not isinstance(row["financing_policy"], str) or not row["financing_policy"]:
            raise DojoHistoricalTrainControlError(
                f"{scenario} financing policy is invalid"
            )
        values = {key: row[key] for key in keys if key != "financing_policy"}
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in values.values()
        ):
            raise DojoHistoricalTrainControlError(
                f"{scenario} cost profile contains an invalid numeric value"
            )
        result[scenario] = {
            "financing_policy": row["financing_policy"],
            **{key: float(value) for key, value in values.items()},
        }
    for key in keys - {"financing_policy"}:
        if result["STRESS"][key] < result["BASE"][key]:
            raise DojoHistoricalTrainControlError(f"STRESS cost {key} is below BASE")
    return result


def _risk_envelope(registry: Mapping[str, Any]) -> dict[str, Any]:
    allocator = registry.get("allocator")
    if not isinstance(allocator, Mapping):
        raise DojoHistoricalTrainControlError("allocator envelope is missing")
    broker_leverage = allocator.get("broker_leverage", 25.0)
    maximum_gross = allocator.get("maximum_gross_leverage")
    per_position = allocator.get("per_position_leverage")
    slots = allocator.get("simultaneous_slots")
    closeout = allocator.get("margin_closeout_fraction")
    admission = allocator.get("new_position_margin_admission_fraction_max")
    stop_risk = allocator.get("portfolio_stop_risk_fraction")
    per_family = allocator.get("max_concurrent_per_family")
    per_pair = allocator.get("max_concurrent_per_pair")
    numeric = (
        broker_leverage,
        maximum_gross,
        per_position,
        closeout,
        admission,
        stop_risk,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric
    ) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (slots, per_family, per_pair)
    ):
        raise DojoHistoricalTrainControlError("allocator envelope types are invalid")
    if not (
        0 < float(per_position) <= float(maximum_gross) <= float(broker_leverage) <= 25
        and math.isclose(
            float(maximum_gross),
            float(per_position) * int(slots),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and 0 < float(admission) < float(closeout) < 1
        and 0 < float(stop_risk) <= 0.25
        and int(per_family) <= int(slots)
        and int(per_pair) <= int(slots)
    ):
        raise DojoHistoricalTrainControlError(
            "allocator violates the sealed safety relationships"
        )
    max_lock_seconds = allocator.get("max_lock_seconds", 14_400)
    if (
        isinstance(max_lock_seconds, bool)
        or not isinstance(max_lock_seconds, int)
        or not 60 <= max_lock_seconds <= 86_400
    ):
        raise DojoHistoricalTrainControlError("allocator lock horizon is unsafe")
    expected = {
        "leverage": float(broker_leverage),
        "margin_closeout_fraction": float(closeout),
        "max_currency_gross_notional_fraction": allocator.get(
            "max_currency_gross_notional_fraction", maximum_gross
        ),
        "max_cluster_gross_notional_fraction": allocator.get(
            "max_cluster_gross_notional_fraction", maximum_gross
        ),
        "max_lock_seconds": max_lock_seconds,
        "max_margin_utilization_fraction": float(admission),
        "max_open_and_pending_per_family": int(per_family),
        "max_open_and_pending_per_pair": int(per_pair),
        "max_open_and_pending_total": int(slots),
        "max_portfolio_stop_risk_fraction": float(stop_risk),
        "maximum_gross_leverage": float(maximum_gross),
        "per_position_leverage": float(per_position),
    }
    for key in (
        "max_currency_gross_notional_fraction",
        "max_cluster_gross_notional_fraction",
    ):
        value = expected[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not float(per_position) <= float(value) <= float(broker_leverage)
        ):
            raise DojoHistoricalTrainControlError(f"allocator {key} is unsafe")
        expected[key] = float(value)
    return expected


def _implementation_digests(
    *,
    repo_root: Path,
    runtime_seal: Mapping[str, Any],
    registry: Mapping[str, Any],
    control: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    costs = _cost_profiles(control)
    separated_control_plane = (
        control is not None
        and _is_room_generation(control["fixed_inputs"])
        and _generation_ordinal(control["fixed_inputs"]) >= 3
    )
    economic_semantic_paths = (
        "src/quant_rabbit/dojo_long_horizon_economic_runner.py",
        "src/quant_rabbit/dojo_long_horizon_execution.py",
        "src/quant_rabbit/dojo_long_horizon_plan.py",
        "src/quant_rabbit/dojo_long_horizon_schedule.py",
        "src/quant_rabbit/dojo_market_calendar.py",
        "src/quant_rabbit/dojo_portfolio_replay_reducer.py",
        "src/quant_rabbit/dojo_shared_worker_protocol.py",
        "src/quant_rabbit/dojo_sparse_replay.py",
        "src/quant_rabbit/dojo_sparse_source_slice_v2.py",
    )
    legacy_combined_paths = economic_semantic_paths + (
        "src/quant_rabbit/dojo_historical_train_control.py",
        "src/quant_rabbit/dojo_historical_job_archive.py",
        "src/quant_rabbit/dojo_historical_raw_reclaim.py",
        "src/quant_rabbit/dojo_historical_supersede_receipt.py",
        "scripts/run-dojo-historical-raw-reclaim.py",
        "scripts/run-dojo-historical-supersede-receipt.py",
    )
    replay_engine_files_sha256 = _file_binding(
        repo_root,
        economic_semantic_paths if separated_control_plane else legacy_combined_paths,
    )
    values = {
        "base_cost_policy_sha256": canonical_sha256(costs["BASE"]),
        "m1_precision_policy_sha256": canonical_sha256(
            {
                "contexts": ["M1_CORE5_2020_2026H1", "M1_FULL28_2025_2026H1"],
                "context_overlap_is_not_replication": True,
                "promotion_from_m5_is_automatic": False,
            }
        ),
        "replay_engine_sha256": replay_engine_files_sha256,
        "risk_policy_sha256": canonical_sha256(_risk_envelope(registry)),
        "scorer_sha256": _file_binding(
            repo_root, ("src/quant_rabbit/dojo_long_horizon_result_reducer.py",)
        ),
        "strategy_bundle_sha256": runtime_seal["runtime_binding_sha256"],
        "stress_cost_policy_sha256": canonical_sha256(costs["STRESS"]),
        "trainer_sha256": _file_binding(
            repo_root,
            (
                "src/quant_rabbit/dojo_ai_trainer_packet.py",
                "src/quant_rabbit/dojo_ai_tuning_state.py",
                "src/quant_rabbit/dojo_immutable_generation_loop.py",
            ),
        ),
    }
    if set(values) != set(IMPLEMENTATION_DIGEST_KEYS):
        raise DojoHistoricalTrainControlError("implementation digest grid drifted")
    return values


def _resource_policy(
    control: Mapping[str, Any],
    *,
    result_coordinate_count: int,
    max_job_coordinate_count: int,
) -> dict[str, int]:
    return {
        "max_resident_coordinates": min(
            max_job_coordinate_count, result_coordinate_count
        ),
        "max_rss_bytes": int(control["execution"].get("max_rss_bytes", 8 * 1024**3)),
        "max_open_files": int(control["execution"].get("max_open_files", 1024)),
        "min_free_disk_bytes": control["execution"]["minimum_free_disk_bytes"],
        "max_checkpoint_bytes": int(
            control["execution"].get("max_checkpoint_bytes", 16 * 1024 * 1024)
        ),
        "max_terminal_bytes": int(
            control["execution"].get("max_terminal_bytes", 16 * 1024 * 1024)
        ),
        "max_parallel_jobs": 1,
    }


def prepare_generation(
    *,
    repo_root: Path,
    run_control_path: Path,
) -> dict[str, Any]:
    """Seal the G2 denominator and initialize its crash-safe execution state."""

    repo = Path(repo_root).resolve(strict=True)
    control = _verified_control(run_control_path, repo_root=repo)
    fixed = control["fixed_inputs"]
    registry = _read_json(repo / fixed["registry_path"], field="G2 registry")
    registry_sha = _registry_artifact(registry)
    if registry_sha != fixed["registry_artifact_sha256"]:
        raise DojoHistoricalTrainControlError("run control names another G2 registry")
    source_manifest = verify_long_horizon_source_manifest_seal(
        _read_json(repo / fixed["source_manifest_path"], field="source manifest")
    )
    if source_manifest["source_manifest_sha256"] != fixed["source_manifest_sha256"]:
        raise DojoHistoricalTrainControlError(
            "run control names another source manifest"
        )
    proposals = _candidate_proposals(registry)
    room_generation = _is_room_generation(fixed)
    room_bindings = _configured_room_bindings(fixed) if room_generation else None
    if room_generation and sorted(row["family"] for row in proposals) != sorted(
        row["family_id"] for row in room_bindings
    ):
        raise DojoHistoricalTrainControlError(
            "G2 room taxonomy does not match the sealed strategy registry"
        )
    runtime_seal = build_tuned_strategy_runtime_seal(
        repo,
        candidate_proposals=proposals,
        generation_ordinal=_generation_ordinal(fixed),
        generation_binding_sha256=registry_sha,
    )
    runtime_seal = verify_tuned_strategy_runtime_seal(runtime_seal, repo_root=repo)
    source_inputs = long_horizon_plan_digest_inputs(source_manifest)
    families = sorted({row["family_id"] for row in runtime_seal["worker_catalog"]})
    implementation_digests = _implementation_digests(
        repo_root=repo,
        runtime_seal=runtime_seal,
        registry=registry,
        control=control,
    )
    archive_runtime_binding = _archive_runtime_binding(control)
    custody_control_plane_binding = (
        _custody_control_plane_binding(repo)
        if room_generation and _generation_ordinal(fixed) >= 3
        else None
    )
    plan = build_long_horizon_train_plan(
        portfolio_families=families,
        source_digests=source_inputs["source_digests"],
        corpus_digests=source_inputs["corpus_digests"],
        implementation_digests=implementation_digests,
        study_profile=fixed["study_profile"],
        room_bindings=room_bindings,
        room_train_months=(fixed.get("train_months") if room_generation else None),
    )
    worker_bindings = [
        {key: row[key] for key in ("worker_id", "family_id", "config_sha256")}
        for row in runtime_seal["worker_catalog"]
    ]
    schedule = build_long_horizon_stream_schedule(plan, worker_bindings=worker_bindings)
    output_root = Path(control["execution"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {key: output_root / name for key, name in _ARTIFACT_NAMES.items()}
    implementation_manifest_body = {
        "contract": "QR_DOJO_IMPLEMENTATION_DIGEST_MANIFEST_V1",
        "schema_version": 1,
        "archive_runtime_binding": archive_runtime_binding,
        "implementation_boundary": (
            "ECONOMIC_SEMANTIC_CORE_SEPARATE_FROM_CUSTODY_CONTROL_PLANE"
            if custody_control_plane_binding is not None
            else "LEGACY_COMBINED_IMPLEMENTATION_BOUNDARY"
        ),
        "custody_control_plane_binding": custody_control_plane_binding,
        "implementation_digests": implementation_digests,
        "implementation_digests_sha256": canonical_sha256(implementation_digests),
        **_AUTHORITY,
    }
    implementation_manifest = {
        **implementation_manifest_body,
        "implementation_manifest_sha256": canonical_sha256(
            implementation_manifest_body
        ),
    }
    sealed_input_dir = output_root / "sealed-inputs"
    sealed_input_values = {
        "IMPLEMENTATION_MANIFEST": implementation_manifest,
        "RUN_CONTROL": control,
        "SOURCE_MANIFEST": source_manifest,
        "STRATEGY_REGISTRY": registry,
    }
    sealed_input_names = {
        "IMPLEMENTATION_MANIFEST": "implementation-manifest.json",
        "RUN_CONTROL": "run-control.json",
        "SOURCE_MANIFEST": "source-manifest.json",
        "STRATEGY_REGISTRY": "strategy-registry.json",
    }
    for artifact_id in sorted(sealed_input_values):
        _write_once(
            sealed_input_dir / sealed_input_names[artifact_id],
            sealed_input_values[artifact_id],
        )
    sealed_input_artifacts = []
    for artifact_id in sorted(sealed_input_values):
        path = sealed_input_dir / sealed_input_names[artifact_id]
        raw = _stable_regular_bytes(path, field=f"sealed input {artifact_id}")
        sealed_input_artifacts.append(
            {
                "artifact_id": artifact_id,
                "relative_path": path.relative_to(output_root).as_posix(),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "file_size_bytes": len(raw),
            }
        )
    resource_policy = _resource_policy(
        control,
        result_coordinate_count=schedule["result_coordinate_count"],
        max_job_coordinate_count=max(
            job["coordinate_count"] for job in schedule["jobs"]
        ),
    )
    proposal_wrapper = {
        "contract": "QR_DOJO_G2_TRAINER_CANDIDATE_PROPOSALS_V1",
        "schema_version": 1,
        "generation": fixed["generation"],
        "registry_artifact_sha256": registry_sha,
        "trainer_candidate_proposals": proposals,
        **_AUTHORITY,
    }
    catalog_wrapper = {
        "contract": "QR_DOJO_LONG_HORIZON_WORKER_CATALOG_V1",
        "schema_version": 1,
        "worker_catalog": runtime_seal["worker_catalog"],
        **_AUTHORITY,
    }
    _write_once(paths["proposals"], proposal_wrapper)
    _write_once(paths["runtime_seal"], runtime_seal)
    _write_once(paths["worker_catalog"], catalog_wrapper)
    _write_once(paths["plan"], plan)
    _write_once(paths["schedule"], schedule)
    _write_once(paths["resource_policy"], resource_policy)
    runner_binding = {
        "runner_contract": "QR_DOJO_LONG_HORIZON_ECONOMIC_RUNNER_OUTPUT_REQUIREMENTS_V1",
        "runner_code_sha256": plan["implementation_binding"]["digests"][
            "replay_engine_sha256"
        ],
        "result_contract": CELL_CONTRACT,
    }
    initialize_long_horizon_execution_state(
        output_root / "execution-state",
        schedule=schedule,
        plan=plan,
        runner_binding=runner_binding,
        resource_policy=resource_policy,
    )
    room_study_profile_policy = (
        _validate_room_study_profile_policy(
            fixed.get("study_profile_policy"),
            train_months=fixed.get("train_months"),
            trainer_milestones=control["trainer_milestones"],
        )
        if room_generation and _generation_ordinal(fixed) >= 3
        else None
    )
    body = {
        "contract": MANIFEST_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_ONLY",
        "rapid_train_evaluation_mode": "INDEPENDENT_MONTH",
        "continuous_account_role": ("SEPARATE_LONG_PROFILE_NOT_RAPID_TUNING_GATE"),
        **(
            {
                "non_overlapping_review_blocks_required": True,
                "study_profile_policy": room_study_profile_policy,
            }
            if room_study_profile_policy is not None
            else {"non_overlapping_six_month_blocks_required": True}
        ),
        "generation": fixed["generation"],
        "registry_artifact_sha256": registry_sha,
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "run_control_sha256": hashlib.sha256(
            Path(run_control_path).read_bytes()
        ).hexdigest(),
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "runtime_binding_sha256": runtime_seal["runtime_binding_sha256"],
        "archive_runtime_binding": archive_runtime_binding,
        "custody_control_plane_binding": custody_control_plane_binding,
        "worker_count": len(runtime_seal["worker_catalog"]),
        "family_count": len(families),
        "room_count": len(room_bindings or ()),
        "room_family_bindings_sha256": (
            _room_binding_sha256(
                room_bindings,
                taxonomy_revision=fixed["room_taxonomy_revision"],
            )
            if room_generation
            else None
        ),
        "stream_job_count": schedule["stream_job_count"],
        "result_coordinate_count": schedule["result_coordinate_count"],
        "trainer_milestone_policy": dict(control["trainer_milestones"]),
        "sealed_input_artifacts": sealed_input_artifacts,
        "sealed_input_artifacts_sha256": canonical_sha256(sealed_input_artifacts),
        "artifact_sha256": {
            key: hashlib.sha256(paths[key].read_bytes()).hexdigest()
            for key in (
                "plan",
                "proposals",
                "resource_policy",
                "runtime_seal",
                "schedule",
                "worker_catalog",
            )
        },
        **_AUTHORITY,
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    _write_once(paths["control_manifest"], manifest)
    return {
        "status": "PREPARED",
        "output_root": str(output_root),
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "worker_count": len(runtime_seal["worker_catalog"]),
        "family_count": len(families),
        "stream_job_count": schedule["stream_job_count"],
        "result_coordinate_count": schedule["result_coordinate_count"],
        **_AUTHORITY,
    }


def _load_generation(
    *, repo_root: Path, run_control_path: Path, operation: str = "run"
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if operation not in {"run", "custody"}:
        raise DojoHistoricalTrainControlError("generation load operation is invalid")
    repo = Path(repo_root).resolve(strict=True)
    control = _verified_control(run_control_path, repo_root=repo)
    root = Path(control["execution"]["output_root"]).resolve(strict=True)
    manifest = _read_json(
        root / _ARTIFACT_NAMES["control_manifest"], field="control manifest"
    )
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("contract") != MANIFEST_CONTRACT or manifest.get(
        "manifest_sha256"
    ) != canonical_sha256(body):
        raise DojoHistoricalTrainControlError("generation control manifest drifted")
    if (
        manifest.get("run_control_sha256")
        != hashlib.sha256(Path(run_control_path).read_bytes()).hexdigest()
    ):
        raise DojoHistoricalTrainControlError(
            "generation run-control bytes differ from the prepared manifest"
        )
    sealed_input_rows = manifest.get("sealed_input_artifacts")
    if (
        not isinstance(sealed_input_rows, list)
        or manifest.get("sealed_input_artifacts_sha256")
        != canonical_sha256(sealed_input_rows)
        or [row.get("artifact_id") for row in sealed_input_rows]
        != [
            "IMPLEMENTATION_MANIFEST",
            "RUN_CONTROL",
            "SOURCE_MANIFEST",
            "STRATEGY_REGISTRY",
        ]
    ):
        raise DojoHistoricalTrainControlError(
            "generation sealed input inventory drifted"
        )
    sealed_inputs: dict[str, dict[str, Any]] = {}
    for row in sealed_input_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "artifact_id",
            "relative_path",
            "file_sha256",
            "file_size_bytes",
        }:
            raise DojoHistoricalTrainControlError(
                "generation sealed input row is invalid"
            )
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise DojoHistoricalTrainControlError(
                "generation sealed input path is invalid"
            )
        path = root / relative
        raw = _stable_regular_bytes(path, field=f"sealed input {row['artifact_id']}")
        if (
            hashlib.sha256(raw).hexdigest() != row["file_sha256"]
            or len(raw) != row["file_size_bytes"]
        ):
            raise DojoHistoricalTrainControlError(
                "generation sealed input bytes drifted"
            )
        sealed_inputs[row["artifact_id"]] = _read_json(
            path, field=f"sealed input {row['artifact_id']}"
        )
    if sealed_inputs["RUN_CONTROL"] != control:
        raise DojoHistoricalTrainControlError(
            "generation sealed run control differs from current control"
        )
    current_archive_runtime_binding = _archive_runtime_binding(control)
    if manifest.get("archive_runtime_binding") != current_archive_runtime_binding:
        raise DojoHistoricalTrainControlError(
            "generation archive runtime binding drifted"
        )
    implementation_manifest = sealed_inputs["IMPLEMENTATION_MANIFEST"]
    implementation_body = {
        key: value
        for key, value in implementation_manifest.items()
        if key != "implementation_manifest_sha256"
    }
    if implementation_manifest.get(
        "implementation_manifest_sha256"
    ) != canonical_sha256(implementation_body) or implementation_manifest.get(
        "implementation_digests_sha256"
    ) != canonical_sha256(implementation_manifest.get("implementation_digests")):
        raise DojoHistoricalTrainControlError(
            "generation implementation manifest drifted"
        )
    if (
        implementation_manifest.get("archive_runtime_binding")
        != current_archive_runtime_binding
    ):
        raise DojoHistoricalTrainControlError(
            "generation implementation archive runtime binding drifted"
        )
    separated_control_plane = (
        _is_room_generation(control["fixed_inputs"])
        and _generation_ordinal(control["fixed_inputs"]) >= 3
    )
    sealed_custody = implementation_manifest.get("custody_control_plane_binding")
    if manifest.get("custody_control_plane_binding") != sealed_custody:
        raise DojoHistoricalTrainControlError(
            "generation custody control-plane manifests disagree"
        )
    if separated_control_plane:
        current_custody = _custody_control_plane_binding(repo)
        if (
            implementation_manifest.get("implementation_boundary")
            != "ECONOMIC_SEMANTIC_CORE_SEPARATE_FROM_CUSTODY_CONTROL_PLANE"
            or not isinstance(sealed_custody, Mapping)
            or sealed_custody.get("compatibility_contract")
            != current_custody["compatibility_contract"]
            or (operation == "run" and sealed_custody != current_custody)
        ):
            raise DojoHistoricalTrainControlError(
                "generation custody control-plane compatibility failed"
            )
    elif sealed_custody is not None:
        raise DojoHistoricalTrainControlError(
            "legacy generation unexpectedly seals a separated custody plane"
        )
    artifacts = {}
    for key in (
        "plan",
        "proposals",
        "resource_policy",
        "runtime_seal",
        "schedule",
        "worker_catalog",
    ):
        path = root / _ARTIFACT_NAMES[key]
        if (
            hashlib.sha256(
                _stable_regular_bytes(path, field=f"generation {key}")
            ).hexdigest()
            != manifest["artifact_sha256"][key]
        ):
            raise DojoHistoricalTrainControlError(f"generation {key} bytes drifted")
        artifacts[key] = _read_json(path, field=key)
    plan = validate_long_horizon_train_plan(artifacts["plan"])
    schedule = validate_long_horizon_stream_schedule(artifacts["schedule"], plan=plan)
    runtime_seal = verify_tuned_strategy_runtime_seal(
        artifacts["runtime_seal"], repo_root=repo
    )
    registry = sealed_inputs["STRATEGY_REGISTRY"]
    if _registry_artifact(registry) != manifest.get("registry_artifact_sha256"):
        raise DojoHistoricalTrainControlError("generation registry binding drifted")
    current_digests = _implementation_digests(
        repo_root=repo,
        runtime_seal=runtime_seal,
        registry=registry,
        control=control,
    )
    if plan["implementation_binding"]["digests"] != current_digests:
        raise DojoHistoricalTrainControlError(
            "generation implementation bytes differ from the prepared plan"
        )
    if implementation_manifest["implementation_digests"] != current_digests:
        raise DojoHistoricalTrainControlError(
            "generation implementation manifest differs from current bytes"
        )
    source_manifest = verify_long_horizon_source_manifest_seal(
        sealed_inputs["SOURCE_MANIFEST"]
    )
    return (
        control,
        root,
        plan,
        schedule,
        runtime_seal,
        source_manifest,
        artifacts["worker_catalog"],
    )


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def _conversion_routes(pairs: Sequence[str]) -> list[dict[str, str]]:
    routes = []
    for currency in sorted(
        {pair.split("_")[1] for pair in pairs if pair.split("_")[1] != "JPY"}
    ):
        direct = f"{currency}_JPY"
        inverse = f"JPY_{currency}"
        if direct in pairs:
            route_pair, orientation = direct, "JPY_PER_CURRENCY"
        elif inverse in pairs:
            route_pair, orientation = inverse, "CURRENCY_PER_JPY"
        else:
            raise DojoHistoricalTrainControlError(
                f"feed has no JPY conversion route for {currency}"
            )
        routes.append(
            {"currency": currency, "pair": route_pair, "orientation": orientation}
        )
    return routes


def _coordinate_runtimes(
    *,
    handoff: Mapping[str, Any],
    plan: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    control: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    job = handoff["job"]
    digests = plan["implementation_binding"]["digests"]
    risk = _risk_envelope(registry)
    costs = _cost_profiles(control)
    result = {}
    for coordinate in job["coordinates"]:
        coordinate_id = coordinate["coordinate_id"]
        if coordinate_id not in handoff["runnable_coordinate_ids"]:
            continue
        active = [
            dict(row)
            for row, bit in zip(catalog, coordinate["active_worker_mask"], strict=True)
            if bit == "1"
        ]
        trade_pairs = [
            pair
            for pair, bit in zip(
                job["feed_pairs"], coordinate["trade_pair_mask"], strict=True
            )
            if bit == "1"
        ]
        scenario = coordinate["cost_scenario"]
        cost = costs[scenario]
        slippage_pips = cost["slippage_pips_per_adverse_fill"]
        policy = seal_portfolio_policy(
            {
                "policy_id": f"g2-{scenario.lower()}-{coordinate_id[:24]}",
                "expected_quote_pairs": list(job["feed_pairs"]),
                "tradable_pairs": trade_pairs,
                "active_worker_bindings": active,
                "leverage": risk["leverage"],
                "margin_closeout_fraction": risk["margin_closeout_fraction"],
                "max_margin_utilization_fraction": risk[
                    "max_margin_utilization_fraction"
                ],
                "max_portfolio_stop_risk_fraction": risk[
                    "max_portfolio_stop_risk_fraction"
                ],
                "max_open_and_pending_total": risk["max_open_and_pending_total"],
                "max_open_and_pending_per_pair": risk["max_open_and_pending_per_pair"],
                "max_open_and_pending_per_family": risk[
                    "max_open_and_pending_per_family"
                ],
                "max_currency_gross_notional_fraction": risk[
                    "max_currency_gross_notional_fraction"
                ],
                "max_cluster_gross_notional_fraction": risk[
                    "max_cluster_gross_notional_fraction"
                ],
                "max_lock_seconds": risk["max_lock_seconds"],
                "slippage_by_pair": [
                    {
                        "pair": pair,
                        "entry_slippage_price": slippage_pips * _pip_size(pair),
                        "exit_slippage_price": slippage_pips * _pip_size(pair),
                    }
                    for pair in job["feed_pairs"]
                ],
                "financing_by_pair": [
                    {
                        "pair": pair,
                        "long_cost_jpy_per_unit_day": (
                            cost["jpy_quote_financing_jpy_per_unit_day"]
                            if pair.endswith("_JPY")
                            else cost["non_jpy_quote_financing_jpy_per_unit_day"]
                        ),
                        "short_cost_jpy_per_unit_day": (
                            cost["jpy_quote_financing_jpy_per_unit_day"]
                            if pair.endswith("_JPY")
                            else cost["non_jpy_quote_financing_jpy_per_unit_day"]
                        ),
                    }
                    for pair in job["feed_pairs"]
                ],
                "conversion_routes": _conversion_routes(job["feed_pairs"]),
                "correlation_bindings": [],
            }
        )
        cost_sha = digests[
            "base_cost_policy_sha256"
            if scenario == "BASE"
            else "stress_cost_policy_sha256"
        ]
        binding_body = {
            "coordinate_id": coordinate_id,
            "cost_scenario": scenario,
            "portfolio_policy_sha256": policy["policy_sha256"],
            "cost_policy_sha256": cost_sha,
            "risk_policy_sha256": digests["risk_policy_sha256"],
            "replay_engine_sha256": digests["replay_engine_sha256"],
            "allocator_policy": coordinate["allocator_policy"],
            "initial_balance_jpy": 200_000,
        }
        result[coordinate_id] = {
            "coordinate_id": coordinate_id,
            "cost_scenario": scenario,
            "trade_pairs": trade_pairs,
            "portfolio_policy": policy,
            "cost_policy_sha256": cost_sha,
            "risk_policy_sha256": digests["risk_policy_sha256"],
            "replay_engine_sha256": digests["replay_engine_sha256"],
            "portfolio_policy_binding_sha256": canonical_sha256(binding_body),
        }
    return result


def _verified_job_result(
    value: Mapping[str, Any], *, handoff: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        key: item for key, item in value.items() if key != "economic_job_result_sha256"
    }
    if (
        value.get("contract") != ECONOMIC_JOB_RESULT_CONTRACT
        or value.get("economic_job_result_sha256") != canonical_sha256(body)
        or value.get("job_sha256") != handoff["job"]["job_sha256"]
        or value.get("claim_sha256") != handoff["claim"]["claim_sha256"]
        or value.get("coordinate_result_count")
        != len(handoff["runnable_coordinate_ids"])
    ):
        raise DojoHistoricalTrainControlError("economic job result is invalid")
    return dict(value)


def _carry_inputs(root: Path, handoff: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    slots = sorted(
        {
            row["predecessor_state_slot_id"]
            for row in handoff["job"]["coordinates"]
            if row["coordinate_id"] in handoff["runnable_coordinate_ids"]
            and row["predecessor_state_slot_id"] is not None
        }
    )
    result = {}
    for slot in slots:
        result[slot] = _read_json(
            root / "economic-carry-states" / f"{slot}.json",
            field=f"economic carry {slot}",
        )
    return result


def _publish_carries(root: Path, result: Mapping[str, Any]) -> None:
    carries = result.get("economic_carry_states_by_slot")
    if not isinstance(carries, Mapping):
        raise DojoHistoricalTrainControlError("economic result carry map is invalid")
    for slot, carry in carries.items():
        if not isinstance(slot, str) or not isinstance(carry, Mapping):
            raise DojoHistoricalTrainControlError("economic carry row is invalid")
        _write_once(root / "economic-carry-states" / f"{slot}.json", carry)


def _milestone_status(
    root: Path, schedule: Mapping[str, Any], *, publish: bool = True
) -> dict[str, Any]:
    manifest_path = root / _ARTIFACT_NAMES["control_manifest"]
    if not manifest_path.is_file():
        raise DojoHistoricalTrainControlError(
            "trainer milestone requires the sealed control manifest"
        )
    manifest = _read_json(manifest_path, field="control manifest")
    trainer_policy = manifest.get("trainer_milestone_policy")
    configured = (
        trainer_policy.get("m5_completed_months_per_review")
        if isinstance(trainer_policy, Mapping)
        else None
    )
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured < 1
    ):
        raise DojoHistoricalTrainControlError("trainer review cadence is invalid")
    review_months = configured
    raw_study_policy = manifest.get("study_profile_policy")
    review_blocks: list[list[str]] | None = None
    if raw_study_policy is not None:
        if not isinstance(raw_study_policy, Mapping):
            raise DojoHistoricalTrainControlError(
                "sealed room study profile policy is malformed"
            )
        raw_review_blocks = raw_study_policy.get("review_blocks")
        if not isinstance(raw_review_blocks, list):
            raise DojoHistoricalTrainControlError(
                "sealed room review block partition is malformed"
            )
        flattened = [
            month
            for block in raw_review_blocks
            if isinstance(block, Mapping)
            for month in block.get("train_months", [])
        ]
        validated_study_policy = _validate_room_study_profile_policy(
            raw_study_policy,
            train_months=flattened,
            trainer_milestones=trainer_policy,
        )
        review_blocks = [
            list(block["train_months"])
            for block in validated_study_policy["review_blocks"]
        ]
    jobs_by_sha = {job["job_sha256"]: job for job in schedule["jobs"]}
    completed_paths: dict[str, set[str]] = {}
    completed_jobs = 0
    result_root = root / "job-results"
    if result_root.is_dir():
        for path in sorted(result_root.glob("*.json")):
            value = _read_json(path, field="job result")
            job = jobs_by_sha.get(value.get("job_sha256"))
            if job is None or value.get("job_status") != "COMPLETE":
                continue
            completed_jobs += 1
            if job["source_binding_id"] == "M5_EXACT28_2020_2026H1":
                completed_paths.setdefault(job["month"], set()).add(
                    job["intrabar_path"]
                )
    completed_months = sorted(
        month for month, paths in completed_paths.items() if paths == {"OHLC", "OLHC"}
    )
    if review_blocks is None:
        block_count = len(completed_months) // review_months
        trainer_review_due = (
            len(completed_months) > 0 and len(completed_months) % review_months == 0
        )
    else:
        completed_month_set = set(completed_months)
        block_count = 0
        for block in review_blocks:
            if not set(block).issubset(completed_month_set):
                break
            block_count += 1
        completed_prefix = {
            month for block in review_blocks[:block_count] for month in block
        }
        trainer_review_due = block_count > 0 and completed_month_set == completed_prefix
    body = {
        "contract": MILESTONE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_ONLY",
        "rapid_train_evaluation_mode": "INDEPENDENT_MONTH",
        "continuous_account_role": ("SEPARATE_LONG_PROFILE_NOT_RAPID_TUNING_GATE"),
        **(
            {"non_overlapping_review_blocks_required": True}
            if review_blocks is not None
            else {"non_overlapping_six_month_blocks_required": True}
        ),
        "completed_job_count": completed_jobs,
        "completed_m5_month_count": len(completed_months),
        "completed_m5_months": completed_months,
        **(
            {"complete_review_block_count": block_count}
            if review_blocks is not None
            else {"complete_six_month_trainer_block_count": block_count}
        ),
        "trainer_review_month_count": review_months,
        "complete_trainer_block_count": block_count,
        "next_trainer_review_at_completed_m5_month_count": (block_count + 1)
        * review_months,
        "trainer_review_due": trainer_review_due,
        "partial_month_economics_may_tune": False,
        "parameter_change_requires_new_generation": True,
        "three_x_status": "3X_NOT_REACHABLE",
        **_AUTHORITY,
    }
    milestone = {**body, "milestone_sha256": canonical_sha256(body)}
    if publish:
        _write_once(
            root
            / "trainer-milestones"
            / f"jobs-{completed_jobs:04d}-{milestone['milestone_sha256']}.json",
            milestone,
        )
    return milestone


def _seal_source_failure(
    *,
    root: Path,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    handoff: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    """Count a deterministic source failure in the fixed denominator."""

    job = handoff["job"]
    claim = handoff["claim"]
    runnable_count = len(handoff["runnable_coordinate_ids"])
    coordinate_count = len(job["coordinates"])
    recorded_count = handoff["recorded_coordinate_count"]
    if (
        handoff["pending_coordinate_count"] != runnable_count
        or handoff["predecessor_blocked_coordinate_count"] != 0
        or recorded_count + runnable_count != coordinate_count
    ):
        raise DojoHistoricalTrainControlError(
            "source failure denominator is neither recorded-predecessor plus runnable"
        )
    message = str(error)
    if not message or len(message) > 4096:
        raise DojoHistoricalTrainControlError(
            "source failure message is empty or exceeds its evidence bound"
        )
    evidence_body = {
        "contract": "QR_DOJO_HISTORICAL_SOURCE_FAILURE_V1",
        "schema_version": SCHEMA_VERSION,
        "job_sha256": job["job_sha256"],
        "claim_sha256": claim["claim_sha256"],
        "stage": "SPARSE_SOURCE_MATERIALIZATION",
        "error_type": type(error).__name__,
        "error": message,
        **_AUTHORITY,
    }
    evidence = {
        **evidence_body,
        "failure_evidence_sha256": canonical_sha256(evidence_body),
    }
    job_dir = root / "jobs" / job["job_sha256"]
    _write_once(job_dir / "source-failure.json", evidence)
    cells = [
        build_long_horizon_coordinate_result(
            job=job,
            claim=claim,
            coordinate_id=coordinate_id,
            status="FAILED",
            failure={
                "code": "SOURCE_SLICE_MATERIALIZATION_FAILURE",
                "retryable": False,
                "evidence_sha256": evidence["failure_evidence_sha256"],
            },
        )
        for coordinate_id in handoff["runnable_coordinate_ids"]
    ]
    record_long_horizon_coordinate_results(
        root / "execution-state",
        schedule=schedule,
        plan=plan,
        claim_sha256=claim["claim_sha256"],
        results=cells,
    )
    terminal = seal_long_horizon_attempt(
        root / "execution-state",
        schedule=schedule,
        plan=plan,
        claim_sha256=claim["claim_sha256"],
    )
    terminal_cells = terminal["terminal_manifest"]["cells"]
    complete_coordinate_count = sum(
        row["status"] == "COMPLETE" for row in terminal_cells
    )
    failed_coordinate_count = sum(row["status"] == "FAILED" for row in terminal_cells)
    completion_body = {
        "contract": JOB_COMPLETION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "job_sha256": job["job_sha256"],
        "claim_sha256": claim["claim_sha256"],
        "source_binding_id": job["source_binding_id"],
        "month": job["month"],
        "intrabar_path": job["intrabar_path"],
        "coordinate_result_count": len(terminal_cells),
        "new_source_failure_coordinate_count": len(cells),
        "predecessor_failure_coordinate_count": recorded_count,
        "complete_coordinate_count": complete_coordinate_count,
        "failed_coordinate_count": failed_coordinate_count,
        "job_status": "FAILED_SOURCE",
        "economic_job_result_sha256": None,
        "terminal_sha256": terminal["terminal_manifest"]["terminal_sha256"],
        "free_disk_bytes_after": shutil.disk_usage(root).free,
        **_AUTHORITY,
    }
    completion = {
        **completion_body,
        "completion_sha256": canonical_sha256(completion_body),
    }
    _write_once(job_dir / "completion.json", completion)
    milestone = _milestone_status(root, schedule)
    return {
        "status": "JOB_FAILED_SOURCE",
        "output_root": str(root),
        "job": {
            "job_sha256": job["job_sha256"],
            "source_binding_id": job["source_binding_id"],
            "month": job["month"],
            "intrabar_path": job["intrabar_path"],
        },
        "coordinate_result_count": len(terminal_cells),
        "new_source_failure_coordinate_count": len(cells),
        "predecessor_failure_coordinate_count": recorded_count,
        "complete_coordinate_count": complete_coordinate_count,
        "failed_coordinate_count": failed_coordinate_count,
        "failure_evidence_sha256": evidence["failure_evidence_sha256"],
        "trainer_milestone": milestone,
        **_AUTHORITY,
    }


def _archive_root(control: Mapping[str, Any]) -> Path | None:
    value = control["execution"].get("archive_root")
    return Path(value) if isinstance(value, str) else None


def _archive_runtime_options(control: Mapping[str, Any]) -> dict[str, Any]:
    execution = control["execution"]
    binding = _archive_runtime_binding(control)
    options: dict[str, Any] = {}
    local_staging_root = (
        binding["local_staging_root"]
        if binding is not None
        else execution.get("archive_local_staging_root")
    )
    if isinstance(local_staging_root, str):
        options["local_staging_root"] = Path(local_staging_root)
    zstd_executable = (
        binding["zstd_executable"]
        if binding is not None
        else execution.get("archive_zstd_executable")
    )
    if isinstance(zstd_executable, str):
        options["zstd_bin"] = zstd_executable
    return options


def _archive_pending_completed_jobs(
    *,
    root: Path,
    control: Mapping[str, Any],
    mutation_guard: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    destination = _archive_root(control)
    if destination is None:
        return []
    receipts_root = destination / "receipts"
    archived = []
    for completion_path in sorted((root / "jobs").glob("*/completion.json")):
        job_sha = completion_path.parent.name
        restored = sorted(
            (root / "restore-v2-receipts").glob(f"restore-{job_sha}-*.json")
        ) or sorted((root / "restore-receipts").glob(f"restore-{job_sha}-*.json"))
        if restored:
            continue
        reclaimed = sorted(
            (root / "reclaim-v2-receipts").glob(f"reclaim-{job_sha}-*.json")
        ) or sorted((root / "reclaim-receipts").glob(f"reclaim-{job_sha}-*.json"))
        if reclaimed:
            continue
        existing = sorted(receipts_root.glob(f"job-{job_sha}-*.json"))
        if existing:
            continue
        runtime_options = _archive_runtime_options(control)
        try:
            recovery_capacity = inspect_historical_job_archive_recovery_capacity(
                run_root=root,
                job_sha256=job_sha,
                archive_root=destination,
                **runtime_options,
            )
        except DojoHistoricalJobArchiveError as exc:
            raise DojoHistoricalTrainControlError(
                f"archive recovery capacity inspection failed for {job_sha}: {exc}"
            ) from exc
        _assert_archive_staging_capacity(
            root=root,
            control=control,
            recovery_capacity=recovery_capacity,
        )
        if mutation_guard is not None:
            mutation_guard()
        try:
            receipt = archive_completed_historical_job(
                run_root=root,
                job_sha256=job_sha,
                archive_root=destination,
                expected_manifest_sha256=recovery_capacity["manifest_sha256"],
                **runtime_options,
            )
        except DojoHistoricalJobArchiveError as exc:
            raise DojoHistoricalTrainControlError(
                f"completed job archive failed for {job_sha}: {exc}"
            ) from exc
        archived.append(receipt)
        # One archive can consume substantial CPU and Drive bandwidth.  A
        # heartbeat recovers at most one orphan before yielding.
        break
    return archived


def _deep_verify_completed_job_custody(
    *, root: Path, control: Mapping[str, Any]
) -> None:
    destination = _archive_root(control)
    if destination is None:
        raise DojoHistoricalTrainControlError("archive root is unavailable")
    zstd_options = {
        key: value
        for key, value in _archive_runtime_options(control).items()
        if key == "zstd_bin"
    }
    for completion_path in sorted((root / "jobs").glob("*/completion.json")):
        job_sha = completion_path.parent.name
        local_paths = sorted((destination / "receipts").glob(f"job-{job_sha}-*.json"))
        signed_remote_paths = sorted(
            (destination / "remote-receipts").glob(f"signed-job-{job_sha}-*.json")
        )
        if len(local_paths) != 1:
            raise DojoHistoricalTrainControlError(
                f"completed job custody requires exact-one local receipt: {job_sha}"
            )
        if len(signed_remote_paths) > 1:
            raise DojoHistoricalTrainControlError(
                f"multiple signed remote attestations name one job: {job_sha}"
            )
        restored_v2 = sorted(
            (root / "restore-v2-receipts").glob(f"restore-{job_sha}-*.json")
        )
        reclaimed_v2 = sorted(
            (root / "reclaim-v2-receipts").glob(f"reclaim-{job_sha}-*.json")
        )
        restored = restored_v2 or (
            []
            if reclaimed_v2
            else sorted((root / "restore-receipts").glob(f"restore-{job_sha}-*.json"))
        )
        reclaimed = reclaimed_v2 or (
            []
            if restored_v2
            else sorted((root / "reclaim-receipts").glob(f"reclaim-{job_sha}-*.json"))
        )
        try:
            if restored:
                verify_existing_historical_job_raw_restore(
                    run_root=root,
                    archive_root=destination,
                    job_sha256=job_sha,
                    **zstd_options,
                )
            elif reclaimed:
                verify_existing_historical_job_raw_reclaim(
                    run_root=root,
                    archive_root=destination,
                    job_sha256=job_sha,
                    expected_drive_parent_id=control["execution"][
                        "archive_drive_readback_parent_id"
                    ],
                    **zstd_options,
                )
            elif signed_remote_paths:
                local = _compact_archive_receipt(
                    receipt_path=local_paths[0],
                    archive_root=destination,
                    expected_job_sha256=job_sha,
                )
                _, authority_path = _compact_signed_remote_receipt(
                    path=signed_remote_paths[0],
                    local_receipt=local,
                    archive_root=destination,
                    expected_drive_parent_id=control["execution"][
                        "archive_drive_readback_parent_id"
                    ],
                )
                verify_historical_job_raw_reclaim(
                    run_root=root,
                    archive_receipt_path=local_paths[0],
                    remote_receipt_path=signed_remote_paths[0],
                    expected_drive_parent_id=control["execution"][
                        "archive_drive_readback_parent_id"
                    ],
                    attestation_authority_seal_path=authority_path,
                    **zstd_options,
                )
            else:
                verify_existing_historical_job_archive(
                    run_root=root,
                    job_sha256=job_sha,
                    archive_root=destination,
                    **zstd_options,
                )
        except (DojoHistoricalJobArchiveError, DojoHistoricalRawReclaimError) as exc:
            raise DojoHistoricalTrainControlError(
                f"completed job custody failed for {job_sha}: {exc}"
            ) from exc


def _baseline_raw_bytes(
    *, control: Mapping[str, Any], planned_coordinate_count: int
) -> int:
    if planned_coordinate_count < 1:
        raise DojoHistoricalTrainControlError(
            "planned coordinate count for capacity estimate is invalid"
        )
    baseline = control["execution"].get("capacity_baseline")
    if not isinstance(baseline, Mapping):
        return 0
    coordinate_count = baseline.get("coordinate_count")
    raw_bytes = baseline.get("raw_bytes_per_job")
    source_bytes = baseline.get("source_bytes_per_job")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (coordinate_count, raw_bytes, source_bytes)
    ) or int(source_bytes) >= int(raw_bytes):
        raise DojoHistoricalTrainControlError(
            "capacity baseline cannot be normalized by coordinate count"
        )
    evidence_bytes = int(raw_bytes) - int(source_bytes)
    normalized_evidence = (
        evidence_bytes * planned_coordinate_count + int(coordinate_count) - 1
    ) // int(coordinate_count)
    return int(source_bytes) + normalized_evidence


def _estimated_next_job_bytes(
    *,
    control: Mapping[str, Any],
    archive_root: Path | None,
    planned_coordinate_count: int,
) -> tuple[int, int, int]:
    execution = control["execution"]
    bootstrap = execution.get("bootstrap_job_working_set_bytes")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 1:
        if _is_room_generation(control["fixed_inputs"]):
            raise DojoHistoricalTrainControlError(
                "room generation requires a sealed bootstrap working-set estimate"
            )
        bootstrap = 6 * 1024**3
    observed = []
    if archive_root is not None and (archive_root / "receipts").is_dir():
        for path in sorted((archive_root / "receipts").glob("job-*.json")):
            try:
                receipt = _read_json(path, field="historical archive receipt")
            except DojoHistoricalTrainControlError:
                continue
            body = {
                key: value for key, value in receipt.items() if key != "receipt_sha256"
            }
            archive_path = Path(receipt.get("archive_path", ""))
            if (
                receipt.get("contract") != ARCHIVE_RECEIPT_CONTRACT
                or receipt.get("receipt_sha256") != canonical_sha256(body)
                or not archive_path.is_file()
                or archive_root not in archive_path.parents
                or receipt.get("local_payload_verified") is not True
            ):
                continue
            value = receipt.get("total_source_bytes")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                observed.append(value)
    measured = max(observed, default=0)
    normalized_baseline = _baseline_raw_bytes(
        control=control,
        planned_coordinate_count=planned_coordinate_count,
    )
    headroom = execution.get("job_space_headroom_fraction", 0.15)
    if (
        isinstance(headroom, bool)
        or not isinstance(headroom, (int, float))
        or not 0 <= float(headroom) <= 1
    ):
        raise DojoHistoricalTrainControlError("job-space headroom fraction is invalid")
    raw_reserve = math.ceil(
        max(bootstrap, measured, normalized_baseline) * (1.0 + float(headroom))
    )
    prospective_tar_pax_overhead = max(
        64 * 1024**2,
        math.ceil(raw_reserve * 0.02),
        planned_coordinate_count * 256 * 1024,
    )
    prospective_tar = raw_reserve + prospective_tar_pax_overhead
    prospective_zstd_framing = max(
        128 * 1024,
        math.ceil(prospective_tar * 0.01),
    )
    archive_upper = prospective_tar + prospective_zstd_framing
    archive_destination = archive_upper * 2 + min(ARCHIVE_PART_BYTES, archive_upper)
    aggregate_peak = raw_reserve + archive_upper + archive_destination
    return raw_reserve, archive_upper, aggregate_peak


def _effective_archive_staging_fraction(
    *, control: Mapping[str, Any], archive_root: Path | None
) -> float:
    execution = control["execution"]
    staging = execution.get("archive_staging_fraction", 0.25)
    if (
        isinstance(staging, bool)
        or not isinstance(staging, (int, float))
        or not 0.20 <= float(staging) <= 1.0
    ):
        raise DojoHistoricalTrainControlError("archive staging fraction is invalid")
    observed: list[float] = []
    if archive_root is not None and (archive_root / "receipts").is_dir():
        for path in sorted((archive_root / "receipts").glob("job-*.json")):
            receipt = _read_json(path, field="historical archive receipt")
            body = {
                key: value for key, value in receipt.items() if key != "receipt_sha256"
            }
            readback = receipt.get("remote_readback_objects")
            source_bytes = receipt.get("total_source_bytes")
            archive_bytes = receipt.get("archive_size_bytes")
            readback_bytes = (
                readback.get("total_size_bytes")
                if isinstance(readback, Mapping)
                else None
            )
            if (
                receipt.get("contract") != ARCHIVE_RECEIPT_CONTRACT
                or receipt.get("receipt_sha256") != canonical_sha256(body)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in (source_bytes, archive_bytes, readback_bytes)
                )
            ):
                raise DojoHistoricalTrainControlError(
                    "archive receipt cannot calibrate staging capacity"
                )
            observed.append(
                (int(archive_bytes) + int(readback_bytes)) / int(source_bytes)
            )
    return max(float(staging), max(observed, default=0.0))


def _assert_archive_staging_capacity(
    *,
    root: Path,
    control: Mapping[str, Any],
    recovery_capacity: Mapping[str, Any],
) -> dict[str, Any]:
    capacity_body = {
        key: value
        for key, value in recovery_capacity.items()
        if key != "capacity_inspection_sha256"
    }
    required_values = {
        "total_source_bytes": recovery_capacity.get("total_source_bytes"),
        "archive_upper_bound_bytes": recovery_capacity.get("archive_upper_bound_bytes"),
        "manifest_payload_bytes": recovery_capacity.get("manifest_payload_bytes"),
        "tar_pax_upper_bound_bytes": recovery_capacity.get("tar_pax_upper_bound_bytes"),
        "zstd_framing_upper_bound_bytes": recovery_capacity.get(
            "zstd_framing_upper_bound_bytes"
        ),
        "remaining_local_staging_bytes": recovery_capacity.get(
            "remaining_local_staging_bytes"
        ),
        "remaining_archive_filesystem_bytes": recovery_capacity.get(
            "remaining_archive_filesystem_bytes"
        ),
    }
    if (
        recovery_capacity.get("contract") != ARCHIVE_RECOVERY_CAPACITY_CONTRACT
        or recovery_capacity.get("schema_version") != SCHEMA_VERSION
        or not isinstance(recovery_capacity.get("manifest_sha256"), str)
        or len(recovery_capacity["manifest_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in recovery_capacity["manifest_sha256"]
        )
        or recovery_capacity.get("compression_ratio_assumed") is not False
        or recovery_capacity.get("capacity_inspection_sha256")
        != canonical_sha256(capacity_body)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in required_values.values()
        )
        or required_values["total_source_bytes"] < 1
        or required_values["archive_upper_bound_bytes"]
        < required_values["total_source_bytes"]
    ):
        raise DojoHistoricalTrainControlError(
            "archive recovery capacity inspection is invalid"
        )
    source_bytes = int(required_values["total_source_bytes"])
    remaining_local = int(required_values["remaining_local_staging_bytes"])
    remaining_archive = int(required_values["remaining_archive_filesystem_bytes"])
    archive_root = _archive_root(control)
    if archive_root is None:
        raise DojoHistoricalTrainControlError("archive root is unavailable")
    recovery_floor = max(
        control["execution"]["minimum_free_disk_bytes"],
        control["execution"]["bootstrap_job_working_set_bytes"],
    )
    local_staging_value = control["execution"].get("archive_local_staging_root")
    local_staging_root = (
        Path(local_staging_value)
        if isinstance(local_staging_value, str)
        else Path(tempfile.gettempdir())
    )
    # The archive inspector deep-validates reusable crash bytes.  Reserve only
    # the still-missing local/archive deltas; free space already accounts for
    # invalid or unrelated bytes that cannot be reused.
    reservations = _filesystem_capacity_reservations(
        (
            ("run", root, 0, recovery_floor),
            ("local_staging", local_staging_root, remaining_local, recovery_floor),
            ("archive", archive_root, remaining_archive, recovery_floor),
        )
    )
    _assert_filesystem_reservations(reservations)
    role_rows = reservations["roles"]
    local_row = role_rows["local_staging"]
    archive_row = role_rows["archive"]
    run_row = role_rows["run"]
    return {
        "shared_filesystem": local_row["device"] == archive_row["device"],
        "source_bytes": source_bytes,
        "compression_ratio_assumed": False,
        "archive_upper_bound_bytes": required_values["archive_upper_bound_bytes"],
        "manifest_payload_bytes": required_values["manifest_payload_bytes"],
        "tar_pax_upper_bound_bytes": required_values["tar_pax_upper_bound_bytes"],
        "zstd_framing_upper_bound_bytes": required_values[
            "zstd_framing_upper_bound_bytes"
        ],
        "validated_existing_bytes": {
            key: value
            for key, value in recovery_capacity.items()
            if key.startswith("validated_") and key.endswith("_bytes")
        },
        "remaining_local_staging_bytes": remaining_local,
        "remaining_archive_filesystem_bytes": remaining_archive,
        "staging_bytes": remaining_local + remaining_archive,
        "local_staging_required_bytes": remaining_local + recovery_floor,
        "archive_required_bytes": remaining_archive + recovery_floor,
        "recovery_floor_bytes": recovery_floor,
        "run_free_bytes": run_row["free_bytes"],
        "local_staging_free_bytes": local_row["free_bytes"],
        "archive_free_bytes": archive_row["free_bytes"],
        "filesystem_reservations": reservations["devices"],
    }


def _compact_archive_receipt(
    *,
    receipt_path: Path,
    archive_root: Path,
    expected_job_sha256: str,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path, field="historical archive receipt")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    archive_path = Path(receipt.get("archive_path", ""))
    readback = receipt.get("remote_readback_objects")
    if (
        receipt.get("contract") != ARCHIVE_RECEIPT_CONTRACT
        or receipt.get("job_sha256") != expected_job_sha256
        or receipt.get("receipt_sha256") != canonical_sha256(body)
        or receipt.get("local_payload_verified") is not True
        or not archive_path.is_absolute()
        or archive_root not in archive_path.parents
        or not isinstance(readback, Mapping)
    ):
        raise DojoHistoricalTrainControlError(
            "historical archive compact receipt is invalid"
        )
    try:
        state = archive_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalTrainControlError(
            "historical archive bytes are unavailable"
        ) from exc
    if (
        archive_path.is_symlink()
        or not stat.S_ISREG(state.st_mode)
        or state.st_size != receipt.get("archive_size_bytes")
    ):
        raise DojoHistoricalTrainControlError("historical archive metadata drifted")
    return receipt


def _archive_local_footprint_bytes(
    receipt: Mapping[str, Any], *, archive_root: Path
) -> int:
    archive_path = Path(receipt["archive_path"])
    relative = archive_path.relative_to(archive_root).as_posix()
    sizes = {relative: int(receipt["archive_size_bytes"])}
    readback = receipt.get("remote_readback_objects")
    objects = readback.get("objects") if isinstance(readback, Mapping) else None
    if not isinstance(objects, list):
        raise DojoHistoricalTrainControlError(
            "archive readback object inventory is unavailable"
        )
    for row in objects:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("relative_path"), str)
            or isinstance(row.get("size_bytes"), bool)
            or not isinstance(row.get("size_bytes"), int)
            or row["size_bytes"] < 1
        ):
            raise DojoHistoricalTrainControlError(
                "archive readback object inventory is invalid"
            )
        sizes[row["relative_path"]] = int(row["size_bytes"])
    return sum(sizes.values())


def _compact_remote_receipt(
    *, path: Path, local_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _read_json(path, field="remote archive readback receipt")
    body = {
        key: value for key, value in receipt.items() if key != "remote_receipt_sha256"
    }
    digest = receipt.get("remote_receipt_sha256")
    expected_name = (
        f"remote-job-{local_receipt['job_sha256']}-"
        f"{local_receipt['manifest_sha256']}-{digest}.json"
    )
    if (
        receipt.get("contract") != REMOTE_READBACK_RECEIPT_CONTRACT
        or receipt.get("status") != "REMOTE_VERIFIED"
        or receipt.get("job_sha256") != local_receipt["job_sha256"]
        or receipt.get("manifest_sha256") != local_receipt["manifest_sha256"]
        or receipt.get("local_archive_receipt_sha256")
        != local_receipt["receipt_sha256"]
        or receipt.get("remote_verified") is not True
        or receipt.get("raw_reclaim_eligible") is not True
        or receipt.get("promotion_eligible") is not False
        or receipt.get("live_permission") is not False
        or receipt.get("order_authority") != "NONE"
        or receipt.get("broker_mutation_allowed") is not False
        or not isinstance(digest, str)
        or digest != canonical_sha256(body)
        or path.name != expected_name
    ):
        raise DojoHistoricalTrainControlError(
            "remote readback compact receipt is invalid"
        )
    return receipt


def _lower_hex_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _attestation_authority_seals_for_job(
    *, archive_root: Path, job_sha256: str
) -> list[tuple[Path, dict[str, Any]]]:
    authority_root = archive_root / "remote-authorities"
    if not authority_root.exists():
        return []
    try:
        state = authority_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalTrainControlError(
            "attestation authority namespace is unavailable"
        ) from exc
    if authority_root.is_symlink() or not stat.S_ISDIR(state.st_mode):
        raise DojoHistoricalTrainControlError(
            "attestation authority namespace is unsafe"
        )
    prefix = f"key-job-{job_sha256}-"
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(authority_root.iterdir()):
        name_claims_job = path.name.startswith(prefix)
        if path.suffix != ".json":
            if name_claims_job:
                raise DojoHistoricalTrainControlError(
                    "attestation authority namespace has a malformed job seal"
                )
            continue
        if path.is_symlink() or not path.is_file():
            raise DojoHistoricalTrainControlError(
                "attestation authority namespace contains an unsafe JSON entry"
            )
        try:
            value = _read_json(path, field="attestation public-key seal")
        except DojoHistoricalTrainControlError:
            if name_claims_job:
                raise
            continue
        if name_claims_job or value.get("job_sha256") == job_sha256:
            matches.append((path, value))
    return matches


def _compact_attestation_authority_seal(
    *,
    archive_root: Path,
    local_receipt: Mapping[str, Any],
    expected_drive_parent_id: str,
) -> tuple[Path, dict[str, Any]]:
    job_sha256 = str(local_receipt["job_sha256"])
    matches = _attestation_authority_seals_for_job(
        archive_root=archive_root,
        job_sha256=job_sha256,
    )
    if len(matches) != 1:
        raise DojoHistoricalTrainControlError(
            "signed attestation requires exact-one public-key authority seal"
        )
    path, seal = matches[0]
    body = {key: value for key, value in seal.items() if key != "authority_seal_sha256"}
    public_key = seal.get("public_key_hex")
    public_key_sha = (
        hashlib.sha256(bytes.fromhex(public_key)).hexdigest()
        if isinstance(public_key, str)
        and len(public_key) == 64
        and all(character in "0123456789abcdef" for character in public_key)
        else None
    )
    digest = seal.get("authority_seal_sha256")
    expected_name = (
        f"key-job-{job_sha256}-{local_receipt['manifest_sha256']}-{digest}.json"
    )
    if (
        set(seal) != _ATTESTATION_PUBLIC_KEY_SEAL_KEYS
        or seal.get("contract") != ATTESTATION_PUBLIC_KEY_SEAL_CONTRACT
        or seal.get("schema_version") != 2
        or isinstance(seal.get("schema_version"), bool)
        or seal.get("status") != "OPERATOR_PUBLIC_KEY_ENROLLED_BEFORE_READBACK"
        or seal.get("job_sha256") != job_sha256
        or seal.get("manifest_sha256") != local_receipt["manifest_sha256"]
        or seal.get("local_archive_receipt_sha256") != local_receipt["receipt_sha256"]
        or seal.get("expected_drive_parent_id") != expected_drive_parent_id
        or seal.get("algorithm") != "ED25519"
        or public_key_sha is None
        or seal.get("public_key_sha256") != public_key_sha
        or seal.get("private_key_material_accepted") is not False
        or seal.get("historical_train_is_proof") is not False
        or seal.get("promotion_eligible") is not False
        or seal.get("live_permission") is not False
        or seal.get("order_authority") != "NONE"
        or seal.get("broker_mutation_allowed") is not False
        or not isinstance(seal.get("enrolled_at_utc"), str)
        or not _lower_hex_sha(digest)
        or digest != canonical_sha256(body)
        or path.name != expected_name
    ):
        raise DojoHistoricalTrainControlError(
            "signed attestation public-key authority seal is invalid"
        )
    return path, seal


def _compact_signed_remote_receipt(
    *,
    path: Path,
    local_receipt: Mapping[str, Any],
    archive_root: Path,
    expected_drive_parent_id: str,
) -> tuple[dict[str, Any], Path]:
    receipt = _read_json(path, field="signed remote archive readback receipt")
    authority_path, authority = _compact_attestation_authority_seal(
        archive_root=archive_root,
        local_receipt=local_receipt,
        expected_drive_parent_id=expected_drive_parent_id,
    )
    body = receipt.get("body")
    envelope_body = {
        key: value for key, value in receipt.items() if key != "remote_receipt_sha256"
    }
    digest = receipt.get("remote_receipt_sha256")
    expected_name = (
        f"signed-job-{local_receipt['job_sha256']}-"
        f"{local_receipt['manifest_sha256']}-{digest}.json"
    )
    if (
        set(receipt) != _SIGNED_ATTESTATION_KEYS
        or receipt.get("contract") != REMOTE_READBACK_SIGNED_ATTESTATION_CONTRACT
        or receipt.get("schema_version") != 2
        or isinstance(receipt.get("schema_version"), bool)
        or receipt.get("algorithm") != "ED25519"
        or receipt.get("public_key_sha256") != authority["public_key_sha256"]
        or not isinstance(receipt.get("signature_base64"), str)
        or not 40 <= len(receipt["signature_base64"]) <= 256
        or not _lower_hex_sha(digest)
        or digest != canonical_sha256(envelope_body)
        or path.name != expected_name
        or not isinstance(body, Mapping)
        or set(body) != _SIGNED_ATTESTATION_BODY_KEYS
        or body.get("contract") != REMOTE_READBACK_ATTESTATION_BODY_CONTRACT
        or body.get("schema_version") != 2
        or isinstance(body.get("schema_version"), bool)
        or body.get("job_sha256") != local_receipt["job_sha256"]
        or body.get("manifest_sha256") != local_receipt["manifest_sha256"]
        or body.get("local_archive_receipt_sha256") != local_receipt["receipt_sha256"]
        or body.get("expected_drive_parent_id") != expected_drive_parent_id
        or body.get("provider") != "GOOGLE_DRIVE"
        or body.get("remote_verified") is not True
        or body.get("raw_reclaim_eligible") is not True
        or body.get("external_readback_attested") is not True
        or body.get("historical_train_is_proof") is not False
        or body.get("promotion_eligible") is not False
        or body.get("live_permission") is not False
        or body.get("order_authority") != "NONE"
        or body.get("broker_mutation_allowed") is not False
    ):
        raise DojoHistoricalTrainControlError(
            "signed remote readback compact receipt is invalid"
        )
    return receipt, authority_path


def _compact_reclaim_receipt(*, path: Path, job_sha256: str) -> dict[str, Any]:
    receipt = _read_json(path, field="historical raw reclaim receipt")
    body = {
        key: value for key, value in receipt.items() if key != "reclaim_receipt_sha256"
    }
    digest = receipt.get("reclaim_receipt_sha256")
    if (
        receipt.get("contract") != RECLAIM_RECEIPT_CONTRACT
        or receipt.get("status") != "RAW_RECLAIMED"
        or receipt.get("job_sha256") != job_sha256
        or receipt.get("promotion_eligible") is not False
        or receipt.get("live_permission") is not False
        or receipt.get("order_authority") != "NONE"
        or receipt.get("broker_mutation_allowed") is not False
        or isinstance(receipt.get("reclaimed_logical_bytes"), bool)
        or not isinstance(receipt.get("reclaimed_logical_bytes"), int)
        or receipt["reclaimed_logical_bytes"] < 1
        or not isinstance(digest, str)
        or digest != canonical_sha256(body)
        or path.name != f"reclaim-{job_sha256}-{digest}.json"
    ):
        raise DojoHistoricalTrainControlError(
            "historical raw reclaim compact receipt is invalid"
        )
    return receipt


def _compact_restore_receipt(*, path: Path, job_sha256: str) -> dict[str, Any]:
    receipt = _read_json(path, field="historical raw restore receipt")
    body = {
        key: value for key, value in receipt.items() if key != "restore_receipt_sha256"
    }
    digest = receipt.get("restore_receipt_sha256")
    if (
        receipt.get("contract") != RESTORE_RECEIPT_CONTRACT
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "RAW_RESTORED"
        or receipt.get("job_sha256") != job_sha256
        or isinstance(receipt.get("restored_file_count"), bool)
        or not isinstance(receipt.get("restored_file_count"), int)
        or receipt["restored_file_count"] < 1
        or not isinstance(receipt.get("restored_files"), list)
        or len(receipt["restored_files"]) != receipt["restored_file_count"]
        or isinstance(receipt.get("restored_logical_bytes"), bool)
        or not isinstance(receipt.get("restored_logical_bytes"), int)
        or receipt["restored_logical_bytes"] < 1
        or receipt.get("local_archive_deep_verified") is not True
        or receipt.get("retained_bytes_verified") is not True
        or receipt.get("all_raw_targets_present") is not True
        or receipt.get("remote_receipt_trusted") is not False
        or receipt.get("historical_train_is_proof") is not False
        or receipt.get("promotion_eligible") is not False
        or receipt.get("live_permission") is not False
        or receipt.get("order_authority") != "NONE"
        or receipt.get("broker_mutation_allowed") is not False
        or not isinstance(digest, str)
        or digest != canonical_sha256(body)
        or path.name != f"restore-{job_sha256}-{digest}.json"
    ):
        raise DojoHistoricalTrainControlError(
            "historical raw restore compact receipt is invalid"
        )
    return receipt


def _job_artifact_files(
    root: Path, relative: str, pattern: str
) -> dict[str, list[Path]]:
    base = root / relative
    rows: dict[str, list[Path]] = {}
    if not base.exists():
        return rows
    try:
        state = base.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalTrainControlError(
            f"lifecycle artifact root is unavailable: {relative}"
        ) from exc
    if not stat.S_ISDIR(state.st_mode):
        raise DojoHistoricalTrainControlError(
            f"lifecycle artifact root is not a directory: {relative}"
        )
    if pattern.startswith("*/"):
        child_pattern = pattern[2:]
        paths = []
        for directory in sorted(base.iterdir()):
            try:
                directory_state = directory.stat(follow_symlinks=False)
            except OSError as exc:
                raise DojoHistoricalTrainControlError(
                    f"lifecycle artifact directory is unavailable: {directory}"
                ) from exc
            if directory.is_symlink() or not stat.S_ISDIR(directory_state.st_mode):
                raise DojoHistoricalTrainControlError(
                    f"lifecycle artifact directory is unsafe: {directory}"
                )
            paths.extend(sorted(directory.glob(child_pattern)))
    else:
        paths = sorted(base.glob(pattern))
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise DojoHistoricalTrainControlError(
                f"lifecycle artifact is unsafe: {path}"
            )
        if path.parent == base:
            name = path.name
            parts = name.split("-")
            job_sha = (
                parts[2]
                if (name.startswith("remote-job-") or name.startswith("signed-job-"))
                and len(parts) > 3
                else parts[1]
                if len(parts) > 2
                else ""
            )
        else:
            job_sha = path.parent.name
        rows.setdefault(job_sha, []).append(path)
    return rows


def _compact_completion(
    *, path: Path, job_sha256: str, final_terminal_sha256: str
) -> dict[str, Any]:
    completion = _read_json(path, field="historical job completion")
    body = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    if (
        completion.get("contract") != JOB_COMPLETION_CONTRACT
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("job_sha256") != job_sha256
        or completion.get("terminal_sha256") != final_terminal_sha256
        or completion.get("completion_sha256") != canonical_sha256(body)
        or path != path.parent / "completion.json"
        or path.parent.name != job_sha256
        or any(completion.get(key) != value for key, value in _AUTHORITY.items())
    ):
        raise DojoHistoricalTrainControlError(
            "historical job completion does not bind the final terminal"
        )
    return completion


def evaluate_historical_lifecycle(
    *,
    root: Path,
    control: Mapping[str, Any],
    plan: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one fail-closed transition from the exact lifecycle artifacts."""

    execution = long_horizon_execution_status(
        root / "execution-state", schedule=schedule, plan=plan
    )
    manifest_path = root / _ARTIFACT_NAMES["control_manifest"]
    milestone = (
        _milestone_status(root, schedule, publish=False)
        if manifest_path.is_file()
        else None
    )
    trainer_review_due = bool(milestone is not None and milestone["trainer_review_due"])
    scheduled = [job["job_sha256"] for job in schedule["jobs"]]
    scheduled_set = set(scheduled)
    terminal_history = _job_artifact_files(
        root / "execution-state", "terminals", "*/*.json"
    )
    claim_history = _job_artifact_files(root / "execution-state", "claims", "*/*.json")
    latest_terminals: dict[str, Path] = {}
    active_ids: set[str] = set()
    history_blockers: list[str] = []
    for job_sha in sorted(set(claim_history) | set(terminal_history)):
        claims_by_attempt: dict[int, Path] = {}
        terminals_by_attempt: dict[int, Path] = {}
        for path in claim_history.get(job_sha, []):
            try:
                attempt = int(path.stem.split("-", 1)[1])
            except (IndexError, ValueError):
                history_blockers.append(f"INVALID_CLAIM_FILENAME:{job_sha}")
                continue
            if attempt in claims_by_attempt:
                history_blockers.append(f"DUPLICATE_CLAIM_ATTEMPT:{job_sha}")
            claims_by_attempt[attempt] = path
        for path in terminal_history.get(job_sha, []):
            try:
                attempt = int(path.stem.split("-", 2)[1])
            except (IndexError, ValueError):
                history_blockers.append(f"INVALID_TERMINAL_FILENAME:{job_sha}")
                continue
            if attempt in terminals_by_attempt:
                history_blockers.append(f"DUPLICATE_TERMINAL_ATTEMPT:{job_sha}")
            terminals_by_attempt[attempt] = path
        if set(terminals_by_attempt) - set(claims_by_attempt):
            history_blockers.append(f"TERMINAL_WITHOUT_CLAIM:{job_sha}")
        if claims_by_attempt:
            latest_attempt = max(claims_by_attempt)
            if latest_attempt in terminals_by_attempt:
                latest_terminals[job_sha] = terminals_by_attempt[latest_attempt]
            else:
                active_ids.add(job_sha)
        elif terminals_by_attempt:
            history_blockers.append(f"TERMINAL_WITHOUT_CLAIM:{job_sha}")
    terminal_ids = set(latest_terminals)
    completions = _job_artifact_files(root, "jobs", "*/completion.json")
    completion_ids = set(completions)
    unknown = (
        set(claim_history) | set(terminal_history) | completion_ids
    ) - scheduled_set
    blockers: list[str] = list(history_blockers)
    if unknown:
        blockers.append("UNKNOWN_LIFECYCLE_JOB:" + ",".join(sorted(unknown)))
    if len(terminal_ids) != execution["terminal_job_count"]:
        blockers.append("TERMINAL_ARTIFACT_COUNT_MISMATCH")
    missing_completion = terminal_ids - completion_ids
    completion_without_terminal = completion_ids - terminal_ids
    if missing_completion:
        blockers.append(
            "INCOMPLETE_TERMINAL_PUBLICATION:" + ",".join(sorted(missing_completion))
        )
    if completion_without_terminal:
        blockers.append(
            "COMPLETION_WITHOUT_TERMINAL:"
            + ",".join(sorted(completion_without_terminal))
        )
    for job_sha in sorted(terminal_ids & completion_ids):
        if len(completions[job_sha]) != 1:
            blockers.append(f"DUPLICATE_COMPLETION:{job_sha}")
            continue
        last_terminal = latest_terminals[job_sha]
        final_terminal_sha256 = last_terminal.stem.rsplit("-", 1)[-1]
        if len(final_terminal_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in final_terminal_sha256
        ):
            blockers.append(f"INVALID_TERMINAL_FILENAME:{job_sha}")
            continue
        _compact_completion(
            path=completions[job_sha][0],
            job_sha256=job_sha,
            final_terminal_sha256=final_terminal_sha256,
        )

    archive_root = _archive_root(control)
    local_receipts: dict[str, list[Path]] = {}
    unsigned_remote_receipts: dict[str, list[Path]] = {}
    signed_remote_receipts: dict[str, list[Path]] = {}
    if archive_root is not None:
        local_receipts = _job_artifact_files(archive_root, "receipts", "job-*.json")
        unsigned_remote_receipts = _job_artifact_files(
            archive_root, "remote-receipts", "remote-job-*.json"
        )
        signed_remote_receipts = _job_artifact_files(
            archive_root, "remote-receipts", "signed-job-*.json"
        )
    reclaim_receipts = _job_artifact_files(root, "reclaim-receipts", "reclaim-*.json")
    reclaim_v2_receipts = _job_artifact_files(
        root, "reclaim-v2-receipts", "reclaim-*.json"
    )
    restore_receipts = _job_artifact_files(root, "restore-receipts", "restore-*.json")
    restore_v2_receipts = _job_artifact_files(
        root, "restore-v2-receipts", "restore-*.json"
    )
    custody_ids = (
        set(local_receipts)
        | set(unsigned_remote_receipts)
        | set(signed_remote_receipts)
        | set(reclaim_receipts)
        | set(restore_receipts)
        | set(reclaim_v2_receipts)
        | set(restore_v2_receipts)
    )
    unknown_custody = custody_ids - scheduled_set
    orphan_custody = custody_ids - completion_ids
    if unknown_custody:
        blockers.append("UNKNOWN_CUSTODY_JOB:" + ",".join(sorted(unknown_custody)))
    if orphan_custody:
        blockers.append(
            "CUSTODY_WITHOUT_COMPLETION:" + ",".join(sorted(orphan_custody))
        )
    remote_without_local = (
        set(unsigned_remote_receipts) | set(signed_remote_receipts)
    ) - set(local_receipts)
    reclaimed_without_local = (set(reclaim_receipts) | set(reclaim_v2_receipts)) - set(
        local_receipts
    )
    restored_without_local = (set(restore_receipts) | set(restore_v2_receipts)) - set(
        local_receipts
    )
    if remote_without_local:
        blockers.append(
            "REMOTE_WITHOUT_LOCAL_ARCHIVE:" + ",".join(sorted(remote_without_local))
        )
    if reclaimed_without_local:
        blockers.append(
            "RECLAIM_WITHOUT_LOCAL_ARCHIVE:" + ",".join(sorted(reclaimed_without_local))
        )
    if restored_without_local:
        blockers.append(
            "RESTORE_WITHOUT_LOCAL_ARCHIVE:" + ",".join(sorted(restored_without_local))
        )
    job_states = []
    for job_sha in scheduled:
        local = local_receipts.get(job_sha, [])
        unsigned_remote = unsigned_remote_receipts.get(job_sha, [])
        signed_remote = signed_remote_receipts.get(job_sha, [])
        reclaimed_v2 = reclaim_v2_receipts.get(job_sha, [])
        restored_v2 = restore_v2_receipts.get(job_sha, [])
        reclaimed = reclaimed_v2 or (
            [] if restored_v2 else reclaim_receipts.get(job_sha, [])
        )
        restored = restored_v2 or (
            [] if reclaimed_v2 else restore_receipts.get(job_sha, [])
        )
        custody_validation = "NOT_APPLICABLE"
        local_archive_deep_verified = False
        remote_attestation_verified = False
        claim_barrier_satisfied = job_sha not in completion_ids
        if any(
            len(rows) > 1
            for rows in (
                local,
                unsigned_remote,
                signed_remote,
                reclaimed,
                restored,
            )
        ):
            blockers.append(f"DUPLICATE_LIFECYCLE_ARTIFACT:{job_sha}")
            state_name = "BLOCKED"
        elif job_sha in active_ids:
            state_name = "RUNNING"
        elif job_sha in missing_completion:
            state_name = "TERMINAL_UNPUBLISHED"
        elif job_sha in completion_without_terminal:
            state_name = "BLOCKED"
        elif job_sha in completion_ids:
            if not local:
                state_name = "COMPLETION_PUBLISHED"
            else:
                receipt = None
                if signed_remote or unsigned_remote or not (restored or reclaimed):
                    receipt = _compact_archive_receipt(
                        receipt_path=local[0],
                        archive_root=archive_root,
                        expected_job_sha256=job_sha,
                    )
                if unsigned_remote:
                    assert receipt is not None
                    _compact_remote_receipt(
                        path=unsigned_remote[0], local_receipt=receipt
                    )
                if signed_remote:
                    assert receipt is not None
                    expected_parent_id = control["execution"].get(
                        "archive_drive_readback_parent_id"
                    )
                    if not isinstance(expected_parent_id, str):
                        raise DojoHistoricalTrainControlError(
                            "signed attestation expected Drive parent is unavailable"
                        )
                    _compact_signed_remote_receipt(
                        path=signed_remote[0],
                        local_receipt=receipt,
                        archive_root=archive_root,
                        expected_drive_parent_id=expected_parent_id,
                    )
                if restored:
                    _compact_restore_receipt(path=restored[0], job_sha256=job_sha)
                    custody_validation = "COMPACT_ONLY_DEEP_REQUIRED_BEFORE_CLAIM"
                    state_name = "RESTORED"
                elif reclaimed:
                    _compact_reclaim_receipt(path=reclaimed[0], job_sha256=job_sha)
                    custody_validation = "COMPACT_ONLY_DEEP_REQUIRED_BEFORE_CLAIM"
                    state_name = "RAW_RECLAIMED"
                    claim_barrier_satisfied = bool(reclaimed_v2 and signed_remote)
                elif signed_remote:
                    custody_validation = "COMPACT_ONLY_DEEP_REQUIRED_BEFORE_CLAIM"
                    state_name = "SIGNED_REMOTE_ATTESTATION_CANDIDATE"
                elif unsigned_remote:
                    custody_validation = "COMPACT_ONLY_DEEP_REQUIRED_BEFORE_CLAIM"
                    state_name = "UNSIGNED_REMOTE_RECEIPT_CANDIDATE"
                else:
                    custody_validation = "COMPACT_ONLY_DEEP_REQUIRED_BEFORE_CLAIM"
                    state_name = "LOCAL_ARCHIVED"
        else:
            state_name = "READY_TO_CLAIM"
        job_states.append(
            {
                "job_sha256": job_sha,
                "state": state_name,
                "custody_validation": custody_validation,
                "local_archive_deep_verified": local_archive_deep_verified,
                "remote_attestation_verified": remote_attestation_verified,
                "signed_remote_attestation_present": len(signed_remote) == 1,
                "exact_v2_raw_reclaim_present": len(reclaimed_v2) == 1,
                "claim_barrier_satisfied": claim_barrier_satisfied,
            }
        )

    states = [row["state"] for row in job_states]
    barrier_policy_enabled = (
        control.get("execution", {}).get("lifecycle_barrier_policy")
        == _LIFECYCLE_BARRIER_POLICY
    )
    unsettled_for_claim = [
        row
        for row in job_states
        if row["job_sha256"] in completion_ids and not row["claim_barrier_satisfied"]
    ]
    if barrier_policy_enabled and len(unsettled_for_claim) > 1:
        blockers.append("MAX_UNRECLAIMED_TERMINAL_JOBS_EXCEEDED")
    if len(active_ids) != execution["active_job_count"]:
        blockers.append("ACTIVE_ARTIFACT_COUNT_MISMATCH")
    if execution["active_job_count"] and any(
        value in {"TERMINAL_UNPUBLISHED", "COMPLETION_PUBLISHED"} for value in states
    ):
        blockers.append("RUNNING_WITH_UNSETTLED_PREDECESSOR")
    if execution.get("exhausted_job_count"):
        blockers.append("EXHAUSTED_UNTERMINAL_JOB")
    hard_blockers = [
        blocker
        for blocker in blockers
        if not blocker.startswith("INCOMPLETE_TERMINAL_PUBLICATION:")
    ]
    if hard_blockers:
        state = "BLOCKED"
        transition = "NONE"
    elif execution["active_job_count"]:
        state = "RUNNING"
        transition = "NONE"
    elif "TERMINAL_UNPUBLISHED" in states:
        state = "TERMINAL_UNPUBLISHED"
        transition = "NONE"
    elif "COMPLETION_PUBLISHED" in states:
        state = "COMPLETION_PUBLISHED"
        transition = "ARCHIVE_NEXT"
    elif (
        barrier_policy_enabled and execution["ready_job_count"] and unsettled_for_claim
    ):
        if any(row["signed_remote_attestation_present"] for row in unsettled_for_claim):
            state = "WAIT_FOR_EXACT_V2_RAW_RECLAIM"
        else:
            state = "WAIT_FOR_SIGNED_REMOTE_ATTESTATION"
        transition = "NONE"
    elif trainer_review_due and execution["ready_job_count"]:
        state = "TRAINER_REVIEW_REQUIRED"
        transition = "NONE"
        blockers.append("TRAINER_REVIEW_DUE")
    elif execution["ready_job_count"]:
        state = "READY_TO_CLAIM"
        transition = "CLAIM_NEXT_JOB"
    elif states:
        settled_priority = (
            "LOCAL_ARCHIVED",
            "SIGNED_REMOTE_ATTESTATION_CANDIDATE",
            "UNSIGNED_REMOTE_RECEIPT_CANDIDATE",
            "RAW_RECLAIMED",
            "RESTORED",
        )
        state = next(
            (candidate for candidate in settled_priority if candidate in states),
            "BLOCKED",
        )
        transition = "NONE"
    else:
        state = "BLOCKED"
        transition = "NONE"
        blockers.append("EMPTY_LIFECYCLE")
    body = {
        "state": state,
        "next_transition": transition,
        "blockers": blockers,
        "job_states": job_states,
        "execution": execution,
        "execution_status_sha256": execution["status_sha256"],
        "remote_attestation_authorized": False,
        "raw_reclaim_transition_allowed": False,
        "status_custody_validation": "COMPACT_ONLY",
        "deep_custody_verification_before_claim": True,
        "trainer_review_due": trainer_review_due,
        "lifecycle_barrier_policy_enabled": barrier_policy_enabled,
        "unreclaimed_terminal_job_count": len(unsettled_for_claim),
        "max_unreclaimed_terminal_jobs": (
            _LIFECYCLE_BARRIER_POLICY["max_unreclaimed_terminal_jobs"]
            if barrier_policy_enabled
            else None
        ),
    }
    return {**body, "lifecycle_sha256": canonical_sha256(body)}


def _nearest_existing_parent(path: Path) -> Path:
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise DojoHistoricalTrainControlError(
                f"no existing parent is available for capacity path: {path}"
            )
        candidate = parent
    return candidate


def _filesystem_capacity_reservations(
    roles: Sequence[tuple[str, Path, int, int]],
) -> dict[str, Any]:
    """Aggregate conservative additions and one recovery floor per filesystem."""

    devices: dict[int, dict[str, Any]] = {}
    role_rows: dict[str, dict[str, Any]] = {}
    for role, path, additional_bytes, floor_bytes in roles:
        if (
            role in role_rows
            or isinstance(additional_bytes, bool)
            or not isinstance(additional_bytes, int)
            or additional_bytes < 0
            or isinstance(floor_bytes, bool)
            or not isinstance(floor_bytes, int)
            or floor_bytes < 0
        ):
            raise DojoHistoricalTrainControlError(
                "filesystem capacity reservation is invalid"
            )
        probe = _nearest_existing_parent(Path(path))
        try:
            state = probe.stat(follow_symlinks=False)
            free_bytes = shutil.disk_usage(probe).free
        except OSError as exc:
            raise DojoHistoricalTrainControlError(
                f"filesystem capacity probe is unavailable: {path}"
            ) from exc
        device = int(state.st_dev)
        device_row = devices.setdefault(
            device,
            {
                "device": device,
                "probe_path": os.fspath(probe),
                "roles": [],
                "free_bytes": int(free_bytes),
                "additional_bytes": 0,
                "recovery_floor_bytes": 0,
            },
        )
        # The same filesystem may be reached through mounts/aliases.  The
        # lowest observed free count is the conservative value.
        device_row["free_bytes"] = min(device_row["free_bytes"], int(free_bytes))
        device_row["additional_bytes"] += additional_bytes
        device_row["recovery_floor_bytes"] = max(
            device_row["recovery_floor_bytes"], floor_bytes
        )
        device_row["roles"].append(role)
        role_rows[role] = {
            "device": device,
            "probe_path": os.fspath(probe),
            "free_bytes": int(free_bytes),
            "additional_bytes": additional_bytes,
            "recovery_floor_bytes": floor_bytes,
        }
    output = []
    for device in sorted(devices):
        row = devices[device]
        row["roles"].sort()
        row["required_bytes"] = row["additional_bytes"] + row["recovery_floor_bytes"]
        output.append(row)
    return {"devices": output, "roles": role_rows}


def _assert_filesystem_reservations(reservations: Mapping[str, Any]) -> None:
    for row in reservations["devices"]:
        if row["free_bytes"] < row["required_bytes"]:
            raise DojoHistoricalTrainControlError(
                "filesystem cannot cover conservative DOJO reservations: "
                f"device={row['device']}, roles={','.join(row['roles'])}, "
                f"free={row['free_bytes']}, required={row['required_bytes']}"
            )


def _disk_capacity_snapshot(
    *,
    root: Path,
    control: Mapping[str, Any],
    estimated_raw_bytes: int,
    estimated_peak_bytes: int,
    estimated_archive_upper_bytes: int | None = None,
) -> dict[str, Any]:
    del estimated_peak_bytes
    execution = control["execution"]
    floor = execution["minimum_free_disk_bytes"]
    archive_floor = max(
        floor,
        int(execution.get("bootstrap_job_working_set_bytes", floor)),
    )
    archive_root = _archive_root(control)
    if archive_root is None:
        reservations = _filesystem_capacity_reservations(
            (("run", root, estimated_raw_bytes, floor),)
        )
        run_row = reservations["roles"]["run"]
        return {
            "shared_filesystem": None,
            "compression_ratio_assumed": False,
            "run_free_bytes": run_row["free_bytes"],
            "run_required_bytes": floor + estimated_raw_bytes,
            "archive_free_bytes": None,
            "archive_required_bytes": 0,
            "filesystem_reservations": reservations["devices"],
        }
    staging_value = execution.get("archive_local_staging_root")
    staging_root = (
        Path(staging_value)
        if isinstance(staging_value, str)
        else Path(tempfile.gettempdir())
    )
    archive_upper = (
        estimated_archive_upper_bytes
        if isinstance(estimated_archive_upper_bytes, int)
        and not isinstance(estimated_archive_upper_bytes, bool)
        and estimated_archive_upper_bytes >= estimated_raw_bytes
        else estimated_raw_bytes
    )
    archive_destination = archive_upper * 2 + min(ARCHIVE_PART_BYTES, archive_upper)
    reservations = _filesystem_capacity_reservations(
        (
            ("run", root, estimated_raw_bytes, floor),
            ("local_staging", staging_root, archive_upper, archive_floor),
            ("archive", archive_root, archive_destination, archive_floor),
        )
    )
    role_rows = reservations["roles"]
    run_device = role_rows["run"]["device"]
    archive_device = role_rows["archive"]["device"]
    run_device_row = next(
        row for row in reservations["devices"] if row["device"] == run_device
    )
    archive_device_row = next(
        row for row in reservations["devices"] if row["device"] == archive_device
    )
    return {
        "shared_filesystem": run_device == archive_device,
        "compression_ratio_assumed": False,
        "run_free_bytes": run_device_row["free_bytes"],
        "run_required_bytes": run_device_row["required_bytes"],
        "archive_free_bytes": archive_device_row["free_bytes"],
        "archive_required_bytes": archive_device_row["required_bytes"],
        "local_staging_device": role_rows["local_staging"]["device"],
        "archive_device": archive_device,
        "run_device": run_device,
        "estimated_archive_upper_bytes": archive_upper,
        "filesystem_reservations": reservations["devices"],
    }


def _assert_disk_capacity(snapshot: Mapping[str, Any]) -> None:
    reservations = snapshot.get("filesystem_reservations")
    if isinstance(reservations, list):
        _assert_filesystem_reservations({"devices": reservations})
        return
    if snapshot["run_free_bytes"] < snapshot["run_required_bytes"]:
        raise DojoHistoricalTrainControlError(
            "run filesystem cannot cover the sealed floor and dynamic next-job "
            f"reservation: free={snapshot['run_free_bytes']}, "
            f"required={snapshot['run_required_bytes']}"
        )


def _configured_conflicting_pairs(
    control: Mapping[str, Any],
) -> list[tuple[Path, Path]]:
    execution = control["execution"]
    roots = execution.get("conflicting_execution_roots", [])
    locks = execution.get("conflicting_run_lock_paths", [])
    if (
        not isinstance(roots, list)
        or not isinstance(locks, list)
        or len(roots) != len(locks)
    ):
        raise DojoHistoricalTrainControlError(
            "conflicting roots and locks must be one-to-one"
        )
    pairs = []
    for root_value, lock_value in zip(roots, locks, strict=True):
        if (
            not isinstance(root_value, str)
            or not isinstance(lock_value, str)
            or not Path(root_value).is_absolute()
            or Path(lock_value) != Path(root_value) / ".historical-train.lock"
        ):
            raise DojoHistoricalTrainControlError(
                "conflicting root and lock binding is invalid"
            )
        pairs.append((Path(root_value), Path(lock_value)))
    if pairs != sorted(pairs, key=lambda row: os.fspath(row[0])):
        raise DojoHistoricalTrainControlError(
            "conflicting roots must use canonical lock order"
        )
    return pairs


def _validate_conflicting_lock(path: Path) -> os.stat_result:
    try:
        state = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalTrainControlError(
            f"configured conflicting lock is unavailable: {path}"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(state.st_mode):
        raise DojoHistoricalTrainControlError(
            f"configured conflicting lock is unsafe: {path}"
        )
    return state


def _find_supersede_receipt_for_root(
    *, current_root: Path, conflicting_root: Path
) -> Path | None:
    store = current_root / "transition-receipts"
    if not store.exists():
        return None
    try:
        state = store.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalTrainControlError(
            "generation transition receipt store is unavailable"
        ) from exc
    if store.is_symlink() or not stat.S_ISDIR(state.st_mode):
        raise DojoHistoricalTrainControlError(
            "generation transition receipt store is unsafe"
        )
    matches = []
    for path in sorted(store.iterdir()):
        if _SUPERSEDE_PENDING_ANCHOR_RE.fullmatch(path.name) is not None:
            if path.is_symlink() or not path.is_file():
                raise DojoHistoricalTrainControlError(
                    "generation transition receipt store contains an unknown entry"
                )
            # The V2 publisher deliberately retains one or more hidden hard-link
            # anchors.  The subsequent receipt verifier checks their bytes,
            # inode linkage, naming, and uniqueness against the final JSON.
            continue
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise DojoHistoricalTrainControlError(
                "generation transition receipt store contains an unknown entry"
            )
        value = _read_json(path, field="generation supersede receipt")
        if value.get("contract") != SUPERSEDE_RECEIPT_CONTRACT:
            continue
        old_generation = value.get("old_generation")
        if isinstance(old_generation, Mapping) and old_generation.get(
            "root"
        ) == os.fspath(conflicting_root):
            matches.append(path)
    if len(matches) > 1:
        raise DojoHistoricalTrainControlError(
            "multiple supersede receipts name one conflicting generation"
        )
    return matches[0] if matches else None


def _verify_supersede_receipt_chain(
    control: Mapping[str, Any],
    *,
    current_root: Path,
    conflicting_root: Path,
    locked_descriptors: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Verify the unique append-only successor path into the current generation."""

    candidate_successors = [
        root for root, _ in _configured_conflicting_pairs(control)
    ] + [current_root]
    if len(candidate_successors) != len(set(candidate_successors)):
        raise DojoHistoricalTrainControlError(
            "generation supersede lineage contains duplicate roots"
        )
    cursor = conflicting_root
    seen = {cursor}
    receipts: list[dict[str, Any]] = []
    for _ in range(len(candidate_successors)):
        matches: list[tuple[Path, Path]] = []
        for successor in candidate_successors:
            if successor in seen or not successor.exists():
                continue
            receipt_path = _find_supersede_receipt_for_root(
                current_root=successor,
                conflicting_root=cursor,
            )
            if receipt_path is not None:
                matches.append((successor, receipt_path))
        if not matches:
            return []
        if len(matches) != 1:
            raise DojoHistoricalTrainControlError(
                "generation supersede lineage has multiple successor paths"
            )
        successor, receipt_path = matches[0]
        try:
            descriptor = (
                locked_descriptors.get(os.fspath(cursor))
                if locked_descriptors is not None
                else None
            )
            if descriptor is None:
                receipt = verify_historical_supersede_receipt_file(
                    receipt_path,
                    old_root=cursor,
                    new_root=successor,
                )
            else:
                receipt = verify_historical_supersede_receipt_store_locked(
                    old_root=cursor,
                    new_root=successor,
                    old_lock_descriptor=descriptor,
                )
        except DojoHistoricalSupersedeReceiptError as exc:
            raise DojoHistoricalTrainControlError(
                "generation supersede receipt chain failed verification: " f"{exc}"
            ) from exc
        receipts.append(receipt)
        cursor = successor
        if cursor == current_root:
            return receipts
        seen.add(cursor)
    raise DojoHistoricalTrainControlError(
        "generation supersede lineage exceeded its configured root bound"
    )


def _conflicting_generation_statuses(
    control: Mapping[str, Any],
    *,
    current_root: Path | None = None,
    locked_descriptors: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for root, lock_path in _configured_conflicting_pairs(control):
        if not root.exists():
            rows.append(
                {
                    "output_root": str(root),
                    "exists": False,
                    "active_job_count": 0,
                    "terminal_job_count": 0,
                    "superseded_by_current_generation": False,
                    "supersede_receipt_sha256": None,
                }
            )
            continue
        _validate_conflicting_lock(lock_path)
        plan = validate_long_horizon_train_plan(
            _read_json(root / "plan.json", field="conflicting generation plan")
        )
        schedule = validate_long_horizon_stream_schedule(
            _read_json(root / "schedule.json", field="conflicting generation schedule"),
            plan=plan,
        )
        status = long_horizon_execution_status(
            root / "execution-state", schedule=schedule, plan=plan
        )
        superseded = False
        supersede_receipt_sha256 = None
        if status["active_job_count"] > 0 and current_root is not None:
            receipt_chain = _verify_supersede_receipt_chain(
                control,
                current_root=current_root,
                conflicting_root=root,
                locked_descriptors=locked_descriptors,
            )
            if receipt_chain:
                superseded = True
                supersede_receipt_sha256 = canonical_sha256(
                    [receipt["receipt_sha256"] for receipt in receipt_chain]
                )
        rows.append(
            {
                "output_root": str(root),
                "exists": True,
                "active_job_count": status["active_job_count"],
                "terminal_job_count": status["terminal_job_count"],
                "status_sha256": status["status_sha256"],
                "superseded_by_current_generation": superseded,
                "supersede_receipt_sha256": supersede_receipt_sha256,
            }
        )
    return rows


def _assert_dynamic_machine_capacity(
    control: Mapping[str, Any],
    *,
    current_root: Path | None = None,
    locked_descriptors: Mapping[str, int] | None = None,
) -> None:
    execution = control["execution"]
    load_fraction = execution.get("max_one_minute_load_per_cpu", 0.8)
    if (
        isinstance(load_fraction, bool)
        or not isinstance(load_fraction, (int, float))
        or not 0.1 <= float(load_fraction) <= 2.0
    ):
        raise DojoHistoricalTrainControlError("load-per-CPU gate is invalid")
    cpu_count = os.cpu_count() or 1
    one_minute_load = os.getloadavg()[0]
    load_limit = cpu_count * float(load_fraction)
    if one_minute_load > load_limit:
        raise DojoHistoricalTrainControlError(
            "machine load is above the dynamic historical TRAIN gate: "
            f"load_1m={one_minute_load:.2f}, limit={load_limit:.2f}, "
            f"logical_cpu={cpu_count}"
        )
    active_conflicts = [
        row
        for row in _conflicting_generation_statuses(
            control,
            current_root=current_root,
            locked_descriptors=locked_descriptors,
        )
        if row["active_job_count"] > 0 and not row["superseded_by_current_generation"]
    ]
    if active_conflicts:
        raise DojoHistoricalTrainControlError(
            "a conflicting historical generation has an active or orphaned claim: "
            + ", ".join(row["output_root"] for row in active_conflicts)
        )


def _acquire_conflicting_run_locks(control: Mapping[str, Any]) -> dict[str, int]:
    descriptors: dict[str, int] = {}
    try:
        for root, path in _configured_conflicting_pairs(control):
            if not root.exists():
                continue
            _validate_conflicting_lock(path)
            descriptor = _open_stable_lock_file(path, create=False)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise DojoHistoricalTrainControlError(
                        f"another heavy DOJO run owns the configured lock: {path}"
                    ) from exc
                _assert_lock_descriptor_identity(descriptor)
            except BaseException:
                _release_one_historical_operation_lock(descriptor)
                raise
            try:
                descriptors[os.fspath(root)] = descriptor
            except BaseException:
                _release_one_historical_operation_lock(descriptor)
                raise
        return descriptors
    except BaseException:
        first_cleanup_error: BaseException | None = None
        for descriptor in reversed(list(descriptors.values())):
            try:
                _release_one_historical_operation_lock(descriptor)
            except BaseException as exc:
                if first_cleanup_error is None:
                    first_cleanup_error = exc
        if first_cleanup_error is not None:
            raise first_cleanup_error
        raise


def _release_historical_operation_locks(
    *,
    run_lock_descriptor: int | None,
    global_lock_descriptor: int | None,
    conflicting_lock_descriptors: Mapping[str, int],
) -> None:
    first_error: BaseException | None = None

    def release(descriptor: int) -> None:
        nonlocal first_error
        try:
            _release_one_historical_operation_lock(descriptor)
        except BaseException as exc:
            if first_error is None:
                first_error = exc

    for descriptor in reversed(list(conflicting_lock_descriptors.values())):
        release(descriptor)
    if global_lock_descriptor is not None:
        release(global_lock_descriptor)
    if run_lock_descriptor is not None:
        release(run_lock_descriptor)
    if first_error is not None:
        raise first_error


def _lock_identity(state: os.stat_result) -> tuple[int, int]:
    return state.st_dev, state.st_ino


def _assert_lock_descriptor_identity(descriptor: int) -> None:
    binding = _LOCK_BINDINGS.get(descriptor)
    if binding is None:
        raise DojoHistoricalTrainControlError(
            "historical operation lock descriptor has no path binding"
        )
    parent_descriptor, parent_path, name, parent_identity, lock_identity = binding
    try:
        opened_parent = os.fstat(parent_descriptor)
        path_parent = parent_path.stat(follow_symlinks=False)
        opened_lock = os.fstat(descriptor)
        named_lock = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise DojoHistoricalTrainControlError(
            f"historical operation lock binding is unavailable: {parent_path / name}"
        ) from exc
    if (
        not stat.S_ISDIR(opened_parent.st_mode)
        or not stat.S_ISDIR(path_parent.st_mode)
        or _lock_identity(opened_parent) != parent_identity
        or _lock_identity(path_parent) != parent_identity
        or not stat.S_ISREG(opened_lock.st_mode)
        or not stat.S_ISREG(named_lock.st_mode)
        or _lock_identity(opened_lock) != lock_identity
        or _lock_identity(named_lock) != lock_identity
    ):
        raise DojoHistoricalTrainControlError(
            f"historical operation lock or parent was replaced: {parent_path / name}"
        )


def _assert_historical_operation_lock_identities(
    *,
    run_lock_descriptor: int,
    global_lock_descriptor: int | None,
    conflicting_lock_descriptors: Mapping[str, int],
) -> None:
    _assert_lock_descriptor_identity(run_lock_descriptor)
    if global_lock_descriptor is not None:
        _assert_lock_descriptor_identity(global_lock_descriptor)
    for descriptor in conflicting_lock_descriptors.values():
        _assert_lock_descriptor_identity(descriptor)


def _release_one_historical_operation_lock(descriptor: int) -> None:
    first_error: BaseException | None = None
    try:
        _assert_lock_descriptor_identity(descriptor)
    except BaseException as exc:
        first_error = exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except BaseException as exc:
        if first_error is None:
            first_error = exc
    binding = _LOCK_BINDINGS.pop(descriptor, None)
    try:
        os.close(descriptor)
    except BaseException as exc:
        if first_error is None:
            first_error = exc
    if binding is not None:
        try:
            os.close(binding[0])
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _open_stable_lock_file(path: Path, *, create: bool = True) -> int:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.name in {"", ".", ".."}:
        raise DojoHistoricalTrainControlError(
            f"historical operation lock path is invalid: {path}"
        )
    parent = candidate.parent
    parent_descriptor: int | None = None
    descriptor: int | None = None
    try:
        parent_before = parent.stat(follow_symlinks=False)
        if not stat.S_ISDIR(parent_before.st_mode):
            raise DojoHistoricalTrainControlError(
                f"historical operation lock parent is unsafe: {parent}"
            )
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        parent_opened = os.fstat(parent_descriptor)
        parent_after = parent.stat(follow_symlinks=False)
        if _lock_identity(parent_before) != _lock_identity(
            parent_opened
        ) or _lock_identity(parent_after) != _lock_identity(parent_opened):
            raise DojoHistoricalTrainControlError(
                f"historical operation lock parent changed while opened: {parent}"
            )
        try:
            before = os.stat(
                candidate.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            before = None
        if before is not None and not stat.S_ISREG(before.st_mode):
            raise DojoHistoricalTrainControlError(
                f"historical operation lock is unsafe: {candidate}"
            )
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        if create:
            flags |= os.O_CREAT
        descriptor = os.open(candidate.name, flags, 0o600, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        after = os.stat(
            candidate.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or _lock_identity(opened) != _lock_identity(after)
            or (before is not None and _lock_identity(before) != _lock_identity(opened))
        ):
            raise DojoHistoricalTrainControlError(
                f"historical operation lock changed while opened: {candidate}"
            )
        _LOCK_BINDINGS[descriptor] = (
            parent_descriptor,
            parent,
            candidate.name,
            _lock_identity(parent_opened),
            _lock_identity(opened),
        )
        parent_descriptor = None
        result = descriptor
        descriptor = None
        return result
    except DojoHistoricalTrainControlError:
        raise
    except OSError as exc:
        raise DojoHistoricalTrainControlError(
            f"historical operation lock could not be opened safely: {candidate}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _acquire_historical_operation_locks(
    *, root: Path, control: Mapping[str, Any]
) -> tuple[int, int | None, dict[str, int]]:
    run_lock_descriptor: int | None = None
    global_lock_descriptor: int | None = None
    conflicting_lock_descriptors: dict[str, int] = {}
    try:
        run_lock_descriptor = _open_stable_lock_file(root / ".historical-train.lock")
        try:
            fcntl.flock(run_lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DojoHistoricalTrainControlError(
                "another historical TRAIN job already owns the run lock"
            ) from exc
        _assert_lock_descriptor_identity(run_lock_descriptor)
        global_lock_value = control["execution"].get("global_heavy_lock_path")
        if isinstance(global_lock_value, str):
            global_lock_path = Path(global_lock_value)
            global_lock_path.parent.mkdir(parents=True, exist_ok=True)
            global_lock_descriptor = _open_stable_lock_file(global_lock_path)
            try:
                fcntl.flock(global_lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DojoHistoricalTrainControlError(
                    "another DOJO heavy operation owns the machine-wide lease"
                ) from exc
            _assert_lock_descriptor_identity(global_lock_descriptor)
        conflicting_lock_descriptors = _acquire_conflicting_run_locks(control)
        return (
            run_lock_descriptor,
            global_lock_descriptor,
            conflicting_lock_descriptors,
        )
    except BaseException:
        _release_historical_operation_locks(
            run_lock_descriptor=run_lock_descriptor,
            global_lock_descriptor=global_lock_descriptor,
            conflicting_lock_descriptors=conflicting_lock_descriptors,
        )
        raise


def _reload_generation_under_lock(
    *,
    repo_root: Path,
    run_control_path: Path,
    prelock_generation: tuple[
        dict[str, Any],
        Path,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
    operation: str = "run",
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    locked_generation = _load_generation(
        repo_root=repo_root,
        run_control_path=run_control_path,
        operation=operation,
    )
    if locked_generation != prelock_generation:
        raise DojoHistoricalTrainControlError(
            "generation artifacts changed after operation locks were acquired"
        )
    return locked_generation


def archive_next_completed_job(
    *,
    repo_root: Path,
    run_control_path: Path,
) -> dict[str, Any]:
    """Archive at most one terminal job and never claim a replay job."""

    prelock_generation = _load_generation(
        repo_root=repo_root,
        run_control_path=run_control_path,
        operation="custody",
    )
    control, root, _, _, _, _, _ = prelock_generation
    (
        run_lock_descriptor,
        global_lock_descriptor,
        conflicting_lock_descriptors,
    ) = _acquire_historical_operation_locks(root=root, control=control)
    try:
        locked_generation = _reload_generation_under_lock(
            repo_root=repo_root,
            run_control_path=run_control_path,
            prelock_generation=prelock_generation,
            operation="custody",
        )
        control, locked_root, plan, schedule, _, _, _ = locked_generation
        if locked_root != root:
            raise DojoHistoricalTrainControlError(
                "archive generation changed after its operation locks were acquired"
            )
        lifecycle = evaluate_historical_lifecycle(
            root=root,
            control=control,
            plan=plan,
            schedule=schedule,
        )
        if lifecycle["state"] in {"BLOCKED", "TERMINAL_UNPUBLISHED", "RUNNING"}:
            raise DojoHistoricalTrainControlError(
                "archive-next lifecycle is blocked: "
                + ",".join(lifecycle["blockers"] or [lifecycle["state"]])
            )
        if lifecycle["state"] != "COMPLETION_PUBLISHED":
            return {
                "status": "ARCHIVE_NOT_PENDING",
                "output_root": str(root),
                "archive": None,
                "lifecycle": lifecycle,
                "next_job_started": False,
                **_AUTHORITY,
            }
        _assert_dynamic_machine_capacity(
            control,
            current_root=root,
            locked_descriptors=conflicting_lock_descriptors,
        )
        recovered_archives = _archive_pending_completed_jobs(
            root=root,
            control=control,
            mutation_guard=lambda: _assert_historical_operation_lock_identities(
                run_lock_descriptor=run_lock_descriptor,
                global_lock_descriptor=global_lock_descriptor,
                conflicting_lock_descriptors=conflicting_lock_descriptors,
            ),
        )
        if recovered_archives:
            return {
                "status": "ARCHIVE_RECOVERED",
                "output_root": str(root),
                "archive": recovered_archives[0],
                "lifecycle_before": lifecycle,
                "next_job_started": False,
                **_AUTHORITY,
            }
        return {
            "status": "ARCHIVE_NOT_PENDING",
            "output_root": str(root),
            "archive": None,
            "next_job_started": False,
            **_AUTHORITY,
        }
    finally:
        _release_historical_operation_locks(
            run_lock_descriptor=run_lock_descriptor,
            global_lock_descriptor=global_lock_descriptor,
            conflicting_lock_descriptors=conflicting_lock_descriptors,
        )


def advance_one_historical_transition(
    *, repo_root: Path, run_control_path: Path
) -> dict[str, Any]:
    """Heartbeat entrypoint: select and execute at most one lifecycle transition."""

    control, root, plan, schedule, _, _, _ = _load_generation(
        repo_root=repo_root,
        run_control_path=run_control_path,
        operation="custody",
    )
    lifecycle = evaluate_historical_lifecycle(
        root=root,
        control=control,
        plan=plan,
        schedule=schedule,
    )
    selected = lifecycle["next_transition"]
    if selected == "ARCHIVE_NEXT":
        operation = archive_next_completed_job(
            repo_root=repo_root,
            run_control_path=run_control_path,
        )
    elif selected == "CLAIM_NEXT_JOB":
        operation = run_next_job(
            repo_root=repo_root,
            run_control_path=run_control_path,
        )
    elif selected == "NONE":
        return {
            "status": "NO_LIFECYCLE_TRANSITION",
            "output_root": str(root),
            "selected_transition": "NONE",
            "transition_execution_count": 0,
            "fallthrough_allowed": False,
            "lifecycle_before": lifecycle,
            **_AUTHORITY,
        }
    else:
        raise DojoHistoricalTrainControlError(
            f"lifecycle selected an unknown transition: {selected}"
        )
    return {
        **operation,
        "heartbeat_step": {
            "selected_transition": selected,
            "transition_execution_count": 1,
            "fallthrough_allowed": False,
            "lifecycle_before_sha256": lifecycle["lifecycle_sha256"],
            "operation_revalidated_under_lock": True,
        },
    }


def run_next_job(
    *,
    repo_root: Path,
    run_control_path: Path,
) -> dict[str, Any]:
    """Execute exactly one claimed historical job under the sealed G2 policy."""

    prelock_generation = _load_generation(
        repo_root=repo_root, run_control_path=run_control_path
    )
    (
        control,
        root,
        plan,
        schedule,
        runtime_seal,
        source_manifest,
        catalog_wrapper,
    ) = prelock_generation
    (
        lock_descriptor,
        global_lock_descriptor,
        conflicting_lock_descriptors,
    ) = _acquire_historical_operation_locks(root=root, control=control)
    try:
        (
            control,
            locked_root,
            plan,
            schedule,
            runtime_seal,
            source_manifest,
            catalog_wrapper,
        ) = _reload_generation_under_lock(
            repo_root=repo_root,
            run_control_path=run_control_path,
            prelock_generation=prelock_generation,
        )
        if locked_root != root:
            raise DojoHistoricalTrainControlError(
                "run generation changed after its operation locks were acquired"
            )
        lifecycle = evaluate_historical_lifecycle(
            root=root,
            control=control,
            plan=plan,
            schedule=schedule,
        )
        milestone = _milestone_status(root, schedule, publish=False)
        if (
            milestone["trainer_review_due"]
            and lifecycle["execution"]["ready_job_count"]
        ):
            raise DojoHistoricalTrainControlError(
                "trainer review is due before the next historical claim"
            )
        if lifecycle["state"] != "READY_TO_CLAIM":
            raise DojoHistoricalTrainControlError(
                "run-next lifecycle does not permit a claim: "
                + ",".join(lifecycle["blockers"] or [lifecycle["state"]])
            )
        _assert_dynamic_machine_capacity(
            control,
            current_root=root,
            locked_descriptors=conflicting_lock_descriptors,
        )
        _deep_verify_completed_job_custody(root=root, control=control)
        planned_coordinate_count = max(
            job["coordinate_count"] for job in schedule["jobs"]
        )
        (
            estimated_raw_bytes,
            estimated_archive_upper_bytes,
            estimated_peak_bytes,
        ) = _estimated_next_job_bytes(
            control=control,
            archive_root=_archive_root(control),
            planned_coordinate_count=planned_coordinate_count,
        )
        _assert_disk_capacity(
            _disk_capacity_snapshot(
                root=root,
                control=control,
                estimated_raw_bytes=estimated_raw_bytes,
                estimated_archive_upper_bytes=estimated_archive_upper_bytes,
                estimated_peak_bytes=estimated_peak_bytes,
            )
        )

        def mutation_guard() -> None:
            _assert_historical_operation_lock_identities(
                run_lock_descriptor=lock_descriptor,
                global_lock_descriptor=global_lock_descriptor,
                conflicting_lock_descriptors=conflicting_lock_descriptors,
            )

        mutation_guard()
        handoff = claim_next_long_horizon_job(
            root / "execution-state",
            schedule=schedule,
            plan=plan,
            runner_id=_runner_id(control["fixed_inputs"]),
        )
        job = handoff["job"]
        claim = handoff["claim"]
        job_sha = job["job_sha256"]
        job_dir = root / "jobs" / job_sha
        mutation_guard()
        job_dir.mkdir(parents=True, exist_ok=True)
        mutation_guard()
        _write_once(job_dir / "runner-handoff.json", handoff)
        source_root = root / "source-slices"
        relative = Path(
            job["source_binding_id"],
            job["month"],
            f"{job['intrabar_path']}-{job_sha}.jsonl",
        )
        mutation_guard()
        (source_root / relative.parent).mkdir(parents=True, exist_ok=True)
        receipt_path = job_dir / "source-slice-receipt.json"
        if receipt_path.exists():
            receipt = validate_sparse_month_source_slice_receipt(
                _read_json(receipt_path, field="source slice receipt"),
                job=job,
                source_manifest=source_manifest,
            )
        else:
            if (source_root / relative).exists():
                raise DojoHistoricalTrainControlError(
                    "orphan source slice exists without its immutable receipt"
                )
            try:
                mutation_guard()
                receipt = build_sparse_month_source_slice_receipt(
                    source_root=source_root,
                    relative_path=relative.as_posix(),
                    job=job,
                    source_manifest=source_manifest,
                )
            except (DojoLongHorizonEconomicRunnerError, SparseReplayError) as exc:
                mutation_guard()
                return _seal_source_failure(
                    root=root,
                    schedule=schedule,
                    plan=plan,
                    handoff=handoff,
                    error=exc,
                )
            mutation_guard()
            _write_once(receipt_path, receipt)
        repo = Path(repo_root).resolve(strict=True)
        registry = _read_json(
            repo / control["fixed_inputs"]["registry_path"], field="G2 registry"
        )
        catalog = catalog_wrapper.get("worker_catalog")
        if not isinstance(catalog, list):
            raise DojoHistoricalTrainControlError("worker catalog is invalid")
        runtimes = _coordinate_runtimes(
            handoff=handoff,
            plan=plan,
            catalog=catalog,
            registry=registry,
            control=control,
        )
        mutation_guard()
        _write_once(
            job_dir / "coordinate-runtimes.json",
            {"coordinate_runtimes": runtimes},
        )
        carries = _carry_inputs(root, handoff)
        if carries:
            mutation_guard()
            _write_once(
                job_dir / "carry-states-input.json",
                {"economic_carry_states_by_slot": carries},
            )
        evidence_root = job_dir / "economic-evidence"
        mutation_guard()
        evidence_root.mkdir(parents=True, exist_ok=True)
        result_path = root / "job-results" / f"{job_sha}.json"
        if result_path.exists():
            result = _verified_job_result(
                _read_json(result_path, field="economic job result"), handoff=handoff
            )
        else:
            runtime_factory = build_tuned_strategy_runtime_factory(
                runtime_seal, repo_root=repo
            )
            mutation_guard()
            result = run_long_horizon_economic_job(
                runner_handoff=handoff,
                plan=plan,
                source_root=source_root,
                source_manifest=source_manifest,
                source_slice_receipt=receipt,
                economic_evidence_root=evidence_root,
                worker_catalog=catalog,
                coordinate_runtimes=runtimes,
                worker_runtime_factory=runtime_factory,
                worker_runtime_binding_sha256=runtime_seal["runtime_binding_sha256"],
                worker_runtime_seal=runtime_seal,
                worker_runtime_repo_root=repo,
                carry_states_by_slot=carries,
            )
            result = _verified_job_result(result, handoff=handoff)
            mutation_guard()
            _write_once(result_path, result)
        mutation_guard()
        _publish_carries(root, result)
        mutation_guard()
        record_long_horizon_coordinate_results(
            root / "execution-state",
            schedule=schedule,
            plan=plan,
            claim_sha256=claim["claim_sha256"],
            results=result["coordinate_results"],
        )
        mutation_guard()
        terminal = seal_long_horizon_attempt(
            root / "execution-state",
            schedule=schedule,
            plan=plan,
            claim_sha256=claim["claim_sha256"],
        )
        completion_body = {
            "contract": JOB_COMPLETION_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "job_sha256": job_sha,
            "claim_sha256": claim["claim_sha256"],
            "source_binding_id": job["source_binding_id"],
            "month": job["month"],
            "intrabar_path": job["intrabar_path"],
            "coordinate_result_count": result["coordinate_result_count"],
            "complete_coordinate_count": result["complete_coordinate_count"],
            "failed_coordinate_count": result["failed_coordinate_count"],
            "job_status": result["job_status"],
            "economic_job_result_sha256": result["economic_job_result_sha256"],
            "terminal_sha256": terminal["terminal_manifest"]["terminal_sha256"],
            "free_disk_bytes_after": shutil.disk_usage(root).free,
            **_AUTHORITY,
        }
        completion = {
            **completion_body,
            "completion_sha256": canonical_sha256(completion_body),
        }
        mutation_guard()
        _write_once(job_dir / "completion.json", completion)
        mutation_guard()
        milestone = _milestone_status(root, schedule)
        return {
            "status": "JOB_COMPLETE",
            "output_root": str(root),
            "job": {
                "job_sha256": job_sha,
                "source_binding_id": job["source_binding_id"],
                "month": job["month"],
                "intrabar_path": job["intrabar_path"],
            },
            "coordinate_result_count": result["coordinate_result_count"],
            "complete_coordinate_count": result["complete_coordinate_count"],
            "failed_coordinate_count": result["failed_coordinate_count"],
            "archive": None,
            "next_transition": "ARCHIVE_NEXT",
            "trainer_milestone": milestone,
            **_AUTHORITY,
        }
    finally:
        _release_historical_operation_locks(
            run_lock_descriptor=lock_descriptor,
            global_lock_descriptor=global_lock_descriptor,
            conflicting_lock_descriptors=conflicting_lock_descriptors,
        )


def generation_status(*, repo_root: Path, run_control_path: Path) -> dict[str, Any]:
    """Return verified compact progress without opening partial economics."""

    control, root, plan, schedule, _, _, _ = _load_generation(
        repo_root=repo_root,
        run_control_path=run_control_path,
        operation="custody",
    )
    lifecycle = evaluate_historical_lifecycle(
        root=root,
        control=control,
        plan=plan,
        schedule=schedule,
    )
    execution = lifecycle["execution"]
    milestone = _milestone_status(root, schedule, publish=False)
    archive_root = _archive_root(control)
    archive_receipts = []
    reclaimed_candidates = []
    restored_candidates = []
    remote_receipt_candidate_count = 0
    remote_receipt_candidate_logical_bytes = 0
    signed_remote_attestation_candidate_count = 0
    signed_remote_attestation_candidate_logical_bytes = 0
    reclaimable_bytes = 0
    if archive_root is not None and (archive_root / "receipts").is_dir():
        matched_receipt_paths: set[Path] = set()
        for completion_path in sorted((root / "jobs").glob("*/completion.json")):
            job_sha = completion_path.parent.name
            matches = sorted((archive_root / "receipts").glob(f"job-{job_sha}-*.json"))
            if not matches:
                continue
            if len(matches) != 1:
                raise DojoHistoricalTrainControlError(
                    "multiple historical archive receipts name one job"
                )
            receipt = _compact_archive_receipt(
                receipt_path=matches[0],
                archive_root=archive_root,
                expected_job_sha256=job_sha,
            )
            restored_v2 = sorted(
                (root / "restore-v2-receipts").glob(f"restore-{job_sha}-*.json")
            )
            reclaimed_v2 = sorted(
                (root / "reclaim-v2-receipts").glob(f"reclaim-{job_sha}-*.json")
            )
            restored = restored_v2 or (
                []
                if reclaimed_v2
                else sorted(
                    (root / "restore-receipts").glob(f"restore-{job_sha}-*.json")
                )
            )
            reclaimed = reclaimed_v2 or (
                []
                if restored_v2
                else sorted(
                    (root / "reclaim-receipts").glob(f"reclaim-{job_sha}-*.json")
                )
            )
            if restored:
                if len(restored) != 1:
                    raise DojoHistoricalTrainControlError(
                        "multiple raw restore receipts name one job"
                    )
                restored_candidates.append(
                    _compact_restore_receipt(path=restored[0], job_sha256=job_sha)
                )
            elif reclaimed:
                if len(reclaimed) != 1:
                    raise DojoHistoricalTrainControlError(
                        "multiple raw reclaim receipts name one job"
                    )
                reclaimed_candidates.append(
                    _compact_reclaim_receipt(path=reclaimed[0], job_sha256=job_sha)
                )
            else:
                unsigned_remote_paths = sorted(
                    (archive_root / "remote-receipts").glob(
                        f"remote-job-{job_sha}-{receipt['manifest_sha256']}-*.json"
                    )
                )
                signed_remote_paths = sorted(
                    (archive_root / "remote-receipts").glob(
                        f"signed-job-{job_sha}-{receipt['manifest_sha256']}-*.json"
                    )
                )
                if unsigned_remote_paths:
                    if len(unsigned_remote_paths) != 1:
                        raise DojoHistoricalTrainControlError(
                            "multiple unsigned remote readback receipts name one job"
                        )
                    _compact_remote_receipt(
                        path=unsigned_remote_paths[0], local_receipt=receipt
                    )
                    remote_receipt_candidate_count += 1
                    remote_receipt_candidate_logical_bytes += int(
                        receipt["total_source_bytes"]
                    )
                if signed_remote_paths:
                    if len(signed_remote_paths) != 1:
                        raise DojoHistoricalTrainControlError(
                            "multiple signed remote attestations name one job"
                        )
                    _compact_signed_remote_receipt(
                        path=signed_remote_paths[0],
                        local_receipt=receipt,
                        archive_root=archive_root,
                        expected_drive_parent_id=control["execution"][
                            "archive_drive_readback_parent_id"
                        ],
                    )
                    signed_remote_attestation_candidate_count += 1
                    signed_remote_attestation_candidate_logical_bytes += int(
                        receipt["total_source_bytes"]
                    )
            archive_receipts.append(receipt)
            matched_receipt_paths.update(matches)
        all_receipt_paths = set((archive_root / "receipts").glob("job-*.json"))
        if all_receipt_paths != matched_receipt_paths:
            raise DojoHistoricalTrainControlError(
                "historical archive receipt does not name a completed job"
            )
    completion_paths = list((root / "jobs").glob("*/completion.json"))
    completion_count = len(completion_paths)
    archived_job_count = len(archive_receipts)
    baseline = control["execution"].get("capacity_baseline", {})
    baseline_raw = (
        baseline.get("raw_bytes_per_job") if isinstance(baseline, Mapping) else None
    )
    planned_coordinate_count = max(job["coordinate_count"] for job in schedule["jobs"])
    next_raw, next_archive_upper, next_peak = _estimated_next_job_bytes(
        control=control,
        archive_root=archive_root,
        planned_coordinate_count=planned_coordinate_count,
    )
    disk_capacity = _disk_capacity_snapshot(
        root=root,
        control=control,
        estimated_raw_bytes=next_raw,
        estimated_archive_upper_bytes=next_archive_upper,
        estimated_peak_bytes=next_peak,
    )
    capacity_comparison = {
        "baseline": dict(baseline) if isinstance(baseline, Mapping) else {},
        "planned_coordinate_count_per_job": max(
            job["coordinate_count"] for job in schedule["jobs"]
        ),
        "estimated_next_raw_bytes": next_raw,
        "estimated_next_archive_upper_bound_bytes": next_archive_upper,
        "estimated_next_peak_with_archive_staging_bytes": next_peak,
        "next_job_capacity_reservation_policy": {
            "compression_ratio_assumed": False,
            "run_source_multiplier": 1,
            "tar_pax_overhead_reserved": True,
            "zstd_framing_overhead_reserved": True,
            "local_staging_uses_archive_upper_bound": True,
            "archive_final_and_readback_copies": 2,
            "archive_part_temp_bytes": min(ARCHIVE_PART_BYTES, next_archive_upper),
        },
        "estimated_raw_to_baseline_ratio": (
            float(next_raw) / int(baseline_raw)
            if isinstance(next_raw, int)
            and isinstance(baseline_raw, int)
            and baseline_raw > 0
            else None
        ),
        "terminal_completion_count": completion_count,
        "locally_verified_archive_count": 0,
        "locally_compact_validated_archive_count": archived_job_count,
        "locally_deep_verified_archive_count": 0,
        "unarchived_terminal_count": max(0, completion_count - archived_job_count),
        "raw_bytes_bound_by_archive_receipts": sum(
            row["total_source_bytes"] for row in archive_receipts
        ),
        "archive_bytes": sum(row["archive_size_bytes"] for row in archive_receipts),
        "archive_and_readback_local_footprint_bytes": sum(
            _archive_local_footprint_bytes(row, archive_root=archive_root)
            for row in archive_receipts
        )
        if archive_root is not None
        else 0,
        "heartbeat_archive_verification_level": (
            "RECEIPT_SEAL_AND_LOCAL_METADATA_ONLY"
        ),
        "local_archive_deep_verified_in_status": False,
        "deep_archive_verification_before_next_job": True,
        "remote_verified_archive_count": 0,
        "remote_receipt_candidate_count": remote_receipt_candidate_count,
        "remote_receipt_candidate_logical_bytes": (
            remote_receipt_candidate_logical_bytes
        ),
        "remote_receipt_candidates_require_deep_verification": True,
        "signed_remote_attestation_candidate_count": (
            signed_remote_attestation_candidate_count
        ),
        "signed_remote_attestation_candidate_logical_bytes": (
            signed_remote_attestation_candidate_logical_bytes
        ),
        "signed_remote_attestations_require_deep_verification": True,
        "unsigned_v1_remote_receipts_authoritative": False,
        "raw_reclaimed_job_count": 0,
        "raw_reclaimed_logical_bytes": 0,
        "raw_reclaim_receipt_candidate_count": len(reclaimed_candidates),
        "raw_reclaim_receipt_candidate_logical_bytes": sum(
            row["reclaimed_logical_bytes"] for row in reclaimed_candidates
        ),
        "raw_restore_receipt_candidate_count": len(restored_candidates),
        "reclaimable_bytes": reclaimable_bytes,
        "deep_verified_reclaimable_bytes": reclaimable_bytes,
        "reclaimable_bytes_verification_level": "DEEP_REMOTE_READBACK_REQUIRED",
        "reclaimable_bytes_requires_deep_verification": True,
        "raw_reclaim_requires_remote_readback": True,
    }
    return {
        "contract": "QR_DOJO_HISTORICAL_TRAIN_STATUS_V1",
        "schema_version": SCHEMA_VERSION,
        "output_root": str(root),
        "execution": execution,
        "lifecycle": lifecycle,
        "trainer_milestone": milestone,
        "capacity_comparison": capacity_comparison,
        "disk_capacity": disk_capacity,
        "machine_load": {
            "logical_cpu_count": os.cpu_count() or 1,
            "load_average_1m": os.getloadavg()[0],
            "configured_load_per_cpu_max": control["execution"].get(
                "max_one_minute_load_per_cpu"
            ),
        },
        "conflicting_generation_statuses": _conflicting_generation_statuses(
            control,
            current_root=root,
        ),
        "free_disk_bytes": shutil.disk_usage(root).free,
        **_AUTHORITY,
    }


__all__ = [
    "DojoHistoricalTrainControlError",
    "advance_one_historical_transition",
    "archive_next_completed_job",
    "evaluate_historical_lifecycle",
    "generation_status",
    "prepare_generation",
    "run_next_job",
]

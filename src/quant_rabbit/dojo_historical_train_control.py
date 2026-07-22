"""Reviewed, research-only controller for continuous DOJO historical TRAIN.

The controller closes the gap between the sealed long-horizon primitives and
an actual replay process.  It accepts one repository-owned run-control policy,
materializes one observed-only month slice, executes one immutable job, then
publishes the complete coordinate denominator and carry state.  It has no
broker client, credential input, live permission, promotion path, or model
client.

Generation mutation is deliberately outside this module.  Economic results
become trainer input only after both intrabar paths for a fixed six-month M5
block are terminal.  A trainer response must create a new immutable generation;
it cannot modify the running G2 schedule.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_bot_trainer import (
    PROPOSAL_CONTRACT,
    seal_candidate_proposal,
)
from quant_rabbit.dojo_historical_job_archive import (
    ARCHIVE_RECEIPT_CONTRACT,
    DojoHistoricalJobArchiveError,
    archive_completed_historical_job,
    verify_existing_historical_job_archive,
)
from quant_rabbit.dojo_historical_raw_reclaim import (
    DojoHistoricalRawReclaimError,
    verify_existing_historical_job_raw_reclaim,
    verify_historical_job_raw_reclaim,
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
    RAPID_G2_ROOM_2025H1_PROFILE,
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
    return (
        isinstance(fixed.get("generation"), str)
        and bool(fixed.get("generation"))
        and fixed.get("generation") != "G2"
        and fixed.get("study_profile") == RAPID_G2_ROOM_2025H1_PROFILE
    )


def _generation_ordinal(fixed: Mapping[str, Any]) -> int:
    value = fixed.get("strategy_generation_ordinal", GENERATION_ORDINAL)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10_000:
        raise DojoHistoricalTrainControlError(
            "strategy generation ordinal is invalid"
        )
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
                "room_family_bindings_sha256": fixed[
                    "room_family_bindings_sha256"
                ],
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
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
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


def _file_binding(repo_root: Path, relative_paths: Sequence[str]) -> str:
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
    return canonical_sha256(rows)


def _verified_control(
    run_control_path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    control = _read_json(run_control_path, field="run control")
    if control.get("contract") != CONTROL_CONTRACT or control.get("schema_version") != 1:
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
        train_months = fixed.get("train_months")
        if not isinstance(train_months, list) or not train_months:
            raise DojoHistoricalTrainControlError(
                "room run-control train window is missing"
            )
        _generation_ordinal(fixed)
        _cost_profiles(control)
    if (
        not (legacy_generation or room_generation)
        or execution.get("evidence_tier") != "WORN_HISTORICAL_TRAIN_ONLY"
        or execution.get("max_parallel_jobs") != 1
        or execution.get("max_jobs_per_invocation") != 1
        or milestone.get("m5_completed_months_per_review") != 6
        or milestone.get("minimum_completed_intrabar_paths_per_month") != 2
        or milestone.get("rapid_train_evaluation_mode") != "INDEPENDENT_MONTH"
        or milestone.get("continuous_account_role")
        != "SEPARATE_LONG_PROFILE_NOT_RAPID_TUNING_GATE"
        or milestone.get("non_overlapping_six_month_blocks_required") is not True
        or milestone.get("partial_month_tuning_allowed") is not False
        or milestone.get("parameter_change_applies_only_to_new_generation") is not True
        or milestone.get("target_multiple_may_backsolve_sizing") is not False
    ):
        raise DojoHistoricalTrainControlError("run-control fixed boundary drifted")
    for key in ("registry_path", "source_manifest_path"):
        relative = fixed.get(key)
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
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
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
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
                    character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
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
        baseline = execution.get("capacity_baseline")
        required_baseline_keys = {
            "baseline_id",
            "coordinate_count",
            "raw_bytes_per_job",
            "source_bytes_per_job",
        }
        if not isinstance(baseline, Mapping) or set(baseline) != required_baseline_keys:
            raise DojoHistoricalTrainControlError(
                "room run-control capacity baseline schema is invalid"
            )
        if not isinstance(baseline["baseline_id"], str) or not baseline[
            "baseline_id"
        ]:
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
    minimum_free = execution.get("minimum_free_disk_bytes")
    if isinstance(minimum_free, bool) or not isinstance(minimum_free, int) or minimum_free < 20 * 1024**3:
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
        if (
            proposal["config_sha256"] != worker.get("config_sha256")
            or proposal["catalog_sha256"] != worker.get("catalog_sha256")
        ):
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
        if not isinstance(row["financing_policy"], str) or not row[
            "financing_policy"
        ]:
            raise DojoHistoricalTrainControlError(
                f"{scenario} financing policy is invalid"
            )
        values = {
            key: row[key]
            for key in keys
            if key != "financing_policy"
        }
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
            raise DojoHistoricalTrainControlError(
                f"STRESS cost {key} is below BASE"
            )
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
    values = {
        "base_cost_policy_sha256": canonical_sha256(costs["BASE"]),
        "m1_precision_policy_sha256": canonical_sha256(
            {
                "contexts": ["M1_CORE5_2020_2026H1", "M1_FULL28_2025_2026H1"],
                "context_overlap_is_not_replication": True,
                "promotion_from_m5_is_automatic": False,
            }
        ),
        "replay_engine_sha256": _file_binding(
            repo_root,
            (
                "src/quant_rabbit/dojo_long_horizon_economic_runner.py",
                "src/quant_rabbit/dojo_long_horizon_execution.py",
                "src/quant_rabbit/dojo_historical_train_control.py",
                "src/quant_rabbit/dojo_historical_job_archive.py",
                "src/quant_rabbit/dojo_historical_raw_reclaim.py",
                "src/quant_rabbit/dojo_long_horizon_plan.py",
                "src/quant_rabbit/dojo_long_horizon_schedule.py",
                "src/quant_rabbit/dojo_market_calendar.py",
                "src/quant_rabbit/dojo_portfolio_replay_reducer.py",
                "src/quant_rabbit/dojo_shared_worker_protocol.py",
                "src/quant_rabbit/dojo_sparse_replay.py",
                "src/quant_rabbit/dojo_sparse_source_slice_v2.py",
                "scripts/run-dojo-historical-raw-reclaim.py",
            ),
        ),
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
        "max_rss_bytes": int(
            control["execution"].get("max_rss_bytes", 8 * 1024**3)
        ),
        "max_open_files": int(control["execution"].get("max_open_files", 1024)),
        "min_free_disk_bytes": control["execution"]["minimum_free_disk_bytes"],
        "max_checkpoint_bytes": int(
            control["execution"].get(
                "max_checkpoint_bytes", 16 * 1024 * 1024
            )
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
        {
            key: row[key]
            for key in ("worker_id", "family_id", "config_sha256")
        }
        for row in runtime_seal["worker_catalog"]
    ]
    schedule = build_long_horizon_stream_schedule(
        plan, worker_bindings=worker_bindings
    )
    output_root = Path(control["execution"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {key: output_root / name for key, name in _ARTIFACT_NAMES.items()}
    implementation_manifest_body = {
        "contract": "QR_DOJO_IMPLEMENTATION_DIGEST_MANIFEST_V1",
        "schema_version": 1,
        "implementation_digests": implementation_digests,
        "implementation_digests_sha256": canonical_sha256(
            implementation_digests
        ),
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
    body = {
        "contract": MANIFEST_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_ONLY",
        "rapid_train_evaluation_mode": "INDEPENDENT_MONTH",
        "continuous_account_role": (
            "SEPARATE_LONG_PROFILE_NOT_RAPID_TUNING_GATE"
        ),
        "non_overlapping_six_month_blocks_required": True,
        "generation": fixed["generation"],
        "registry_artifact_sha256": registry_sha,
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "run_control_sha256": hashlib.sha256(Path(run_control_path).read_bytes()).hexdigest(),
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "runtime_binding_sha256": runtime_seal["runtime_binding_sha256"],
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
        "sealed_input_artifacts_sha256": canonical_sha256(
            sealed_input_artifacts
        ),
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
    *, repo_root: Path, run_control_path: Path
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo = Path(repo_root).resolve(strict=True)
    control = _verified_control(run_control_path, repo_root=repo)
    root = Path(control["execution"]["output_root"]).resolve(strict=True)
    manifest = _read_json(root / _ARTIFACT_NAMES["control_manifest"], field="control manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("contract") != MANIFEST_CONTRACT
        or manifest.get("manifest_sha256") != canonical_sha256(body)
    ):
        raise DojoHistoricalTrainControlError("generation control manifest drifted")
    if manifest.get("run_control_sha256") != hashlib.sha256(
        Path(run_control_path).read_bytes()
    ).hexdigest():
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
        raw = _stable_regular_bytes(
            path, field=f"sealed input {row['artifact_id']}"
        )
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
    implementation_manifest = sealed_inputs["IMPLEMENTATION_MANIFEST"]
    implementation_body = {
        key: value
        for key, value in implementation_manifest.items()
        if key != "implementation_manifest_sha256"
    }
    if (
        implementation_manifest.get("implementation_manifest_sha256")
        != canonical_sha256(implementation_body)
        or implementation_manifest.get("implementation_digests_sha256")
        != canonical_sha256(
            implementation_manifest.get("implementation_digests")
        )
    ):
        raise DojoHistoricalTrainControlError(
            "generation implementation manifest drifted"
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
        if hashlib.sha256(
            _stable_regular_bytes(path, field=f"generation {key}")
        ).hexdigest() != manifest["artifact_sha256"][key]:
            raise DojoHistoricalTrainControlError(f"generation {key} bytes drifted")
        artifacts[key] = _read_json(path, field=key)
    plan = validate_long_horizon_train_plan(artifacts["plan"])
    schedule = validate_long_horizon_stream_schedule(
        artifacts["schedule"], plan=plan
    )
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
            for pair, bit in zip(job["feed_pairs"], coordinate["trade_pair_mask"], strict=True)
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
                "max_open_and_pending_per_pair": risk[
                    "max_open_and_pending_per_pair"
                ],
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
    body = {key: item for key, item in value.items() if key != "economic_job_result_sha256"}
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
    review_months = 6
    manifest_path = root / _ARTIFACT_NAMES["control_manifest"]
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, field="control manifest")
        policy = manifest.get("trainer_milestone_policy")
        configured = (
            policy.get("m5_completed_months_per_review")
            if isinstance(policy, Mapping)
            else None
        )
        if (
            isinstance(configured, bool)
            or not isinstance(configured, int)
            or configured < 1
        ):
            raise DojoHistoricalTrainControlError(
                "trainer review cadence is invalid"
            )
        review_months = configured
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
    block_count = len(completed_months) // review_months
    body = {
        "contract": MILESTONE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_ONLY",
        "rapid_train_evaluation_mode": "INDEPENDENT_MONTH",
        "continuous_account_role": (
            "SEPARATE_LONG_PROFILE_NOT_RAPID_TUNING_GATE"
        ),
        "non_overlapping_six_month_blocks_required": True,
        "completed_job_count": completed_jobs,
        "completed_m5_month_count": len(completed_months),
        "completed_m5_months": completed_months,
        "complete_six_month_trainer_block_count": block_count,
        "trainer_review_month_count": review_months,
        "complete_trainer_block_count": block_count,
        "next_trainer_review_at_completed_m5_month_count": (
            block_count + 1
        )
        * review_months,
        "trainer_review_due": (
            len(completed_months) > 0
            and len(completed_months) % review_months == 0
        ),
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


def _archive_pending_completed_jobs(
    *, root: Path, control: Mapping[str, Any]
) -> list[dict[str, Any]]:
    destination = _archive_root(control)
    if destination is None:
        return []
    receipts_root = destination / "receipts"
    archived = []
    for completion_path in sorted((root / "jobs").glob("*/completion.json")):
        job_sha = completion_path.parent.name
        reclaimed = sorted(
            (root / "reclaim-receipts").glob(f"reclaim-{job_sha}-*.json")
        )
        if reclaimed:
            try:
                verify_existing_historical_job_raw_reclaim(
                    run_root=root,
                    archive_root=destination,
                    job_sha256=job_sha,
                    expected_drive_parent_id=control["execution"][
                        "archive_drive_readback_parent_id"
                    ],
                )
            except DojoHistoricalRawReclaimError as exc:
                raise DojoHistoricalTrainControlError(
                    f"completed job raw reclaim failed for {job_sha}: {exc}"
                ) from exc
            continue
        existing = sorted(receipts_root.glob(f"job-{job_sha}-*.json"))
        if existing:
            try:
                verify_existing_historical_job_archive(
                    run_root=root,
                    job_sha256=job_sha,
                    archive_root=destination,
                )
            except DojoHistoricalJobArchiveError as exc:
                raise DojoHistoricalTrainControlError(
                    f"existing completed job archive failed for {job_sha}: {exc}"
                ) from exc
            continue
        try:
            receipt = archive_completed_historical_job(
                run_root=root,
                job_sha256=job_sha,
                archive_root=destination,
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
    per_coordinate = math.ceil(evidence_bytes / int(coordinate_count))
    return int(source_bytes) + per_coordinate * planned_coordinate_count


def _estimated_next_job_bytes(
    *,
    control: Mapping[str, Any],
    archive_root: Path | None,
    planned_coordinate_count: int,
) -> tuple[int, int]:
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
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
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
        raise DojoHistoricalTrainControlError(
            "job-space headroom fraction is invalid"
        )
    raw_reserve = math.ceil(
        max(bootstrap, measured, normalized_baseline) * (1.0 + float(headroom))
    )
    staging = execution.get("archive_staging_fraction", 0.10)
    if (
        isinstance(staging, bool)
        or not isinstance(staging, (int, float))
        or not 0 <= float(staging) <= 0.5
    ):
        raise DojoHistoricalTrainControlError(
            "archive staging fraction is invalid"
        )
    return raw_reserve, raw_reserve + math.ceil(raw_reserve * float(staging))


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


def _disk_capacity_snapshot(
    *,
    root: Path,
    control: Mapping[str, Any],
    estimated_raw_bytes: int,
    estimated_peak_bytes: int,
) -> dict[str, Any]:
    floor = control["execution"]["minimum_free_disk_bytes"]
    run_probe = _nearest_existing_parent(root)
    run_free = shutil.disk_usage(run_probe).free
    archive_root = _archive_root(control)
    if archive_root is None:
        return {
            "shared_filesystem": None,
            "run_free_bytes": run_free,
            "run_required_bytes": floor + estimated_raw_bytes,
            "archive_free_bytes": None,
            "archive_required_bytes": 0,
        }
    archive_probe = _nearest_existing_parent(archive_root)
    archive_free = shutil.disk_usage(archive_probe).free
    shared = os.stat(run_probe).st_dev == os.stat(archive_probe).st_dev
    staging_bytes = estimated_peak_bytes - estimated_raw_bytes
    return {
        "shared_filesystem": shared,
        "run_free_bytes": run_free,
        "run_required_bytes": (
            floor + estimated_peak_bytes
            if shared
            else floor + estimated_raw_bytes
        ),
        "archive_free_bytes": archive_free,
        "archive_required_bytes": 0 if shared else staging_bytes,
    }


def _assert_disk_capacity(snapshot: Mapping[str, Any]) -> None:
    if snapshot["run_free_bytes"] < snapshot["run_required_bytes"]:
        raise DojoHistoricalTrainControlError(
            "run filesystem cannot cover the sealed floor and dynamic next-job "
            f"reservation: free={snapshot['run_free_bytes']}, "
            f"required={snapshot['run_required_bytes']}"
        )
    archive_free = snapshot.get("archive_free_bytes")
    archive_required = snapshot.get("archive_required_bytes")
    if (
        isinstance(archive_free, int)
        and isinstance(archive_required, int)
        and archive_free < archive_required
    ):
        raise DojoHistoricalTrainControlError(
            "archive filesystem cannot cover dynamic staging: "
            f"free={archive_free}, required={archive_required}"
        )


def _conflicting_generation_statuses(
    control: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for value in control["execution"].get("conflicting_execution_roots", []):
        root = Path(value)
        if not root.exists():
            rows.append(
                {
                    "output_root": str(root),
                    "exists": False,
                    "active_job_count": 0,
                    "terminal_job_count": 0,
                }
            )
            continue
        plan = validate_long_horizon_train_plan(
            _read_json(root / "plan.json", field="conflicting generation plan")
        )
        schedule = validate_long_horizon_stream_schedule(
            _read_json(
                root / "schedule.json", field="conflicting generation schedule"
            ),
            plan=plan,
        )
        status = long_horizon_execution_status(
            root / "execution-state", schedule=schedule, plan=plan
        )
        rows.append(
            {
                "output_root": str(root),
                "exists": True,
                "active_job_count": status["active_job_count"],
                "terminal_job_count": status["terminal_job_count"],
                "status_sha256": status["status_sha256"],
            }
        )
    return rows


def _assert_dynamic_machine_capacity(control: Mapping[str, Any]) -> None:
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
    conflicts = execution.get("conflicting_run_lock_paths", [])
    if not isinstance(conflicts, list) or not all(
        isinstance(value, str) and Path(value).is_absolute() for value in conflicts
    ):
        raise DojoHistoricalTrainControlError("conflicting run-lock list is invalid")
    active_conflicts = [
        row
        for row in _conflicting_generation_statuses(control)
        if row["active_job_count"] > 0
    ]
    if active_conflicts:
        raise DojoHistoricalTrainControlError(
            "a conflicting historical generation has an active or orphaned claim: "
            + ", ".join(row["output_root"] for row in active_conflicts)
        )


def _acquire_conflicting_run_locks(control: Mapping[str, Any]) -> list[int]:
    descriptors: list[int] = []
    try:
        for value in control["execution"].get("conflicting_run_lock_paths", []):
            path = Path(value)
            if not path.exists():
                continue
            descriptor = os.open(
                path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(descriptor)
                raise DojoHistoricalTrainControlError(
                    f"another heavy DOJO run owns the configured lock: {path}"
                ) from exc
            descriptors.append(descriptor)
        return descriptors
    except BaseException:
        for descriptor in reversed(descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        raise


def run_next_job(
    *,
    repo_root: Path,
    run_control_path: Path,
) -> dict[str, Any]:
    """Execute exactly one claimed historical job under the sealed G2 policy."""

    (
        control,
        root,
        plan,
        schedule,
        runtime_seal,
        source_manifest,
        catalog_wrapper,
    ) = _load_generation(repo_root=repo_root, run_control_path=run_control_path)
    lock_path = root / ".historical-train.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    global_lock_descriptor: int | None = None
    conflicting_lock_descriptors: list[int] = []
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DojoHistoricalTrainControlError(
                "another historical TRAIN job already owns the run lock"
            ) from exc
        global_lock_value = control["execution"].get("global_heavy_lock_path")
        if isinstance(global_lock_value, str):
            global_lock_path = Path(global_lock_value)
            global_lock_path.parent.mkdir(parents=True, exist_ok=True)
            global_lock_descriptor = os.open(
                global_lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                fcntl.flock(
                    global_lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError as exc:
                raise DojoHistoricalTrainControlError(
                    "another DOJO heavy operation owns the machine-wide lease"
                ) from exc
        conflicting_lock_descriptors = _acquire_conflicting_run_locks(control)
        _assert_dynamic_machine_capacity(control)
        recovered_archives = _archive_pending_completed_jobs(
            root=root, control=control
        )
        if recovered_archives:
            return {
                "status": "ARCHIVE_RECOVERED",
                "output_root": str(root),
                "archive": recovered_archives[0],
                "next_job_started": False,
                **_AUTHORITY,
            }
        planned_coordinate_count = max(
            job["coordinate_count"] for job in schedule["jobs"]
        )
        estimated_raw_bytes, estimated_peak_bytes = _estimated_next_job_bytes(
            control=control,
            archive_root=_archive_root(control),
            planned_coordinate_count=planned_coordinate_count,
        )
        _assert_disk_capacity(
            _disk_capacity_snapshot(
                root=root,
                control=control,
                estimated_raw_bytes=estimated_raw_bytes,
                estimated_peak_bytes=estimated_peak_bytes,
            )
        )
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
        job_dir.mkdir(parents=True, exist_ok=True)
        _write_once(job_dir / "runner-handoff.json", handoff)
        source_root = root / "source-slices"
        relative = Path(
            job["source_binding_id"],
            job["month"],
            f"{job['intrabar_path']}-{job_sha}.jsonl",
        )
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
                receipt = build_sparse_month_source_slice_receipt(
                    source_root=source_root,
                    relative_path=relative.as_posix(),
                    job=job,
                    source_manifest=source_manifest,
                )
            except (DojoLongHorizonEconomicRunnerError, SparseReplayError) as exc:
                return _seal_source_failure(
                    root=root,
                    schedule=schedule,
                    plan=plan,
                    handoff=handoff,
                    error=exc,
                )
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
        _write_once(
            job_dir / "coordinate-runtimes.json",
            {"coordinate_runtimes": runtimes},
        )
        carries = _carry_inputs(root, handoff)
        if carries:
            _write_once(
                job_dir / "carry-states-input.json",
                {"economic_carry_states_by_slot": carries},
            )
        evidence_root = job_dir / "economic-evidence"
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
                worker_runtime_binding_sha256=runtime_seal[
                    "runtime_binding_sha256"
                ],
                worker_runtime_seal=runtime_seal,
                worker_runtime_repo_root=repo,
                carry_states_by_slot=carries,
            )
            result = _verified_job_result(result, handoff=handoff)
            _write_once(result_path, result)
        _publish_carries(root, result)
        record_long_horizon_coordinate_results(
            root / "execution-state",
            schedule=schedule,
            plan=plan,
            claim_sha256=claim["claim_sha256"],
            results=result["coordinate_results"],
        )
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
        _write_once(job_dir / "completion.json", completion)
        archive_receipt = None
        archive_destination = _archive_root(control)
        if archive_destination is not None:
            try:
                archive_receipt = archive_completed_historical_job(
                    run_root=root,
                    job_sha256=job_sha,
                    archive_root=archive_destination,
                )
            except DojoHistoricalJobArchiveError as exc:
                raise DojoHistoricalTrainControlError(
                    f"completed job is sealed but its archive failed: {exc}"
                ) from exc
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
            "archive": archive_receipt,
            "trainer_milestone": milestone,
            **_AUTHORITY,
        }
    finally:
        for descriptor in reversed(conflicting_lock_descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        if global_lock_descriptor is not None:
            try:
                fcntl.flock(global_lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(global_lock_descriptor)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def generation_status(
    *, repo_root: Path, run_control_path: Path
) -> dict[str, Any]:
    """Return verified compact progress without opening partial economics."""

    control, root, plan, schedule, _, _, _ = _load_generation(
        repo_root=repo_root, run_control_path=run_control_path
    )
    execution = long_horizon_execution_status(
        root / "execution-state", schedule=schedule, plan=plan
    )
    milestone = _milestone_status(root, schedule, publish=False)
    archive_root = _archive_root(control)
    archive_receipts = []
    reclaimed_verifications = []
    remotely_verified_jobs: set[str] = set()
    reclaimable_bytes = 0
    if archive_root is not None and (archive_root / "receipts").is_dir():
        matched_receipt_paths: set[Path] = set()
        for completion_path in sorted((root / "jobs").glob("*/completion.json")):
            job_sha = completion_path.parent.name
            matches = sorted(
                (archive_root / "receipts").glob(f"job-{job_sha}-*.json")
            )
            if not matches:
                continue
            reclaimed = sorted(
                (root / "reclaim-receipts").glob(f"reclaim-{job_sha}-*.json")
            )
            if reclaimed:
                try:
                    verification = verify_existing_historical_job_raw_reclaim(
                        run_root=root,
                        archive_root=archive_root,
                        job_sha256=job_sha,
                        expected_drive_parent_id=control["execution"][
                            "archive_drive_readback_parent_id"
                        ],
                    )
                except DojoHistoricalRawReclaimError as exc:
                    raise DojoHistoricalTrainControlError(
                        "historical raw reclaim failed deep verification: "
                        f"{exc}"
                    ) from exc
                receipt = _read_json(
                    matches[0], field="reclaimed historical archive receipt"
                )
                reclaimed_verifications.append(verification)
                remotely_verified_jobs.add(job_sha)
            else:
                try:
                    receipt = verify_existing_historical_job_archive(
                        run_root=root,
                        job_sha256=job_sha,
                        archive_root=archive_root,
                    )
                except DojoHistoricalJobArchiveError as exc:
                    raise DojoHistoricalTrainControlError(
                        "historical archive receipt failed deep verification: "
                        f"{exc}"
                    ) from exc
                remote_paths = sorted(
                    (archive_root / "remote-receipts").glob(
                        f"remote-job-{job_sha}-{receipt['manifest_sha256']}-*.json"
                    )
                )
                if remote_paths:
                    if len(remote_paths) != 1:
                        raise DojoHistoricalTrainControlError(
                            "multiple remote readback receipts name one job"
                        )
                    try:
                        eligibility = verify_historical_job_raw_reclaim(
                            run_root=root,
                            archive_receipt_path=matches[0],
                            remote_receipt_path=remote_paths[0],
                            expected_drive_parent_id=control["execution"][
                                "archive_drive_readback_parent_id"
                            ],
                        )
                    except DojoHistoricalRawReclaimError as exc:
                        raise DojoHistoricalTrainControlError(
                            "remote readback receipt failed verification: "
                            f"{exc}"
                        ) from exc
                    remotely_verified_jobs.add(job_sha)
                    reclaimable_bytes += eligibility["plan"]["target_bytes"]
            archive_receipts.append(receipt)
            matched_receipt_paths.update(matches)
        all_receipt_paths = set(
            (archive_root / "receipts").glob("job-*.json")
        )
        if all_receipt_paths != matched_receipt_paths:
            raise DojoHistoricalTrainControlError(
                "historical archive receipt does not name a completed job"
            )
    completion_paths = list((root / "jobs").glob("*/completion.json"))
    completion_count = len(completion_paths)
    archived_job_count = len(archive_receipts)
    baseline = control["execution"].get("capacity_baseline", {})
    baseline_raw = (
        baseline.get("raw_bytes_per_job")
        if isinstance(baseline, Mapping)
        else None
    )
    planned_coordinate_count = max(
        job["coordinate_count"] for job in schedule["jobs"]
    )
    next_raw, next_peak = _estimated_next_job_bytes(
        control=control,
        archive_root=archive_root,
        planned_coordinate_count=planned_coordinate_count,
    )
    disk_capacity = _disk_capacity_snapshot(
        root=root,
        control=control,
        estimated_raw_bytes=next_raw,
        estimated_peak_bytes=next_peak,
    )
    capacity_comparison = {
        "baseline": dict(baseline) if isinstance(baseline, Mapping) else {},
        "planned_coordinate_count_per_job": max(
            job["coordinate_count"] for job in schedule["jobs"]
        ),
        "estimated_next_raw_bytes": next_raw,
        "estimated_next_peak_with_archive_staging_bytes": next_peak,
        "estimated_raw_to_baseline_ratio": (
            float(next_raw) / int(baseline_raw)
            if isinstance(next_raw, int)
            and isinstance(baseline_raw, int)
            and baseline_raw > 0
            else None
        ),
        "terminal_completion_count": completion_count,
        "locally_verified_archive_count": archived_job_count,
        "unarchived_terminal_count": max(
            0, completion_count - archived_job_count
        ),
        "raw_bytes_bound_by_archive_receipts": sum(
            row["total_source_bytes"] for row in archive_receipts
        ),
        "archive_bytes": sum(row["archive_size_bytes"] for row in archive_receipts),
        "remote_verified_archive_count": len(remotely_verified_jobs),
        "raw_reclaimed_job_count": len(reclaimed_verifications),
        "raw_reclaimed_logical_bytes": sum(
            row["reclaimed_logical_bytes"] for row in reclaimed_verifications
        ),
        "reclaimable_bytes": reclaimable_bytes,
        "raw_reclaim_requires_remote_readback": True,
    }
    return {
        "contract": "QR_DOJO_HISTORICAL_TRAIN_STATUS_V1",
        "schema_version": SCHEMA_VERSION,
        "output_root": str(root),
        "execution": execution,
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
            control
        ),
        "free_disk_bytes": shutil.disk_usage(root).free,
        **_AUTHORITY,
    }


__all__ = [
    "DojoHistoricalTrainControlError",
    "generation_status",
    "prepare_generation",
    "run_next_job",
]

"""Crash-safe, broker-free execution state for the DOJO long-horizon schedule.

This module is deliberately an orchestration boundary, not a replay engine or
an economic scorer.  It validates the sealed long-horizon stream schedule,
issues one content-addressed runner handoff per job, retains every COMPLETE or
FAILED coordinate receipt, enforces continuous-account predecessor carry
slots, and emits a typed reducer handoff.  It never imports a broker, opens
market data, executes strategy code, calls a model, or grants promotion/live
authority.

Filesystem mutations are append-only ``O_EXCL`` writes followed by file and
directory ``fsync``.  A crashed runner resumes the same durable claim and
fills only missing coordinate slots.  A terminal FAILED attempt is retained;
the next bounded attempt receives a new claim rather than rewriting history.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_long_horizon_plan import canonical_sha256
from quant_rabbit.dojo_long_horizon_schedule import (
    MAX_WORKERS,
    MAX_WORKERS_PER_FAMILY,
    DojoLongHorizonScheduleError,
    validate_long_horizon_stream_schedule,
)


MANIFEST_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_EXECUTION_MANIFEST_V1"
CLAIM_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_JOB_CLAIM_V1"
RUNNER_HANDOFF_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_RUNNER_HANDOFF_V1"
CELL_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_COORDINATE_RESULT_V1"
TERMINAL_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_TERMINAL_MANIFEST_V1"
CARRY_SLOT_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_CARRY_SLOT_V1"
REDUCER_HANDOFF_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_REDUCER_HANDOFF_V1"
STATE_STATUS_CONTRACT: Final = "QR_DOJO_LONG_HORIZON_EXECUTION_STATUS_V1"
SCHEMA_VERSION: Final = 1

# One immutable attempt makes a terminal failure absorbing.  Crash recovery
# resumes the same preterminal claim; it never receives a free new sample after
# observing an unfavorable cell.  This is an anti-selection evidence rule, not
# a trading, market, or sizing parameter.
MAX_ATTEMPTS_PER_JOB: Final = 1
# One canonical artifact is bounded to keep a malformed runner from exhausting
# local research storage.  Large trade/event evidence belongs in separately
# content-addressed compact evidence, not in this control-plane JSON.
MAX_CONTROL_ARTIFACT_BYTES: Final = 16 * 1024 * 1024
# Runner and failure identifiers are bounded provenance labels, not strategy
# parameters.  Longer prose belongs in a content-addressed evidence artifact.
MAX_IDENTIFIER_LENGTH: Final = 128

_ZERO_SHA256: Final = "0" * 64
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_FAILURE_CODE_RE: Final = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
# Four exact validated parent triples avoid rebuilding 32,112 coordinates for
# every cell checkpoint in one process while bounding resident control data.
# The key hashes every input byte-equivalent JSON value; this is a performance
# cache only and never relaxes validation after any content change.
_MAX_VALIDATION_CACHE_ENTRIES: Final = 4
_VALIDATED_MANIFEST_CACHE: OrderedDict[tuple[str, str, str], bytes] = OrderedDict()
_VALIDATED_MANIFEST_CACHE_LOCK = threading.RLock()


class DojoLongHorizonExecutionError(ValueError):
    """Execution state or a runner/reducer handoff is malformed."""


def _authority() -> dict[str, Any]:
    return {
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "diagnostic_only": True,
        "live_permission": False,
        "order_authority": "NONE",
        "portfolio_economics_computed_by_state_machine": False,
        "promotion_eligible": False,
    }


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
        raise DojoLongHorizonExecutionError(
            "artifact is not strict canonical JSON"
        ) from exc


def _exact(value: Any, keys: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise DojoLongHorizonExecutionError(
            f"{field} must contain exactly: {','.join(sorted(keys))}"
        )
    if any(not isinstance(key, str) for key in value):
        raise DojoLongHorizonExecutionError(f"{field} keys must be strings")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DojoLongHorizonExecutionError(f"{field} must be a sequence")
    return value


def _sha(value: Any, *, field: str, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if (
        not isinstance(value, str)
        or _SHA256_RE.fullmatch(value) is None
        or value == _ZERO_SHA256
    ):
        raise DojoLongHorizonExecutionError(
            f"{field} must be a non-zero lowercase SHA-256"
        )
    return value


def _identifier(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > MAX_IDENTIFIER_LENGTH
        or _IDENTIFIER_RE.fullmatch(value) is None
    ):
        raise DojoLongHorizonExecutionError(f"{field} is invalid")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoLongHorizonExecutionError(f"{field} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    nullable: bool = False,
) -> float | int | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoLongHorizonExecutionError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise DojoLongHorizonExecutionError(f"{field} is outside its finite range")
    return value


def _nullable_integer(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field=field)


def _verify_authority(value: Any, *, field: str) -> None:
    if dict(_exact(value, set(_authority()), field=field)) != _authority():
        raise DojoLongHorizonExecutionError(f"{field} is not fail-closed")


def _runner_binding(value: Any) -> dict[str, str]:
    row = _exact(
        value,
        {"runner_contract", "runner_code_sha256", "result_contract"},
        field="runner_binding",
    )
    result = {
        "runner_contract": _identifier(
            row["runner_contract"], field="runner_binding.runner_contract"
        ),
        "runner_code_sha256": str(
            _sha(
                row["runner_code_sha256"],
                field="runner_binding.runner_code_sha256",
            )
        ),
        "result_contract": _identifier(
            row["result_contract"], field="runner_binding.result_contract"
        ),
    }
    if result["result_contract"] != CELL_CONTRACT:
        raise DojoLongHorizonExecutionError(
            f"runner result_contract must be {CELL_CONTRACT}"
        )
    return result


def _resource_policy(value: Any, *, result_coordinate_count: int) -> dict[str, int]:
    row = _exact(
        value,
        {
            "max_resident_coordinates",
            "max_rss_bytes",
            "max_open_files",
            "min_free_disk_bytes",
            "max_checkpoint_bytes",
            "max_terminal_bytes",
            "max_parallel_jobs",
        },
        field="resource_policy",
    )
    result = {
        field: _integer(row[field], field=f"resource_policy.{field}", minimum=1)
        for field in row
    }
    if result["max_resident_coordinates"] > result_coordinate_count:
        raise DojoLongHorizonExecutionError(
            "max_resident_coordinates exceeds the sealed denominator"
        )
    if result["max_terminal_bytes"] > MAX_CONTROL_ARTIFACT_BYTES:
        raise DojoLongHorizonExecutionError(
            "max_terminal_bytes exceeds the control artifact ceiling"
        )
    if result["max_parallel_jobs"] > MAX_WORKERS:
        raise DojoLongHorizonExecutionError(
            "max_parallel_jobs exceeds the global worker ceiling"
        )
    return result


def _jobs_by_sha(schedule: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in _sequence(schedule.get("jobs"), field="schedule.jobs"):
        if not isinstance(raw, Mapping):
            raise DojoLongHorizonExecutionError("schedule job must be an object")
        job_sha = str(_sha(raw.get("job_sha256"), field="schedule.job_sha256"))
        if job_sha in result:
            raise DojoLongHorizonExecutionError(
                "schedule job SHA values are not unique"
            )
        result[job_sha] = raw
    return result


def _coordinate_map(job: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in _sequence(job.get("coordinates"), field="job.coordinates"):
        if not isinstance(raw, Mapping):
            raise DojoLongHorizonExecutionError("coordinate must be an object")
        coordinate_id = str(
            _sha(raw.get("coordinate_id"), field="coordinate.coordinate_id")
        )
        if coordinate_id in result:
            raise DojoLongHorizonExecutionError(
                "job coordinate identifiers are not unique"
            )
        result[coordinate_id] = raw
    if len(result) != job.get("coordinate_count"):
        raise DojoLongHorizonExecutionError("job coordinate count is inconsistent")
    return result


def _slot_ids(job: Mapping[str, Any], *, key: str, omit_null: bool = True) -> list[str]:
    values: list[str] = []
    for coordinate in _coordinate_map(job).values():
        value = coordinate.get(key)
        if value is None and omit_null:
            continue
        values.append(str(_sha(value, field=f"coordinate.{key}")))
    if len(values) != len(set(values)):
        raise DojoLongHorizonExecutionError(f"duplicate {key} in one job")
    return sorted(values)


def build_long_horizon_execution_manifest(
    schedule: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    runner_binding: Mapping[str, Any],
    resource_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the compact immutable job/state-machine manifest."""

    try:
        sealed_schedule = validate_long_horizon_stream_schedule(schedule, plan=plan)
    except DojoLongHorizonScheduleError as exc:
        raise DojoLongHorizonExecutionError(
            "long-horizon stream schedule is invalid"
        ) from exc
    runner = _runner_binding(runner_binding)
    resources = _resource_policy(
        resource_policy,
        result_coordinate_count=sealed_schedule["result_coordinate_count"],
    )
    worker_bindings = sealed_schedule["worker_set"]["bindings"]
    family_counts: dict[str, int] = {}
    for binding in worker_bindings:
        family = binding["family_id"]
        family_counts[family] = family_counts.get(family, 0) + 1
    if len(worker_bindings) > MAX_WORKERS:
        raise DojoLongHorizonExecutionError("worker count exceeds global limit")
    if any(count > MAX_WORKERS_PER_FAMILY for count in family_counts.values()):
        raise DojoLongHorizonExecutionError("worker count exceeds family limit")

    jobs: list[dict[str, Any]] = []
    for ordinal, job in enumerate(sealed_schedule["jobs"]):
        required = _slot_ids(job, key="predecessor_state_slot_id")
        produced = _slot_ids(job, key="carry_out_state_slot_id")
        coordinate_ids = [row["coordinate_id"] for row in job["coordinates"]]
        body = {
            "job_ordinal": ordinal,
            "job_sha256": job["job_sha256"],
            "phase": job["phase"],
            "source_binding_id": job["source_binding_id"],
            "month": job["month"],
            "intrabar_path": job["intrabar_path"],
            "coordinate_count": job["coordinate_count"],
            "coordinate_ids_sha256": canonical_sha256(coordinate_ids),
            "required_predecessor_state_slot_count": len(required),
            "required_predecessor_state_slot_ids": required,
            "required_predecessor_state_slot_ids_sha256": canonical_sha256(required),
            "produced_carry_out_state_slot_count": len(produced),
            "produced_carry_out_state_slot_ids": produced,
            "produced_carry_out_state_slot_ids_sha256": canonical_sha256(produced),
        }
        jobs.append({**body, "execution_job_sha256": canonical_sha256(body)})
    body = {
        "contract": MANIFEST_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": sealed_schedule["classification"],
        "schedule_sha256": sealed_schedule["schedule_sha256"],
        "plan_sha256": sealed_schedule["plan_sha256"],
        "worker_set_sha256": sealed_schedule["worker_set"]["worker_set_sha256"],
        "runner_binding": runner,
        "resource_limits": {
            "max_workers_global": MAX_WORKERS,
            "max_workers_per_family": MAX_WORKERS_PER_FAMILY,
            "worker_count": len(worker_bindings),
            "worker_count_by_family": dict(sorted(family_counts.items())),
            "max_attempts_per_job": MAX_ATTEMPTS_PER_JOB,
            **resources,
        },
        "resource_enforcement": {
            "runner_must_measure_limits": True,
            "state_machine_observes_process_metrics": False,
            "resource_limit_action": "PAUSE_AND_RESUME_SAME_CLAIM",
            "resource_limit_may_create_failed_terminal": False,
            "resource_limit_may_create_new_claim": False,
            "unbounded_execution_allowed": False,
        },
        "state_machine": {
            "claim_write_policy": "O_EXCL_FSYNC_APPEND_ONLY",
            "cell_write_policy": "O_EXCL_FSYNC_APPEND_ONLY",
            "failed_cells_retained": True,
            "complete_cells_retained": True,
            "failed_attempts_retained": True,
            "terminal_failure_absorbing": True,
            "resume_policy": "SAME_CLAIM_FILL_MISSING_COORDINATES_ONLY",
            "claim_next_order_policy": "SEALED_SCHEDULE_ORDER_ONLY",
            "continuous_predecessor_policy": (
                "COORDINATE_SCOPED_CARRY_OR_PREDECESSOR_FAILED"
            ),
            "missing_cell_policy": "COUNT_IN_DENOMINATOR_AND_FAIL_CLOSED",
            "live_or_broker_execution_supported": False,
        },
        "job_count": len(jobs),
        "result_coordinate_count": sealed_schedule["result_coordinate_count"],
        "jobs": jobs,
        "all_execution_job_ids_sha256": canonical_sha256(
            [row["execution_job_sha256"] for row in jobs]
        ),
        "authority": _authority(),
    }
    return {**body, "execution_manifest_sha256": canonical_sha256(body)}


def validate_long_horizon_execution_manifest(
    manifest: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild a manifest and reject self-rehashed semantic drift."""

    row = _exact(
        manifest,
        {
            "contract",
            "schema_version",
            "classification",
            "schedule_sha256",
            "plan_sha256",
            "worker_set_sha256",
            "runner_binding",
            "resource_limits",
            "resource_enforcement",
            "state_machine",
            "job_count",
            "result_coordinate_count",
            "jobs",
            "all_execution_job_ids_sha256",
            "authority",
            "execution_manifest_sha256",
        },
        field="execution_manifest",
    )
    _verify_authority(row["authority"], field="execution_manifest.authority")
    cache_key = (
        canonical_sha256(plan),
        canonical_sha256(schedule),
        canonical_sha256(dict(row)),
    )
    with _VALIDATED_MANIFEST_CACHE_LOCK:
        cached = _VALIDATED_MANIFEST_CACHE.get(cache_key)
        if cached is not None:
            _VALIDATED_MANIFEST_CACHE.move_to_end(cache_key)
            loaded = json.loads(cached.decode("utf-8"))
            if not isinstance(loaded, dict):  # pragma: no cover - internal invariant
                raise DojoLongHorizonExecutionError("validation cache is malformed")
            return loaded
    rebuilt = build_long_horizon_execution_manifest(
        schedule,
        plan=plan,
        runner_binding=_runner_binding(row["runner_binding"]),
        resource_policy={
            key: row["resource_limits"][key]
            for key in (
                "max_resident_coordinates",
                "max_rss_bytes",
                "max_open_files",
                "min_free_disk_bytes",
                "max_checkpoint_bytes",
                "max_terminal_bytes",
                "max_parallel_jobs",
            )
        },
    )
    if dict(row) != rebuilt:
        raise DojoLongHorizonExecutionError("execution manifest drifted")
    with _VALIDATED_MANIFEST_CACHE_LOCK:
        _VALIDATED_MANIFEST_CACHE[cache_key] = _canonical_bytes(rebuilt)
        _VALIDATED_MANIFEST_CACHE.move_to_end(cache_key)
        while len(_VALIDATED_MANIFEST_CACHE) > _MAX_VALIDATION_CACHE_ENTRIES:
            _VALIDATED_MANIFEST_CACHE.popitem(last=False)
    return rebuilt


def _claim_body(
    *,
    manifest: Mapping[str, Any],
    job_sha256: str,
    required_predecessor_slot_ids: Sequence[str],
    attempt_ordinal: int,
    runner_id: str,
) -> dict[str, Any]:
    body = {
        "contract": CLAIM_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "job_sha256": job_sha256,
        "attempt_ordinal": attempt_ordinal,
        "runner_id": runner_id,
        "required_predecessor_state_slot_count": len(required_predecessor_slot_ids),
        "required_predecessor_state_slot_ids_sha256": canonical_sha256(
            list(required_predecessor_slot_ids)
        ),
        "authority": _authority(),
    }
    return {**body, "claim_sha256": canonical_sha256(body)}


def _validate_claim(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "contract",
            "schema_version",
            "execution_manifest_sha256",
            "schedule_sha256",
            "job_sha256",
            "attempt_ordinal",
            "runner_id",
            "required_predecessor_state_slot_count",
            "required_predecessor_state_slot_ids_sha256",
            "authority",
            "claim_sha256",
        },
        field="claim",
    )
    if row["contract"] != CLAIM_CONTRACT or row["schema_version"] != SCHEMA_VERSION:
        raise DojoLongHorizonExecutionError("claim contract is unsupported")
    if (
        row["execution_manifest_sha256"] != manifest["execution_manifest_sha256"]
        or row["schedule_sha256"] != manifest["schedule_sha256"]
        or row["job_sha256"] != job["job_sha256"]
    ):
        raise DojoLongHorizonExecutionError("claim parent binding drifted")
    attempt = _integer(row["attempt_ordinal"], field="claim.attempt_ordinal", minimum=1)
    if attempt > MAX_ATTEMPTS_PER_JOB:
        raise DojoLongHorizonExecutionError("claim attempt exceeds its bounded limit")
    runner_id = _identifier(row["runner_id"], field="claim.runner_id")
    expected_slots = _slot_ids(job, key="predecessor_state_slot_id")
    if row["required_predecessor_state_slot_count"] != len(expected_slots) or row[
        "required_predecessor_state_slot_ids_sha256"
    ] != canonical_sha256(expected_slots):
        raise DojoLongHorizonExecutionError(
            "claim does not bind the exact predecessor slot denominator"
        )
    expected = _claim_body(
        manifest=manifest,
        job_sha256=job["job_sha256"],
        required_predecessor_slot_ids=expected_slots,
        attempt_ordinal=attempt,
        runner_id=runner_id,
    )
    if dict(row) != expected:
        raise DojoLongHorizonExecutionError("claim content or digest drifted")
    return expected


_CELL_KEYS = {
    "contract",
    "schema_version",
    "job_sha256",
    "coordinate_id",
    "coordinate_digest_sha256",
    "claim_sha256",
    "status",
    "starting_balance_jpy",
    "starting_equity_jpy",
    "ending_balance_jpy",
    "ending_equity_jpy",
    "minimum_mtm_equity_jpy",
    "minimum_free_margin_jpy",
    "max_mtm_drawdown_fraction",
    "peak_margin_usage_fraction",
    "margin_closeout_count",
    "ruin_event_count",
    "trade_count",
    "fill_count",
    "margin_reject_count",
    "financing_jpy",
    "transaction_cost_jpy",
    "source_slice_receipt_sha256",
    "batch_chain_sha256",
    "compact_evidence_sha256",
    "quote_coverage_complete",
    "active_worker_ack_complete",
    "predecessor_state_slot_id",
    "predecessor_state_sha256",
    "carry_out_state_slot_id",
    "carry_out_state_sha256",
    "failure",
    "authority",
    "cell_sha256",
}


def validate_long_horizon_coordinate_result(
    value: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact COMPLETE/FAILED runner result payload."""

    row = _exact(value, _CELL_KEYS, field="coordinate_result")
    if row["contract"] != CELL_CONTRACT or row["schema_version"] != SCHEMA_VERSION:
        raise DojoLongHorizonExecutionError("coordinate result contract is unsupported")
    if row["job_sha256"] != job["job_sha256"]:
        raise DojoLongHorizonExecutionError("coordinate result job binding drifted")
    if row["claim_sha256"] != claim["claim_sha256"]:
        raise DojoLongHorizonExecutionError("coordinate result claim binding drifted")
    coordinate_id = str(
        _sha(row["coordinate_id"], field="coordinate_result.coordinate_id")
    )
    coordinates = _coordinate_map(job)
    if coordinate_id not in coordinates:
        raise DojoLongHorizonExecutionError("coordinate is absent from the sealed job")
    if row["coordinate_digest_sha256"] != coordinate_id:
        raise DojoLongHorizonExecutionError(
            "coordinate digest must equal the sealed coordinate identity"
        )
    coordinate = coordinates[coordinate_id]
    if row["predecessor_state_slot_id"] != coordinate["predecessor_state_slot_id"]:
        raise DojoLongHorizonExecutionError("predecessor state slot drifted")
    if row["carry_out_state_slot_id"] != coordinate["carry_out_state_slot_id"]:
        raise DojoLongHorizonExecutionError("carry-out state slot drifted")
    _verify_authority(row["authority"], field="coordinate_result.authority")
    status = row["status"]
    numeric_fields = (
        "starting_balance_jpy",
        "starting_equity_jpy",
        "ending_balance_jpy",
        "ending_equity_jpy",
        "minimum_mtm_equity_jpy",
        "minimum_free_margin_jpy",
        "max_mtm_drawdown_fraction",
        "peak_margin_usage_fraction",
        "financing_jpy",
        "transaction_cost_jpy",
    )
    integer_fields = (
        "margin_closeout_count",
        "ruin_event_count",
        "trade_count",
        "fill_count",
        "margin_reject_count",
    )
    evidence_fields = (
        "source_slice_receipt_sha256",
        "batch_chain_sha256",
        "compact_evidence_sha256",
    )
    predecessor_required = coordinate["predecessor_state_slot_id"] is not None
    carry_required = coordinate["carry_out_state_slot_id"] is not None
    if status == "COMPLETE":
        for field in numeric_fields:
            minimum = (
                0.0
                if field
                in {
                    "max_mtm_drawdown_fraction",
                    "peak_margin_usage_fraction",
                    "transaction_cost_jpy",
                }
                else None
            )
            _number(row[field], field=f"coordinate_result.{field}", minimum=minimum)
        for field in integer_fields:
            _integer(row[field], field=f"coordinate_result.{field}")
        for field in evidence_fields:
            _sha(row[field], field=f"coordinate_result.{field}")
        if (
            row["quote_coverage_complete"] is not True
            or row["active_worker_ack_complete"] is not True
        ):
            raise DojoLongHorizonExecutionError(
                "COMPLETE result requires full quote and active-worker acknowledgements"
            )
        if float(row["minimum_mtm_equity_jpy"]) > min(
            float(row["starting_equity_jpy"]), float(row["ending_equity_jpy"])
        ):
            raise DojoLongHorizonExecutionError(
                "minimum MTM equity exceeds an observed endpoint equity"
            )
        if (float(row["minimum_mtm_equity_jpy"]) <= 0.0) != (
            int(row["ruin_event_count"]) > 0
        ):
            raise DojoLongHorizonExecutionError(
                "minimum MTM equity and ruin event count disagree"
            )
        if row["failure"] is not None:
            raise DojoLongHorizonExecutionError("COMPLETE result cannot have failure")
        if predecessor_required:
            _sha(
                row["predecessor_state_sha256"],
                field="coordinate_result.predecessor_state_sha256",
            )
        elif row["predecessor_state_sha256"] is not None:
            raise DojoLongHorizonExecutionError(
                "first/independent coordinate cannot invent predecessor state"
            )
        if carry_required:
            _sha(
                row["carry_out_state_sha256"],
                field="coordinate_result.carry_out_state_sha256",
            )
        elif row["carry_out_state_sha256"] is not None:
            raise DojoLongHorizonExecutionError(
                "independent coordinate cannot invent carry-out state"
            )
    elif status == "FAILED":
        nullable = [
            *numeric_fields,
            *integer_fields,
            *evidence_fields,
            "predecessor_state_sha256",
            "carry_out_state_sha256",
            "quote_coverage_complete",
            "active_worker_ack_complete",
        ]
        if any(row[field] is not None for field in nullable):
            raise DojoLongHorizonExecutionError(
                "FAILED result must not masquerade as partial/zero economics"
            )
        failure = _exact(
            row["failure"],
            {"code", "retryable", "evidence_sha256"},
            field="coordinate_result.failure",
        )
        if (
            not isinstance(failure["code"], str)
            or _FAILURE_CODE_RE.fullmatch(failure["code"]) is None
            or not isinstance(failure["retryable"], bool)
        ):
            raise DojoLongHorizonExecutionError("coordinate failure is malformed")
        _sha(failure["evidence_sha256"], field="coordinate failure evidence")
    else:
        raise DojoLongHorizonExecutionError(
            "coordinate result status must be COMPLETE or FAILED"
        )
    body = {key: row[key] for key in _CELL_KEYS if key != "cell_sha256"}
    expected_sha = canonical_sha256(body)
    if row["cell_sha256"] != expected_sha:
        raise DojoLongHorizonExecutionError("coordinate result digest drifted")
    return {**body, "cell_sha256": expected_sha}


def build_long_horizon_coordinate_result(
    *,
    job: Mapping[str, Any],
    claim: Mapping[str, Any],
    coordinate_id: str,
    status: str,
    starting_balance_jpy: float | int | None = None,
    starting_equity_jpy: float | int | None = None,
    ending_balance_jpy: float | int | None = None,
    ending_equity_jpy: float | int | None = None,
    minimum_mtm_equity_jpy: float | int | None = None,
    minimum_free_margin_jpy: float | int | None = None,
    max_mtm_drawdown_fraction: float | int | None = None,
    peak_margin_usage_fraction: float | int | None = None,
    margin_closeout_count: int | None = None,
    ruin_event_count: int | None = None,
    trade_count: int | None = None,
    fill_count: int | None = None,
    margin_reject_count: int | None = None,
    financing_jpy: float | int | None = None,
    transaction_cost_jpy: float | int | None = None,
    source_slice_receipt_sha256: str | None = None,
    batch_chain_sha256: str | None = None,
    compact_evidence_sha256: str | None = None,
    quote_coverage_complete: bool | None = None,
    active_worker_ack_complete: bool | None = None,
    predecessor_state_sha256: str | None = None,
    carry_out_state_sha256: str | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a strict cell payload for an external broker-free runner."""

    coordinate = _coordinate_map(job).get(coordinate_id)
    if coordinate is None:
        raise DojoLongHorizonExecutionError("coordinate is absent from the sealed job")
    body = {
        "contract": CELL_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "job_sha256": job["job_sha256"],
        "coordinate_id": coordinate_id,
        "coordinate_digest_sha256": coordinate_id,
        "claim_sha256": claim["claim_sha256"],
        "status": status,
        "starting_balance_jpy": starting_balance_jpy,
        "starting_equity_jpy": starting_equity_jpy,
        "ending_balance_jpy": ending_balance_jpy,
        "ending_equity_jpy": ending_equity_jpy,
        "minimum_mtm_equity_jpy": minimum_mtm_equity_jpy,
        "minimum_free_margin_jpy": minimum_free_margin_jpy,
        "max_mtm_drawdown_fraction": max_mtm_drawdown_fraction,
        "peak_margin_usage_fraction": peak_margin_usage_fraction,
        "margin_closeout_count": margin_closeout_count,
        "ruin_event_count": ruin_event_count,
        "trade_count": trade_count,
        "fill_count": fill_count,
        "margin_reject_count": margin_reject_count,
        "financing_jpy": financing_jpy,
        "transaction_cost_jpy": transaction_cost_jpy,
        "source_slice_receipt_sha256": source_slice_receipt_sha256,
        "batch_chain_sha256": batch_chain_sha256,
        "compact_evidence_sha256": compact_evidence_sha256,
        "quote_coverage_complete": quote_coverage_complete,
        "active_worker_ack_complete": active_worker_ack_complete,
        "predecessor_state_slot_id": coordinate["predecessor_state_slot_id"],
        "predecessor_state_sha256": predecessor_state_sha256,
        "carry_out_state_slot_id": coordinate["carry_out_state_slot_id"],
        "carry_out_state_sha256": carry_out_state_sha256,
        "failure": dict(failure) if failure is not None else None,
        "authority": _authority(),
    }
    candidate = {**body, "cell_sha256": canonical_sha256(body)}
    return validate_long_horizon_coordinate_result(candidate, job=job, claim=claim)


def _terminal_body(
    *,
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    claim: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    coordinate_by_id = _coordinate_map(job)
    coordinate_ids = list(coordinate_by_id)
    if [row["coordinate_id"] for row in cells] != coordinate_ids:
        raise DojoLongHorizonExecutionError(
            "terminal cells must equal the full sealed coordinate order"
        )
    complete = sum(row["status"] == "COMPLETE" for row in cells)
    failed = len(cells) - complete
    failure_ids = [row["coordinate_id"] for row in cells if row["status"] == "FAILED"]
    carry_slots = [
        row["carry_out_state_slot_id"]
        for row in cells
        if row["status"] == "COMPLETE" and row["carry_out_state_slot_id"] is not None
    ]
    return {
        "contract": TERMINAL_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "job_sha256": job["job_sha256"],
        "claim": dict(claim),
        "attempt_ordinal": claim["attempt_ordinal"],
        "terminal_status": "COMPLETE" if failed == 0 else "FAILED",
        "coordinate_count": len(cells),
        "complete_coordinate_count": complete,
        "failed_coordinate_count": failed,
        "failed_coordinate_ids": failure_ids,
        "failed_coordinate_ids_sha256": canonical_sha256(failure_ids),
        "cells": [dict(row) for row in cells],
        "cell_sha256_values_sha256": canonical_sha256(
            [row["cell_sha256"] for row in cells]
        ),
        "produced_carry_out_state_slot_count": len(carry_slots),
        "produced_carry_out_state_slot_ids_sha256": canonical_sha256(carry_slots),
        "missing_cell_policy": "COUNT_IN_DENOMINATOR_AND_FAIL_CLOSED",
        "reducer_required": True,
        "authority": _authority(),
    }


def _validate_terminal_prevalidated(
    value: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "contract",
            "schema_version",
            "execution_manifest_sha256",
            "schedule_sha256",
            "job_sha256",
            "claim",
            "attempt_ordinal",
            "terminal_status",
            "coordinate_count",
            "complete_coordinate_count",
            "failed_coordinate_count",
            "failed_coordinate_ids",
            "failed_coordinate_ids_sha256",
            "cells",
            "cell_sha256_values_sha256",
            "produced_carry_out_state_slot_count",
            "produced_carry_out_state_slot_ids_sha256",
            "missing_cell_policy",
            "reducer_required",
            "authority",
            "terminal_sha256",
        },
        field="terminal_manifest",
    )
    if row["contract"] != TERMINAL_CONTRACT or row["schema_version"] != SCHEMA_VERSION:
        raise DojoLongHorizonExecutionError("terminal manifest contract is unsupported")
    jobs = _jobs_by_sha(schedule)
    job_sha = str(_sha(row["job_sha256"], field="terminal_manifest.job_sha256"))
    if job_sha not in jobs:
        raise DojoLongHorizonExecutionError("terminal job is absent from schedule")
    job = jobs[job_sha]
    claim = _validate_claim(row["claim"], manifest=manifest, job=job)
    raw_cells = _sequence(row["cells"], field="terminal_manifest.cells")
    cells = [
        validate_long_horizon_coordinate_result(item, job=job, claim=claim)
        for item in raw_cells
    ]
    body = _terminal_body(manifest=manifest, job=job, claim=claim, cells=cells)
    expected = {**body, "terminal_sha256": canonical_sha256(body)}
    if dict(row) != expected:
        raise DojoLongHorizonExecutionError("terminal manifest content drifted")
    return expected


def validate_long_horizon_terminal_manifest(
    value: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Public pure validator for one scorer-facing terminal manifest."""

    manifest = validate_long_horizon_execution_manifest(
        execution_manifest, schedule=schedule, plan=plan
    )
    return _validate_terminal_prevalidated(value, schedule=schedule, manifest=manifest)


def validate_long_horizon_terminal_manifests(
    values: Sequence[Mapping[str, Any]],
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the exact all-job terminal denominator with one parent rebuild."""

    manifest = validate_long_horizon_execution_manifest(
        execution_manifest, schedule=schedule, plan=plan
    )
    rows = _sequence(values, field="terminal_manifests")
    if len(rows) != manifest["job_count"]:
        raise DojoLongHorizonExecutionError(
            "terminal manifests do not cover the exact job denominator"
        )
    verified = [
        _validate_terminal_prevalidated(value, schedule=schedule, manifest=manifest)
        for value in rows
    ]
    expected_job_order = [job["job_sha256"] for job in schedule["jobs"]]
    actual_job_order = [terminal["job_sha256"] for terminal in verified]
    if actual_job_order != expected_job_order:
        raise DojoLongHorizonExecutionError(
            "terminal manifests must follow the complete sealed schedule order"
        )
    _validate_terminal_continuity(verified)
    return verified


def _validate_terminal_continuity(
    terminals: Sequence[Mapping[str, Any]],
) -> None:
    producer_by_slot: dict[str, Mapping[str, Any]] = {}
    for terminal in terminals:
        for cell in terminal["cells"]:
            slot = cell["carry_out_state_slot_id"]
            if slot is None:
                continue
            if slot in producer_by_slot:
                raise DojoLongHorizonExecutionError(
                    "terminal denominator has duplicate carry producers"
                )
            producer_by_slot[slot] = cell
    for terminal in terminals:
        for cell in terminal["cells"]:
            slot = cell["predecessor_state_slot_id"]
            if slot is None:
                continue
            producer = producer_by_slot.get(slot)
            if producer is None:
                raise DojoLongHorizonExecutionError(
                    "terminal predecessor has no exact producer cell"
                )
            if producer["status"] == "FAILED":
                if (
                    cell["status"] != "FAILED"
                    or cell["failure"]["code"] != "PREDECESSOR_FAILED"
                    or cell["failure"]["evidence_sha256"] != producer["cell_sha256"]
                ):
                    raise DojoLongHorizonExecutionError(
                        "failed continuous state did not propagate exactly"
                    )
            elif cell["status"] == "COMPLETE":
                if (
                    cell["predecessor_state_sha256"]
                    != producer["carry_out_state_sha256"]
                ):
                    raise DojoLongHorizonExecutionError(
                        "continuous carry state hash changed across months"
                    )
            elif cell["failure"]["code"] == "PREDECESSOR_FAILED":
                raise DojoLongHorizonExecutionError(
                    "PREDECESSOR_FAILED cannot follow a complete producer cell"
                )


def _build_reducer_handoff(
    terminal: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "contract": REDUCER_HANDOFF_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "job_sha256": terminal["job_sha256"],
        "terminal_sha256": terminal["terminal_sha256"],
        "terminal_status": terminal["terminal_status"],
        "coordinate_count": terminal["coordinate_count"],
        "complete_coordinate_count": terminal["complete_coordinate_count"],
        "failed_coordinate_count": terminal["failed_coordinate_count"],
        "cell_sha256_values_sha256": terminal["cell_sha256_values_sha256"],
        "reducer_action": (
            "REDUCE_EXACT_FIXED_DENOMINATOR"
            if terminal["terminal_status"] == "COMPLETE"
            else "RETAIN_FAILURE_AND_FAIL_CLOSED"
        ),
        "portfolio_economics_computed": False,
        "authority": _authority(),
    }
    return {**body, "reducer_handoff_sha256": canonical_sha256(body)}


def validate_long_horizon_reducer_handoff(
    value: Mapping[str, Any],
    *,
    terminal: Mapping[str, Any],
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the typed terminal-to-reducer boundary."""

    manifest = validate_long_horizon_execution_manifest(
        execution_manifest, schedule=schedule, plan=plan
    )
    verified_terminal = _validate_terminal_prevalidated(
        terminal, schedule=schedule, manifest=manifest
    )
    expected = _build_reducer_handoff(verified_terminal, manifest=manifest)
    if dict(value) != expected:
        raise DojoLongHorizonExecutionError("reducer handoff content drifted")
    return expected


def validate_long_horizon_reducer_handoffs(
    values: Sequence[Mapping[str, Any]],
    *,
    terminals: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate exact all-job reducer handoffs without quadratic parent rebuilds."""

    manifest = validate_long_horizon_execution_manifest(
        execution_manifest, schedule=schedule, plan=plan
    )
    verified_terminals = _sequence(terminals, field="terminal_manifests")
    if len(verified_terminals) != manifest["job_count"]:
        raise DojoLongHorizonExecutionError(
            "terminal manifests do not cover the exact job denominator"
        )
    terminal_rows = [
        _validate_terminal_prevalidated(value, schedule=schedule, manifest=manifest)
        for value in verified_terminals
    ]
    expected_job_order = [job["job_sha256"] for job in schedule["jobs"]]
    if [row["job_sha256"] for row in terminal_rows] != expected_job_order:
        raise DojoLongHorizonExecutionError(
            "terminal manifests must follow the complete sealed schedule order"
        )
    _validate_terminal_continuity(terminal_rows)
    handoffs = _sequence(values, field="reducer_handoffs")
    if len(handoffs) != len(terminal_rows):
        raise DojoLongHorizonExecutionError(
            "reducer handoffs do not cover the exact job denominator"
        )
    result: list[dict[str, Any]] = []
    for index, (value, terminal_row) in enumerate(
        zip(handoffs, terminal_rows, strict=True)
    ):
        expected = _build_reducer_handoff(terminal_row, manifest=manifest)
        if dict(value) != expected:
            raise DojoLongHorizonExecutionError(
                f"reducer handoff[{index}] content drifted"
            )
        result.append(expected)
    return result


def validate_long_horizon_terminal_reducer_bundle(
    *,
    terminals: Sequence[Mapping[str, Any]],
    handoffs: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all terminals and reducer handoffs in one parent/terminal pass."""

    manifest = validate_long_horizon_execution_manifest(
        execution_manifest, schedule=schedule, plan=plan
    )
    terminal_values = _sequence(terminals, field="terminal_manifests")
    if len(terminal_values) != manifest["job_count"]:
        raise DojoLongHorizonExecutionError(
            "terminal manifests do not cover the exact job denominator"
        )
    terminal_rows = [
        _validate_terminal_prevalidated(value, schedule=schedule, manifest=manifest)
        for value in terminal_values
    ]
    expected_job_order = [job["job_sha256"] for job in schedule["jobs"]]
    if [row["job_sha256"] for row in terminal_rows] != expected_job_order:
        raise DojoLongHorizonExecutionError(
            "terminal manifests must follow the complete sealed schedule order"
        )
    _validate_terminal_continuity(terminal_rows)
    handoff_values = _sequence(handoffs, field="reducer_handoffs")
    if len(handoff_values) != len(terminal_rows):
        raise DojoLongHorizonExecutionError(
            "reducer handoffs do not cover the exact job denominator"
        )
    handoff_rows: list[dict[str, Any]] = []
    for index, (value, terminal_row) in enumerate(
        zip(handoff_values, terminal_rows, strict=True)
    ):
        expected = _build_reducer_handoff(terminal_row, manifest=manifest)
        if dict(value) != expected:
            raise DojoLongHorizonExecutionError(
                f"reducer handoff[{index}] content drifted"
            )
        handoff_rows.append(expected)
    return {
        "terminal_manifests": terminal_rows,
        "reducer_handoffs": handoff_rows,
    }


def _safe_root(path: Path) -> Path:
    candidate = path.absolute()
    if candidate == candidate.parent:
        raise DojoLongHorizonExecutionError(
            "state directory cannot be a filesystem root"
        )
    parts = candidate.parts
    if len(parts) < 3:
        raise DojoLongHorizonExecutionError("state directory is too broad")
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        try:
            state = current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=0o700)
            state = current.lstat()
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
            raise DojoLongHorizonExecutionError(
                "state path components must be real directories"
            )
    return candidate.resolve(strict=True)


def _ensure_directory(path: Path) -> Path:
    root = _safe_root(path.parent)
    candidate = root / path.name
    try:
        candidate.mkdir(mode=0o700)
    except FileExistsError:
        pass
    state = candidate.lstat()
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise DojoLongHorizonExecutionError("state child must be a real directory")
    return candidate.resolve(strict=True)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> bool:
    payload = _canonical_bytes(value) + b"\n"
    if not payload or len(payload) > MAX_CONTROL_ARTIFACT_BYTES:
        raise DojoLongHorizonExecutionError("control artifact byte limit exceeded")
    directory = _ensure_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    created = False
    file_synced = False
    directory_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            existing = _read_json(path, field=str(path))
            if _canonical_bytes(existing) + b"\n" != payload:
                raise DojoLongHorizonExecutionError(
                    f"immutable artifact already exists with different bytes: {path.name}"
                )
            return False
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            written = handle.write(payload)
            if written != len(payload):
                raise DojoLongHorizonExecutionError("exclusive write was incomplete")
            handle.flush()
            os.fsync(handle.fileno())
            file_synced = True
        os.fsync(directory_fd)
        return True
    except BaseException as exc:
        # Before the file itself is durable, its immutable final name must not
        # survive a failed write.  Once file fsync has succeeded, however, a
        # directory-fsync failure means publication durability is unknown.  In
        # that phase keep the complete artifact: a retry verifies the exact
        # bytes above and returns False instead of deleting a possibly
        # published value.
        if created and not file_synced:
            try:
                os.unlink(path.name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                exc.add_note(
                    "failed to remove incomplete exclusive artifact: "
                    f"{cleanup_error!r}"
                )
            else:
                try:
                    os.fsync(directory_fd)
                except OSError as cleanup_error:
                    exc.add_note(
                        "failed to fsync incomplete-artifact removal: "
                        f"{cleanup_error!r}"
                    )
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoLongHorizonExecutionError(f"cannot inspect {field}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_size <= 0
        or before.st_size > MAX_CONTROL_ARTIFACT_BYTES
    ):
        raise DojoLongHorizonExecutionError(f"{field} is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            raw = handle.read(MAX_CONTROL_ARTIFACT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise DojoLongHorizonExecutionError(f"cannot read {field}") from exc
    if (
        before.st_dev != opened.st_dev
        or before.st_ino != opened.st_ino
        or opened.st_dev != after.st_dev
        or opened.st_ino != after.st_ino
        or opened.st_size != after.st_size
        or opened.st_mtime_ns != after.st_mtime_ns
        or len(raw) != opened.st_size
    ):
        raise DojoLongHorizonExecutionError(f"{field} changed while read")

    def reject_constant(token: str) -> None:
        raise DojoLongHorizonExecutionError(f"non-finite JSON is forbidden: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DojoLongHorizonExecutionError(
                    f"duplicate JSON key is forbidden: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except DojoLongHorizonExecutionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoLongHorizonExecutionError(
            f"strict JSON parse failed: {field}"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise DojoLongHorizonExecutionError(f"{field} is not canonical JSON")
    return value


def initialize_long_horizon_execution_state(
    state_dir: Path,
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    runner_binding: Mapping[str, Any],
    resource_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Create or idempotently verify one immutable execution state root."""

    root = _safe_root(state_dir)
    for child in ("claims", "cells", "terminals", "carry", "reducers"):
        _ensure_directory(root / child)
    manifest = build_long_horizon_execution_manifest(
        schedule,
        plan=plan,
        runner_binding=runner_binding,
        resource_policy=resource_policy,
    )
    _write_exclusive(root / "execution-manifest.json", manifest)
    return manifest


def _load_manifest(
    state_dir: Path,
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    validated_session: LongHorizonExecutionSession | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = _safe_root(state_dir)
    raw = _read_json(root / "execution-manifest.json", field="execution manifest")
    if validated_session is not None:
        if root != validated_session.state_dir:
            raise DojoLongHorizonExecutionError(
                "validated execution session belongs to a different state root"
            )
        if (
            schedule is not validated_session.schedule
            or plan is not validated_session.plan
            or raw != validated_session.execution_manifest
        ):
            raise DojoLongHorizonExecutionError(
                "validated execution session input or manifest drifted"
            )
        return root, validated_session.execution_manifest
    return root, validate_long_horizon_execution_manifest(
        raw, schedule=schedule, plan=plan
    )


def _claim_files(root: Path, job_sha256: str) -> list[Path]:
    directory = root / "claims" / job_sha256
    if not directory.exists():
        return []
    _safe_root(directory)
    result = sorted(directory.glob("attempt-*.json"))
    if any(not re.fullmatch(r"attempt-[0-9]{4}\.json", item.name) for item in result):
        raise DojoLongHorizonExecutionError("claim directory has an unknown artifact")
    return result


def _terminal_files(root: Path, job_sha256: str) -> list[Path]:
    directory = root / "terminals" / job_sha256
    if not directory.exists():
        return []
    _safe_root(directory)
    result = sorted(directory.glob("attempt-*.json"))
    if any(
        not re.fullmatch(r"attempt-[0-9]{4}-[0-9a-f]{64}-[0-9a-f]{64}\.json", item.name)
        for item in result
    ):
        raise DojoLongHorizonExecutionError(
            "terminal directory has an unknown artifact"
        )
    return result


def _attempt_history(
    root: Path,
    *,
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    claims = []
    for path in _claim_files(root, job["job_sha256"]):
        claim = _validate_claim(
            _read_json(path, field="claim"), manifest=manifest, job=job
        )
        expected_name = f"attempt-{claim['attempt_ordinal']:04d}.json"
        if path.name != expected_name:
            raise DojoLongHorizonExecutionError("claim filename binding drifted")
        claims.append(claim)
    claims.sort(key=lambda row: row["attempt_ordinal"])
    if [row["attempt_ordinal"] for row in claims] != list(range(1, len(claims) + 1)):
        raise DojoLongHorizonExecutionError("claim attempt sequence is not contiguous")
    terminals_by_claim: dict[str, dict[str, Any]] = {}
    for path in _terminal_files(root, job["job_sha256"]):
        terminal = _validate_terminal_prevalidated(
            _read_json(path, field="terminal manifest"),
            schedule=schedule,
            manifest=manifest,
        )
        expected_name = (
            f"attempt-{terminal['attempt_ordinal']:04d}-"
            f"{terminal['claim']['claim_sha256']}-{terminal['terminal_sha256']}.json"
        )
        if path.name != expected_name:
            raise DojoLongHorizonExecutionError("terminal filename binding drifted")
        claim_sha = terminal["claim"]["claim_sha256"]
        if claim_sha in terminals_by_claim:
            raise DojoLongHorizonExecutionError("claim has multiple terminal manifests")
        terminals_by_claim[claim_sha] = terminal
    history = [
        (claim, terminals_by_claim.pop(claim["claim_sha256"], None)) for claim in claims
    ]
    if terminals_by_claim:
        raise DojoLongHorizonExecutionError("terminal exists without a durable claim")
    for index, (_, terminal) in enumerate(history[:-1]):
        if terminal is None:
            raise DojoLongHorizonExecutionError("new attempt followed an active claim")
        if terminal["terminal_status"] != "FAILED":
            raise DojoLongHorizonExecutionError("attempt followed a COMPLETE terminal")
    return history


def _validate_carry_receipt(value: Any) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "contract",
            "schema_version",
            "execution_manifest_sha256",
            "schedule_sha256",
            "state_slot_id",
            "state_sha256",
            "producer_job_sha256",
            "producer_coordinate_id",
            "producer_claim_sha256",
            "producer_terminal_sha256",
            "authority",
            "carry_receipt_sha256",
        },
        field="carry_receipt",
    )
    if (
        row["contract"] != CARRY_SLOT_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
    ):
        raise DojoLongHorizonExecutionError("carry receipt contract is unsupported")
    for field in (
        "execution_manifest_sha256",
        "schedule_sha256",
        "state_slot_id",
        "state_sha256",
        "producer_job_sha256",
        "producer_coordinate_id",
        "producer_claim_sha256",
        "producer_terminal_sha256",
    ):
        _sha(row[field], field=f"carry_receipt.{field}")
    _verify_authority(row["authority"], field="carry_receipt.authority")
    body = {key: row[key] for key in row if key != "carry_receipt_sha256"}
    if row["carry_receipt_sha256"] != canonical_sha256(body):
        raise DojoLongHorizonExecutionError("carry receipt digest drifted")
    return dict(row)


def _carry_producer_index(
    schedule: Mapping[str, Any],
) -> dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    result: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for job in schedule["jobs"]:
        for coordinate in job["coordinates"]:
            slot = coordinate["carry_out_state_slot_id"]
            if slot is None:
                continue
            if slot in result:
                raise DojoLongHorizonExecutionError(
                    "carry state slot has multiple producer coordinates"
                )
            result[slot] = (job, coordinate)
    return result


def _predecessor_states(
    root: Path,
    *,
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    producer_cache: dict[str, dict[str, Any]] = {}
    producer_history_cache: dict[
        str, list[tuple[dict[str, Any], dict[str, Any] | None]]
    ] = {}
    producer_index = _carry_producer_index(schedule)
    for slot in _slot_ids(job, key="predecessor_state_slot_id"):
        path = root / "carry" / f"{slot}.json"
        if not path.exists():
            producer_binding = producer_index.get(slot)
            if producer_binding is None:
                raise DojoLongHorizonExecutionError(
                    "predecessor state slot has no sealed producer coordinate"
                )
            producer_job, producer_coordinate = producer_binding
            history = producer_history_cache.get(producer_job["job_sha256"])
            if history is None:
                history = _attempt_history(
                    root,
                    job=producer_job,
                    manifest=manifest,
                    schedule=schedule,
                    plan=plan,
                )
                producer_history_cache[producer_job["job_sha256"]] = history
            if not history or history[-1][1] is None:
                result[slot] = {"status": "PENDING", "state_slot_id": slot}
                continue
            producer_terminal = history[-1][1]
            assert producer_terminal is not None
            producer_cell = next(
                cell
                for cell in producer_terminal["cells"]
                if cell["coordinate_id"] == producer_coordinate["coordinate_id"]
            )
            if producer_cell["status"] == "FAILED":
                result[slot] = {
                    "status": "FAILED",
                    "state_slot_id": slot,
                    "producer_job_sha256": producer_job["job_sha256"],
                    "producer_coordinate_id": producer_cell["coordinate_id"],
                    "producer_cell_sha256": producer_cell["cell_sha256"],
                    "producer_failure_code": producer_cell["failure"]["code"],
                }
                continue
            # A terminal may be durable before a crash-safe carry publication
            # finishes.  The producer's `seal` retry will publish the missing
            # slot; do not reinterpret the valid COMPLETE cell as failure.
            result[slot] = {"status": "PENDING", "state_slot_id": slot}
            continue
        receipt = _validate_carry_receipt(_read_json(path, field="carry receipt"))
        if (
            receipt["execution_manifest_sha256"]
            != manifest["execution_manifest_sha256"]
            or receipt["schedule_sha256"] != manifest["schedule_sha256"]
            or receipt["state_slot_id"] != slot
        ):
            raise DojoLongHorizonExecutionError("predecessor carry binding drifted")
        terminal_path = (
            root
            / "terminals"
            / receipt["producer_job_sha256"]
            / (
                f"attempt-*-{receipt['producer_claim_sha256']}-"
                f"{receipt['producer_terminal_sha256']}.json"
            )
        )
        matches = list(terminal_path.parent.glob(terminal_path.name))
        if len(matches) != 1:
            raise DojoLongHorizonExecutionError(
                "carry slot does not have one retained producer terminal"
            )
        producer = producer_cache.get(receipt["producer_terminal_sha256"])
        if producer is None:
            producer = _validate_terminal_prevalidated(
                _read_json(matches[0], field="carry producer terminal"),
                schedule=schedule,
                manifest=manifest,
            )
            producer_cache[receipt["producer_terminal_sha256"]] = producer
        producer_cells = {cell["coordinate_id"]: cell for cell in producer["cells"]}
        producer_cell = producer_cells.get(receipt["producer_coordinate_id"])
        if (
            producer["terminal_sha256"] != receipt["producer_terminal_sha256"]
            or producer_cell is None
            or producer_cell["status"] != "COMPLETE"
            or producer_cell["carry_out_state_slot_id"] != slot
            or producer_cell["carry_out_state_sha256"] != receipt["state_sha256"]
        ):
            raise DojoLongHorizonExecutionError(
                "carry slot disagrees with its complete producer terminal"
            )
        result[slot] = {
            "status": "READY",
            "state_slot_id": slot,
            "state_sha256": receipt["state_sha256"],
            "carry_receipt_sha256": receipt["carry_receipt_sha256"],
            "producer_job_sha256": receipt["producer_job_sha256"],
            "producer_coordinate_id": receipt["producer_coordinate_id"],
            "producer_cell_sha256": producer_cell["cell_sha256"],
        }
    return result


def claim_long_horizon_job(
    state_dir: Path,
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    job_sha256: str,
    runner_id: str,
    _validated_session: LongHorizonExecutionSession | None = None,
) -> dict[str, Any]:
    """Claim one exact job using an append-only ``O_EXCL`` attempt slot."""

    root, manifest = _load_manifest(
        state_dir,
        schedule=schedule,
        plan=plan,
        validated_session=_validated_session,
    )
    jobs = _jobs_by_sha(schedule)
    if job_sha256 not in jobs:
        raise DojoLongHorizonExecutionError("job is absent from sealed schedule")
    job = jobs[job_sha256]
    runner = _identifier(runner_id, field="runner_id")
    history = _attempt_history(
        root,
        job=job,
        manifest=manifest,
        schedule=schedule,
        plan=plan,
    )
    if history:
        last_claim, last_terminal = history[-1]
        if last_terminal is None:
            if last_claim["runner_id"] == runner:
                return resume_long_horizon_claim(
                    state_dir,
                    schedule=schedule,
                    plan=plan,
                    claim_sha256=last_claim["claim_sha256"],
                    _validated_session=_validated_session,
                )
            raise DojoLongHorizonExecutionError("job already has an active claim")
        if last_terminal["terminal_status"] == "COMPLETE":
            raise DojoLongHorizonExecutionError("job already completed")
        raise DojoLongHorizonExecutionError(
            "FAILED terminal is absorbing and cannot be reclaimed"
        )
    attempt = len(history) + 1
    if attempt > MAX_ATTEMPTS_PER_JOB:
        raise DojoLongHorizonExecutionError("job exhausted its retained attempt budget")
    claim = _claim_body(
        manifest=manifest,
        job_sha256=job_sha256,
        required_predecessor_slot_ids=_slot_ids(job, key="predecessor_state_slot_id"),
        attempt_ordinal=attempt,
        runner_id=runner,
    )
    claim_path = root / "claims" / job_sha256 / f"attempt-{attempt:04d}.json"
    try:
        _write_exclusive(claim_path, claim)
    except DojoLongHorizonExecutionError:
        # A concurrent claimer may have won the O_EXCL race.  Never skip ahead
        # based on stale state; reload the exact job and fail/return idempotently.
        history = _attempt_history(
            root,
            job=job,
            manifest=manifest,
            schedule=schedule,
            plan=plan,
        )
        if history and history[-1][1] is None and history[-1][0]["runner_id"] == runner:
            claim = history[-1][0]
        else:
            raise
    return resume_long_horizon_claim(
        state_dir,
        schedule=schedule,
        plan=plan,
        claim_sha256=claim["claim_sha256"],
        _validated_session=_validated_session,
    )


def _find_claim(
    root: Path,
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    claim_sha256: str,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any] | None]:
    target = str(_sha(claim_sha256, field="claim_sha256"))
    jobs = _jobs_by_sha(schedule)
    claims_root = root / "claims"
    entries = list(claims_root.iterdir())
    if any(item.is_symlink() or not item.is_dir() for item in entries):
        raise DojoLongHorizonExecutionError("claims root has a non-directory artifact")
    directories = sorted(entries)
    if any(
        _SHA256_RE.fullmatch(item.name) is None or item.name not in jobs
        for item in directories
    ):
        raise DojoLongHorizonExecutionError("claims root has an unknown job directory")
    for directory in directories:
        job = jobs[directory.name]
        history = _attempt_history(
            root,
            job=job,
            manifest=manifest,
            schedule=schedule,
            plan=plan,
        )
        for claim, terminal in history:
            if claim["claim_sha256"] == target:
                return job, claim, terminal
    raise DojoLongHorizonExecutionError("claim is absent from execution state")


def _cell_directory(root: Path, claim: Mapping[str, Any]) -> Path:
    return (
        root
        / "cells"
        / claim["job_sha256"]
        / f"attempt-{claim['attempt_ordinal']:04d}-{claim['claim_sha256']}"
    )


def _load_cells(
    root: Path,
    *,
    job: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    directory = _cell_directory(root, claim)
    if not directory.exists():
        return {}
    _safe_root(directory)
    paths = sorted(directory.glob("*.json"))
    if any(_SHA256_RE.fullmatch(item.stem) is None for item in paths):
        raise DojoLongHorizonExecutionError("cell directory has an unknown artifact")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        cell = validate_long_horizon_coordinate_result(
            _read_json(path, field="coordinate result"), job=job, claim=claim
        )
        if path.stem != cell["coordinate_id"] or cell["coordinate_id"] in result:
            raise DojoLongHorizonExecutionError("coordinate result filename drifted")
        result[cell["coordinate_id"]] = cell
    return result


def _materialize_failed_predecessor_cells(
    root: Path,
    *,
    job: Mapping[str, Any],
    claim: Mapping[str, Any],
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Append deterministic failures only for chains whose producer cell failed."""

    cells = _load_cells(root, job=job, claim=claim)
    states = _predecessor_states(
        root,
        job=job,
        manifest=manifest,
        schedule=schedule,
        plan=plan,
    )
    for coordinate_id, coordinate in _coordinate_map(job).items():
        if coordinate_id in cells:
            continue
        slot = coordinate["predecessor_state_slot_id"]
        state = states.get(slot) if slot is not None else None
        if state is None or state["status"] != "FAILED":
            continue
        cell = build_long_horizon_coordinate_result(
            job=job,
            claim=claim,
            coordinate_id=coordinate_id,
            status="FAILED",
            failure={
                "code": "PREDECESSOR_FAILED",
                "retryable": False,
                "evidence_sha256": state["producer_cell_sha256"],
            },
        )
        _write_exclusive(_cell_directory(root, claim) / f"{coordinate_id}.json", cell)
        cells[coordinate_id] = cell
    return cells


def resume_long_horizon_claim(
    state_dir: Path,
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    claim_sha256: str,
    _validated_session: LongHorizonExecutionSession | None = None,
) -> dict[str, Any]:
    """Return the same typed runner handoff after a crash/restart."""

    root, manifest = _load_manifest(
        state_dir,
        schedule=schedule,
        plan=plan,
        validated_session=_validated_session,
    )
    job, claim, terminal = _find_claim(
        root,
        schedule=schedule,
        plan=plan,
        manifest=manifest,
        claim_sha256=claim_sha256,
    )
    cells = (
        _load_cells(root, job=job, claim=claim)
        if terminal is not None
        else _materialize_failed_predecessor_cells(
            root,
            job=job,
            claim=claim,
            manifest=manifest,
            schedule=schedule,
            plan=plan,
        )
    )
    predecessor_states = _predecessor_states(
        root,
        job=job,
        manifest=manifest,
        schedule=schedule,
        plan=plan,
    )
    coordinate_by_id = _coordinate_map(job)
    coordinate_ids = list(coordinate_by_id)
    pending = [
        coordinate_id for coordinate_id in coordinate_ids if coordinate_id not in cells
    ]
    runnable: list[str] = []
    blocked: list[str] = []
    for coordinate_id in pending:
        coordinate = coordinate_by_id[coordinate_id]
        slot = coordinate["predecessor_state_slot_id"]
        if slot is None or predecessor_states[slot]["status"] == "READY":
            runnable.append(coordinate_id)
        else:
            blocked.append(coordinate_id)
    ready_receipts = [
        state
        for slot, state in sorted(predecessor_states.items())
        if state["status"] == "READY"
    ]
    recorded_ids = [
        coordinate_id for coordinate_id in coordinate_ids if coordinate_id in cells
    ]
    body = {
        "contract": RUNNER_HANDOFF_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "runner_binding": manifest["runner_binding"],
        "claim": claim,
        "job": dict(job),
        "coordinate_count": len(coordinate_ids),
        "recorded_coordinate_count": len(cells),
        "pending_coordinate_count": len(pending),
        "recorded_coordinate_ids_sha256": canonical_sha256(recorded_ids),
        "pending_coordinate_ids": pending,
        "pending_coordinate_ids_sha256": canonical_sha256(pending),
        "runnable_coordinate_count": len(runnable),
        "runnable_coordinate_ids": runnable,
        "runnable_coordinate_ids_sha256": canonical_sha256(runnable),
        "predecessor_blocked_coordinate_count": len(blocked),
        "predecessor_blocked_coordinate_ids": blocked,
        "predecessor_blocked_coordinate_ids_sha256": canonical_sha256(blocked),
        "ready_predecessor_carry_receipts": ready_receipts,
        "ready_predecessor_carry_receipts_sha256": canonical_sha256(ready_receipts),
        "terminal_status": terminal["terminal_status"] if terminal else None,
        "resource_limits": manifest["resource_limits"],
        "resource_enforcement": manifest["resource_enforcement"],
        "runner_obligations": {
            "one_synchronized_source_stream": True,
            "fanout_before_any_coordinate_decision": True,
            "source_reopen_or_resort_allowed": False,
            "all_coordinates_must_receive_terminal_result": True,
            "failed_result_zero_fill_allowed": False,
            "broker_or_live_path_allowed": False,
            "portfolio_economics_implemented_by_state_machine": False,
            "resource_exhaustion_action": "PAUSE_AND_RESUME_SAME_CLAIM",
        },
        "authority": _authority(),
    }
    return {**body, "runner_handoff_sha256": canonical_sha256(body)}


def record_long_horizon_coordinate_result(
    state_dir: Path,
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    claim_sha256: str,
    result: Mapping[str, Any],
    _validated_session: LongHorizonExecutionSession | None = None,
) -> dict[str, Any]:
    """Append one immutable coordinate result to an active claim."""

    return record_long_horizon_coordinate_results(
        state_dir,
        schedule=schedule,
        plan=plan,
        claim_sha256=claim_sha256,
        results=[result],
        _validated_session=_validated_session,
    )[0]


def record_long_horizon_coordinate_results(
    state_dir: Path,
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    claim_sha256: str,
    results: Sequence[Mapping[str, Any]],
    _validated_session: LongHorizonExecutionSession | None = None,
) -> list[dict[str, Any]]:
    """Append a bounded batch while preserving one O_EXCL slot per cell."""

    root, manifest = _load_manifest(
        state_dir,
        schedule=schedule,
        plan=plan,
        validated_session=_validated_session,
    )
    job, claim, terminal = _find_claim(
        root,
        schedule=schedule,
        plan=plan,
        manifest=manifest,
        claim_sha256=claim_sha256,
    )
    if terminal is not None:
        raise DojoLongHorizonExecutionError("terminal claim cannot accept new cells")
    _materialize_failed_predecessor_cells(
        root,
        job=job,
        claim=claim,
        manifest=manifest,
        schedule=schedule,
        plan=plan,
    )
    raw_results = _sequence(results, field="coordinate results")
    if not raw_results:
        raise DojoLongHorizonExecutionError("coordinate result batch cannot be empty")
    if len(raw_results) > job["coordinate_count"]:
        raise DojoLongHorizonExecutionError(
            "coordinate result batch exceeds job denominator"
        )
    cells = [
        validate_long_horizon_coordinate_result(result, job=job, claim=claim)
        for result in raw_results
    ]
    if len({cell["coordinate_id"] for cell in cells}) != len(cells):
        raise DojoLongHorizonExecutionError("coordinate result batch has duplicate ids")
    predecessor_states = _predecessor_states(
        root,
        job=job,
        manifest=manifest,
        schedule=schedule,
        plan=plan,
    )
    for cell in cells:
        predecessor_slot = cell["predecessor_state_slot_id"]
        if predecessor_slot is None:
            continue
        state = predecessor_states[predecessor_slot]
        if cell["status"] == "COMPLETE":
            if (
                state["status"] != "READY"
                or cell["predecessor_state_sha256"] != state["state_sha256"]
            ):
                raise DojoLongHorizonExecutionError(
                    "cell predecessor state is not a verified ready carry slot"
                )
        elif state["status"] == "FAILED" and (
            cell["failure"]["code"] != "PREDECESSOR_FAILED"
            or cell["failure"]["evidence_sha256"] != state["producer_cell_sha256"]
        ):
            raise DojoLongHorizonExecutionError(
                "failed predecessor must retain deterministic propagation evidence"
            )
        elif state["status"] == "PENDING":
            raise DojoLongHorizonExecutionError(
                "predecessor-pending coordinate cannot record a terminal result"
            )
    for cell in cells:
        path = _cell_directory(root, claim) / f"{cell['coordinate_id']}.json"
        _write_exclusive(path, cell)
    return cells


def _publish_carry_slots(
    root: Path,
    *,
    terminal: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    for cell in terminal["cells"]:
        if cell["status"] != "COMPLETE":
            continue
        slot = cell["carry_out_state_slot_id"]
        if slot is None:
            continue
        body = {
            "contract": CARRY_SLOT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "execution_manifest_sha256": manifest["execution_manifest_sha256"],
            "schedule_sha256": manifest["schedule_sha256"],
            "state_slot_id": slot,
            "state_sha256": cell["carry_out_state_sha256"],
            "producer_job_sha256": terminal["job_sha256"],
            "producer_coordinate_id": cell["coordinate_id"],
            "producer_claim_sha256": terminal["claim"]["claim_sha256"],
            "producer_terminal_sha256": terminal["terminal_sha256"],
            "authority": _authority(),
        }
        receipt = {**body, "carry_receipt_sha256": canonical_sha256(body)}
        _write_exclusive(root / "carry" / f"{slot}.json", receipt)


def seal_long_horizon_attempt(
    state_dir: Path,
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    claim_sha256: str,
    _validated_session: LongHorizonExecutionSession | None = None,
) -> dict[str, Any]:
    """Seal the full denominator, then publish carry and reducer handoffs."""

    root, manifest = _load_manifest(
        state_dir,
        schedule=schedule,
        plan=plan,
        validated_session=_validated_session,
    )
    job, claim, existing_terminal = _find_claim(
        root,
        schedule=schedule,
        plan=plan,
        manifest=manifest,
        claim_sha256=claim_sha256,
    )
    cells_by_id = _load_cells(root, job=job, claim=claim)
    coordinate_ids = list(_coordinate_map(job))
    missing = [
        coordinate_id
        for coordinate_id in coordinate_ids
        if coordinate_id not in cells_by_id
    ]
    if missing:
        raise DojoLongHorizonExecutionError(
            f"attempt cannot seal with {len(missing)} missing coordinates"
        )
    cells = [cells_by_id[coordinate_id] for coordinate_id in coordinate_ids]
    body = _terminal_body(manifest=manifest, job=job, claim=claim, cells=cells)
    terminal = {**body, "terminal_sha256": canonical_sha256(body)}
    terminal = _validate_terminal_prevalidated(
        terminal,
        schedule=schedule,
        manifest=manifest,
    )
    if (
        len(_canonical_bytes(terminal)) + 1
        > manifest["resource_limits"]["max_terminal_bytes"]
    ):
        raise DojoLongHorizonExecutionError(
            "terminal exceeds its sealed resource limit; resume the same claim"
        )
    if existing_terminal is not None and existing_terminal != terminal:
        raise DojoLongHorizonExecutionError("claim already has a different terminal")
    terminal_path = (
        root
        / "terminals"
        / job["job_sha256"]
        / (
            f"attempt-{claim['attempt_ordinal']:04d}-{claim['claim_sha256']}-"
            f"{terminal['terminal_sha256']}.json"
        )
    )
    _write_exclusive(terminal_path, terminal)
    _publish_carry_slots(root, terminal=terminal, manifest=manifest)
    reducer = _build_reducer_handoff(terminal, manifest=manifest)
    _write_exclusive(
        root / "reducers" / job["job_sha256"] / f"{terminal['terminal_sha256']}.json",
        reducer,
    )
    return {"terminal_manifest": terminal, "reducer_handoff": reducer}


def claim_next_long_horizon_job(
    state_dir: Path,
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    runner_id: str,
    _validated_session: LongHorizonExecutionSession | None = None,
) -> dict[str, Any]:
    """Claim the first ready nonterminal job in sealed schedule order."""

    root, manifest = _load_manifest(
        state_dir,
        schedule=schedule,
        plan=plan,
        validated_session=_validated_session,
    )
    for job in schedule["jobs"]:
        history = _attempt_history(
            root,
            job=job,
            manifest=manifest,
            schedule=schedule,
            plan=plan,
        )
        if history:
            claim, terminal = history[-1]
            if terminal is None:
                if claim["runner_id"] == runner_id:
                    return resume_long_horizon_claim(
                        state_dir,
                        schedule=schedule,
                        plan=plan,
                        claim_sha256=claim["claim_sha256"],
                        _validated_session=_validated_session,
                    )
                continue
            if terminal["terminal_status"] == "COMPLETE":
                continue
            if len(history) >= MAX_ATTEMPTS_PER_JOB:
                continue
        try:
            return claim_long_horizon_job(
                state_dir,
                schedule=schedule,
                plan=plan,
                job_sha256=job["job_sha256"],
                runner_id=runner_id,
                _validated_session=_validated_session,
            )
        except DojoLongHorizonExecutionError as exc:
            if "active claim" in str(exc):
                continue
            raise
    raise DojoLongHorizonExecutionError("no ready long-horizon job remains")


def long_horizon_execution_status(
    state_dir: Path,
    *,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    _validated_session: LongHorizonExecutionSession | None = None,
) -> dict[str, Any]:
    """Return deterministic counts without interpreting partial P/L."""

    root, manifest = _load_manifest(
        state_dir,
        schedule=schedule,
        plan=plan,
        validated_session=_validated_session,
    )
    complete_jobs = 0
    terminal_jobs = 0
    failed_terminal_jobs = 0
    active_jobs = 0
    failed_attempts = 0
    recorded_cells = 0
    pending_cells = 0
    ready_jobs = 0
    predecessor_blocked_jobs = 0
    exhausted_jobs = 0
    reducer_handoffs = 0
    for job in schedule["jobs"]:
        history = _attempt_history(
            root,
            job=job,
            manifest=manifest,
            schedule=schedule,
            plan=plan,
        )
        failed_attempts += sum(
            terminal is not None and terminal["terminal_status"] == "FAILED"
            for _, terminal in history
        )
        last_claim: Mapping[str, Any] | None = history[-1][0] if history else None
        last_terminal = history[-1][1] if history else None
        if last_terminal is not None:
            terminal_jobs += 1
            if last_terminal["terminal_status"] == "COMPLETE":
                complete_jobs += 1
            else:
                failed_terminal_jobs += 1
            reducer_path = (
                root
                / "reducers"
                / job["job_sha256"]
                / f"{last_terminal['terminal_sha256']}.json"
            )
            if reducer_path.exists():
                reducer_handoffs += 1
            continue
        if last_claim is not None and last_terminal is None:
            active_jobs += 1
            cells = _load_cells(root, job=job, claim=last_claim)
            recorded_cells += len(cells)
            pending_cells += job["coordinate_count"] - len(cells)
            states = _predecessor_states(
                root,
                job=job,
                manifest=manifest,
                schedule=schedule,
                plan=plan,
            )
            if any(state["status"] == "PENDING" for state in states.values()):
                predecessor_blocked_jobs += 1
            continue
        if len(history) >= MAX_ATTEMPTS_PER_JOB:
            exhausted_jobs += 1
            continue
        ready_jobs += 1
    body = {
        "contract": STATE_STATUS_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "job_count": manifest["job_count"],
        "terminal_job_count": terminal_jobs,
        "complete_job_count": complete_jobs,
        "failed_terminal_job_count": failed_terminal_jobs,
        "active_job_count": active_jobs,
        "ready_job_count": ready_jobs,
        "predecessor_blocked_job_count": predecessor_blocked_jobs,
        "exhausted_job_count": exhausted_jobs,
        "failed_attempt_count": failed_attempts,
        "active_recorded_coordinate_count": recorded_cells,
        "active_pending_coordinate_count": pending_cells,
        "reducer_handoff_count": reducer_handoffs,
        "terminal_complete": terminal_jobs == manifest["job_count"],
        "all_cells_complete": complete_jobs == manifest["job_count"],
        "profit_or_3x_status_computed": False,
        "resource_limit_enforcement_by_state_machine": False,
        "runner_pause_same_claim_on_resource_limit_required": True,
        "authority": _authority(),
    }
    return {**body, "status_sha256": canonical_sha256(body)}


class LongHorizonExecutionSession:
    """One fully validated long-lived execution session.

    Construction performs the expensive exact 32,112-coordinate parent
    rebuild once and detaches canonical copies.  Subsequent claim/checkpoint/
    resume/seal calls still reread and byte-compare the immutable state
    manifest, but do not rebuild or rehash the full schedule per cell.  Create
    a new session after any plan, schedule, manifest, or process boundary.
    """

    def __init__(
        self,
        state_dir: Path,
        *,
        schedule: Mapping[str, Any],
        plan: Mapping[str, Any],
    ) -> None:
        root, manifest = _load_manifest(state_dir, schedule=schedule, plan=plan)
        detached_schedule = json.loads(_canonical_bytes(schedule).decode("utf-8"))
        detached_plan = json.loads(_canonical_bytes(plan).decode("utf-8"))
        detached_manifest = json.loads(_canonical_bytes(manifest).decode("utf-8"))
        if not all(
            isinstance(value, dict)
            for value in (detached_schedule, detached_plan, detached_manifest)
        ):
            raise DojoLongHorizonExecutionError(
                "validated execution session parents must be JSON objects"
            )
        self.state_dir = root
        self.schedule: dict[str, Any] = detached_schedule
        self.plan: dict[str, Any] = detached_plan
        self.execution_manifest: dict[str, Any] = detached_manifest

    def claim_job(self, *, job_sha256: str, runner_id: str) -> dict[str, Any]:
        return claim_long_horizon_job(
            self.state_dir,
            schedule=self.schedule,
            plan=self.plan,
            job_sha256=job_sha256,
            runner_id=runner_id,
            _validated_session=self,
        )

    def claim_next(self, *, runner_id: str) -> dict[str, Any]:
        return claim_next_long_horizon_job(
            self.state_dir,
            schedule=self.schedule,
            plan=self.plan,
            runner_id=runner_id,
            _validated_session=self,
        )

    def resume(self, *, claim_sha256: str) -> dict[str, Any]:
        return resume_long_horizon_claim(
            self.state_dir,
            schedule=self.schedule,
            plan=self.plan,
            claim_sha256=claim_sha256,
            _validated_session=self,
        )

    def record_result(
        self, *, claim_sha256: str, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        return record_long_horizon_coordinate_result(
            self.state_dir,
            schedule=self.schedule,
            plan=self.plan,
            claim_sha256=claim_sha256,
            result=result,
            _validated_session=self,
        )

    def record_results(
        self,
        *,
        claim_sha256: str,
        results: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return record_long_horizon_coordinate_results(
            self.state_dir,
            schedule=self.schedule,
            plan=self.plan,
            claim_sha256=claim_sha256,
            results=results,
            _validated_session=self,
        )

    def seal(self, *, claim_sha256: str) -> dict[str, Any]:
        return seal_long_horizon_attempt(
            self.state_dir,
            schedule=self.schedule,
            plan=self.plan,
            claim_sha256=claim_sha256,
            _validated_session=self,
        )

    def status(self) -> dict[str, Any]:
        return long_horizon_execution_status(
            self.state_dir,
            schedule=self.schedule,
            plan=self.plan,
            _validated_session=self,
        )


__all__ = [
    "CARRY_SLOT_CONTRACT",
    "CELL_CONTRACT",
    "CLAIM_CONTRACT",
    "MANIFEST_CONTRACT",
    "MAX_ATTEMPTS_PER_JOB",
    "REDUCER_HANDOFF_CONTRACT",
    "RUNNER_HANDOFF_CONTRACT",
    "STATE_STATUS_CONTRACT",
    "TERMINAL_CONTRACT",
    "DojoLongHorizonExecutionError",
    "LongHorizonExecutionSession",
    "build_long_horizon_coordinate_result",
    "build_long_horizon_execution_manifest",
    "claim_long_horizon_job",
    "claim_next_long_horizon_job",
    "initialize_long_horizon_execution_state",
    "long_horizon_execution_status",
    "record_long_horizon_coordinate_result",
    "record_long_horizon_coordinate_results",
    "resume_long_horizon_claim",
    "seal_long_horizon_attempt",
    "validate_long_horizon_coordinate_result",
    "validate_long_horizon_execution_manifest",
    "validate_long_horizon_reducer_handoff",
    "validate_long_horizon_reducer_handoffs",
    "validate_long_horizon_terminal_manifest",
    "validate_long_horizon_terminal_manifests",
    "validate_long_horizon_terminal_reducer_bundle",
]

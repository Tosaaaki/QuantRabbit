"""Exact stream/fan-out schedule for the sealed DOJO long-horizon TRAIN.

``dojo_long_horizon_plan`` fixes aggregate denominator counts.  This module
turns those counts into the complete coordinate list that a future runner must
execute.  One job represents one immutable month/path price stream; every
cost arm, account mode, portfolio/LOPO counterfactual and strategy worker is
fed from that one synchronized stream instead of reopening or independently
sorting the corpus.

The schedule is declarative and side-effect free.  It does not open market
data, run a strategy, score evidence, archive artifacts, call a model or grant
any order/live authority.  Runtime terminal manifests must later prove that a
runner actually honored this fan-out contract.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from quant_rabbit.dojo_long_horizon_plan import (
    EVIDENCE_TIER,
    M5_BINDING_ID,
    DojoLongHorizonPlanError,
    canonical_sha256,
    validate_long_horizon_train_plan,
)


CONTRACT: Final = "QR_DOJO_LONG_HORIZON_STREAM_SCHEDULE_V2"
# This is an artifact schema identity, not a market/risk tuning constant.
SCHEMA_VERSION: Final = 2
# These are fan-out resource bounds, not market or sizing parameters.  Thirty-
# two total accounts keeps one stream job reviewable; eight variants per family
# prevents one causal family from consuming the complete research budget.  A
# measured runner resource profile should replace them if these limits change.
MAX_WORKERS: Final = 32
MAX_WORKERS_PER_FAMILY: Final = 8
# The two M1 rectangles overlap in pair/month labels, but they expose different
# synchronized market contexts (core-five versus all 28 pairs).  A worker may
# legitimately use those surrounding quotes for features, conversion or shared
# capital allocation, so the observations are not interchangeable replicas.
# Every raw coordinate therefore remains one full result in the denominator.
M1_CONTEXT_WEIGHT: Final = 1.0
_ZERO_SHA256: Final = "0" * 64
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_WORKER_ID_RE: Final = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")


class DojoLongHorizonScheduleError(ValueError):
    """The sealed stream schedule is malformed or contradicts its plan."""


def _authority() -> dict[str, Any]:
    return {
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "diagnostic_only": True,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }


def _execution_contract() -> dict[str, bool]:
    return {
        "single_synchronized_price_stream_per_job": True,
        "all_coordinates_receive_identical_batch_before_decisions": True,
        "one_shared_account_per_coordinate": True,
        "account_state_shared_between_coordinates": False,
        "continuous_account_state_chained_across_months": True,
        "independent_month_state_reset": True,
        "m1_overlapping_rectangles_are_distinct_contexts": True,
        "continuous_predecessor_state_slot_required_after_first_month": True,
        "runtime_terminal_manifest_per_coordinate_required": True,
        "runner_implementation_verified_by_this_artifact": False,
    }


def _strict_sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256_RE.fullmatch(value) is None
        or value == _ZERO_SHA256
    ):
        raise DojoLongHorizonScheduleError(
            f"{field} must be a non-zero lowercase SHA-256"
        )
    return value


def _worker_bindings(
    value: Sequence[Mapping[str, Any]], *, families: Sequence[str]
) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DojoLongHorizonScheduleError("worker_bindings must be a sequence")
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {
            "worker_id",
            "family_id",
            "config_sha256",
        }:
            raise DojoLongHorizonScheduleError(
                f"worker_bindings[{index}] schema is not exact"
            )
        worker_id = raw["worker_id"]
        family_id = raw["family_id"]
        if not isinstance(worker_id, str) or _WORKER_ID_RE.fullmatch(worker_id) is None:
            raise DojoLongHorizonScheduleError(
                f"worker_bindings[{index}].worker_id is invalid"
            )
        if not isinstance(family_id, str) or family_id not in families:
            raise DojoLongHorizonScheduleError(
                f"worker_bindings[{index}].family_id is not sealed in the plan"
            )
        rows.append(
            {
                "worker_id": worker_id,
                "family_id": family_id,
                "config_sha256": _strict_sha(
                    raw["config_sha256"],
                    field=f"worker_bindings[{index}].config_sha256",
                ),
            }
        )
    if not rows:
        raise DojoLongHorizonScheduleError("at least one worker binding is required")
    if len(rows) > MAX_WORKERS:
        raise DojoLongHorizonScheduleError(
            f"worker count exceeds the hard cap of {MAX_WORKERS}"
        )
    if rows != sorted(rows, key=lambda row: row["worker_id"]):
        raise DojoLongHorizonScheduleError(
            "worker_bindings must be sorted by worker_id"
        )
    worker_ids = [row["worker_id"] for row in rows]
    if len(worker_ids) != len(set(worker_ids)):
        raise DojoLongHorizonScheduleError("worker_id values must be unique")
    coverage = Counter(row["family_id"] for row in rows)
    if set(coverage) != set(families):
        raise DojoLongHorizonScheduleError(
            "every sealed portfolio family must have at least one worker"
        )
    oversized = sorted(
        family for family, count in coverage.items() if count > MAX_WORKERS_PER_FAMILY
    )
    if oversized:
        raise DojoLongHorizonScheduleError(
            "worker count exceeds the per-family hard cap: " + ",".join(oversized)
        )
    return rows


def _active_worker_variants(
    worker_rows: Sequence[Mapping[str, str]], *, families: Sequence[str]
) -> tuple[list[dict[str, Any]], str]:
    variants: list[dict[str, Any]] = []
    for excluded_family in (None, *families):
        active = [
            dict(row)
            for row in worker_rows
            if excluded_family is None or row["family_id"] != excluded_family
        ]
        active_ids = {row["worker_id"] for row in active}
        variants.append(
            {
                "excluded_family_id": excluded_family,
                "active_worker_mask": "".join(
                    "1" if row["worker_id"] in active_ids else "0"
                    for row in worker_rows
                ),
                "active_worker_bindings_sha256": canonical_sha256({"bindings": active}),
            }
        )
    body = {"variants": variants}
    return variants, canonical_sha256(body)


def _month_bounds(month: str) -> tuple[str, str]:
    year = int(month[:4])
    number = int(month[5:])
    next_year, next_month = (year + 1, 1) if number == 12 else (year, number + 1)
    return (
        f"{year:04d}-{number:02d}-01T00:00:00Z",
        f"{next_year:04d}-{next_month:02d}-01T00:00:00Z",
    )


def _coordinate(
    *,
    phase: str,
    price_stream_id: str,
    source_binding_id: str,
    month: str,
    evaluation_mode: str,
    intrabar_path: str,
    cost_scenario: str,
    stage: str,
    fold_label: str | None,
    precision_pair: str | None,
    feed_pairs: Sequence[str],
    trade_pairs: Sequence[str],
    active_families: Sequence[str],
    all_families: Sequence[str],
    active_worker_rows: Sequence[Mapping[str, str]],
    all_worker_rows: Sequence[Mapping[str, str]],
    plan_sha256: str,
    worker_set_sha256: str,
    month_ordinal: int,
    month_count: int,
    replica_group_id: str | None,
    replica_expected_count: int,
    aggregation_weight: float,
    replica_paired_consistency_required: bool,
) -> dict[str, Any]:
    terminal_policy = (
        "MONTH_END_FLAT_SETTLEMENT"
        if evaluation_mode == "INDEPENDENT_MONTH"
        else "MONTH_END_MTM_WITH_STATE_HANDOFF"
    )
    active_worker_ids = {row["worker_id"] for row in active_worker_rows}
    active_worker_mask = "".join(
        "1" if row["worker_id"] in active_worker_ids else "0" for row in all_worker_rows
    )
    active_worker_body = {"bindings": [dict(row) for row in active_worker_rows]}
    active_worker_bindings_sha256 = canonical_sha256(active_worker_body)
    trade_pair_ids = set(trade_pairs)
    trade_pair_mask = "".join(
        "1" if pair in trade_pair_ids else "0" for pair in feed_pairs
    )
    trade_pair_set_sha256 = canonical_sha256({"pairs": list(trade_pairs)})
    active_family_ids = set(active_families)
    active_family_mask = "".join(
        "1" if family in active_family_ids else "0" for family in all_families
    )
    active_family_set_sha256 = canonical_sha256({"families": list(active_families)})
    chain_material = {
        "plan_sha256": plan_sha256,
        "worker_set_sha256": worker_set_sha256,
        "phase": phase,
        "source_binding_id": source_binding_id,
        "evaluation_mode": evaluation_mode,
        "intrabar_path": intrabar_path,
        "cost_scenario": cost_scenario,
        "stage": stage,
        "fold_label": fold_label,
        "precision_pair": precision_pair,
        "trade_pair_mask": trade_pair_mask,
        "trade_pair_set_sha256": trade_pair_set_sha256,
        "active_family_mask": active_family_mask,
        "active_family_set_sha256": active_family_set_sha256,
        "active_worker_mask": active_worker_mask,
        "active_worker_bindings_sha256": active_worker_bindings_sha256,
        "replica_group_id": replica_group_id,
    }
    continuous_chain_id = (
        canonical_sha256(chain_material)
        if evaluation_mode == "CONTINUOUS_ACCOUNT"
        else None
    )
    if evaluation_mode == "CONTINUOUS_ACCOUNT":
        carry_in_state_slot_id = canonical_sha256(
            {
                "continuous_account_chain_id": continuous_chain_id,
                "boundary_ordinal": month_ordinal,
            }
        )
        carry_out_state_slot_id = canonical_sha256(
            {
                "continuous_account_chain_id": continuous_chain_id,
                "boundary_ordinal": month_ordinal + 1,
            }
        )
        predecessor_state_slot_id = (
            None if month_ordinal == 0 else carry_in_state_slot_id
        )
        continuous_chain_ordinal: int | None = month_ordinal
        continuous_chain_first = month_ordinal == 0
        continuous_chain_last = month_ordinal == month_count - 1
    else:
        carry_in_state_slot_id = None
        carry_out_state_slot_id = None
        predecessor_state_slot_id = None
        continuous_chain_ordinal = None
        continuous_chain_first = False
        continuous_chain_last = False
    body = {
        "plan_sha256": plan_sha256,
        "worker_set_sha256": worker_set_sha256,
        "phase": phase,
        "price_stream_id": price_stream_id,
        "source_binding_id": source_binding_id,
        "month": month,
        "evaluation_mode": evaluation_mode,
        "intrabar_path": intrabar_path,
        "cost_scenario": cost_scenario,
        "stage": stage,
        "fold_label": fold_label,
        "precision_pair": precision_pair,
        "feed_pairs": list(feed_pairs),
        "trade_pair_count": len(trade_pairs),
        "trade_pair_mask": trade_pair_mask,
        "trade_pair_set_sha256": trade_pair_set_sha256,
        "active_family_count": len(active_families),
        "active_family_mask": active_family_mask,
        "active_family_set_sha256": active_family_set_sha256,
        "active_worker_count": len(active_worker_rows),
        "active_worker_mask": active_worker_mask,
        "active_worker_bindings_sha256": active_worker_bindings_sha256,
        "shared_account_per_coordinate": True,
        "allocator_policy": "EXITS_FIRST_THEN_INPUT_ORDER_INVARIANT_RANKED_ENTRIES",
        "terminal_policy": terminal_policy,
        "continuous_account_chain_id": continuous_chain_id,
        "continuous_chain_ordinal": continuous_chain_ordinal,
        "continuous_chain_first": continuous_chain_first,
        "continuous_chain_last": continuous_chain_last,
        "carry_in_state_slot_id": carry_in_state_slot_id,
        "carry_out_state_slot_id": carry_out_state_slot_id,
        "predecessor_state_slot_id": predecessor_state_slot_id,
        "replica_group_id": replica_group_id,
        "replica_expected_count": replica_expected_count,
        "aggregation_weight": aggregation_weight,
        "replica_paired_consistency_required": replica_paired_consistency_required,
        "runtime_terminal_manifest_required": True,
    }
    # Job-level source/month/path/feed fields participate in the coordinate
    # identity but are stored once on the parent job.  Avoiding 32,112 copies
    # keeps the sealed schedule compact enough to archive and revalidate.
    descriptor = {
        key: value
        for key, value in body.items()
        if key
        not in {
            "phase",
            "price_stream_id",
            "source_binding_id",
            "month",
            "intrabar_path",
            "feed_pairs",
            "plan_sha256",
            "worker_set_sha256",
        }
    }
    return {"coordinate_id": canonical_sha256(body), **descriptor}


def _m5_coordinates(
    *,
    plan: Mapping[str, Any],
    source_binding_id: str,
    month: str,
    intrabar_path: str,
    price_stream_id: str,
    feed_pairs: Sequence[str],
    worker_rows: Sequence[Mapping[str, str]],
    plan_sha256: str,
    worker_set_sha256: str,
    month_ordinal: int,
    month_count: int,
) -> list[dict[str, Any]]:
    families = list(plan["portfolio"]["families"])
    modes = [row["mode"] for row in plan["evaluation"]["modes"]]
    scenarios = list(plan["evaluation"]["cost_scenarios"])
    currencies = list(plan["exact_denominator"]["portfolio_stages"][3]["labels"])
    result: list[dict[str, Any]] = []
    for mode in modes:
        for scenario in scenarios:
            common = {
                "phase": "BROAD_DISCOVERY_M5",
                "price_stream_id": price_stream_id,
                "source_binding_id": source_binding_id,
                "month": month,
                "evaluation_mode": mode,
                "intrabar_path": intrabar_path,
                "cost_scenario": scenario,
                "feed_pairs": feed_pairs,
            }
            result.append(
                _coordinate(
                    **common,
                    stage="PORTFOLIO_MAIN",
                    fold_label=None,
                    precision_pair=None,
                    trade_pairs=feed_pairs,
                    active_families=families,
                    all_families=families,
                    active_worker_rows=worker_rows,
                    all_worker_rows=worker_rows,
                    plan_sha256=plan_sha256,
                    worker_set_sha256=worker_set_sha256,
                    month_ordinal=month_ordinal,
                    month_count=month_count,
                    replica_group_id=None,
                    replica_expected_count=1,
                    aggregation_weight=1.0,
                    replica_paired_consistency_required=False,
                )
            )
            for held_out_pair in feed_pairs:
                result.append(
                    _coordinate(
                        **common,
                        stage="PAIR_LOPO",
                        fold_label=held_out_pair,
                        precision_pair=None,
                        trade_pairs=[
                            pair for pair in feed_pairs if pair != held_out_pair
                        ],
                        active_families=families,
                        all_families=families,
                        active_worker_rows=worker_rows,
                        all_worker_rows=worker_rows,
                        plan_sha256=plan_sha256,
                        worker_set_sha256=worker_set_sha256,
                        month_ordinal=month_ordinal,
                        month_count=month_count,
                        replica_group_id=None,
                        replica_expected_count=1,
                        aggregation_weight=1.0,
                        replica_paired_consistency_required=False,
                    )
                )
            for held_out_family in families:
                active_rows = [
                    row for row in worker_rows if row["family_id"] != held_out_family
                ]
                result.append(
                    _coordinate(
                        **common,
                        stage="FAMILY_LOPO",
                        fold_label=held_out_family,
                        precision_pair=None,
                        trade_pairs=feed_pairs,
                        active_families=[
                            family for family in families if family != held_out_family
                        ],
                        all_families=families,
                        active_worker_rows=active_rows,
                        all_worker_rows=worker_rows,
                        plan_sha256=plan_sha256,
                        worker_set_sha256=worker_set_sha256,
                        month_ordinal=month_ordinal,
                        month_count=month_count,
                        replica_group_id=None,
                        replica_expected_count=1,
                        aggregation_weight=1.0,
                        replica_paired_consistency_required=False,
                    )
                )
            for held_out_currency in currencies:
                result.append(
                    _coordinate(
                        **common,
                        stage="CURRENCY_LOPO",
                        fold_label=held_out_currency,
                        precision_pair=None,
                        trade_pairs=[
                            pair
                            for pair in feed_pairs
                            if held_out_currency not in pair.split("_")
                        ],
                        active_families=families,
                        all_families=families,
                        active_worker_rows=worker_rows,
                        all_worker_rows=worker_rows,
                        plan_sha256=plan_sha256,
                        worker_set_sha256=worker_set_sha256,
                        month_ordinal=month_ordinal,
                        month_count=month_count,
                        replica_group_id=None,
                        replica_expected_count=1,
                        aggregation_weight=1.0,
                        replica_paired_consistency_required=False,
                    )
                )
    return result


def _m1_coordinates(
    *,
    plan: Mapping[str, Any],
    source_binding_id: str,
    month: str,
    intrabar_path: str,
    price_stream_id: str,
    feed_pairs: Sequence[str],
    worker_rows: Sequence[Mapping[str, str]],
    plan_sha256: str,
    worker_set_sha256: str,
    month_ordinal: int,
    month_count: int,
) -> list[dict[str, Any]]:
    families = list(plan["portfolio"]["families"])
    modes = [row["mode"] for row in plan["evaluation"]["modes"]]
    scenarios = list(plan["evaluation"]["cost_scenarios"])
    result: list[dict[str, Any]] = []
    for precision_pair in feed_pairs:
        for mode in modes:
            for scenario in scenarios:
                result.append(
                    _coordinate(
                        phase="PRECISION_M1",
                        price_stream_id=price_stream_id,
                        source_binding_id=source_binding_id,
                        month=month,
                        evaluation_mode=mode,
                        intrabar_path=intrabar_path,
                        cost_scenario=scenario,
                        stage="M1_PRECISION_MAIN",
                        fold_label=None,
                        precision_pair=precision_pair,
                        feed_pairs=feed_pairs,
                        trade_pairs=[precision_pair],
                        active_families=families,
                        all_families=families,
                        active_worker_rows=worker_rows,
                        all_worker_rows=worker_rows,
                        plan_sha256=plan_sha256,
                        worker_set_sha256=worker_set_sha256,
                        month_ordinal=month_ordinal,
                        month_count=month_count,
                        replica_group_id=None,
                        replica_expected_count=1,
                        aggregation_weight=M1_CONTEXT_WEIGHT,
                        replica_paired_consistency_required=False,
                    )
                )
    return result


def _source_specs(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    m5 = plan["source_bindings"]["m5"]
    rectangles = plan["source_bindings"]["m1_precision_rectangles"]
    return [
        {
            "source_binding_id": M5_BINDING_ID,
            "phase": "BROAD_DISCOVERY_M5",
            "granularity": "M5",
            "pairs": list(m5["pairs"]),
            "months": list(m5["months"]),
            "source_digest_sha256": m5["source_digest_sha256"],
            "corpus_digest_sha256": m5["corpus_digest_sha256"],
        },
        *[
            {
                "source_binding_id": rectangle["rectangle_id"],
                "phase": "PRECISION_M1",
                "granularity": "M1",
                "pairs": list(rectangle["pairs"]),
                "months": list(rectangle["months"]),
                "source_digest_sha256": rectangle["source_digest_sha256"],
                "corpus_digest_sha256": rectangle["corpus_digest_sha256"],
            }
            for rectangle in rectangles
        ],
    ]


def build_long_horizon_stream_schedule(
    plan: Mapping[str, Any],
    *,
    worker_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Expand the plan into every result coordinate and one-stream fan-out job."""

    try:
        sealed_plan = validate_long_horizon_train_plan(plan)
    except DojoLongHorizonPlanError as exc:
        raise DojoLongHorizonScheduleError("long-horizon plan is invalid") from exc
    families = list(sealed_plan["portfolio"]["families"])
    workers = _worker_bindings(worker_bindings, families=families)
    worker_body = {"bindings": workers}
    worker_set_sha256 = canonical_sha256(worker_body)
    plan_sha256 = sealed_plan["plan_sha256"]
    active_worker_variants, active_worker_variants_sha256 = _active_worker_variants(
        workers, families=families
    )
    paths = list(sealed_plan["evaluation"]["intrabar_paths"])
    rectangles = sealed_plan["source_bindings"]["m1_precision_rectangles"]
    overlap_pairs = frozenset(rectangles[0]["pairs"]) & frozenset(
        rectangles[1]["pairs"]
    )
    overlap_months = frozenset(rectangles[0]["months"]) & frozenset(
        rectangles[1]["months"]
    )
    jobs: list[dict[str, Any]] = []
    all_coordinate_ids: list[str] = []
    for source in _source_specs(sealed_plan):
        source_month_count = len(source["months"])
        for month_ordinal, month in enumerate(source["months"]):
            month_from, month_to = _month_bounds(month)
            for path in paths:
                # Source identity deliberately excludes worker, allocator and
                # strategy/plan hashes.  Execution identity remains bound by
                # the coordinate ids and job_sha256 below.
                stream_body = {
                    "source_binding_id": source["source_binding_id"],
                    "month": month,
                    "intrabar_path": path,
                    "source_digest_sha256": source["source_digest_sha256"],
                    "corpus_digest_sha256": source["corpus_digest_sha256"],
                }
                price_stream_id = canonical_sha256(stream_body)
                if source["granularity"] == "M5":
                    coordinates = _m5_coordinates(
                        plan=sealed_plan,
                        source_binding_id=source["source_binding_id"],
                        month=month,
                        intrabar_path=path,
                        price_stream_id=price_stream_id,
                        feed_pairs=source["pairs"],
                        worker_rows=workers,
                        plan_sha256=plan_sha256,
                        worker_set_sha256=worker_set_sha256,
                        month_ordinal=month_ordinal,
                        month_count=source_month_count,
                    )
                else:
                    coordinates = _m1_coordinates(
                        plan=sealed_plan,
                        source_binding_id=source["source_binding_id"],
                        month=month,
                        intrabar_path=path,
                        price_stream_id=price_stream_id,
                        feed_pairs=source["pairs"],
                        worker_rows=workers,
                        plan_sha256=plan_sha256,
                        worker_set_sha256=worker_set_sha256,
                        month_ordinal=month_ordinal,
                        month_count=source_month_count,
                    )
                coordinate_ids = [row["coordinate_id"] for row in coordinates]
                all_coordinate_ids.extend(coordinate_ids)
                job_body = {
                    "plan_sha256": plan_sha256,
                    "worker_set_sha256": worker_set_sha256,
                    "active_worker_variants_sha256": active_worker_variants_sha256,
                    "price_stream_id": price_stream_id,
                    "source_binding_id": source["source_binding_id"],
                    "phase": source["phase"],
                    "granularity": source["granularity"],
                    "month": month,
                    "from_utc": month_from,
                    "to_utc": month_to,
                    "intrabar_path": path,
                    "feed_pairs": list(source["pairs"]),
                    "source_digest_sha256": source["source_digest_sha256"],
                    "corpus_digest_sha256": source["corpus_digest_sha256"],
                    "source_stream_instance_count": 1,
                    "quote_schedule": "PAIR_SYNCHRONIZED_EPOCH_UNION",
                    "fanout_before_any_coordinate_decision": True,
                    "coordinate_may_reopen_or_resort_source": False,
                    "synthetic_quote_count": 0,
                    "runtime_month_source_slice_receipt_required": True,
                    "runtime_batch_chain_receipt_required": True,
                    "coordinate_count": len(coordinates),
                    "coordinates": coordinates,
                    "coordinate_ids_sha256": canonical_sha256(coordinate_ids),
                }
                jobs.append({"job_sha256": canonical_sha256(job_body), **job_body})
    if len(all_coordinate_ids) != len(set(all_coordinate_ids)):
        raise DojoLongHorizonScheduleError("generated coordinate ids are not unique")
    expected = sealed_plan["exact_denominator"]["total_required_result_cell_count"]
    if len(all_coordinate_ids) != expected:
        raise DojoLongHorizonScheduleError(
            "expanded coordinate count differs from the sealed denominator"
        )
    effective_coordinate_count = len(all_coordinate_ids)
    denominator = sealed_plan["exact_denominator"]
    mode_count = len(sealed_plan["evaluation"]["modes"])
    scenario_count = len(sealed_plan["evaluation"]["cost_scenarios"])
    expected_effective = denominator["total_required_result_cell_count"]
    if effective_coordinate_count != expected_effective:
        raise DojoLongHorizonScheduleError(
            "effective weighted denominator differs from the sealed fixed denominator"
        )
    overlap_pair_month_count = len(overlap_pairs) * len(overlap_months)
    m1_dimension_count = mode_count * len(paths) * scenario_count
    body = {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": EVIDENCE_TIER,
        "plan_sha256": sealed_plan["plan_sha256"],
        "worker_set": {
            **worker_body,
            "worker_set_sha256": worker_set_sha256,
            "active_worker_variants": active_worker_variants,
            "active_worker_variants_sha256": active_worker_variants_sha256,
        },
        "execution_contract": _execution_contract(),
        "stream_job_count": len(jobs),
        "result_coordinate_count": len(all_coordinate_ids),
        "expected_result_coordinate_count": expected,
        "effective_weighted_result_coordinate_count": effective_coordinate_count,
        "expected_effective_weighted_result_coordinate_count": expected_effective,
        "m1_context_contract": {
            "overlap_pairs": sorted(overlap_pairs),
            "overlap_months": sorted(overlap_months),
            "overlap_pair_month_count": overlap_pair_month_count,
            "contexts_per_overlap_pair_month": 2,
            "aggregation_weight_per_context": M1_CONTEXT_WEIGHT,
            "raw_overlap_context_coordinate_count": overlap_pair_month_count
            * m1_dimension_count
            * 2,
            "effective_overlap_result_count": overlap_pair_month_count
            * m1_dimension_count
            * 2,
            "paired_terminal_consistency_required": False,
            "reason": "CORE5_AND_FULL28_EXPOSE_DIFFERENT_SYNCHRONIZED_FEED_CONTEXTS",
        },
        "jobs": jobs,
        "all_coordinate_ids_sha256": canonical_sha256(all_coordinate_ids),
        "authority": _authority(),
    }
    return {**body, "schedule_sha256": canonical_sha256(body)}


def validate_long_horizon_stream_schedule(
    schedule: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the schedule and reject even self-rehashed semantic drift."""

    if not isinstance(schedule, Mapping):
        raise DojoLongHorizonScheduleError("schedule must be one JSON object")
    expected_keys = {
        "contract",
        "schema_version",
        "classification",
        "plan_sha256",
        "worker_set",
        "execution_contract",
        "stream_job_count",
        "result_coordinate_count",
        "expected_result_coordinate_count",
        "effective_weighted_result_coordinate_count",
        "expected_effective_weighted_result_coordinate_count",
        "m1_context_contract",
        "jobs",
        "all_coordinate_ids_sha256",
        "authority",
        "schedule_sha256",
    }
    if set(schedule) != expected_keys:
        raise DojoLongHorizonScheduleError("schedule top-level schema is not exact")
    if schedule.get("execution_contract") != _execution_contract():
        raise DojoLongHorizonScheduleError(
            "schedule content or canonical hash drifted from the fixed contract"
        )
    worker_set = schedule.get("worker_set")
    if not isinstance(worker_set, Mapping) or set(worker_set) != {
        "bindings",
        "worker_set_sha256",
        "active_worker_variants",
        "active_worker_variants_sha256",
    }:
        raise DojoLongHorizonScheduleError("schedule worker_set schema is not exact")
    bindings = worker_set["bindings"]
    rebuilt = build_long_horizon_stream_schedule(plan, worker_bindings=bindings)
    if dict(schedule) != rebuilt:
        raise DojoLongHorizonScheduleError(
            "schedule content or canonical hash drifted from the fixed contract"
        )
    return rebuilt


__all__ = [
    "CONTRACT",
    "MAX_WORKERS",
    "MAX_WORKERS_PER_FAMILY",
    "M1_CONTEXT_WEIGHT",
    "DojoLongHorizonScheduleError",
    "build_long_horizon_stream_schedule",
    "validate_long_horizon_stream_schedule",
]

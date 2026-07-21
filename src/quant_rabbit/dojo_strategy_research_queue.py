"""Immutable, content-addressed backlog for new DOJO strategy research.

The queue is deliberately smaller than a strategy generator.  It freezes three
falsifiable designs that do not duplicate the six G2 baseline families and it
provides a pure reservation transition for an external AI trainer.  The
transition accepts only cost-complete TRAIN results that are terminal or
materially changed.  It never reads a holdout, executes a replay, edits a
strategy, calls a model, grants proof, or creates live authority.

Queue and reservation artifacts are canonical-JSON SHA-256 addressed.  Queue
validation is exact: changing prose, a threshold, candidate order, authority,
or a dependency requires a new version instead of resealing V1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_g2_baseline import build_g2_baseline


QUEUE_CONTRACT: Final = "QR_DOJO_STRATEGY_RESEARCH_QUEUE_V1"
TRIGGER_CONTRACT: Final = "QR_DOJO_STRATEGY_RESEARCH_TRIGGER_V1"
STATE_CONTRACT: Final = "QR_DOJO_STRATEGY_RESEARCH_RESERVATION_STATE_V1"
RESERVATION_CONTRACT: Final = "QR_DOJO_STRATEGY_RESEARCH_RESERVATION_V1"
DECISION_CONTRACT: Final = "QR_DOJO_STRATEGY_RESEARCH_DECISION_V1"
SCHEMA_VERSION: Final = 1
QUEUE_ID: Final = "dojo-new-strategy-research-v1"
STATUS: Final = "DESIGN_BACKLOG_NOT_EXECUTED"
EVIDENCE_CLASS: Final = "WORN_HISTORICAL_TRAIN_ONLY"
SELECTION_BASIS: Final = "ROBUST_POST_COST_EVIDENCE_ONLY"
ROOM_ISOLATION_CONTRACT: Final = "QR_DOJO_STRATEGY_RESEARCH_ROOM_ISOLATION_V1"
QUEUE_ROOM_BINDING_CONTRACT: Final = "QR_DOJO_STRATEGY_QUEUE_ROOM_BINDING_V1"

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_ZERO_SHA256: Final = "0" * 64
_TRIGGER_KINDS: Final = frozenset({"TERMINAL_RESULT", "MATERIAL_RESULT_CHANGE"})

_AUTHORITY: Final = {
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
    "automatic_deployment_allowed": False,
    "trainer_may_change_live_configuration": False,
}


class DojoStrategyResearchQueueError(ValueError):
    """The research queue or a reservation transition is not admissible."""


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of strict canonical JSON."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoStrategyResearchQueueError(
            "value is not strict canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _strict_copy(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoStrategyResearchQueueError(f"{label} must be a JSON object")
    try:
        return json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise DojoStrategyResearchQueueError(
            f"{label} must contain strict JSON values"
        ) from exc


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256_RE.fullmatch(value) is None
        or value == _ZERO_SHA256
    ):
        raise DojoStrategyResearchQueueError(
            f"{label} must be a non-zero lowercase SHA-256"
        )
    return value


def _candidate(body: Mapping[str, Any]) -> dict[str, Any]:
    copied = _strict_copy(body, "candidate")
    return {**copied, "candidate_sha256": canonical_sha256(copied)}


def _minimum_train(
    *,
    chronological_months: int,
    distinct_years: int,
    observation_name: str,
    observation_count: int,
    active_months: int,
    pair_count: int,
    rationale: str,
) -> dict[str, Any]:
    # These are workload/coverage floors for hypothesis rejection, not risk or
    # promotion knobs.  They deliberately span several seasonal regimes; a
    # later empirical power analysis must replace them in a new queue version.
    return {
        "evidence_class": EVIDENCE_CLASS,
        "chronological_months_min": chronological_months,
        "distinct_calendar_years_min": distinct_years,
        "active_months_min": active_months,
        "pair_count_min": pair_count,
        "observation_floor": {
            "name": observation_name,
            "count_min": observation_count,
        },
        "both_intrabar_paths_required": True,
        "base_and_stress_cost_arms_required": True,
        "fixed_denominator_required": True,
        "zero_trade_is_failure": True,
        "coverage_floor_rationale": rationale,
        "train_is_hypothesis_generation_not_edge_proof": True,
    }


def _independent_window_requirement() -> dict[str, Any]:
    return {
        "required": True,
        "next_window_role": "SEPARATELY_PREREGISTERED_INDEPENDENT_TRAIN_DIAGNOSTIC",
        "chronological_non_overlap_required": True,
        "candidate_and_evaluator_sealed_before_window_read": True,
        "burn_registry_check_required": True,
        "global_untouched_holdout_claim_allowed": False,
        "historical_holdout_access_allowed": False,
        "prospective_data_access_allowed": False,
        "opened_window_must_be_burned_after_one_evaluation": True,
        "reason": (
            "The available 2024-2026 historical corpus is worn; independence is "
            "lineage-local TRAIN discipline and must not be relabeled as proof."
        ),
    }


def _economics_requirements(*, lopo_unit: str) -> dict[str, Any]:
    return {
        "recorded_bid_ask_required": True,
        "base_and_stress_cost_arms_required": True,
        "slippage_on_every_adverse_fill_path_required": True,
        "financing_through_forced_close_required": True,
        "continuous_account_mtm_after_each_action_required": True,
        "terminal_open_exposure_mtm_or_forced_close_required": True,
        "hedge_net_margin_replay_required": True,
        "margin_reject_and_closeout_events_required": True,
        "ruin_and_drawdown_distribution_required": True,
        "lopo_required": True,
        "lopo_unit": lopo_unit,
        "positive_main_and_every_lopo_required_for_survival": True,
        "missing_cost_mtm_margin_or_lopo_is_terminal_reject": True,
    }


def _room_isolation(
    *,
    room_id: str,
    strategy_identity: str,
    input_class: str,
) -> dict[str, Any]:
    room_slug = f"{room_id}-{strategy_identity}"
    return {
        "contract": ROOM_ISOLATION_CONTRACT,
        "dojo_room_id": room_id,
        "strategy_identity": strategy_identity,
        "input_class": input_class,
        "trainer_lineage_id": f"dojo-{room_slug}-lineage-v1",
        "search_budget": {
            "budget_id": f"dojo-{room_slug}-search-budget-v1",
            "scope": "ROOM_ONLY",
            # Four proposal slots are a fixed multiple-testing cap for this V1
            # room, not a market/risk parameter.  A later power or search design
            # must create a new contract rather than silently widening it.
            "proposal_slots_max": 4,
            "proposal_slots_consumed": 0,
            "borrow_from_other_rooms_allowed": False,
            "failed_or_rejected_slot_refunded": False,
            "budget_change_requires_new_queue_version": True,
        },
        "artifact_root": (f"research/strategy_rooms/{QUEUE_ID}/train/{room_slug}"),
        "room_private_fields": [
            "strategy",
            "thesis",
            "input_class",
            "trainer_lineage",
            "search_budget",
            "artifact_root",
            "parameters",
            "results",
        ],
        "shared_contracts_only": [
            "EVALUATOR_CONTRACT",
            "COST_CONTRACT",
            "RISK_CONTRACT",
            "SOURCE_CONTRACT",
        ],
    }


def _candidate_bodies() -> tuple[dict[str, Any], ...]:
    independent = _independent_window_requirement()
    return (
        {
            "candidate_id": "asia_sweep_reclaim_be",
            "family": "asia_sweep_reclaim_be",
            "dojo_room_id": "room-01",
            "design_kind": "SESSION_LIQUIDITY_RECLAIM_WITH_CONDITIONAL_BREAKEVEN",
            "status": STATUS,
            "hypothesis": (
                "A completed Asia-session range sweep that closes back inside the "
                "range has post-cost reversal edge when entered only on the next "
                "executable quote and protected by a one-way break-even transition."
            ),
            "causal_inputs": [
                "completed M5 bid/ask candles available before the decision epoch",
                "a predeclared DST-aware session calendar and completed Asia range",
                "entry-time ATR computed only from completed bars",
                "the next executable bid/ask quote after close-confirmed reclaim",
                "post-entry MFE and completed bars only for the break-even transition",
            ],
            "forbidden_inputs": [
                "the decision bar's not-yet-observed high low or close",
                "a later session range or later reclaim outcome",
                "future spread fill MFE MAE TP SL or answer labels",
            ],
            "novelty": {
                "not_a_g2_family": True,
                "distinguishing_mechanism": (
                    "session-bounded liquidity sweep plus close-confirmed reclaim and "
                    "stateful break-even; not an unconditional spike, previous-day, "
                    "round-number, pullback, burst, or 24-hour mean-reversion signal"
                ),
                "comparison_controls": [
                    "same entry with FIXED exit",
                    "session-matched no-sweep control",
                ],
            },
            "implementation_dependencies": [
                "DST-aware Asia session range detector over completed bars",
                "next-quote reclaim entry vehicle in the economic runner",
                "BREAKEVEN exit overlay that never loosens an existing stop",
                "compact continuous-MTM transcript and independent scorer",
            ],
            "room_isolation": _room_isolation(
                room_id="room-01",
                strategy_identity="asia_sweep_reclaim_be",
                input_class="SESSION_RANGE_SWEEP_RECLAIM_CAUSAL_M5",
            ),
            "falsification_conditions": [
                "worst intrabar path is non-positive after STRESS costs",
                "the reclaim arm does not beat its session-matched no-sweep control",
                "BREAKEVEN is non-positive and does not improve the FIXED downside tail",
                "any required continuous MTM margin or pair-LOPO result is missing",
                "any pair-LOPO terminal result is non-positive",
            ],
            "minimum_train": _minimum_train(
                chronological_months=24,
                distinct_years=3,
                observation_name="closed_trades",
                observation_count=250,
                active_months=18,
                pair_count=8,
                rationale=(
                    "Two calendar years plus a third year boundary and 18 active "
                    "months expose session and volatility seasonality before redesign."
                ),
            ),
            "independent_window_requirement": independent,
            "economics_requirements": _economics_requirements(lopo_unit="PAIR"),
            "authority": dict(_AUTHORITY),
        },
        {
            "candidate_id": "h1_donchian_break_atr_trailing",
            "family": "h1_donchian_break_atr_trailing",
            "dojo_room_id": "room-02",
            "design_kind": "CLOSE_CONFIRMED_H1_BREAKOUT_WITH_ATR_TRAILING",
            "status": STATUS,
            "hypothesis": (
                "A close-confirmed H1 Donchian breakout has a fat-tail continuation "
                "edge when entered on the next executable quote and exited by a "
                "completed-bar ATR trail that never widens."
            ),
            "causal_inputs": [
                "completed H1 bid/ask candles and a predeclared Donchian lookback",
                "completed H1 ATR history available at signal close",
                "the first executable M5 or finer bid/ask quote after the H1 close",
                "completed post-entry bars for one-way ATR trailing updates",
            ],
            "forbidden_inputs": [
                "the breaking H1 bar before its close",
                "same-bar post-entry extremes when entry occurs inside that bar",
                "future ATR peak trough spread or breakout follow-through labels",
            ],
            "novelty": {
                "not_a_g2_family": True,
                "distinguishing_mechanism": (
                    "multi-hour channel close confirmation plus stateful volatility "
                    "trail; unlike the G2 short-horizon burst family and all fade arms"
                ),
                "comparison_controls": [
                    "same entry with FIXED exit",
                    "same entry with BREAKEVEN exit",
                ],
            },
            "implementation_dependencies": [
                "causal H1 aggregation with exact close boundaries",
                "versioned Donchian close-break detector",
                "ATR_TRAILING overlay updated only from completed bars",
                "gap-through stop and next-quote execution accounting",
                "compact continuous-MTM transcript and independent scorer",
            ],
            "room_isolation": _room_isolation(
                room_id="room-02",
                strategy_identity="h1_donchian_break_atr_trailing",
                input_class="CLOSE_CONFIRMED_DONCHIAN_CAUSAL_H1",
            ),
            "falsification_conditions": [
                "worst intrabar path is non-positive after STRESS costs",
                "ATR_TRAILING is non-positive and fails to improve FIXED tail loss",
                "breakout profits are concentrated in one pair or one calendar regime",
                "any required continuous MTM margin or pair-LOPO result is missing",
                "any pair-LOPO terminal result is non-positive",
            ],
            "minimum_train": _minimum_train(
                chronological_months=36,
                distinct_years=4,
                observation_name="closed_trades",
                observation_count=200,
                active_months=24,
                pair_count=8,
                rationale=(
                    "Slow H1 breakouts need three years, four year labels, and at "
                    "least 24 active months to expose trend and range regimes."
                ),
            ),
            "independent_window_requirement": independent,
            "economics_requirements": _economics_requirements(lopo_unit="PAIR"),
            "authority": dict(_AUTHORITY),
        },
        {
            "candidate_id": "g8_relative_strength_risk_budget",
            "family": "g8_relative_strength_risk_budget",
            "dojo_room_id": "room-03",
            "design_kind": "CROSS_SECTIONAL_G8_STRENGTH_WITH_PORTFOLIO_RISK_BUDGET",
            "status": STATUS,
            "hypothesis": (
                "Closed-H1 cross-sectional G8 currency strength can allocate a fixed "
                "portfolio risk budget across independent strong-versus-weak legs "
                "with better post-cost capital productivity than isolated signals."
            ),
            "causal_inputs": [
                "a synchronized exact-28-pair completed-H1 bid/ask matrix",
                "currency strength derived only from returns ending before rebalance",
                "pre-rebalance spread ATR margin and currency-exposure state",
                "trailing realized covariance from completed returns only",
                "the next executable bid/ask quote after a fixed rebalance epoch",
            ],
            "forbidden_inputs": [
                "unsynchronized later quotes from another pair at the same epoch",
                "future cross-sectional ranks covariance returns or fills",
                "netting away gross margin stop risk or shared currency exposure",
            ],
            "novelty": {
                "not_a_g2_family": True,
                "distinguishing_mechanism": (
                    "portfolio-level cross-sectional currency ranking and shared "
                    "risk-budget allocation rather than a single-pair entry pattern"
                ),
                "comparison_controls": [
                    "equal-risk selected-leg allocation",
                    "cash/no-trade control",
                ],
            },
            "implementation_dependencies": [
                "causal synchronized exact-28 H1 feature matrix",
                "currency-level strength decomposition without duplicate pair votes",
                "order-independent gross risk margin and currency-shadow allocator",
                "dependence-aware slot and per-currency exposure limits",
                "compact portfolio MTM transcript and currency-LOPO scorer",
            ],
            "room_isolation": _room_isolation(
                room_id="room-03",
                strategy_identity="g8_relative_strength_risk_budget",
                input_class="SYNCHRONIZED_EXACT28_CROSS_SECTIONAL_CAUSAL_H1",
            ),
            "falsification_conditions": [
                "worst intrabar path is non-positive after STRESS costs",
                "risk-budget allocation does not beat equal-risk post-cost productivity",
                "positive return depends on one currency or one rebalance regime",
                "gross margin risk or shared-currency exposure is understated",
                "any required continuous MTM margin or currency-LOPO result is missing",
                "any currency-LOPO terminal result is non-positive",
            ],
            "minimum_train": _minimum_train(
                chronological_months=36,
                distinct_years=4,
                observation_name="portfolio_rebalance_decisions",
                observation_count=300,
                active_months=24,
                pair_count=28,
                rationale=(
                    "Cross-sectional allocation requires the complete G8 graph and "
                    "several trend, carry, and correlation regimes before redesign."
                ),
            ),
            "independent_window_requirement": independent,
            "economics_requirements": _economics_requirements(lopo_unit="CURRENCY"),
            "authority": dict(_AUTHORITY),
        },
    )


def build_research_queue() -> dict[str, Any]:
    """Build the one exact immutable V1 research queue."""

    g2 = build_g2_baseline()
    g2_families = [str(worker["family"]) for worker in g2["workers"]]
    candidates = [_candidate(body) for body in _candidate_bodies()]
    families = [str(candidate["family"]) for candidate in candidates]
    if len(families) != len(set(families)):
        raise DojoStrategyResearchQueueError("candidate families must be unique")
    overlap = sorted(set(families) & set(g2_families))
    if overlap:
        raise DojoStrategyResearchQueueError(
            f"research queue duplicates G2 families: {overlap}"
        )

    body = {
        "contract": QUEUE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "queue_id": QUEUE_ID,
        "status": STATUS,
        "baseline_binding": {
            "contract": g2["contract"],
            "artifact_sha256": g2["artifact_sha256"],
            "existing_worker_ids": [worker["worker_id"] for worker in g2["workers"]],
            "existing_families": g2_families,
            "duplicate_family_admission_allowed": False,
        },
        "room_isolation_contract": {
            "contract": ROOM_ISOLATION_CONTRACT,
            "room_ids": [candidate["dojo_room_id"] for candidate in candidates],
            "room_private_fields": [
                "strategy",
                "thesis",
                "input_class",
                "trainer_lineage",
                "search_budget",
                "artifact_root",
                "parameters",
                "results",
            ],
            "shared_contracts_only": [
                "EVALUATOR_CONTRACT",
                "COST_CONTRACT",
                "RISK_CONTRACT",
                "SOURCE_CONTRACT",
            ],
            "cross_room_result_or_parameter_copy_policy": {
                "copy_is_new_hypothesis": True,
                "new_candidate_id_required": True,
                "new_candidate_sha256_required": True,
                "destination_room_budget_debit_required": True,
                "source_room_result_inheritance_allowed": False,
                "silent_parameter_inheritance_allowed": False,
            },
            "common_sparring_arena": {
                "arena_id": "dojo-common-sparring-arena-v1",
                "role": "WORN_HISTORICAL_TRAIN_PORTFOLIO_COMPARISON_ONLY",
                "artifact_root": (
                    f"research/strategy_rooms/{QUEUE_ID}/common-sparring-arena"
                ),
                "admitted_room_ids": [
                    candidate["dojo_room_id"] for candidate in candidates
                ],
                # Four shared slots express the requested comparison topology;
                # they are not permission to reverse-size toward the 3x target.
                "shared_capital_slots": 4,
                "max_concurrent_positions_per_room": 1,
                "allocator_and_slot_order_presealed_required": True,
                "same_evaluator_cost_risk_source_contracts_required": True,
                "room_owned_positions_and_results_remain_isolated": True,
                "target_multiple_backsolve_allowed": False,
                "proof_eligible": False,
            },
            "holdout_examination_arena": {
                "arena_id": "dojo-holdout-examination-arena-v1",
                "artifact_root": (
                    f"research/strategy_rooms/{QUEUE_ID}/holdout-examination-arena"
                ),
                "separate_from_train_and_sparring": True,
                "trainer_access_allowed": False,
                "currently_open": False,
                "admission_requires_new_preregistration": True,
                "live_permission": False,
            },
            "prospective_forward_arena": {
                "arena_id": "dojo-prospective-forward-arena-v1",
                "artifact_root": (
                    f"research/strategy_rooms/{QUEUE_ID}/prospective-forward-arena"
                ),
                "separate_from_train_sparring_and_holdout": True,
                "trainer_access_allowed": False,
                "currently_open": False,
                "fixed_candidate_and_evaluator_required_before_start": True,
                "live_permission": False,
            },
        },
        "reservation_policy": {
            "reservation_actor": "EXTERNAL_AI_TRAINER_SUPERVISOR",
            "quant_rabbit_model_api_calls_allowed": False,
            "eligible_trigger_kinds": sorted(_TRIGGER_KINDS),
            "unchanged_semantic_result_action": "NO_OP_RESULT_UNCHANGED",
            "max_active_reservations": 1,
            "max_new_reservations_per_trigger": 1,
            "reservation_order": [
                candidate["candidate_id"] for candidate in candidates
            ],
            "holdout_opening_allowed": False,
            "prospective_window_opening_allowed": False,
            "target_multiple_backsolve_allowed": False,
            "selection_basis": SELECTION_BASIS,
            "terminal_economics_complete_required": True,
            "reservation_is_execution_permission": False,
            "reservation_is_proof_or_promotion": False,
        },
        "candidates": candidates,
        "authority": dict(_AUTHORITY),
    }
    return {**body, "artifact_sha256": canonical_sha256(body)}


def validate_research_queue(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require exact V1 queue content and every content-address binding."""

    candidate = _strict_copy(value, "research queue")
    digest = _require_sha256(candidate.get("artifact_sha256"), "artifact_sha256")
    body = {key: item for key, item in candidate.items() if key != "artifact_sha256"}
    if canonical_sha256(body) != digest:
        raise DojoStrategyResearchQueueError("queue artifact SHA-256 mismatch")

    rows = candidate.get("candidates")
    if not isinstance(rows, list) or len(rows) != 3:
        raise DojoStrategyResearchQueueError(
            "V1 queue must contain exactly three candidates"
        )
    g2_families = {str(worker["family"]) for worker in build_g2_baseline()["workers"]}
    seen_ids: set[str] = set()
    seen_families: set[str] = set()
    seen_rooms: set[str] = set()
    seen_room_roots: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DojoStrategyResearchQueueError(
                f"candidates[{index}] must be an object"
            )
        row_copy = _strict_copy(row, f"candidates[{index}]")
        row_digest = _require_sha256(
            row_copy.get("candidate_sha256"),
            f"candidates[{index}].candidate_sha256",
        )
        row_body = {
            key: item for key, item in row_copy.items() if key != "candidate_sha256"
        }
        if canonical_sha256(row_body) != row_digest:
            raise DojoStrategyResearchQueueError(
                f"candidates[{index}] SHA-256 mismatch"
            )
        candidate_id = row_copy.get("candidate_id")
        family = row_copy.get("family")
        room_id = row_copy.get("dojo_room_id")
        if not isinstance(candidate_id, str) or candidate_id in seen_ids:
            raise DojoStrategyResearchQueueError("candidate IDs must be unique strings")
        if not isinstance(family, str) or family in seen_families:
            raise DojoStrategyResearchQueueError(
                "candidate families must be unique strings"
            )
        if family in g2_families:
            raise DojoStrategyResearchQueueError(
                f"candidate family duplicates G2: {family}"
            )
        if not isinstance(room_id, str) or room_id in seen_rooms:
            raise DojoStrategyResearchQueueError(
                "candidate dojo_room_id values must be unique strings"
            )
        room = row_copy.get("room_isolation")
        if not isinstance(room, Mapping):
            raise DojoStrategyResearchQueueError("candidate room isolation is missing")
        if (
            room.get("contract") != ROOM_ISOLATION_CONTRACT
            or room.get("dojo_room_id") != room_id
            or room.get("strategy_identity") != candidate_id
        ):
            raise DojoStrategyResearchQueueError(
                "candidate room identity binding mismatch"
            )
        artifact_root = room.get("artifact_root")
        if not isinstance(artifact_root, str) or artifact_root in seen_room_roots:
            raise DojoStrategyResearchQueueError(
                "candidate room artifact roots must be unique strings"
            )
        budget = room.get("search_budget")
        if not isinstance(budget, Mapping) or budget.get("scope") != "ROOM_ONLY":
            raise DojoStrategyResearchQueueError(
                "candidate search budget is not room-local"
            )
        if room.get("shared_contracts_only") != [
            "EVALUATOR_CONTRACT",
            "COST_CONTRACT",
            "RISK_CONTRACT",
            "SOURCE_CONTRACT",
        ]:
            raise DojoStrategyResearchQueueError(
                "candidate room shares forbidden state"
            )
        seen_ids.add(candidate_id)
        seen_families.add(family)
        seen_rooms.add(room_id)
        seen_room_roots.add(artifact_root)

    isolation = candidate.get("room_isolation_contract")
    if not isinstance(isolation, Mapping):
        raise DojoStrategyResearchQueueError("room isolation contract is missing")
    if isolation.get("room_ids") != ["room-01", "room-02", "room-03"]:
        raise DojoStrategyResearchQueueError("V1 room order or denominator changed")
    common = isolation.get("common_sparring_arena")
    holdout = isolation.get("holdout_examination_arena")
    forward = isolation.get("prospective_forward_arena")
    if not all(isinstance(value, Mapping) for value in (common, holdout, forward)):
        raise DojoStrategyResearchQueueError("research arenas are incomplete")
    if common.get("shared_capital_slots") != 4:
        raise DojoStrategyResearchQueueError(
            "common arena must keep four capital slots"
        )
    arena_roots = {
        common.get("artifact_root"),
        holdout.get("artifact_root"),
        forward.get("artifact_root"),
    }
    if None in arena_roots or len(arena_roots) != 3 or arena_roots & seen_room_roots:
        raise DojoStrategyResearchQueueError(
            "room and arena artifact roots must be separate"
        )

    expected = build_research_queue()
    if candidate != expected:
        raise DojoStrategyResearchQueueError(
            "research queue differs from the exact immutable V1 contract"
        )
    return expected


def resolve_queue_room_binding(
    value: Mapping[str, Any], *, dojo_room_id: str
) -> dict[str, Any]:
    """Return the one canonical queue candidate bound to ``dojo_room_id``.

    Training-room receipts use this small projection instead of trusting a
    caller-supplied candidate name or family.  The complete immutable queue is
    validated first, so the projection is anchored to both the queue artifact
    and the candidate content address.
    """

    queue = validate_research_queue(value)
    if not isinstance(dojo_room_id, str) or not dojo_room_id:
        raise DojoStrategyResearchQueueError("dojo_room_id must be a string")
    matches = [
        candidate
        for candidate in queue["candidates"]
        if candidate["dojo_room_id"] == dojo_room_id
    ]
    if len(matches) != 1:
        raise DojoStrategyResearchQueueError(
            "dojo room must bind exactly one canonical queue candidate"
        )
    candidate = matches[0]
    return {
        "contract": QUEUE_ROOM_BINDING_CONTRACT,
        "queue_contract": queue["contract"],
        "queue_id": queue["queue_id"],
        "queue_artifact_sha256": queue["artifact_sha256"],
        "dojo_room_id": dojo_room_id,
        "canonical_candidate_id": candidate["candidate_id"],
        "canonical_candidate_sha256": candidate["candidate_sha256"],
        "canonical_family": candidate["family"],
    }


def registry_relative_path() -> str:
    """Return the filename bound to the canonical queue artifact digest."""

    digest = build_research_queue()["artifact_sha256"]
    return f"research/registries/dojo_strategy_research_queue_v1-{digest}.json"


def load_research_queue(path: Path) -> dict[str, Any]:
    """Load V1 and require its filename to match the artifact content address."""

    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise DojoStrategyResearchQueueError(
            f"cannot read research queue: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise DojoStrategyResearchQueueError("research queue JSON must be an object")
    queue = validate_research_queue(value)
    expected_name = Path(registry_relative_path()).name
    if Path(path).name != expected_name:
        raise DojoStrategyResearchQueueError(
            "research queue filename does not match its artifact SHA-256"
        )
    return queue


def _state_body(
    *,
    queue_sha256: str,
    completed_candidate_ids: Sequence[str],
    active_reservation: Mapping[str, Any] | None,
    last_result_artifact_sha256: str | None,
    last_semantic_result_sha256: str | None,
    reservation_count: int,
) -> dict[str, Any]:
    return {
        "contract": STATE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "queue_artifact_sha256": queue_sha256,
        "completed_candidate_ids": list(completed_candidate_ids),
        "active_reservation": (
            None
            if active_reservation is None
            else _strict_copy(active_reservation, "active reservation")
        ),
        "last_result_artifact_sha256": last_result_artifact_sha256,
        "last_semantic_result_sha256": last_semantic_result_sha256,
        # This is a monotonic queue sequence, not a market or sizing parameter.
        "reservation_count": reservation_count,
        "authority": dict(_AUTHORITY),
    }


def _seal_state(body: Mapping[str, Any]) -> dict[str, Any]:
    copied = _strict_copy(body, "reservation state body")
    return {**copied, "state_sha256": canonical_sha256(copied)}


def build_initial_reservation_state(
    queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic empty state for the exact V1 queue."""

    validated = validate_research_queue(
        build_research_queue() if queue is None else queue
    )
    return _seal_state(
        _state_body(
            queue_sha256=validated["artifact_sha256"],
            completed_candidate_ids=[],
            active_reservation=None,
            last_result_artifact_sha256=None,
            last_semantic_result_sha256=None,
            reservation_count=0,
        )
    )


def _validate_reservation(
    value: Mapping[str, Any], *, queue: Mapping[str, Any], expected_candidate_id: str
) -> dict[str, Any]:
    reservation = _strict_copy(value, "active reservation")
    expected_keys = {
        "contract",
        "schema_version",
        "queue_artifact_sha256",
        "candidate_id",
        "dojo_room_id",
        "candidate_sha256",
        "trigger_kind",
        "result_artifact_sha256",
        "semantic_result_sha256",
        "previous_state_sha256",
        "selection_basis",
        "authority",
        "reservation_sha256",
    }
    if set(reservation) != expected_keys:
        raise DojoStrategyResearchQueueError("active reservation schema mismatch")
    digest = _require_sha256(
        reservation.get("reservation_sha256"), "reservation_sha256"
    )
    body = {
        key: item for key, item in reservation.items() if key != "reservation_sha256"
    }
    if canonical_sha256(body) != digest:
        raise DojoStrategyResearchQueueError("active reservation SHA-256 mismatch")
    if (
        reservation["contract"] != RESERVATION_CONTRACT
        or reservation["schema_version"] != 1
    ):
        raise DojoStrategyResearchQueueError("active reservation contract mismatch")
    if reservation["queue_artifact_sha256"] != queue["artifact_sha256"]:
        raise DojoStrategyResearchQueueError(
            "active reservation queue binding mismatch"
        )
    if reservation["candidate_id"] != expected_candidate_id:
        raise DojoStrategyResearchQueueError(
            "active reservation is not the next queue candidate"
        )
    candidate = next(
        row
        for row in queue["candidates"]
        if row["candidate_id"] == expected_candidate_id
    )
    if reservation["dojo_room_id"] != candidate["dojo_room_id"]:
        raise DojoStrategyResearchQueueError("active reservation room binding mismatch")
    if reservation["candidate_sha256"] != candidate["candidate_sha256"]:
        raise DojoStrategyResearchQueueError(
            "active reservation candidate binding mismatch"
        )
    if reservation["trigger_kind"] not in _TRIGGER_KINDS:
        raise DojoStrategyResearchQueueError(
            "active reservation trigger kind is invalid"
        )
    _require_sha256(reservation["result_artifact_sha256"], "result_artifact_sha256")
    _require_sha256(reservation["semantic_result_sha256"], "semantic_result_sha256")
    _require_sha256(reservation["previous_state_sha256"], "previous_state_sha256")
    if reservation["selection_basis"] != SELECTION_BASIS:
        raise DojoStrategyResearchQueueError(
            "active reservation selection basis mismatch"
        )
    if reservation["authority"] != _AUTHORITY:
        raise DojoStrategyResearchQueueError("active reservation authority mismatch")
    return reservation


def validate_reservation_state(
    value: Mapping[str, Any], queue: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate a content-addressed state and its prefix/active-candidate ordering."""

    validated_queue = validate_research_queue(
        build_research_queue() if queue is None else queue
    )
    state = _strict_copy(value, "reservation state")
    expected_keys = {
        "contract",
        "schema_version",
        "queue_artifact_sha256",
        "completed_candidate_ids",
        "active_reservation",
        "last_result_artifact_sha256",
        "last_semantic_result_sha256",
        "reservation_count",
        "authority",
        "state_sha256",
    }
    if set(state) != expected_keys:
        raise DojoStrategyResearchQueueError("reservation state schema mismatch")
    digest = _require_sha256(state.get("state_sha256"), "state_sha256")
    body = {key: item for key, item in state.items() if key != "state_sha256"}
    if canonical_sha256(body) != digest:
        raise DojoStrategyResearchQueueError("reservation state SHA-256 mismatch")
    if state["contract"] != STATE_CONTRACT or state["schema_version"] != 1:
        raise DojoStrategyResearchQueueError("reservation state contract mismatch")
    if state["queue_artifact_sha256"] != validated_queue["artifact_sha256"]:
        raise DojoStrategyResearchQueueError("reservation state queue binding mismatch")
    if state["authority"] != _AUTHORITY:
        raise DojoStrategyResearchQueueError("reservation state authority mismatch")

    order = [row["candidate_id"] for row in validated_queue["candidates"]]
    completed = state["completed_candidate_ids"]
    if not isinstance(completed, list) or completed != order[: len(completed)]:
        raise DojoStrategyResearchQueueError(
            "completed candidates must be a unique queue-order prefix"
        )
    reservation_count = state["reservation_count"]
    if (
        not isinstance(reservation_count, int)
        or isinstance(reservation_count, bool)
        or reservation_count < 0
    ):
        raise DojoStrategyResearchQueueError(
            "reservation_count must be non-negative int"
        )
    active = state["active_reservation"]
    expected_count = len(completed) + (1 if active is not None else 0)
    if reservation_count != expected_count:
        raise DojoStrategyResearchQueueError(
            "reservation_count does not match queue state"
        )
    if active is not None:
        if len(completed) >= len(order) or not isinstance(active, Mapping):
            raise DojoStrategyResearchQueueError(
                "active reservation cannot follow exhaustion"
            )
        state["active_reservation"] = _validate_reservation(
            active,
            queue=validated_queue,
            expected_candidate_id=order[len(completed)],
        )
    for key in ("last_result_artifact_sha256", "last_semantic_result_sha256"):
        value_or_none = state[key]
        if value_or_none is not None:
            _require_sha256(value_or_none, key)
    if reservation_count == 0:
        if (
            state["last_result_artifact_sha256"] is not None
            or state["last_semantic_result_sha256"] is not None
        ):
            raise DojoStrategyResearchQueueError(
                "empty state cannot claim a prior result"
            )
    elif (
        state["last_result_artifact_sha256"] is None
        or state["last_semantic_result_sha256"] is None
    ):
        raise DojoStrategyResearchQueueError(
            "non-empty state must bind its last result"
        )
    return state


def validate_trigger(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one terminal/material TRAIN result trigger without opening holdout."""

    trigger = _strict_copy(value, "research trigger")
    expected_keys = {
        "contract",
        "schema_version",
        "trigger_kind",
        "result_artifact_sha256",
        "semantic_result_sha256",
        "result_candidate_id",
        "source_partition",
        "evidence_class",
        "terminal_result",
        "material_change",
        "holdout_opened",
        "prospective_window_opened",
        "global_untouched_holdout_claimed",
        "target_multiple_backsolve_used",
        "selection_basis",
        "economics",
    }
    if set(trigger) != expected_keys:
        raise DojoStrategyResearchQueueError("research trigger schema mismatch")
    if trigger["contract"] != TRIGGER_CONTRACT or trigger["schema_version"] != 1:
        raise DojoStrategyResearchQueueError("research trigger contract mismatch")
    kind = trigger["trigger_kind"]
    if kind not in _TRIGGER_KINDS:
        raise DojoStrategyResearchQueueError("trigger is not terminal or material")
    if kind == "TERMINAL_RESULT":
        if (
            trigger["terminal_result"] is not True
            or trigger["material_change"] is not True
        ):
            raise DojoStrategyResearchQueueError(
                "terminal trigger must be a material terminal result"
            )
    elif (
        trigger["terminal_result"] is not False
        or trigger["material_change"] is not True
    ):
        raise DojoStrategyResearchQueueError("material trigger flags are inconsistent")
    _require_sha256(trigger["result_artifact_sha256"], "result_artifact_sha256")
    _require_sha256(trigger["semantic_result_sha256"], "semantic_result_sha256")
    result_candidate_id = trigger["result_candidate_id"]
    if result_candidate_id is not None and not isinstance(result_candidate_id, str):
        raise DojoStrategyResearchQueueError(
            "result_candidate_id must be string or null"
        )
    if (
        trigger["source_partition"] != "TRAIN"
        or trigger["evidence_class"] != EVIDENCE_CLASS
    ):
        raise DojoStrategyResearchQueueError(
            "only worn historical TRAIN may trigger research"
        )
    if (
        trigger["holdout_opened"] is not False
        or trigger["prospective_window_opened"] is not False
        or trigger["global_untouched_holdout_claimed"] is not False
    ):
        raise DojoStrategyResearchQueueError("holdout/prospective access is forbidden")
    if trigger["target_multiple_backsolve_used"] is not False:
        raise DojoStrategyResearchQueueError("monthly target backsolving is forbidden")
    if trigger["selection_basis"] != SELECTION_BASIS:
        raise DojoStrategyResearchQueueError("trigger selection basis is invalid")
    economics = trigger["economics"]
    required_economics = {
        "recorded_bid_ask_costed": True,
        "slippage_costed": True,
        "financing_costed": True,
        "continuous_mtm_complete": True,
        "margin_replayed": True,
        "lopo_complete": True,
        "fixed_denominator_complete": True,
    }
    if economics != required_economics:
        raise DojoStrategyResearchQueueError(
            "trigger economics must prove cost MTM margin LOPO and denominator completeness"
        )
    return trigger


def _seal_reservation(body: Mapping[str, Any]) -> dict[str, Any]:
    copied = _strict_copy(body, "reservation body")
    return {**copied, "reservation_sha256": canonical_sha256(copied)}


def _seal_decision(body: Mapping[str, Any]) -> dict[str, Any]:
    copied = _strict_copy(body, "decision body")
    return {**copied, "decision_sha256": canonical_sha256(copied)}


def plan_reservation(
    *,
    queue: Mapping[str, Any],
    trigger: Mapping[str, Any],
    previous_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Purely reserve at most one next design for one admissible changed result."""

    validated_queue = validate_research_queue(queue)
    validated_trigger = validate_trigger(trigger)
    state = validate_reservation_state(previous_state, validated_queue)
    semantic_sha = validated_trigger["semantic_result_sha256"]

    if semantic_sha == state["last_semantic_result_sha256"]:
        return _seal_decision(
            {
                "contract": DECISION_CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "action": "NO_OP_RESULT_UNCHANGED",
                "reason_codes": ["SEMANTIC_RESULT_SHA256_UNCHANGED"],
                "queue_artifact_sha256": validated_queue["artifact_sha256"],
                "trigger_result_artifact_sha256": validated_trigger[
                    "result_artifact_sha256"
                ],
                "trigger_semantic_result_sha256": semantic_sha,
                "previous_state_sha256": state["state_sha256"],
                "reservation": None,
                "next_state": state,
                "authority": dict(_AUTHORITY),
            }
        )

    order = [row["candidate_id"] for row in validated_queue["candidates"]]
    completed = list(state["completed_candidate_ids"])
    active = state["active_reservation"]
    result_candidate_id = validated_trigger["result_candidate_id"]
    if active is not None:
        if result_candidate_id != active["candidate_id"]:
            return _seal_decision(
                {
                    "contract": DECISION_CONTRACT,
                    "schema_version": SCHEMA_VERSION,
                    "action": "NO_OP_ACTIVE_RESERVATION_RESULT_UNBOUND",
                    "reason_codes": ["RESULT_DOES_NOT_CLOSE_ACTIVE_RESERVATION"],
                    "queue_artifact_sha256": validated_queue["artifact_sha256"],
                    "trigger_result_artifact_sha256": validated_trigger[
                        "result_artifact_sha256"
                    ],
                    "trigger_semantic_result_sha256": semantic_sha,
                    "previous_state_sha256": state["state_sha256"],
                    "reservation": None,
                    "next_state": state,
                    "authority": dict(_AUTHORITY),
                }
            )
        completed.append(active["candidate_id"])
    elif result_candidate_id is not None:
        raise DojoStrategyResearchQueueError(
            "a result candidate cannot be bound without an active reservation"
        )

    if len(completed) == len(order):
        next_state = _seal_state(
            _state_body(
                queue_sha256=validated_queue["artifact_sha256"],
                completed_candidate_ids=completed,
                active_reservation=None,
                last_result_artifact_sha256=validated_trigger["result_artifact_sha256"],
                last_semantic_result_sha256=semantic_sha,
                reservation_count=state["reservation_count"],
            )
        )
        return _seal_decision(
            {
                "contract": DECISION_CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "action": "QUEUE_EXHAUSTED",
                "reason_codes": ["ALL_V1_DESIGNS_COMPLETED"],
                "queue_artifact_sha256": validated_queue["artifact_sha256"],
                "trigger_result_artifact_sha256": validated_trigger[
                    "result_artifact_sha256"
                ],
                "trigger_semantic_result_sha256": semantic_sha,
                "previous_state_sha256": state["state_sha256"],
                "reservation": None,
                "next_state": next_state,
                "authority": dict(_AUTHORITY),
            }
        )

    next_candidate = validated_queue["candidates"][len(completed)]
    reservation = _seal_reservation(
        {
            "contract": RESERVATION_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "queue_artifact_sha256": validated_queue["artifact_sha256"],
            "candidate_id": next_candidate["candidate_id"],
            "dojo_room_id": next_candidate["dojo_room_id"],
            "candidate_sha256": next_candidate["candidate_sha256"],
            "trigger_kind": validated_trigger["trigger_kind"],
            "result_artifact_sha256": validated_trigger["result_artifact_sha256"],
            "semantic_result_sha256": semantic_sha,
            "previous_state_sha256": state["state_sha256"],
            "selection_basis": SELECTION_BASIS,
            "authority": dict(_AUTHORITY),
        }
    )
    next_state = _seal_state(
        _state_body(
            queue_sha256=validated_queue["artifact_sha256"],
            completed_candidate_ids=completed,
            active_reservation=reservation,
            last_result_artifact_sha256=validated_trigger["result_artifact_sha256"],
            last_semantic_result_sha256=semantic_sha,
            reservation_count=len(completed) + 1,
        )
    )
    validate_reservation_state(next_state, validated_queue)
    return _seal_decision(
        {
            "contract": DECISION_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "action": "RESERVE_ONE_CANDIDATE",
            "reason_codes": ["TERMINAL_OR_MATERIAL_TRAIN_RESULT_CHANGED"],
            "queue_artifact_sha256": validated_queue["artifact_sha256"],
            "trigger_result_artifact_sha256": validated_trigger[
                "result_artifact_sha256"
            ],
            "trigger_semantic_result_sha256": semantic_sha,
            "previous_state_sha256": state["state_sha256"],
            "reservation": reservation,
            "next_state": next_state,
            "authority": dict(_AUTHORITY),
        }
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DojoStrategyResearchQueueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DojoStrategyResearchQueueError(f"{label} must be a JSON object")
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate the immutable queue.")
    validate.add_argument("--registry", type=Path, required=True)
    plan = commands.add_parser(
        "plan-reservation",
        help="Plan one pure reservation transition and print it without writing files.",
    )
    plan.add_argument("--registry", type=Path, required=True)
    plan.add_argument("--trigger", type=Path, required=True)
    plan.add_argument("--state", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only validator/planner CLI."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        queue = load_research_queue(args.registry)
        if args.command == "validate":
            result: dict[str, Any] = {
                "contract": QUEUE_CONTRACT,
                "status": "VALID",
                "artifact_sha256": queue["artifact_sha256"],
                "candidate_count": len(queue["candidates"]),
                "live_permission": False,
                "order_authority": "NONE",
            }
        elif args.command == "plan-reservation":
            state = (
                build_initial_reservation_state(queue)
                if args.state is None
                else _read_json(args.state, "reservation state")
            )
            result = plan_reservation(
                queue=queue,
                trigger=_read_json(args.trigger, "research trigger"),
                previous_state=state,
            )
        else:
            raise DojoStrategyResearchQueueError("unsupported command")
    except DojoStrategyResearchQueueError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


REGISTRY_RELATIVE_PATH: Final = registry_relative_path()


__all__ = [
    "DECISION_CONTRACT",
    "DojoStrategyResearchQueueError",
    "EVIDENCE_CLASS",
    "QUEUE_CONTRACT",
    "REGISTRY_RELATIVE_PATH",
    "ROOM_ISOLATION_CONTRACT",
    "RESERVATION_CONTRACT",
    "SCHEMA_VERSION",
    "SELECTION_BASIS",
    "STATE_CONTRACT",
    "STATUS",
    "TRIGGER_CONTRACT",
    "build_initial_reservation_state",
    "build_research_queue",
    "canonical_sha256",
    "load_research_queue",
    "main",
    "plan_reservation",
    "registry_relative_path",
    "validate_research_queue",
    "validate_reservation_state",
    "validate_trigger",
]


if __name__ == "__main__":
    raise SystemExit(main())

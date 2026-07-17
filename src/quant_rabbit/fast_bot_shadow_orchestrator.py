"""Compact, fail-closed orchestration for the 28-pair technical shadow grid.

The current causal route is the H08 no-trade control.  This module therefore
does not convert any of the 5,096 pair/candidate cells into an order candidate.
It deep-validates each upstream technical shadow by deterministic rebuild,
binds one fresh risk-sizing identity, and records the whole Cartesian product
with a compact run-length representation.  The artifact is evidence plumbing:
it has no broker, promotion, profitability, or risk-allocation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from quant_rabbit.fast_bot_technical_grid_backtest import (
    ORIENTATIONS,
    PLANNED_ARM_COUNT,
    PLANNED_CANDIDATE_COUNT,
    PLANNED_DIRECTIONAL_FAMILIES,
    build_fast_bot_technical_grid_catalog_v1,
)
from quant_rabbit.fast_bot_technical_hypotheses import (
    EVALUATOR_POLICY_V1,
    FEATURE_SNAPSHOT_CONTRACT,
    SHADOW_CONTRACT,
    TIMEFRAMES,
    technical_hypothesis_shadow_valid,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


SHADOW_ADMISSION_BINDING_CONTRACT_V1 = "QR_FAST_BOT_SHADOW_ADMISSION_BINDING_V1"
SHADOW_RISK_SIZING_CONTRACT_V1 = "QR_FAST_BOT_SHADOW_RISK_SIZING_IDENTITY_V1"
SHADOW_ORCHESTRATION_CONTRACT_V1 = "QR_FAST_BOT_SHADOW_ORCHESTRATION_V1"
SHADOW_MATRIX_ENCODING_V1 = "CARTESIAN_PRODUCT_SINGLE_STATE_RLE_V1"
SHADOW_ORCHESTRATION_POLICY_V1 = "EXACT_28_PAIR_X_182_CANDIDATE_H08_PERMANENT_SHADOW_V1"

CURRENT_ROUTE_HYPOTHESIS_ID = "H08"
CURRENT_CELL_STATE = "PERMANENT_SHADOW_H08_NO_TRADE_CONTROL"
EXPECTED_PAIR_COUNT = len(DEFAULT_TRADER_PAIRS)
EXPECTED_MATRIX_CARDINALITY = EXPECTED_PAIR_COUNT * PLANNED_CANDIDATE_COUNT

# This is a serialization/transport bound, not a market or risk decision.  The
# compact matrix should remain far below 128 KiB; replace this with a shared
# repository artifact limit if one is introduced.
MAX_SHADOW_ORCHESTRATION_BYTES = 128 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# The 128-character bound is a receipt-key transport limit, not a trading
# parameter.  Replace it only if the shared receipt identity contract changes.
_CYCLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

_ZERO_AUTHORITY = {
    "historical_only": False,
    "diagnostic_only": True,
    "forward_proof_eligible": False,
    "automatic_promotion_allowed": False,
    "promotion_allowed": False,
    "order_authority": "NONE",
    "primary_effect": False,
    "risk_effect": False,
    "shadow_only": True,
    "live_permission": False,
    "broker_mutation_allowed": False,
}

_ROUTE_ROW_KEYS = frozenset(
    {
        "pair",
        "decision_at_utc",
        "selected_hypothesis_id",
        "selected_state",
        "feature_snapshot_sha256",
        "hypothesis_shadow_sha256",
        "h08_hypothesis_sha256",
        "evaluator_policy",
        "deep_validation",
        "active_hypothesis_ids",
        "h08_evidence",
        "candidate_state",
        "candidate_count",
        "go_count",
        "order_authority",
        "live_permission",
        "broker_mutation_allowed",
        "receipt_sha256",
    }
)
_MATRIX_KEYS = frozenset(
    {
        "encoding",
        "pair_order",
        "candidate_order",
        "candidate_identity_sha256",
        "pair_count",
        "candidate_count_per_pair",
        "matrix_cardinality",
        "flat_index_formula",
        "state_runs",
        "matrix_identity_sha256",
    }
)
_RISK_IDENTITY_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "cycle_id",
        "as_of_utc",
        "sizing_nav_jpy",
        "daily_loss_capacity_before_open_jpy",
        "fresh_open_risk_jpy",
        "pending_risk_jpy",
        "per_trade_risk_cap_jpy",
        "available_capacity_after_open_and_pending_jpy",
        "capacity_accounting_policy",
        "broker_snapshot_sha256",
        "daily_target_state_sha256",
        "execution_ledger_tip_sha256",
        "risk_sizing_identity_sha256",
        "candidate_risk_allocations",
        "go_risk_jpy",
        *_ZERO_AUTHORITY,
        "order_intents",
        "contract_sha256",
    }
)
_ORCHESTRATION_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "orchestration_policy",
        "cycle_id",
        "cycle_started_at_utc",
        "decision_at_utc",
        "observed_at_utc",
        "freshness_deadline_utc",
        "fresh_at_build",
        "freshness_must_be_revalidated_before_use",
        "catalog_contract_sha256",
        "catalog_dimensions",
        "pair_routes",
        "matrix",
        "current_route_hypothesis_id",
        "go_count",
        "go_candidate_refs",
        "future_multiple_go_shape_supported",
        "go_generation_enabled",
        "go_blocker_codes",
        "economic_status",
        "profitability_proven",
        "economic_conclusion_allowed",
        "risk_sizing_identity_sha256",
        "risk_sizing_receipt_sha256",
        "risk_sizing_identity",
        "go_risk_jpy",
        "discarded_candidates_retained_in_shadow",
        "h08_control_retained_per_pair",
        *_ZERO_AUTHORITY,
        "order_intents",
        "contract_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class PairCausalShadowInput:
    """One upstream H01..H08 evaluation plus every input needed to rebuild it."""

    feature_snapshot: Mapping[str, Any]
    hypothesis_shadow: Mapping[str, Any]
    attempt_direction: str
    branch_outcome: str
    route_family: str
    spread_pips: float
    m5_atr_pips: float
    spread_to_m5_atr: float


def build_fast_bot_shadow_admission_binding_v1(
    *,
    cycle_id: str,
    decision_utc: datetime,
    hold_minutes: int,
    open_positions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    currency_cap_fraction: float = 0.5,
) -> dict[str, Any]:
    """Bind the pre-entry admission gates into the future GO contract shape.

    Seals, per cycle, the causal close-distance gate, the pre-declared
    high-cost window mask, and per-candidate currency-exposure verdicts so
    a future GO route must present this artifact.  It admits or refuses
    hypothetical candidates only; GO stays 0 and order intents stay empty.
    """

    from quant_rabbit.close_distance_gate import evaluate_close_distance_gate
    from quant_rabbit.cost_window_mask import evaluate_cost_window
    from quant_rabbit.currency_exposure_guard import evaluate_currency_exposure

    cycle = _validated_cycle_id(cycle_id)
    decision = _aware_utc(decision_utc, "decision_utc")
    close_decision = evaluate_close_distance_gate(
        decision, hold_minutes=hold_minutes
    )
    cost_decision = evaluate_cost_window(decision)
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        exposure = evaluate_currency_exposure(
            open_positions,
            candidate,
            currency_cap_fraction=currency_cap_fraction,
        )
        admitted = bool(
            close_decision.admitted and cost_decision.admitted and exposure.admitted
        )
        candidate_rows.append(
            {
                "pair": str(candidate.get("pair")),
                "side": str(candidate.get("side")),
                "admitted": admitted,
                "refusal_reasons": [
                    reason
                    for reason, ok in (
                        (close_decision.reason, close_decision.admitted),
                        (cost_decision.reason, cost_decision.admitted),
                        (exposure.reason, exposure.admitted),
                    )
                    if not ok
                ],
                "exposure": exposure.payload(),
            }
        )
    body = {
        "contract": SHADOW_ADMISSION_BINDING_CONTRACT_V1,
        "schema_version": 1,
        "cycle_id": cycle,
        "decision_utc": decision.isoformat(),
        "close_distance_gate": close_decision.payload(),
        "cost_window_mask": cost_decision.payload(),
        "currency_cap_fraction": float(currency_cap_fraction),
        "candidate_rows": candidate_rows,
        "admitted_candidate_count": sum(
            row["admitted"] for row in candidate_rows
        ),
        "future_go_contract_must_bind_this_artifact": True,
        "go_risk_jpy": 0.0,
        **_zero_authority(),
        "order_intents": [],
    }
    return _seal(body)


def build_fast_bot_shadow_risk_sizing_identity_v1(
    *,
    cycle_id: str,
    as_of_utc: datetime,
    sizing_nav_jpy: float,
    daily_loss_capacity_before_open_jpy: float,
    fresh_open_risk_jpy: float,
    pending_risk_jpy: float,
    per_trade_risk_cap_jpy: float,
    broker_snapshot_sha256: str,
    daily_target_state_sha256: str,
    execution_ledger_tip_sha256: str,
) -> dict[str, Any]:
    """Seal risk inputs without allocating risk to any shadow candidate.

    The capacity identity mirrors the live contract's before-open-risk basis so
    open and pending risk are subtracted exactly once.  It is observability for
    the shadow cycle, never a sizing recommendation or permission to trade.
    """

    cycle = _validated_cycle_id(cycle_id)
    as_of = _aware_utc(as_of_utc, "as_of_utc")
    nav = _finite_number(sizing_nav_jpy, "sizing_nav_jpy", positive=True)
    capacity = _finite_number(
        daily_loss_capacity_before_open_jpy,
        "daily_loss_capacity_before_open_jpy",
        nonnegative=True,
    )
    open_risk = _finite_number(
        fresh_open_risk_jpy,
        "fresh_open_risk_jpy",
        nonnegative=True,
    )
    pending_risk = _finite_number(
        pending_risk_jpy,
        "pending_risk_jpy",
        nonnegative=True,
    )
    per_trade_cap = _finite_number(
        per_trade_risk_cap_jpy,
        "per_trade_risk_cap_jpy",
        nonnegative=True,
    )
    available = max(0.0, capacity - open_risk - pending_risk)
    if per_trade_cap > available:
        raise ValueError("per_trade_risk_cap_jpy exceeds current available capacity")
    identity = {
        "cycle_id": cycle,
        "as_of_utc": as_of.isoformat(),
        "sizing_nav_jpy": _rounded(nav),
        "daily_loss_capacity_before_open_jpy": _rounded(capacity),
        "fresh_open_risk_jpy": _rounded(open_risk),
        "pending_risk_jpy": _rounded(pending_risk),
        "per_trade_risk_cap_jpy": _rounded(per_trade_cap),
        "available_capacity_after_open_and_pending_jpy": _rounded(available),
        "capacity_accounting_policy": (
            "BEFORE_OPEN_CAPACITY_MINUS_FRESH_OPEN_MINUS_PENDING_EXACTLY_ONCE"
        ),
        "broker_snapshot_sha256": _validated_sha(
            broker_snapshot_sha256, "broker_snapshot_sha256"
        ),
        "daily_target_state_sha256": _validated_sha(
            daily_target_state_sha256, "daily_target_state_sha256"
        ),
        "execution_ledger_tip_sha256": _validated_sha(
            execution_ledger_tip_sha256, "execution_ledger_tip_sha256"
        ),
    }
    body = {
        "contract": SHADOW_RISK_SIZING_CONTRACT_V1,
        "schema_version": 1,
        **identity,
        "risk_sizing_identity_sha256": _canonical_sha(identity),
        "candidate_risk_allocations": [],
        "go_risk_jpy": 0.0,
        **_zero_authority(),
        "order_intents": [],
    }
    return _seal(body)


def build_fast_bot_shadow_orchestration_v1(
    *,
    cycle_id: str,
    cycle_started_at_utc: datetime,
    decision_at_utc: datetime,
    freshness_deadline_utc: datetime,
    observed_at_utc: datetime,
    pair_inputs: Sequence[PairCausalShadowInput | Mapping[str, Any]],
    risk_sizing_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Deep-validate and compactly orchestrate all 28 x 182 shadow cells.

    Freshness is explicit and caller-bound: no wall-clock read or stale fallback
    is hidden inside the builder.  Every pair must have an exact H08 ACTIVE_SHADOW
    route.  Any H01..H07 route remains upstream research and cannot be promoted
    by this contract.
    """

    cycle = _validated_cycle_id(cycle_id)
    started = _aware_utc(cycle_started_at_utc, "cycle_started_at_utc")
    decision = _aware_utc(decision_at_utc, "decision_at_utc")
    deadline = _aware_utc(freshness_deadline_utc, "freshness_deadline_utc")
    observed = _aware_utc(observed_at_utc, "observed_at_utc")
    if not started <= decision <= observed < deadline:
        raise ValueError(
            "cycle clocks must satisfy started <= decision <= observed < deadline"
        )
    risk = _validated_risk_sizing_identity(risk_sizing_identity)
    if risk["cycle_id"] != cycle:
        raise ValueError("risk sizing identity belongs to a different cycle")
    risk_as_of = _parse_utc(risk["as_of_utc"], "risk sizing as_of_utc")
    if not decision <= risk_as_of <= observed:
        raise ValueError("risk sizing identity is outside the current cycle window")

    if isinstance(pair_inputs, (str, bytes)) or not isinstance(pair_inputs, Sequence):
        raise ValueError("pair_inputs must be a sequence")
    coerced = [_coerce_pair_input(item) for item in pair_inputs]
    if len(coerced) != EXPECTED_PAIR_COUNT:
        raise ValueError("orchestration requires exactly all 28 trader pairs")

    by_pair: dict[str, PairCausalShadowInput] = {}
    for item in coerced:
        pair = str(item.feature_snapshot.get("pair") or "")
        if pair not in DEFAULT_TRADER_PAIRS or pair in by_pair:
            raise ValueError("pair inputs are unknown, duplicated, or incomplete")
        by_pair[pair] = item
    if tuple(pair for pair in DEFAULT_TRADER_PAIRS if pair in by_pair) != tuple(
        DEFAULT_TRADER_PAIRS
    ):
        raise ValueError("pair inputs do not cover the canonical trader-pair order")

    route_rows = [
        _deep_validated_h08_route(
            pair=pair,
            item=by_pair[pair],
            decision=decision,
        )
        for pair in DEFAULT_TRADER_PAIRS
    ]

    catalog = build_fast_bot_technical_grid_catalog_v1()
    if catalog.get("planned_candidate_count") != PLANNED_CANDIDATE_COUNT:
        raise ValueError("technical grid catalog candidate count drifted")
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != PLANNED_CANDIDATE_COUNT:
        raise ValueError("technical grid catalog is malformed")
    candidate_order = [str(row.get("candidate_id") or "") for row in candidates]
    candidate_identity = [
        {
            "candidate_id": candidate_id,
            "candidate_sha256": _validated_sha(
                row.get("candidate_sha256"), "candidate_sha256"
            ),
        }
        for candidate_id, row in zip(candidate_order, candidates, strict=True)
    ]
    if len(set(candidate_order)) != PLANNED_CANDIDATE_COUNT or any(
        not candidate_id for candidate_id in candidate_order
    ):
        raise ValueError("technical grid catalog candidate identity is invalid")

    pair_order = list(DEFAULT_TRADER_PAIRS)
    matrix_identity = {
        "catalog_contract_sha256": _validated_sha(
            catalog.get("contract_sha256"), "catalog contract_sha256"
        ),
        "pair_order": pair_order,
        "candidate_identity": candidate_identity,
        "cell_state": CURRENT_CELL_STATE,
        "route_receipt_sha256_by_pair": [
            {
                "pair": row["pair"],
                "receipt_sha256": row["receipt_sha256"],
            }
            for row in route_rows
        ],
    }
    matrix = {
        "encoding": SHADOW_MATRIX_ENCODING_V1,
        "pair_order": pair_order,
        "candidate_order": candidate_order,
        "candidate_identity_sha256": _canonical_sha(candidate_identity),
        "pair_count": EXPECTED_PAIR_COUNT,
        "candidate_count_per_pair": PLANNED_CANDIDATE_COUNT,
        "matrix_cardinality": EXPECTED_MATRIX_CARDINALITY,
        "flat_index_formula": (
            "pair_index * candidate_count_per_pair + candidate_catalog_order"
        ),
        "state_runs": [
            {
                "flat_start_inclusive": 0,
                "flat_end_exclusive": EXPECTED_MATRIX_CARDINALITY,
                "length": EXPECTED_MATRIX_CARDINALITY,
                "state": CURRENT_CELL_STATE,
            }
        ],
        "matrix_identity_sha256": _canonical_sha(matrix_identity),
    }
    body = {
        "contract": SHADOW_ORCHESTRATION_CONTRACT_V1,
        "schema_version": 1,
        "orchestration_policy": SHADOW_ORCHESTRATION_POLICY_V1,
        "cycle_id": cycle,
        "cycle_started_at_utc": started.isoformat(),
        "decision_at_utc": decision.isoformat(),
        "observed_at_utc": observed.isoformat(),
        "freshness_deadline_utc": deadline.isoformat(),
        "fresh_at_build": True,
        "freshness_must_be_revalidated_before_use": True,
        "catalog_contract_sha256": matrix_identity["catalog_contract_sha256"],
        "catalog_dimensions": {
            "directional_hypotheses": PLANNED_DIRECTIONAL_FAMILIES,
            "orientations": len(ORIENTATIONS),
            "ofat_arms": PLANNED_ARM_COUNT,
            "candidate_count": PLANNED_CANDIDATE_COUNT,
            "timeframes": list(TIMEFRAMES),
        },
        "pair_routes": route_rows,
        "matrix": matrix,
        "current_route_hypothesis_id": CURRENT_ROUTE_HYPOTHESIS_ID,
        "go_count": 0,
        "go_candidate_refs": [],
        "future_multiple_go_shape_supported": True,
        "go_generation_enabled": False,
        "go_blocker_codes": [
            "CURRENT_ROUTE_H08_NO_TRADE_CONTROL",
            "NO_DEEP_CAUSAL_EXECUTION_GEOMETRY",
            "NO_VERIFIED_ECONOMIC_PROMOTION_RECEIPT",
        ],
        "economic_status": "UNPROVEN_NO_PROFIT_CLAIM",
        "profitability_proven": False,
        "economic_conclusion_allowed": False,
        "risk_sizing_identity_sha256": risk["risk_sizing_identity_sha256"],
        "risk_sizing_receipt_sha256": risk["contract_sha256"],
        "risk_sizing_identity": risk,
        "go_risk_jpy": 0.0,
        "discarded_candidates_retained_in_shadow": True,
        "h08_control_retained_per_pair": True,
        **_zero_authority(),
        "order_intents": [],
    }
    artifact = _seal(body)
    if len(_canonical_json_bytes(artifact)) > MAX_SHADOW_ORCHESTRATION_BYTES:
        raise ValueError("shadow orchestration exceeds its bounded artifact size")
    return artifact


def fast_bot_shadow_orchestration_valid_v1(
    value: Any,
    *,
    now_utc: datetime,
) -> bool:
    """Validate the compact artifact and reject it at its explicit deadline."""

    try:
        if not isinstance(value, Mapping):
            return False
        if frozenset(value) != _ORCHESTRATION_KEYS:
            return False
        if value.get("contract") != SHADOW_ORCHESTRATION_CONTRACT_V1:
            return False
        if value.get("schema_version") != 1:
            return False
        if value.get("orchestration_policy") != SHADOW_ORCHESTRATION_POLICY_V1:
            return False
        if not _sealed_valid(value):
            return False
        now = _aware_utc(now_utc, "now_utc")
        started = _parse_utc(value.get("cycle_started_at_utc"), "cycle start")
        decision = _parse_utc(value.get("decision_at_utc"), "decision")
        observed = _parse_utc(value.get("observed_at_utc"), "observed")
        deadline = _parse_utc(value.get("freshness_deadline_utc"), "deadline")
        if not started <= decision <= observed <= now < deadline:
            return False
        if value.get("fresh_at_build") is not True:
            return False
        if value.get("freshness_must_be_revalidated_before_use") is not True:
            return False
        expected_dimensions = {
            "directional_hypotheses": PLANNED_DIRECTIONAL_FAMILIES,
            "orientations": len(ORIENTATIONS),
            "ofat_arms": PLANNED_ARM_COUNT,
            "candidate_count": PLANNED_CANDIDATE_COUNT,
            "timeframes": list(TIMEFRAMES),
        }
        if value.get("catalog_dimensions") != expected_dimensions:
            return False
        matrix = value.get("matrix")
        routes = value.get("pair_routes")
        if not isinstance(matrix, Mapping) or not isinstance(routes, list):
            return False
        if frozenset(matrix) != _MATRIX_KEYS:
            return False
        if [row.get("pair") for row in routes] != list(DEFAULT_TRADER_PAIRS):
            return False
        if any(not _route_row_valid(row, decision=decision) for row in routes):
            return False
        catalog = build_fast_bot_technical_grid_catalog_v1()
        candidates = catalog.get("candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) != PLANNED_CANDIDATE_COUNT
        ):
            return False
        expected_order = [str(row.get("candidate_id") or "") for row in candidates]
        candidate_identity = [
            {
                "candidate_id": candidate_id,
                "candidate_sha256": str(row.get("candidate_sha256") or ""),
            }
            for candidate_id, row in zip(expected_order, candidates, strict=True)
        ]
        if matrix.get("encoding") != SHADOW_MATRIX_ENCODING_V1:
            return False
        if matrix.get("pair_order") != list(DEFAULT_TRADER_PAIRS):
            return False
        if matrix.get("pair_count") != EXPECTED_PAIR_COUNT:
            return False
        if matrix.get("candidate_count_per_pair") != PLANNED_CANDIDATE_COUNT:
            return False
        if matrix.get("flat_index_formula") != (
            "pair_index * candidate_count_per_pair + candidate_catalog_order"
        ):
            return False
        candidate_order = matrix.get("candidate_order")
        if candidate_order != expected_order:
            return False
        if matrix.get("candidate_identity_sha256") != _canonical_sha(
            candidate_identity
        ):
            return False
        catalog_sha = str(catalog.get("contract_sha256") or "")
        if value.get("catalog_contract_sha256") != catalog_sha:
            return False
        if matrix.get("matrix_cardinality") != EXPECTED_MATRIX_CARDINALITY:
            return False
        if matrix.get("state_runs") != [
            {
                "flat_start_inclusive": 0,
                "flat_end_exclusive": EXPECTED_MATRIX_CARDINALITY,
                "length": EXPECTED_MATRIX_CARDINALITY,
                "state": CURRENT_CELL_STATE,
            }
        ]:
            return False
        matrix_identity = {
            "catalog_contract_sha256": catalog_sha,
            "pair_order": list(DEFAULT_TRADER_PAIRS),
            "candidate_identity": candidate_identity,
            "cell_state": CURRENT_CELL_STATE,
            "route_receipt_sha256_by_pair": [
                {
                    "pair": row["pair"],
                    "receipt_sha256": row["receipt_sha256"],
                }
                for row in routes
            ],
        }
        if matrix.get("matrix_identity_sha256") != _canonical_sha(matrix_identity):
            return False
        if value.get("current_route_hypothesis_id") != CURRENT_ROUTE_HYPOTHESIS_ID:
            return False
        if value.get("go_count") != 0 or value.get("go_candidate_refs") != []:
            return False
        if value.get("go_risk_jpy") != 0.0 or value.get("order_intents") != []:
            return False
        if value.get("profitability_proven") is not False:
            return False
        if value.get("economic_conclusion_allowed") is not False:
            return False
        if value.get("economic_status") != "UNPROVEN_NO_PROFIT_CLAIM":
            return False
        if value.get("future_multiple_go_shape_supported") is not True:
            return False
        if value.get("go_generation_enabled") is not False:
            return False
        if value.get("go_blocker_codes") != [
            "CURRENT_ROUTE_H08_NO_TRADE_CONTROL",
            "NO_DEEP_CAUSAL_EXECUTION_GEOMETRY",
            "NO_VERIFIED_ECONOMIC_PROMOTION_RECEIPT",
        ]:
            return False
        if value.get("discarded_candidates_retained_in_shadow") is not True:
            return False
        if value.get("h08_control_retained_per_pair") is not True:
            return False
        risk = _validated_risk_sizing_identity(value.get("risk_sizing_identity"))
        if risk.get("cycle_id") != value.get("cycle_id"):
            return False
        risk_as_of = _parse_utc(risk.get("as_of_utc"), "risk sizing as_of_utc")
        if not decision <= risk_as_of <= observed:
            return False
        if value.get("risk_sizing_identity_sha256") != risk.get(
            "risk_sizing_identity_sha256"
        ):
            return False
        if value.get("risk_sizing_receipt_sha256") != risk.get("contract_sha256"):
            return False
        if not all(
            _SHA256_RE.fullmatch(str(value.get(name) or ""))
            for name in (
                "risk_sizing_identity_sha256",
                "risk_sizing_receipt_sha256",
            )
        ):
            return False
        if not _zero_authority_valid(value):
            return False
        return len(_canonical_json_bytes(value)) <= MAX_SHADOW_ORCHESTRATION_BYTES
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _deep_validated_h08_route(
    *,
    pair: str,
    item: PairCausalShadowInput,
    decision: datetime,
) -> dict[str, Any]:
    snapshot = item.feature_snapshot
    shadow = item.hypothesis_shadow
    if snapshot.get("contract") != FEATURE_SNAPSHOT_CONTRACT:
        raise ValueError("pair feature snapshot contract is invalid")
    if snapshot.get("pair") != pair or shadow.get("pair") != pair:
        raise ValueError("pair route provenance does not bind the canonical pair")
    if (
        _parse_utc(
            snapshot.get("handoff_cycle_generated_at_utc"), "feature snapshot cycle"
        )
        != decision
    ):
        raise ValueError("pair feature snapshot belongs to a different decision clock")
    if shadow.get("contract") != SHADOW_CONTRACT:
        raise ValueError("technical hypothesis shadow contract is invalid")
    if not technical_hypothesis_shadow_valid(
        shadow,
        feature_snapshot=snapshot,
        attempt_direction=item.attempt_direction,
        branch_outcome=item.branch_outcome,
        route_family=item.route_family,
        spread_pips=item.spread_pips,
        m5_atr_pips=item.m5_atr_pips,
        spread_to_m5_atr=item.spread_to_m5_atr,
    ):
        raise ValueError(
            "technical hypothesis shadow failed deterministic deep validation"
        )
    hypotheses = shadow.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ValueError("technical hypothesis shadow has no hypothesis rows")
    h08 = [row for row in hypotheses if row.get("hypothesis_id") == "H08"]
    if len(h08) != 1 or h08[0].get("status") != "ACTIVE_SHADOW":
        raise ValueError("current pair route is not the active H08 no-trade control")
    if h08[0].get("predicted_side") is not None:
        raise ValueError("H08 control cannot carry a predicted side")
    body = {
        "pair": pair,
        "decision_at_utc": decision.isoformat(),
        "selected_hypothesis_id": CURRENT_ROUTE_HYPOTHESIS_ID,
        "selected_state": "ACTIVE_SHADOW",
        "feature_snapshot_sha256": _validated_sha(
            snapshot.get("contract_sha256"), "feature snapshot contract_sha256"
        ),
        "hypothesis_shadow_sha256": _validated_sha(
            shadow.get("contract_sha256"), "hypothesis shadow contract_sha256"
        ),
        "h08_hypothesis_sha256": _validated_sha(
            h08[0].get("hypothesis_sha256"), "H08 hypothesis_sha256"
        ),
        "evaluator_policy": EVALUATOR_POLICY_V1,
        "deep_validation": "DETERMINISTIC_REBUILD_EXACT_MATCH",
        "active_hypothesis_ids": list(shadow.get("active_hypothesis_ids") or []),
        "h08_evidence": list(h08[0].get("evidence") or []),
        "candidate_state": CURRENT_CELL_STATE,
        "candidate_count": PLANNED_CANDIDATE_COUNT,
        "go_count": 0,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return {**body, "receipt_sha256": _canonical_sha(body)}


def _validated_risk_sizing_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("risk sizing identity must be a mapping")
    result = dict(value)
    if frozenset(result) != _RISK_IDENTITY_KEYS:
        raise ValueError("risk sizing identity has non-canonical fields")
    if result.get("contract") != SHADOW_RISK_SIZING_CONTRACT_V1:
        raise ValueError("risk sizing identity contract is invalid")
    if not _sealed_valid(result) or not _zero_authority_valid(result):
        raise ValueError("risk sizing identity seal or authority is invalid")
    if result.get("candidate_risk_allocations") != []:
        raise ValueError("shadow risk sizing cannot allocate candidate risk")
    if result.get("go_risk_jpy") != 0.0 or result.get("order_intents") != []:
        raise ValueError("shadow risk sizing cannot carry GO risk or order intents")
    if result.get("capacity_accounting_policy") != (
        "BEFORE_OPEN_CAPACITY_MINUS_FRESH_OPEN_MINUS_PENDING_EXACTLY_ONCE"
    ):
        raise ValueError("risk sizing capacity accounting policy is invalid")
    identity = {
        key: result.get(key)
        for key in (
            "cycle_id",
            "as_of_utc",
            "sizing_nav_jpy",
            "daily_loss_capacity_before_open_jpy",
            "fresh_open_risk_jpy",
            "pending_risk_jpy",
            "per_trade_risk_cap_jpy",
            "available_capacity_after_open_and_pending_jpy",
            "capacity_accounting_policy",
            "broker_snapshot_sha256",
            "daily_target_state_sha256",
            "execution_ledger_tip_sha256",
        )
    }
    if result.get("risk_sizing_identity_sha256") != _canonical_sha(identity):
        raise ValueError("risk sizing identity digest is invalid")
    _validated_cycle_id(result.get("cycle_id"))
    _parse_utc(result.get("as_of_utc"), "risk sizing as_of_utc")
    for name in (
        "sizing_nav_jpy",
        "daily_loss_capacity_before_open_jpy",
        "fresh_open_risk_jpy",
        "pending_risk_jpy",
        "per_trade_risk_cap_jpy",
        "available_capacity_after_open_and_pending_jpy",
    ):
        _finite_number(
            result.get(name),
            name,
            positive=name == "sizing_nav_jpy",
            nonnegative=name != "sizing_nav_jpy",
        )
    expected_available = max(
        0.0,
        float(result["daily_loss_capacity_before_open_jpy"])
        - float(result["fresh_open_risk_jpy"])
        - float(result["pending_risk_jpy"]),
    )
    if result.get("available_capacity_after_open_and_pending_jpy") != _rounded(
        expected_available
    ):
        raise ValueError("risk sizing capacity identity does not reconcile")
    if float(result["per_trade_risk_cap_jpy"]) > expected_available:
        raise ValueError("risk sizing per-trade cap exceeds available capacity")
    for name in (
        "broker_snapshot_sha256",
        "daily_target_state_sha256",
        "execution_ledger_tip_sha256",
    ):
        _validated_sha(result.get(name), name)
    return result


def _coerce_pair_input(
    value: PairCausalShadowInput | Mapping[str, Any],
) -> PairCausalShadowInput:
    if isinstance(value, PairCausalShadowInput):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("pair input must be PairCausalShadowInput or a mapping")
    try:
        return PairCausalShadowInput(
            feature_snapshot=value["feature_snapshot"],
            hypothesis_shadow=value["hypothesis_shadow"],
            attempt_direction=str(value["attempt_direction"]),
            branch_outcome=str(value["branch_outcome"]),
            route_family=str(value["route_family"]),
            spread_pips=float(value["spread_pips"]),
            m5_atr_pips=float(value["m5_atr_pips"]),
            spread_to_m5_atr=float(value["spread_to_m5_atr"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("pair input mapping is invalid") from exc


def _route_row_valid(value: Any, *, decision: datetime) -> bool:
    if not isinstance(value, Mapping):
        return False
    if frozenset(value) != _ROUTE_ROW_KEYS:
        return False
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return bool(
        value.get("receipt_sha256") == _canonical_sha(body)
        and value.get("selected_hypothesis_id") == CURRENT_ROUTE_HYPOTHESIS_ID
        and value.get("selected_state") == "ACTIVE_SHADOW"
        and value.get("candidate_state") == CURRENT_CELL_STATE
        and value.get("candidate_count") == PLANNED_CANDIDATE_COUNT
        and value.get("go_count") == 0
        and value.get("order_authority") == "NONE"
        and value.get("live_permission") is False
        and value.get("broker_mutation_allowed") is False
        and _parse_utc(value.get("decision_at_utc"), "route decision") == decision
        and value.get("deep_validation") == "DETERMINISTIC_REBUILD_EXACT_MATCH"
        and value.get("evaluator_policy") == EVALUATOR_POLICY_V1
        and isinstance(value.get("active_hypothesis_ids"), list)
        and CURRENT_ROUTE_HYPOTHESIS_ID in value.get("active_hypothesis_ids", [])
        and len(value.get("active_hypothesis_ids", []))
        == len(set(value.get("active_hypothesis_ids", [])))
        and all(
            hypothesis_id in {f"H{index:02d}" for index in range(1, 9)}
            for hypothesis_id in value.get("active_hypothesis_ids", [])
        )
        and isinstance(value.get("h08_evidence"), list)
        and bool(value.get("h08_evidence"))
        and value.get("h08_evidence") == sorted(set(value.get("h08_evidence", [])))
        and all(
            isinstance(reason, str) and bool(reason)
            for reason in value.get("h08_evidence", [])
        )
        and all(
            _SHA256_RE.fullmatch(str(value.get(name) or ""))
            for name in (
                "feature_snapshot_sha256",
                "hypothesis_shadow_sha256",
                "h08_hypothesis_sha256",
            )
        )
    )


def _zero_authority() -> dict[str, Any]:
    return dict(_ZERO_AUTHORITY)


def _zero_authority_valid(value: Mapping[str, Any]) -> bool:
    return all(value.get(key) == expected for key, expected in _ZERO_AUTHORITY.items())


def _validated_cycle_id(value: Any) -> str:
    if not isinstance(value, str) or not _CYCLE_ID_RE.fullmatch(value):
        raise ValueError("cycle_id is invalid")
    return value


def _validated_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_number(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _aware_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    # Round-tripping through UTC makes equal instants canonical regardless of
    # the caller's timezone while keeping wall-clock reads outside this module.
    return value.astimezone(timezone.utc)


def _parse_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 string") from exc
    return _aware_utc(parsed, name)


def _rounded(value: float) -> float:
    # Eight decimal places preserve sub-JPY calculation precision while
    # removing binary-float serialization noise.  Replace this with a shared
    # money-decimal serializer if the repository introduces one.
    return round(float(value), 8)


def _sealed_valid(value: Mapping[str, Any]) -> bool:
    expected = value.get("contract_sha256")
    if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
        return False
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return expected == _canonical_sha(body)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    return {**body, "contract_sha256": _canonical_sha(body)}


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "CURRENT_CELL_STATE",
    "CURRENT_ROUTE_HYPOTHESIS_ID",
    "EXPECTED_MATRIX_CARDINALITY",
    "MAX_SHADOW_ORCHESTRATION_BYTES",
    "PairCausalShadowInput",
    "SHADOW_ORCHESTRATION_CONTRACT_V1",
    "SHADOW_RISK_SIZING_CONTRACT_V1",
    "build_fast_bot_shadow_orchestration_v1",
    "build_fast_bot_shadow_risk_sizing_identity_v1",
    "fast_bot_shadow_orchestration_valid_v1",
]

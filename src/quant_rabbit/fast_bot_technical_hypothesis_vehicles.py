"""Pre-outcome execution vehicles for the fast-bot technical catalog.

The V1 technical evaluator decides whether H01..H08 are active.  This module
does not reinterpret that decision.  It binds every active hypothesis to one
predeclared execution vehicle using only the already-sealed feature snapshot,
episode anchor/route, learning seat, and confirmation clock.  It deliberately
has no truth adapter, broker client, result reader, tuning path, or live order
authority.

V2 is append-only.  A later geometry experiment must add a new policy/version;
it must not reinterpret rows emitted under this contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any, Mapping, Sequence

from quant_rabbit.fast_bot_episode import (
    CONFIRMATION_TTL_SECONDS,
    EPISODE_KIND,
    EPISODE_RULE_VERSION,
    HORIZON_LANE,
    SELECTION_STATUS,
)
from quant_rabbit.fast_bot_learning import (
    BASE_ENTRY_TTL_SECONDS,
    BASE_MAX_HOLD_SECONDS,
    LEARNING_SEAT_CONTRACT,
    _learning_seat_valid,
)
from quant_rabbit.fast_bot_technical_hypotheses import (
    technical_hypothesis_shadow_valid,
)
from quant_rabbit.instruments import instrument_pip_factor


CATALOG_CONTRACT_V2 = "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_VEHICLE_CATALOG_V2"
SHADOW_CONTRACT_V2 = "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_VEHICLE_SHADOW_V2"
VEHICLE_CONTRACT_V2 = "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_VEHICLE_V2"
VEHICLE_POLICY_V2 = "PRE_OUTCOME_CAUSAL_TECHNICAL_VEHICLES_V2"
NATIVE_MARKET_UNIT_COHORT_POLICY_V2 = (
    "EXACT_FROZEN_NATIVE_MARKET_UNIT_NO_TUNED_MULTIPLIER_OR_FIXED_PIPS_V1"
)

# OANDA G8 FX quotes carry one pipette per one-tenth pip.  This is broker price
# precision, not a market threshold or a tuned geometry multiplier.
OANDA_FX_TICKS_PER_PIP = 10
# Engineering-only comparison tolerance for the detector's decimal JSON rail
# arithmetic.  It does not admit, reject, widen, or shrink market geometry.
SEALED_RAIL_FLOAT_ABS_TOLERANCE = 1e-9
# Stable diagnostic serialization precision; this is not an edge threshold.
PIP_METRIC_SERIALIZATION_DECIMALS = 6

EXACT_STOP_GAP_POLICY_V2 = (
    "REAL_S5_ONLY;LONG_ENTRY_ASK_SHORT_ENTRY_BID;ENTRY_GAP_AT_EXECUTABLE_OPEN;"
    "SL_GAP_AT_EXECUTABLE_OPEN;TP_NO_OPTIMISTIC_GAP_IMPROVEMENT;"
    "NEXT_REAL_OPEN_OWNS_VERIFIED_NO_TICK_GAP"
)
EXACT_STOP_INTRABAR_POLICY_V2 = (
    "ENTRY_CANDLE_ANY_ATTACHED_EXIT_IS_SL_FIRST;" "LATER_SAME_S5_TP_AND_SL_IS_SL_FIRST"
)
EXACT_STOP_HOLD_POLICY_V2 = (
    "UNFILLED_EXPIRY_ZERO;FILLED_UNRESOLVED_CHARGE_FULL_FROZEN_SL;"
    "RETAIN_SEPARATE_EXECUTABLE_HOLD_END_MARK"
)
BASE_PROXY_GAP_POLICY_V2 = "INHERIT_EXACT_FROZEN_BASE_ARM_TRUTH_POLICY"
BASE_PROXY_INTRABAR_POLICY_V2 = "INHERIT_EXACT_FROZEN_BASE_ARM_TRUTH_POLICY"
ZERO_CONTROL_POLICY_V2 = "NO_ENTRY_NO_GAP_NO_INTRABAR_ZERO_PNL"

CAUSAL_SOURCE_PATHS_V2 = (
    "confirmed_at_utc",
    "technical_feature_snapshot.contract_sha256",
    "technical_feature_snapshot.pair",
    "technical_feature_snapshot.handoff_cycle_generated_at_utc",
    "technical_feature_snapshot.timeframes[].timeframe",
    "technical_feature_snapshot.timeframes[].complete_candle_close_utc",
    "technical_feature_snapshot.timeframes[].indicators.close",
    "technical_feature_snapshot.timeframes[].indicators.donchian_high",
    "technical_feature_snapshot.timeframes[].indicators.donchian_low",
    "technical_feature_snapshot.timeframes[].indicators.bb_middle",
    "technical_feature_snapshot.timeframes[].indicators.atr_pips",
    "technical_hypothesis_shadow.contract_sha256",
    "technical_hypothesis_shadow.cost_state",
    "technical_hypothesis_shadow.hypotheses[].hypothesis_id",
    "technical_hypothesis_shadow.hypotheses[].hypothesis_sha256",
    "technical_hypothesis_shadow.hypotheses[].status",
    "technical_hypothesis_shadow.hypotheses[].predicted_side",
    "episode_anchor.attempt_direction",
    "episode_anchor.setup_candle_utc",
    "episode_anchor.attempt_candle_utc",
    "episode_anchor.rail",
    "episode_anchor.source_evidence_sha256",
    "episode_route.branch_outcome",
    "episode_route.trade_side",
    "episode_route.candidate_methods",
    "episode_route.route_family",
    "episode_route.branch_candle_utc",
    "episode_route.branch_close",
    "learning_seat.contract_sha256",
    "learning_seat.seat_id",
    "learning_seat.generated_at_utc",
    "learning_seat.m1_closed_candle_utc",
    "learning_seat.quote_timestamp_utc",
    "learning_seat.quote_bid",
    "learning_seat.quote_ask",
    "learning_seat.executable_spread_pips",
    "learning_seat.m5_atr_pips",
    "learning_seat.spread_to_m5_atr",
    "learning_seat.causal_input_proof_eligible",
    "learning_seat.paired_direction_proof_eligible",
    "learning_seat.candidates[].candidate_id",
    "learning_seat.candidates[].candidate_sha256",
    "learning_seat.candidates[].side",
    "learning_seat.candidates[].method",
    "learning_seat.candidates[].input_blocked",
    "learning_seat.candidates[].technical_hypothesis_proof_eligible",
    "learning_seat.candidates[].arms[arm_id=BASE]",
)

_ANCHOR_KEYS = frozenset(
    {
        "episode_kind",
        "episode_rule_version",
        "horizon_lane",
        "confirmation_ttl_seconds",
        "setup_candle_utc",
        "attempt_candle_utc",
        "attempt_direction",
        "rail",
        "source_evidence_sha256",
    }
)
_RAIL_KEYS = frozenset({"upper", "lower", "width", "buffer", "buffer_ratio"})
_ROUTE_KEYS = frozenset(
    {
        "branch_outcome",
        "trade_side",
        "candidate_methods",
        "route_family",
        "branch_candle_utc",
        "branch_close",
        "selection_status",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HYPOTHESIS_IDS = tuple(f"H{number:02d}" for number in range(1, 9))
_SIDES = frozenset({"LONG", "SHORT"})


@dataclass(frozen=True, slots=True)
class TechnicalHypothesisVehicleDefinitionV2:
    hypothesis_id: str
    execution_kind: str
    order_type: str
    entry_formula: str
    entry_ttl_policy: str
    take_profit_formula: str
    stop_loss_formula: str
    max_hold_policy: str
    gap_policy: str
    intrabar_policy: str
    applicability: str
    cost_policy: str
    geometry_unit_policy: str
    proxy_method: str | None = None


TECHNICAL_HYPOTHESIS_VEHICLE_CATALOG_V2 = (
    TechnicalHypothesisVehicleDefinitionV2(
        hypothesis_id="H01",
        execution_kind="EXACT_ATTACHED_BARRIER_STOP",
        order_type="STOP_GTD",
        entry_formula=(
            "LONG=CEIL_TICK(MAX(FROZEN_QUOTE_ASK,M1_DONCHIAN_HIGH)+BROKER_TICK);"
            "SHORT=FLOOR_TICK(MIN(FROZEN_QUOTE_BID,M1_DONCHIAN_LOW)-BROKER_TICK)"
        ),
        entry_ttl_policy="IMPORT_FAST_BOT_LEARNING_BASE_ENTRY_TTL_SECONDS",
        take_profit_formula="ENTRY_PLUS_MINUS_FROZEN_M15_ATR_PIPS",
        stop_loss_formula="ENTRY_MINUS_PLUS_FROZEN_LEARNING_SEAT_M5_ATR_PIPS",
        max_hold_policy="IMPORT_FAST_BOT_LEARNING_BASE_MAX_HOLD_SECONDS",
        gap_policy=EXACT_STOP_GAP_POLICY_V2,
        intrabar_policy=EXACT_STOP_INTRABAR_POLICY_V2,
        applicability="V1_H01_ACTIVE_WITH_EXACT_PREDICTED_SIDE",
        cost_policy="OBSERVE_COST_COHORT_NEVER_GATE_ON_SPREAD_TO_TP",
        geometry_unit_policy=(
            "ONE_FROZEN_M15_ATR_TP_AND_ONE_FROZEN_M5_ATR_SL;"
            "SOURCE_MEASURES_USED_DIRECTLY"
        ),
    ),
    TechnicalHypothesisVehicleDefinitionV2(
        hypothesis_id="H02",
        execution_kind="EXACT_ATTACHED_BARRIER_STOP",
        order_type="STOP_GTD",
        entry_formula=(
            "LONG=CEIL_TICK(MAX(FROZEN_QUOTE_ASK,M1_CLOSE)+BROKER_TICK);"
            "SHORT=FLOOR_TICK(MIN(FROZEN_QUOTE_BID,M1_CLOSE)-BROKER_TICK)"
        ),
        entry_ttl_policy="IMPORT_FAST_BOT_LEARNING_BASE_ENTRY_TTL_SECONDS",
        take_profit_formula="ENTRY_PLUS_MINUS_FROZEN_M15_ATR_PIPS",
        stop_loss_formula="ENTRY_MINUS_PLUS_FROZEN_LEARNING_SEAT_M5_ATR_PIPS",
        max_hold_policy="IMPORT_FAST_BOT_LEARNING_BASE_MAX_HOLD_SECONDS",
        gap_policy=EXACT_STOP_GAP_POLICY_V2,
        intrabar_policy=EXACT_STOP_INTRABAR_POLICY_V2,
        applicability="V1_H02_ACTIVE_AFTER_M1_REACCELERATION",
        cost_policy="OBSERVE_COST_COHORT_NEVER_GATE_ON_SPREAD_TO_TP",
        geometry_unit_policy=(
            "ONE_FROZEN_M15_ATR_TP_AND_ONE_FROZEN_M5_ATR_SL;"
            "SOURCE_MEASURES_USED_DIRECTLY"
        ),
    ),
    TechnicalHypothesisVehicleDefinitionV2(
        hypothesis_id="H03",
        execution_kind="FROZEN_BASE_ARM_PROXY",
        order_type="INHERIT_FROZEN_BASE_ARM",
        entry_formula="BIND_EXACT_PREDICTED_SIDE_RANGE_ROTATION_BASE_ARM_BY_DIGEST",
        entry_ttl_policy="INHERIT_FROZEN_BASE_ARM_WITHOUT_NUMERIC_COPY",
        take_profit_formula="INHERIT_FROZEN_BASE_ARM_WITHOUT_NUMERIC_COPY",
        stop_loss_formula="INHERIT_FROZEN_BASE_ARM_WITHOUT_NUMERIC_COPY",
        max_hold_policy="INHERIT_FROZEN_BASE_ARM_WITHOUT_NUMERIC_COPY",
        gap_policy=BASE_PROXY_GAP_POLICY_V2,
        intrabar_policy=BASE_PROXY_INTRABAR_POLICY_V2,
        applicability="V1_H03_ACTIVE_AND_EXACT_SIDE_METHOD_BASE_ARM_EXISTS",
        cost_policy="OBSERVE_FROZEN_SEAT_COST_COHORT",
        geometry_unit_policy="INHERIT_EXACT_FROZEN_BASE_ARM_BY_DIGEST",
        proxy_method="RANGE_ROTATION",
    ),
    TechnicalHypothesisVehicleDefinitionV2(
        hypothesis_id="H04",
        execution_kind="EXACT_ATTACHED_BARRIER_STOP",
        order_type="STOP_GTD",
        entry_formula=(
            "LONG=CEIL_TICK(MAX(FROZEN_QUOTE_ASK,M1_DONCHIAN_HIGH,"
            "RAIL_UPPER_PLUS_BUFFER)+BROKER_TICK);"
            "SHORT=FLOOR_TICK(MIN(FROZEN_QUOTE_BID,M1_DONCHIAN_LOW,"
            "RAIL_LOWER_MINUS_BUFFER)-BROKER_TICK)"
        ),
        entry_ttl_policy="IMPORT_FAST_BOT_LEARNING_BASE_ENTRY_TTL_SECONDS",
        take_profit_formula="ENTRY_PLUS_MINUS_FROZEN_EPISODE_RAIL_WIDTH",
        stop_loss_formula=(
            "LONG=FROZEN_RAIL_UPPER_MINUS_BUFFER;" "SHORT=FROZEN_RAIL_LOWER_PLUS_BUFFER"
        ),
        max_hold_policy="IMPORT_FAST_BOT_LEARNING_BASE_MAX_HOLD_SECONDS",
        gap_policy=EXACT_STOP_GAP_POLICY_V2,
        intrabar_policy=EXACT_STOP_INTRABAR_POLICY_V2,
        applicability="V1_H04_ACTIVE_ACCEPTED_BRANCH_AND_ATTEMPT_SIDE_MATCH",
        cost_policy="OBSERVE_COST_COHORT_NEVER_GATE_ON_SPREAD_TO_TP",
        geometry_unit_policy=(
            "ONE_EXACT_FROZEN_EPISODE_RAIL_WIDTH_TP;"
            "EXACT_FROZEN_BRANCH_INVALIDATION_SL"
        ),
    ),
    TechnicalHypothesisVehicleDefinitionV2(
        hypothesis_id="H05",
        execution_kind="FROZEN_BASE_ARM_PROXY",
        order_type="INHERIT_FROZEN_BASE_ARM",
        entry_formula="BIND_EXACT_PREDICTED_SIDE_BREAKOUT_FAILURE_BASE_ARM_BY_DIGEST",
        entry_ttl_policy="INHERIT_FROZEN_BASE_ARM_WITHOUT_NUMERIC_COPY",
        take_profit_formula="INHERIT_FROZEN_BASE_ARM_WITHOUT_NUMERIC_COPY",
        stop_loss_formula="INHERIT_FROZEN_BASE_ARM_WITHOUT_NUMERIC_COPY",
        max_hold_policy="INHERIT_FROZEN_BASE_ARM_WITHOUT_NUMERIC_COPY",
        gap_policy=BASE_PROXY_GAP_POLICY_V2,
        intrabar_policy=BASE_PROXY_INTRABAR_POLICY_V2,
        applicability="V1_H05_ACTIVE_AND_EXACT_SIDE_METHOD_BASE_ARM_EXISTS",
        cost_policy="OBSERVE_FROZEN_SEAT_COST_COHORT",
        geometry_unit_policy="INHERIT_EXACT_FROZEN_BASE_ARM_BY_DIGEST",
        proxy_method="BREAKOUT_FAILURE",
    ),
    TechnicalHypothesisVehicleDefinitionV2(
        hypothesis_id="H06",
        execution_kind="EXACT_ATTACHED_BARRIER_STOP",
        order_type="STOP_GTD",
        entry_formula=(
            "LONG=CEIL_TICK(MAX(FROZEN_QUOTE_ASK,M1_CLOSE)+BROKER_TICK);"
            "SHORT=FLOOR_TICK(MIN(FROZEN_QUOTE_BID,M1_CLOSE)-BROKER_TICK)"
        ),
        entry_ttl_policy="IMPORT_FAST_BOT_LEARNING_BASE_ENTRY_TTL_SECONDS",
        take_profit_formula="ABSOLUTE_FROZEN_M1_BOLLINGER_MIDDLE",
        stop_loss_formula=(
            "LONG=FROZEN_M1_DONCHIAN_LOW_MINUS_BROKER_TICK;"
            "SHORT=FROZEN_M1_DONCHIAN_HIGH_PLUS_BROKER_TICK"
        ),
        max_hold_policy="IMPORT_FAST_BOT_LEARNING_BASE_MAX_HOLD_SECONDS",
        gap_policy=EXACT_STOP_GAP_POLICY_V2,
        intrabar_policy=EXACT_STOP_INTRABAR_POLICY_V2,
        applicability=(
            "V1_H06_ACTIVE_AND_BOLLINGER_TARGET_FAVORABLE_AND_" "DONCHIAN_STOP_ADVERSE"
        ),
        cost_policy="OBSERVE_COST_COHORT_NEVER_GATE_ON_SPREAD_TO_TP",
        geometry_unit_policy="EXACT_FROZEN_BOLLINGER_TARGET_AND_DONCHIAN_INVALIDATION",
    ),
    TechnicalHypothesisVehicleDefinitionV2(
        hypothesis_id="H07",
        execution_kind="EXACT_ATTACHED_BARRIER_STOP",
        order_type="STOP_GTD",
        entry_formula=(
            "LONG=CEIL_TICK(MAX(FROZEN_QUOTE_ASK,M5_DONCHIAN_HIGH,"
            "RAIL_UPPER_PLUS_BUFFER)+BROKER_TICK);"
            "SHORT=FLOOR_TICK(MIN(FROZEN_QUOTE_BID,M5_DONCHIAN_LOW,"
            "RAIL_LOWER_MINUS_BUFFER)-BROKER_TICK)"
        ),
        entry_ttl_policy="IMPORT_FAST_BOT_LEARNING_BASE_ENTRY_TTL_SECONDS",
        take_profit_formula="ENTRY_PLUS_MINUS_FROZEN_EPISODE_RAIL_WIDTH",
        stop_loss_formula=(
            "LONG=FROZEN_RAIL_UPPER_MINUS_BUFFER;" "SHORT=FROZEN_RAIL_LOWER_PLUS_BUFFER"
        ),
        max_hold_policy="IMPORT_FAST_BOT_LEARNING_BASE_MAX_HOLD_SECONDS",
        gap_policy=EXACT_STOP_GAP_POLICY_V2,
        intrabar_policy=EXACT_STOP_INTRABAR_POLICY_V2,
        applicability=(
            "V1_H07_ACTIVE_SESSION_EVIDENCE_ACCEPTED_BRANCH_AND_ATTEMPT_SIDE_MATCH"
        ),
        cost_policy="OBSERVE_COST_COHORT_NEVER_GATE_ON_SPREAD_TO_TP",
        geometry_unit_policy=(
            "ONE_EXACT_FROZEN_EPISODE_RAIL_WIDTH_TP;"
            "EXACT_FROZEN_BRANCH_INVALIDATION_SL"
        ),
    ),
    TechnicalHypothesisVehicleDefinitionV2(
        hypothesis_id="H08",
        execution_kind="ZERO_PNL_CONTROL",
        order_type="NONE",
        entry_formula="NO_ENTRY",
        entry_ttl_policy="NONE",
        take_profit_formula="ZERO_PNL_IDENTITY",
        stop_loss_formula="ZERO_PNL_IDENTITY",
        max_hold_policy="NONE",
        gap_policy=ZERO_CONTROL_POLICY_V2,
        intrabar_policy=ZERO_CONTROL_POLICY_V2,
        applicability="V1_H08_ACTIVE_NO_TRADE_CONTROL",
        cost_policy="CONTROL_HAS_NO_EXECUTION_COST",
        geometry_unit_policy="NO_GEOMETRY_ZERO_PNL_CONTROL",
    ),
)


def technical_hypothesis_vehicle_catalog_v2() -> dict[str, Any]:
    """Return the sealed, immutable V2 vehicle definitions."""

    rows = [_definition_dict(item) for item in TECHNICAL_HYPOTHESIS_VEHICLE_CATALOG_V2]
    body = {
        "contract": CATALOG_CONTRACT_V2,
        "schema_version": 2,
        "vehicle_policy": VEHICLE_POLICY_V2,
        "causal_source_paths": list(CAUSAL_SOURCE_PATHS_V2),
        "activation_clock_policy": (
            "MAX_CONFIRMED_AT_SEAT_GENERATED_AT_QUOTE_TIMESTAMP"
        ),
        "price_precision_policy": ("OANDA_G8_FX_PIPETTE_FROM_INSTRUMENT_PIP_FACTOR"),
        "missing_input_policy": "REASON_CODED_UNSCORABLE_NO_FALLBACK",
        "native_market_unit_cohort_policy": NATIVE_MARKET_UNIT_COHORT_POLICY_V2,
        "native_market_unit_cohort_definition": (
            "ATR and exact episode-rail distances are prospectively frozen native "
            "market measures used directly. No tuned numeric multiplier or fixed-pip "
            "distance is introduced by this vehicle catalog."
        ),
        "tuned_geometry_multipliers_allowed": False,
        "fixed_pip_geometry_allowed": False,
        "cost_policy": ("OBSERVE_FROZEN_COST_COHORT_NEVER_EXCLUDE_BY_SPREAD_TO_TP"),
        "spread_larger_than_take_profit_is_scorable": True,
        "post_emission_geometry_change_allowed": False,
        "vehicles": rows,
        "diagnostic_only": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "primary_effect": False,
        "risk_effect": False,
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_intents": [],
    }
    return _seal(body)


def build_fast_bot_technical_hypothesis_vehicles_v2(
    *,
    technical_feature_snapshot: Mapping[str, Any],
    technical_hypothesis_shadow: Mapping[str, Any],
    episode_anchor: Mapping[str, Any],
    episode_route: Mapping[str, Any],
    learning_seat: Mapping[str, Any],
    confirmed_at_utc: str,
) -> dict[str, Any]:
    """Freeze all V2 vehicles before any later price path may be read."""

    catalog = technical_hypothesis_vehicle_catalog_v2()
    authority = _zero_authority()
    errors = _causal_source_errors(
        technical_feature_snapshot=technical_feature_snapshot,
        technical_hypothesis_shadow=technical_hypothesis_shadow,
        episode_anchor=episode_anchor,
        episode_route=episode_route,
        learning_seat=learning_seat,
        confirmed_at_utc=confirmed_at_utc,
    )
    if errors:
        return _seal(
            {
                "contract": SHADOW_CONTRACT_V2,
                "schema_version": 2,
                "vehicle_policy": VEHICLE_POLICY_V2,
                "catalog_contract_sha256": catalog["contract_sha256"],
                "status": "INVALID_CAUSAL_INPUT",
                "causal_input_errors": sorted(set(errors)),
                "causal_input_proof_eligible": False,
                "paired_direction_proof_eligible": False,
                "technical_hypothesis_proof_eligible": False,
                "scorecard_eligible": False,
                "scorecard_ineligibility_reasons": ["INVALID_CAUSAL_INPUT"],
                "vehicle_count": 0,
                "diagnostic_vehicle_count": 0,
                "diagnostic_evaluable_vehicle_count": 0,
                "scoring_vehicle_count": 0,
                "vehicles": [],
                **authority,
            }
        )

    pair = str(technical_feature_snapshot["pair"])
    confirmed_at = _parse_utc(confirmed_at_utc)
    seat_generated_at = _parse_utc(learning_seat["generated_at_utc"])
    quote_timestamp = _parse_utc(learning_seat["quote_timestamp_utc"])
    if confirmed_at is None or seat_generated_at is None or quote_timestamp is None:
        raise AssertionError("validated causal clocks became unavailable")
    activation_at = max(confirmed_at, seat_generated_at, quote_timestamp)
    anchor_sha = _canonical_sha(dict(episode_anchor))
    route_sha = _canonical_sha(dict(episode_route))
    source_binding = {
        "technical_feature_snapshot_sha256": str(
            technical_feature_snapshot["contract_sha256"]
        ),
        "technical_hypothesis_shadow_sha256": str(
            technical_hypothesis_shadow["contract_sha256"]
        ),
        "technical_hypothesis_catalog_sha256": str(
            technical_hypothesis_shadow["catalog_contract_sha256"]
        ),
        "technical_cost_state_sha256": str(
            technical_hypothesis_shadow["cost_state_sha256"]
        ),
        "episode_anchor_sha256": anchor_sha,
        "episode_route_sha256": route_sha,
        "learning_seat_contract_sha256": str(learning_seat["contract_sha256"]),
        "learning_seat_id": str(learning_seat["seat_id"]),
        "learning_seat_generated_at_utc": seat_generated_at.isoformat(),
        "learning_seat_quote_timestamp_utc": quote_timestamp.isoformat(),
        "learning_seat_m1_closed_candle_utc": str(
            learning_seat["m1_closed_candle_utc"]
        ),
        "confirmed_at_utc": confirmed_at.isoformat(),
        "activation_at_utc": activation_at.isoformat(),
    }
    cost_cohort = {
        "technical_cost_state_sha256": str(
            technical_hypothesis_shadow["cost_state_sha256"]
        ),
        "executable_spread_pips": float(learning_seat["executable_spread_pips"]),
        "m5_atr_pips": float(learning_seat["m5_atr_pips"]),
        "spread_to_m5_atr": float(learning_seat["spread_to_m5_atr"]),
        "cost_is_direction": False,
        "cost_is_applicability_gate": False,
        "spread_larger_than_take_profit_is_excluded": False,
    }
    proof_eligibility = _seat_proof_eligibility(learning_seat)
    hypotheses = {
        str(row["hypothesis_id"]): row
        for row in technical_hypothesis_shadow["hypotheses"]
    }
    definitions = {
        item.hypothesis_id: item for item in TECHNICAL_HYPOTHESIS_VEHICLE_CATALOG_V2
    }
    rows: list[dict[str, Any]] = []
    for hypothesis_id in _HYPOTHESIS_IDS:
        hypothesis = hypotheses[hypothesis_id]
        definition = definitions[hypothesis_id]
        rows.append(
            _build_vehicle_row(
                pair=pair,
                definition=definition,
                hypothesis=hypothesis,
                feature_snapshot=technical_feature_snapshot,
                anchor=episode_anchor,
                route=episode_route,
                seat=learning_seat,
                activation_at=activation_at,
                catalog_sha=str(catalog["contract_sha256"]),
                common_source_binding=source_binding,
                cost_cohort=cost_cohort,
                proof_eligibility=proof_eligibility,
            )
        )
    body = {
        "contract": SHADOW_CONTRACT_V2,
        "schema_version": 2,
        "vehicle_policy": VEHICLE_POLICY_V2,
        "catalog_contract_sha256": catalog["contract_sha256"],
        "status": (
            "EMITTED"
            if proof_eligibility["scorecard_eligible"] is True
            else "EMITTED_DIAGNOSTIC_ONLY"
        ),
        "pair": pair,
        "confirmed_at_utc": confirmed_at.isoformat(),
        "activation_at_utc": activation_at.isoformat(),
        "source_binding": source_binding,
        "source_binding_sha256": _canonical_sha(source_binding),
        "cost_cohort": cost_cohort,
        "cost_cohort_sha256": _canonical_sha(cost_cohort),
        **proof_eligibility,
        "vehicle_count": len(rows),
        "diagnostic_vehicle_count": sum(
            row["diagnostic_vehicle_available"] is True for row in rows
        ),
        "diagnostic_evaluable_vehicle_count": sum(
            row["diagnostic_evaluable"] is True for row in rows
        ),
        "scoring_vehicle_count": sum(
            row["scoring_vehicle_available"] is True for row in rows
        ),
        "vehicles": rows,
        **authority,
    }
    return _seal(body)


def technical_hypothesis_vehicle_shadow_v2_valid(
    value: Any,
    *,
    technical_feature_snapshot: Mapping[str, Any],
    technical_hypothesis_shadow: Mapping[str, Any],
    episode_anchor: Mapping[str, Any],
    episode_route: Mapping[str, Any],
    learning_seat: Mapping[str, Any],
    confirmed_at_utc: str,
) -> bool:
    """Rebuild one V2 artifact from the same causal bytes and compare exactly."""

    if not isinstance(value, Mapping) or not _sealed_valid(value, SHADOW_CONTRACT_V2):
        return False
    try:
        expected = build_fast_bot_technical_hypothesis_vehicles_v2(
            technical_feature_snapshot=technical_feature_snapshot,
            technical_hypothesis_shadow=technical_hypothesis_shadow,
            episode_anchor=episode_anchor,
            episode_route=episode_route,
            learning_seat=learning_seat,
            confirmed_at_utc=confirmed_at_utc,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        expected.get("status") in {"EMITTED", "EMITTED_DIAGNOSTIC_ONLY"}
        and dict(value) == expected
    )


def _seat_proof_eligibility(seat: Mapping[str, Any]) -> dict[str, Any]:
    candidates = seat.get("candidates")
    candidate_rows = (
        [item for item in candidates if isinstance(item, Mapping)]
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes))
        else []
    )
    causal_eligible = seat.get("causal_input_proof_eligible") is True
    paired_eligible = seat.get("paired_direction_proof_eligible") is True
    technical_eligible = bool(
        causal_eligible
        and candidate_rows
        and len(candidate_rows) == len(candidates)
        and all(
            item.get("technical_hypothesis_proof_eligible") is True
            for item in candidate_rows
        )
    )
    reasons: list[str] = []
    if any(item.get("input_blocked") is True for item in candidate_rows):
        reasons.append("INPUT_BLOCKED_SHADOW_DIAGNOSTIC_ONLY")
    if not causal_eligible:
        reasons.append("LEARNING_SEAT_CAUSAL_INPUT_PROOF_INELIGIBLE")
    if not paired_eligible:
        reasons.append("LEARNING_SEAT_PAIRED_DIRECTION_PROOF_INELIGIBLE")
    if not technical_eligible:
        reasons.append("LEARNING_SEAT_TECHNICAL_HYPOTHESIS_PROOF_INELIGIBLE")
    scorecard_eligible = bool(
        causal_eligible and paired_eligible and technical_eligible
    )
    return {
        "causal_input_proof_eligible": causal_eligible,
        "paired_direction_proof_eligible": paired_eligible,
        "technical_hypothesis_proof_eligible": technical_eligible,
        "scorecard_eligible": scorecard_eligible,
        "scorecard_ineligibility_reasons": reasons,
    }


def _combined_reasons(*groups: Sequence[str]) -> list[str]:
    return sorted({str(item) for group in groups for item in group if str(item)})


def _build_vehicle_row(
    *,
    pair: str,
    definition: TechnicalHypothesisVehicleDefinitionV2,
    hypothesis: Mapping[str, Any],
    feature_snapshot: Mapping[str, Any],
    anchor: Mapping[str, Any],
    route: Mapping[str, Any],
    seat: Mapping[str, Any],
    activation_at: datetime,
    catalog_sha: str,
    common_source_binding: Mapping[str, Any],
    cost_cohort: Mapping[str, Any],
    proof_eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    hypothesis_id = definition.hypothesis_id
    active = hypothesis.get("status") == "ACTIVE_SHADOW"
    side = str(hypothesis.get("predicted_side") or "")
    definition_dict = _definition_dict(definition)
    row_source_binding = {
        **dict(common_source_binding),
        "hypothesis_sha256": str(hypothesis["hypothesis_sha256"]),
        "vehicle_definition_sha256": _canonical_sha(definition_dict),
    }
    base = {
        "contract": VEHICLE_CONTRACT_V2,
        "schema_version": 2,
        "vehicle_policy": VEHICLE_POLICY_V2,
        "catalog_contract_sha256": catalog_sha,
        "hypothesis_id": hypothesis_id,
        "hypothesis_sha256": str(hypothesis["hypothesis_sha256"]),
        "hypothesis_status_at_emission": str(hypothesis["status"]),
        "predicted_side": side or None,
        "execution_kind": definition.execution_kind,
        "native_market_unit_cohort_policy": NATIVE_MARKET_UNIT_COHORT_POLICY_V2,
        "geometry_unit_policy": definition.geometry_unit_policy,
        "tuned_geometry_multiplier_used": False,
        "fixed_pip_geometry_used": False,
        "vehicle_definition_sha256": _canonical_sha(definition_dict),
        "source_binding": row_source_binding,
        "source_binding_sha256": _canonical_sha(row_source_binding),
        "cost_cohort_sha256": _canonical_sha(cost_cohort),
        "causal_input_proof_eligible": proof_eligibility["causal_input_proof_eligible"],
        "paired_direction_proof_eligible": proof_eligibility[
            "paired_direction_proof_eligible"
        ],
        "technical_hypothesis_proof_eligible": proof_eligibility[
            "technical_hypothesis_proof_eligible"
        ],
        "proxy_binding": None,
        "control": None,
        **_zero_authority(),
    }

    if hypothesis_id == "H08":
        control = {
            "control_type": "NO_TRADE_ZERO_PNL",
            "net_pips": 0.0,
            "filled": False,
        }
        diagnostic_available = active
        scoring_available = bool(
            diagnostic_available and proof_eligibility["scorecard_eligible"] is True
        )
        scorecard_reasons = _combined_reasons(
            proof_eligibility["scorecard_ineligibility_reasons"],
            [] if diagnostic_available else ["HYPOTHESIS_INACTIVE"],
        )
        return _seal_vehicle_row(
            {
                **base,
                "status": (
                    "ZERO_PNL_CONTROL_READY"
                    if scoring_available
                    else (
                        "ZERO_PNL_CONTROL_DIAGNOSTIC_ONLY"
                        if diagnostic_available
                        else "ZERO_PNL_CONTROL_INACTIVE"
                    )
                ),
                "reasons": ([] if active else ["HYPOTHESIS_INACTIVE"]),
                "diagnostic_vehicle_available": diagnostic_available,
                "scorecard_eligible": scoring_available,
                "scorecard_ineligibility_reasons": scorecard_reasons,
                "scoring_vehicle_available": scoring_available,
                "execution": None,
                "control": control,
                "cost_observation": {
                    "executable_spread_pips": None,
                    "gross_take_profit_pips": None,
                    "spread_to_gross_take_profit": None,
                    "cost_is_applicability_gate": False,
                },
            }
        )

    if not active:
        return _seal_vehicle_row(
            {
                **base,
                "status": "INAPPLICABLE_INACTIVE_HYPOTHESIS",
                "reasons": ["HYPOTHESIS_INACTIVE"],
                "diagnostic_vehicle_available": False,
                "scorecard_eligible": False,
                "scorecard_ineligibility_reasons": _combined_reasons(
                    proof_eligibility["scorecard_ineligibility_reasons"],
                    ["HYPOTHESIS_INACTIVE"],
                ),
                "scoring_vehicle_available": False,
                "execution": None,
                "cost_observation": _empty_cost_observation(cost_cohort),
            }
        )
    if side not in _SIDES:
        return _seal_vehicle_row(
            {
                **base,
                "status": "UNSCORABLE_CAUSAL_INPUT",
                "reasons": ["ACTIVE_HYPOTHESIS_PREDICTED_SIDE_MISSING"],
                "diagnostic_vehicle_available": False,
                "scorecard_eligible": False,
                "scorecard_ineligibility_reasons": _combined_reasons(
                    proof_eligibility["scorecard_ineligibility_reasons"],
                    ["ACTIVE_HYPOTHESIS_PREDICTED_SIDE_MISSING"],
                ),
                "scoring_vehicle_available": False,
                "execution": None,
                "cost_observation": _empty_cost_observation(cost_cohort),
            }
        )

    applicability_errors = _applicability_errors(
        hypothesis_id=hypothesis_id,
        side=side,
        anchor=anchor,
        route=route,
    )
    if applicability_errors:
        return _seal_vehicle_row(
            {
                **base,
                "status": "UNSCORABLE_CAUSAL_INPUT",
                "reasons": applicability_errors,
                "diagnostic_vehicle_available": False,
                "scorecard_eligible": False,
                "scorecard_ineligibility_reasons": _combined_reasons(
                    proof_eligibility["scorecard_ineligibility_reasons"],
                    applicability_errors,
                ),
                "scoring_vehicle_available": False,
                "execution": None,
                "cost_observation": _empty_cost_observation(cost_cohort),
            }
        )

    if definition.execution_kind == "FROZEN_BASE_ARM_PROXY":
        binding, reasons = _base_proxy_binding(
            seat=seat,
            side=side,
            method=str(definition.proxy_method or ""),
            activation_at=activation_at,
        )
        diagnostic_available = binding is not None
        scoring_available = bool(
            diagnostic_available and proof_eligibility["scorecard_eligible"] is True
        )
        return _seal_vehicle_row(
            {
                **base,
                "status": (
                    "FROZEN_BASE_PROXY_READY"
                    if scoring_available
                    else (
                        "FROZEN_BASE_PROXY_DIAGNOSTIC_ONLY"
                        if diagnostic_available
                        else "UNSCORABLE_CAUSAL_INPUT"
                    )
                ),
                "reasons": reasons,
                "diagnostic_vehicle_available": diagnostic_available,
                "scorecard_eligible": scoring_available,
                "scorecard_ineligibility_reasons": _combined_reasons(
                    proof_eligibility["scorecard_ineligibility_reasons"],
                    [] if diagnostic_available else reasons,
                ),
                "scoring_vehicle_available": scoring_available,
                "execution": None,
                "proxy_binding": binding,
                "cost_observation": _empty_cost_observation(cost_cohort),
            }
        )

    execution, reasons = _exact_execution(
        pair=pair,
        hypothesis_id=hypothesis_id,
        side=side,
        feature_snapshot=feature_snapshot,
        anchor=anchor,
        seat=seat,
        activation_at=activation_at,
    )
    cost_observation = (
        _execution_cost_observation(
            execution=execution,
            pair=pair,
            cost_cohort=cost_cohort,
        )
        if execution is not None
        else _empty_cost_observation(cost_cohort)
    )
    diagnostic_available = execution is not None
    scoring_available = bool(
        diagnostic_available and proof_eligibility["scorecard_eligible"] is True
    )
    return _seal_vehicle_row(
        {
            **base,
            "status": (
                "EXACT_STOP_READY"
                if scoring_available
                else (
                    "EXACT_STOP_DIAGNOSTIC_ONLY"
                    if diagnostic_available
                    else "UNSCORABLE_CAUSAL_INPUT"
                )
            ),
            "reasons": reasons,
            "diagnostic_vehicle_available": diagnostic_available,
            "scorecard_eligible": scoring_available,
            "scorecard_ineligibility_reasons": _combined_reasons(
                proof_eligibility["scorecard_ineligibility_reasons"],
                [] if diagnostic_available else reasons,
            ),
            "scoring_vehicle_available": scoring_available,
            "execution": execution,
            "cost_observation": cost_observation,
        }
    )


def _exact_execution(
    *,
    pair: str,
    hypothesis_id: str,
    side: str,
    feature_snapshot: Mapping[str, Any],
    anchor: Mapping[str, Any],
    seat: Mapping[str, Any],
    activation_at: datetime,
) -> tuple[dict[str, Any] | None, list[str]]:
    tick = _price_tick(pair)
    pip_size = Decimal(1) / Decimal(instrument_pip_factor(pair))
    quote_bid = _decimal_number(seat.get("quote_bid"))
    quote_ask = _decimal_number(seat.get("quote_ask"))
    if quote_bid is None or quote_ask is None or quote_ask <= quote_bid:
        return None, ["FROZEN_EXECUTABLE_QUOTE_INVALID"]

    timeframe = "M5" if hypothesis_id == "H07" else "M1"
    entry_indicator = (
        "close"
        if hypothesis_id in {"H02", "H06"}
        else ("donchian_high" if side == "LONG" else "donchian_low")
    )
    entry_indicator_value = _indicator_decimal(
        feature_snapshot, timeframe, entry_indicator
    )
    if entry_indicator_value is None:
        return None, [f"{timeframe}_{entry_indicator.upper()}_MISSING"]

    entry_sources: list[tuple[str, Decimal]] = [
        (
            "FROZEN_QUOTE_ASK" if side == "LONG" else "FROZEN_QUOTE_BID",
            quote_ask if side == "LONG" else quote_bid,
        ),
        (f"{timeframe}_{entry_indicator.upper()}", entry_indicator_value),
    ]
    rail = anchor["rail"]
    upper = _decimal_number(rail.get("upper"))
    lower = _decimal_number(rail.get("lower"))
    width = _decimal_number(rail.get("width"))
    buffer = _decimal_number(rail.get("buffer"))
    if any(item is None for item in (upper, lower, width, buffer)):
        return None, ["FROZEN_EPISODE_RAIL_INVALID"]
    assert (
        upper is not None
        and lower is not None
        and width is not None
        and buffer is not None
    )
    if hypothesis_id in {"H04", "H07"}:
        rail_trigger = upper + buffer if side == "LONG" else lower - buffer
        entry_sources.append(
            (
                "RAIL_UPPER_PLUS_BUFFER"
                if side == "LONG"
                else "RAIL_LOWER_MINUS_BUFFER",
                rail_trigger,
            )
        )

    if side == "LONG":
        entry = _ceil_tick(max(value for _, value in entry_sources) + tick, tick)
    else:
        entry = _floor_tick(min(value for _, value in entry_sources) - tick, tick)

    geometry_inputs: list[dict[str, Any]] = [
        {"source": label, "value": _price_float(value, tick)}
        for label, value in entry_sources
    ]
    geometry_distance_cohort: dict[str, Any]
    if hypothesis_id in {"H01", "H02"}:
        m15_atr = _indicator_decimal(feature_snapshot, "M15", "atr_pips")
        m5_atr = _decimal_number(seat.get("m5_atr_pips"))
        reasons = []
        if m15_atr is None or m15_atr <= 0:
            reasons.append("M15_ATR_PIPS_MISSING_OR_NONPOSITIVE")
        if m5_atr is None or m5_atr <= 0:
            reasons.append("LEARNING_SEAT_M5_ATR_PIPS_MISSING_OR_NONPOSITIVE")
        if reasons:
            return None, reasons
        assert m15_atr is not None and m5_atr is not None
        tp_distance = m15_atr * pip_size
        sl_distance = m5_atr * pip_size
        take_profit = _nearest_tick(
            entry + tp_distance if side == "LONG" else entry - tp_distance,
            tick,
        )
        stop_loss = _nearest_tick(
            entry - sl_distance if side == "LONG" else entry + sl_distance,
            tick,
        )
        geometry_inputs.extend(
            (
                {"source": "M15_ATR_PIPS", "value": float(m15_atr)},
                {"source": "LEARNING_SEAT_M5_ATR_PIPS", "value": float(m5_atr)},
            )
        )
        geometry_distance_cohort = {
            "take_profit_measure": "FROZEN_M15_ATR_PIPS",
            "stop_loss_measure": "FROZEN_LEARNING_SEAT_M5_ATR_PIPS",
            "measure_use_policy": "ONE_NATIVE_MEASURE_USED_DIRECTLY",
            "tuned_numeric_multiplier": None,
            "fixed_pip_distance": None,
        }
    elif hypothesis_id in {"H04", "H07"}:
        if width <= 0:
            return None, ["FROZEN_EPISODE_RAIL_WIDTH_NONPOSITIVE"]
        take_profit = _nearest_tick(
            entry + width if side == "LONG" else entry - width,
            tick,
        )
        stop_loss = _nearest_tick(
            upper - buffer if side == "LONG" else lower + buffer,
            tick,
        )
        geometry_inputs.extend(
            (
                {"source": "EPISODE_RAIL_WIDTH", "value": float(width)},
                {
                    "source": (
                        "RAIL_UPPER_MINUS_BUFFER"
                        if side == "LONG"
                        else "RAIL_LOWER_PLUS_BUFFER"
                    ),
                    "value": _price_float(
                        upper - buffer if side == "LONG" else lower + buffer,
                        tick,
                    ),
                },
            )
        )
        geometry_distance_cohort = {
            "take_profit_measure": "EXACT_FROZEN_EPISODE_RAIL_WIDTH",
            "stop_loss_measure": "EXACT_FROZEN_BRANCH_INVALIDATION_BOUNDARY",
            "measure_use_policy": "ONE_EXACT_RAIL_MEASURE_USED_DIRECTLY",
            "tuned_numeric_multiplier": None,
            "fixed_pip_distance": None,
        }
    elif hypothesis_id == "H06":
        mean = _indicator_decimal(feature_snapshot, "M1", "bb_middle")
        donchian = _indicator_decimal(
            feature_snapshot,
            "M1",
            "donchian_low" if side == "LONG" else "donchian_high",
        )
        reasons = []
        if mean is None:
            reasons.append("M1_BB_MIDDLE_MISSING")
        if donchian is None:
            reasons.append("M1_OPPOSITE_DONCHIAN_EXTREME_MISSING")
        if reasons:
            return None, reasons
        assert mean is not None and donchian is not None
        take_profit = _nearest_tick(mean, tick)
        stop_loss = _nearest_tick(
            donchian - tick if side == "LONG" else donchian + tick,
            tick,
        )
        geometry_inputs.extend(
            (
                {"source": "M1_BB_MIDDLE", "value": _price_float(mean, tick)},
                {
                    "source": (
                        "M1_DONCHIAN_LOW_MINUS_BROKER_TICK"
                        if side == "LONG"
                        else "M1_DONCHIAN_HIGH_PLUS_BROKER_TICK"
                    ),
                    "value": _price_float(stop_loss, tick),
                },
            )
        )
        geometry_distance_cohort = {
            "take_profit_measure": "EXACT_FROZEN_M1_BOLLINGER_MIDDLE",
            "stop_loss_measure": "EXACT_FROZEN_M1_DONCHIAN_INVALIDATION",
            "measure_use_policy": "ABSOLUTE_FROZEN_TECHNICAL_LEVELS",
            "tuned_numeric_multiplier": None,
            "fixed_pip_distance": None,
        }
    else:
        return None, ["EXACT_VEHICLE_FORMULA_NOT_IMPLEMENTED"]

    ordering_errors = _geometry_ordering_errors(
        side=side,
        entry=entry,
        take_profit=take_profit,
        stop_loss=stop_loss,
    )
    if ordering_errors:
        return None, ordering_errors
    expiry = activation_at + timedelta(seconds=BASE_ENTRY_TTL_SECONDS)
    maturity = expiry + timedelta(seconds=BASE_MAX_HOLD_SECONDS)
    return (
        {
            "vehicle_type": "ATTACHED_BARRIER_STOP_GTD",
            "order_type": "STOP",
            "time_in_force": "GTD",
            "price_bound": None,
            "entry_price": _price_float(entry, tick),
            "take_profit_price": _price_float(take_profit, tick),
            "stop_loss_price": _price_float(stop_loss, tick),
            "entry_ttl_seconds": BASE_ENTRY_TTL_SECONDS,
            "max_hold_seconds": BASE_MAX_HOLD_SECONDS,
            "activation_at_utc": activation_at.isoformat(),
            "entry_expires_at_utc": expiry.isoformat(),
            "latest_maturity_at_utc": maturity.isoformat(),
            "natural_trigger_component": "ASK" if side == "LONG" else "BID",
            "exit_component": "BID" if side == "LONG" else "ASK",
            "entry_gap_policy": EXACT_STOP_GAP_POLICY_V2,
            "barrier_gap_policy": EXACT_STOP_GAP_POLICY_V2,
            "intrabar_policy": EXACT_STOP_INTRABAR_POLICY_V2,
            "hold_end_policy": EXACT_STOP_HOLD_POLICY_V2,
            "missing_s5_policy": "NO_SYNTHETIC_CANDLE",
            "native_market_unit_cohort_policy": NATIVE_MARKET_UNIT_COHORT_POLICY_V2,
            "geometry_distance_cohort": geometry_distance_cohort,
            "frozen_geometry_inputs": geometry_inputs,
        },
        [],
    )


def _base_proxy_binding(
    *, seat: Mapping[str, Any], side: str, method: str, activation_at: datetime
) -> tuple[dict[str, Any] | None, list[str]]:
    candidates = [
        candidate
        for candidate in seat.get("candidates", [])
        if isinstance(candidate, Mapping)
        and candidate.get("side") == side
        and candidate.get("method") == method
    ]
    if len(candidates) != 1:
        return None, ["EXACT_SIDE_METHOD_CANDIDATE_NOT_UNIQUE"]
    candidate = candidates[0]
    arms = [
        arm
        for arm in candidate.get("arms", [])
        if isinstance(arm, Mapping) and arm.get("arm_id") == "BASE"
    ]
    if len(arms) != 1:
        return None, ["FROZEN_BASE_ARM_NOT_UNIQUE"]
    arm = arms[0]
    return (
        {
            "binding_type": "EXACT_FROZEN_LEARNING_SEAT_BASE_ARM_REFERENCE",
            "learning_seat_contract_sha256": str(seat["contract_sha256"]),
            "learning_seat_id": str(seat["seat_id"]),
            "candidate_id": str(candidate["candidate_id"]),
            "candidate_sha256": str(candidate["candidate_sha256"]),
            "side": side,
            "method": method,
            "arm_id": "BASE",
            "arm_sha256": _canonical_sha(dict(arm)),
            "activation_at_utc": activation_at.isoformat(),
            "scoring_start_at_utc": activation_at.isoformat(),
            "path_lower_bound_policy": "S5_INTERVAL_START_GTE_SCORING_START_AT_UTC",
            "pre_activation_price_path_allowed": False,
            "numeric_geometry_embedded": False,
            "source_resolution_required": True,
            "gap_policy": BASE_PROXY_GAP_POLICY_V2,
            "intrabar_policy": BASE_PROXY_INTRABAR_POLICY_V2,
        },
        [],
    )


def _applicability_errors(
    *,
    hypothesis_id: str,
    side: str,
    anchor: Mapping[str, Any],
    route: Mapping[str, Any],
) -> list[str]:
    if hypothesis_id not in {"H04", "H07"}:
        return []
    attempt = str(anchor.get("attempt_direction") or "")
    expected_side = "LONG" if attempt == "UP" else "SHORT" if attempt == "DOWN" else ""
    errors = []
    if route.get("branch_outcome") != "ACCEPTED":
        errors.append("ACCEPTED_BRANCH_REQUIRED")
    if route.get("route_family") != "BREAKOUT_CONTINUATION":
        errors.append("BREAKOUT_CONTINUATION_ROUTE_REQUIRED")
    if side != expected_side or route.get("trade_side") != expected_side:
        errors.append("ATTEMPT_DIRECTION_SIDE_MISMATCH")
    return errors


def _causal_source_errors(
    *,
    technical_feature_snapshot: Mapping[str, Any],
    technical_hypothesis_shadow: Mapping[str, Any],
    episode_anchor: Mapping[str, Any],
    episode_route: Mapping[str, Any],
    learning_seat: Mapping[str, Any],
    confirmed_at_utc: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(learning_seat, Mapping):
        return ["LEARNING_SEAT_NOT_MAPPING"]
    try:
        if not _learning_seat_valid(learning_seat):
            errors.append("LEARNING_SEAT_INVALID")
    except (KeyError, TypeError, ValueError, OverflowError):
        errors.append("LEARNING_SEAT_INVALID")
    if learning_seat.get("contract") != LEARNING_SEAT_CONTRACT:
        errors.append("LEARNING_SEAT_CONTRACT_UNSUPPORTED")
    errors.extend(_anchor_route_errors(episode_anchor, episode_route))
    confirmed = _parse_utc(confirmed_at_utc)
    if confirmed is None:
        errors.append("CONFIRMED_AT_INVALID")
    if not isinstance(technical_feature_snapshot, Mapping):
        errors.append("TECHNICAL_FEATURE_SNAPSHOT_NOT_MAPPING")
        return errors
    if not isinstance(technical_hypothesis_shadow, Mapping):
        errors.append("TECHNICAL_HYPOTHESIS_SHADOW_NOT_MAPPING")
        return errors
    pair = str(technical_feature_snapshot.get("pair") or "")
    if not pair or learning_seat.get("pair") != pair:
        errors.append("PAIR_BINDING_MISMATCH")
    generated = _parse_utc(learning_seat.get("generated_at_utc"))
    quote_timestamp = _parse_utc(learning_seat.get("quote_timestamp_utc"))
    snapshot_cycle = _parse_utc(
        technical_feature_snapshot.get("handoff_cycle_generated_at_utc")
    )
    m1_close = _parse_utc(learning_seat.get("m1_closed_candle_utc"))
    feature_m1_close = _feature_timeframe_close(
        technical_feature_snapshot,
        "M1",
    )
    if generated is None or quote_timestamp is None or snapshot_cycle is None:
        errors.append("FROZEN_ACTIVATION_CLOCK_INVALID")
    elif generated != snapshot_cycle or quote_timestamp > generated:
        errors.append("FROZEN_ACTIVATION_CLOCK_BINDING_INVALID")
    if generated is not None and m1_close is not None and m1_close > generated:
        errors.append("LEARNING_SEAT_M1_CLOSE_AFTER_GENERATION")
    if confirmed is not None and generated is not None and confirmed > generated:
        errors.append("CONFIRMED_AT_AFTER_SEAT_GENERATION")
    if confirmed is not None and m1_close != confirmed:
        errors.append("CONFIRMED_AT_M1_CLOSE_BINDING_INVALID")
    if confirmed is not None and feature_m1_close != confirmed:
        errors.append("FEATURE_M1_CLOSE_CONFIRMED_AT_BINDING_INVALID")
    if confirmed is not None and isinstance(episode_anchor, Mapping):
        attempt_at = _parse_utc(episode_anchor.get("attempt_candle_utc"))
        setup_at = _parse_utc(episode_anchor.get("setup_candle_utc"))
        if (
            setup_at is None
            or attempt_at is None
            or not setup_at < attempt_at < confirmed
        ):
            errors.append("EPISODE_ANCHOR_CLOCK_ORDER_INVALID")
    if confirmed is not None and isinstance(episode_route, Mapping):
        branch_at = _parse_utc(episode_route.get("branch_candle_utc"))
        if branch_at is None or branch_at >= confirmed:
            errors.append("EPISODE_ROUTE_CLOCK_ORDER_INVALID")
    if isinstance(episode_anchor, Mapping) and isinstance(episode_route, Mapping):
        try:
            shadow_valid = technical_hypothesis_shadow_valid(
                technical_hypothesis_shadow,
                feature_snapshot=technical_feature_snapshot,
                attempt_direction=str(episode_anchor.get("attempt_direction") or ""),
                branch_outcome=str(episode_route.get("branch_outcome") or ""),
                route_family=str(episode_route.get("route_family") or ""),
                spread_pips=float(learning_seat["executable_spread_pips"]),
                m5_atr_pips=float(learning_seat["m5_atr_pips"]),
                spread_to_m5_atr=float(learning_seat["spread_to_m5_atr"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            shadow_valid = False
        if not shadow_valid:
            errors.append("TECHNICAL_HYPOTHESIS_SHADOW_BINDING_INVALID")
    rows = technical_hypothesis_shadow.get("hypotheses")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        errors.append("TECHNICAL_HYPOTHESIS_ROWS_INVALID")
    else:
        ids = [
            str(row.get("hypothesis_id") or "")
            for row in rows
            if isinstance(row, Mapping)
        ]
        if ids != list(_HYPOTHESIS_IDS):
            errors.append("TECHNICAL_HYPOTHESIS_ID_SET_INVALID")
    return errors


def _anchor_route_errors(
    anchor: Mapping[str, Any], route: Mapping[str, Any]
) -> list[str]:
    if not isinstance(anchor, Mapping) or set(anchor) != _ANCHOR_KEYS:
        return ["EPISODE_ANCHOR_SCHEMA_INVALID"]
    if not isinstance(route, Mapping) or set(route) != _ROUTE_KEYS:
        return ["EPISODE_ROUTE_SCHEMA_INVALID"]
    errors: list[str] = []
    if not bool(
        anchor.get("episode_kind") == EPISODE_KIND
        and anchor.get("episode_rule_version") == EPISODE_RULE_VERSION
        and anchor.get("horizon_lane") == HORIZON_LANE
        and anchor.get("confirmation_ttl_seconds") == CONFIRMATION_TTL_SECONDS
        and anchor.get("attempt_direction") in {"UP", "DOWN"}
        and isinstance(anchor.get("source_evidence_sha256"), str)
        and _SHA256_RE.fullmatch(str(anchor.get("source_evidence_sha256"))) is not None
    ):
        errors.append("EPISODE_ANCHOR_SEMANTICS_INVALID")
    rail = anchor.get("rail")
    if not isinstance(rail, Mapping) or set(rail) != _RAIL_KEYS:
        errors.append("EPISODE_RAIL_SCHEMA_INVALID")
    else:
        upper = _number(rail.get("upper"))
        lower = _number(rail.get("lower"))
        width = _number(rail.get("width"))
        buffer = _number(rail.get("buffer"))
        ratio = _number(rail.get("buffer_ratio"))
        if not bool(
            upper is not None
            and lower is not None
            and width is not None
            and buffer is not None
            and ratio is not None
            and upper > lower
            and width > 0.0
            and buffer > 0.0
            and ratio > 0.0
            and math.isclose(
                width,
                upper - lower,
                rel_tol=0.0,
                abs_tol=SEALED_RAIL_FLOAT_ABS_TOLERANCE,
            )
            and math.isclose(
                buffer,
                width * ratio,
                rel_tol=0.0,
                abs_tol=SEALED_RAIL_FLOAT_ABS_TOLERANCE,
            )
        ):
            errors.append("EPISODE_RAIL_SEMANTICS_INVALID")
    attempt = str(anchor.get("attempt_direction") or "")
    branch = str(route.get("branch_outcome") or "")
    expected_side = "LONG" if attempt == "UP" else "SHORT"
    expected = {
        "ACCEPTED": (
            expected_side,
            ["TREND_CONTINUATION"],
            "BREAKOUT_CONTINUATION",
        ),
        "REJECTED": (
            "SHORT" if expected_side == "LONG" else "LONG",
            ["BREAKOUT_FAILURE", "RANGE_ROTATION"],
            "RANGE_RECLAIM_OR_BREAKOUT_FAILURE",
        ),
    }.get(branch)
    branch_close = _number(route.get("branch_close"))
    if not bool(
        expected is not None
        and route.get("trade_side") == expected[0]
        and route.get("candidate_methods") == expected[1]
        and route.get("route_family") == expected[2]
        and route.get("selection_status") == SELECTION_STATUS
        and branch_close is not None
    ):
        errors.append("EPISODE_ROUTE_SEMANTICS_INVALID")
    return errors


def _geometry_ordering_errors(
    *, side: str, entry: Decimal, take_profit: Decimal, stop_loss: Decimal
) -> list[str]:
    errors = []
    if side == "LONG":
        if take_profit <= entry:
            errors.append("TAKE_PROFIT_NOT_FAVORABLE")
        if stop_loss >= entry:
            errors.append("STOP_LOSS_NOT_ADVERSE")
    else:
        if take_profit >= entry:
            errors.append("TAKE_PROFIT_NOT_FAVORABLE")
        if stop_loss <= entry:
            errors.append("STOP_LOSS_NOT_ADVERSE")
    return errors


def _execution_cost_observation(
    *,
    execution: Mapping[str, Any],
    pair: str,
    cost_cohort: Mapping[str, Any],
) -> dict[str, Any]:
    gross = abs(
        float(execution["take_profit_price"]) - float(execution["entry_price"])
    ) * float(instrument_pip_factor(pair))
    spread = float(cost_cohort["executable_spread_pips"])
    return {
        "executable_spread_pips": round(spread, PIP_METRIC_SERIALIZATION_DECIMALS),
        "gross_take_profit_pips": round(gross, PIP_METRIC_SERIALIZATION_DECIMALS),
        "spread_to_gross_take_profit": round(
            spread / gross, PIP_METRIC_SERIALIZATION_DECIMALS
        ),
        "cost_is_applicability_gate": False,
    }


def _empty_cost_observation(cost_cohort: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "executable_spread_pips": float(cost_cohort["executable_spread_pips"]),
        "gross_take_profit_pips": None,
        "spread_to_gross_take_profit": None,
        "cost_is_applicability_gate": False,
    }


def _indicator_decimal(
    snapshot: Mapping[str, Any], timeframe: str, key: str
) -> Decimal | None:
    rows = snapshot.get("timeframes")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    matched = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("timeframe") == timeframe
    ]
    if len(matched) != 1 or not isinstance(matched[0].get("indicators"), Mapping):
        return None
    return _decimal_number(matched[0]["indicators"].get(key))


def _feature_timeframe_close(
    snapshot: Mapping[str, Any], timeframe: str
) -> datetime | None:
    rows = snapshot.get("timeframes")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    matched = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("timeframe") == timeframe
    ]
    if len(matched) != 1:
        return None
    return _parse_utc(matched[0].get("complete_candle_close_utc"))


def _price_tick(pair: str) -> Decimal:
    return Decimal(1) / (
        Decimal(instrument_pip_factor(pair)) * Decimal(OANDA_FX_TICKS_PER_PIP)
    )


def _ceil_tick(value: Decimal, tick: Decimal) -> Decimal:
    return (value / tick).to_integral_value(rounding=ROUND_CEILING) * tick


def _floor_tick(value: Decimal, tick: Decimal) -> Decimal:
    return (value / tick).to_integral_value(rounding=ROUND_FLOOR) * tick


def _nearest_tick(value: Decimal, tick: Decimal) -> Decimal:
    return (value / tick).to_integral_value(rounding=ROUND_HALF_UP) * tick


def _price_float(value: Decimal, tick: Decimal) -> float:
    return float(value.quantize(tick))


def _decimal_number(value: Any) -> Decimal | None:
    parsed = _number(value)
    return Decimal(str(parsed)) if parsed is not None else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _definition_dict(
    value: TechnicalHypothesisVehicleDefinitionV2,
) -> dict[str, Any]:
    return asdict(value)


def _zero_authority() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "shadow_only": True,
        "order_authority": "NONE",
        "primary_effect": False,
        "risk_effect": False,
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_intents": [],
    }


def _seal_vehicle_row(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["diagnostic_evaluable"] = body.get("diagnostic_vehicle_available") is True
    body["scoring_vehicle"] = body.get("scoring_vehicle_available") is True
    return {**body, "vehicle_sha256": _canonical_sha(body)}


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    return {**body, "contract_sha256": _canonical_sha(body)}


def _sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    if value.get("contract") != contract:
        return False
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    try:
        return value.get("contract_sha256") == _canonical_sha(body)
    except (TypeError, ValueError, OverflowError):
        return False


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CATALOG_CONTRACT_V2",
    "CAUSAL_SOURCE_PATHS_V2",
    "NATIVE_MARKET_UNIT_COHORT_POLICY_V2",
    "SHADOW_CONTRACT_V2",
    "TECHNICAL_HYPOTHESIS_VEHICLE_CATALOG_V2",
    "VEHICLE_CONTRACT_V2",
    "VEHICLE_POLICY_V2",
    "build_fast_bot_technical_hypothesis_vehicles_v2",
    "technical_hypothesis_vehicle_catalog_v2",
    "technical_hypothesis_vehicle_shadow_v2_valid",
]

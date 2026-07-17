"""Causal, historical-only technical-grid research contracts.

This module deliberately stops before orchestration.  It freezes the finite
H01..H07 x DIRECT/INVERSE x OFAT-arm universe, validates a complete closed-M1
multi-timeframe feature snapshot, freezes one executable signal, resolves that
signal on exact sparse OANDA S5 bid/ask candles, and creates train/validation
selection receipts.  It never reads a holdout outcome, places an order, changes
runtime strategy state, or grants promotion/live authority.

The existing technical-hypothesis V1 and vehicle V2 contracts are immutable.
The contracts here are a separate research layer; in particular, executable
time-close is not a reinterpretation of V2's conservative full-stop-at-hold
evidence policy.  This module is not a strategy evaluator: its source SHA and
aggregate metric rows are caller assertions, so its receipts are structural
dry-run evidence and cannot establish profitability until an upstream causal
hypothesis/vehicle receipt and metric provenance contract are bound and
validated.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping, Sequence

from quant_rabbit.fast_bot_technical_hypotheses import (
    ENUM_INDICATOR_FEATURES,
    FEATURE_SNAPSHOT_CONTRACT,
    HYPOTHESIS_FAMILIES,
    INDICATOR_SERIES_FEATURES,
    MAX_TECHNICAL_FEATURE_SNAPSHOT_BYTES,
    MARKET_STATE_ENUM_FEATURES,
    MAX_INDICATOR_SERIES_VALUES,
    NUMERIC_INDICATOR_FEATURES,
    TIMEFRAMES,
    technical_hypothesis_catalog,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


GRID_CATALOG_CONTRACT_V1 = "QR_FAST_BOT_TECHNICAL_GRID_CATALOG_V1"
GRID_CANDIDATE_CONTRACT_V1 = "QR_FAST_BOT_TECHNICAL_GRID_CANDIDATE_V1"
GRID_SIGNAL_CONTRACT_V1 = "QR_FAST_BOT_TECHNICAL_GRID_SIGNAL_V1"
TIME_CLOSE_OUTCOME_CONTRACT_V1 = "QR_FAST_BOT_TECHNICAL_GRID_TIME_CLOSE_OUTCOME_V1"
SPLIT_RECEIPT_CONTRACT_V1 = "QR_FAST_BOT_TECHNICAL_GRID_SPLIT_RECEIPT_V1"
SELECTION_RECEIPT_CONTRACT_V1 = "QR_FAST_BOT_TECHNICAL_GRID_SELECTION_RECEIPT_V1"

GRID_POLICY_V1 = "H01_H07_DIRECT_INVERSE_13_OFAT_V1"
TIME_CLOSE_POLICY_V1 = "EXACT_S5_BID_ASK_EXECUTABLE_TIME_CLOSE_SL_FIRST_V1"
SELECTION_POLICY_V1 = (
    "TRAIN_POSITIVE_ECONOMICS_VALIDATION_HOLM_ONE_SE_SIMPLEST_PER_FAMILY_V1"
)
SPLIT_POLICY_V1 = "EMISSION_CLOCK_WITH_LATEST_MATURITY_PURGE_AND_UNOPENED_HOLDOUT_V1"

ORIENTATIONS = ("DIRECT", "INVERSE")
ORDER_TYPES = frozenset({"STOP", "LIMIT"})
S5_SECONDS = 5
PLANNED_DIRECTIONAL_FAMILIES = 7
PLANNED_ARM_COUNT = 13
PLANNED_CANDIDATE_COUNT = (
    PLANNED_DIRECTIONAL_FAMILIES * len(ORIENTATIONS) * PLANNED_ARM_COUNT
)
SELECTION_ALPHA = 0.05
MIN_SELECTION_CLUSTERS = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_SIGNAL_IDENTITY_KEYS = (
    "contract",
    "schema_version",
    "candidate_id",
    "candidate_sha256",
    "catalog_contract_sha256",
    "pair",
    "hypothesis_id",
    "orientation",
    "source_predicted_side",
    "side",
    "arm",
    "order_type",
    "activation_at_utc",
    "entry_expires_at_utc",
    "latest_maturity_at_utc",
    "entry_price",
    "take_profit_price",
    "stop_loss_price",
    "base_take_profit_price",
    "base_stop_loss_price",
    "natural_entry_component",
    "exit_component",
    "entry_gap_policy",
    "same_s5_policy",
    "hold_policy",
    "missing_s5_policy",
    "causal_source_sha256",
)

_ZERO_AUTHORITY_SCALARS = {
    "historical_only": True,
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


def _zero_authority() -> dict[str, Any]:
    """Return a fresh authority payload so artifacts never share a list."""

    return {**_ZERO_AUTHORITY_SCALARS, "order_intents": []}


@dataclass(frozen=True, slots=True)
class TechnicalGridArm:
    """One predeclared one-factor-at-a-time execution arm.

    The numeric values mirror the already-published learning-arm policy.  They
    are a finite research universe, not a production geometry recommendation.
    BASE supplies the unchanged native H vehicle geometry; every other arm
    changes exactly one axis.
    """

    arm_id: str
    take_profit_multiplier: float
    stop_loss_multiplier: float
    entry_ttl_seconds: int
    max_hold_seconds: int
    changed_axis: str
    complexity_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "take_profit_multiplier": self.take_profit_multiplier,
            "stop_loss_multiplier": self.stop_loss_multiplier,
            "entry_ttl_seconds": self.entry_ttl_seconds,
            "max_hold_seconds": self.max_hold_seconds,
            "changed_axis": self.changed_axis,
            "complexity_rank": self.complexity_rank,
        }


@dataclass(frozen=True, slots=True)
class CausalTimeframeFeature:
    """Already-computed features bound to one complete candle close."""

    timeframe: str
    complete_candle_close_utc: datetime
    market_state: Mapping[str, Any]
    indicators: Mapping[str, Any]
    indicator_series: Mapping[str, Sequence[float]]


_ARMS_V1 = (
    TechnicalGridArm("BASE", 1.00, 1.00, 90, 900, "BASELINE", 0),
    TechnicalGridArm("TP050", 0.50, 1.00, 90, 900, "TAKE_PROFIT", 1),
    TechnicalGridArm("TP075", 0.75, 1.00, 90, 900, "TAKE_PROFIT", 1),
    TechnicalGridArm("TP125", 1.25, 1.00, 90, 900, "TAKE_PROFIT", 1),
    TechnicalGridArm("SL075", 1.00, 0.75, 90, 900, "STOP_LOSS", 1),
    TechnicalGridArm("SL125", 1.00, 1.25, 90, 900, "STOP_LOSS", 1),
    TechnicalGridArm("SL150", 1.00, 1.50, 90, 900, "STOP_LOSS", 1),
    TechnicalGridArm("HOLD180", 1.00, 1.00, 90, 180, "HOLD_HORIZON", 1),
    TechnicalGridArm("HOLD300", 1.00, 1.00, 90, 300, "HOLD_HORIZON", 1),
    TechnicalGridArm("HOLD600", 1.00, 1.00, 90, 600, "HOLD_HORIZON", 1),
    TechnicalGridArm("HOLD1800", 1.00, 1.00, 90, 1800, "HOLD_HORIZON", 1),
    TechnicalGridArm("TTL45", 1.00, 1.00, 45, 900, "ENTRY_TTL", 1),
    TechnicalGridArm("TTL180", 1.00, 1.00, 180, 900, "ENTRY_TTL", 1),
)


def technical_grid_arms_v1() -> tuple[TechnicalGridArm, ...]:
    """Return the immutable, ordered 13-arm OFAT grid."""

    return _ARMS_V1


def build_fast_bot_technical_grid_catalog_v1() -> dict[str, Any]:
    """Freeze H01..H07 x two orientations x 13 arms plus H08 control."""

    source_catalog = technical_hypothesis_catalog()
    hypotheses = source_catalog.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) != 8:
        raise ValueError("technical hypothesis catalog does not contain H01..H08")
    directional = hypotheses[:7]
    if [row.get("hypothesis_id") for row in directional] != [
        f"H{index:02d}" for index in range(1, 8)
    ] or hypotheses[-1].get("hypothesis_id") != "H08":
        raise ValueError("technical hypothesis catalog order is invalid")

    candidates: list[dict[str, Any]] = []
    order = 0
    for hypothesis in directional:
        hypothesis_id = str(hypothesis["hypothesis_id"])
        hypothesis_definition_sha = _canonical_sha(hypothesis)
        for orientation_rank, orientation in enumerate(ORIENTATIONS):
            for arm in _ARMS_V1:
                candidate_id = f"{hypothesis_id}:{orientation}:{arm.arm_id}"
                body = {
                    "contract": GRID_CANDIDATE_CONTRACT_V1,
                    "schema_version": 1,
                    "candidate_id": candidate_id,
                    "catalog_order": order,
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_definition_sha256": hypothesis_definition_sha,
                    "family": str(hypothesis["family"]),
                    "orientation": orientation,
                    "orientation_rank": orientation_rank,
                    "arm": arm.to_dict(),
                    "complexity_rank": arm.complexity_rank,
                    "exit_contract": "EXECUTABLE_BIDASK_TIME_CLOSE_V1",
                    **_zero_authority(),
                }
                candidates.append({**body, "candidate_sha256": _canonical_sha(body)})
                order += 1
    if len(candidates) != PLANNED_CANDIDATE_COUNT:
        raise AssertionError("technical grid candidate count drifted")

    h08 = dict(hypotheses[-1])
    control = {
        "hypothesis_id": "H08",
        "family": str(h08["family"]),
        "control_type": "NO_TRADE_ZERO_PNL",
        "post_cost_realized_pips": 0.0,
        "planned_candidate": False,
        "multiple_testing_denominator_member": False,
        "source_definition_sha256": _canonical_sha(h08),
        **_zero_authority(),
    }
    body = {
        "contract": GRID_CATALOG_CONTRACT_V1,
        "schema_version": 1,
        "grid_policy": GRID_POLICY_V1,
        "source_hypothesis_catalog_sha256": source_catalog["contract_sha256"],
        "maximum_one_regime_gate": True,
        "maximum_one_direction_core": True,
        "maximum_one_optional_confirmer_group": True,
        "correlated_indicator_majority_vote_forbidden": True,
        "orientations": list(ORIENTATIONS),
        "arms": [arm.to_dict() for arm in _ARMS_V1],
        "planned_directional_family_count": PLANNED_DIRECTIONAL_FAMILIES,
        "planned_arm_count": PLANNED_ARM_COUNT,
        "planned_candidate_count": len(candidates),
        "multiple_testing_denominator": len(candidates),
        "candidate_order_policy": "HYPOTHESIS_THEN_DIRECT_INVERSE_THEN_OFAT_ORDER",
        "candidates": candidates,
        "h08_control": control,
        "full_stop_hold_policy_role": "PARITY_STRESS_ONLY_NOT_WINNER_SELECTION",
        "primary_exit_policy": TIME_CLOSE_POLICY_V1,
        "strategy_evaluator_included": False,
        "economic_backtest_ready": False,
        **_zero_authority(),
    }
    return _seal(body)


def build_causal_technical_feature_snapshot_v1(
    *,
    pair: str,
    decision_at_utc: datetime,
    timeframes: Sequence[CausalTimeframeFeature | Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal all seven complete-timeframe features at one closed-M1 decision.

    This function intentionally accepts already-computed features.  Historical
    S5 resampling must prove OANDA timeframe alignment separately; silently
    treating UTC H4/D buckets as broker H4/D candles would be a causal-shape
    error.  The M1 clock must equal the decision clock and every other complete
    candle must close no later than it.
    """

    pair_name = _validated_pair(pair)
    decision = _aware_utc(decision_at_utc)
    if decision.second != 0 or decision.microsecond != 0:
        raise ValueError("decision_at_utc must be an exact complete-M1 close")
    if isinstance(timeframes, (str, bytes)) or not isinstance(timeframes, Sequence):
        raise ValueError("timeframes must be a sequence")

    normalized: dict[str, CausalTimeframeFeature] = {}
    for raw in timeframes:
        item = _coerce_timeframe_feature(raw)
        timeframe = item.timeframe.upper()
        if timeframe not in TIMEFRAMES or timeframe in normalized:
            raise ValueError("timeframe feature set is unknown or duplicated")
        normalized[timeframe] = item
    if set(normalized) != set(TIMEFRAMES):
        raise ValueError("causal snapshot requires all seven timeframes")

    rows: list[dict[str, Any]] = []
    for timeframe in TIMEFRAMES:
        item = normalized[timeframe]
        close = _aware_utc(item.complete_candle_close_utc)
        if close > decision:
            raise ValueError("timeframe close exceeds the M1 decision clock")
        if timeframe == "M1" and close != decision:
            raise ValueError("M1 close must equal the decision clock")
        market = _validated_market_state(item.market_state)
        indicators = _validated_indicators(item.indicators)
        series = _validated_indicator_series(item.indicator_series)
        row_body = {
            "timeframe": timeframe,
            "complete_candle_close_utc": close.isoformat(),
            "market_state": market,
            "indicators": indicators,
            "indicator_series": series,
        }
        rows.append({**row_body, "feature_sha256": _canonical_sha(row_body)})

    body = {
        "contract": FEATURE_SNAPSHOT_CONTRACT,
        "schema_version": 1,
        "pair": pair_name,
        "handoff_cycle_generated_at_utc": decision.isoformat(),
        "feature_allowlist_version": 1,
        "timeframes": rows,
        "hypothesis_families": list(HYPOTHESIS_FAMILIES),
        "raw_chart_packet_embedded": False,
        "diagnostic_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    snapshot = _seal(body)
    if len(_canonical_json_bytes(snapshot)) > MAX_TECHNICAL_FEATURE_SNAPSHOT_BYTES:
        raise ValueError("causal feature snapshot exceeds its bounded contract")
    return snapshot


def freeze_fast_bot_technical_grid_signal_v1(
    *,
    pair: str,
    hypothesis_id: str,
    orientation: str,
    source_predicted_side: str,
    side: str,
    arm_id: str,
    order_type: str,
    activation_at_utc: datetime,
    entry_price: float,
    base_take_profit_price: float,
    base_stop_loss_price: float,
    causal_source_sha256: str,
) -> dict[str, Any]:
    """Freeze one caller-supplied actual-side primitive before later S5 prices.

    This validates the finite candidate and its execution geometry, but does
    not rebuild H01..H07 from the source snapshot.  The source SHA is therefore
    only a syntactic caller binding in this increment.
    """

    catalog = build_fast_bot_technical_grid_catalog_v1()
    pair_name = _validated_pair(pair)
    hypothesis = str(hypothesis_id).upper()
    orientation_name = str(orientation).upper()
    source_side = str(source_predicted_side).upper()
    actual_side = str(side).upper()
    order = str(order_type).upper()
    candidate_id = f"{hypothesis}:{orientation_name}:{str(arm_id).upper()}"
    candidate_by_id = {str(row["candidate_id"]): row for row in catalog["candidates"]}
    candidate = candidate_by_id.get(candidate_id)
    if candidate is None:
        raise ValueError("signal does not belong to the fixed 182-candidate catalog")
    if source_side not in {"LONG", "SHORT"} or actual_side not in {"LONG", "SHORT"}:
        raise ValueError("signal side is invalid")
    expected_side = (
        source_side if orientation_name == "DIRECT" else _opposite_side(source_side)
    )
    if actual_side != expected_side:
        raise ValueError("inverse signal must freeze independent actual-side geometry")
    if order not in ORDER_TYPES:
        raise ValueError("grid signal order_type must be STOP or LIMIT")
    activation = _aware_utc(activation_at_utc)
    if activation.second != 0 or activation.microsecond != 0:
        raise ValueError("signal activation must equal a complete-M1 close")
    source_sha = _validated_sha(causal_source_sha256, "causal_source_sha256")
    entry = _positive_finite(entry_price, "entry_price")
    base_tp = _positive_finite(base_take_profit_price, "base_take_profit_price")
    base_sl = _positive_finite(base_stop_loss_price, "base_stop_loss_price")
    if not _geometry_valid(actual_side, entry, base_tp, base_sl):
        raise ValueError("base geometry is invalid for actual signal side")

    arm = _arm_by_id(str(arm_id).upper())
    tick = _price_tick(pair_name)
    entry_decimal = Decimal(str(entry))
    if _nearest_tick(entry_decimal, tick) != entry_decimal:
        raise ValueError("entry_price is not aligned to broker pipette precision")
    tp_distance = abs(Decimal(str(base_tp)) - Decimal(str(entry)))
    sl_distance = abs(Decimal(str(entry)) - Decimal(str(base_sl)))
    scaled_tp_distance = tp_distance * Decimal(str(arm.take_profit_multiplier))
    scaled_sl_distance = sl_distance * Decimal(str(arm.stop_loss_multiplier))
    if actual_side == "LONG":
        take_profit = _nearest_tick(entry_decimal + scaled_tp_distance, tick)
        stop_loss = _nearest_tick(entry_decimal - scaled_sl_distance, tick)
    else:
        take_profit = _nearest_tick(entry_decimal - scaled_tp_distance, tick)
        stop_loss = _nearest_tick(entry_decimal + scaled_sl_distance, tick)
    frozen_entry = _nearest_tick(entry_decimal, tick)
    entry_float = float(frozen_entry)
    tp_float = float(take_profit)
    sl_float = float(stop_loss)
    if not _geometry_valid(actual_side, entry_float, tp_float, sl_float):
        raise ValueError("scaled grid geometry collapsed at broker precision")

    entry_expires = activation + timedelta(seconds=arm.entry_ttl_seconds)
    latest_maturity = entry_expires + timedelta(seconds=arm.max_hold_seconds)
    execution_fields = {
        "contract": GRID_SIGNAL_CONTRACT_V1,
        "schema_version": 1,
        "candidate_id": candidate_id,
        "candidate_sha256": candidate["candidate_sha256"],
        "pair": pair_name,
        "hypothesis_id": hypothesis,
        "orientation": orientation_name,
        "source_predicted_side": source_side,
        "side": actual_side,
        "arm": arm.to_dict(),
        "order_type": order,
        "activation_at_utc": activation.isoformat(),
        "entry_expires_at_utc": entry_expires.isoformat(),
        "latest_maturity_at_utc": latest_maturity.isoformat(),
        "entry_price": entry_float,
        "take_profit_price": tp_float,
        "stop_loss_price": sl_float,
        "base_take_profit_price": base_tp,
        "base_stop_loss_price": base_sl,
        "natural_entry_component": "ASK" if actual_side == "LONG" else "BID",
        "exit_component": "BID" if actual_side == "LONG" else "ASK",
        "entry_gap_policy": "STOP_WORSE_REAL_OPEN_LIMIT_REAL_OPEN_IMPROVEMENT",
        "same_s5_policy": (
            "FILL_CANDLE_ANY_ATTACHED_EXIT_SL_FIRST;LATER_TP_AND_SL_SL_FIRST"
        ),
        "hold_policy": "FIRST_REAL_EXECUTABLE_OPEN_AT_OR_AFTER_HOLD_BOUNDARY",
        "missing_s5_policy": "NO_SYNTHETIC_CANDLE_NEXT_REAL_OPEN_OWNS_GAP",
        "causal_source_sha256": source_sha,
        "catalog_contract_sha256": catalog["contract_sha256"],
        "causal_strategy_evaluator_binding": "CALLER_ASSERTED_SHA_ONLY",
        "strategy_evaluator_binding_verified": False,
        "economic_backtest_eligible": False,
        **_zero_authority(),
    }
    signal_id = _canonical_sha(_signal_identity_payload(execution_fields))[:24]
    body = {**execution_fields, "signal_id": signal_id}
    return {**body, "signal_sha256": _canonical_sha(body)}


def resolve_executable_bidask_time_close_v1(
    signal: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    truth_source_receipt_sha256: str,
    truth_provenance_end_utc: datetime | None = None,
) -> dict[str, Any]:
    """Resolve one frozen signal on sparse, chronological exact S5 BID/ASK.

    Entry inspection ends at the frozen TTL.  Exit inspection is not capped by
    the theoretical latest-maturity grid: a sparse market remains open until
    the first real S5 at or after the hold boundary.  A caller may provide an
    exclusive provenance/split boundary; prices at or beyond it are rejected
    and an otherwise-open position becomes explicitly unresolved.
    """

    _validate_frozen_signal(signal)
    receipt_sha = _validated_sha(
        truth_source_receipt_sha256,
        "truth_source_receipt_sha256",
    )
    activation = _parse_utc(signal["activation_at_utc"])
    expiry = _parse_utc(signal["entry_expires_at_utc"])
    latest_maturity = _parse_utc(signal["latest_maturity_at_utc"])
    provenance_end = (
        _aware_utc(truth_provenance_end_utc)
        if truth_provenance_end_utc is not None
        else None
    )
    if provenance_end is not None and provenance_end <= activation:
        raise ValueError("truth provenance boundary must follow activation")
    if (
        provenance_end is not None
        and not isinstance(candles, (str, bytes))
        and isinstance(candles, Sequence)
        and any(
            candle.__class__ is S5BidAskCandle
            and _aware_utc(candle.timestamp_utc) >= provenance_end
            for candle in candles
        )
    ):
        # Check clocks before validating any OHLC so a caller cannot
        # accidentally expose split/HOLDOUT prices to this resolver.
        raise ValueError(
            "S5 truth reaches or crosses the exclusive provenance boundary"
        )
    path = _validated_s5_path(candles)
    relevant = tuple(candle for candle in path if activation <= candle.timestamp_utc)
    ttl_coverage_sentinel = next(
        (
            candle
            for candle in path
            if candle.timestamp_utc >= activation
            and candle.timestamp_utc + timedelta(seconds=S5_SECONDS) >= expiry
        ),
        None,
    )
    side = str(signal["side"])
    order_type = str(signal["order_type"])
    entry = float(signal["entry_price"])
    target = float(signal["take_profit_price"])
    stop = float(signal["stop_loss_price"])
    hold_seconds = int(signal["arm"]["max_hold_seconds"])
    pip_factor = float(instrument_pip_factor(str(signal["pair"])))

    fill_candle: S5BidAskCandle | None = None
    fill_price: float | None = None
    fill_kind: str | None = None
    for candle in relevant:
        if candle.timestamp_utc + timedelta(seconds=S5_SECONDS) > expiry:
            break
        triggered, triggered_at_open = _entry_trigger(
            order_type,
            side,
            candle,
            entry,
        )
        if not triggered:
            continue
        fill_candle = candle
        fill_kind = "OPEN_TRIGGER" if triggered_at_open else "INTRABAR_TRIGGER"
        fill_price = _fill_price(
            order_type,
            side,
            candle,
            entry,
            triggered_at_open=triggered_at_open,
        )
        break

    if fill_candle is None or fill_price is None or fill_kind is None:
        relevant = _evidence_through_candle(relevant, ttl_coverage_sentinel)
        truth_path_sha = _truth_path_digest(
            receipt_sha=receipt_sha,
            pair=str(signal["pair"]),
            activation=activation,
            latest_maturity=latest_maturity,
            provenance_end=provenance_end,
            candles=relevant,
            ttl_coverage_sentinel=ttl_coverage_sentinel,
        )
        if provenance_end is not None and expiry > provenance_end:
            return _sealed_outcome(
                signal,
                truth_path_sha=truth_path_sha,
                truth_source_receipt_sha=receipt_sha,
                status="UNRESOLVED_ENTRY_TTL_CROSSES_PROVENANCE_BOUNDARY",
                filled=False,
                fill_at=None,
                fill_price=None,
                fill_kind=None,
                exit_at=None,
                exit_price=None,
                exit_reason="ENTRY_TTL_CROSSES_PROVENANCE_BOUNDARY",
                realized_pips=None,
                ambiguous=False,
                entry_gap_pips=0.0,
                stop_gap_worse=False,
                result_available=False,
                relevant=relevant,
                activation=activation,
                latest_maturity=latest_maturity,
            )
        if ttl_coverage_sentinel is None:
            return _sealed_outcome(
                signal,
                truth_path_sha=truth_path_sha,
                truth_source_receipt_sha=receipt_sha,
                status="UNRESOLVED_MISSING_ENTRY_TTL_COVERAGE",
                filled=False,
                fill_at=None,
                fill_price=None,
                fill_kind=None,
                exit_at=None,
                exit_price=None,
                exit_reason="ENTRY_TTL_COVERAGE_SENTINEL_MISSING",
                realized_pips=None,
                ambiguous=False,
                entry_gap_pips=0.0,
                stop_gap_worse=False,
                result_available=False,
                relevant=relevant,
                activation=activation,
                latest_maturity=latest_maturity,
            )
        return _sealed_outcome(
            signal,
            truth_path_sha=truth_path_sha,
            truth_source_receipt_sha=receipt_sha,
            status="MATURE_UNFILLED",
            filled=False,
            fill_at=None,
            fill_price=None,
            fill_kind=None,
            exit_at=expiry,
            exit_price=None,
            exit_reason="UNFILLED_EXPIRED",
            realized_pips=0.0,
            ambiguous=False,
            entry_gap_pips=0.0,
            stop_gap_worse=False,
            result_available=True,
            relevant=relevant,
            activation=activation,
            latest_maturity=latest_maturity,
            ttl_coverage_at=ttl_coverage_sentinel.timestamp_utc,
        )

    entry_gap_pips = _entry_gap_pips(
        order_type,
        side,
        entry,
        fill_price,
        pip_factor,
    )
    hold_boundary = fill_candle.timestamp_utc + timedelta(seconds=hold_seconds)
    first_real_at_time_close = next(
        (candle for candle in relevant if candle.timestamp_utc >= hold_boundary),
        None,
    )
    relevant = _evidence_through_candle(relevant, first_real_at_time_close)
    truth_path_sha = _truth_path_digest(
        receipt_sha=receipt_sha,
        pair=str(signal["pair"]),
        activation=activation,
        latest_maturity=latest_maturity,
        provenance_end=provenance_end,
        candles=relevant,
        ttl_coverage_sentinel=ttl_coverage_sentinel,
    )
    if not _geometry_valid(side, fill_price, target, stop):
        return _sealed_outcome(
            signal,
            truth_path_sha=truth_path_sha,
            truth_source_receipt_sha=receipt_sha,
            status="CANCEL_NO_FILL_INVALID_DEPENDENT_GEOMETRY",
            filled=False,
            fill_at=None,
            fill_price=None,
            fill_kind=None,
            exit_at=fill_candle.timestamp_utc,
            exit_price=None,
            exit_reason="CANCEL_NO_FILL_INVALID_DEPENDENT_GEOMETRY",
            realized_pips=0.0,
            ambiguous=False,
            entry_gap_pips=0.0,
            stop_gap_worse=False,
            result_available=True,
            relevant=relevant,
            activation=activation,
            latest_maturity=latest_maturity,
        )
    fill_index = relevant.index(fill_candle)
    for offset, candle in enumerate(relevant[fill_index:]):
        is_fill_candle = offset == 0
        if not is_fill_candle and candle.timestamp_utc >= hold_boundary:
            stop_gap = _stop_crossed_at_open(side, candle, stop)
            if stop_gap:
                exit_price = _exit_open(side, candle)
                return _filled_outcome(
                    signal,
                    truth_path_sha=truth_path_sha,
                    truth_source_receipt_sha=receipt_sha,
                    fill_candle=fill_candle,
                    fill_price=fill_price,
                    fill_kind=fill_kind,
                    entry_gap_pips=entry_gap_pips,
                    exit_candle=candle,
                    exit_price=exit_price,
                    exit_reason="STOP_LOSS_GAP_AT_TIME_CLOSE",
                    pip_factor=pip_factor,
                    ambiguous=False,
                    stop_gap=True,
                    relevant=relevant,
                    activation=activation,
                    latest_maturity=latest_maturity,
                )
            if _target_crossed_at_open(side, candle, target):
                return _filled_outcome(
                    signal,
                    truth_path_sha=truth_path_sha,
                    truth_source_receipt_sha=receipt_sha,
                    fill_candle=fill_candle,
                    fill_price=fill_price,
                    fill_kind=fill_kind,
                    entry_gap_pips=entry_gap_pips,
                    exit_candle=candle,
                    exit_price=target,
                    exit_reason="TAKE_PROFIT_AT_TIME_CLOSE_OPEN",
                    pip_factor=pip_factor,
                    ambiguous=False,
                    stop_gap=False,
                    relevant=relevant,
                    activation=activation,
                    latest_maturity=latest_maturity,
                )
            return _filled_outcome(
                signal,
                truth_path_sha=truth_path_sha,
                truth_source_receipt_sha=receipt_sha,
                fill_candle=fill_candle,
                fill_price=fill_price,
                fill_kind=fill_kind,
                entry_gap_pips=entry_gap_pips,
                exit_candle=candle,
                exit_price=_exit_open(side, candle),
                exit_reason="EXECUTABLE_TIME_CLOSE",
                pip_factor=pip_factor,
                ambiguous=False,
                stop_gap=False,
                relevant=relevant,
                activation=activation,
                latest_maturity=latest_maturity,
            )

        target_hit, stop_hit = _exit_hits(side, candle, target, stop)
        if is_fill_candle and (target_hit or stop_hit):
            # If entry was first touched intrabar, the candle open predates the
            # fill and may not be used as a stop gap.  This prevents a
            # pre-entry price from becoming an executable post-entry loss.
            gap_allowed = fill_kind == "OPEN_TRIGGER"
            return _stop_first_outcome(
                signal,
                truth_path_sha=truth_path_sha,
                truth_source_receipt_sha=receipt_sha,
                fill_candle=fill_candle,
                fill_price=fill_price,
                fill_kind=fill_kind,
                entry_gap_pips=entry_gap_pips,
                exit_candle=candle,
                stop=stop,
                pip_factor=pip_factor,
                ambiguous=True,
                reason="STOP_LOSS_AMBIGUOUS_FILL_S5",
                gap_allowed=gap_allowed,
                relevant=relevant,
                activation=activation,
                latest_maturity=latest_maturity,
            )
        if not is_fill_candle and target_hit and stop_hit:
            return _stop_first_outcome(
                signal,
                truth_path_sha=truth_path_sha,
                truth_source_receipt_sha=receipt_sha,
                fill_candle=fill_candle,
                fill_price=fill_price,
                fill_kind=fill_kind,
                entry_gap_pips=entry_gap_pips,
                exit_candle=candle,
                stop=stop,
                pip_factor=pip_factor,
                ambiguous=True,
                reason="STOP_LOSS_AMBIGUOUS_SAME_S5",
                gap_allowed=True,
                relevant=relevant,
                activation=activation,
                latest_maturity=latest_maturity,
            )
        if not is_fill_candle and stop_hit:
            return _stop_first_outcome(
                signal,
                truth_path_sha=truth_path_sha,
                truth_source_receipt_sha=receipt_sha,
                fill_candle=fill_candle,
                fill_price=fill_price,
                fill_kind=fill_kind,
                entry_gap_pips=entry_gap_pips,
                exit_candle=candle,
                stop=stop,
                pip_factor=pip_factor,
                ambiguous=False,
                reason="STOP_LOSS",
                gap_allowed=True,
                relevant=relevant,
                activation=activation,
                latest_maturity=latest_maturity,
            )
        if not is_fill_candle and target_hit:
            return _filled_outcome(
                signal,
                truth_path_sha=truth_path_sha,
                truth_source_receipt_sha=receipt_sha,
                fill_candle=fill_candle,
                fill_price=fill_price,
                fill_kind=fill_kind,
                entry_gap_pips=entry_gap_pips,
                exit_candle=candle,
                exit_price=target,
                exit_reason="TAKE_PROFIT",
                pip_factor=pip_factor,
                ambiguous=False,
                stop_gap=False,
                relevant=relevant,
                activation=activation,
                latest_maturity=latest_maturity,
            )

    crosses_provenance = provenance_end is not None
    return _sealed_outcome(
        signal,
        truth_path_sha=truth_path_sha,
        truth_source_receipt_sha=receipt_sha,
        status=(
            "UNRESOLVED_EXIT_CROSSES_PROVENANCE_BOUNDARY"
            if crosses_provenance
            else "UNRESOLVED_TIME_CLOSE_REAL_OPEN_MISSING"
        ),
        filled=True,
        fill_at=fill_candle.timestamp_utc,
        fill_price=fill_price,
        fill_kind=fill_kind,
        exit_at=None,
        exit_price=None,
        exit_reason=(
            "EXIT_REMAINS_OPEN_AT_PROVENANCE_BOUNDARY"
            if crosses_provenance
            else "TIME_CLOSE_REAL_OPEN_MISSING"
        ),
        realized_pips=None,
        ambiguous=False,
        entry_gap_pips=entry_gap_pips,
        stop_gap_worse=False,
        result_available=False,
        relevant=relevant,
        activation=activation,
        latest_maturity=latest_maturity,
    )


def deoverlap_same_pair_signal_specs_v1(
    signals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Retain one half-open window per exact pair/candidate specification.

    Different H families and orientations remain independent, even on the same
    pair.  This prevents minute-by-minute duplicate stacking without restoring
    a top-one-per-pair assumption.
    """

    validated = [_validated_signal_copy(item) for item in signals]
    _reject_duplicate_signal_ids(validated)
    ordered = sorted(
        validated,
        key=lambda item: (
            _parse_utc(item["activation_at_utc"]),
            str(item["pair"]),
            str(item["candidate_id"]),
            str(item["signal_id"]),
        ),
    )
    occupied_until: dict[tuple[str, str], datetime] = {}
    selected: list[str] = []
    rejected: list[dict[str, str]] = []
    for signal in ordered:
        key = (str(signal["pair"]), str(signal["candidate_id"]))
        start = _parse_utc(signal["activation_at_utc"])
        end = _parse_utc(signal["latest_maturity_at_utc"])
        if start < occupied_until.get(key, start):
            rejected.append(
                {
                    "signal_id": str(signal["signal_id"]),
                    "reason": "OVERLAPPING_SAME_PAIR_SIGNAL_SPEC",
                }
            )
            continue
        selected.append(str(signal["signal_id"]))
        occupied_until[key] = end
    body = {
        "policy": "PAIR_CANDIDATE_HALF_OPEN_MAX_MATURITY_V1",
        "source_signal_count": len(ordered),
        "selected_signal_ids": selected,
        "rejected": rejected,
        "different_hypotheses_may_coexist": True,
        "different_orientations_may_coexist": True,
        **_zero_authority(),
    }
    return {**body, "receipt_sha256": _canonical_sha(body)}


def assign_fixed_split_roles_v1(
    signals: Sequence[Mapping[str, Any]],
    *,
    train_end_utc: datetime,
    validation_end_utc: datetime,
    holdout_end_utc: datetime,
) -> dict[str, Any]:
    """Assign roles from frozen clocks without opening a holdout outcome."""

    train_end = _aware_utc(train_end_utc)
    validation_end = _aware_utc(validation_end_utc)
    holdout_end = _aware_utc(holdout_end_utc)
    if not train_end < validation_end < holdout_end:
        raise ValueError("split boundaries must be strictly chronological")
    validated = [_validated_signal_copy(item) for item in signals]
    _reject_duplicate_signal_ids(validated)
    assignments: list[dict[str, str]] = []
    for signal in sorted(
        validated,
        key=lambda item: (
            _parse_utc(item["activation_at_utc"]),
            str(item["signal_id"]),
        ),
    ):
        activation = _parse_utc(signal["activation_at_utc"])
        maturity = _parse_utc(signal["latest_maturity_at_utc"])
        if activation < train_end:
            role = "TRAIN" if maturity <= train_end else "PURGED_TRAIN_VALIDATION"
        elif activation < validation_end:
            role = (
                "VALIDATION"
                if maturity <= validation_end
                else "PURGED_VALIDATION_HOLDOUT"
            )
        elif activation < holdout_end:
            role = "RESERVED_HOLDOUT_UNOPENED"
        else:
            role = "OUTSIDE_FIXED_WINDOW"
        assignments.append({"signal_id": str(signal["signal_id"]), "role": role})
    body = {
        "contract": SPLIT_RECEIPT_CONTRACT_V1,
        "schema_version": 1,
        "split_policy": SPLIT_POLICY_V1,
        "train_end_utc": train_end.isoformat(),
        "validation_end_utc": validation_end.isoformat(),
        "holdout_end_utc": holdout_end.isoformat(),
        "selection_allowed_roles": ["TRAIN", "VALIDATION"],
        "holdout_role": "RESERVED_HOLDOUT_UNOPENED",
        "holdout_outcomes_opened": False,
        "assignments": assignments,
        **_zero_authority(),
    }
    return _seal(body)


def select_validation_one_se_v1(
    metrics: Sequence[Mapping[str, Any]],
    *,
    alpha: float = SELECTION_ALPHA,
) -> dict[str, Any]:
    """Select at most one validation winner per H family without holdout data.

    Each metric row names one fixed candidate and one role (TRAIN or
    VALIDATION). Required economic fields are cluster_count,
    mean_daily_post_cost_pips, standard_error_daily_post_cost_pips,
    mean_post_cost_pips, profit_factor, and one_sided_p_value. Missing planned
    candidates remain in the Holm denominator with p=1.
    """

    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("selection alpha must be between zero and one-half")
    catalog = build_fast_bot_technical_grid_catalog_v1()
    candidates = {str(row["candidate_id"]): row for row in catalog["candidates"]}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in metrics:
        if not isinstance(raw, Mapping):
            raise ValueError("selection metric must be a mapping")
        role = str(raw.get("data_role") or "").upper()
        if role not in {"TRAIN", "VALIDATION"}:
            if "HOLDOUT" in role:
                raise ValueError("HOLDOUT metrics are forbidden in selection")
            raise ValueError("selection metric role must be TRAIN or VALIDATION")
        candidate_id = str(raw.get("candidate_id") or "")
        if candidate_id not in candidates:
            raise ValueError("selection metric candidate is outside fixed catalog")
        key = (candidate_id, role)
        if key in by_key:
            raise ValueError("duplicate candidate/data-role selection metric")
        by_key[key] = _validated_selection_metric(raw, role=role)

    validation_p = {
        candidate_id: (
            by_key[(candidate_id, "VALIDATION")]["one_sided_p_value"]
            if (candidate_id, "VALIDATION") in by_key
            else 1.0
        )
        for candidate_id in candidates
    }
    adjusted = _holm_adjusted_p_values(validation_p)
    economic: dict[str, dict[str, Any]] = {}
    for candidate_id, candidate in candidates.items():
        train = by_key.get((candidate_id, "TRAIN"))
        validation = by_key.get((candidate_id, "VALIDATION"))
        train_pass = _economic_metric_passes(train, require_p=False, alpha=alpha)
        validation_pass = _economic_metric_passes(
            validation,
            require_p=True,
            alpha=alpha,
            adjusted_p=adjusted[candidate_id],
        )
        economic[candidate_id] = {
            "candidate_id": candidate_id,
            "hypothesis_id": candidate["hypothesis_id"],
            "train_gate_passed": train_pass,
            "validation_gate_passed": validation_pass,
            "holm_adjusted_p_value": adjusted[candidate_id],
            "eligible": bool(train_pass and validation_pass),
        }

    winners: list[dict[str, Any]] = []
    for hypothesis_id in (f"H{index:02d}" for index in range(1, 8)):
        eligible = [
            candidates[candidate_id]
            for candidate_id, row in economic.items()
            if row["eligible"] is True and row["hypothesis_id"] == hypothesis_id
        ]
        if not eligible:
            continue
        best = max(
            eligible,
            key=lambda row: by_key[(str(row["candidate_id"]), "VALIDATION")][
                "mean_daily_post_cost_pips"
            ],
        )
        best_metric = by_key[(str(best["candidate_id"]), "VALIDATION")]
        one_se_floor = (
            best_metric["mean_daily_post_cost_pips"]
            - best_metric["standard_error_daily_post_cost_pips"]
        )
        within_one_se = [
            row
            for row in eligible
            if by_key[(str(row["candidate_id"]), "VALIDATION")][
                "mean_daily_post_cost_pips"
            ]
            >= one_se_floor
        ]
        selected = min(
            within_one_se,
            key=lambda row: (
                int(row["complexity_rank"]),
                int(row["orientation_rank"]),
                int(row["catalog_order"]),
            ),
        )
        winners.append(
            {
                "hypothesis_id": hypothesis_id,
                "candidate_id": selected["candidate_id"],
                "best_candidate_id": best["candidate_id"],
                "best_mean_daily_post_cost_pips": best_metric[
                    "mean_daily_post_cost_pips"
                ],
                "best_standard_error_daily_post_cost_pips": best_metric[
                    "standard_error_daily_post_cost_pips"
                ],
                "one_se_floor": _round(one_se_floor),
                "selected_by_simplicity_within_one_se": True,
            }
        )

    body = {
        "contract": SELECTION_RECEIPT_CONTRACT_V1,
        "schema_version": 1,
        "selection_policy": SELECTION_POLICY_V1,
        "catalog_contract_sha256": catalog["contract_sha256"],
        "alpha": float(alpha),
        "minimum_clusters": MIN_SELECTION_CLUSTERS,
        "planned_candidate_count": PLANNED_CANDIDATE_COUNT,
        "multiple_testing_denominator": PLANNED_CANDIDATE_COUNT,
        "provided_metric_row_count": len(by_key),
        "holm_missing_candidate_policy": "MISSING_VALIDATION_P_EQUALS_ONE",
        "eligible_candidate_count": sum(
            row["eligible"] is True for row in economic.values()
        ),
        "selected_family_count": len(winners),
        "selected": winners,
        "locked_portfolio_candidate_ids": [],
        "holdout_metrics_consumed": False,
        "holdout_candidate_replacement_allowed": False,
        "candidate_drop_allowed": False,
        "metric_provenance_binding_verified": False,
        "selection_status": "STRUCTURAL_DRY_RUN_ONLY",
        "economic_conclusion_allowed": False,
        "backtest_complete": False,
        "provisional_validation_candidate_ids": [
            str(row["candidate_id"]) for row in winners
        ],
        "candidate_economics": [economic[candidate_id] for candidate_id in candidates],
        **_zero_authority(),
    }
    return _seal(body)


def _validated_selection_metric(raw: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    cluster_count = raw.get("cluster_count")
    if cluster_count.__class__ is not int or cluster_count < 0:
        raise ValueError("selection cluster_count is invalid")
    fields = {
        name: _finite_number(raw.get(name), name=name, nonnegative=nonnegative)
        for name, nonnegative in (
            ("mean_daily_post_cost_pips", False),
            ("standard_error_daily_post_cost_pips", True),
            ("mean_post_cost_pips", False),
            ("one_sided_p_value", True),
        )
    }
    p_value = fields["one_sided_p_value"]
    if p_value > 1.0:
        raise ValueError("selection p-value exceeds one")
    profit_factor = raw.get("profit_factor")
    if profit_factor is None:
        parsed_pf = 0.0
    else:
        try:
            parsed_pf = float(profit_factor)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("selection profit_factor is invalid") from error
        if math.isnan(parsed_pf) or parsed_pf < 0.0:
            raise ValueError("selection profit_factor is invalid")
    return {
        "candidate_id": str(raw["candidate_id"]),
        "data_role": role,
        "cluster_count": cluster_count,
        **fields,
        "profit_factor": parsed_pf,
    }


def _economic_metric_passes(
    metric: Mapping[str, Any] | None,
    *,
    require_p: bool,
    alpha: float,
    adjusted_p: float = 1.0,
) -> bool:
    if metric is None:
        return False
    return bool(
        int(metric["cluster_count"]) >= MIN_SELECTION_CLUSTERS
        and float(metric["mean_daily_post_cost_pips"]) > 0.0
        and float(metric["mean_post_cost_pips"]) > 0.0
        and float(metric["profit_factor"]) > 1.0
        and (not require_p or adjusted_p <= alpha)
    )


def _holm_adjusted_p_values(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (candidate_id, raw_p) in enumerate(ordered):
        current = min(1.0, (total - index) * float(raw_p))
        running = max(running, current)
        # Preserve full precision for the alpha comparison.  Rounding an
        # adjusted p-value before the gate can turn a value just above alpha
        # into a false pass.
        adjusted[candidate_id] = running
    return adjusted


def _stop_first_outcome(
    signal: Mapping[str, Any],
    *,
    truth_path_sha: str,
    truth_source_receipt_sha: str,
    fill_candle: S5BidAskCandle,
    fill_price: float,
    fill_kind: str,
    entry_gap_pips: float,
    exit_candle: S5BidAskCandle,
    stop: float,
    pip_factor: float,
    ambiguous: bool,
    reason: str,
    gap_allowed: bool,
    relevant: Sequence[S5BidAskCandle],
    activation: datetime,
    latest_maturity: datetime,
) -> dict[str, Any]:
    stop_gap = gap_allowed and _stop_crossed_at_open(
        str(signal["side"]), exit_candle, stop
    )
    exit_price = _exit_open(str(signal["side"]), exit_candle) if stop_gap else stop
    final_reason = f"{reason}_GAP" if stop_gap else reason
    return _filled_outcome(
        signal,
        truth_path_sha=truth_path_sha,
        truth_source_receipt_sha=truth_source_receipt_sha,
        fill_candle=fill_candle,
        fill_price=fill_price,
        fill_kind=fill_kind,
        entry_gap_pips=entry_gap_pips,
        exit_candle=exit_candle,
        exit_price=exit_price,
        exit_reason=final_reason,
        pip_factor=pip_factor,
        ambiguous=ambiguous,
        stop_gap=stop_gap,
        relevant=relevant,
        activation=activation,
        latest_maturity=latest_maturity,
    )


def _filled_outcome(
    signal: Mapping[str, Any],
    *,
    truth_path_sha: str,
    truth_source_receipt_sha: str,
    fill_candle: S5BidAskCandle,
    fill_price: float,
    fill_kind: str,
    entry_gap_pips: float,
    exit_candle: S5BidAskCandle,
    exit_price: float,
    exit_reason: str,
    pip_factor: float,
    ambiguous: bool,
    stop_gap: bool,
    relevant: Sequence[S5BidAskCandle],
    activation: datetime,
    latest_maturity: datetime,
) -> dict[str, Any]:
    side = str(signal["side"])
    realized = (
        (exit_price - fill_price) * pip_factor
        if side == "LONG"
        else (fill_price - exit_price) * pip_factor
    )
    status = (
        "MATURE_FILLED_TAKE_PROFIT"
        if exit_reason.startswith("TAKE_PROFIT")
        else "MATURE_FILLED_TIME_CLOSE"
        if exit_reason == "EXECUTABLE_TIME_CLOSE"
        else "MATURE_FILLED_STOP_LOSS"
    )
    return _sealed_outcome(
        signal,
        truth_path_sha=truth_path_sha,
        truth_source_receipt_sha=truth_source_receipt_sha,
        status=status,
        filled=True,
        fill_at=fill_candle.timestamp_utc,
        fill_price=fill_price,
        fill_kind=fill_kind,
        exit_at=exit_candle.timestamp_utc,
        exit_price=exit_price,
        exit_reason=exit_reason,
        realized_pips=realized,
        ambiguous=ambiguous,
        entry_gap_pips=entry_gap_pips,
        stop_gap_worse=stop_gap,
        result_available=True,
        relevant=relevant,
        activation=activation,
        latest_maturity=latest_maturity,
    )


def _sealed_outcome(
    signal: Mapping[str, Any],
    *,
    truth_path_sha: str,
    truth_source_receipt_sha: str,
    status: str,
    filled: bool,
    fill_at: datetime | None,
    fill_price: float | None,
    fill_kind: str | None,
    exit_at: datetime | None,
    exit_price: float | None,
    exit_reason: str,
    realized_pips: float | None,
    ambiguous: bool,
    entry_gap_pips: float,
    stop_gap_worse: bool,
    result_available: bool,
    relevant: Sequence[S5BidAskCandle],
    activation: datetime,
    latest_maturity: datetime,
    ttl_coverage_at: datetime | None = None,
) -> dict[str, Any]:
    grid_count = _grid_slot_count(activation, latest_maturity)
    grid_candle_count = sum(
        activation <= candle.timestamp_utc < latest_maturity for candle in relevant
    )
    body = {
        "contract": TIME_CLOSE_OUTCOME_CONTRACT_V1,
        "schema_version": 1,
        "scoring_policy": TIME_CLOSE_POLICY_V1,
        "signal_id": signal["signal_id"],
        "signal_sha256": signal["signal_sha256"],
        "candidate_id": signal["candidate_id"],
        "pair": signal["pair"],
        "side": signal["side"],
        "activation_at_utc": signal["activation_at_utc"],
        "latest_maturity_at_utc": signal["latest_maturity_at_utc"],
        "status": status,
        "filled": filled,
        "fill_at_utc": fill_at.isoformat() if fill_at is not None else None,
        "fill_price": _round(fill_price) if fill_price is not None else None,
        "fill_kind": fill_kind,
        "entry_gap_pips": _round(entry_gap_pips),
        "exit_at_utc": exit_at.isoformat() if exit_at is not None else None,
        "exit_price": _round(exit_price) if exit_price is not None else None,
        "exit_reason": exit_reason,
        "post_cost_realized_pips": (
            _round(realized_pips) if realized_pips is not None else None
        ),
        "ambiguous_same_s5": ambiguous,
        "stop_gap_worse": stop_gap_worse,
        "scorecard_result_available": result_available,
        "truth_source": "CALLER_SEALED_EXACT_S5_BID_ASK",
        "truth_source_receipt_sha256": truth_source_receipt_sha,
        "truth_path_sha256": truth_path_sha,
        "entry_ttl_coverage_sentinel_at_utc": (
            ttl_coverage_at.isoformat() if ttl_coverage_at is not None else None
        ),
        "entry_ttl_coverage_complete": bool(filled or ttl_coverage_at is not None),
        "truth_grid_slot_count": grid_count,
        "truth_candle_count": len(relevant),
        "truth_grid_candle_count": grid_candle_count,
        "truth_post_latest_maturity_candle_count": len(relevant) - grid_candle_count,
        "truth_no_tick_slot_count": grid_count - grid_candle_count,
        "missing_no_tick_intervals_synthesized": False,
        "strategy_evaluator_binding_verified": False,
        "economic_evidence_eligible": False,
        **_zero_authority(),
    }
    return _seal(body)


def _coerce_timeframe_feature(
    raw: CausalTimeframeFeature | Mapping[str, Any],
) -> CausalTimeframeFeature:
    if isinstance(raw, CausalTimeframeFeature):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("timeframe feature must be a mapping")
    try:
        close = raw["complete_candle_close_utc"]
        parsed_close = close if isinstance(close, datetime) else _parse_utc(close)
        return CausalTimeframeFeature(
            timeframe=str(raw["timeframe"]),
            complete_candle_close_utc=parsed_close,
            market_state=raw.get("market_state", {}),
            indicators=raw.get("indicators", {}),
            indicator_series=raw.get("indicator_series", {}),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("timeframe feature shape is invalid") from error


def _validated_market_state(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("market_state must be a mapping")
    result: dict[str, Any] = {}
    for key, raw in value.items():
        if key in MARKET_STATE_ENUM_FEATURES:
            if not isinstance(raw, str) or raw not in MARKET_STATE_ENUM_FEATURES[key]:
                raise ValueError(f"invalid market-state feature: {key}")
            result[key] = raw
        elif key == "confidence":
            number = _finite_number(raw, name=key, nonnegative=True)
            if number > 1.0:
                raise ValueError("market-state confidence exceeds one")
            result[key] = number
        elif key == "evidence_complete":
            if not isinstance(raw, bool):
                raise ValueError("market-state evidence_complete must be boolean")
            result[key] = raw
        else:
            raise ValueError(f"unknown market-state feature: {key}")
    return result


def _validated_indicators(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("indicators must be a mapping")
    result: dict[str, Any] = {}
    for key, raw in value.items():
        if raw is None:
            continue
        if key in NUMERIC_INDICATOR_FEATURES:
            result[key] = _finite_number(raw, name=key, nonnegative=False)
        elif key in ENUM_INDICATOR_FEATURES:
            if not isinstance(raw, str) or not raw or len(raw) > 64:
                raise ValueError(f"invalid enum indicator: {key}")
            result[key] = raw
        else:
            raise ValueError(f"unknown indicator feature: {key}")
    return result


def _validated_indicator_series(
    value: Mapping[str, Sequence[float]],
) -> dict[str, list[float | int]]:
    if not isinstance(value, Mapping):
        raise ValueError("indicator_series must be a mapping")
    result: dict[str, list[float | int]] = {}
    for key, raw in value.items():
        if key not in INDICATOR_SERIES_FEATURES:
            raise ValueError(f"unknown indicator series: {key}")
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError(f"invalid indicator series: {key}")
        if len(raw) > MAX_INDICATOR_SERIES_VALUES:
            raise ValueError(f"indicator series exceeds bounded history: {key}")
        result[key] = [
            _finite_number(item, name=key, nonnegative=False) for item in raw
        ]
    return result


def _validate_frozen_signal(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("frozen signal must be a mapping")
    arm = value.get("arm")
    if not isinstance(arm, Mapping):
        raise ValueError("frozen signal arm is invalid")
    body = {key: item for key, item in value.items() if key != "signal_sha256"}
    try:
        sealed = value.get("signal_sha256") == _canonical_sha(body)
    except (TypeError, ValueError, OverflowError):
        sealed = False
    if not sealed or value.get("contract") != GRID_SIGNAL_CONTRACT_V1:
        raise ValueError("frozen signal seal or authority is invalid")

    pair = _validated_pair(value.get("pair"))
    hypothesis = str(value.get("hypothesis_id") or "")
    orientation = str(value.get("orientation") or "")
    source_side = str(value.get("source_predicted_side") or "")
    side = str(value.get("side") or "")
    candidate_id = f"{hypothesis}:{orientation}:{arm.get('arm_id')}"
    catalog = build_fast_bot_technical_grid_catalog_v1()
    candidate = next(
        (row for row in catalog["candidates"] if row["candidate_id"] == candidate_id),
        None,
    )
    if not bool(
        candidate is not None
        and value.get("candidate_id") == candidate_id
        and value.get("candidate_sha256") == candidate["candidate_sha256"]
        and value.get("catalog_contract_sha256") == catalog["contract_sha256"]
        and dict(arm) == candidate["arm"]
        and value.get("order_type") in ORDER_TYPES
        and source_side in {"LONG", "SHORT"}
        and side
        == (source_side if orientation == "DIRECT" else _opposite_side(source_side))
        and value.get("natural_entry_component") == ("ASK" if side == "LONG" else "BID")
        and value.get("exit_component") == ("BID" if side == "LONG" else "ASK")
        and value.get("causal_strategy_evaluator_binding") == "CALLER_ASSERTED_SHA_ONLY"
        and value.get("strategy_evaluator_binding_verified") is False
        and value.get("economic_backtest_eligible") is False
        and all(
            value.get(key) == expected for key, expected in _zero_authority().items()
        )
    ):
        raise ValueError("frozen signal catalog identity or authority is invalid")

    _validated_sha(value.get("causal_source_sha256"), "causal_source_sha256")
    activation = _parse_utc(value["activation_at_utc"])
    expiry = _parse_utc(value["entry_expires_at_utc"])
    maturity = _parse_utc(value["latest_maturity_at_utc"])
    try:
        ttl_seconds = int(arm["entry_ttl_seconds"])
        hold_seconds = int(arm["max_hold_seconds"])
        entry = float(value["entry_price"])
        target = float(value["take_profit_price"])
        stop = float(value["stop_loss_price"])
        base_target = float(value["base_take_profit_price"])
        base_stop = float(value["base_stop_loss_price"])
        tick = _price_tick(pair)
        entry_decimal = Decimal(str(entry))
        if _nearest_tick(entry_decimal, tick) != entry_decimal:
            raise ValueError("frozen signal entry is not pipette aligned")
        tp_distance = abs(Decimal(str(base_target)) - entry_decimal)
        sl_distance = abs(entry_decimal - Decimal(str(base_stop)))
        arm_tp = Decimal(str(arm["take_profit_multiplier"]))
        arm_sl = Decimal(str(arm["stop_loss_multiplier"]))
        if side == "LONG":
            expected_target = _nearest_tick(entry_decimal + tp_distance * arm_tp, tick)
            expected_stop = _nearest_tick(entry_decimal - sl_distance * arm_sl, tick)
        else:
            expected_target = _nearest_tick(entry_decimal - tp_distance * arm_tp, tick)
            expected_stop = _nearest_tick(entry_decimal + sl_distance * arm_sl, tick)
        clocks_and_geometry_valid = bool(
            activation.second == 0
            and activation.microsecond == 0
            and expiry == activation + timedelta(seconds=ttl_seconds)
            and maturity == expiry + timedelta(seconds=hold_seconds)
            and _geometry_valid(side, entry, base_target, base_stop)
            and _geometry_valid(side, entry, target, stop)
            and target == float(expected_target)
            and stop == float(expected_stop)
        )
    except (KeyError, TypeError, ValueError, OverflowError, ArithmeticError):
        clocks_and_geometry_valid = False
    if not clocks_and_geometry_valid:
        raise ValueError("frozen signal clocks or geometry are invalid")

    if value.get("signal_id") != _canonical_sha(_signal_identity_payload(value))[:24]:
        raise ValueError("frozen signal identity is invalid")


def _validated_signal_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    _validate_frozen_signal(value)
    return dict(value)


def _reject_duplicate_signal_ids(signals: Sequence[Mapping[str, Any]]) -> None:
    signal_ids = [str(signal["signal_id"]) for signal in signals]
    if len(signal_ids) != len(set(signal_ids)):
        raise ValueError("duplicate signal_id is forbidden")


def _signal_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {key: value[key] for key in _SIGNAL_IDENTITY_KEYS}
    except (KeyError, TypeError) as error:
        raise ValueError("frozen signal execution identity is incomplete") from error


def _validated_s5_path(
    candles: Sequence[S5BidAskCandle],
) -> tuple[S5BidAskCandle, ...]:
    if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence):
        raise ValueError("candles must be a sequence")
    normalized = tuple(candles)
    previous: datetime | None = None
    for candle in normalized:
        if candle.__class__ is not S5BidAskCandle:
            raise ValueError("S5 candle class is invalid")
        timestamp = _aware_utc(candle.timestamp_utc)
        if timestamp.microsecond != 0 or int(timestamp.timestamp()) % S5_SECONDS != 0:
            raise ValueError("S5 candle timestamp is not grid aligned")
        if previous is not None and timestamp <= previous:
            raise ValueError("S5 candles must be chronological and unique")
        previous = timestamp
        bid = (candle.bid_o, candle.bid_h, candle.bid_l, candle.bid_c)
        ask = (candle.ask_o, candle.ask_h, candle.ask_l, candle.ask_c)
        if any(not math.isfinite(item) or item <= 0.0 for item in (*bid, *ask)):
            raise ValueError("S5 candle price is invalid")
        if not bool(
            candle.bid_l <= min(candle.bid_o, candle.bid_c)
            and max(candle.bid_o, candle.bid_c) <= candle.bid_h
            and candle.ask_l <= min(candle.ask_o, candle.ask_c)
            and max(candle.ask_o, candle.ask_c) <= candle.ask_h
            and all(bid_item < ask_item for bid_item, ask_item in zip(bid, ask))
        ):
            raise ValueError("S5 candle OHLC or bid/ask envelope is invalid")
    return normalized


def _entry_trigger(
    order_type: str,
    side: str,
    candle: S5BidAskCandle,
    entry: float,
) -> tuple[bool, bool]:
    if order_type == "STOP":
        if side == "LONG":
            return candle.ask_h >= entry, candle.ask_o >= entry
        return candle.bid_l <= entry, candle.bid_o <= entry
    if side == "LONG":
        return candle.ask_l <= entry, candle.ask_o <= entry
    return candle.bid_h >= entry, candle.bid_o >= entry


def _fill_price(
    order_type: str,
    side: str,
    candle: S5BidAskCandle,
    entry: float,
    *,
    triggered_at_open: bool,
) -> float:
    if not triggered_at_open:
        return entry
    executable_open = candle.ask_o if side == "LONG" else candle.bid_o
    if order_type == "STOP":
        return (
            max(entry, executable_open)
            if side == "LONG"
            else min(entry, executable_open)
        )
    return (
        min(entry, executable_open) if side == "LONG" else max(entry, executable_open)
    )


def _entry_gap_pips(
    order_type: str,
    side: str,
    entry: float,
    fill_price: float,
    pip_factor: float,
) -> float:
    if order_type != "STOP":
        return 0.0
    return max(
        0.0,
        (fill_price - entry) * pip_factor
        if side == "LONG"
        else (entry - fill_price) * pip_factor,
    )


def _exit_hits(
    side: str,
    candle: S5BidAskCandle,
    target: float,
    stop: float,
) -> tuple[bool, bool]:
    if side == "LONG":
        return candle.bid_h >= target, candle.bid_l <= stop
    return candle.ask_l <= target, candle.ask_h >= stop


def _stop_crossed_at_open(side: str, candle: S5BidAskCandle, stop: float) -> bool:
    return candle.bid_o < stop if side == "LONG" else candle.ask_o > stop


def _target_crossed_at_open(side: str, candle: S5BidAskCandle, target: float) -> bool:
    return candle.bid_o >= target if side == "LONG" else candle.ask_o <= target


def _exit_open(side: str, candle: S5BidAskCandle) -> float:
    return candle.bid_o if side == "LONG" else candle.ask_o


def _candle_payload(candle: S5BidAskCandle) -> dict[str, Any]:
    return {
        "timestamp_utc": _aware_utc(candle.timestamp_utc).isoformat(),
        "bid": [candle.bid_o, candle.bid_h, candle.bid_l, candle.bid_c],
        "ask": [candle.ask_o, candle.ask_h, candle.ask_l, candle.ask_c],
    }


def _evidence_through_candle(
    candles: Sequence[S5BidAskCandle],
    terminal: S5BidAskCandle | None,
) -> tuple[S5BidAskCandle, ...]:
    normalized = tuple(candles)
    if terminal is None:
        return normalized
    return normalized[: normalized.index(terminal) + 1]


def _truth_path_digest(
    *,
    receipt_sha: str,
    pair: str,
    activation: datetime,
    latest_maturity: datetime,
    provenance_end: datetime | None,
    candles: Sequence[S5BidAskCandle],
    ttl_coverage_sentinel: S5BidAskCandle | None,
) -> str:
    return _canonical_sha(
        {
            "truth_source_receipt_sha256": receipt_sha,
            "pair": pair,
            "activation_at_utc": activation.isoformat(),
            "latest_maturity_at_utc": latest_maturity.isoformat(),
            "truth_provenance_end_utc": (
                provenance_end.isoformat() if provenance_end is not None else None
            ),
            "candles": [_candle_payload(candle) for candle in candles],
            "ttl_coverage_sentinel": (
                _candle_payload(ttl_coverage_sentinel)
                if ttl_coverage_sentinel is not None
                else None
            ),
        }
    )


def _arm_by_id(arm_id: str) -> TechnicalGridArm:
    matches = [arm for arm in _ARMS_V1 if arm.arm_id == arm_id]
    if len(matches) != 1:
        raise ValueError("arm_id is outside the fixed OFAT grid")
    return matches[0]


def _geometry_valid(side: str, entry: float, target: float, stop: float) -> bool:
    if not all(math.isfinite(item) and item > 0.0 for item in (entry, target, stop)):
        return False
    return stop < entry < target if side == "LONG" else target < entry < stop


def _opposite_side(side: str) -> str:
    return "SHORT" if side == "LONG" else "LONG"


def _validated_pair(value: Any) -> str:
    pair = str(value or "")
    if pair not in DEFAULT_TRADER_PAIRS or not _PAIR_RE.fullmatch(pair):
        raise ValueError("pair is outside the canonical G8 universe")
    return pair


def _validated_sha(value: Any, name: str) -> str:
    text = str(value or "")
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _positive_finite(value: Any, name: str) -> float:
    parsed = _finite_number(value, name=name, nonnegative=False)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return float(parsed)


def _finite_number(value: Any, *, name: str, nonnegative: bool) -> float | int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(parsed) or (nonnegative and parsed < 0.0):
        raise ValueError(f"{name} must be finite")
    if isinstance(value, int):
        return value
    return parsed


def _price_tick(pair: str) -> Decimal:
    return Decimal(1) / Decimal(instrument_pip_factor(pair) * 10)


def _nearest_tick(value: Decimal, tick: Decimal) -> Decimal:
    return (value / tick).quantize(Decimal(1), rounding=ROUND_HALF_UP) * tick


def _grid_slot_count(start: datetime, end: datetime) -> int:
    return max(0, int((end - start).total_seconds() // S5_SECONDS))


def _parse_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _aware_utc(value)
    if not isinstance(value, str) or not value:
        raise ValueError("UTC timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("UTC timestamp is invalid") from error
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _round(value: float | int) -> float:
    return round(float(value), 6)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    return {**body, "contract_sha256": _canonical_sha(body)}


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "CausalTimeframeFeature",
    "GRID_CATALOG_CONTRACT_V1",
    "GRID_POLICY_V1",
    "PLANNED_CANDIDATE_COUNT",
    "SELECTION_RECEIPT_CONTRACT_V1",
    "SPLIT_RECEIPT_CONTRACT_V1",
    "TIME_CLOSE_OUTCOME_CONTRACT_V1",
    "TechnicalGridArm",
    "assign_fixed_split_roles_v1",
    "build_causal_technical_feature_snapshot_v1",
    "build_fast_bot_technical_grid_catalog_v1",
    "deoverlap_same_pair_signal_specs_v1",
    "freeze_fast_bot_technical_grid_signal_v1",
    "resolve_executable_bidask_time_close_v1",
    "select_validation_one_se_v1",
    "technical_grid_arms_v1",
]

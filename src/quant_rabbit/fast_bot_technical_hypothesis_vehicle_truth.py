"""Pure exact-S5 truth for pre-outcome technical-hypothesis vehicles.

The vehicle builder freezes H01..H08 before any outcome is visible.  This
module rebuilds that artifact from the same causal inputs and then evaluates
only its diagnostic vehicles on caller-supplied, complete OANDA S5 bid/ask
candles.  Missing S5 slots remain missing; no price path is synthesized.

This is historical diagnostic evidence only.  It cannot promote a vehicle,
grant live permission, construct an order intent, or mutate broker state.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from quant_rabbit.fast_bot_learning_truth import (
    ARM_OUTCOME_CONTRACT,
    CANDIDATE_OUTCOME_CONTRACT,
    OUTCOME_CONTRACT as LEARNING_OUTCOME_CONTRACT,
    _outcome_valid_for_seat,
)
from quant_rabbit.fast_bot_technical_hypothesis_vehicles import (
    SHADOW_CONTRACT_V2,
    VEHICLE_CONTRACT_V2,
    build_fast_bot_technical_hypothesis_vehicles_v2,
)
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle


TRUTH_CONTRACT_V1 = "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_VEHICLE_TRUTH_V1"
VEHICLE_OUTCOME_CONTRACT_V1 = "QR_FAST_BOT_TECHNICAL_HYPOTHESIS_VEHICLE_OUTCOME_V1"
SCORING_POLICY_V1 = "EXACT_S5_BID_ASK_STOP_SL_FIRST_DIAGNOSTIC_V1"
TRUTH_SOURCE_V1 = "CALLER_SEALED_EXACT_S5_BID_ASK"
ROUND_DIGITS = 6
S5_SECONDS = 5

_SHA256_LENGTH = 64
_SIDES = frozenset({"LONG", "SHORT"})
_UNRESOLVED_PROXY_REASONS = frozenset(
    {
        "SEALED_MATCHED_BASE_ARM_OUTCOME_MISSING",
        "SEALED_MATCHED_BASE_ARM_OUTCOME_NOT_UNIQUE",
        "SEALED_BASE_ARM_SCORING_START_MISMATCH",
        "SEALED_BASE_ARM_OUTCOME_AFTER_RESOLUTION_CLOCK",
    }
)
_AUTHORITY = {
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
    "order_intents": [],
}


def resolve_fast_bot_technical_hypothesis_vehicle_truth_v1(
    vehicle_shadow: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    technical_feature_snapshot: Mapping[str, Any],
    technical_hypothesis_shadow: Mapping[str, Any],
    episode_anchor: Mapping[str, Any],
    episode_route: Mapping[str, Any],
    learning_seat: Mapping[str, Any],
    confirmed_at_utc: str,
    resolved_at_utc: datetime,
    truth_source_receipt_sha256: str,
    base_arm_outcomes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Rebuild one frozen vehicle shadow and score its diagnostic rows.

    ``candles`` is a sparse exact path.  It may omit no-tick S5 slots, but each
    supplied candle must be a valid, unique, chronological S5 bid/ask candle
    inside the frozen activation/maturity interval.  H03/H05 remain unresolved
    unless an explicitly supplied, sealed learning-seat outcome contains the
    exact BASE arm named by their proxy binding.
    """

    source_receipt_sha = _sha256_text(
        truth_source_receipt_sha256,
        name="truth_source_receipt_sha256",
    )
    rebuilt = build_fast_bot_technical_hypothesis_vehicles_v2(
        technical_feature_snapshot=technical_feature_snapshot,
        technical_hypothesis_shadow=technical_hypothesis_shadow,
        episode_anchor=episode_anchor,
        episode_route=episode_route,
        learning_seat=learning_seat,
        confirmed_at_utc=confirmed_at_utc,
    )
    if rebuilt.get("status") not in {"EMITTED", "EMITTED_DIAGNOSTIC_ONLY"}:
        raise ValueError("rebuilt vehicle shadow is not emitted")
    if dict(vehicle_shadow) != rebuilt:
        raise ValueError("vehicle shadow differs from its causal rebuild")
    if not _sealed_valid(rebuilt, SHADOW_CONTRACT_V2):
        raise ValueError("vehicle shadow seal is invalid")

    activation = _parse_utc(rebuilt.get("activation_at_utc"), name="activation")
    resolved = _aware_utc(resolved_at_utc)
    diagnostic_rows = [
        row
        for row in rebuilt.get("vehicles", [])
        if isinstance(row, Mapping) and row.get("diagnostic_vehicle_available") is True
    ]
    if not diagnostic_rows:
        raise ValueError("vehicle shadow has no diagnostic vehicles")
    for row in diagnostic_rows:
        if not _vehicle_row_valid(row):
            raise ValueError("diagnostic vehicle row is invalid")

    proxy_outcomes = _validated_base_outcomes(
        base_arm_outcomes,
        learning_seat=learning_seat,
    )
    exact_maturities = [
        _parse_utc(row["execution"]["latest_maturity_at_utc"], name="maturity")
        for row in diagnostic_rows
        if isinstance(row.get("execution"), Mapping)
    ]
    latest_exact_maturity = max(exact_maturities, default=activation)
    if resolved < latest_exact_maturity:
        raise ValueError("exact vehicle truth is not mature")
    normalized = _validated_truth_path(
        candles,
        activation=activation,
        maturity=latest_exact_maturity,
    )
    path_payload = [_candle_payload(candle) for candle in normalized]
    truth_path_sha = _canonical_sha(
        {
            "truth_source_receipt_sha256": source_receipt_sha,
            "pair": rebuilt.get("pair"),
            "activation_at_utc": activation.isoformat(),
            "maturity_at_utc": latest_exact_maturity.isoformat(),
            "candles": path_payload,
        }
    )
    pip_factor = float(instrument_pip_factor(str(rebuilt["pair"])))
    outcomes: list[dict[str, Any]] = []
    for row in diagnostic_rows:
        hypothesis_id = str(row["hypothesis_id"])
        if hypothesis_id == "H08":
            outcome = _zero_control_outcome(
                row,
                truth_path_sha256=truth_path_sha,
            )
        elif isinstance(row.get("execution"), Mapping):
            outcome = _resolve_exact_stop(
                row,
                candles=normalized,
                pip_factor=pip_factor,
                truth_path_sha256=truth_path_sha,
            )
        elif isinstance(row.get("proxy_binding"), Mapping):
            outcome = _resolve_base_proxy(
                row,
                pair=str(rebuilt["pair"]),
                base_outcomes=proxy_outcomes,
                truth_path_sha256=truth_path_sha,
                resolved_at_utc=resolved,
            )
        else:
            raise ValueError("diagnostic vehicle has no scoring definition")
        if not technical_hypothesis_vehicle_outcome_v1_valid(
            outcome,
            vehicle_row=row,
        ):
            raise AssertionError("constructed vehicle outcome is invalid")
        outcomes.append(outcome)

    grid_count = _grid_slot_count(activation, latest_exact_maturity)
    body = {
        "contract": TRUTH_CONTRACT_V1,
        "schema_version": 1,
        "scoring_policy": SCORING_POLICY_V1,
        "vehicle_shadow_contract": SHADOW_CONTRACT_V2,
        "vehicle_shadow_contract_sha256": str(rebuilt["contract_sha256"]),
        "pair": str(rebuilt["pair"]),
        "activation_at_utc": activation.isoformat(),
        "maturity_at_utc": latest_exact_maturity.isoformat(),
        "resolved_at_utc": resolved.isoformat(),
        "truth_source": TRUTH_SOURCE_V1,
        "truth_source_receipt_sha256": source_receipt_sha,
        "truth_interval_from_utc": _ceil_s5(activation).isoformat(),
        "truth_interval_to_utc": _floor_s5(latest_exact_maturity).isoformat(),
        "truth_grid_slot_count": grid_count,
        "truth_candle_count": len(normalized),
        "truth_no_tick_slot_count": grid_count - len(normalized),
        "truth_path_sha256": truth_path_sha,
        "missing_no_tick_intervals_synthesized": False,
        "diagnostic_vehicle_count": len(diagnostic_rows),
        "vehicle_outcome_count": len(outcomes),
        "causal_input_proof_eligible": rebuilt.get("causal_input_proof_eligible")
        is True,
        "paired_direction_proof_eligible": rebuilt.get(
            "paired_direction_proof_eligible"
        )
        is True,
        "technical_hypothesis_proof_eligible": rebuilt.get(
            "technical_hypothesis_proof_eligible"
        )
        is True,
        "scorecard_eligible": rebuilt.get("scorecard_eligible") is True,
        "scorecard_ineligibility_reasons": list(
            rebuilt.get("scorecard_ineligibility_reasons", [])
        ),
        "scorecard_eligible_outcome_count": sum(
            row["scorecard_eligible"] is True for row in outcomes
        ),
        "scorecard_result_available_outcome_count": sum(
            row["scorecard_result_available"] is True for row in outcomes
        ),
        "unresolved_proxy_outcome_count": sum(
            row["status"] == "UNRESOLVED_BASE_ARM_OUTCOME_REQUIRED" for row in outcomes
        ),
        "vehicle_outcomes": outcomes,
        **_AUTHORITY,
    }
    return _seal(body)


def technical_hypothesis_vehicle_truth_v1_valid(
    value: Any,
    vehicle_shadow: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    technical_feature_snapshot: Mapping[str, Any],
    technical_hypothesis_shadow: Mapping[str, Any],
    episode_anchor: Mapping[str, Any],
    episode_route: Mapping[str, Any],
    learning_seat: Mapping[str, Any],
    confirmed_at_utc: str,
    resolved_at_utc: datetime,
    truth_source_receipt_sha256: str,
    base_arm_outcomes: Sequence[Mapping[str, Any]] = (),
) -> bool:
    """Re-resolve from sealed sources and require byte-for-byte semantics."""

    if not isinstance(value, Mapping) or not _sealed_valid(value, TRUTH_CONTRACT_V1):
        return False
    try:
        expected = resolve_fast_bot_technical_hypothesis_vehicle_truth_v1(
            vehicle_shadow,
            candles,
            technical_feature_snapshot=technical_feature_snapshot,
            technical_hypothesis_shadow=technical_hypothesis_shadow,
            episode_anchor=episode_anchor,
            episode_route=episode_route,
            learning_seat=learning_seat,
            confirmed_at_utc=confirmed_at_utc,
            resolved_at_utc=resolved_at_utc,
            truth_source_receipt_sha256=truth_source_receipt_sha256,
            base_arm_outcomes=base_arm_outcomes,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return dict(value) == expected


def technical_hypothesis_vehicle_outcome_v1_valid(
    value: Any,
    *,
    vehicle_row: Mapping[str, Any],
) -> bool:
    """Validate the sealed result and its full vehicle-bound semantics."""

    if not isinstance(value, Mapping) or not _sealed_named_valid(
        value,
        VEHICLE_OUTCOME_CONTRACT_V1,
        "vehicle_outcome_sha256",
    ):
        return False
    if not _vehicle_row_valid(vehicle_row):
        return False
    if not all(value.get(key) == expected for key, expected in _AUTHORITY.items()):
        return False
    proof_eligibility_keys = (
        "causal_input_proof_eligible",
        "paired_direction_proof_eligible",
        "technical_hypothesis_proof_eligible",
    )
    status = str(value.get("status") or "")
    resolution_reasons = value.get("resolution_reasons")
    if not isinstance(resolution_reasons, list) or not all(
        isinstance(reason, str) and reason for reason in resolution_reasons
    ):
        return False
    result_available = status != "UNRESOLVED_BASE_ARM_OUTCOME_REQUIRED"
    ex_ante_eligible = vehicle_row.get("scorecard_eligible") is True
    ex_ante_reasons = list(vehicle_row.get("scorecard_ineligibility_reasons", []))
    expected_result_reasons = (
        ex_ante_reasons
        if result_available
        else sorted(
            {
                *ex_ante_reasons,
                "OUTCOME_RESULT_UNAVAILABLE",
                *resolution_reasons,
            }
        )
    )
    if not bool(
        value.get("schema_version") == 1
        and value.get("scoring_policy") == SCORING_POLICY_V1
        and value.get("hypothesis_id") == vehicle_row.get("hypothesis_id")
        and value.get("vehicle_sha256") == vehicle_row.get("vehicle_sha256")
        and value.get("predicted_side") == vehicle_row.get("predicted_side")
        and all(
            value.get(key) == (vehicle_row.get(key) is True)
            for key in proof_eligibility_keys
        )
        and value.get("ex_ante_scorecard_eligible") is ex_ante_eligible
        and value.get("ex_ante_scorecard_ineligibility_reasons") == ex_ante_reasons
        and value.get("scorecard_result_available") is result_available
        and value.get("scorecard_eligible") is (ex_ante_eligible and result_available)
        and value.get("scorecard_ineligibility_reasons") == expected_result_reasons
        and value.get("truth_path_sha256")
        and _is_sha256(value.get("truth_path_sha256"))
        and value.get("missing_no_tick_intervals_synthesized") is False
    ):
        return False
    try:
        activation = _parse_utc(
            value.get("activation_at_utc"),
            name="outcome activation_at_utc",
        )
        expected_activation = _vehicle_activation(vehicle_row)
    except (TypeError, ValueError, OverflowError):
        return False
    if activation != expected_activation:
        return False
    try:
        if status == "MATURE_UNFILLED":
            return _unfilled_outcome_semantics_valid(value, vehicle_row)
        if status == "ZERO_PNL_CONTROL":
            return _zero_outcome_semantics_valid(value, vehicle_row)
        if status in {
            "MATURE_FILLED_TAKE_PROFIT",
            "MATURE_FILLED_STOP_LOSS",
            "MATURE_FILLED_HOLD_END_FULL_STOP",
        }:
            return _exact_filled_outcome_semantics_valid(value, vehicle_row)
        if status == "MATURE_PROXY_RESOLVED":
            return _resolved_proxy_semantics_valid(value, vehicle_row)
        if status == "UNRESOLVED_BASE_ARM_OUTCOME_REQUIRED":
            return _unresolved_proxy_semantics_valid(value, vehicle_row)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return False


def _unfilled_outcome_semantics_valid(
    value: Mapping[str, Any],
    vehicle: Mapping[str, Any],
) -> bool:
    execution = vehicle.get("execution")
    if not isinstance(execution, Mapping):
        return False
    activation, expiry, _maturity, _hold = _execution_clocks(execution)
    return bool(
        value.get("filled") is False
        and value.get("fill_at_utc") is None
        and value.get("fill_price") is None
        and value.get("entry_gap_worse") is False
        and value.get("entry_gap_pips") == 0.0
        and value.get("exit_reason") == "UNFILLED_EXPIRED"
        and value.get("exit_at_utc") is None
        and value.get("exit_price") is None
        and value.get("stop_gap_worse") is False
        and value.get("ambiguous_same_s5") is False
        and value.get("post_cost_realized_pips") == 0.0
        and _mark_fields_are_null(value)
        and value.get("resolution_reasons") == []
        and _truth_counts_match(value, start=activation, end=expiry)
    )


def _zero_outcome_semantics_valid(
    value: Mapping[str, Any],
    vehicle: Mapping[str, Any],
) -> bool:
    return bool(
        vehicle.get("hypothesis_id") == "H08"
        and isinstance(vehicle.get("control"), Mapping)
        and value.get("filled") is False
        and value.get("fill_at_utc") is None
        and value.get("fill_price") is None
        and value.get("entry_gap_worse") is False
        and value.get("entry_gap_pips") == 0.0
        and value.get("exit_reason") == "NO_TRADE_ZERO_PNL"
        and value.get("exit_at_utc") is None
        and value.get("exit_price") is None
        and value.get("stop_gap_worse") is False
        and value.get("ambiguous_same_s5") is False
        and value.get("post_cost_realized_pips") == 0.0
        and _mark_fields_are_null(value)
        and value.get("truth_candle_count") == 0
        and value.get("truth_no_tick_slot_count") == 0
        and value.get("resolution_reasons") == []
    )


def _exact_filled_outcome_semantics_valid(
    value: Mapping[str, Any],
    vehicle: Mapping[str, Any],
) -> bool:
    execution = vehicle.get("execution")
    side = str(vehicle.get("predicted_side") or "")
    if not isinstance(execution, Mapping) or side not in _SIDES:
        return False
    activation, expiry, latest_maturity, hold_seconds = _execution_clocks(execution)
    fill_at = _parse_utc(value.get("fill_at_utc"), name="fill_at_utc")
    exit_at = _parse_utc(value.get("exit_at_utc"), name="exit_at_utc")
    fill_horizon = min(
        fill_at + timedelta(seconds=hold_seconds),
        latest_maturity,
    )
    if not bool(
        activation <= fill_at
        and _floor_s5(fill_at) == fill_at
        and _floor_s5(exit_at) == exit_at
        and fill_at + timedelta(seconds=S5_SECONDS) <= expiry
        and fill_at <= exit_at <= fill_horizon
        and _truth_counts_match(value, start=activation, end=fill_horizon)
        and value.get("filled") is True
        and value.get("truth_candle_count", 0) >= 1
        and value.get("resolution_reasons") == []
    ):
        return False
    entry = _positive(execution.get("entry_price"), name="entry")
    target = _positive(execution.get("take_profit_price"), name="target")
    stop = _positive(execution.get("stop_loss_price"), name="stop")
    fill_price = _positive(value.get("fill_price"), name="fill_price")
    exit_price = _positive(value.get("exit_price"), name="exit_price")
    realized = _finite(value.get("post_cost_realized_pips"), allow_none=False)
    pip_factor = _vehicle_pip_factor(vehicle, entry=entry, target=target)
    expected_entry_gap = (
        (fill_price - entry) * pip_factor
        if side == "LONG"
        else (entry - fill_price) * pip_factor
    )
    if not bool(
        (fill_price >= entry if side == "LONG" else fill_price <= entry)
        and value.get("entry_gap_worse") is (expected_entry_gap > 0.0)
        and _same_number(value.get("entry_gap_pips"), expected_entry_gap)
    ):
        return False
    status = str(value["status"])
    if status == "MATURE_FILLED_TAKE_PROFIT":
        expected_realized = (
            (target - fill_price) * pip_factor
            if side == "LONG"
            else (fill_price - target) * pip_factor
        )
        return bool(
            fill_at < exit_at < fill_horizon
            and value.get("exit_reason") == "TAKE_PROFIT"
            and _same_number(exit_price, target)
            and value.get("stop_gap_worse") is False
            and value.get("ambiguous_same_s5") is False
            and _same_number(realized, expected_realized)
            and _mark_fields_are_null(value)
        )
    if status == "MATURE_FILLED_HOLD_END_FULL_STOP":
        expected_realized = (
            (stop - fill_price) * pip_factor
            if side == "LONG"
            else (fill_price - stop) * pip_factor
        )
        mark_at = _parse_utc(value.get("hold_end_mark_at_utc"), name="mark_at")
        mark_price = _positive(value.get("hold_end_mark_price"), name="mark_price")
        expected_mark = (
            (mark_price - fill_price) * pip_factor
            if side == "LONG"
            else (fill_price - mark_price) * pip_factor
        )
        return bool(
            exit_at == fill_horizon
            and value.get("exit_reason") == "HOLD_END_FULL_FROZEN_STOP"
            and _same_number(exit_price, stop)
            and value.get("stop_gap_worse") is False
            and value.get("ambiguous_same_s5") is False
            and _same_number(realized, expected_realized)
            and fill_at < mark_at <= fill_horizon
            and _same_number(value.get("hold_end_mark_pips"), expected_mark)
        )
    return _stop_outcome_semantics_valid(
        value,
        side=side,
        fill_at=fill_at,
        exit_at=exit_at,
        fill_horizon=fill_horizon,
        fill_price=fill_price,
        stop=stop,
        exit_price=exit_price,
        realized=realized,
        pip_factor=pip_factor,
    )


def _stop_outcome_semantics_valid(
    value: Mapping[str, Any],
    *,
    side: str,
    fill_at: datetime,
    exit_at: datetime,
    fill_horizon: datetime,
    fill_price: float,
    stop: float,
    exit_price: float,
    realized: float,
    pip_factor: float,
) -> bool:
    reason = str(value.get("exit_reason") or "")
    semantics = {
        "STOP_LOSS": (False, False, "LATER"),
        "STOP_LOSS_GAP": (True, False, "LATER"),
        "STOP_LOSS_AMBIGUOUS_FILL_S5": (False, True, "FILL"),
        "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5": (True, True, "FILL"),
        "STOP_LOSS_AMBIGUOUS_SAME_S5": (False, True, "LATER"),
        "STOP_LOSS_GAP_AMBIGUOUS_SAME_S5": (True, True, "LATER"),
    }.get(reason)
    if semantics is None:
        return False
    gap, ambiguous, timing = semantics
    expected_realized = (
        (exit_price - fill_price) * pip_factor
        if side == "LONG"
        else (fill_price - exit_price) * pip_factor
    )
    return bool(
        exit_at < fill_horizon
        and (exit_at == fill_at if timing == "FILL" else exit_at > fill_at)
        and value.get("stop_gap_worse") is gap
        and value.get("ambiguous_same_s5") is ambiguous
        and (
            (exit_price < stop if side == "LONG" else exit_price > stop)
            if gap
            else _same_number(exit_price, stop)
        )
        and _same_number(realized, expected_realized)
        and _mark_fields_are_null(value)
    )


def _resolved_proxy_semantics_valid(
    value: Mapping[str, Any],
    vehicle: Mapping[str, Any],
) -> bool:
    source = value.get("base_proxy_source")
    binding = vehicle.get("proxy_binding")
    filled = value.get("filled")
    if not isinstance(binding, Mapping) or not isinstance(source, Mapping):
        return False
    try:
        source_start = _vehicle_activation(vehicle)
        source_maturity = _parse_utc(
            source.get("source_maturity_at_utc"), name="source_maturity"
        )
        source_resolved = _parse_utc(
            source.get("source_resolved_at_utc"), name="source_resolved"
        )
        source_entry = _positive(
            source.get("source_entry_price"), name="source_entry_price"
        )
        source_ttl = _positive_int(
            source.get("source_entry_ttl_seconds"), name="source_entry_ttl_seconds"
        )
        source_hold = _positive_int(
            source.get("source_max_hold_seconds"), name="source_max_hold_seconds"
        )
        source_expiry = _parse_utc(
            source.get("source_entry_expires_at_utc"), name="source_entry_expires"
        )
        realized = _finite(value.get("post_cost_realized_pips"), allow_none=False)
    except (TypeError, ValueError, OverflowError):
        return False
    if not bool(
        filled.__class__ is bool
        and math.isfinite(realized)
        and source_expiry == source_start + timedelta(seconds=source_ttl)
        and source_maturity == source_expiry + timedelta(seconds=source_hold)
        and source_resolved >= source_maturity
        and all(
            _is_sha256(source.get(key))
            for key in (
                "learning_seat_contract_sha256",
                "candidate_input_sha256",
                "arm_input_sha256",
                "learning_outcome_contract_sha256",
                "candidate_outcome_sha256",
                "arm_outcome_sha256",
                "truth_path_sha256",
            )
        )
        and source.get("learning_seat_contract_sha256")
        == binding.get("learning_seat_contract_sha256")
        and source.get("candidate_id") == binding.get("candidate_id")
        and source.get("candidate_input_sha256") == binding.get("candidate_sha256")
        and source.get("arm_id") == binding.get("arm_id")
        and source.get("arm_input_sha256") == binding.get("arm_sha256")
        and source.get("source_side") == binding.get("side")
        and source.get("source_filled") is filled
        and source.get("source_fill_at_utc") == value.get("fill_at_utc")
        and source.get("source_exit_at_utc") == value.get("exit_at_utc")
        and source.get("source_exit_reason") == value.get("exit_reason")
        and source.get("source_post_cost_realized_pips")
        == value.get("post_cost_realized_pips")
        and source.get("source_ambiguous_same_s5") is value.get("ambiguous_same_s5")
        and source.get("source_truth_candle_count") == value.get("truth_candle_count")
        and source.get("source_truth_no_tick_slot_count")
        == value.get("truth_no_tick_slot_count")
        and value.get("entry_gap_worse") is None
        and value.get("entry_gap_pips") is None
        and value.get("exit_price") is None
        and isinstance(value.get("stop_gap_worse"), bool)
        and isinstance(value.get("ambiguous_same_s5"), bool)
        and _mark_fields_are_null(value)
        and _nonnegative_count(value.get("truth_candle_count"))
        and _nonnegative_count(value.get("truth_no_tick_slot_count"))
        and value.get("resolution_reasons") == []
    ):
        return False
    if filled:
        fill_at = _parse_utc(value.get("fill_at_utc"), name="proxy fill")
        exit_at = _parse_utc(value.get("exit_at_utc"), name="proxy exit")
        reason = str(value.get("exit_reason") or "")
        timing = _proxy_exit_timing(reason)
        fill_horizon = min(
            fill_at + timedelta(seconds=source_hold),
            source_maturity,
        )
        return bool(
            timing is not None
            and source_start <= fill_at
            and _floor_s5(fill_at) == fill_at
            and fill_at + timedelta(seconds=S5_SECONDS) <= source_expiry
            and fill_at <= exit_at <= fill_horizon
            and _floor_s5(exit_at) == exit_at
            and _same_number(value.get("fill_price"), source_entry)
            and value.get("truth_candle_count", 0) >= 1
            and value.get("stop_gap_worse") is ("GAP" in reason)
            and value.get("ambiguous_same_s5") is ("AMBIGUOUS" in reason)
            and (realized > 0.0 if reason == "TAKE_PROFIT" else realized < 0.0)
            and (
                exit_at == fill_at
                if timing == "FILL"
                else exit_at == fill_horizon
                if timing == "HORIZON"
                else fill_at < exit_at < fill_horizon
            )
        )
    return bool(
        value.get("fill_at_utc") is None
        and value.get("fill_price") is None
        and value.get("exit_at_utc") is None
        and value.get("exit_reason") == "UNFILLED"
        and value.get("stop_gap_worse") is False
        and value.get("ambiguous_same_s5") is False
        and realized == 0.0
    )


def _unresolved_proxy_semantics_valid(
    value: Mapping[str, Any],
    vehicle: Mapping[str, Any],
) -> bool:
    nullable = (
        "filled",
        "fill_at_utc",
        "fill_price",
        "entry_gap_worse",
        "entry_gap_pips",
        "exit_reason",
        "exit_at_utc",
        "exit_price",
        "stop_gap_worse",
        "ambiguous_same_s5",
        "post_cost_realized_pips",
        "hold_end_mark_at_utc",
        "hold_end_mark_price",
        "hold_end_mark_pips",
        "truth_candle_count",
        "truth_no_tick_slot_count",
        "base_proxy_source",
    )
    return bool(
        isinstance(vehicle.get("proxy_binding"), Mapping)
        and all(value.get(key) is None for key in nullable)
        and value.get("resolution_reasons")
        == sorted(set(value.get("resolution_reasons", [])))
        and set(value.get("resolution_reasons", [])).issubset(_UNRESOLVED_PROXY_REASONS)
    )


def _proxy_exit_timing(reason: str) -> str | None:
    if reason == "TAKE_PROFIT" or reason in {"STOP_LOSS", "STOP_LOSS_GAP"}:
        return "LATER"
    if reason in {
        "STOP_LOSS_AMBIGUOUS_FILL_S5",
        "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5",
    }:
        return "FILL"
    if reason in {
        "STOP_LOSS_AMBIGUOUS_SAME_S5",
        "STOP_LOSS_GAP_AMBIGUOUS_SAME_S5",
    }:
        return "LATER"
    if reason == "HORIZON_FULL_STOP_LOSS":
        return "HORIZON"
    return None


def _execution_clocks(
    execution: Mapping[str, Any],
) -> tuple[datetime, datetime, datetime, int]:
    activation = _parse_utc(execution.get("activation_at_utc"), name="activation")
    expiry = _parse_utc(execution.get("entry_expires_at_utc"), name="expiry")
    maturity = _parse_utc(execution.get("latest_maturity_at_utc"), name="maturity")
    hold = _positive_int(execution.get("max_hold_seconds"), name="max_hold")
    if not activation < expiry < maturity:
        raise ValueError("execution clocks are invalid")
    return activation, expiry, maturity, hold


def _vehicle_activation(vehicle: Mapping[str, Any]) -> datetime:
    for key in ("execution", "proxy_binding", "source_binding"):
        source = vehicle.get(key)
        if isinstance(source, Mapping) and source.get("activation_at_utc"):
            return _parse_utc(source["activation_at_utc"], name="vehicle activation")
    raise ValueError("vehicle activation is missing")


def _vehicle_pip_factor(
    vehicle: Mapping[str, Any],
    *,
    entry: float,
    target: float,
) -> float:
    observation = vehicle.get("cost_observation")
    if not isinstance(observation, Mapping):
        raise ValueError("vehicle cost observation is missing")
    gross = _positive(
        observation.get("gross_take_profit_pips"),
        name="gross_take_profit_pips",
    )
    distance = abs(target - entry)
    if distance <= 0.0:
        raise ValueError("target distance is invalid")
    return gross / distance


def _truth_counts_match(
    value: Mapping[str, Any],
    *,
    start: datetime,
    end: datetime,
) -> bool:
    candles = value.get("truth_candle_count")
    no_tick = value.get("truth_no_tick_slot_count")
    return bool(
        _nonnegative_count(candles)
        and _nonnegative_count(no_tick)
        and candles + no_tick == _grid_slot_count(start, end)
    )


def _nonnegative_count(value: Any) -> bool:
    return value.__class__ is int and value >= 0


def _mark_fields_are_null(value: Mapping[str, Any]) -> bool:
    return all(
        value.get(key) is None
        for key in (
            "hold_end_mark_at_utc",
            "hold_end_mark_price",
            "hold_end_mark_pips",
        )
    )


def _same_number(left: Any, right: float) -> bool:
    try:
        parsed = _finite(left, allow_none=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isclose(parsed, _round(right), rel_tol=0.0, abs_tol=1e-6)


def _resolve_exact_stop(
    vehicle: Mapping[str, Any],
    *,
    candles: Sequence[S5BidAskCandle],
    pip_factor: float,
    truth_path_sha256: str,
) -> dict[str, Any]:
    execution = vehicle["execution"]
    side = str(vehicle["predicted_side"])
    if side not in _SIDES:
        raise ValueError("exact STOP vehicle side is invalid")
    activation = _parse_utc(execution.get("activation_at_utc"), name="activation")
    expiry = _parse_utc(execution.get("entry_expires_at_utc"), name="expiry")
    maturity = _parse_utc(
        execution.get("latest_maturity_at_utc"),
        name="maturity",
    )
    if not activation < expiry < maturity:
        raise ValueError("exact STOP vehicle clocks are invalid")
    entry = _positive(execution.get("entry_price"), name="entry")
    target = _positive(execution.get("take_profit_price"), name="target")
    stop = _positive(execution.get("stop_loss_price"), name="stop")
    max_hold_seconds = _positive_int(
        execution.get("max_hold_seconds"),
        name="max_hold_seconds",
    )
    if side == "LONG" and not stop < entry < target:
        raise ValueError("LONG STOP geometry is invalid")
    if side == "SHORT" and not target < entry < stop:
        raise ValueError("SHORT STOP geometry is invalid")
    path = [
        candle
        for candle in candles
        if _ceil_s5(activation) <= candle.timestamp_utc < _floor_s5(maturity)
    ]
    entry_path = [candle for candle in path if candle.timestamp_utc < _floor_s5(expiry)]
    fill_index = next(
        (
            index
            for index, candle in enumerate(entry_path)
            if (side == "LONG" and candle.ask_h >= entry)
            or (side == "SHORT" and candle.bid_l <= entry)
        ),
        None,
    )
    base = _outcome_base(vehicle, truth_path_sha256=truth_path_sha256)
    if fill_index is None:
        return _seal_outcome(
            {
                **base,
                "status": "MATURE_UNFILLED",
                "filled": False,
                "fill_at_utc": None,
                "fill_price": None,
                "entry_gap_worse": False,
                "entry_gap_pips": 0.0,
                "exit_reason": "UNFILLED_EXPIRED",
                "exit_at_utc": None,
                "exit_price": None,
                "stop_gap_worse": False,
                "ambiguous_same_s5": False,
                "post_cost_realized_pips": 0.0,
                "hold_end_mark_at_utc": None,
                "hold_end_mark_price": None,
                "hold_end_mark_pips": None,
                "truth_candle_count": len(entry_path),
                "truth_no_tick_slot_count": _grid_slot_count(activation, expiry)
                - len(entry_path),
                "missing_no_tick_intervals_synthesized": False,
                "resolution_reasons": [],
            }
        )

    fill_candle = entry_path[fill_index]
    fill_horizon = min(
        fill_candle.timestamp_utc + timedelta(seconds=max_hold_seconds),
        maturity,
    )
    scoring_path = [
        candle for candle in path if candle.timestamp_utc < _floor_s5(fill_horizon)
    ]
    fill_position = scoring_path.index(fill_candle)
    opening_trigger = fill_candle.ask_o if side == "LONG" else fill_candle.bid_o
    entry_triggered_at_open = (
        opening_trigger >= entry if side == "LONG" else opening_trigger <= entry
    )
    fill_price = (
        max(entry, opening_trigger) if side == "LONG" else min(entry, opening_trigger)
    )
    entry_gap_pips = (
        (fill_price - entry) * pip_factor
        if side == "LONG"
        else (entry - fill_price) * pip_factor
    )
    hold_mark_candle: S5BidAskCandle | None = None
    for offset, candle in enumerate(scoring_path[fill_position:]):
        target_hit, stop_hit = _exit_hits(
            side,
            candle,
            take_profit=target,
            stop_loss=stop,
        )
        is_fill_candle = offset == 0
        if is_fill_candle and (target_hit or stop_hit):
            return _stop_outcome(
                base,
                vehicle=vehicle,
                candle=candle,
                fill_at=fill_candle.timestamp_utc,
                fill_price=fill_price,
                entry_gap_pips=entry_gap_pips,
                stop_loss=stop,
                pip_factor=pip_factor,
                reason="STOP_LOSS_AMBIGUOUS_FILL_S5",
                ambiguous=True,
                opening_gap_allowed=entry_triggered_at_open,
                path=scoring_path,
                activation=activation,
                maturity=fill_horizon,
            )
        if not is_fill_candle and target_hit and stop_hit:
            return _stop_outcome(
                base,
                vehicle=vehicle,
                candle=candle,
                fill_at=fill_candle.timestamp_utc,
                fill_price=fill_price,
                entry_gap_pips=entry_gap_pips,
                stop_loss=stop,
                pip_factor=pip_factor,
                reason="STOP_LOSS_AMBIGUOUS_SAME_S5",
                ambiguous=True,
                opening_gap_allowed=True,
                path=scoring_path,
                activation=activation,
                maturity=fill_horizon,
            )
        if not is_fill_candle and stop_hit:
            return _stop_outcome(
                base,
                vehicle=vehicle,
                candle=candle,
                fill_at=fill_candle.timestamp_utc,
                fill_price=fill_price,
                entry_gap_pips=entry_gap_pips,
                stop_loss=stop,
                pip_factor=pip_factor,
                reason="STOP_LOSS",
                ambiguous=False,
                opening_gap_allowed=True,
                path=scoring_path,
                activation=activation,
                maturity=fill_horizon,
            )
        if not is_fill_candle and target_hit:
            realized = (
                (target - fill_price) * pip_factor
                if side == "LONG"
                else (fill_price - target) * pip_factor
            )
            return _seal_outcome(
                {
                    **base,
                    "status": "MATURE_FILLED_TAKE_PROFIT",
                    "filled": True,
                    "fill_at_utc": fill_candle.timestamp_utc.isoformat(),
                    "fill_price": _round(fill_price),
                    "entry_gap_worse": entry_gap_pips > 0.0,
                    "entry_gap_pips": _round(entry_gap_pips),
                    "exit_reason": "TAKE_PROFIT",
                    "exit_at_utc": candle.timestamp_utc.isoformat(),
                    "exit_price": _round(target),
                    "stop_gap_worse": False,
                    "ambiguous_same_s5": False,
                    "post_cost_realized_pips": _round(realized),
                    "hold_end_mark_at_utc": None,
                    "hold_end_mark_price": None,
                    "hold_end_mark_pips": None,
                    "truth_candle_count": len(scoring_path),
                    "truth_no_tick_slot_count": _grid_slot_count(
                        activation,
                        fill_horizon,
                    )
                    - len(scoring_path),
                    "missing_no_tick_intervals_synthesized": False,
                    "resolution_reasons": [],
                }
            )
        hold_mark_candle = candle

    if hold_mark_candle is None:
        raise AssertionError("filled STOP path lost its fill candle")
    mark_price = hold_mark_candle.bid_c if side == "LONG" else hold_mark_candle.ask_c
    mark_pips = (
        (mark_price - fill_price) * pip_factor
        if side == "LONG"
        else (fill_price - mark_price) * pip_factor
    )
    full_stop_pips = (
        (stop - fill_price) * pip_factor
        if side == "LONG"
        else (fill_price - stop) * pip_factor
    )
    return _seal_outcome(
        {
            **base,
            "status": "MATURE_FILLED_HOLD_END_FULL_STOP",
            "filled": True,
            "fill_at_utc": fill_candle.timestamp_utc.isoformat(),
            "fill_price": _round(fill_price),
            "entry_gap_worse": entry_gap_pips > 0.0,
            "entry_gap_pips": _round(entry_gap_pips),
            "exit_reason": "HOLD_END_FULL_FROZEN_STOP",
            "exit_at_utc": fill_horizon.isoformat(),
            "exit_price": _round(stop),
            "stop_gap_worse": False,
            "ambiguous_same_s5": False,
            "post_cost_realized_pips": _round(full_stop_pips),
            "hold_end_mark_at_utc": (
                hold_mark_candle.timestamp_utc + timedelta(seconds=S5_SECONDS)
            ).isoformat(),
            "hold_end_mark_price": _round(mark_price),
            "hold_end_mark_pips": _round(mark_pips),
            "truth_candle_count": len(scoring_path),
            "truth_no_tick_slot_count": _grid_slot_count(activation, fill_horizon)
            - len(scoring_path),
            "missing_no_tick_intervals_synthesized": False,
            "resolution_reasons": [],
        }
    )


def _stop_outcome(
    base: Mapping[str, Any],
    *,
    vehicle: Mapping[str, Any],
    candle: S5BidAskCandle,
    fill_at: datetime,
    fill_price: float,
    entry_gap_pips: float,
    stop_loss: float,
    pip_factor: float,
    reason: str,
    ambiguous: bool,
    opening_gap_allowed: bool,
    path: Sequence[S5BidAskCandle],
    activation: datetime,
    maturity: datetime,
) -> dict[str, Any]:
    side = str(vehicle["predicted_side"])
    opening_exit = candle.bid_o if side == "LONG" else candle.ask_o
    stop_gap = bool(
        opening_gap_allowed
        and (opening_exit < stop_loss if side == "LONG" else opening_exit > stop_loss)
    )
    exit_price = opening_exit if stop_gap else stop_loss
    realized = (
        (exit_price - fill_price) * pip_factor
        if side == "LONG"
        else (fill_price - exit_price) * pip_factor
    )
    final_reason = _stop_gap_reason(reason) if stop_gap else reason
    return _seal_outcome(
        {
            **base,
            "status": "MATURE_FILLED_STOP_LOSS",
            "filled": True,
            "fill_at_utc": fill_at.isoformat(),
            "fill_price": _round(fill_price),
            "entry_gap_worse": entry_gap_pips > 0.0,
            "entry_gap_pips": _round(entry_gap_pips),
            "exit_reason": final_reason,
            "exit_at_utc": candle.timestamp_utc.isoformat(),
            "exit_price": _round(exit_price),
            "stop_gap_worse": stop_gap,
            "ambiguous_same_s5": ambiguous,
            "post_cost_realized_pips": _round(realized),
            "hold_end_mark_at_utc": None,
            "hold_end_mark_price": None,
            "hold_end_mark_pips": None,
            "truth_candle_count": len(path),
            "truth_no_tick_slot_count": _grid_slot_count(activation, maturity)
            - len(path),
            "missing_no_tick_intervals_synthesized": False,
            "resolution_reasons": [],
        }
    )


def _resolve_base_proxy(
    vehicle: Mapping[str, Any],
    *,
    pair: str,
    base_outcomes: Sequence[Mapping[str, Any]],
    truth_path_sha256: str,
    resolved_at_utc: datetime,
) -> dict[str, Any]:
    binding = vehicle["proxy_binding"]
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    for outcome in base_outcomes:
        if (
            outcome.get("seat_contract_sha256")
            != binding.get("learning_seat_contract_sha256")
            or outcome.get("pair") != pair
        ):
            continue
        for candidate in outcome.get("candidates", []):
            if not isinstance(candidate, Mapping):
                continue
            if candidate.get("candidate_id") != binding.get(
                "candidate_id"
            ) or candidate.get("candidate_sha256") != binding.get("candidate_sha256"):
                continue
            for arm in candidate.get("arms", []):
                if isinstance(arm, Mapping) and bool(
                    arm.get("arm_id") == binding.get("arm_id")
                    and arm.get("arm_input_sha256") == binding.get("arm_sha256")
                ):
                    matches.append((outcome, candidate, arm))
    base = _outcome_base(vehicle, truth_path_sha256=truth_path_sha256)
    if len(matches) != 1:
        reason = (
            "SEALED_MATCHED_BASE_ARM_OUTCOME_MISSING"
            if not matches
            else "SEALED_MATCHED_BASE_ARM_OUTCOME_NOT_UNIQUE"
        )
        return _unresolved_proxy_outcome(base, reasons=[reason])
    learning_outcome, candidate, arm = matches[0]
    source_resolved = _parse_utc(
        learning_outcome.get("resolved_at_utc"),
        name="base outcome resolved_at_utc",
    )
    source_maturity = _parse_utc(
        arm.get("maturity_at_utc"),
        name="base arm maturity_at_utc",
    )
    scoring_start = _parse_utc(
        binding.get("scoring_start_at_utc"),
        name="base proxy scoring_start_at_utc",
    )
    source_start = _parse_utc(
        learning_outcome.get("seat_generated_at_utc"),
        name="base outcome seat_generated_at_utc",
    )
    clock_reasons: list[str] = []
    if source_start != scoring_start:
        clock_reasons.append("SEALED_BASE_ARM_SCORING_START_MISMATCH")
    if source_maturity > resolved_at_utc or source_resolved > resolved_at_utc:
        clock_reasons.append("SEALED_BASE_ARM_OUTCOME_AFTER_RESOLUTION_CLOCK")
    if clock_reasons:
        return _unresolved_proxy_outcome(
            base,
            reasons=clock_reasons,
        )
    realized = _finite(arm.get("post_cost_realized_pips"), allow_none=False)
    return _seal_outcome(
        {
            **base,
            "status": "MATURE_PROXY_RESOLVED",
            "filled": arm.get("filled") is True,
            "fill_at_utc": arm.get("fill_at_utc"),
            "fill_price": arm.get("entry") if arm.get("filled") is True else None,
            "entry_gap_worse": None,
            "entry_gap_pips": None,
            "exit_reason": arm.get("exit_reason"),
            "exit_at_utc": arm.get("exit_at_utc"),
            "exit_price": None,
            "stop_gap_worse": "GAP" in str(arm.get("exit_reason") or ""),
            "ambiguous_same_s5": arm.get("ambiguous_same_s5"),
            "post_cost_realized_pips": _round(realized),
            "hold_end_mark_at_utc": None,
            "hold_end_mark_price": None,
            "hold_end_mark_pips": None,
            "truth_candle_count": arm.get("truth_candle_count"),
            "truth_no_tick_slot_count": arm.get("truth_no_tick_slot_count"),
            "missing_no_tick_intervals_synthesized": False,
            "base_proxy_source": {
                "learning_seat_contract_sha256": binding[
                    "learning_seat_contract_sha256"
                ],
                "candidate_id": binding["candidate_id"],
                "candidate_input_sha256": binding["candidate_sha256"],
                "arm_id": binding["arm_id"],
                "arm_input_sha256": binding["arm_sha256"],
                "source_side": candidate["side"],
                "source_entry_price": arm["entry"],
                "source_entry_ttl_seconds": arm["entry_ttl_seconds"],
                "source_max_hold_seconds": arm["max_hold_seconds"],
                "source_entry_expires_at_utc": (
                    source_start + timedelta(seconds=int(arm["entry_ttl_seconds"]))
                ).isoformat(),
                "source_filled": arm["filled"],
                "source_fill_at_utc": arm["fill_at_utc"],
                "source_exit_at_utc": arm["exit_at_utc"],
                "source_exit_reason": arm["exit_reason"],
                "source_post_cost_realized_pips": arm["post_cost_realized_pips"],
                "source_ambiguous_same_s5": arm["ambiguous_same_s5"],
                "source_truth_candle_count": arm["truth_candle_count"],
                "source_truth_no_tick_slot_count": arm["truth_no_tick_slot_count"],
                "learning_outcome_contract_sha256": learning_outcome["contract_sha256"],
                "candidate_outcome_sha256": candidate["candidate_outcome_sha256"],
                "arm_outcome_sha256": arm["arm_outcome_sha256"],
                "truth_path_sha256": arm["truth_path_sha256"],
                "source_maturity_at_utc": source_maturity.isoformat(),
                "source_resolved_at_utc": source_resolved.isoformat(),
            },
            "resolution_reasons": [],
        }
    )


def _unresolved_proxy_outcome(
    base: Mapping[str, Any],
    *,
    reasons: Sequence[str],
) -> dict[str, Any]:
    return _seal_outcome(
        {
            **base,
            "scorecard_result_available": False,
            "scorecard_eligible": False,
            "scorecard_ineligibility_reasons": sorted(
                {
                    *base.get("scorecard_ineligibility_reasons", []),
                    "OUTCOME_RESULT_UNAVAILABLE",
                    *reasons,
                }
            ),
            "status": "UNRESOLVED_BASE_ARM_OUTCOME_REQUIRED",
            "filled": None,
            "fill_at_utc": None,
            "fill_price": None,
            "entry_gap_worse": None,
            "entry_gap_pips": None,
            "exit_reason": None,
            "exit_at_utc": None,
            "exit_price": None,
            "stop_gap_worse": None,
            "ambiguous_same_s5": None,
            "post_cost_realized_pips": None,
            "hold_end_mark_at_utc": None,
            "hold_end_mark_price": None,
            "hold_end_mark_pips": None,
            "truth_candle_count": None,
            "truth_no_tick_slot_count": None,
            "missing_no_tick_intervals_synthesized": False,
            "base_proxy_source": None,
            "resolution_reasons": sorted(set(reasons)),
        }
    )


def _zero_control_outcome(
    vehicle: Mapping[str, Any],
    *,
    truth_path_sha256: str,
) -> dict[str, Any]:
    return _seal_outcome(
        {
            **_outcome_base(vehicle, truth_path_sha256=truth_path_sha256),
            "status": "ZERO_PNL_CONTROL",
            "filled": False,
            "fill_at_utc": None,
            "fill_price": None,
            "entry_gap_worse": False,
            "entry_gap_pips": 0.0,
            "exit_reason": "NO_TRADE_ZERO_PNL",
            "exit_at_utc": None,
            "exit_price": None,
            "stop_gap_worse": False,
            "ambiguous_same_s5": False,
            "post_cost_realized_pips": 0.0,
            "hold_end_mark_at_utc": None,
            "hold_end_mark_price": None,
            "hold_end_mark_pips": None,
            "truth_candle_count": 0,
            "truth_no_tick_slot_count": 0,
            "missing_no_tick_intervals_synthesized": False,
            "resolution_reasons": [],
        }
    )


def _validated_base_outcomes(
    values: Sequence[Mapping[str, Any]],
    *,
    learning_seat: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("base_arm_outcomes must be a sequence")
    rows: list[Mapping[str, Any]] = []
    for outcome in values:
        if not isinstance(outcome, Mapping) or not _sealed_valid(
            outcome,
            LEARNING_OUTCOME_CONTRACT,
        ):
            raise ValueError("base learning outcome seal is invalid")
        candidates = outcome.get("candidates")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise ValueError("base learning outcome candidates are invalid")
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not _sealed_named_valid(
                candidate,
                CANDIDATE_OUTCOME_CONTRACT,
                "candidate_outcome_sha256",
            ):
                raise ValueError("base candidate outcome seal is invalid")
            arms = candidate.get("arms")
            if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence):
                raise ValueError("base candidate arms are invalid")
            for arm in arms:
                if not isinstance(arm, Mapping) or not _sealed_named_valid(
                    arm,
                    ARM_OUTCOME_CONTRACT,
                    "arm_outcome_sha256",
                ):
                    raise ValueError("base arm outcome seal is invalid")
                if arm.get("truth_path_sha256") != outcome.get("truth_path_sha256"):
                    raise ValueError("base arm truth path binding is invalid")
        if outcome.get("seat_contract_sha256") == learning_seat.get(
            "contract_sha256"
        ) and not _outcome_valid_for_seat(outcome, learning_seat):
            raise ValueError("matched base learning outcome is invalid for its seat")
        rows.append(outcome)
    return rows


def _validated_truth_path(
    candles: Sequence[S5BidAskCandle],
    *,
    activation: datetime,
    maturity: datetime,
) -> tuple[S5BidAskCandle, ...]:
    if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence):
        raise ValueError("S5 candles must be a sequence")
    first = _ceil_s5(activation)
    end = _floor_s5(maturity)
    grid_count = _grid_slot_count(activation, maturity)
    normalized = tuple(candles)
    previous: datetime | None = None
    for candle in normalized:
        if candle.__class__ is not S5BidAskCandle:
            raise ValueError("S5 candle class is invalid")
        timestamp = _aware_utc(candle.timestamp_utc)
        if not first <= timestamp < end or _floor_s5(timestamp) != timestamp:
            raise ValueError("S5 candle lies outside the exact frozen interval")
        if previous is not None and timestamp <= previous:
            raise ValueError("S5 candles must be chronological and unique")
        previous = timestamp
        _validate_candle(candle)
    if len(normalized) > grid_count:
        raise ValueError("S5 candles exceed frozen grid")
    return normalized


def _validate_candle(candle: S5BidAskCandle) -> None:
    bid = (candle.bid_o, candle.bid_h, candle.bid_l, candle.bid_c)
    ask = (candle.ask_o, candle.ask_h, candle.ask_l, candle.ask_c)
    if any(_finite(value, allow_none=False) <= 0.0 for value in (*bid, *ask)):
        raise ValueError("S5 candle contains invalid price")
    if not bool(
        candle.bid_l <= min(candle.bid_o, candle.bid_c)
        and max(candle.bid_o, candle.bid_c) <= candle.bid_h
        and candle.ask_l <= min(candle.ask_o, candle.ask_c)
        and max(candle.ask_o, candle.ask_c) <= candle.ask_h
        and all(bid_value < ask_value for bid_value, ask_value in zip(bid, ask))
    ):
        raise ValueError("S5 candle OHLC or bid/ask envelope is invalid")


def _vehicle_row_valid(value: Mapping[str, Any]) -> bool:
    if not _sealed_named_valid(value, VEHICLE_CONTRACT_V2, "vehicle_sha256"):
        return False
    return bool(
        value.get("diagnostic_vehicle_available") is True
        and isinstance(value.get("causal_input_proof_eligible"), bool)
        and isinstance(value.get("paired_direction_proof_eligible"), bool)
        and isinstance(value.get("technical_hypothesis_proof_eligible"), bool)
        and isinstance(value.get("scorecard_eligible"), bool)
        and isinstance(value.get("scorecard_ineligibility_reasons"), list)
        and value.get("order_authority") == "NONE"
        and value.get("live_permission") is False
        and value.get("broker_mutation_allowed") is False
        and value.get("promotion_allowed") is False
        and value.get("automatic_promotion_allowed") is False
    )


def _outcome_base(
    vehicle: Mapping[str, Any],
    *,
    truth_path_sha256: str,
) -> dict[str, Any]:
    activation = (
        vehicle.get("execution", {}).get("activation_at_utc")
        if isinstance(vehicle.get("execution"), Mapping)
        else vehicle.get("proxy_binding", {}).get("activation_at_utc")
        if isinstance(vehicle.get("proxy_binding"), Mapping)
        else vehicle.get("source_binding", {}).get("activation_at_utc")
    )
    ex_ante_scorecard_eligible = vehicle.get("scorecard_eligible") is True
    ex_ante_ineligibility_reasons = list(
        vehicle.get("scorecard_ineligibility_reasons", [])
    )
    return {
        "contract": VEHICLE_OUTCOME_CONTRACT_V1,
        "schema_version": 1,
        "scoring_policy": SCORING_POLICY_V1,
        "hypothesis_id": str(vehicle["hypothesis_id"]),
        "vehicle_sha256": str(vehicle["vehicle_sha256"]),
        "predicted_side": vehicle.get("predicted_side"),
        "activation_at_utc": activation,
        "causal_input_proof_eligible": vehicle.get("causal_input_proof_eligible")
        is True,
        "paired_direction_proof_eligible": vehicle.get(
            "paired_direction_proof_eligible"
        )
        is True,
        "technical_hypothesis_proof_eligible": vehicle.get(
            "technical_hypothesis_proof_eligible"
        )
        is True,
        "ex_ante_scorecard_eligible": ex_ante_scorecard_eligible,
        "ex_ante_scorecard_ineligibility_reasons": ex_ante_ineligibility_reasons,
        "scorecard_result_available": True,
        "scorecard_eligible": ex_ante_scorecard_eligible,
        "scorecard_ineligibility_reasons": ex_ante_ineligibility_reasons,
        "truth_path_sha256": truth_path_sha256,
        **_AUTHORITY,
    }


def _exit_hits(
    side: str,
    candle: S5BidAskCandle,
    *,
    take_profit: float,
    stop_loss: float,
) -> tuple[bool, bool]:
    if side == "LONG":
        return candle.bid_h >= take_profit, candle.bid_l <= stop_loss
    return candle.ask_l <= take_profit, candle.ask_h >= stop_loss


def _stop_gap_reason(reason: str) -> str:
    return {
        "STOP_LOSS": "STOP_LOSS_GAP",
        "STOP_LOSS_AMBIGUOUS_FILL_S5": "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5",
        "STOP_LOSS_AMBIGUOUS_SAME_S5": "STOP_LOSS_GAP_AMBIGUOUS_SAME_S5",
    }[reason]


def _candle_payload(candle: S5BidAskCandle) -> dict[str, Any]:
    return {
        "timestamp_utc": _aware_utc(candle.timestamp_utc).isoformat(),
        "bid": [candle.bid_o, candle.bid_h, candle.bid_l, candle.bid_c],
        "ask": [candle.ask_o, candle.ask_h, candle.ask_l, candle.ask_c],
    }


def _grid_slot_count(start: datetime, end: datetime) -> int:
    return max(0, int((_floor_s5(end) - _ceil_s5(start)).total_seconds() // S5_SECONDS))


def _ceil_s5(value: datetime) -> datetime:
    micros = int(round(_aware_utc(value).timestamp() * 1_000_000))
    aligned = ((micros + 4_999_999) // 5_000_000) * 5_000_000
    return datetime.fromtimestamp(aligned / 1_000_000, tz=timezone.utc)


def _floor_s5(value: datetime) -> datetime:
    micros = int(round(_aware_utc(value).timestamp() * 1_000_000))
    aligned = (micros // 5_000_000) * 5_000_000
    return datetime.fromtimestamp(aligned / 1_000_000, tz=timezone.utc)


def _parse_utc(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an aware UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an aware UTC timestamp") from exc
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _positive(value: Any, *, name: str) -> float:
    parsed = _finite(value, allow_none=False)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_int(value: Any, *, name: str) -> int:
    if value.__class__ is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: Any, *, allow_none: bool) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("number must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("number must be finite")
    return parsed


def _round(value: float | int) -> float:
    return round(float(value), ROUND_DIGITS)


def _sha256_text(value: Any, *, name: str) -> str:
    text = str(value or "")
    if not _is_sha256(text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return bool(
        len(text) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in text)
    )


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _canonical_sha(body)}


def _seal_outcome(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "vehicle_outcome_sha256"}
    return {**body, "vehicle_outcome_sha256": _canonical_sha(body)}


def _sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    if value.get("contract") != contract:
        return False
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    try:
        return value.get("contract_sha256") == _canonical_sha(body)
    except (TypeError, ValueError, OverflowError):
        return False


def _sealed_named_valid(
    value: Mapping[str, Any],
    contract: str,
    digest_key: str,
) -> bool:
    if value.get("contract") != contract:
        return False
    body = {key: item for key, item in value.items() if key != digest_key}
    try:
        return value.get(digest_key) == _canonical_sha(body)
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
    "SCORING_POLICY_V1",
    "TRUTH_CONTRACT_V1",
    "VEHICLE_OUTCOME_CONTRACT_V1",
    "resolve_fast_bot_technical_hypothesis_vehicle_truth_v1",
    "technical_hypothesis_vehicle_outcome_v1_valid",
    "technical_hypothesis_vehicle_truth_v1_valid",
]

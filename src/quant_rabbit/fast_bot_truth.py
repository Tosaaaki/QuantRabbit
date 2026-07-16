"""Exact OANDA S5 bid/ask outcomes for deterministic fast-bot shadows."""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.fast_bot import (
    ENTRY_ARM_SPREAD_FRACTIONS,
    ENTRY_EXPERIMENT_CONTRACT,
    ENTRY_TTL_SECONDS,
    HORIZON_LANE,
    MAX_HOLD_SECONDS,
    METHODS,
    SIGNAL_CONTRACT,
    _entry_experiment_arms,
    _shadow_geometry_pips,
)
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle
from quant_rabbit.technical_forecast_forward_truth import fetch_frozen_s5_truth


OUTCOME_CONTRACT = "QR_FAST_BOT_S5_BID_ASK_OUTCOME_V1"
SCORECARD_CONTRACT = "QR_FAST_BOT_FORWARD_SCORECARD_V1"
TRUTH_ADAPTER_CONTRACT = "QR_FAST_BOT_S5_TRUTH_ADAPTER_V1"
SCORING_POLICY = "QR_FAST_BOT_RECEIPTED_SPARSE_S5_CONSERVATIVE_V3"
MAX_DUE_PER_RUN = 12
MAX_WORKERS = 4
TRUTH_CHUNK_CANDLE_LIMIT = 4500
# Outcome pips are persisted at six decimal places throughout this contract.
# Classifying gaps at that same precision prevents binary-float noise at the
# attached-stop boundary from minting a GAP label that its sealed validator
# cannot reproduce.  A future price-decimal contract should replace this.
OUTCOME_PIP_ROUND_DIGITS = 6
SIGNAL_IDENTITY_CONTRACT_V3 = "QR_FAST_BOT_SHADOW_IDENTITY_V3"


def resolve_fast_bot_signal(
    signal: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    resolved_at_utc: datetime,
    truth_chunk_sha256: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve one passive LIMIT signal using executable bid/ask sides.

    No-fill is zero.  A filled signal that has not reached TP before the fixed
    horizon is conservatively charged the full attached SL.  Same-S5 TP/SL is
    stop-first.  This scorer therefore cannot manufacture edge through a
    favorable intrabar or time-close assumption.
    """

    _validate_signal(signal)
    normalized_truth_hashes = [str(item) for item in truth_chunk_sha256]
    if not _truth_chunk_hashes_valid(normalized_truth_hashes):
        raise ValueError("valid truth chunk SHA-256 evidence is required")
    resolved = _aware_utc(resolved_at_utc)
    generated = _parse_utc(signal["generated_at_utc"])
    pip_factor = float(instrument_pip_factor(str(signal["pair"])))
    side = str(signal["side"])
    entry = float(signal["entry"])
    tp = float(signal["take_profit"])
    sl = float(signal["stop_loss"])
    tp_pips = float(signal["take_profit_pips"])
    sl_pips = float(signal["stop_loss_pips"])
    ttl = int(signal["entry_ttl_seconds"])
    hold = int(signal["max_hold_seconds"])
    primary = _resolve_entry_hypothesis(
        side=side,
        generated=generated,
        resolved=resolved,
        candles=candles,
        entry=entry,
        take_profit=tp,
        stop_loss=sl,
        take_profit_pips=tp_pips,
        stop_loss_pips=sl_pips,
        entry_ttl_seconds=ttl,
        max_hold_seconds=hold,
        pip_factor=pip_factor,
    )
    maturity = _parse_utc(primary["maturity_at_utc"])
    expected_chunk_count = math.ceil(
        _expected_truth_candle_count(generated, maturity)
        / TRUTH_CHUNK_CANDLE_LIMIT
    )
    if len(normalized_truth_hashes) != expected_chunk_count:
        raise ValueError("truth chunk hash count does not cover the exact request")
    experiment: dict[str, Any] | None = None
    if signal.get("schema_version") in {2, 3}:
        arm_results = []
        for arm in signal.get("entry_experiment_arms", []):
            arm_result = _resolve_entry_hypothesis(
                side=side,
                generated=generated,
                resolved=resolved,
                candles=candles,
                entry=float(arm["entry"]),
                take_profit=float(arm["take_profit"]),
                stop_loss=float(arm["stop_loss"]),
                take_profit_pips=float(arm["take_profit_pips"]),
                stop_loss_pips=float(arm["stop_loss_pips"]),
                entry_ttl_seconds=int(arm["entry_ttl_seconds"]),
                max_hold_seconds=int(arm["max_hold_seconds"]),
                pip_factor=pip_factor,
            )
            arm_results.append(
                {
                    "arm_id": str(arm["arm_id"]),
                    "spread_fraction_toward_market": float(
                        arm["spread_fraction_toward_market"]
                    ),
                    "effective_spread_fraction_toward_market": float(
                        arm["effective_spread_fraction_toward_market"]
                    ),
                    "entry": float(arm["entry"]),
                    "take_profit": float(arm["take_profit"]),
                    "stop_loss": float(arm["stop_loss"]),
                    **arm_result,
                }
            )
        experiment = {
            "contract": ENTRY_EXPERIMENT_CONTRACT,
            "precommitted_in_signal": True,
            "automatic_parameter_change_allowed": False,
            "arms": arm_results,
        }
    body = {
        "contract": OUTCOME_CONTRACT,
        "schema_version": int(signal["schema_version"]),
        "scoring_policy": SCORING_POLICY,
        "signal_id": str(signal["signal_id"]),
        "pair": str(signal["pair"]),
        "side": side,
        "method": str(signal["method"]),
        "signal_generated_at_utc": generated.isoformat(),
        "resolved_at_utc": resolved.isoformat(),
        "maturity_at_utc": maturity.isoformat(),
        "filled": primary["filled"],
        "fill_at_utc": primary["fill_at_utc"],
        "exit_at_utc": primary["exit_at_utc"],
        "exit_reason": primary["exit_reason"],
        "realized_pips": primary["realized_pips"],
        "ambiguous_same_s5": primary["ambiguous_same_s5"],
        "truth_source": "OANDA_S5_BID_ASK",
        "truth_request_from_utc": _ceil_s5(generated).isoformat(),
        "truth_request_to_utc": _floor_s5(maturity).isoformat(),
        "truth_request_coverage_proved": True,
        "truth_grid_slot_count": primary["truth_grid_slot_count"],
        "truth_candle_count": primary["truth_candle_count"],
        "truth_no_tick_slot_count": primary["truth_no_tick_slot_count"],
        "evaluated_candle_count": primary["evaluated_candle_count"],
        "evaluated_grid_slot_count": primary["evaluated_grid_slot_count"],
        "evaluated_no_tick_slot_count": primary[
            "evaluated_no_tick_slot_count"
        ],
        "truth_chunk_sha256": normalized_truth_hashes,
        "signal_sha256": str(signal["signal_sha256"]),
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation": False,
        **({"entry_experiment": experiment} if experiment is not None else {}),
    }
    return _seal(body)


def _resolve_entry_hypothesis(
    *,
    side: str,
    generated: datetime,
    resolved: datetime,
    candles: Sequence[S5BidAskCandle],
    entry: float,
    take_profit: float,
    stop_loss: float,
    take_profit_pips: float,
    stop_loss_pips: float,
    entry_ttl_seconds: int,
    max_hold_seconds: int,
    pip_factor: float,
) -> dict[str, Any]:
    """Score one precommitted entry geometry on the same frozen S5 path."""

    fill_deadline = generated + timedelta(seconds=entry_ttl_seconds)
    maturity = fill_deadline + timedelta(seconds=max_hold_seconds)
    if resolved < maturity:
        raise ValueError("fast-bot signal is not mature")
    ordered = sorted(candles, key=lambda item: item.timestamp_utc)
    if not _truth_grid_coverage_valid(
        ordered,
        generated=generated,
        maturity=maturity,
    ):
        raise ValueError("invalid fast-bot S5 truth grid coverage")
    fill_at: datetime | None = None
    exit_at: datetime | None = None
    exit_reason = "UNFILLED"
    realized_pips = 0.0
    ambiguous = False
    observed = 0
    for candle in ordered:
        if (
            fill_at is not None
            and candle.timestamp_utc
            >= fill_at + timedelta(seconds=max_hold_seconds)
        ):
            break
        observed += 1
        newly_filled = False
        if fill_at is None:
            filled = (
                candle.ask_l <= entry
                if side == "LONG"
                else candle.bid_h >= entry
            )
            if (
                not filled
                or candle.timestamp_utc + timedelta(seconds=5) > fill_deadline
            ):
                continue
            fill_at = candle.timestamp_utc
            newly_filled = True
        if side == "LONG":
            tp_hit = candle.bid_h >= take_profit
            sl_hit = candle.bid_l <= stop_loss
        else:
            tp_hit = candle.ask_l <= take_profit
            sl_hit = candle.ask_h >= stop_loss
        if newly_filled and (tp_hit or sl_hit):
            ambiguous = True
            realized_pips = _round_outcome_pips(
                _stop_loss_realized_pips(
                    side=side,
                    entry=entry,
                    stop_loss_pips=stop_loss_pips,
                    candle=candle,
                    pip_factor=pip_factor,
                    newly_filled=False,
                )
            )
            exit_reason = (
                "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5"
                if _is_executable_gap_pips(realized_pips, stop_loss_pips)
                else "STOP_LOSS_AMBIGUOUS_FILL_S5"
            )
            exit_at = candle.timestamp_utc
            break
        if tp_hit and sl_hit:
            ambiguous = True
            realized_pips = _round_outcome_pips(
                _stop_loss_realized_pips(
                    side=side,
                    entry=entry,
                    stop_loss_pips=stop_loss_pips,
                    candle=candle,
                    pip_factor=pip_factor,
                    newly_filled=newly_filled,
                )
            )
            exit_reason = (
                "STOP_LOSS_GAP_AMBIGUOUS_SAME_S5"
                if _is_executable_gap_pips(realized_pips, stop_loss_pips)
                else "STOP_LOSS_AMBIGUOUS_SAME_S5"
            )
            exit_at = candle.timestamp_utc
            break
        if sl_hit:
            realized_pips = _round_outcome_pips(
                _stop_loss_realized_pips(
                    side=side,
                    entry=entry,
                    stop_loss_pips=stop_loss_pips,
                    candle=candle,
                    pip_factor=pip_factor,
                    newly_filled=newly_filled,
                )
            )
            exit_reason = (
                "STOP_LOSS_GAP"
                if _is_executable_gap_pips(realized_pips, stop_loss_pips)
                else "STOP_LOSS"
            )
            exit_at = candle.timestamp_utc
            break
        if tp_hit:
            exit_reason = "TAKE_PROFIT"
            realized_pips = take_profit_pips
            exit_at = candle.timestamp_utc
            break
    if fill_at is not None and exit_at is None:
        exit_reason = "HORIZON_FULL_STOP_LOSS"
        realized_pips = -stop_loss_pips
        exit_at = min(fill_at + timedelta(seconds=max_hold_seconds), resolved)
    evaluated_grid_slot_count = _evaluated_grid_slot_count(
        generated=generated,
        maturity=maturity,
        exit_reason=exit_reason,
        exit_at=exit_at,
    )
    evaluated_no_tick_slot_count = evaluated_grid_slot_count - observed
    if evaluated_no_tick_slot_count < 0:
        raise ValueError("evaluated candle count exceeds its exact S5 prefix")
    return {
        "maturity_at_utc": maturity.isoformat(),
        "filled": fill_at is not None,
        "fill_at_utc": fill_at.isoformat() if fill_at else None,
        "exit_at_utc": exit_at.isoformat() if exit_at else None,
        "exit_reason": exit_reason,
        "realized_pips": _round_outcome_pips(realized_pips),
        "ambiguous_same_s5": ambiguous,
        "truth_grid_slot_count": _expected_truth_candle_count(generated, maturity),
        "truth_candle_count": len(ordered),
        "truth_no_tick_slot_count": (
            _expected_truth_candle_count(generated, maturity) - len(ordered)
        ),
        "evaluated_candle_count": observed,
        "evaluated_grid_slot_count": evaluated_grid_slot_count,
        "evaluated_no_tick_slot_count": evaluated_no_tick_slot_count,
    }


def _truth_grid_coverage_valid(
    candles: Sequence[S5BidAskCandle],
    *,
    generated: datetime,
    maturity: datetime,
) -> bool:
    first = _ceil_s5(generated)
    horizon_floor = _floor_s5(maturity)
    expected_count = _expected_truth_candle_count(generated, maturity)
    timestamps = [candle.timestamp_utc for candle in candles]
    return bool(
        expected_count > 0
        and len(timestamps) <= expected_count
        and len(set(timestamps)) == len(timestamps)
        and all(
            first <= timestamp < horizon_floor
            and _floor_s5(timestamp) == timestamp
            for timestamp in timestamps
        )
    )


def _expected_truth_candle_count(
    generated: datetime,
    maturity: datetime,
) -> int:
    first = _ceil_s5(generated)
    horizon_floor = _floor_s5(maturity)
    return max(0, int((horizon_floor - first).total_seconds() // 5))


def _evaluated_grid_slot_count(
    *,
    generated: datetime,
    maturity: datetime,
    exit_reason: str,
    exit_at: datetime | None,
) -> int:
    first = _ceil_s5(generated)
    if exit_reason == "UNFILLED":
        return _expected_truth_candle_count(generated, maturity)
    if exit_at is None or _floor_s5(exit_at) != exit_at:
        raise ValueError("filled outcome requires an aligned exit timestamp")
    if exit_reason == "HORIZON_FULL_STOP_LOSS":
        # The hold-cutoff candle is not evaluated.  This prefix is therefore
        # first-grid <= t < fill+hold.
        count = int((exit_at - first).total_seconds() // 5)
    else:
        # Price exits evaluate the exit candle itself.
        count = int((exit_at - first).total_seconds() // 5) + 1
    full_count = _expected_truth_candle_count(generated, maturity)
    if not 0 < count <= full_count:
        raise ValueError("evaluated S5 prefix is outside the truth grid")
    return count


def _ceil_s5(value: datetime) -> datetime:
    step_microseconds = 5_000_000
    epoch_microseconds = int(round(value.timestamp() * 1_000_000))
    aligned = (
        (epoch_microseconds + step_microseconds - 1) // step_microseconds
    ) * step_microseconds
    return datetime.fromtimestamp(aligned / 1_000_000, tz=timezone.utc)


def _floor_s5(value: datetime) -> datetime:
    step_microseconds = 5_000_000
    epoch_microseconds = int(round(value.timestamp() * 1_000_000))
    aligned = (epoch_microseconds // step_microseconds) * step_microseconds
    return datetime.fromtimestamp(aligned / 1_000_000, tz=timezone.utc)


def _stop_loss_realized_pips(
    *,
    side: str,
    entry: float,
    stop_loss_pips: float,
    candle: S5BidAskCandle,
    pip_factor: float,
    newly_filled: bool,
) -> float:
    if newly_filled:
        return -stop_loss_pips
    opening_exit_pips = (
        (candle.bid_o - entry) * pip_factor
        if side == "LONG"
        else (entry - candle.ask_o) * pip_factor
    )
    return min(-stop_loss_pips, opening_exit_pips)


def _round_outcome_pips(value: float) -> float:
    return round(float(value), OUTCOME_PIP_ROUND_DIGITS)


def _is_executable_gap_pips(realized_pips: float, stop_loss_pips: float) -> bool:
    """Require a serialized executable loss strictly beyond attached SL."""

    return _round_outcome_pips(realized_pips) < -_round_outcome_pips(
        stop_loss_pips
    )


def build_fast_bot_scorecard(
    signals: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: datetime,
) -> dict[str, Any]:
    valid_signal_rows = [item for item in signals if _fast_bot_signal_valid(item)]
    valid_signals = _dedupe_signal_identities(valid_signal_rows)
    valid_outcomes = [
        item
        for item in outcomes
        if _sealed_valid(item, OUTCOME_CONTRACT)
        and item.get("scoring_policy") == SCORING_POLICY
    ]
    emitted = valid_signals
    resolved: list[Mapping[str, Any]] = []
    resolved_by_signal_sha: dict[str, Mapping[str, Any]] = {}
    duplicate_valid_outcomes_ignored = 0
    for signal in emitted:
        signal_sha = str(signal["signal_sha256"])
        matches = [
            outcome
            for outcome in valid_outcomes
            if outcome.get("signal_sha256") == signal_sha
            and outcome.get("signal_id") == signal.get("signal_id")
            and _fast_bot_outcome_valid_for_signal(outcome, signal)
        ]
        if len(matches) == 1:
            resolved.append(matches[0])
            resolved_by_signal_sha[signal_sha] = matches[0]
        elif len(matches) > 1:
            duplicate_valid_outcomes_ignored += len(matches)
    fills = [item for item in resolved if item.get("filled") is True]
    values = [float(item["realized_pips"]) for item in fills]
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    active_days = {
        _parse_utc(item["signal_generated_at_utc"]).date().isoformat()
        for item in fills
        if isinstance(item.get("signal_generated_at_utc"), str)
    }
    mean = statistics.fmean(values) if values else None
    daily_values = _daily_means(
        (
            _parse_utc(item["signal_generated_at_utc"]).date().isoformat(),
            float(item["realized_pips"]),
        )
        for item in fills
    )
    lower = _one_sided_95_mean_lower(list(daily_values.values()))
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0.0
        else math.inf if gross_profit > 0.0 else None
    )
    passed = bool(
        len(fills) >= 100
        and len(active_days) >= 10
        and mean is not None
        and mean > 0.0
        and lower is not None
        and lower > 0.0
        and profit_factor is not None
        and profit_factor >= 1.25
    )
    entry_experiment = _build_entry_experiment_scorecard(
        emitted,
        resolved_by_signal_sha,
    )
    body = {
        "contract": SCORECARD_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": _aware_utc(as_of_utc).isoformat(),
        "status": "FORWARD_EVIDENCE_PASSED" if passed else "COLLECTING_FORWARD_EVIDENCE",
        "emitted_signals": len(emitted),
        "duplicate_identity_signals_ignored": len(valid_signal_rows) - len(valid_signals),
        "resolved_signals": len(resolved),
        "duplicate_valid_outcomes_ignored": duplicate_valid_outcomes_ignored,
        "filled_signals": len(fills),
        "unfilled_signals": sum(item.get("filled") is False for item in resolved),
        "active_days": len(active_days),
        "wins": len(wins),
        "losses": len(losses),
        "fill_rate": round(len(fills) / len(resolved), 6) if resolved else None,
        "win_rate": round(len(wins) / len(fills), 6) if fills else None,
        "net_pips": round(sum(values), 6),
        "mean_pips_per_fill": round(mean, 6) if mean is not None else None,
        "expectancy_inference_unit": "FILLED_SIGNAL_DAY",
        "daily_mean_observations": len(daily_values),
        "one_sided_95_daily_mean_lower_pips": (
            round(lower, 6)
            if lower is not None and math.isfinite(lower)
            else None
        ),
        "one_sided_95_mean_lower_pips": (
            round(lower, 6)
            if lower is not None and math.isfinite(lower)
            else None
        ),
        "profit_factor": round(profit_factor, 6) if profit_factor is not None and math.isfinite(profit_factor) else "INF" if profit_factor == math.inf else None,
        "forward_evidence_passed": passed,
        "live_permission": False,
        "promotion_allowed": False,
        "promotion_blockers": (
            [
                "HORIZON_AWARE_MULTI_GO_PORTFOLIO_FORWARD_PROOF_REQUIRED",
                "COUNTERFACTUAL_LEARNING_OUTCOME_PROOF_REQUIRED",
                "SEPARATE_CONTENT_ADDRESSED_LIVE_PROMOTION_REQUIRED",
            ]
            if passed
            else [
                "MINIMUM_100_EXACT_S5_FILLS_NOT_MET" if len(fills) < 100 else None,
                "MINIMUM_10_ACTIVE_DAYS_NOT_MET" if len(active_days) < 10 else None,
                "POST_COST_EXPECTANCY_LOWER_BOUND_NOT_POSITIVE" if lower is None or lower <= 0.0 else None,
                "PROFIT_FACTOR_1_25_NOT_MET" if profit_factor is None or profit_factor < 1.25 else None,
                "HORIZON_AWARE_MULTI_GO_PORTFOLIO_FORWARD_PROOF_REQUIRED",
                "COUNTERFACTUAL_LEARNING_OUTCOME_PROOF_REQUIRED",
                "SEPARATE_CONTENT_ADDRESSED_LIVE_PROMOTION_REQUIRED",
            ]
        ),
        "entry_experiment": entry_experiment,
        "shadow_only": True,
        "broker_mutation": False,
    }
    body["promotion_blockers"] = [item for item in body["promotion_blockers"] if item]
    return _seal(body)


def _build_entry_experiment_scorecard(
    signals: Sequence[Mapping[str, Any]],
    outcomes_by_signal_sha: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate paired, precommitted arms without changing primary metrics."""

    resolved_experiments: list[tuple[str, Mapping[str, Any]]] = []
    experiment_days: set[str] = set()
    for signal in signals:
        if signal.get("schema_version") not in {2, 3}:
            continue
        outcome = outcomes_by_signal_sha.get(str(signal.get("signal_sha256") or ""))
        experiment = outcome.get("entry_experiment") if isinstance(outcome, Mapping) else None
        if (
            not isinstance(experiment, Mapping)
            or experiment.get("contract") != ENTRY_EXPERIMENT_CONTRACT
            or experiment.get("precommitted_in_signal") is not True
            or not isinstance(experiment.get("arms"), list)
            or not _fast_bot_outcome_valid_for_signal(outcome, signal)
            or not _entry_experiment_outcome_matches_signal(signal, experiment)
        ):
            continue
        day = _parse_utc(signal["generated_at_utc"]).date().isoformat()
        resolved_experiments.append((day, experiment))
        experiment_days.add(day)

    arms: list[dict[str, Any]] = []
    for arm_id, fraction in ENTRY_ARM_SPREAD_FRACTIONS:
        rows: list[Mapping[str, Any]] = []
        paired_delta_rows: list[tuple[str, float]] = []
        distinct_from_primary = 0
        for day, experiment in resolved_experiments:
            matched = [
                row
                for row in experiment["arms"]
                if isinstance(row, Mapping) and row.get("arm_id") == arm_id
            ]
            if len(matched) == 1:
                rows.append(matched[0])
                primary = [
                    row
                    for row in experiment["arms"]
                    if isinstance(row, Mapping)
                    and row.get("arm_id") == ENTRY_ARM_SPREAD_FRACTIONS[0][0]
                ]
                if len(primary) == 1:
                    paired_delta_rows.append(
                        (
                            day,
                            float(matched[0].get("realized_pips") or 0.0)
                            - float(primary[0].get("realized_pips") or 0.0),
                        )
                    )
                    if matched[0].get("entry") != primary[0].get("entry"):
                        distinct_from_primary += 1
        filled = [row for row in rows if row.get("filled") is True]
        values_per_signal = [float(row.get("realized_pips") or 0.0) for row in rows]
        values_per_fill = [float(row.get("realized_pips") or 0.0) for row in filled]
        wins = [value for value in values_per_fill if value > 0.0]
        losses = [value for value in values_per_fill if value < 0.0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0.0
            else math.inf if gross_profit > 0.0 else None
        )
        paired_deltas = [value for _, value in paired_delta_rows]
        daily_deltas = _daily_means(paired_delta_rows)
        delta_lower = _one_sided_95_mean_lower(list(daily_deltas.values()))
        paired_count = len(paired_deltas)
        arms.append(
            {
                "arm_id": arm_id,
                "spread_fraction_toward_market": fraction,
                "paired_signals_vs_near_side": paired_count,
                "paired_active_days": len(daily_deltas),
                "distinct_from_near_side_signals": distinct_from_primary,
                "entry_collapse_rate_vs_near_side": (
                    round(1.0 - distinct_from_primary / paired_count, 6)
                    if paired_count
                    else None
                ),
                "resolved_signals": len(rows),
                "filled_signals": len(filled),
                "unfilled_signals": len(rows) - len(filled),
                "fill_rate": round(len(filled) / len(rows), 6) if rows else None,
                "wins": len(wins),
                "losses": len(losses),
                "net_pips": round(sum(values_per_signal), 6),
                "net_delta_pips_vs_near_side": round(sum(paired_deltas), 6),
                "mean_pips_per_signal": (
                    round(statistics.fmean(values_per_signal), 6)
                    if values_per_signal
                    else None
                ),
                "mean_pips_per_fill": (
                    round(statistics.fmean(values_per_fill), 6)
                    if values_per_fill
                    else None
                ),
                "mean_delta_pips_vs_near_side": (
                    round(statistics.fmean(paired_deltas), 6)
                    if paired_deltas
                    else None
                ),
                "one_sided_95_delta_lower_pips": (
                    round(delta_lower, 6)
                    if delta_lower is not None and math.isfinite(delta_lower)
                    else None
                ),
                "one_sided_95_daily_delta_lower_pips": (
                    round(delta_lower, 6)
                    if delta_lower is not None and math.isfinite(delta_lower)
                    else None
                ),
                "delta_inference_unit": "PAIRED_SIGNAL_DAY",
                "profit_factor": (
                    round(profit_factor, 6)
                    if profit_factor is not None and math.isfinite(profit_factor)
                    else "INF" if profit_factor == math.inf else None
                ),
            }
        )
    resolved_count = len(resolved_experiments)
    active_days = len(experiment_days)
    alternatives_have_distinct_ticks = all(
        arm["arm_id"] == ENTRY_ARM_SPREAD_FRACTIONS[0][0]
        or arm["distinct_from_near_side_signals"] >= 100
        for arm in arms
    )
    review_ready = bool(
        resolved_count >= 100
        and active_days >= 10
        and alternatives_have_distinct_ticks
    )
    return {
        "contract": ENTRY_EXPERIMENT_CONTRACT,
        "status": (
            "REVIEW_READY"
            if review_ready
            else "COLLECTING_PAIRED_FORWARD_EVIDENCE"
        ),
        "resolved_precommitted_signals": resolved_count,
        "active_days": active_days,
        "all_alternatives_have_100_distinct_ticks": alternatives_have_distinct_ticks,
        "review_ready": review_ready,
        "automatic_parameter_change_allowed": False,
        "selected_arm_id": None,
        "arms": arms,
    }


def _entry_experiment_outcome_matches_signal(
    signal: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> bool:
    """Bind every scored arm back to its immutable, pre-outcome geometry."""

    expected_arms = signal.get("entry_experiment_arms")
    observed_arms = experiment.get("arms")
    if (
        experiment.get("automatic_parameter_change_allowed") is not False
        or not isinstance(expected_arms, list)
        or not isinstance(observed_arms, list)
        or len(expected_arms) != len(observed_arms)
    ):
        return False
    try:
        generated = _parse_utc(signal["generated_at_utc"])
        ttl = int(signal["entry_ttl_seconds"])
        hold = int(signal["max_hold_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    maturity = generated + timedelta(seconds=ttl + hold)
    expected_truth_count = _expected_truth_candle_count(generated, maturity)
    for expected, observed in zip(expected_arms, observed_arms):
        if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
            return False
        for key in (
            "arm_id",
            "spread_fraction_toward_market",
            "effective_spread_fraction_toward_market",
            "entry",
            "take_profit",
            "stop_loss",
        ):
            if observed.get(key) != expected.get(key):
                return False
        try:
            realized_pips = float(observed["realized_pips"])
            arm_maturity = _parse_utc(observed["maturity_at_utc"])
            grid_slot_count = observed["truth_grid_slot_count"]
            truth_count = observed["truth_candle_count"]
            no_tick_count = observed["truth_no_tick_slot_count"]
            evaluated_count = observed["evaluated_candle_count"]
            evaluated_grid_count = observed["evaluated_grid_slot_count"]
            evaluated_no_tick_count = observed[
                "evaluated_no_tick_slot_count"
            ]
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if (
            not isinstance(observed.get("filled"), bool)
            or not math.isfinite(realized_pips)
            or not str(observed.get("exit_reason") or "")
            or not _outcome_result_semantics_valid(
                observed,
                take_profit_pips=float(expected["take_profit_pips"]),
                stop_loss_pips=float(expected["stop_loss_pips"]),
            )
            or arm_maturity != maturity
            or not isinstance(grid_slot_count, int)
            or isinstance(grid_slot_count, bool)
            or grid_slot_count != expected_truth_count
            or not isinstance(truth_count, int)
            or isinstance(truth_count, bool)
            or not 0 <= truth_count <= grid_slot_count
            or not isinstance(no_tick_count, int)
            or isinstance(no_tick_count, bool)
            or no_tick_count != grid_slot_count - truth_count
            or not isinstance(evaluated_count, int)
            or isinstance(evaluated_count, bool)
            or not 0 <= evaluated_count <= truth_count
            or not isinstance(evaluated_grid_count, int)
            or isinstance(evaluated_grid_count, bool)
            or not isinstance(evaluated_no_tick_count, int)
            or isinstance(evaluated_no_tick_count, bool)
            or not _outcome_timestamps_valid(
                observed,
                generated=generated,
                entry_ttl_seconds=ttl,
                max_hold_seconds=hold,
            )
            or not _evaluated_count_valid(
                observed,
                generated=generated,
                maturity=maturity,
            )
        ):
            return False
    return True


def _fast_bot_outcome_valid_for_signal(
    outcome: Mapping[str, Any],
    signal: Mapping[str, Any],
) -> bool:
    try:
        generated = _parse_utc(signal["generated_at_utc"])
        outcome_generated = _parse_utc(outcome["signal_generated_at_utc"])
        maturity = _parse_utc(outcome["maturity_at_utc"])
        request_from = _parse_utc(outcome["truth_request_from_utc"])
        request_to = _parse_utc(outcome["truth_request_to_utc"])
        grid_slot_count = outcome["truth_grid_slot_count"]
        truth_count = outcome["truth_candle_count"]
        no_tick_count = outcome["truth_no_tick_slot_count"]
        evaluated_count = outcome["evaluated_candle_count"]
        evaluated_grid_count = outcome["evaluated_grid_slot_count"]
        evaluated_no_tick_count = outcome[
            "evaluated_no_tick_slot_count"
        ]
        resolved = _parse_utc(outcome["resolved_at_utc"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    expected_maturity = generated + timedelta(
        seconds=int(signal["entry_ttl_seconds"]) + int(signal["max_hold_seconds"])
    )
    expected_schema = int(signal.get("schema_version") or 0)
    if (
        outcome.get("schema_version") != expected_schema
        or outcome.get("scoring_policy") != SCORING_POLICY
        or outcome.get("signal_sha256") != signal.get("signal_sha256")
        or outcome.get("signal_id") != signal.get("signal_id")
        or outcome.get("pair") != signal.get("pair")
        or outcome.get("side") != signal.get("side")
        or outcome.get("method") != signal.get("method")
        or outcome_generated != generated
        or maturity != expected_maturity
        or request_from != _ceil_s5(generated)
        or request_to != _floor_s5(maturity)
        or outcome.get("truth_request_coverage_proved") is not True
        or resolved < maturity
        or not isinstance(grid_slot_count, int)
        or isinstance(grid_slot_count, bool)
        or grid_slot_count != _expected_truth_candle_count(generated, maturity)
        or not isinstance(truth_count, int)
        or isinstance(truth_count, bool)
        or not 0 <= truth_count <= grid_slot_count
        or not isinstance(no_tick_count, int)
        or isinstance(no_tick_count, bool)
        or no_tick_count != grid_slot_count - truth_count
        or not isinstance(evaluated_count, int)
        or isinstance(evaluated_count, bool)
        or not 0 <= evaluated_count <= truth_count
        or not isinstance(evaluated_grid_count, int)
        or isinstance(evaluated_grid_count, bool)
        or not isinstance(evaluated_no_tick_count, int)
        or isinstance(evaluated_no_tick_count, bool)
        or outcome.get("truth_source") != "OANDA_S5_BID_ASK"
        or not _truth_chunk_hashes_valid(outcome.get("truth_chunk_sha256"))
        or len(outcome["truth_chunk_sha256"])
        != math.ceil(grid_slot_count / TRUTH_CHUNK_CANDLE_LIMIT)
        or outcome.get("shadow_only") is not True
        or outcome.get("live_permission") is not False
        or outcome.get("broker_mutation") is not False
        or not _outcome_result_semantics_valid(
            outcome,
            take_profit_pips=float(signal["take_profit_pips"]),
            stop_loss_pips=float(signal["stop_loss_pips"]),
        )
        or not _outcome_timestamps_valid(
            outcome,
            generated=generated,
            entry_ttl_seconds=int(signal["entry_ttl_seconds"]),
            max_hold_seconds=int(signal["max_hold_seconds"]),
        )
        or not _evaluated_count_valid(
            outcome,
            generated=generated,
            maturity=maturity,
        )
    ):
        return False
    experiment = outcome.get("entry_experiment")
    if expected_schema == 1:
        return experiment is None
    if not bool(
        isinstance(experiment, Mapping)
        and experiment.get("contract") == ENTRY_EXPERIMENT_CONTRACT
        and experiment.get("precommitted_in_signal") is True
        and _entry_experiment_outcome_matches_signal(signal, experiment)
    ):
        return False
    primary_arms = [
        row
        for row in experiment["arms"]
        if isinstance(row, Mapping)
        and row.get("arm_id") == ENTRY_ARM_SPREAD_FRACTIONS[0][0]
    ]
    if len(primary_arms) != 1:
        return False
    primary = primary_arms[0]
    return all(
        outcome.get(key) == primary.get(key)
        for key in (
            "filled",
            "fill_at_utc",
            "exit_at_utc",
            "exit_reason",
            "realized_pips",
            "ambiguous_same_s5",
            "truth_grid_slot_count",
            "truth_candle_count",
            "truth_no_tick_slot_count",
            "evaluated_candle_count",
            "evaluated_grid_slot_count",
            "evaluated_no_tick_slot_count",
            "maturity_at_utc",
        )
    )


def _outcome_result_semantics_valid(
    result: Mapping[str, Any],
    *,
    take_profit_pips: float,
    stop_loss_pips: float,
) -> bool:
    try:
        realized_pips = float(result["realized_pips"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    filled = result.get("filled")
    reason = str(result.get("exit_reason") or "")
    fill_at = result.get("fill_at_utc")
    exit_at = result.get("exit_at_utc")
    ambiguous = result.get("ambiguous_same_s5")
    if not isinstance(ambiguous, bool):
        return False
    if filled is False:
        return bool(
            reason == "UNFILLED"
            and realized_pips == 0.0
            and fill_at is None
            and exit_at is None
            and ambiguous is False
        )
    if filled is not True or not isinstance(fill_at, str) or not isinstance(exit_at, str):
        return False
    if reason == "TAKE_PROFIT":
        return ambiguous is False and math.isclose(
            realized_pips,
            take_profit_pips,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    if reason in {
        "STOP_LOSS",
        "HORIZON_FULL_STOP_LOSS",
    }:
        return ambiguous is False and math.isclose(
            realized_pips,
            -stop_loss_pips,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    if reason in {
        "STOP_LOSS_AMBIGUOUS_FILL_S5",
        "STOP_LOSS_AMBIGUOUS_SAME_S5",
    }:
        return ambiguous is True and math.isclose(
            realized_pips,
            -stop_loss_pips,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    if reason in {
        "STOP_LOSS_GAP",
        "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5",
        "STOP_LOSS_GAP_AMBIGUOUS_SAME_S5",
    }:
        return (
            ambiguous is (reason != "STOP_LOSS_GAP")
            and _is_executable_gap_pips(realized_pips, stop_loss_pips)
        )
    return False


def _outcome_timestamps_valid(
    result: Mapping[str, Any],
    *,
    generated: datetime,
    entry_ttl_seconds: int,
    max_hold_seconds: int,
) -> bool:
    if result.get("filled") is False:
        return result.get("fill_at_utc") is None and result.get("exit_at_utc") is None
    try:
        fill_at = _parse_utc(result["fill_at_utc"])
        exit_at = _parse_utc(result["exit_at_utc"])
    except (KeyError, TypeError, ValueError):
        return False
    fill_deadline = generated + timedelta(seconds=entry_ttl_seconds)
    maturity = fill_deadline + timedelta(seconds=max_hold_seconds)
    if not (
        generated <= fill_at
        and _floor_s5(fill_at) == fill_at
        and _floor_s5(exit_at) == exit_at
        and fill_at + timedelta(seconds=5) <= fill_deadline
        and fill_at <= exit_at <= maturity
    ):
        return False
    reason = str(result.get("exit_reason") or "")
    if reason == "HORIZON_FULL_STOP_LOSS":
        return exit_at == fill_at + timedelta(seconds=max_hold_seconds)
    return exit_at < min(
        fill_at + timedelta(seconds=max_hold_seconds),
        maturity,
    )


def _evaluated_count_valid(
    result: Mapping[str, Any],
    *,
    generated: datetime,
    maturity: datetime,
) -> bool:
    try:
        observed = result["evaluated_candle_count"]
        truth_count = result["truth_candle_count"]
        truth_no_tick_count = result["truth_no_tick_slot_count"]
        evaluated_grid_count = result["evaluated_grid_slot_count"]
        evaluated_no_tick_count = result[
            "evaluated_no_tick_slot_count"
        ]
    except KeyError:
        return False
    if (
        not isinstance(observed, int)
        or isinstance(observed, bool)
        or not isinstance(truth_count, int)
        or isinstance(truth_count, bool)
        or not isinstance(truth_no_tick_count, int)
        or isinstance(truth_no_tick_count, bool)
        or not isinstance(evaluated_grid_count, int)
        or isinstance(evaluated_grid_count, bool)
        or not isinstance(evaluated_no_tick_count, int)
        or isinstance(evaluated_no_tick_count, bool)
    ):
        return False
    reason = str(result.get("exit_reason") or "")
    try:
        exit_at = (
            _parse_utc(result["exit_at_utc"])
            if reason != "UNFILLED"
            else None
        )
        expected_grid_count = _evaluated_grid_slot_count(
            generated=generated,
            maturity=maturity,
            exit_reason=reason,
            exit_at=exit_at,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        evaluated_grid_count == expected_grid_count
        and 0 <= evaluated_no_tick_count <= truth_no_tick_count
        and evaluated_grid_count - evaluated_no_tick_count == observed
        and 0 <= observed <= truth_count
        and (
            observed == truth_count
            and evaluated_no_tick_count == truth_no_tick_count
            if reason == "UNFILLED"
            else observed > 0
        )
    )


def _truth_chunk_hashes_valid(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(_sha256_text(item) for item in value)
    )


def resolve_due_fast_bot_outcomes_from_oanda(
    *,
    shadow_ledger_path: Path,
    outcome_ledger_path: Path,
    scorecard_path: Path,
    client_factory: Callable[[], Any] = OandaReadOnlyClient,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    now = _aware_utc((clock or (lambda: datetime.now(timezone.utc)))())
    loaded_signals = _load_jsonl(shadow_ledger_path)
    valid_loaded_signals = [
        item for item in loaded_signals if _fast_bot_signal_valid(item)
    ]
    signals = _dedupe_signal_identities(valid_loaded_signals)
    outcomes = _load_jsonl(outcome_ledger_path)
    current_rows_by_identity: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = {}
    for outcome in outcomes:
        identity = _current_outcome_identity(outcome)
        if identity is not None:
            current_rows_by_identity.setdefault(identity, []).append(outcome)
    resolved_identities: set[tuple[str, str]] = set()
    conflict_identities: set[tuple[str, str]] = set()
    outcome_identity_conflicts: list[dict[str, Any]] = []
    for signal in signals:
        identity = _signal_outcome_identity(signal)
        current_rows = current_rows_by_identity.get(identity, [])
        valid_rows = [
            outcome
            for outcome in current_rows
            if (
            _sealed_valid(outcome, OUTCOME_CONTRACT)
            and outcome.get("scoring_policy") == SCORING_POLICY
            and _fast_bot_outcome_valid_for_signal(outcome, signal)
            )
        ]
        if len(valid_rows) == 1:
            resolved_identities.add(identity)
        elif current_rows:
            conflict_identities.add(identity)
            outcome_identity_conflicts.append(
                {
                    "signal_id": str(signal.get("signal_id") or ""),
                    "signal_sha256": str(signal.get("signal_sha256") or ""),
                    "pair": str(signal.get("pair") or ""),
                    "current_policy_row_count": len(current_rows),
                    "valid_current_policy_row_count": len(valid_rows),
                    "reason": (
                        "CURRENT_POLICY_OUTCOME_IDENTITY_HAS_NO_VALID_ROW"
                        if not valid_rows
                        else "CURRENT_POLICY_OUTCOME_IDENTITY_HAS_MULTIPLE_VALID_ROWS"
                    ),
                }
            )
    due = []
    for signal in signals:
        identity = _signal_outcome_identity(signal)
        if (
            not _fast_bot_signal_valid(signal)
            or identity in resolved_identities
            or identity in conflict_identities
        ):
            continue
        generated = _parse_utc(signal["generated_at_utc"])
        maturity = generated + timedelta(
            seconds=int(signal["entry_ttl_seconds"]) + int(signal["max_hold_seconds"])
        )
        if maturity <= now:
            due.append((maturity, signal))
    due.sort(key=lambda item: (item[0], str(item[1].get("signal_id"))))
    selection_offset, selected_rows = _rotating_due_selection(due, now=now)
    selected = [item[1] for item in selected_rows]
    base = {
        "contract": TRUTH_ADAPTER_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation": False,
        "due_count": len(due),
        "selected_due_count": len(selected),
        "selection_offset": selection_offset,
        "duplicate_identity_signals_ignored": (
            len(valid_loaded_signals) - len(signals)
        ),
        "outcome_identity_conflict_count": len(outcome_identity_conflicts),
        "outcome_identity_conflicts": outcome_identity_conflicts,
    }
    if not selected:
        scorecard = build_fast_bot_scorecard(loaded_signals, outcomes, as_of_utc=now)
        _write_json_atomic(scorecard_path, scorecard)
        return {
            **base,
            "status": (
                "OUTCOME_IDENTITY_CONFLICT"
                if outcome_identity_conflicts
                else "NO_DUE_SIGNALS"
            ),
            "broker_read": False,
            "ledger_appended": 0,
            "ledger_duplicate_outcome_identities_skipped": 0,
            "scorecard_status": scorecard["status"],
        }

    client = client_factory()
    resolved: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def resolve(signal: Mapping[str, Any]) -> dict[str, Any]:
        generated = _parse_utc(signal["generated_at_utc"])
        maturity = generated + timedelta(
            seconds=int(signal["entry_ttl_seconds"]) + int(signal["max_hold_seconds"])
        )
        candles, hashes = fetch_frozen_s5_truth(
            client,
            pair=str(signal["pair"]),
            time_from=generated,
            time_to=maturity,
            chunk_candle_limit=TRUTH_CHUNK_CANDLE_LIMIT,
        )
        return resolve_fast_bot_signal(signal, candles, resolved_at_utc=now, truth_chunk_sha256=hashes)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(resolve, signal): signal for signal in selected}
        for future in concurrent.futures.as_completed(futures):
            signal = futures[future]
            try:
                resolved.append(future.result())
            except Exception as exc:  # pragma: no cover - network boundary
                errors.append({
                    "signal_id": str(signal.get("signal_id") or ""),
                    "pair": str(signal.get("pair") or ""),
                    "error": f"{type(exc).__name__}: {exc}"[:320],
                })
    append_result = _append_outcomes_once(outcome_ledger_path, resolved)
    all_outcomes = _load_jsonl(outcome_ledger_path)
    scorecard = build_fast_bot_scorecard(
        loaded_signals,
        all_outcomes,
        as_of_utc=now,
    )
    _write_json_atomic(scorecard_path, scorecard)
    return {
        **base,
        "status": (
            "RESOLVED_WITH_ERRORS"
            if (
                errors
                or outcome_identity_conflicts
                or append_result["duplicate_identity_count"]
            )
            else "RESOLVED"
        ),
        "broker_read": True,
        "ledger_appended": append_result["appended"],
        "ledger_duplicate_outcome_identities_skipped": append_result[
            "duplicate_identity_count"
        ],
        "ledger_duplicate_outcome_signal_ids": append_result[
            "duplicate_identity_signal_ids"
        ],
        "errors": errors,
        "scorecard_status": scorecard["status"],
        "forward_evidence_passed": scorecard["forward_evidence_passed"],
    }


def _rotating_due_selection(
    due: Sequence[tuple[datetime, Mapping[str, Any]]],
    *,
    now: datetime,
) -> tuple[int, list[tuple[datetime, Mapping[str, Any]]]]:
    """Bound each read cycle without starving later permanently failing rows."""

    if len(due) <= MAX_DUE_PER_RUN:
        return 0, list(due)
    offset = ((int(now.timestamp()) // 30) * MAX_DUE_PER_RUN) % len(due)
    rotated = [*due[offset:], *due[:offset]]
    return offset, rotated[:MAX_DUE_PER_RUN]


def _validate_signal(signal: Mapping[str, Any]) -> None:
    if not _fast_bot_signal_valid(signal):
        raise ValueError("invalid fast-bot signal")


def _fast_bot_signal_valid(signal: Mapping[str, Any]) -> bool:
    try:
        signal_body = {key: item for key, item in signal.items() if key != "signal_sha256"}
        signal_sha = str(signal["signal_sha256"])
        regime_sha = str(signal["regime_contract_sha256"])
        raw_schema = signal["schema_version"]
        if isinstance(raw_schema, bool) or not isinstance(raw_schema, int):
            return False
        schema = raw_schema
        signal_id = str(signal["signal_id"])
        pair = str(signal["pair"])
        side = str(signal.get("side") or "")
        method = str(signal["method"])
        entry = float(signal["entry"])
        tp = float(signal["take_profit"])
        sl = float(signal["stop_loss"])
        tp_pips = float(signal["take_profit_pips"])
        sl_pips = float(signal["stop_loss_pips"])
        reward_risk = float(signal["reward_risk"])
        generated = _parse_utc(signal["generated_at_utc"])
        quote_at = _parse_utc(signal["quote_timestamp_utc"])
        m1_closed = _parse_utc(signal["m1_closed_candle_utc"])
        ttl = int(signal["entry_ttl_seconds"])
        hold = int(signal["max_hold_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    geometry_ok = (
        tp > entry > sl if side == "LONG" else sl > entry > tp if side == "SHORT" else False
    )
    digest_ok = (
        _sha256_text(signal_sha)
        and _sha256_text(regime_sha)
        and signal_sha == _canonical_sha(signal_body)
    )
    quote_age_seconds = (generated - quote_at).total_seconds()
    timing_ok = (
        0.0 <= quote_age_seconds <= 45.0
        and 0.0 <= (generated - m1_closed).total_seconds() <= 120.0
    )
    common_valid = bool(
        signal.get("contract") == SIGNAL_CONTRACT
        and schema in {1, 2, 3}
        and digest_ok
        and len(signal_id) == 24
        and all(character in "0123456789abcdef" for character in signal_id)
        and pair == pair.upper()
        and "_" in pair
        and method in METHODS
        and signal.get("shadow_only") is True
        and signal.get("live_permission") is False
        and signal.get("broker_mutation_allowed") is False
        and str(signal.get("order_type") or "") == "LIMIT"
        and str(signal.get("entry_reference") or "") == "PASSIVE_NEAR_SIDE"
        and signal.get("attached_take_profit_required") is True
        and signal.get("attached_stop_loss_required") is True
        and geometry_ok
        and timing_ok
        and tp_pips > 0.0
        and sl_pips > 0.0
        and math.isclose(
            reward_risk,
            tp_pips / sl_pips,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and ttl == ENTRY_TTL_SECONDS
        and hold == MAX_HOLD_SECONDS
    )
    if not common_valid:
        return False
    if schema == 1:
        return True
    if not _fast_bot_v2_geometry_valid(
        signal,
        pair=pair,
        side=side,
        method=method,
        entry=entry,
        take_profit=tp,
        stop_loss=sl,
        take_profit_pips=tp_pips,
        stop_loss_pips=sl_pips,
    ):
        return False
    return schema == 2 or _fast_bot_v3_identity_valid(signal)


def _fast_bot_v3_identity_valid(signal: Mapping[str, Any]) -> bool:
    """Bind every V3 side/method seat to its canonical horizon identity."""

    identity = {
        "identity_contract": SIGNAL_IDENTITY_CONTRACT_V3,
        "pair": str(signal.get("pair") or ""),
        "m1_closed_candle_utc": str(
            signal.get("m1_closed_candle_utc") or ""
        ),
        "side": str(signal.get("side") or ""),
        "method": str(signal.get("method") or ""),
        "horizon_lane": HORIZON_LANE,
    }
    return bool(
        signal.get("identity_contract") == SIGNAL_IDENTITY_CONTRACT_V3
        and signal.get("horizon_lane") == HORIZON_LANE
        and signal.get("signal_id") == _canonical_sha(identity)[:24]
    )


def _fast_bot_v2_geometry_valid(
    signal: Mapping[str, Any],
    *,
    pair: str,
    side: str,
    method: str,
    entry: float,
    take_profit: float,
    stop_loss: float,
    take_profit_pips: float,
    stop_loss_pips: float,
) -> bool:
    """Require the shared V2/V3 quote geometry and arms to reproduce exactly."""

    try:
        bid = float(signal["quote_bid"])
        ask = float(signal["quote_ask"])
        spread = float(signal["spread_pips"])
        m5_atr = float(signal["m5_atr_pips"])
        arms = signal["entry_experiment_arms"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if (
        not all(math.isfinite(value) and value > 0.0 for value in (bid, ask, spread, m5_atr))
        or ask <= bid
        or bid != _rounded_price(pair, bid)
        or ask != _rounded_price(pair, ask)
        or signal.get("geometry_policy") != "METHOD_SPREAD_M5_ATR_V1"
        or signal.get("entry_experiment_contract") != ENTRY_EXPERIMENT_CONTRACT
        or not isinstance(arms, list)
    ):
        return False
    observed_spread = round(
        (ask - bid) * float(instrument_pip_factor(pair)),
        6,
    )
    expected_tp_pips, expected_sl_pips = _shadow_geometry_pips(
        method,
        spread=spread,
        m5_atr=m5_atr,
    )
    expected_arms = _entry_experiment_arms(
        pair=pair,
        side=side,
        bid=bid,
        ask=ask,
        tp_pips=expected_tp_pips,
        sl_pips=expected_sl_pips,
    )
    primary = expected_arms[0]
    return bool(
        math.isclose(spread, observed_spread, rel_tol=0.0, abs_tol=1e-6)
        and take_profit_pips == float(primary["take_profit_pips"])
        and stop_loss_pips == float(primary["stop_loss_pips"])
        and entry == float(primary["entry"])
        and take_profit == float(primary["take_profit"])
        and stop_loss == float(primary["stop_loss"])
        and arms == expected_arms
    )


def _rounded_price(pair: str, value: float) -> float:
    return round(value, 3 if pair.endswith("_JPY") else 5)


def _daily_means(rows: Iterable[tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for day, value in rows:
        grouped.setdefault(day, []).append(value)
    return {
        day: statistics.fmean(values)
        for day, values in grouped.items()
    }


def _one_sided_95_mean_lower(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return -math.inf
    stdev = statistics.stdev(values)
    if stdev == 0.0:
        return mean
    critical = _student_t_one_sided_95(len(values) - 1)
    return mean - critical * stdev / math.sqrt(len(values))


def _dedupe_signal_identities(
    signals: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Preserve legacy M1 seats while keeping every distinct V3 GO seat."""

    seen_shas: set[str] = set()
    seen_identities: set[tuple[str, ...]] = set()
    out: list[Mapping[str, Any]] = []
    for signal in signals:
        signal_sha = str(signal.get("signal_sha256") or "")
        if signal_sha in seen_shas:
            continue
        if signal.get("schema_version") == 3:
            identity = (
                "V3",
                str(signal.get("pair") or ""),
                str(signal.get("m1_closed_candle_utc") or ""),
                str(signal.get("side") or ""),
                str(signal.get("method") or ""),
                str(signal.get("horizon_lane") or ""),
            )
        else:
            identity = (
                "LEGACY",
                str(signal.get("pair") or ""),
                str(signal.get("m1_closed_candle_utc") or ""),
            )
        if identity in seen_identities:
            continue
        seen_shas.add(signal_sha)
        seen_identities.add(identity)
        out.append(signal)
    return out


def _student_t_one_sided_95(df: int) -> float:
    table = {
        1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
        6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
        12: 1.782, 15: 1.753, 20: 1.725, 25: 1.708, 30: 1.697,
        40: 1.684, 60: 1.671, 120: 1.658,
    }
    for bound in sorted(table):
        if df <= bound:
            return table[bound]
    return 1.645


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid JSONL row at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"non-object JSONL row at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _signal_outcome_identity(signal: Mapping[str, Any]) -> tuple[str, str]:
    return (str(signal.get("signal_sha256") or ""), SCORING_POLICY)


def _current_outcome_identity(
    outcome: Mapping[str, Any],
) -> tuple[str, str] | None:
    signal_sha = str(outcome.get("signal_sha256") or "")
    policy = str(outcome.get("scoring_policy") or "")
    if (
        outcome.get("contract") != OUTCOME_CONTRACT
        or policy != SCORING_POLICY
        or not _sha256_text(signal_sha)
    ):
        return None
    return (signal_sha, policy)


def _append_outcomes_once(
    path: Path,
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not outcomes:
        return {
            "appended": 0,
            "duplicate_identity_count": 0,
            "duplicate_identity_signal_ids": [],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen_digests: set[str] = set()
        seen_identities: set[tuple[str, str]] = set()
        for line_number, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid JSONL row at {path}:{line_number}"
                ) from exc
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"non-object JSONL row at {path}:{line_number}"
                )
            if item.get("contract_sha256"):
                seen_digests.add(str(item["contract_sha256"]))
            identity = _current_outcome_identity(item)
            if identity is not None:
                # Identity is intentionally retained even when the historical
                # row fails today's deep validator.  Appending a new sealed
                # row every 30 seconds cannot repair immutable history; it
                # only hides the conflict and burns broker reads.
                seen_identities.add(identity)
        handle.seek(0, os.SEEK_END)
        appended = 0
        duplicate_identity_signal_ids: list[str] = []
        for outcome in outcomes:
            signal_id = str(outcome.get("signal_id") or "")
            outcome_sha = str(outcome.get("contract_sha256") or "")
            identity = _current_outcome_identity(outcome)
            if (
                not signal_id
                or not outcome_sha
                or outcome_sha in seen_digests
                or outcome.get("scoring_policy") != SCORING_POLICY
                or not _sealed_valid(outcome, OUTCOME_CONTRACT)
            ):
                continue
            if identity is None:
                continue
            if identity in seen_identities:
                duplicate_identity_signal_ids.append(signal_id)
                continue
            handle.write(json.dumps(dict(outcome), ensure_ascii=False, sort_keys=True) + "\n")
            seen_digests.add(outcome_sha)
            seen_identities.add(identity)
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {
        "appended": appended,
        "duplicate_identity_count": len(duplicate_identity_signal_ids),
        "duplicate_identity_signal_ids": sorted(
            set(duplicate_identity_signal_ids)
        ),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _canonical_sha(body)}


def _sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != contract:
        return False
    stored = str(value.get("contract_sha256") or "")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return bool(stored and stored == _canonical_sha(body))


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _parse_utc(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be aware UTC") from exc
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "OUTCOME_CONTRACT",
    "SCORING_POLICY",
    "SCORECARD_CONTRACT",
    "build_fast_bot_scorecard",
    "resolve_due_fast_bot_outcomes_from_oanda",
    "resolve_fast_bot_signal",
]

"""Exact, read-only truth for permanent fast-bot counterfactual seats.

One learning seat is one frozen market path.  Every pair/side/method candidate
and every precommitted OFAT arm in that seat is evaluated against that exact
same OANDA S5 bid/ask response.  This module cannot emit an order, change the
primary scorecard, promote an arm, or mutate broker state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.fast_bot_learning import (
    CANDIDATE_CLASSES,
    CELL_ORDER,
    LEARNING_ARM_SPECS,
    LEARNING_SEAT_CONTRACT,
    METHODS,
    SIDES,
    _learning_seat_valid,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle
from quant_rabbit.technical_forecast_forward_truth import fetch_frozen_s5_truth


OUTCOME_CONTRACT = "QR_FAST_BOT_LEARNING_S5_BID_ASK_OUTCOME_V1"
CANDIDATE_OUTCOME_CONTRACT = "QR_FAST_BOT_LEARNING_CANDIDATE_OUTCOME_V1"
ARM_OUTCOME_CONTRACT = "QR_FAST_BOT_LEARNING_ARM_OUTCOME_V1"
SCORECARD_CONTRACT = "QR_FAST_BOT_LEARNING_SCORECARD_V1"
TRUTH_ADAPTER_CONTRACT = "QR_FAST_BOT_LEARNING_TRUTH_ADAPTER_V1"
SCORING_POLICY = "QR_FAST_BOT_LEARNING_SPARSE_S5_CONSERVATIVE_V1"
TRUTH_CHUNK_CANDLE_LIMIT = 4500
MAX_DUE_PER_RUN = 12
ROUND_DIGITS = 6


def resolve_fast_bot_learning_seat(
    seat: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    resolved_at_utc: datetime,
    truth_chunk_sha256: Sequence[str],
) -> dict[str, Any]:
    """Score every arm on one immutable seat-level truth path."""

    _validate_learning_seat(seat)
    resolved = _aware_utc(resolved_at_utc)
    generated = _parse_utc(seat["generated_at_utc"])
    max_maturity = _seat_maturity(seat)
    if resolved < max_maturity:
        raise ValueError("learning seat is not mature")
    ordered = sorted(candles, key=lambda item: item.timestamp_utc)
    _validate_truth_path(ordered, generated=generated, maturity=max_maturity)
    hashes = [str(item) for item in truth_chunk_sha256]
    if not hashes or not all(_sha256_text(item) for item in hashes):
        raise ValueError("valid truth chunk hashes are required")
    expected_chunks = math.ceil(
        _grid_slot_count(generated, max_maturity) / TRUTH_CHUNK_CANDLE_LIMIT
    )
    if len(hashes) != expected_chunks:
        raise ValueError("truth chunk hashes do not cover the seat path")
    path_sha = _truth_path_sha(
        pair=str(seat["pair"]),
        generated=generated,
        maturity=max_maturity,
        candles=ordered,
        chunk_hashes=hashes,
    )
    pip_factor = float(instrument_pip_factor(str(seat["pair"])))
    candidate_results: list[dict[str, Any]] = []
    for candidate in seat["candidates"]:
        arm_results = []
        for arm in candidate["arms"]:
            arm_results.append(
                _resolve_arm(
                    arm,
                    side=str(candidate["side"]),
                    generated=generated,
                    candles=ordered,
                    pip_factor=pip_factor,
                    truth_path_sha256=path_sha,
                    truth_chunk_sha256=hashes,
                )
            )
        candidate_body = {
            "contract": CANDIDATE_OUTCOME_CONTRACT,
            "schema_version": 1,
            "candidate_id": str(candidate["candidate_id"]),
            "candidate_sha256": str(candidate["candidate_sha256"]),
            "candidate_class": str(candidate["candidate_class"]),
            "comparison_role": str(candidate["comparison_role"]),
            "pair": str(seat["pair"]),
            "side": str(candidate["side"]),
            "method": str(candidate["method"]),
            "horizon_lane": str(candidate["horizon_lane"]),
            "counterfactual_comparison_group_id": str(
                candidate["counterfactual_comparison_group_id"]
            ),
            "truth_path_sha256": path_sha,
            "arms": arm_results,
            "automatic_promotion_allowed": False,
            "primary_effect": False,
            "risk_effect": False,
            "order_authority": "NONE",
            "shadow_only": True,
            "live_permission": False,
            "broker_mutation_allowed": False,
        }
        candidate_results.append(_seal_named(candidate_body, "candidate_outcome_sha256"))
    body = {
        "contract": OUTCOME_CONTRACT,
        "schema_version": 1,
        "scoring_policy": SCORING_POLICY,
        "seat_id": str(seat["seat_id"]),
        "seat_contract_sha256": str(seat["contract_sha256"]),
        "pair": str(seat["pair"]),
        "sampling_bucket_utc": str(seat["sampling_bucket_utc"]),
        "seat_generated_at_utc": generated.isoformat(),
        "resolved_at_utc": resolved.isoformat(),
        "maturity_at_utc": max_maturity.isoformat(),
        "candidate_count": len(candidate_results),
        "candidate_class_cost_context": dict(seat["cost_context"]),
        "truth_source": "OANDA_S5_BID_ASK",
        "truth_request_from_utc": _ceil_s5(generated).isoformat(),
        "truth_request_to_utc": _floor_s5(max_maturity).isoformat(),
        "truth_request_coverage_proved": True,
        "truth_grid_slot_count": _grid_slot_count(generated, max_maturity),
        "truth_candle_count": len(ordered),
        "truth_no_tick_slot_count": (
            _grid_slot_count(generated, max_maturity) - len(ordered)
        ),
        "truth_chunk_sha256": hashes,
        "truth_path_sha256": path_sha,
        "candidates": candidate_results,
        "pair_direction_netting_allowed": False,
        "automatic_promotion_allowed": False,
        "primary_effect": False,
        "risk_effect": False,
        "order_authority": "NONE",
        "diagnostic_only": True,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return _seal(body)


def _resolve_arm(
    arm: Mapping[str, Any],
    *,
    side: str,
    generated: datetime,
    candles: Sequence[S5BidAskCandle],
    pip_factor: float,
    truth_path_sha256: str,
    truth_chunk_sha256: Sequence[str],
) -> dict[str, Any]:
    ttl = int(arm["entry_ttl_seconds"])
    hold = int(arm["max_hold_seconds"])
    fill_deadline = generated + timedelta(seconds=ttl)
    maturity = fill_deadline + timedelta(seconds=hold)
    path = [item for item in candles if item.timestamp_utc < _floor_s5(maturity)]
    entry = float(arm["entry"])
    take_profit = float(arm["take_profit"])
    stop_loss = float(arm["stop_loss"])
    tp_pips = float(arm["take_profit_pips"])
    sl_pips = float(arm["stop_loss_pips"])
    fill_at: datetime | None = None
    exit_at: datetime | None = None
    exit_reason = "UNFILLED"
    realized = 0.0
    ambiguous = False
    mfe: float | None = None
    mae: float | None = None
    evaluated_candles = 0
    for candle in path:
        if fill_at is not None and candle.timestamp_utc >= fill_at + timedelta(seconds=hold):
            break
        evaluated_candles += 1
        newly_filled = False
        if fill_at is None:
            touched = candle.ask_l <= entry if side == "LONG" else candle.bid_h >= entry
            if not touched or candle.timestamp_utc + timedelta(seconds=5) > fill_deadline:
                continue
            fill_at = candle.timestamp_utc
            newly_filled = True
            mfe = 0.0
            mae = _adverse_excursion(side, entry, candle, pip_factor)
        tp_hit, sl_hit = _exit_hits(
            side,
            candle,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )
        if newly_filled:
            # Intrabar ordering relative to the passive fill is unknowable.
            # Never credit favorable fill-candle movement; any attached exit
            # touched in that candle is charged stop-first.
            if tp_hit or sl_hit:
                ambiguous = True
                realized = _stop_realized(
                    side=side,
                    entry=entry,
                    stop_loss_pips=sl_pips,
                    candle=candle,
                    pip_factor=pip_factor,
                    gap_allowed=True,
                )
                exit_reason = (
                    "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5"
                    if realized < -_round(sl_pips)
                    else "STOP_LOSS_AMBIGUOUS_FILL_S5"
                )
                exit_at = candle.timestamp_utc
                break
            continue
        adverse = _adverse_excursion(side, entry, candle, pip_factor)
        favorable = _favorable_excursion(side, entry, candle, pip_factor)
        mae = max(float(mae or 0.0), adverse)
        if tp_hit and sl_hit:
            ambiguous = True
            realized = _stop_realized(
                side=side,
                entry=entry,
                stop_loss_pips=sl_pips,
                candle=candle,
                pip_factor=pip_factor,
                gap_allowed=True,
            )
            exit_reason = (
                "STOP_LOSS_GAP_AMBIGUOUS_SAME_S5"
                if realized < -_round(sl_pips)
                else "STOP_LOSS_AMBIGUOUS_SAME_S5"
            )
            exit_at = candle.timestamp_utc
            break
        if sl_hit:
            realized = _stop_realized(
                side=side,
                entry=entry,
                stop_loss_pips=sl_pips,
                candle=candle,
                pip_factor=pip_factor,
                gap_allowed=True,
            )
            exit_reason = "STOP_LOSS_GAP" if realized < -_round(sl_pips) else "STOP_LOSS"
            exit_at = candle.timestamp_utc
            break
        if tp_hit:
            mfe = max(float(mfe or 0.0), favorable)
            realized = tp_pips
            exit_reason = "TAKE_PROFIT"
            exit_at = candle.timestamp_utc
            break
        mfe = max(float(mfe or 0.0), favorable)
    if fill_at is not None and exit_at is None:
        exit_reason = "HORIZON_FULL_STOP_LOSS"
        realized = -sl_pips
        exit_at = fill_at + timedelta(seconds=hold)
    evaluated_grid = _evaluated_grid_count(
        generated=generated,
        maturity=maturity,
        exit_reason=exit_reason,
        exit_at=exit_at,
    )
    evaluated_no_tick = evaluated_grid - evaluated_candles
    if evaluated_no_tick < 0:
        raise ValueError("arm evaluated candles exceed its exact prefix")
    input_sha = _canonical_sha(dict(arm))
    arm_body = {
        "contract": ARM_OUTCOME_CONTRACT,
        "schema_version": 1,
        "arm_id": str(arm["arm_id"]),
        "arm_input_sha256": input_sha,
        "changed_axis": str(arm["changed_axis"]),
        "horizon_lane": str(arm["horizon_lane"]),
        "entry_fraction_toward_market": float(arm["entry_fraction_toward_market"]),
        "effective_entry_fraction_toward_market": float(
            arm["effective_entry_fraction_toward_market"]
        ),
        "entry_ttl_seconds": ttl,
        "max_hold_seconds": hold,
        "entry": entry,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "take_profit_pips": tp_pips,
        "stop_loss_pips": sl_pips,
        "maturity_at_utc": maturity.isoformat(),
        "filled": fill_at is not None,
        "fill_at_utc": fill_at.isoformat() if fill_at else None,
        "time_to_fill_seconds": (
            _round((fill_at - generated).total_seconds()) if fill_at else None
        ),
        "exit_at_utc": exit_at.isoformat() if exit_at else None,
        "exit_reason": exit_reason,
        "post_cost_realized_pips": _round(realized),
        "realized_pips": _round(realized),
        "maximum_favorable_excursion_pips": _round(mfe) if mfe is not None else None,
        "maximum_adverse_excursion_pips": _round(mae) if mae is not None else None,
        "ambiguous_same_s5": ambiguous,
        "truth_grid_slot_count": _grid_slot_count(generated, maturity),
        "truth_candle_count": len(path),
        "truth_no_tick_slot_count": _grid_slot_count(generated, maturity) - len(path),
        "evaluated_candle_count": evaluated_candles,
        "evaluated_grid_slot_count": evaluated_grid,
        "evaluated_no_tick_slot_count": evaluated_no_tick,
        "evaluated_prefix_from_utc": _ceil_s5(generated).isoformat(),
        "evaluated_prefix_to_utc": _evaluated_prefix_end(
            generated=generated,
            maturity=maturity,
            exit_reason=exit_reason,
            exit_at=exit_at,
        ).isoformat(),
        "truth_path_sha256": truth_path_sha256,
        "truth_chunk_sha256": list(truth_chunk_sha256),
        "truth_source": "OANDA_S5_BID_ASK",
        "automatic_promotion_allowed": False,
        "primary_effect": False,
        "risk_effect": False,
        "order_authority": "NONE",
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return _seal_named(arm_body, "arm_outcome_sha256")


def resolve_due_fast_bot_learning_outcomes_from_oanda(
    *,
    shadow_ledger_path: Path,
    outcome_ledger_path: Path,
    scorecard_path: Path,
    client_factory: Callable[[], Any] = OandaReadOnlyClient,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Resolve mature seats once; preflight conflicts before broker reads."""

    now = _aware_utc((clock or (lambda: datetime.now(timezone.utc)))())
    base = {
        "contract": TRUTH_ADAPTER_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "order_authority": "NONE",
        "diagnostic_only": True,
        "automatic_promotion_allowed": False,
        "primary_effect": False,
        "risk_effect": False,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    try:
        seats = _load_learning_seats(shadow_ledger_path)
        outcomes = _load_jsonl(outcome_ledger_path)
    except ValueError as exc:
        return {
            **base,
            "status": "LEDGER_INVALID_FAIL_CLOSED",
            "broker_read": False,
            "ledger_appended": 0,
            "error": str(exc)[:320],
        }
    outcomes_by_identity: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in outcomes:
        identity = _current_identity(row)
        if identity is not None:
            outcomes_by_identity.setdefault(identity, []).append(row)
    resolved_sha: set[str] = set()
    conflicts: list[dict[str, Any]] = []
    due: list[tuple[datetime, Mapping[str, Any]]] = []
    for seat in seats:
        identity = _seat_identity(seat)
        current = outcomes_by_identity.get(identity, [])
        valid = [row for row in current if _outcome_valid_for_seat(row, seat)]
        if len(current) == 1 and len(valid) == 1:
            resolved_sha.add(identity[0])
            continue
        if current:
            conflicts.append(
                {
                    "seat_id": str(seat["seat_id"]),
                    "seat_contract_sha256": identity[0],
                    "current_policy_row_count": len(current),
                    "valid_current_policy_row_count": len(valid),
                    "reason": "CURRENT_POLICY_OUTCOME_IDENTITY_CONFLICT",
                }
            )
            continue
        maturity = _seat_maturity(seat)
        if maturity <= now:
            due.append((maturity, seat))
    due.sort(key=lambda item: (item[0], str(item[1]["seat_id"])))
    selected = [seat for _, seat in due[:MAX_DUE_PER_RUN]]
    if not selected:
        scorecard = build_fast_bot_learning_scorecard(seats, outcomes, as_of_utc=now)
        _write_json_atomic(scorecard_path, scorecard)
        return {
            **base,
            "status": "OUTCOME_IDENTITY_CONFLICT" if conflicts else "NO_DUE_SEATS",
            "broker_read": False,
            "due_count": len(due),
            "selected_due_count": 0,
            "outcome_identity_conflict_count": len(conflicts),
            "outcome_identity_conflicts": conflicts,
            "ledger_appended": 0,
            "scorecard_status": scorecard["status"],
        }
    client = client_factory()
    resolved: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for seat in selected:
        try:
            generated = _parse_utc(seat["generated_at_utc"])
            maturity = _seat_maturity(seat)
            candles, hashes = fetch_frozen_s5_truth(
                client,
                pair=str(seat["pair"]),
                time_from=generated,
                time_to=maturity,
                chunk_candle_limit=TRUTH_CHUNK_CANDLE_LIMIT,
            )
            resolved.append(
                resolve_fast_bot_learning_seat(
                    seat,
                    candles,
                    resolved_at_utc=now,
                    truth_chunk_sha256=hashes,
                )
            )
        except Exception as exc:  # pragma: no cover - network boundary
            errors.append(
                {
                    "seat_id": str(seat.get("seat_id") or ""),
                    "pair": str(seat.get("pair") or ""),
                    "error": f"{type(exc).__name__}: {exc}"[:320],
                }
            )
    try:
        append_result = _append_outcomes_once(outcome_ledger_path, resolved)
        all_outcomes = _load_jsonl(outcome_ledger_path)
        scorecard = build_fast_bot_learning_scorecard(seats, all_outcomes, as_of_utc=now)
        _write_json_atomic(scorecard_path, scorecard)
    except ValueError as exc:
        return {
            **base,
            "status": "OUTCOME_PERSISTENCE_FAILED_CLOSED",
            "broker_read": True,
            "due_count": len(due),
            "selected_due_count": len(selected),
            "resolved_in_memory_count": len(resolved),
            "ledger_appended": 0,
            "errors": [*errors, {"error": str(exc)[:320]}],
        }
    return {
        **base,
        "status": (
            "RESOLVED_WITH_ERRORS"
            if errors or conflicts or append_result["duplicate_identity_count"]
            else "RESOLVED"
        ),
        "broker_read": True,
        "due_count": len(due),
        "selected_due_count": len(selected),
        "resolved_in_memory_count": len(resolved),
        "ledger_appended": append_result["appended"],
        "ledger_duplicate_outcome_identities_skipped": append_result[
            "duplicate_identity_count"
        ],
        "outcome_identity_conflict_count": len(conflicts),
        "outcome_identity_conflicts": conflicts,
        "errors": errors,
        "scorecard_status": scorecard["status"],
    }


def build_fast_bot_learning_scorecard(
    seats: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: datetime,
) -> dict[str, Any]:
    """Aggregate exact, non-netted counterfactual arm outcomes."""

    valid_seats = [seat for seat in seats if _learning_seat_deep_valid(seat)]
    seats_by_sha = {str(seat["contract_sha256"]): seat for seat in valid_seats}
    valid_outcomes: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    identity_counts: dict[tuple[str, str], int] = {}
    for row in outcomes:
        identity = _current_identity(row)
        if identity is not None:
            identity_counts[identity] = identity_counts.get(identity, 0) + 1
    for row in outcomes:
        seat = seats_by_sha.get(str(row.get("seat_contract_sha256") or ""))
        identity = _current_identity(row)
        if (
            seat is not None
            and identity is not None
            and identity_counts.get(identity) == 1
            and _outcome_valid_for_seat(row, seat)
        ):
            valid_outcomes.append((seat, row))
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for seat, outcome in valid_outcomes:
        cost_bucket = str(seat["cost_context"]["cost_pressure_bucket"])
        for candidate in outcome["candidates"]:
            arms = {str(arm["arm_id"]): arm for arm in candidate["arms"]}
            base = arms.get("BASE")
            if base is None:
                continue
            base_value = float(base["post_cost_realized_pips"])
            for arm in candidate["arms"]:
                key = (
                    str(candidate["candidate_class"]),
                    cost_bucket,
                    str(candidate["pair"]),
                    str(candidate["side"]),
                    str(candidate["method"]),
                    str(arm["horizon_lane"]),
                    str(arm["arm_id"]),
                )
                grouped.setdefault(key, []).append(
                    {
                        "arm": arm,
                        "paired_delta": float(arm["post_cost_realized_pips"]) - base_value,
                    }
                )
    groups = [_score_group(key, rows) for key, rows in sorted(grouped.items())]
    body = {
        "contract": SCORECARD_CONTRACT,
        "schema_version": 1,
        "scoring_policy": SCORING_POLICY,
        "generated_at_utc": _aware_utc(as_of_utc).isoformat(),
        "status": "COLLECTING_COUNTERFACTUAL_EVIDENCE",
        "emitted_seats": len(valid_seats),
        "resolved_seats": len(valid_outcomes),
        "grouping_dimensions": [
            "candidate_class",
            "cost_pressure_bucket",
            "pair",
            "side",
            "method",
            "horizon_lane",
            "arm_id",
        ],
        "pair_direction_method_horizon_netting_allowed": False,
        "groups": groups,
        "automatic_parameter_change_allowed": False,
        "automatic_promotion_allowed": False,
        "selected_arm_id": None,
        "primary_effect": False,
        "risk_effect": False,
        "order_authority": "NONE",
        "diagnostic_only": True,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return _seal(body)


def _score_group(key: tuple[str, ...], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms = [row["arm"] for row in rows]
    fills = [arm for arm in arms if arm["filled"] is True]
    values = [float(arm["post_cost_realized_pips"]) for arm in arms]
    fill_values = [float(arm["post_cost_realized_pips"]) for arm in fills]
    wins = [value for value in fill_values if value > 0.0]
    losses = [value for value in fill_values if value < 0.0]
    gross_loss = abs(sum(losses))
    profit_factor: float | None = (
        sum(wins) / gross_loss
        if gross_loss > 0.0
        else math.inf if wins else None
    )
    deltas = [float(row["paired_delta"]) for row in rows]
    mfe = [float(arm["maximum_favorable_excursion_pips"]) for arm in fills]
    mae = [float(arm["maximum_adverse_excursion_pips"]) for arm in fills]
    return {
        "candidate_class": key[0],
        "cost_pressure_bucket": key[1],
        "pair": key[2],
        "side": key[3],
        "method": key[4],
        "horizon_lane": key[5],
        "arm_id": key[6],
        "resolved": len(arms),
        "fills": len(fills),
        "unfilled": len(arms) - len(fills),
        "wins": len(wins),
        "losses": len(losses),
        "fill_rate": _round(len(fills) / len(arms)) if arms else None,
        "net_post_cost_pips": _round(sum(values)),
        "mean_post_cost_pips_per_resolved": _round(statistics.fmean(values)) if values else None,
        "mean_post_cost_pips_per_fill": _round(statistics.fmean(fill_values)) if fill_values else None,
        "profit_factor": (
            _round(profit_factor)
            if profit_factor is not None and math.isfinite(profit_factor)
            else "INF" if profit_factor == math.inf else None
        ),
        "mean_mfe_pips": _round(statistics.fmean(mfe)) if mfe else None,
        "mean_mae_pips": _round(statistics.fmean(mae)) if mae else None,
        "paired_count_vs_base": len(deltas),
        "net_paired_delta_pips_vs_base": _round(sum(deltas)),
        "mean_paired_delta_pips_vs_base": _round(statistics.fmean(deltas)) if deltas else None,
    }


def _validate_learning_seat(seat: Mapping[str, Any]) -> None:
    if not _learning_seat_deep_valid(seat):
        raise ValueError("invalid fast-bot learning seat")


def _learning_seat_deep_valid(seat: Mapping[str, Any]) -> bool:
    try:
        if not _learning_seat_valid(seat):
            return False
        generated = _parse_utc(seat["generated_at_utc"])
        quote_at = _parse_utc(seat["quote_timestamp_utc"])
        pair = str(seat["pair"])
        bid = float(seat["quote_bid"])
        ask = float(seat["quote_ask"])
        spread = float(seat["executable_spread_pips"])
        candidates = seat["candidates"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    try:
        observed_spread = (ask - bid) * float(instrument_pip_factor(pair))
    except (KeyError, TypeError, ValueError):
        return False
    if not bool(
        seat.get("contract") == LEARNING_SEAT_CONTRACT
        and pair in DEFAULT_TRADER_PAIRS
        and bid > 0.0
        and ask > bid
        and 0.0 <= (generated - quote_at).total_seconds() <= 45.0
        and math.isclose(spread, observed_spread, rel_tol=0.0, abs_tol=1e-6)
        and isinstance(candidates, list)
        and 1 <= len(candidates) <= len(CELL_ORDER)
    ):
        return False
    candidate_ids: set[str] = set()
    cells: set[tuple[str, str]] = set()
    expected_arm_ids = [spec[0] for spec in LEARNING_ARM_SPECS]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return False
        body = {key: value for key, value in candidate.items() if key != "candidate_sha256"}
        cell = (str(candidate.get("side") or ""), str(candidate.get("method") or ""))
        arms = candidate.get("arms")
        if not bool(
            str(candidate.get("candidate_sha256") or "") == _canonical_sha(body)
            and str(candidate.get("candidate_id") or "") not in candidate_ids
            and cell not in cells
            and cell[0] in SIDES
            and cell[1] in METHODS
            and candidate.get("candidate_class") in CANDIDATE_CLASSES
            and isinstance(arms, list)
            and [arm.get("arm_id") for arm in arms if isinstance(arm, Mapping)]
            == expected_arm_ids
            and all(_arm_input_valid(arm, side=cell[0]) for arm in arms)
        ):
            return False
        candidate_ids.add(str(candidate["candidate_id"]))
        cells.add(cell)
    return True


def _arm_input_valid(arm: Any, *, side: str) -> bool:
    if not isinstance(arm, Mapping):
        return False
    try:
        entry = float(arm["entry"])
        tp = float(arm["take_profit"])
        sl = float(arm["stop_loss"])
        ttl = arm["entry_ttl_seconds"]
        hold = arm["max_hold_seconds"]
        tp_pips = float(arm["take_profit_pips"])
        sl_pips = float(arm["stop_loss_pips"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        ttl.__class__ is int
        and hold.__class__ is int
        and ttl > 0
        and hold > 0
        and tp_pips > 0.0
        and sl_pips > 0.0
        and all(math.isfinite(value) for value in (entry, tp, sl, tp_pips, sl_pips))
        and ((tp > entry > sl) if side == "LONG" else (sl > entry > tp))
    )


def _outcome_valid_for_seat(outcome: Mapping[str, Any], seat: Mapping[str, Any]) -> bool:
    try:
        if not _sealed_valid(outcome, OUTCOME_CONTRACT):
            return False
        generated = _parse_utc(outcome["seat_generated_at_utc"])
        maturity = _parse_utc(outcome["maturity_at_utc"])
        resolved = _parse_utc(outcome["resolved_at_utc"])
        request_from = _parse_utc(outcome["truth_request_from_utc"])
        request_to = _parse_utc(outcome["truth_request_to_utc"])
        truth_grid_count = outcome["truth_grid_slot_count"]
        truth_candle_count = outcome["truth_candle_count"]
        truth_no_tick_count = outcome["truth_no_tick_slot_count"]
        truth_hashes = outcome["truth_chunk_sha256"]
        candidates = outcome["candidates"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    expected_candidates = {str(row["candidate_sha256"]): row for row in seat["candidates"]}
    if not bool(
        outcome.get("scoring_policy") == SCORING_POLICY
        and outcome.get("seat_id") == seat.get("seat_id")
        and outcome.get("seat_contract_sha256") == seat.get("contract_sha256")
        and outcome.get("pair") == seat.get("pair")
        and generated == _parse_utc(seat["generated_at_utc"])
        and maturity == _seat_maturity(seat)
        and resolved >= maturity
        and request_from == _ceil_s5(generated)
        and request_to == _floor_s5(maturity)
        and outcome.get("truth_request_coverage_proved") is True
        and outcome.get("truth_source") == "OANDA_S5_BID_ASK"
        and truth_grid_count.__class__ is int
        and truth_grid_count == _grid_slot_count(generated, maturity)
        and truth_candle_count.__class__ is int
        and 0 <= truth_candle_count <= truth_grid_count
        and truth_no_tick_count.__class__ is int
        and truth_no_tick_count == truth_grid_count - truth_candle_count
        and _sha256_text(outcome.get("truth_path_sha256"))
        and isinstance(truth_hashes, list)
        and len(truth_hashes)
        == math.ceil(truth_grid_count / TRUTH_CHUNK_CANDLE_LIMIT)
        and all(_sha256_text(item) for item in truth_hashes)
        and outcome.get("candidate_count") == len(expected_candidates)
        and outcome.get("candidate_class_cost_context") == seat.get("cost_context")
        and isinstance(candidates, list)
        and len(candidates) == len(expected_candidates)
        and outcome.get("automatic_promotion_allowed") is False
        and outcome.get("primary_effect") is False
        and outcome.get("risk_effect") is False
        and outcome.get("order_authority") == "NONE"
        and outcome.get("shadow_only") is True
        and outcome.get("live_permission") is False
        and outcome.get("broker_mutation_allowed") is False
    ):
        return False
    seen_candidates: set[str] = set()
    for candidate_outcome in candidates:
        if not isinstance(candidate_outcome, Mapping):
            return False
        candidate_sha = str(candidate_outcome.get("candidate_sha256") or "")
        candidate = expected_candidates.get(candidate_sha)
        if candidate is None or candidate_sha in seen_candidates:
            return False
        if not _candidate_outcome_valid(
            candidate_outcome,
            candidate=candidate,
            seat=seat,
            truth_path_sha256=str(outcome["truth_path_sha256"]),
            truth_chunk_sha256=outcome["truth_chunk_sha256"],
        ):
            return False
        seen_candidates.add(candidate_sha)
    return len(seen_candidates) == len(expected_candidates)


def _candidate_outcome_valid(
    value: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    seat: Mapping[str, Any],
    truth_path_sha256: str,
    truth_chunk_sha256: Sequence[str],
) -> bool:
    if not _sealed_named_valid(value, CANDIDATE_OUTCOME_CONTRACT, "candidate_outcome_sha256"):
        return False
    arms = value.get("arms")
    expected = {str(arm["arm_id"]): arm for arm in candidate["arms"]}
    if not bool(
        value.get("candidate_id") == candidate.get("candidate_id")
        and value.get("candidate_sha256") == candidate.get("candidate_sha256")
        and value.get("candidate_class") == candidate.get("candidate_class")
        and value.get("pair") == seat.get("pair")
        and value.get("side") == candidate.get("side")
        and value.get("method") == candidate.get("method")
        and value.get("truth_path_sha256") == truth_path_sha256
        and isinstance(arms, list)
        and len(arms) == len(expected)
        and value.get("automatic_promotion_allowed") is False
        and value.get("primary_effect") is False
        and value.get("risk_effect") is False
        and value.get("order_authority") == "NONE"
        and value.get("shadow_only") is True
        and value.get("live_permission") is False
        and value.get("broker_mutation_allowed") is False
    ):
        return False
    seen: set[str] = set()
    for arm_outcome in arms:
        arm_id = str(arm_outcome.get("arm_id") or "") if isinstance(arm_outcome, Mapping) else ""
        arm = expected.get(arm_id)
        if arm is None or arm_id in seen or not _arm_outcome_valid(
            arm_outcome,
            arm=arm,
            side=str(candidate["side"]),
            generated=_parse_utc(seat["generated_at_utc"]),
            truth_path_sha256=truth_path_sha256,
            truth_chunk_sha256=truth_chunk_sha256,
        ):
            return False
        seen.add(arm_id)
    return len(seen) == len(expected)


def _arm_outcome_valid(
    value: Any,
    *,
    arm: Mapping[str, Any],
    side: str,
    generated: datetime,
    truth_path_sha256: str,
    truth_chunk_sha256: Sequence[str],
) -> bool:
    if not isinstance(value, Mapping) or not _sealed_named_valid(
        value, ARM_OUTCOME_CONTRACT, "arm_outcome_sha256"
    ):
        return False
    try:
        filled = value["filled"]
        realized = float(value["post_cost_realized_pips"])
        mfe = value["maximum_favorable_excursion_pips"]
        mae = value["maximum_adverse_excursion_pips"]
        maturity = _parse_utc(value["maturity_at_utc"])
        expected_maturity = generated + timedelta(
            seconds=int(arm["entry_ttl_seconds"]) + int(arm["max_hold_seconds"])
        )
        truth_grid_count = value["truth_grid_slot_count"]
        truth_candle_count = value["truth_candle_count"]
        truth_no_tick_count = value["truth_no_tick_slot_count"]
        evaluated_candle_count = value["evaluated_candle_count"]
        evaluated_grid_count = value["evaluated_grid_slot_count"]
        evaluated_no_tick_count = value["evaluated_no_tick_slot_count"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    geometry_keys = (
        "arm_id",
        "changed_axis",
        "horizon_lane",
        "entry_fraction_toward_market",
        "effective_entry_fraction_toward_market",
        "entry_ttl_seconds",
        "max_hold_seconds",
        "entry",
        "take_profit",
        "stop_loss",
        "take_profit_pips",
        "stop_loss_pips",
    )
    if not bool(
        all(value.get(key) == arm.get(key) for key in geometry_keys)
        and value.get("arm_input_sha256") == _canonical_sha(dict(arm))
        and maturity == expected_maturity
        and value.get("truth_path_sha256") == truth_path_sha256
        and value.get("truth_chunk_sha256") == list(truth_chunk_sha256)
        and all(_sha256_text(item) for item in value["truth_chunk_sha256"])
        and value.get("realized_pips") == value.get("post_cost_realized_pips")
        and isinstance(filled, bool)
        and math.isfinite(realized)
        and truth_grid_count.__class__ is int
        and truth_grid_count == _grid_slot_count(generated, expected_maturity)
        and truth_candle_count.__class__ is int
        and 0 <= truth_candle_count <= truth_grid_count
        and truth_no_tick_count.__class__ is int
        and truth_no_tick_count == truth_grid_count - truth_candle_count
        and evaluated_candle_count.__class__ is int
        and 0 <= evaluated_candle_count <= truth_candle_count
        and evaluated_grid_count.__class__ is int
        and evaluated_no_tick_count.__class__ is int
        and evaluated_no_tick_count >= 0
        and evaluated_grid_count - evaluated_no_tick_count == evaluated_candle_count
        and value.get("automatic_promotion_allowed") is False
        and value.get("primary_effect") is False
        and value.get("risk_effect") is False
        and value.get("order_authority") == "NONE"
        and value.get("shadow_only") is True
        and value.get("live_permission") is False
        and value.get("broker_mutation_allowed") is False
    ):
        return False
    if filled is False:
        return bool(
            value.get("fill_at_utc") is None
            and value.get("time_to_fill_seconds") is None
            and value.get("exit_at_utc") is None
            and value.get("exit_reason") == "UNFILLED"
            and realized == 0.0
            and mfe is None
            and mae is None
            and evaluated_grid_count == truth_grid_count
            and evaluated_candle_count == truth_candle_count
            and evaluated_no_tick_count == truth_no_tick_count
            and value.get("evaluated_prefix_from_utc") == _ceil_s5(generated).isoformat()
            and value.get("evaluated_prefix_to_utc")
            == _floor_s5(expected_maturity).isoformat()
        )
    try:
        fill_at = _parse_utc(value["fill_at_utc"])
        exit_at = _parse_utc(value["exit_at_utc"])
        time_to_fill = float(value["time_to_fill_seconds"])
        mfe_number = float(mfe)
        mae_number = float(mae)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    reason = str(value.get("exit_reason") or "")
    valid_reasons = {
        "TAKE_PROFIT",
        "STOP_LOSS",
        "STOP_LOSS_GAP",
        "STOP_LOSS_AMBIGUOUS_FILL_S5",
        "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5",
        "STOP_LOSS_AMBIGUOUS_SAME_S5",
        "STOP_LOSS_GAP_AMBIGUOUS_SAME_S5",
        "HORIZON_FULL_STOP_LOSS",
    }
    try:
        expected_evaluated_grid = _evaluated_grid_count(
            generated=generated,
            maturity=expected_maturity,
            exit_reason=reason,
            exit_at=exit_at,
        )
        expected_prefix_end = _evaluated_prefix_end(
            generated=generated,
            maturity=expected_maturity,
            exit_reason=reason,
            exit_at=exit_at,
        )
    except ValueError:
        return False
    stop_pips = float(arm["stop_loss_pips"])
    tp_pips = float(arm["take_profit_pips"])
    if reason == "TAKE_PROFIT":
        result_semantics = (
            value.get("ambiguous_same_s5") is False
            and math.isclose(realized, tp_pips, rel_tol=0.0, abs_tol=1e-6)
        )
    elif reason in {"STOP_LOSS", "HORIZON_FULL_STOP_LOSS"}:
        result_semantics = (
            value.get("ambiguous_same_s5") is False
            and math.isclose(realized, -stop_pips, rel_tol=0.0, abs_tol=1e-6)
        )
    elif reason in {
        "STOP_LOSS_AMBIGUOUS_FILL_S5",
        "STOP_LOSS_AMBIGUOUS_SAME_S5",
    }:
        result_semantics = (
            value.get("ambiguous_same_s5") is True
            and math.isclose(realized, -stop_pips, rel_tol=0.0, abs_tol=1e-6)
        )
    else:
        result_semantics = (
            reason in {
                "STOP_LOSS_GAP",
                "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5",
                "STOP_LOSS_GAP_AMBIGUOUS_SAME_S5",
            }
            and realized < -_round(stop_pips)
            and value.get("ambiguous_same_s5")
            is (reason != "STOP_LOSS_GAP")
        )
    return bool(
        reason in valid_reasons
        and generated <= fill_at <= exit_at <= expected_maturity
        and _floor_s5(fill_at) == fill_at
        and _floor_s5(exit_at) == exit_at
        and fill_at + timedelta(seconds=5)
        <= generated + timedelta(seconds=int(arm["entry_ttl_seconds"]))
        and math.isclose(time_to_fill, (fill_at - generated).total_seconds(), abs_tol=1e-6)
        and math.isfinite(mfe_number)
        and math.isfinite(mae_number)
        and mfe_number >= 0.0
        and mae_number >= 0.0
        and side in SIDES
        and result_semantics
        and (
            exit_at == fill_at + timedelta(seconds=int(arm["max_hold_seconds"]))
            if reason == "HORIZON_FULL_STOP_LOSS"
            else exit_at
            < min(
                fill_at + timedelta(seconds=int(arm["max_hold_seconds"])),
                expected_maturity,
            )
        )
        and evaluated_grid_count == expected_evaluated_grid
        and value.get("evaluated_prefix_from_utc") == _ceil_s5(generated).isoformat()
        and value.get("evaluated_prefix_to_utc") == expected_prefix_end.isoformat()
    )


def _seat_maturity(seat: Mapping[str, Any]) -> datetime:
    generated = _parse_utc(seat["generated_at_utc"])
    durations = [
        int(arm["entry_ttl_seconds"]) + int(arm["max_hold_seconds"])
        for candidate in seat["candidates"]
        for arm in candidate["arms"]
    ]
    if not durations:
        raise ValueError("learning seat has no arms")
    return generated + timedelta(seconds=max(durations))


def _load_learning_seats(path: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    seats: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_buckets: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=1):
        if not _learning_seat_deep_valid(row):
            raise ValueError(f"invalid learning seat row {index}")
        seat_id = str(row["seat_id"])
        bucket = (str(row["pair"]), str(row["sampling_bucket_utc"]))
        if seat_id in seen_ids or bucket in seen_buckets:
            raise ValueError(f"duplicate learning seat identity at row {index}")
        seen_ids.add(seat_id)
        seen_buckets.add(bucket)
        seats.append(row)
    return seats


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
                raise ValueError(f"malformed JSONL row {line_number} in {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row {line_number} in {path}")
            rows.append(value)
    return rows


def _append_outcomes_once(path: Path, outcomes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    if not outcomes:
        return {"appended": 0, "duplicate_identity_count": 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen: set[tuple[str, str]] = set()
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"malformed outcome row {line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"non-object outcome row {line_number}")
            identity = _current_identity(row)
            if identity is not None:
                if not _sealed_valid(row, OUTCOME_CONTRACT):
                    raise ValueError(
                        f"invalid current-policy outcome row {line_number}"
                    )
                if identity in seen:
                    raise ValueError(f"duplicate current-policy outcome row {line_number}")
                seen.add(identity)
        appended = 0
        duplicates = 0
        handle.seek(0, os.SEEK_END)
        for outcome in outcomes:
            identity = _current_identity(outcome)
            if identity is None or not _sealed_valid(outcome, OUTCOME_CONTRACT):
                raise ValueError("attempted to append invalid learning outcome")
            if identity in seen:
                duplicates += 1
                continue
            handle.write(json.dumps(dict(outcome), ensure_ascii=False, sort_keys=True) + "\n")
            seen.add(identity)
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {"appended": appended, "duplicate_identity_count": duplicates}


def _seat_identity(seat: Mapping[str, Any]) -> tuple[str, str]:
    return str(seat.get("contract_sha256") or ""), SCORING_POLICY


def _current_identity(outcome: Mapping[str, Any]) -> tuple[str, str] | None:
    seat_sha = str(outcome.get("seat_contract_sha256") or "")
    if (
        outcome.get("contract") == OUTCOME_CONTRACT
        and outcome.get("scoring_policy") == SCORING_POLICY
        and _sha256_text(seat_sha)
    ):
        return seat_sha, SCORING_POLICY
    return None


def _validate_truth_path(
    candles: Sequence[S5BidAskCandle], *, generated: datetime, maturity: datetime
) -> None:
    first = _ceil_s5(generated)
    end = _floor_s5(maturity)
    timestamps = [item.timestamp_utc for item in candles]
    if not bool(
        _grid_slot_count(generated, maturity) > 0
        and len(timestamps) <= _grid_slot_count(generated, maturity)
        and len(timestamps) == len(set(timestamps))
        and all(first <= item < end and _floor_s5(item) == item for item in timestamps)
        and all(_candle_valid(item) for item in candles)
    ):
        raise ValueError("invalid learning S5 truth path")


def _candle_valid(candle: S5BidAskCandle) -> bool:
    values = (
        candle.bid_o, candle.bid_h, candle.bid_l, candle.bid_c,
        candle.ask_o, candle.ask_h, candle.ask_l, candle.ask_c,
    )
    return bool(
        all(math.isfinite(float(value)) and float(value) > 0.0 for value in values)
        and candle.bid_l <= min(candle.bid_o, candle.bid_c)
        and max(candle.bid_o, candle.bid_c) <= candle.bid_h
        and candle.ask_l <= min(candle.ask_o, candle.ask_c)
        and max(candle.ask_o, candle.ask_c) <= candle.ask_h
        and candle.ask_o > candle.bid_o
        and candle.ask_h > candle.bid_h
        and candle.ask_l > candle.bid_l
        and candle.ask_c > candle.bid_c
    )


def _truth_path_sha(
    *,
    pair: str,
    generated: datetime,
    maturity: datetime,
    candles: Sequence[S5BidAskCandle],
    chunk_hashes: Sequence[str],
) -> str:
    return _canonical_sha(
        {
            "pair": pair,
            "from": _ceil_s5(generated).isoformat(),
            "to": _floor_s5(maturity).isoformat(),
            "truth_chunk_sha256": list(chunk_hashes),
            "candles": [
                {
                    "timestamp_utc": item.timestamp_utc.isoformat(),
                    "bid": [item.bid_o, item.bid_h, item.bid_l, item.bid_c],
                    "ask": [item.ask_o, item.ask_h, item.ask_l, item.ask_c],
                }
                for item in candles
            ],
        }
    )


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


def _favorable_excursion(
    side: str, entry: float, candle: S5BidAskCandle, pip_factor: float
) -> float:
    raw = (
        (candle.bid_h - entry) * pip_factor
        if side == "LONG"
        else (entry - candle.ask_l) * pip_factor
    )
    return max(0.0, _round(raw))


def _adverse_excursion(
    side: str, entry: float, candle: S5BidAskCandle, pip_factor: float
) -> float:
    raw = (
        (entry - candle.bid_l) * pip_factor
        if side == "LONG"
        else (candle.ask_h - entry) * pip_factor
    )
    return max(0.0, _round(raw))


def _stop_realized(
    *,
    side: str,
    entry: float,
    stop_loss_pips: float,
    candle: S5BidAskCandle,
    pip_factor: float,
    gap_allowed: bool,
) -> float:
    opening = (
        (candle.bid_o - entry) * pip_factor
        if side == "LONG"
        else (entry - candle.ask_o) * pip_factor
    )
    return _round(min(-stop_loss_pips, opening) if gap_allowed else -stop_loss_pips)


def _evaluated_grid_count(
    *, generated: datetime, maturity: datetime, exit_reason: str, exit_at: datetime | None
) -> int:
    first = _ceil_s5(generated)
    if exit_reason == "UNFILLED":
        return _grid_slot_count(generated, maturity)
    if exit_at is None or _floor_s5(exit_at) != exit_at:
        raise ValueError("filled arm requires aligned exit")
    count = int((exit_at - first).total_seconds() // 5)
    if exit_reason != "HORIZON_FULL_STOP_LOSS":
        count += 1
    if not 0 < count <= _grid_slot_count(generated, maturity):
        raise ValueError("evaluated arm prefix outside truth grid")
    return count


def _evaluated_prefix_end(
    *, generated: datetime, maturity: datetime, exit_reason: str, exit_at: datetime | None
) -> datetime:
    if exit_reason == "UNFILLED":
        return _floor_s5(maturity)
    if exit_at is None:
        raise ValueError("filled arm requires exit")
    return exit_at if exit_reason == "HORIZON_FULL_STOP_LOSS" else exit_at + timedelta(seconds=5)


def _grid_slot_count(generated: datetime, maturity: datetime) -> int:
    return max(0, int((_floor_s5(maturity) - _ceil_s5(generated)).total_seconds() // 5))


def _ceil_s5(value: datetime) -> datetime:
    micros = int(round(_aware_utc(value).timestamp() * 1_000_000))
    aligned = ((micros + 4_999_999) // 5_000_000) * 5_000_000
    return datetime.fromtimestamp(aligned / 1_000_000, tz=timezone.utc)


def _floor_s5(value: datetime) -> datetime:
    micros = int(round(_aware_utc(value).timestamp() * 1_000_000))
    aligned = (micros // 5_000_000) * 5_000_000
    return datetime.fromtimestamp(aligned / 1_000_000, tz=timezone.utc)


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be an aware ISO string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be an aware ISO string") from exc
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _round(value: float | int) -> float:
    return round(float(value), ROUND_DIGITS)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _canonical_sha(body)}


def _seal_named(value: Mapping[str, Any], digest_key: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != digest_key}
    return {**body, digest_key: _canonical_sha(body)}


def _sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != contract:
        return False
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return str(value.get("contract_sha256") or "") == _canonical_sha(body)


def _sealed_named_valid(value: Mapping[str, Any], contract: str, digest_key: str) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != contract:
        return False
    body = {key: item for key, item in value.items() if key != digest_key}
    return str(value.get(digest_key) or "") == _canonical_sha(body)


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


__all__ = [
    "OUTCOME_CONTRACT",
    "SCORING_POLICY",
    "SCORECARD_CONTRACT",
    "build_fast_bot_learning_scorecard",
    "resolve_due_fast_bot_learning_outcomes_from_oanda",
    "resolve_fast_bot_learning_seat",
]

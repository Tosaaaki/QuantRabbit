#!/usr/bin/env python3
"""Audit passive RANGE-forecast LIMIT vehicles on local OANDA bid/ask truth.

This validator is intentionally read-only.  It evaluates only information that
existed when a RANGE forecast was emitted: the forecast box, current price,
confidence, and horizon.  A single nearest rail is selected, forecasts for the
same pair are spaced so overlapping persistence is not counted as independent
evidence, and all fills/exits use executable bid/ask sides.

The output is hypothesis evidence, never live permission.  Runtime adoption
still requires the predictive-SCOUT gateway, loss memory, broker freshness,
spread, margin, event, manual-position, and forward-proof gates.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import statistics
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
for path in (SCRIPT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from oanda_history_replay_validate import (  # noqa: E402
    QuoteCandle,
    _history_dirs,
    _is_likely_fx_no_market_window,
    _load_candles,
    _parse_float_csv,
    _parse_pair_filter,
    _parse_time,
    _safe_float,
)
from quant_rabbit.instruments import instrument_pip_factor  # noqa: E402


DEFAULT_TP_GRID_PIPS = (3.0, 5.0, 7.0, 10.0)
DEFAULT_SL_GRID_PIPS = (5.0, 7.0, 10.0, 15.0)
DEFAULT_TTL_MINUTES = 90
DEFAULT_DEDUPE_MINUTES = 90
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_MIN_TRAIN_FILLS = 20
DEFAULT_MIN_VALIDATION_FILLS = 10
DEFAULT_MIN_TOTAL_FILLS = 30
MIN_TRUTH_WINDOW_COVERAGE_RATE = 0.95
JST = timezone(timedelta(hours=9), "JST")
# One-sided 95% Student-t critical values for df=1..30.  These are
# distributional confidence constants, not market-tuned thresholds.  Above 30
# df the standard Cornish-Fisher expansion converges toward the normal 95%
# quantile without adding a SciPy runtime dependency.
ONE_SIDED_T95_BY_DF = (
    math.inf,
    6.3138,
    2.9200,
    2.3534,
    2.1318,
    2.0150,
    1.9432,
    1.8946,
    1.8595,
    1.8331,
    1.8125,
    1.7959,
    1.7823,
    1.7709,
    1.7613,
    1.7531,
    1.7459,
    1.7396,
    1.7341,
    1.7291,
    1.7247,
    1.7207,
    1.7171,
    1.7139,
    1.7109,
    1.7081,
    1.7056,
    1.7033,
    1.7011,
    1.6991,
    1.6973,
)


@dataclass(frozen=True)
class RangeForecastRow:
    source_index: int
    timestamp_utc: datetime
    pair: str
    confidence: float
    current_price: float
    range_low_price: float
    range_high_price: float
    horizon_min: float
    cycle_id: str | None


@dataclass(frozen=True)
class RangeSignal:
    row: RangeForecastRow
    side: str
    entry: float
    range_width_pips: float
    box_position: float


def main() -> int:
    args = _parse_args()
    pairs = _parse_pair_filter(args.pairs)
    rows, load_stats = load_range_forecasts(
        args.forecast_history,
        pairs=pairs,
        dedupe_minutes=args.dedupe_minutes,
    )
    now = datetime.now(timezone.utc)
    mature_rows = [row for row in rows if truth_end(row, ttl_minutes=args.ttl_minutes) <= now]
    no_market_rows = [
        row
        for row in mature_rows
        if range_truth_is_no_market(row, ttl_minutes=args.ttl_minutes)
    ]
    no_market_row_set = set(no_market_rows)
    scorable_rows = [row for row in mature_rows if row not in no_market_row_set]
    history_dirs = _history_dirs(
        args.history_dir,
        granularity=args.granularity,
        auto_min_days=args.auto_history_min_days,
    )
    candles_by_pair, candle_stats = _load_candles(
        history_dirs,
        granularity=args.granularity,
        windows_by_pair=range_truth_windows(scorable_rows, ttl_minutes=args.ttl_minutes),
    )
    results, score_stats = score_range_forecasts(
        scorable_rows,
        candles_by_pair,
        ttl_minutes=args.ttl_minutes,
        tp_grid=args.tp_grid_pips,
        sl_grid=args.sl_grid_pips,
        candle_interval=granularity_interval(args.granularity),
    )
    selections = train_validation_selections(
        results,
        train_fraction=args.train_fraction,
        min_train_fills=args.min_train_fills,
        min_validation_fills=args.min_validation_fills,
        min_total_fills=args.min_total_fills,
        family_truth_coverage={
            (str(item["pair"]), str(item["side"])): item
            for item in score_stats.get("pair_side_truth_coverage", [])
        },
    )
    report = {
        "generated_at_utc": _iso(now),
        "kind": "oanda_range_scout_replay_validate",
        "read_only": True,
        "live_permission_allowed": False,
        "adoption_blockers": [
            "RANGE_SCOUT_TTL_CLOSE_LIVE_CONTRACT_MISMATCH",
            "RANGE_SCOUT_ZERO_LATENCY_ENTRY_ASSUMPTION",
            "RANGE_SCOUT_LIVE_GATE_REPLAY_INCOMPLETE",
        ],
        "source": str(args.forecast_history),
        "history_dirs": [str(path) for path in history_dirs],
        "truth_source": (
            f"local OANDA {args.granularity} bid/ask; LONG LIMIT fills on ask low and exits on bid, "
            "SHORT LIMIT fills on bid high and exits on ask; a fill-candle stop is conservative, "
            "while an ordering-ambiguous fill-candle target "
            "is ignored until a later bar (or the same close) proves a post-fill re-cross"
        ),
        "vehicle": {
            "selector": "NEAREST_RANGE_RAIL_ONE_SIDE",
            "granularity": args.granularity.upper(),
            "entry": "forecast range_low for LONG or range_high for SHORT",
            "entry_eligibility_assumption": (
                "order is active at the first complete S5 boundary at/after forecast emission "
                "with zero additional verifier/gateway latency (0 to <5 seconds of boundary delay)"
            ),
            "ttl_minutes": args.ttl_minutes,
            "candle_interval_seconds": granularity_interval(args.granularity).total_seconds(),
            "same_pair_dedupe_minutes": args.dedupe_minutes,
            "take_profit_pips": list(args.tp_grid_pips),
            "stop_loss_pips": list(args.sl_grid_pips),
            "tp_must_not_cross_midpoint": True,
            "unfilled_is_not_a_trade": True,
            "timeout_exit_included": True,
            "live_contract_alignment": {
                "status": "MISMATCH",
                "simulation_exit": "TP, SL, or executable bid/ask TTL_CLOSE after fill",
                "current_live_exit": "attached broker TP/SL only; GTD expires only an unfilled entry order",
                "required_before_adoption": (
                    "either implement and audit a gateway-controlled filled-position time exit, "
                    "or rerun with the actual attached-TP/SL holding contract"
                ),
            },
            "minimum_pair_side_truth_window_coverage_rate": MIN_TRUTH_WINDOW_COVERAGE_RATE,
        },
        **load_stats,
        "mature_rows": len(mature_rows),
        "pending_future_truth_rows": len(rows) - len(mature_rows),
        "unscorable_no_market_rows": len(no_market_rows),
        **candle_stats,
        **score_stats,
        "comparison_period": {
            "forecast_start_utc": _iso(min(row.timestamp_utc for row in scorable_rows))
            if scorable_rows
            else None,
            "forecast_end_utc": _iso(max(truth_end(row, ttl_minutes=args.ttl_minutes) for row in scorable_rows))
            if scorable_rows
            else None,
        },
        "groups": group_metrics(results),
        "train_validation_selections": selections,
        "evaluated_oos_family_count": len(selections),
        "powered_oos_family_count": sum(
            item.get("oos_powered") is True for item in selections
        ),
        "tested_pair_side_family_count": len(
            {(item["pair"], item["side"]) for item in results}
        ),
        "tested_vehicle_count": len(
            {
                (
                    item["pair"],
                    item["side"],
                    item["take_profit_pips"],
                    item["stop_loss_pips"],
                )
                for item in results
            }
        ),
        "multiple_testing_caveat": (
            "all visible pair/side and TP/SL grids are searched; chronological validation and "
            "Student-t lower bounds reduce but do not eliminate family-wise selection risk"
        ),
        "candidate_rules": [item for item in selections if item.get("hypothesis_candidate") is True],
        "status": _report_status(selections),
    }
    write_report(report, args.output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-history", type=Path, default=Path("data/forecast_history.jsonl"))
    parser.add_argument("--history-dir", type=Path, action="append", default=None)
    parser.add_argument("--granularity", default="S5")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/reports/forecast_improvement/range_scout"))
    parser.add_argument("--auto-history-min-days", type=float, default=30.0)
    parser.add_argument("--ttl-minutes", type=int, default=DEFAULT_TTL_MINUTES)
    parser.add_argument("--dedupe-minutes", type=int, default=DEFAULT_DEDUPE_MINUTES)
    parser.add_argument("--tp-grid-pips", type=_parse_float_csv, default=DEFAULT_TP_GRID_PIPS)
    parser.add_argument("--sl-grid-pips", type=_parse_float_csv, default=DEFAULT_SL_GRID_PIPS)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--min-train-fills", type=int, default=DEFAULT_MIN_TRAIN_FILLS)
    parser.add_argument("--min-validation-fills", type=int, default=DEFAULT_MIN_VALIDATION_FILLS)
    parser.add_argument("--min-total-fills", type=int, default=DEFAULT_MIN_TOTAL_FILLS)
    args = parser.parse_args()
    args.granularity = str(args.granularity).upper()
    if args.granularity != "S5":
        parser.error("--granularity must be S5 for the canonical RANGE SCOUT bid/ask vehicle")
    if args.ttl_minutes <= 0 or args.ttl_minutes > DEFAULT_TTL_MINUTES:
        parser.error(f"--ttl-minutes must be in 1..{DEFAULT_TTL_MINUTES}")
    if args.dedupe_minutes < args.ttl_minutes:
        parser.error("--dedupe-minutes must be >= --ttl-minutes so one pair has no overlapping signal")
    if not 0.5 <= args.train_fraction < 1.0:
        parser.error("--train-fraction must be in [0.5, 1.0)")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in (*args.tp_grid_pips, *args.sl_grid_pips)
    ):
        parser.error("--tp-grid-pips and --sl-grid-pips must contain finite positive values")
    if min(args.min_train_fills, args.min_validation_fills, args.min_total_fills) <= 0:
        parser.error("minimum fill counts must be positive")
    if args.min_total_fills < args.min_train_fills + args.min_validation_fills:
        parser.error("--min-total-fills must cover train plus validation minimums")
    return args


def _report_status(selections: Sequence[dict[str, Any]]) -> str:
    powered = [item for item in selections if item.get("oos_powered") is True]
    if not powered:
        return "INSUFFICIENT_OOS_SAMPLE_UNDER_TTL_CLOSE_RESEARCH_CONTRACT"
    if any(item.get("hypothesis_candidate") is True for item in powered):
        return "RANGE_SCOUT_RESEARCH_HYPOTHESES_EXIT_CONTRACT_MISMATCH"
    return "NO_OOS_EDGE_UNDER_TTL_CLOSE_RESEARCH_CONTRACT"


def load_range_forecasts(
    path: Path,
    *,
    pairs: set[str] | None = None,
    dedupe_minutes: int = DEFAULT_DEDUPE_MINUTES,
) -> tuple[list[RangeForecastRow], dict[str, Any]]:
    pair_filter = {str(item).upper() for item in (pairs or set()) if str(item).strip()}
    parsed: list[RangeForecastRow] = []
    seen: set[tuple[Any, ...]] = set()
    raw_range_rows = 0
    skipped_duplicate = 0
    skipped_invalid = 0
    skipped_pair_filter = 0
    with path.open(encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue
            if str(payload.get("direction") or "").upper() != "RANGE":
                continue
            raw_range_rows += 1
            pair = str(payload.get("pair") or "").upper().strip()
            if pair_filter and pair not in pair_filter:
                skipped_pair_filter += 1
                continue
            timestamp = _parse_time(payload.get("timestamp_utc"))
            current = _safe_float(payload.get("current_price"))
            low = _safe_float(payload.get("range_low_price"))
            high = _safe_float(payload.get("range_high_price"))
            if timestamp is None or not pair or current is None or low is None or high is None or not low < high:
                skipped_invalid += 1
                continue
            confidence = _safe_float(payload.get("confidence")) or 0.0
            horizon = _safe_float(payload.get("horizon_min")) or 0.0
            if horizon <= 0:
                skipped_invalid += 1
                continue
            cycle_id = str(payload.get("cycle_id") or "").strip() or None
            key = (
                ("cycle", cycle_id, pair)
                if cycle_id
                else (
                    "bucket",
                    pair,
                    timestamp.replace(microsecond=0).isoformat(),
                    round(confidence, 6),
                    low,
                    high,
                )
            )
            if key in seen:
                skipped_duplicate += 1
                continue
            seen.add(key)
            parsed.append(
                RangeForecastRow(
                    source_index=source_index,
                    timestamp_utc=timestamp,
                    pair=pair,
                    confidence=confidence,
                    current_price=current,
                    range_low_price=low,
                    range_high_price=high,
                    horizon_min=horizon,
                    cycle_id=cycle_id,
                )
            )

    deduped: list[RangeForecastRow] = []
    last_by_pair: dict[str, datetime] = {}
    persistence_deduped = 0
    for row in sorted(parsed, key=lambda item: (item.timestamp_utc, item.pair, item.source_index)):
        if not row.range_low_price <= row.current_price <= row.range_high_price:
            continue
        last = last_by_pair.get(row.pair)
        if last is not None and row.timestamp_utc - last < timedelta(minutes=dedupe_minutes):
            persistence_deduped += 1
            continue
        last_by_pair[row.pair] = row.timestamp_utc
        deduped.append(row)
    return deduped, {
        "raw_range_rows": raw_range_rows,
        "valid_unique_range_rows": len(parsed),
        "deduped_range_rows": len(deduped),
        "skipped_duplicate_rows": skipped_duplicate,
        "skipped_invalid_rows": skipped_invalid,
        "skipped_pair_filter_rows": skipped_pair_filter,
        "skipped_persistence_rows": persistence_deduped,
        "pair_filter": sorted(pair_filter),
    }


def truth_end(row: RangeForecastRow, *, ttl_minutes: int) -> datetime:
    ttl = min(float(ttl_minutes), row.horizon_min)
    return row.timestamp_utc + timedelta(minutes=ttl)


def range_truth_is_no_market(row: RangeForecastRow, *, ttl_minutes: int) -> bool:
    return _is_likely_fx_no_market_window(
        replace(
            row,
            horizon_min=min(float(ttl_minutes), row.horizon_min),
        )
    )


def range_truth_windows(
    rows: Sequence[RangeForecastRow],
    *,
    ttl_minutes: int,
) -> dict[str, list[tuple[datetime, datetime]]]:
    pad = timedelta(minutes=1)
    by_pair: dict[str, list[tuple[datetime, datetime]]] = collections.defaultdict(list)
    for row in rows:
        by_pair[row.pair].append((row.timestamp_utc - pad, truth_end(row, ttl_minutes=ttl_minutes) + pad))
    return {pair: merge_windows(windows) for pair, windows in by_pair.items()}


def granularity_interval(granularity: str) -> timedelta:
    value = str(granularity or "").strip().upper()
    units = {"S": 1, "M": 60, "H": 60 * 60, "D": 24 * 60 * 60}
    prefix = value[:1]
    try:
        count = int(value[1:]) if prefix != "D" else int(value[1:] or "1")
    except ValueError as exc:
        raise ValueError(f"unsupported candle granularity: {granularity}") from exc
    if prefix not in units or count <= 0:
        raise ValueError(f"unsupported candle granularity: {granularity}")
    return timedelta(seconds=count * units[prefix])


def merge_windows(windows: Iterable[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def range_signal(row: RangeForecastRow) -> RangeSignal:
    pip = 1.0 / instrument_pip_factor(row.pair)
    width_pips = (row.range_high_price - row.range_low_price) / pip
    midpoint = (row.range_low_price + row.range_high_price) / 2.0
    side = "LONG" if row.current_price <= midpoint else "SHORT"
    entry = row.range_low_price if side == "LONG" else row.range_high_price
    box_position = (row.current_price - row.range_low_price) / (row.range_high_price - row.range_low_price)
    return RangeSignal(row=row, side=side, entry=entry, range_width_pips=width_pips, box_position=box_position)


def score_range_forecasts(
    rows: Sequence[RangeForecastRow],
    candles_by_pair: dict[str, list[QuoteCandle]],
    *,
    ttl_minutes: int,
    tp_grid: Sequence[float],
    sl_grid: Sequence[float],
    candle_interval: timedelta,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    eligible_signals = 0
    complete_truth_windows = 0
    filled_signals: set[tuple[int, str]] = set()
    skipped_no_window = 0
    skipped_incomplete_truth_window = 0
    skipped_not_passive = 0
    skipped_unfilled = 0
    ambiguous_fill_bar_target_cases = 0
    pair_side_truth_counts: dict[tuple[str, str], collections.Counter[str]] = (
        collections.defaultdict(collections.Counter)
    )
    pair_side_forecast_rows_by_day: dict[
        tuple[str, str], collections.Counter[str]
    ] = collections.defaultdict(collections.Counter)
    pair_side_complete_windows_by_day: dict[
        tuple[str, str], collections.Counter[str]
    ] = collections.defaultdict(collections.Counter)
    times_by_pair = {
        pair: [item.timestamp_utc for item in candles]
        for pair, candles in candles_by_pair.items()
    }
    for row in rows:
        signal = range_signal(row)
        family = (row.pair, signal.side)
        forecast_day = row.timestamp_utc.date().isoformat()
        truth_counts = pair_side_truth_counts[family]
        truth_counts["forecast_rows"] += 1
        pair_side_forecast_rows_by_day[family][forecast_day] += 1
        candles = candles_by_pair.get(row.pair) or []
        if not candles:
            skipped_no_window += 1
            continue
        times = times_by_pair[row.pair]
        start = bisect.bisect_left(times, row.timestamp_utc)
        # OANDA candle timestamps mark interval start.  Requiring the candle's
        # full interval to end by the truth boundary prevents its high/low or
        # close from reading price action after forecast/TTL expiry.
        last_complete_start = truth_end(row, ttl_minutes=ttl_minutes) - candle_interval
        end = bisect.bisect_right(times, last_complete_start)
        if start >= len(candles) or end <= start:
            skipped_no_window += 1
            continue
        window_candles = candles[start:end]
        if not complete_truth_window(
            window_candles,
            forecast_start=row.timestamp_utc,
            forecast_end=truth_end(row, ttl_minutes=ttl_minutes),
            candle_interval=candle_interval,
        ):
            skipped_incomplete_truth_window += 1
            continue
        complete_truth_windows += 1
        truth_counts["complete_truth_windows"] += 1
        pair_side_complete_windows_by_day[family][forecast_day] += 1
        pip = 1.0 / instrument_pip_factor(row.pair)
        first = candles[start]
        if signal.side == "LONG" and not signal.entry < first.ask.o:
            skipped_not_passive += 1
            continue
        if signal.side == "SHORT" and not signal.entry > first.bid.o:
            skipped_not_passive += 1
            continue
        eligible_signals += 1
        fill_index = next(
            (
                idx
                for idx in range(start, end)
                if (signal.side == "LONG" and candles[idx].ask.l <= signal.entry)
                or (signal.side == "SHORT" and candles[idx].bid.h >= signal.entry)
            ),
            None,
        )
        if fill_index is None:
            skipped_unfilled += 1
            continue
        filled_signals.add((row.source_index, signal.side))
        for take_profit_pips in tp_grid:
            # A rail reversion TP must remain at or before the forecast midpoint.
            if take_profit_pips > signal.range_width_pips / 2.0 + 1e-9:
                continue
            for stop_loss_pips in sl_grid:
                outcome = simulate_filled_signal(
                    signal,
                    candles[fill_index:end],
                    take_profit_pips=float(take_profit_pips),
                    stop_loss_pips=float(stop_loss_pips),
                    pip=pip,
                    candle_interval=candle_interval,
                )
                if outcome.get("fill_bar_target_ambiguous") is True:
                    ambiguous_fill_bar_target_cases += 1
                results.append(
                    {
                        "source_index": row.source_index,
                        "timestamp_utc": _iso(row.timestamp_utc),
                        "campaign_day_jst": row.timestamp_utc.astimezone(JST).date().isoformat(),
                        "granularity": "S5",
                        "pair": row.pair,
                        "side": signal.side,
                        "confidence": round(row.confidence, 6),
                        "confidence_bucket": confidence_bucket(row.confidence),
                        "range_width_pips": round(signal.range_width_pips, 6),
                        "range_width_bucket": range_width_bucket(signal.range_width_pips),
                        "box_position": round(signal.box_position, 6),
                        "take_profit_pips": float(take_profit_pips),
                        "stop_loss_pips": float(stop_loss_pips),
                        **outcome,
                    }
                )
    return results, {
        "eligible_signals": eligible_signals,
        "filled_signals": len(filled_signals),
        "fill_rate": round(len(filled_signals) / eligible_signals, 6) if eligible_signals else 0.0,
        "skipped_no_price_window": skipped_no_window,
        "skipped_incomplete_truth_window": skipped_incomplete_truth_window,
        "skipped_not_passive": skipped_not_passive,
        "skipped_unfilled": skipped_unfilled,
        "ambiguous_fill_bar_target_cases": ambiguous_fill_bar_target_cases,
        "complete_truth_windows": complete_truth_windows,
        "truth_window_coverage_rate": round(
            complete_truth_windows
            / (complete_truth_windows + skipped_no_window + skipped_incomplete_truth_window),
            6,
        )
        if complete_truth_windows + skipped_no_window + skipped_incomplete_truth_window
        else 0.0,
        "pair_side_truth_coverage": [
            {
                "pair": pair,
                "side": side,
                "forecast_rows": int(counts["forecast_rows"]),
                "complete_truth_windows": int(counts["complete_truth_windows"]),
                "coverage_rate": round(
                    counts["complete_truth_windows"] / counts["forecast_rows"],
                    6,
                )
                if counts["forecast_rows"]
                else 0.0,
                "forecast_rows_by_day": dict(
                    sorted(pair_side_forecast_rows_by_day[(pair, side)].items())
                ),
                "complete_truth_windows_by_day": dict(
                    sorted(pair_side_complete_windows_by_day[(pair, side)].items())
                ),
            }
            for (pair, side), counts in sorted(pair_side_truth_counts.items())
        ],
    }


def complete_truth_window(
    candles: Sequence[QuoteCandle],
    *,
    forecast_start: datetime,
    forecast_end: datetime,
    candle_interval: timedelta,
) -> bool:
    if not candles:
        return False
    if candles[0].timestamp_utc - forecast_start >= candle_interval:
        return False
    if forecast_end - (candles[-1].timestamp_utc + candle_interval) >= candle_interval:
        return False
    return all(
        current.timestamp_utc - previous.timestamp_utc <= candle_interval
        for previous, current in zip(candles, candles[1:])
    )


def simulate_filled_signal(
    signal: RangeSignal,
    candles: Sequence[QuoteCandle],
    *,
    take_profit_pips: float,
    stop_loss_pips: float,
    pip: float,
    candle_interval: timedelta = timedelta(seconds=5),
) -> dict[str, Any]:
    if (
        not math.isfinite(float(take_profit_pips))
        or not math.isfinite(float(stop_loss_pips))
        or take_profit_pips <= 0.0
        or stop_loss_pips <= 0.0
    ):
        raise ValueError("RANGE SCOUT TP/SL distances must be finite and positive")
    fill_bar_target_ambiguous = False
    for index, candle in enumerate(candles):
        if signal.side == "LONG":
            target_price = signal.entry + take_profit_pips * pip
            stop_hit = candle.bid.l <= signal.entry - stop_loss_pips * pip
            target_hit = candle.bid.h >= target_price
            close_proves_post_fill_target = candle.bid.c >= target_price
        else:
            target_price = signal.entry - take_profit_pips * pip
            stop_hit = candle.ask.h >= signal.entry + stop_loss_pips * pip
            target_hit = candle.ask.l <= target_price
            close_proves_post_fill_target = candle.ask.c <= target_price
        if stop_hit:
            exit_at = candle.timestamp_utc + candle_interval
            return {
                "realized_pips": -stop_loss_pips,
                "exit_reason": "STOP_LOSS",
                "same_candle_stop_first": target_hit,
                "scorable": True,
                "fill_bar_target_ambiguous": (
                    fill_bar_target_ambiguous or bool(index == 0 and target_hit)
                ),
                "fill_at_utc": _iso(candles[0].timestamp_utc),
                "exit_at_utc": _iso(exit_at),
                "resolved_day_utc": exit_at.date().isoformat(),
            }
        # The fill and target may have occurred in the opposite order inside
        # the same S5 OHLC bar.  If the executable close remains beyond target,
        # price necessarily crossed the target after the passive fill and the
        # TP is proved.  Otherwise ignore that first target and continue from
        # the next complete bar; this scores the adverse target-before-fill
        # ordering instead of deleting the sample from the evidence set.
        if target_hit and index == 0:
            fill_bar_target_ambiguous = True
            if not close_proves_post_fill_target:
                continue
        if target_hit:
            exit_at = candle.timestamp_utc + candle_interval
            return {
                "realized_pips": take_profit_pips,
                "exit_reason": "TAKE_PROFIT",
                "same_candle_stop_first": False,
                "scorable": True,
                "fill_bar_target_ambiguous": fill_bar_target_ambiguous,
                "fill_at_utc": _iso(candles[0].timestamp_utc),
                "exit_at_utc": _iso(exit_at),
                "resolved_day_utc": exit_at.date().isoformat(),
            }
    last = candles[-1]
    executable_close = last.bid.c if signal.side == "LONG" else last.ask.c
    realized = (
        (executable_close - signal.entry) / pip
        if signal.side == "LONG"
        else (signal.entry - executable_close) / pip
    )
    exit_at = last.timestamp_utc + candle_interval
    return {
        "realized_pips": round(realized, 6),
        "exit_reason": "TTL_CLOSE",
        "same_candle_stop_first": False,
        "scorable": True,
        "fill_bar_target_ambiguous": fill_bar_target_ambiguous,
        "fill_at_utc": _iso(candles[0].timestamp_utc),
        "exit_at_utc": _iso(exit_at),
        "resolved_day_utc": exit_at.date().isoformat(),
    }


def group_metrics(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for item in results:
        key = (item["pair"], item["side"], item["take_profit_pips"], item["stop_loss_pips"])
        grouped[key].append(item)
    rows = []
    for (pair, side, tp, sl), items in grouped.items():
        rows.append({"pair": pair, "side": side, "take_profit_pips": tp, "stop_loss_pips": sl, **metrics(items)})
    rows.sort(
        key=lambda item: (
            -float(item["mean_pips"] if item.get("mean_pips") is not None else -999.0),
            -int(item["fills"]),
            item["pair"],
            item["side"],
        )
    )
    return rows


def train_validation_selections(
    results: Sequence[dict[str, Any]],
    *,
    train_fraction: float,
    min_train_fills: int,
    min_validation_fills: int,
    min_total_fills: int,
    family_truth_coverage: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_vehicle: dict[tuple[str, str, float, float], list[dict[str, Any]]] = collections.defaultdict(list)
    for item in results:
        by_vehicle[(item["pair"], item["side"], item["take_profit_pips"], item["stop_loss_pips"])].append(item)
    by_family: dict[tuple[str, str], list[tuple[tuple[str, str, float, float], list[dict[str, Any]]]]] = collections.defaultdict(list)
    for key, items in by_vehicle.items():
        items.sort(key=lambda item: (item["timestamp_utc"], item["source_index"]))
        by_family[(key[0], key[1])].append((key, items))

    selections: list[dict[str, Any]] = []
    for family, vehicles in sorted(by_family.items()):
        result_days = {
            str(item["timestamp_utc"])[:10]
            for _key, items in vehicles
            for item in items
        }
        if family_truth_coverage is None:
            coverage = {
                "pair": family[0],
                "side": family[1],
                "forecast_rows": len(result_days),
                "complete_truth_windows": len(result_days),
                "coverage_rate": 1.0,
                "forecast_rows_by_day": {day: 1 for day in result_days},
                "complete_truth_windows_by_day": {day: 1 for day in result_days},
            }
        else:
            coverage = dict((family_truth_coverage or {}).get(family) or {})
        coverage_days = {
            str(day)
            for day, count in (
                coverage.get("forecast_rows_by_day") or {}
            ).items()
            if int(count or 0) > 0
        }
        family_days = sorted(coverage_days or result_days)
        if len(family_days) < 2:
            continue
        day_cut = max(
            1,
            min(len(family_days) - 1, int(len(family_days) * train_fraction)),
        )
        train_days = set(family_days[:day_cut])
        validation_days = set(family_days[day_cut:])
        train_truth_coverage = _truth_coverage_for_days(coverage, train_days)
        validation_truth_coverage = _truth_coverage_for_days(
            coverage,
            validation_days,
        )
        candidates: list[
            tuple[
                float,
                float,
                int,
                tuple[str, str, float, float],
                list[dict[str, Any]],
            ]
        ] = []
        for key, items in vehicles:
            train = [
                item
                for item in items
                if str(item["timestamp_utc"])[:10] in train_days
            ]
            train_metrics = metrics(train)
            if len(train) < min_train_fills:
                continue
            candidates.append(
                (
                    float(train_metrics["mean_pips"]),
                    _profit_factor_value(train_metrics.get("profit_factor")),
                    len(train),
                    key,
                    items,
                )
            )
        if not candidates:
            continue
        _mean, _pf, _n, key, items = max(candidates)
        train = [
            item
            for item in items
            if str(item["timestamp_utc"])[:10] in train_days
        ]
        validation = [
            item
            for item in items
            if str(item["timestamp_utc"])[:10] not in train_days
        ]
        full_metrics = metrics(items)
        train_metrics = metrics(train)
        validation_metrics = metrics(validation)
        truth_coverage_rate = float(coverage.get("coverage_rate") or 0.0)
        oos_powered = bool(
            len(items) >= min_total_fills
            and len(train) >= min_train_fills
            and len(validation) >= min_validation_fills
            and int(train_metrics["active_days"]) >= 5
            and int(validation_metrics["active_days"]) >= 5
            and truth_coverage_rate >= MIN_TRUTH_WINDOW_COVERAGE_RATE
            and train_truth_coverage["coverage_rate"]
            >= MIN_TRUTH_WINDOW_COVERAGE_RATE
            and validation_truth_coverage["coverage_rate"]
            >= MIN_TRUTH_WINDOW_COVERAGE_RATE
        )
        hypothesis_candidate = bool(
            oos_powered
            and float(train_metrics["mean_pips"]) > 0.0
            and _profit_factor_value(train_metrics.get("profit_factor")) >= 1.2
            and float(validation_metrics["mean_pips"]) > 0.0
            and _profit_factor_value(validation_metrics.get("profit_factor")) >= 1.2
            and float(train_metrics.get("one_sided_95_mean_lower_pips") or 0.0) > 0.0
            and float(validation_metrics.get("one_sided_95_mean_lower_pips") or 0.0) > 0.0
            and float(full_metrics.get("one_sided_95_mean_lower_pips") or 0.0) > 0.0
            and float(train_metrics.get("one_sided_95_daily_mean_lower_pips") or 0.0) > 0.0
            and float(validation_metrics.get("one_sided_95_daily_mean_lower_pips") or 0.0) > 0.0
            and float(full_metrics.get("one_sided_95_daily_mean_lower_pips") or 0.0) > 0.0
            and int(train_metrics["active_days"]) >= 5
            and int(validation_metrics["active_days"]) >= 5
            and int(full_metrics["active_days"]) >= 5
            and float(train_metrics["positive_day_rate"]) >= 2.0 / 3.0
            and float(validation_metrics["positive_day_rate"]) >= 2.0 / 3.0
            and float(full_metrics["positive_day_rate"]) >= 2.0 / 3.0
            and float(train_metrics["max_daily_sample_share"]) <= 0.70
            and float(validation_metrics["max_daily_sample_share"]) <= 0.70
            and float(full_metrics["max_daily_sample_share"]) <= 0.70
        )
        selections.append(
            {
                "pair": family[0],
                "side": family[1],
                "take_profit_pips": key[2],
                "stop_loss_pips": key[3],
                "train": train_metrics,
                "validation": validation_metrics,
                "full": full_metrics,
                "truth_coverage": {
                    **coverage,
                    "train": train_truth_coverage,
                    "validation": validation_truth_coverage,
                },
                "minimum_truth_window_coverage_rate": MIN_TRUTH_WINDOW_COVERAGE_RATE,
                "oos_powered": oos_powered,
                "hypothesis_candidate": hypothesis_candidate,
                "live_permission_allowed": False,
            }
        )
    return selections


def _truth_coverage_for_days(
    coverage: dict[str, Any],
    days: set[str],
) -> dict[str, Any]:
    forecast_by_day = coverage.get("forecast_rows_by_day") or {}
    complete_by_day = coverage.get("complete_truth_windows_by_day") or {}
    forecast_rows = sum(int(forecast_by_day.get(day) or 0) for day in days)
    complete_windows = sum(int(complete_by_day.get(day) or 0) for day in days)
    return {
        "days": sorted(days),
        "forecast_rows": forecast_rows,
        "complete_truth_windows": complete_windows,
        "coverage_rate": round(complete_windows / forecast_rows, 6)
        if forecast_rows
        else 0.0,
    }


def metrics(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "fills": 0,
            "mean_pips": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "active_days": 0,
            "positive_day_rate": 0.0,
            "max_daily_sample_share": 0.0,
            "one_sided_95_mean_lower_pips": None,
            "one_sided_95_daily_mean_lower_pips": None,
            "take_profit_rate": 0.0,
            "stop_loss_rate": 0.0,
            "ttl_close_rate": 0.0,
        }
    pnls = [float(item["realized_pips"]) for item in items]
    gross_profit = sum(value for value in pnls if value > 0.0)
    gross_loss = -sum(value for value in pnls if value < 0.0)
    by_day: dict[str, float] = collections.defaultdict(float)
    day_counts: collections.Counter[str] = collections.Counter()
    for item in items:
        day = str(
            item.get("resolved_day_utc")
            or str(item.get("timestamp_utc") or "")[:10]
            or item.get("campaign_day_jst")
            or ""
        )
        if not day:
            continue
        by_day[day] += float(item["realized_pips"])
        day_counts[day] += 1
    mean = statistics.mean(pnls)
    lower = None
    if len(pnls) >= 2:
        lower = mean - _one_sided_t95_critical(len(pnls) - 1) * statistics.stdev(
            pnls
        ) / math.sqrt(len(pnls))
    daily_pnls = list(by_day.values())
    daily_lower = None
    if len(daily_pnls) >= 2:
        daily_lower = statistics.mean(daily_pnls) - _one_sided_t95_critical(
            len(daily_pnls) - 1
        ) * statistics.stdev(daily_pnls) / math.sqrt(len(daily_pnls))
    return {
        "fills": len(items),
        "mean_pips": round(mean, 6),
        "profit_factor": round(gross_profit / gross_loss, 6) if gross_loss > 0.0 else None,
        "win_rate": round(sum(value > 0.0 for value in pnls) / len(pnls), 6),
        "active_days": len(by_day),
        "positive_day_rate": round(
            sum(value > 0.0 for value in by_day.values()) / len(by_day),
            6,
        )
        if by_day
        else 0.0,
        "max_daily_sample_share": round(max(day_counts.values()) / len(items), 6)
        if day_counts
        else 0.0,
        "one_sided_95_mean_lower_pips": round(lower, 6) if lower is not None else None,
        "one_sided_95_daily_mean_lower_pips": (
            round(daily_lower, 6) if daily_lower is not None else None
        ),
        "take_profit_rate": round(sum(item["exit_reason"] == "TAKE_PROFIT" for item in items) / len(items), 6),
        "stop_loss_rate": round(sum(item["exit_reason"] == "STOP_LOSS" for item in items) / len(items), 6),
        "ttl_close_rate": round(sum(item["exit_reason"] == "TTL_CLOSE" for item in items) / len(items), 6),
    }


def _one_sided_t95_critical(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        return math.inf
    if degrees_of_freedom < len(ONE_SIDED_T95_BY_DF):
        return ONE_SIDED_T95_BY_DF[degrees_of_freedom]
    z = statistics.NormalDist().inv_cdf(0.95)
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def _profit_factor_value(value: Any) -> float:
    """Treat a no-loss slice as unbounded PF without writing Infinity to JSON."""

    return math.inf if value is None else float(value)


def confidence_bucket(confidence: float) -> str:
    if confidence < 0.65:
        return "<0.65"
    if confidence < 0.75:
        return "0.65-0.75"
    if confidence < 0.90:
        return "0.75-0.90"
    return ">=0.90"


def range_width_bucket(width_pips: float) -> str:
    if width_pips < 10.0:
        return "<10"
    if width_pips < 20.0:
        return "10-20"
    if width_pips < 40.0:
        return "20-40"
    return ">=40"


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"oanda_range_scout_replay_validate_{run_ts}.json"
    markdown_path = output_dir / f"oanda_range_scout_replay_validate_{run_ts}.md"
    latest_json = output_dir / "oanda_range_scout_replay_validate_latest.json"
    latest_markdown = output_dir / "oanda_range_scout_replay_validate_latest.md"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    json_path.write_text(encoded, encoding="utf-8")
    latest_json.write_text(encoded, encoding="utf-8")
    markdown = markdown_report(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    latest_markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": report.get("status"),
        "candidate_rules": len(report.get("candidate_rules") or []),
        "filled_signals": report.get("filled_signals"),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }, ensure_ascii=False, indent=2, sort_keys=True))


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# OANDA RANGE SCOUT Replay Validation",
        "",
        f"- Generated at UTC: `{report.get('generated_at_utc')}`",
        f"- Status: `{report.get('status')}`",
        f"- RANGE rows: raw={report.get('raw_range_rows')} deduped={report.get('deduped_range_rows')}",
        f"- Eligible signals: {report.get('eligible_signals')}",
        f"- Filled signals: {report.get('filled_signals')} (rate={report.get('fill_rate')})",
        f"- Candidate rules: {len(report.get('candidate_rules') or [])}",
        "- Live permission: `false`",
        f"- Adoption blockers: {', '.join(f'`{item}`' for item in report.get('adoption_blockers') or [])}",
        f"- Complete truth windows: {report.get('complete_truth_windows')} (coverage={report.get('truth_window_coverage_rate')})",
        f"- Ordering-ambiguous fill-bar target cases conservatively continued: {report.get('ambiguous_fill_bar_target_cases')}",
        "",
        "## Chronological Train / Validation",
        "",
        "| Pair | Side | TP | SL | Train PF | Validation PF | Validation mean | Candidate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in report.get("train_validation_selections") or []:
        train = item.get("train") or {}
        validation = item.get("validation") or {}
        lines.append(
            f"| {item.get('pair')} | {item.get('side')} | {item.get('take_profit_pips')} | "
            f"{item.get('stop_loss_pips')} | {train.get('profit_factor')} | "
            f"{validation.get('profit_factor')} | {validation.get('mean_pips')} | "
            f"{item.get('hypothesis_candidate')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Forecast persistence is deduplicated per pair for the full pending TTL.",
        "- Spread is included through executable bid/ask fills and exits.",
        "- Only candles whose full interval ends by forecast expiry are read.",
        "- Observation starts at the first complete S5 boundary at/after forecast emission; no pre-forecast OHLC range is consumed.",
        "- The best train grid is judged on the later chronological validation slice.",
        f"- Pair/side truth-window coverage must be at least {MIN_TRUTH_WINDOW_COVERAGE_RATE:.0%} before an OOS family is called powered.",
        "- Both trade-level and UTC resolved-day Student-t lower bounds must remain above zero for a research candidate.",
        "- Replay TTL_CLOSE after fill does not match current live SCOUT, whose GTD expires only unfilled entries; adoption is blocked until the exit contracts match.",
        "- A candidate is only a bounded forward hypothesis; it is not positive-expectancy proof or live permission.",
        "",
    ])
    return "\n".join(lines)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

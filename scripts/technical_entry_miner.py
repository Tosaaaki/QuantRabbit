#!/usr/bin/env python3
"""Mine technical conditions that actually led to forecast payoff.

The directional forecast can be right about a brief push but still lose money
when invalidation touches before target. This audit recomputes a broad
technical stack from candles available before each forecast emission and ranks
technical buckets by:

- target before invalidation
- target touch
- final direction
- MFE >= 2 pips

The output is evidence only. It does not grant live trading permission.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from quant_rabbit.analysis.candles import Candle, fetch_candles_between
from quant_rabbit.analysis.families import compute_family_scores
from quant_rabbit.analysis.indicators import IndicatorSet, compute_indicators
from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.instruments import instrument_pip_factor


DIRECTIONAL = {"UP", "DOWN"}
TIMEFRAMES = ("M1", "M5", "M15")

# These are audit-label boundaries, not production trade thresholds. They map
# continuous indicators into coarse categories so the miner can count samples.
# Runtime code must revalidate any mined rule against fresh broker/price truth.
RSI_LOW = 35.0
RSI_HIGH = 65.0
STOCH_LOW = 25.0
STOCH_HIGH = 75.0
ADX_TRENDING = 25.0
BB_EDGE_PCT_B = 0.20
ATR_QUIET_PCTL = 0.25
ATR_HOT_PCTL = 0.75
CHOP_RANGE = 61.8
CHOP_TREND = 38.2


@dataclass(frozen=True)
class ForecastRow:
    source_index: int
    timestamp: datetime
    pair: str
    direction: str
    confidence: float | None
    entry: float | None
    target: float | None
    invalidation: float | None
    horizon_min: float
    cycle_id: str | None


@dataclass(frozen=True)
class PairCandleContext:
    m1: list[Candle]
    m1_timestamps: list[datetime]


def main() -> int:
    args = _parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_out = out_dir / f"technical_entry_mining_{run_ts}.json"
    md_out = out_dir / f"technical_entry_mining_{run_ts}.md"
    latest_json = out_dir / "technical_entry_mining_latest.json"
    latest_md = out_dir / "technical_entry_mining_latest.md"

    pairs = _parse_pair_filter(args.pairs)
    rows, load_stats = _load_rows(
        args.data_root / "forecast_history.jsonl",
        pairs=pairs,
        max_rows=args.max_rows,
    )
    candles_by_pair, fetch_stats = _fetch_m1_context(
        rows,
        lookback_minutes=args.lookback_minutes,
        max_chunk_days=args.max_chunk_days,
    )
    scored_rows, score_stats = _score_rows(
        rows,
        candles_by_pair,
        indicator_bars=args.indicator_bars,
        scalp_tp_pips=args.scalp_tp_pips,
        scalp_stop_pips=args.scalp_stop_pips,
    )
    feature_rows = _mine_feature_buckets(
        scored_rows,
        min_samples=args.min_samples,
        min_touch_samples=args.min_touch_samples,
    )
    strict = [
        row
        for row in feature_rows
        if row["touch_n"] >= args.min_touch_samples
        and (row.get("target_first_wilson95_lower") or 0.0) >= args.min_wilson_lower
    ]
    positive = [
        row
        for row in feature_rows
        if row["n"] >= args.min_samples
        and (row.get("mfe_ge_2pip_wilson95_lower") or 0.0) >= args.min_wilson_lower
    ]
    scalp = [
        row
        for row in feature_rows
        if row["n"] >= args.min_samples
        and (row.get("scalp_tp_first_wilson95_lower") or 0.0) >= args.min_wilson_lower
    ]
    negative = [
        row
        for row in feature_rows
        if row["n"] >= args.min_samples
        and (row.get("avg_final_pips") or 0.0) < 0.0
        and (row.get("final_wilson95_lower") or 0.0) < 0.45
    ]

    report = {
        "generated_at_utc": _iso(datetime.now(timezone.utc)),
        "source": str(args.data_root / "forecast_history.jsonl"),
        "truth_source": (
            "OANDA M1 mid candles; features are computed from candles ending "
            "before each forecast truth window"
        ),
        "timeframes": list(TIMEFRAMES),
        "lookback_minutes": args.lookback_minutes,
        "scalp_tp_pips": args.scalp_tp_pips,
        "scalp_stop_pips": args.scalp_stop_pips,
        **load_stats,
        **fetch_stats,
        **score_stats,
        "summary": _summarize(scored_rows),
        "strict_target_first_candidates": strict[:50],
        "strict_scalp_candidates": scalp[:50],
        "positive_mfe2_candidates": positive[:50],
        "negative_buckets": negative[:50],
        "top_feature_buckets": feature_rows[:120],
    }
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    latest_json.write_text(json_out.read_text(encoding="utf-8"), encoding="utf-8")
    md_out.write_text(_markdown(report), encoding="utf-8")
    latest_md.write_text(md_out.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/reports/forecast_improvement"))
    parser.add_argument("--lookback-minutes", type=int, default=900)
    parser.add_argument("--max-chunk-days", type=float, default=3.0)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-touch-samples", type=int, default=30)
    parser.add_argument("--min-wilson-lower", type=float, default=0.90)
    parser.add_argument(
        "--indicator-bars",
        type=int,
        default=90,
        help="most recent bars per timeframe used for indicator calculation",
    )
    parser.add_argument(
        "--scalp-tp-pips",
        type=float,
        default=5.0,
        help="audit take-profit width for execution-cost-safe harvest entries",
    )
    parser.add_argument(
        "--scalp-stop-pips",
        type=float,
        default=4.0,
        help="audit stop width for execution-cost-safe harvest entries",
    )
    parser.add_argument(
        "--pairs",
        default="",
        help="comma-separated pair filter, e.g. AUD_JPY,EUR_USD",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="score only the most recent N deduped directional rows after filtering",
    )
    return parser.parse_args()


def _parse_pair_filter(value: str) -> set[str] | None:
    pairs = {item.strip().upper() for item in str(value or "").split(",") if item.strip()}
    return pairs or None


def _load_rows(
    path: Path,
    *,
    pairs: set[str] | None,
    max_rows: int,
) -> tuple[list[ForecastRow], dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    rows: list[ForecastRow] = []
    seen: set[tuple[Any, ...]] = set()
    raw_directional = 0
    skipped_invalid = 0
    skipped_duplicate = 0
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue
            direction = str(payload.get("direction") or "").upper()
            if direction not in DIRECTIONAL:
                continue
            raw_directional += 1
            timestamp = _parse_time(payload.get("timestamp_utc"))
            pair = str(payload.get("pair") or "").upper().strip()
            if timestamp is None or not pair:
                skipped_invalid += 1
                continue
            if pairs is not None and pair not in pairs:
                continue
            horizon = _safe_horizon(payload.get("horizon_min"))
            if timestamp + timedelta(minutes=horizon) > now_utc - timedelta(minutes=2):
                skipped_invalid += 1
                continue
            confidence = _safe_float(payload.get("confidence"))
            target = _safe_float(payload.get("target_price"))
            invalidation = _safe_float(payload.get("invalidation_price"))
            cycle_id = str(payload.get("cycle_id") or "").strip() or None
            key = _dedupe_key(
                cycle_id=cycle_id,
                pair=pair,
                timestamp=timestamp,
                direction=direction,
                confidence=confidence,
                target=target,
                invalidation=invalidation,
            )
            if key in seen:
                skipped_duplicate += 1
                continue
            seen.add(key)
            entry = _safe_float(payload.get("current_price"))
            if entry is not None and entry <= 0:
                entry = None
            rows.append(
                ForecastRow(
                    source_index=idx,
                    timestamp=timestamp,
                    pair=pair,
                    direction=direction,
                    confidence=confidence,
                    entry=entry,
                    target=target,
                    invalidation=invalidation,
                    horizon_min=horizon,
                    cycle_id=cycle_id,
                )
            )
    if max_rows > 0 and len(rows) > max_rows:
        rows = rows[-max_rows:]
    return rows, {
        "raw_directional_rows": raw_directional,
        "deduped_directional_rows": len(rows),
        "pair_filter": sorted(pairs) if pairs else None,
        "max_rows": max_rows if max_rows > 0 else None,
        "skipped_invalid_rows": skipped_invalid,
        "skipped_duplicate_rows": skipped_duplicate,
    }


def _fetch_m1_context(
    rows: list[ForecastRow],
    *,
    lookback_minutes: int,
    max_chunk_days: float,
) -> tuple[dict[str, list[Candle]], dict[str, Any]]:
    windows = _merged_windows(rows, lookback_minutes=lookback_minutes)
    client = OandaReadOnlyClient()
    candles: dict[str, list[Candle]] = collections.defaultdict(list)
    errors: list[dict[str, str]] = []
    for pair in sorted(windows):
        for start, end in windows[pair]:
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(end, chunk_start + timedelta(days=max_chunk_days))
                for attempt in range(3):
                    try:
                        candles[pair].extend(
                            fetch_candles_between(
                                pair,
                                "M1",
                                time_from=chunk_start,
                                time_to=chunk_end,
                                client=client,
                            )
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 - evidence plumbing report.
                        if attempt == 2:
                            errors.append(
                                {
                                    "pair": pair,
                                    "from": _iso(chunk_start),
                                    "to": _iso(chunk_end),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                        else:
                            time.sleep(0.8 * (attempt + 1))
                chunk_start = chunk_end
    deduped: dict[str, list[Candle]] = {}
    for pair, values in candles.items():
        by_ts = {c.timestamp_utc.astimezone(timezone.utc): c for c in values}
        deduped[pair] = [by_ts[key] for key in sorted(by_ts)]
    return deduped, {
        "fetch_pairs": len(windows),
        "fetch_windows": sum(len(items) for items in windows.values()),
        "fetched_candles": sum(len(items) for items in deduped.values()),
        "fetch_errors": errors[:50],
    }


def _merged_windows(
    rows: list[ForecastRow],
    *,
    lookback_minutes: int,
) -> dict[str, list[tuple[datetime, datetime]]]:
    intervals: dict[str, list[tuple[datetime, datetime]]] = collections.defaultdict(list)
    for row in rows:
        truth_start = _ceil_minute(row.timestamp)
        intervals[row.pair].append(
            (
                truth_start - timedelta(minutes=lookback_minutes),
                row.timestamp + timedelta(minutes=row.horizon_min) + timedelta(minutes=1),
            )
        )
    merged: dict[str, list[tuple[datetime, datetime]]] = {}
    for pair, values in intervals.items():
        values.sort()
        out: list[tuple[datetime, datetime]] = []
        for start, end in values:
            if not out or start > out[-1][1] + timedelta(minutes=2):
                out.append((start, end))
            else:
                out[-1] = (out[-1][0], max(out[-1][1], end))
        merged[pair] = out
    return merged


def _score_rows(
    rows: list[ForecastRow],
    candles_by_pair: dict[str, list[Candle]],
    *,
    indicator_bars: int,
    scalp_tp_pips: float,
    scalp_stop_pips: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    no_context = 0
    no_truth = 0
    pair_contexts = _prepare_pair_contexts(candles_by_pair)
    for row in rows:
        truth_start = _ceil_minute(row.timestamp)
        pair_context = pair_contexts.get(row.pair)
        if pair_context is None:
            no_context += 1
            continue
        context_end = bisect_left(pair_context.m1_timestamps, truth_start)
        truth_start_idx = bisect_left(pair_context.m1_timestamps, truth_start)
        truth_end_idx = bisect_right(
            pair_context.m1_timestamps,
            row.timestamp + timedelta(minutes=row.horizon_min),
        )
        if context_end < 80:
            no_context += 1
            continue
        truth = pair_context.m1[truth_start_idx:truth_end_idx]
        if not truth:
            no_truth += 1
            continue
        context_tail = _indicator_context_tail(
            pair_context.m1,
            end_index=context_end,
            indicator_bars=indicator_bars,
        )
        tf_candles = _timeframe_candles(context_tail)
        indicators_by_tf: dict[str, IndicatorSet] = {}
        families_by_tf: dict[str, Any] = {}
        for tf, candles in tf_candles.items():
            if len(candles) < 30:
                continue
            indicators = compute_indicators(row.pair, tf, candles[-max(30, indicator_bars):])
            indicators_by_tf[tf] = indicators
            families_by_tf[tf] = compute_family_scores(indicators)
        if "M1" not in indicators_by_tf:
            no_context += 1
            continue
        entry = row.entry if row.entry is not None and row.entry > 0 else truth[0].open
        scored.append(
            _score_one(
                row,
                truth,
                entry=entry,
                entry_source="forecast_current_price" if row.entry else "next_full_m1_open_proxy",
                indicators_by_tf=indicators_by_tf,
                families_by_tf=families_by_tf,
                scalp_tp_pips=scalp_tp_pips,
                scalp_stop_pips=scalp_stop_pips,
            )
        )
    return scored, {
        "scored_rows": len(scored),
        "skipped_no_context_rows": no_context,
        "skipped_no_truth_rows": no_truth,
    }


def _prepare_pair_contexts(candles_by_pair: dict[str, list[Candle]]) -> dict[str, PairCandleContext]:
    contexts: dict[str, PairCandleContext] = {}
    for pair, candles in candles_by_pair.items():
        by_ts = {
            candle.timestamp_utc.astimezone(timezone.utc): candle
            for candle in candles
            if candle.complete
        }
        timestamps = sorted(by_ts)
        m1 = [by_ts[ts] for ts in timestamps]
        contexts[pair] = PairCandleContext(m1=m1, m1_timestamps=timestamps)
    return contexts


def _indicator_context_tail(
    candles: Sequence[Candle],
    *,
    end_index: int,
    indicator_bars: int,
) -> Sequence[Candle]:
    # M15 needs the widest M1 slice. Resampling only this tail avoids repeatedly
    # rebuilding every historical bucket for each forecast row.
    max_tf_minutes = 15
    required_m1 = max(120, indicator_bars * max_tf_minutes + max_tf_minutes)
    start_index = max(0, end_index - required_m1)
    return candles[start_index:end_index]


def _score_one(
    row: ForecastRow,
    truth: list[Candle],
    *,
    entry: float,
    entry_source: str,
    indicators_by_tf: dict[str, IndicatorSet],
    families_by_tf: dict[str, Any],
    scalp_tp_pips: float,
    scalp_stop_pips: float,
) -> dict[str, Any]:
    factor = instrument_pip_factor(row.pair)
    final_close = truth[-1].close
    if row.direction == "UP":
        final_pips = (final_close - entry) * factor
        mfe_pips = (max(c.high for c in truth) - entry) * factor
        mae_pips = max(0.0, (entry - min(c.low for c in truth)) * factor)
        final_hit = final_close > entry
    else:
        final_pips = (entry - final_close) * factor
        mfe_pips = (entry - min(c.low for c in truth)) * factor
        mae_pips = max(0.0, (max(c.high for c in truth) - entry) * factor)
        final_hit = final_close < entry
    first_touch, target_first = _first_touch(row, entry, truth)
    target_touch = _target_touch(row, entry, truth)
    scalp_first_touch, scalp_tp_first = _first_scalp_touch(
        row.direction,
        entry,
        truth,
        factor=factor,
        take_profit_pips=scalp_tp_pips,
        stop_loss_pips=scalp_stop_pips,
    )
    features = _technical_features(row, indicators_by_tf, families_by_tf)
    return {
        "source_index": row.source_index,
        "timestamp_utc": _iso(row.timestamp),
        "pair": row.pair,
        "direction": row.direction,
        "confidence": row.confidence,
        "confidence_bucket": _confidence_bucket(row.confidence),
        "horizon_bucket": _horizon_bucket(row.horizon_min),
        "entry_source": entry_source,
        "target_pips": abs(row.target - entry) * factor if row.target is not None else None,
        "final_direction_hit": bool(final_hit),
        "final_pips": final_pips,
        "mfe_pips": mfe_pips,
        "mae_pips": mae_pips,
        "mfe_ge_2pip": bool(mfe_pips >= 2.0),
        "mfe_ge_5pip": bool(mfe_pips >= 5.0),
        "target_touch_hit": target_touch,
        "first_touch": first_touch,
        "target_before_invalidation_hit": target_first,
        "scalp_first_touch": scalp_first_touch,
        "scalp_take_profit_before_stop_hit": scalp_tp_first,
        "features": sorted(features),
    }


def _technical_features(
    row: ForecastRow,
    indicators_by_tf: dict[str, IndicatorSet],
    families_by_tf: dict[str, Any],
) -> set[str]:
    features = {
        f"pair:{row.pair}",
        f"direction:{row.direction}",
        f"confidence:{_confidence_bucket(row.confidence)}",
        f"horizon:{_horizon_bucket(row.horizon_min)}",
    }
    for tf in TIMEFRAMES:
        indicators = indicators_by_tf.get(tf)
        if indicators is None:
            continue
        features.update(_indicator_features(tf, row.direction, indicators))
        family_scores = families_by_tf.get(tf)
        if family_scores is not None:
            features.update(_family_features(tf, row.direction, family_scores))
    features.update(_cross_tf_features(row.direction, indicators_by_tf, families_by_tf))
    return features


def _indicator_features(tf: str, direction: str, indicators: IndicatorSet) -> set[str]:
    up = direction == "UP"
    features: set[str] = set()
    close = indicators.close
    if indicators.ema_20 is not None:
        aligned = (close > indicators.ema_20) == up
        features.add(f"{tf}:ema20_{'aligned' if aligned else 'opposed'}")
    if indicators.ema_50 is not None:
        aligned = (close > indicators.ema_50) == up
        features.add(f"{tf}:ema50_{'aligned' if aligned else 'opposed'}")
    if indicators.ema_slope_5 is not None:
        aligned = (indicators.ema_slope_5 > 0) == up
        features.add(f"{tf}:ema_slope5_{'aligned' if aligned else 'opposed'}")
    if indicators.macd_hist is not None:
        aligned = (indicators.macd_hist > 0) == up
        features.add(f"{tf}:macd_hist_{'aligned' if aligned else 'opposed'}")
    if indicators.rsi_14 is not None:
        features.add(f"{tf}:rsi_{_rsi_zone(indicators.rsi_14)}")
        reversal = (up and indicators.rsi_14 <= RSI_LOW) or (not up and indicators.rsi_14 >= RSI_HIGH)
        momentum = (up and indicators.rsi_14 >= 50.0) or (not up and indicators.rsi_14 <= 50.0)
        features.add(f"{tf}:rsi_reversal_{'aligned' if reversal else 'not_aligned'}")
        features.add(f"{tf}:rsi_momentum_{'aligned' if momentum else 'opposed'}")
    if indicators.stoch_rsi is not None:
        features.add(f"{tf}:stoch_{_stoch_zone(indicators.stoch_rsi)}")
        reversal = (up and indicators.stoch_rsi <= STOCH_LOW) or (not up and indicators.stoch_rsi >= STOCH_HIGH)
        features.add(f"{tf}:stoch_reversal_{'aligned' if reversal else 'not_aligned'}")
    if indicators.supertrend_dir is not None:
        aligned = (indicators.supertrend_dir > 0) == up
        features.add(f"{tf}:supertrend_{'aligned' if aligned else 'opposed'}")
    if indicators.aroon_up_14 is not None and indicators.aroon_down_14 is not None:
        aligned = (indicators.aroon_up_14 > indicators.aroon_down_14) == up
        features.add(f"{tf}:aroon_{'aligned' if aligned else 'opposed'}")
    if indicators.plus_di_14 is not None and indicators.minus_di_14 is not None:
        aligned = (indicators.plus_di_14 > indicators.minus_di_14) == up
        features.add(f"{tf}:di_{'aligned' if aligned else 'opposed'}")
    if indicators.adx_14 is not None:
        features.add(f"{tf}:adx_{'trend' if indicators.adx_14 >= ADX_TRENDING else 'weak'}")
    pct_b = _bb_pct_b(indicators)
    if pct_b is not None:
        features.add(f"{tf}:bb_{_bb_zone(pct_b)}")
        reversal = (up and pct_b <= BB_EDGE_PCT_B) or (not up and pct_b >= 1.0 - BB_EDGE_PCT_B)
        momentum = (up and pct_b >= 0.5) or (not up and pct_b <= 0.5)
        features.add(f"{tf}:bb_reversion_{'aligned' if reversal else 'not_aligned'}")
        features.add(f"{tf}:bb_momentum_{'aligned' if momentum else 'opposed'}")
    if indicators.donchian_high is not None and indicators.donchian_low is not None:
        if up:
            near_break = close >= indicators.donchian_high
        else:
            near_break = close <= indicators.donchian_low
        features.add(f"{tf}:donchian_breakout_{'aligned' if near_break else 'not_aligned'}")
    if indicators.vwap_gap_pips is not None:
        aligned = (indicators.vwap_gap_pips > 0) == up
        features.add(f"{tf}:vwap_gap_{'aligned' if aligned else 'opposed'}")
    if indicators.bb_squeeze is not None:
        features.add(f"{tf}:bb_squeeze_{int(indicators.bb_squeeze)}")
    if indicators.atr_percentile_100 is not None:
        features.add(f"{tf}:atr_{_percentile_zone(indicators.atr_percentile_100)}")
    if indicators.bb_width_percentile_100 is not None:
        features.add(f"{tf}:bb_width_{_percentile_zone(indicators.bb_width_percentile_100)}")
    if indicators.choppiness_14 is not None:
        if indicators.choppiness_14 >= CHOP_RANGE:
            features.add(f"{tf}:chop_range")
        elif indicators.choppiness_14 <= CHOP_TREND:
            features.add(f"{tf}:chop_trend")
        else:
            features.add(f"{tf}:chop_mid")
    return features


def _family_features(tf: str, direction: str, family_scores: Any) -> set[str]:
    up = direction == "UP"
    features: set[str] = set()
    trend = getattr(family_scores, "trend_score", 0.0)
    mean_rev = getattr(family_scores, "mean_rev_score", 0.0)
    breakout = getattr(family_scores, "breakout_score", 0.0)
    disagreement = getattr(family_scores, "disagreement", 0.0)
    features.add(f"{tf}:family_trend_{'aligned' if (trend > 0) == up else 'opposed'}")
    features.add(f"{tf}:family_meanrev_{'aligned' if (mean_rev > 0) == up else 'opposed'}")
    features.add(f"{tf}:family_breakout_{'expansion' if breakout > 0 else 'spent'}")
    features.add(f"{tf}:family_disagreement_{'high' if disagreement >= 0.75 else 'low'}")
    return features


def _cross_tf_features(
    direction: str,
    indicators_by_tf: dict[str, IndicatorSet],
    families_by_tf: dict[str, Any],
) -> set[str]:
    features: set[str] = set()
    for name, tfs in {
        "M1M5": ("M1", "M5"),
        "M5M15": ("M5", "M15"),
        "M1M5M15": ("M1", "M5", "M15"),
    }.items():
        ema = [_is_ema_aligned(direction, indicators_by_tf.get(tf)) for tf in tfs]
        macd = [_is_macd_aligned(direction, indicators_by_tf.get(tf)) for tf in tfs]
        trend_family = [_is_family_trend_aligned(direction, families_by_tf.get(tf)) for tf in tfs]
        if all(value is True for value in ema):
            features.add(f"cross:{name}:ema_all_aligned")
        if all(value is False for value in ema):
            features.add(f"cross:{name}:ema_all_opposed")
        if all(value is True for value in macd):
            features.add(f"cross:{name}:macd_all_aligned")
        if all(value is True for value in trend_family):
            features.add(f"cross:{name}:trend_family_all_aligned")
    return features


def _mine_feature_buckets(
    rows: list[dict[str, Any]],
    *,
    min_samples: int,
    min_touch_samples: int,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        for feature in row["features"]:
            buckets[feature].append(row)
        for feature in row["features"]:
            if not feature.startswith(("pair:", "direction:", "confidence:", "horizon:")):
                buckets[f"{row['pair']}|{row['direction']}|{feature}"].append(row)
                buckets[f"{row['direction']}|{feature}"].append(row)
    out: list[dict[str, Any]] = []
    for feature, values in buckets.items():
        if len(values) < min_samples and len([v for v in values if v["target_before_invalidation_hit"] is not None]) < min_touch_samples:
            continue
        row = {"feature": feature}
        row.update(_summarize(values))
        out.append(row)
    out.sort(
        key=lambda row: (
            -(row.get("target_first_wilson95_lower") or 0.0),
            -(row.get("scalp_tp_first_wilson95_lower") or 0.0),
            -(row.get("mfe_ge_2pip_wilson95_lower") or 0.0),
            -(row.get("final_wilson95_lower") or 0.0),
            -row["n"],
        )
    )
    return out


def _summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    final_hits = sum(1 for item in items if item["final_direction_hit"])
    mfe2 = sum(1 for item in items if item["mfe_ge_2pip"])
    mfe5 = sum(1 for item in items if item["mfe_ge_5pip"])
    target_items = [item for item in items if item["target_touch_hit"] is not None]
    target_hits = sum(1 for item in target_items if item["target_touch_hit"])
    first_items = [item for item in items if item["target_before_invalidation_hit"] is not None]
    first_hits = sum(1 for item in first_items if item["target_before_invalidation_hit"])
    scalp_items = [item for item in items if item["scalp_take_profit_before_stop_hit"] is not None]
    scalp_hits = sum(1 for item in scalp_items if item["scalp_take_profit_before_stop_hit"])
    final_pips = [float(item["final_pips"]) for item in items]
    return {
        "n": n,
        "final_hits": final_hits,
        "final_hit_rate": final_hits / n if n else None,
        "final_wilson95_lower": _wilson_lower(final_hits, n),
        "mfe_ge_2pip_hits": mfe2,
        "mfe_ge_2pip_rate": mfe2 / n if n else None,
        "mfe_ge_2pip_wilson95_lower": _wilson_lower(mfe2, n),
        "mfe_ge_5pip_hits": mfe5,
        "mfe_ge_5pip_rate": mfe5 / n if n else None,
        "target_touch_n": len(target_items),
        "target_touch_hits": target_hits,
        "target_touch_rate": target_hits / len(target_items) if target_items else None,
        "target_touch_wilson95_lower": _wilson_lower(target_hits, len(target_items)),
        "touch_n": len(first_items),
        "target_first_hits": first_hits,
        "target_first_rate": first_hits / len(first_items) if first_items else None,
        "target_first_wilson95_lower": _wilson_lower(first_hits, len(first_items)),
        "scalp_tp_first_n": len(scalp_items),
        "scalp_tp_first_hits": scalp_hits,
        "scalp_tp_first_rate": scalp_hits / len(scalp_items) if scalp_items else None,
        "scalp_tp_first_wilson95_lower": _wilson_lower(scalp_hits, len(scalp_items)),
        "avg_final_pips": statistics.fmean(final_pips) if final_pips else None,
        "median_final_pips": statistics.median(final_pips) if final_pips else None,
    }


def _timeframe_candles(m1: Sequence[Candle]) -> dict[str, list[Candle]]:
    return {
        "M1": list(m1),
        "M5": _resample(m1, minutes=5),
        "M15": _resample(m1, minutes=15),
    }


def _resample(candles: Sequence[Candle], *, minutes: int) -> list[Candle]:
    buckets: dict[datetime, list[Candle]] = collections.defaultdict(list)
    for candle in candles:
        ts = candle.timestamp_utc.astimezone(timezone.utc)
        bucket_minute = (ts.minute // minutes) * minutes
        bucket = ts.replace(minute=bucket_minute, second=0, microsecond=0)
        buckets[bucket].append(candle)
    out: list[Candle] = []
    for bucket in sorted(buckets):
        values = buckets[bucket]
        if not values:
            continue
        out.append(
            Candle(
                timestamp_utc=bucket,
                open=values[0].open,
                high=max(c.high for c in values),
                low=min(c.low for c in values),
                close=values[-1].close,
                volume=sum(c.volume for c in values),
                complete=all(c.complete for c in values),
            )
        )
    return out


def _first_touch(row: ForecastRow, entry: float, truth: list[Candle]) -> tuple[str | None, bool | None]:
    if row.target is None or row.invalidation is None:
        return None, None
    if row.direction == "UP":
        if row.target <= entry or row.invalidation >= entry:
            return None, None
    elif row.target >= entry or row.invalidation <= entry:
        return None, None
    for candle in truth:
        if row.direction == "UP":
            target_hit = candle.high >= row.target
            invalid_hit = candle.low <= row.invalidation
        else:
            target_hit = candle.low <= row.target
            invalid_hit = candle.high >= row.invalidation
        if target_hit and invalid_hit:
            return "AMBIGUOUS_SAME_M1", False
        if target_hit:
            return "TARGET_FIRST", True
        if invalid_hit:
            return "INVALIDATION_FIRST", False
    return "TIMEOUT", False


def _target_touch(row: ForecastRow, entry: float, truth: list[Candle]) -> bool | None:
    if row.target is None:
        return None
    if row.direction == "UP":
        if row.target <= entry:
            return None
        return any(c.high >= row.target for c in truth)
    if row.target >= entry:
        return None
    return any(c.low <= row.target for c in truth)


def _first_scalp_touch(
    direction: str,
    entry: float,
    truth: list[Candle],
    *,
    factor: float,
    take_profit_pips: float,
    stop_loss_pips: float,
) -> tuple[str | None, bool | None]:
    if take_profit_pips <= 0.0 or stop_loss_pips <= 0.0:
        return None, None
    if direction == "UP":
        take_profit = entry + take_profit_pips / factor
        stop_loss = entry - stop_loss_pips / factor
    else:
        take_profit = entry - take_profit_pips / factor
        stop_loss = entry + stop_loss_pips / factor
    for candle in truth:
        if direction == "UP":
            take_profit_hit = candle.high >= take_profit
            stop_loss_hit = candle.low <= stop_loss
        else:
            take_profit_hit = candle.low <= take_profit
            stop_loss_hit = candle.high >= stop_loss
        if take_profit_hit and stop_loss_hit:
            return "AMBIGUOUS_SAME_M1", False
        if take_profit_hit:
            return "TAKE_PROFIT_FIRST", True
        if stop_loss_hit:
            return "STOP_FIRST", False
    return "TIMEOUT", False


def _bb_pct_b(indicators: IndicatorSet) -> float | None:
    if indicators.bb_upper is None or indicators.bb_lower is None or indicators.bb_upper <= indicators.bb_lower:
        return None
    return (indicators.close - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower)


def _rsi_zone(value: float) -> str:
    if value <= RSI_LOW:
        return "low"
    if value >= RSI_HIGH:
        return "high"
    return "mid"


def _stoch_zone(value: float) -> str:
    if value <= STOCH_LOW:
        return "low"
    if value >= STOCH_HIGH:
        return "high"
    return "mid"


def _bb_zone(pct_b: float) -> str:
    if pct_b <= BB_EDGE_PCT_B:
        return "lower"
    if pct_b >= 1.0 - BB_EDGE_PCT_B:
        return "upper"
    return "middle"


def _percentile_zone(value: float) -> str:
    if value <= ATR_QUIET_PCTL:
        return "low"
    if value >= ATR_HOT_PCTL:
        return "high"
    return "mid"


def _is_ema_aligned(direction: str, indicators: IndicatorSet | None) -> bool | None:
    if indicators is None or indicators.ema_50 is None:
        return None
    return (indicators.close > indicators.ema_50) == (direction == "UP")


def _is_macd_aligned(direction: str, indicators: IndicatorSet | None) -> bool | None:
    if indicators is None or indicators.macd_hist is None:
        return None
    return (indicators.macd_hist > 0.0) == (direction == "UP")


def _is_family_trend_aligned(direction: str, family_scores: Any | None) -> bool | None:
    if family_scores is None:
        return None
    return (getattr(family_scores, "trend_score", 0.0) > 0.0) == (direction == "UP")


def _dedupe_key(
    *,
    cycle_id: str | None,
    pair: str,
    timestamp: datetime,
    direction: str,
    confidence: float | None,
    target: float | None,
    invalidation: float | None,
) -> tuple[Any, ...]:
    if cycle_id:
        return ("cycle", cycle_id, pair)
    return (
        "row",
        pair,
        timestamp.replace(microsecond=0).isoformat(),
        direction,
        round(confidence or 0.0, 6),
        round(target or 0.0, 5),
        round(invalidation or 0.0, 5),
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Technical Entry Mining",
        "",
        f"- generated_at_utc: {report['generated_at_utc']}",
        f"- source: `{report['source']}`",
        f"- truth: {report['truth_source']}",
        f"- scalp audit: TP {report['scalp_tp_pips']:.2f} pips / stop {report['scalp_stop_pips']:.2f} pips",
        f"- rows raw/deduped/scored: {report['raw_directional_rows']} / {report['deduped_directional_rows']} / {report['scored_rows']}",
        f"- fetch pairs/windows/candles/errors: {report['fetch_pairs']} / {report['fetch_windows']} / {report['fetched_candles']} / {len(report['fetch_errors'])}",
        "",
        "## Overall",
        "",
    ]
    lines.extend(_table([report["summary"]], _metric_cols()))
    lines.extend(["", "## Strict Target-First Candidates", ""])
    strict = report["strict_target_first_candidates"]
    lines.extend(_table(strict, _feature_cols(), limit=30) if strict else ["None."])
    lines.extend(["", "## Strict Scalp Candidates", ""])
    scalp = report["strict_scalp_candidates"]
    lines.extend(_table(scalp, _feature_cols(), limit=30) if scalp else ["None."])
    lines.extend(["", "## Positive MFE>=2 Candidates", ""])
    positive = report["positive_mfe2_candidates"]
    lines.extend(_table(positive, _feature_cols(), limit=30) if positive else ["None."])
    lines.extend(["", "## Negative Buckets", ""])
    negative = report["negative_buckets"]
    lines.extend(_table(negative, _feature_cols(), limit=30) if negative else ["None."])
    lines.extend(["", "## Top Feature Buckets", ""])
    lines.extend(_table(report["top_feature_buckets"], _feature_cols(), limit=50))
    return "\n".join(lines) + "\n"


def _metric_cols() -> list[tuple[str, str]]:
    return [
        ("n", "n"),
        ("final hit", "final_hit_rate"),
        ("final Wilson95L", "final_wilson95_lower"),
        ("MFE>=2", "mfe_ge_2pip_rate"),
        ("MFE2 Wilson95L", "mfe_ge_2pip_wilson95_lower"),
        ("touch n", "touch_n"),
        ("target-first", "target_first_rate"),
        ("target-first Wilson95L", "target_first_wilson95_lower"),
        ("scalp TP-first", "scalp_tp_first_rate"),
        ("scalp Wilson95L", "scalp_tp_first_wilson95_lower"),
        ("avg pips", "avg_final_pips"),
    ]


def _feature_cols() -> list[tuple[str, str]]:
    return [("feature", "feature"), *_metric_cols()]


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], limit: int = 12) -> list[str]:
    lines = [
        "| " + " | ".join(header for header, _ in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows[:limit]:
        values: list[str] = []
        for _, key in cols:
            value = row.get(key)
            if isinstance(value, float):
                values.append(_pct(value) if "rate" in key or "wilson" in key else f"{value:.2f}")
            elif value is None:
                values.append("n/a")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_horizon(value: Any) -> float:
    parsed = _safe_float(value)
    if parsed is None or parsed <= 0.0:
        return 60.0
    return min(parsed, 1440.0)


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _ceil_minute(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc)
    base = value.replace(second=0, microsecond=0)
    return base if value == base else base + timedelta(minutes=1)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _confidence_bucket(confidence: float | None) -> str:
    if confidence is None:
        return "missing"
    if confidence >= 0.90:
        return ">=0.90"
    if confidence >= 0.75:
        return "0.75-0.90"
    if confidence >= 0.55:
        return "0.55-0.75"
    return "<0.55"


def _horizon_bucket(minutes: float) -> str:
    if minutes <= 15:
        return "<=15m"
    if minutes <= 60:
        return "<=60m"
    if minutes <= 240:
        return "<=4h"
    return ">4h"


def _wilson_lower(successes: int, trials: int, z: float = 1.96) -> float | None:
    if trials <= 0:
        return None
    p_hat = successes / trials
    denom = 1.0 + z * z / trials
    centre = p_hat + z * z / (2 * trials)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4 * trials)) / trials)
    return max(0.0, (centre - margin) / denom)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())

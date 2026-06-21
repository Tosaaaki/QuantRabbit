#!/usr/bin/env python3
"""Mine cross-pair high-turnover entry shapes from local OANDA candles.

This audit intentionally does not start from ``forecast_history.jsonl``.  It
walks the already-fetched multi-month OANDA bid/ask candle files directly and
tests the same normalized entry shapes across every pair:

* range edge reversion
* failed-break fade
* trend continuation
* pullback continuation
* squeeze breakout

Pair-specific volatility is normalized with M5 ATR, and spread is charged by
entering LONG at ask / exiting at bid and SHORT at bid / exiting at ask.  The
result is evidence only; it does not grant live permission by itself.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor


JST = timezone(timedelta(hours=9), "JST")
DEFAULT_HISTORY_ROOT = Path("logs/replay/oanda_history")
DEFAULT_OUTPUT_DIR = Path("logs/reports/forecast_improvement")
DEFAULT_HISTORY_GLOB = "*_M5_BA_20260316T000000Z_20260620T000000Z.jsonl"

# These are audit-label boundaries that convert normalized continuous market
# state into repeatable buckets. Runtime entries still have to pass current
# broker truth, strategy profile, spread, risk, and live gateway checks.
ATR_PERIOD = 14
RANGE_LOOKBACK = 20
MOMENTUM_FAST_BARS = 3
MOMENTUM_SLOW_BARS = 12
MIN_WARMUP_BARS = max(ATR_PERIOD + 2, RANGE_LOOKBACK + 2, MOMENTUM_SLOW_BARS + 2)
DEFAULT_MAX_HOLD_BARS = 12
DEFAULT_STRIDE_BARS = 1
DEFAULT_MIN_SAMPLES = 240
DEFAULT_MIN_ACTIVE_DAYS = 15
DEFAULT_MIN_PAIR_COUNT = 6
DEFAULT_MAX_PAIR_SAMPLE_SHARE = 0.35
DEFAULT_MAX_DAILY_SAMPLE_SHARE = 0.18
DEFAULT_MIN_POSITIVE_DAY_RATE = 0.55
DEFAULT_MIN_VALIDATION_EXPECTANCY_ATR = 0.015
DEFAULT_MIN_VALIDATION_WIN_RATE = 0.52
DEFAULT_MIN_VALIDATION_SAMPLES = 12
DEFAULT_MIN_PROFIT_FACTOR = 1.08
DEFAULT_HIGH_PRECISION_MIN_WIN_RATE = 0.70
DEFAULT_HIGH_PRECISION_MIN_WILSON_LOWER = 0.50
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_TOP = 30
DEFAULT_DIRECTIONAL_SELECTOR_MIN_SAMPLES = 60
DEFAULT_MULTI_CONFLUENCE_SIZES = (3, 4)
DEFAULT_INVERSION_SELECTOR_CONFLUENCE_SIZES = (2, 3)
DEFAULT_SELECTOR_FEATURE_PREFIXES = (
    "session:",
    "atr_regime:",
    "spread_regime:",
    "range_pos:",
    "fast_mom_abs:",
    "slow_mom_abs:",
    "body_abs:",
    "bar_range:",
)
DEFAULT_EXIT_SHAPES = (
    "tp0.75_sl0.75",
    "tp1_sl0.75",
    "tp1_sl1",
    "tp1.25_sl1",
)


@dataclass(frozen=True)
class BaOhlc:
    timestamp_utc: datetime
    bid_o: float
    bid_h: float
    bid_l: float
    bid_c: float
    ask_o: float
    ask_h: float
    ask_l: float
    ask_c: float
    volume: float

    @property
    def mid_o(self) -> float:
        return (self.bid_o + self.ask_o) / 2.0

    @property
    def mid_h(self) -> float:
        return (self.bid_h + self.ask_h) / 2.0

    @property
    def mid_l(self) -> float:
        return (self.bid_l + self.ask_l) / 2.0

    @property
    def mid_c(self) -> float:
        return (self.bid_c + self.ask_c) / 2.0


@dataclass(frozen=True)
class Candidate:
    timestamp_utc: datetime
    pair: str
    side: str
    shape: str
    features: tuple[str, ...]
    entry_bid: float
    entry_ask: float
    atr_pips: float
    spread_pips: float


def main() -> int:
    args = _parse_args()
    pairs = _parse_pairs(args.pairs)
    exit_shapes = _parse_exit_shapes(args.exit_shapes)
    multi_confluence_sizes = _parse_multi_confluence_sizes(args.multi_confluence_sizes)
    inversion_selector_sizes = _parse_inversion_selector_sizes(args.inversion_selector_sizes)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_out = out_dir / f"oanda_universal_rotation_mining_{run_ts}.json"
    md_out = out_dir / f"oanda_universal_rotation_mining_{run_ts}.md"
    latest_json = out_dir / "oanda_universal_rotation_mining_latest.json"
    latest_md = out_dir / "oanda_universal_rotation_mining_latest.md"

    files = _discover_m5_files(args.history_root, args.history_glob, pairs=pairs)
    rows, inversion_rows, load_stats = _score_files(
        files,
        exit_shapes=exit_shapes,
        max_hold_bars=args.max_hold_bars,
        stride_bars=args.stride_bars,
        min_atr_pips=args.min_atr_pips,
        max_spread_atr=args.max_spread_atr,
        tp_spread_floor=args.tp_spread_floor,
        sl_spread_floor=args.sl_spread_floor,
    )
    report = _build_report(
        rows,
        generated_at_utc=datetime.now(timezone.utc),
        history_root=args.history_root,
        files=files,
        exit_shapes=exit_shapes,
        max_hold_bars=args.max_hold_bars,
        stride_bars=args.stride_bars,
        tp_spread_floor=args.tp_spread_floor,
        sl_spread_floor=args.sl_spread_floor,
        train_fraction=args.train_fraction,
        min_samples=args.min_samples,
        min_active_days=args.min_active_days,
        min_pair_count=args.min_pair_count,
        max_pair_sample_share=args.max_pair_sample_share,
        max_daily_sample_share=args.max_daily_sample_share,
        min_positive_day_rate=args.min_positive_day_rate,
        min_validation_expectancy_atr=args.min_validation_expectancy_atr,
        min_validation_win_rate=args.min_validation_win_rate,
        min_validation_samples=args.min_validation_samples,
        min_profit_factor=args.min_profit_factor,
        high_precision_min_win_rate=args.high_precision_min_win_rate,
        high_precision_min_wilson_lower=args.high_precision_min_wilson_lower,
        multi_confluence_sizes=multi_confluence_sizes,
        inversion_selector_sizes=inversion_selector_sizes,
        top=args.top,
        load_stats=load_stats,
        inversion_rows=inversion_rows,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    json_out.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")
    markdown = _markdown(report)
    md_out.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--history-glob", default=DEFAULT_HISTORY_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairs", default="")
    parser.add_argument("--exit-shapes", default=",".join(DEFAULT_EXIT_SHAPES))
    parser.add_argument("--max-hold-bars", type=int, default=DEFAULT_MAX_HOLD_BARS)
    parser.add_argument("--stride-bars", type=int, default=DEFAULT_STRIDE_BARS)
    parser.add_argument("--min-atr-pips", type=float, default=0.1)
    parser.add_argument("--max-spread-atr", type=float, default=0.45)
    parser.add_argument(
        "--tp-spread-floor",
        type=float,
        default=2.5,
        help="minimum take-profit distance as a multiple of current spread",
    )
    parser.add_argument(
        "--sl-spread-floor",
        type=float,
        default=2.0,
        help="minimum stop distance as a multiple of current spread",
    )
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--min-active-days", type=int, default=DEFAULT_MIN_ACTIVE_DAYS)
    parser.add_argument("--min-pair-count", type=int, default=DEFAULT_MIN_PAIR_COUNT)
    parser.add_argument("--max-pair-sample-share", type=float, default=DEFAULT_MAX_PAIR_SAMPLE_SHARE)
    parser.add_argument("--max-daily-sample-share", type=float, default=DEFAULT_MAX_DAILY_SAMPLE_SHARE)
    parser.add_argument("--min-positive-day-rate", type=float, default=DEFAULT_MIN_POSITIVE_DAY_RATE)
    parser.add_argument(
        "--min-validation-expectancy-atr",
        type=float,
        default=DEFAULT_MIN_VALIDATION_EXPECTANCY_ATR,
    )
    parser.add_argument("--min-validation-win-rate", type=float, default=DEFAULT_MIN_VALIDATION_WIN_RATE)
    parser.add_argument("--min-validation-samples", type=int, default=DEFAULT_MIN_VALIDATION_SAMPLES)
    parser.add_argument("--min-profit-factor", type=float, default=DEFAULT_MIN_PROFIT_FACTOR)
    parser.add_argument(
        "--high-precision-min-win-rate",
        type=float,
        default=DEFAULT_HIGH_PRECISION_MIN_WIN_RATE,
    )
    parser.add_argument(
        "--high-precision-min-wilson-lower",
        type=float,
        default=DEFAULT_HIGH_PRECISION_MIN_WILSON_LOWER,
    )
    parser.add_argument(
        "--multi-confluence-sizes",
        default=",".join(str(size) for size in DEFAULT_MULTI_CONFLUENCE_SIZES),
        help=(
            "comma-separated feature-confluence sizes to mine for pair-specific buckets. "
            "Defaults to 3,4; larger values are intentionally opt-in because combinations grow quickly."
        ),
    )
    parser.add_argument(
        "--inversion-selector-sizes",
        default=",".join(str(size) for size in DEFAULT_INVERSION_SELECTOR_CONFLUENCE_SIZES),
        help=(
            "comma-separated neutral-feature confluence sizes used to test whether the opposite "
            "side of a fired entry shape was actually profitable on the same candles. Defaults to 2,3."
        ),
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    return parser.parse_args()


def _parse_pairs(value: str) -> set[str]:
    items = {item.strip().upper() for item in str(value or "").split(",") if item.strip()}
    return items or set(DEFAULT_TRADER_PAIRS)


def _parse_exit_shapes(value: str) -> tuple[tuple[str, float, float], ...]:
    shapes: list[tuple[str, float, float]] = []
    for raw in str(value or "").split(","):
        text = raw.strip().lower()
        if not text:
            continue
        try:
            left, right = text.split("_", 1)
            tp = float(left.removeprefix("tp"))
            sl = float(right.removeprefix("sl"))
        except (ValueError, AttributeError):
            continue
        if math.isfinite(tp) and math.isfinite(sl) and tp > 0.0 and sl > 0.0:
            shapes.append((f"tp{tp:g}_sl{sl:g}", tp, sl))
    if not shapes:
        raise ValueError("--exit-shapes produced no valid tp/sl ATR ratios")
    return tuple(shapes)


def _parse_multi_confluence_sizes(value: str) -> tuple[int, ...]:
    sizes: set[int] = set()
    for raw in str(value or "").split(","):
        text = raw.strip()
        if not text:
            continue
        try:
            size = int(text)
        except ValueError as exc:
            raise ValueError(f"invalid multi confluence size: {text!r}") from exc
        if size < 3 or size > 8:
            raise ValueError("multi confluence sizes must be between 3 and 8")
        sizes.add(size)
    if not sizes:
        raise ValueError("--multi-confluence-sizes produced no valid sizes")
    return tuple(sorted(sizes))


def _parse_inversion_selector_sizes(value: str) -> tuple[int, ...]:
    sizes: set[int] = set()
    for raw in str(value or "").split(","):
        text = raw.strip()
        if not text:
            continue
        try:
            size = int(text)
        except ValueError as exc:
            raise ValueError(f"invalid inversion selector size: {text!r}") from exc
        if size < 2 or size > 6:
            raise ValueError("inversion selector sizes must be between 2 and 6")
        sizes.add(size)
    if not sizes:
        raise ValueError("--inversion-selector-sizes produced no valid sizes")
    return tuple(sorted(sizes))


def _discover_m5_files(root: Path, history_glob: str, *, pairs: set[str]) -> list[Path]:
    by_pair: dict[str, Path] = {}
    for path in sorted(root.rglob(history_glob)):
        pair = path.parent.name.upper()
        if pair not in pairs:
            continue
        current = by_pair.get(pair)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            by_pair[pair] = path
    return [by_pair[pair] for pair in sorted(by_pair)]


def _score_files(
    files: Sequence[Path],
    *,
    exit_shapes: Sequence[tuple[str, float, float]],
    max_hold_bars: int,
    stride_bars: int,
    min_atr_pips: float,
    max_spread_atr: float,
    tp_spread_floor: float,
    sl_spread_floor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    inversion_scored: list[dict[str, Any]] = []
    pair_stats: list[dict[str, Any]] = []
    for path in files:
        pair = path.parent.name.upper()
        candles = _load_ba_candles(path)
        pair_rows, pair_inversion_rows = _score_pair(
            pair,
            candles,
            exit_shapes=exit_shapes,
            max_hold_bars=max_hold_bars,
            stride_bars=stride_bars,
            min_atr_pips=min_atr_pips,
            max_spread_atr=max_spread_atr,
            tp_spread_floor=tp_spread_floor,
            sl_spread_floor=sl_spread_floor,
        )
        scored.extend(pair_rows)
        inversion_scored.extend(pair_inversion_rows)
        pair_stats.append(
            {
                "pair": pair,
                "candles": len(candles),
                "scored_outcomes": len(pair_rows),
                "inversion_scored_outcomes": len(pair_inversion_rows),
            }
        )
    return scored, inversion_scored, {
        "history_files": len(files),
        "history_pairs": len({path.parent.name.upper() for path in files}),
        "history_file_paths": [str(path) for path in files],
        "pair_load_stats": pair_stats,
        "scored_outcomes": len(scored),
        "inversion_scored_outcomes": len(inversion_scored),
    }


def _load_ba_candles(path: Path) -> list[BaOhlc]:
    candles: list[BaOhlc] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not payload.get("complete", True):
                continue
            bid = payload.get("bid") or {}
            ask = payload.get("ask") or {}
            ts = _parse_time(payload.get("time"))
            if ts is None:
                continue
            try:
                candles.append(
                    BaOhlc(
                        timestamp_utc=ts,
                        bid_o=float(bid["o"]),
                        bid_h=float(bid["h"]),
                        bid_l=float(bid["l"]),
                        bid_c=float(bid["c"]),
                        ask_o=float(ask["o"]),
                        ask_h=float(ask["h"]),
                        ask_l=float(ask["l"]),
                        ask_c=float(ask["c"]),
                        volume=float(payload.get("volume") or 0.0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    candles.sort(key=lambda item: item.timestamp_utc)
    return candles


def _score_pair(
    pair: str,
    candles: Sequence[BaOhlc],
    *,
    exit_shapes: Sequence[tuple[str, float, float]],
    max_hold_bars: int,
    stride_bars: int,
    min_atr_pips: float,
    max_spread_atr: float,
    tp_spread_floor: float,
    sl_spread_floor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(candles) < MIN_WARMUP_BARS + max_hold_bars + 2:
        return [], []
    factor = instrument_pip_factor(pair)
    atrs = _atr_pips(candles, factor=factor, period=ATR_PERIOD)
    atr_quantiles = _quantile_boundaries(
        [value for value in atrs if value is not None and value > 0.0],
        lower=0.25,
        upper=0.75,
    )
    rows: list[dict[str, Any]] = []
    inversion_rows: list[dict[str, Any]] = []
    stride = max(1, int(stride_bars))
    for idx in range(MIN_WARMUP_BARS, len(candles) - max_hold_bars - 1, stride):
        atr = atrs[idx]
        if atr is None or atr < min_atr_pips:
            continue
        spread_pips = (candles[idx].ask_c - candles[idx].bid_c) * factor
        if spread_pips <= 0.0:
            continue
        spread_atr = spread_pips / atr
        if spread_atr > max_spread_atr:
            continue
        common = _market_state_features(
            candles,
            idx,
            factor=factor,
            atr_pips=atr,
            spread_atr=spread_atr,
            atr_quantiles=atr_quantiles,
        )
        neutral_features = _neutral_features(common)
        for side in ("LONG", "SHORT"):
            side_features = _side_features(candles, idx, side=side, atr_pips=atr, common=common)
            for shape in _entry_shapes(side_features):
                candidate = Candidate(
                    timestamp_utc=candles[idx].timestamp_utc,
                    pair=pair,
                    side=side,
                    shape=shape,
                    features=tuple(sorted(side_features | {f"shape:{shape}", f"side:{side}"})),
                    entry_bid=candles[idx].bid_c,
                    entry_ask=candles[idx].ask_c,
                    atr_pips=atr,
                    spread_pips=spread_pips,
                )
                inverse_side = _opposite_side(side)
                inverse_candidate = Candidate(
                    timestamp_utc=candles[idx].timestamp_utc,
                    pair=pair,
                    side=inverse_side,
                    shape=shape,
                    features=(),
                    entry_bid=candles[idx].bid_c,
                    entry_ask=candles[idx].ask_c,
                    atr_pips=atr,
                    spread_pips=spread_pips,
                )
                for exit_name, tp_atr, sl_atr in exit_shapes:
                    result = _score_exit(
                        candidate,
                        candles,
                        idx,
                        factor=factor,
                        tp_atr=tp_atr,
                        sl_atr=sl_atr,
                        max_hold_bars=max_hold_bars,
                        tp_spread_floor=tp_spread_floor,
                        sl_spread_floor=sl_spread_floor,
                    )
                    rows.append(
                        {
                            "timestamp_utc": _iso(candidate.timestamp_utc),
                            "jst_day": candidate.timestamp_utc.astimezone(JST).date().isoformat(),
                            "pair": pair,
                            "side": side,
                            "shape": shape,
                            "exit_shape": exit_name,
                            "atr_pips": round(atr, 6),
                            "spread_pips": round(spread_pips, 6),
                            "spread_atr": round(spread_atr, 6),
                            "take_profit_pips": result["take_profit_pips"],
                            "stop_loss_pips": result["stop_loss_pips"],
                            "realized_pips": result["realized_pips"],
                            "realized_atr": result["realized_pips"] / atr,
                            "win": result["realized_pips"] > 0.0,
                            "outcome": result["outcome"],
                            "features": list(candidate.features),
                            "neutral_features": neutral_features,
                        }
                    )
                    inverse_result = _score_exit(
                        inverse_candidate,
                        candles,
                        idx,
                        factor=factor,
                        tp_atr=tp_atr,
                        sl_atr=sl_atr,
                        max_hold_bars=max_hold_bars,
                        tp_spread_floor=tp_spread_floor,
                        sl_spread_floor=sl_spread_floor,
                    )
                    inversion_rows.append(
                        {
                            "timestamp_utc": _iso(candidate.timestamp_utc),
                            "jst_day": candidate.timestamp_utc.astimezone(JST).date().isoformat(),
                            "pair": pair,
                            "shape": shape,
                            "source_shape": shape,
                            "source_side": side,
                            "selected_side": inverse_side,
                            "side": inverse_side,
                            "exit_shape": exit_name,
                            "atr_pips": round(atr, 6),
                            "spread_pips": round(spread_pips, 6),
                            "spread_atr": round(spread_atr, 6),
                            "take_profit_pips": inverse_result["take_profit_pips"],
                            "stop_loss_pips": inverse_result["stop_loss_pips"],
                            "realized_pips": inverse_result["realized_pips"],
                            "realized_atr": inverse_result["realized_pips"] / atr,
                            "win": inverse_result["realized_pips"] > 0.0,
                            "outcome": inverse_result["outcome"],
                            "source_realized_pips": result["realized_pips"],
                            "source_realized_atr": result["realized_pips"] / atr,
                            "source_win": result["realized_pips"] > 0.0,
                            "source_outcome": result["outcome"],
                            "source_features": list(candidate.features),
                            "neutral_features": neutral_features,
                        }
                    )
    return rows, inversion_rows


def _market_state_features(
    candles: Sequence[BaOhlc],
    idx: int,
    *,
    factor: int,
    atr_pips: float,
    spread_atr: float,
    atr_quantiles: tuple[float, float],
) -> dict[str, Any]:
    c = candles[idx]
    prev3 = candles[idx - MOMENTUM_FAST_BARS]
    prev12 = candles[idx - MOMENTUM_SLOW_BARS]
    window = candles[idx - RANGE_LOOKBACK + 1: idx + 1]
    range_high = max(item.mid_h for item in window)
    range_low = min(item.mid_l for item in window)
    width = max(range_high - range_low, 1e-12)
    range_pos = (c.mid_c - range_low) / width
    fast_mom = (c.mid_c - prev3.mid_c) * factor / atr_pips
    slow_mom = (c.mid_c - prev12.mid_c) * factor / atr_pips
    body = (c.mid_c - c.mid_o) * factor / atr_pips
    full_range = max((c.mid_h - c.mid_l) * factor / atr_pips, 1e-9)
    upper_wick = (c.mid_h - max(c.mid_o, c.mid_c)) * factor / atr_pips
    lower_wick = (min(c.mid_o, c.mid_c) - c.mid_l) * factor / atr_pips
    q_low, q_high = atr_quantiles
    return {
        "session": _session(c.timestamp_utc),
        "atr_regime": "low" if atr_pips <= q_low else "high" if atr_pips >= q_high else "mid",
        "spread_regime": "low" if spread_atr <= 0.15 else "high" if spread_atr >= 0.30 else "mid",
        "range_pos": range_pos,
        "fast_mom": fast_mom,
        "slow_mom": slow_mom,
        "body": body,
        "upper_wick": upper_wick / full_range,
        "lower_wick": lower_wick / full_range,
        "bar_range_atr": full_range,
        "volume": c.volume,
    }


def _neutral_features(common: dict[str, Any]) -> list[str]:
    return sorted(
        {
            f"session:{common['session']}",
            f"atr_regime:{common['atr_regime']}",
            f"spread_regime:{common['spread_regime']}",
            f"range_pos:{_range_pos_bucket(float(common['range_pos']))}",
            f"fast_mom_abs:{_absolute_signed_bucket(float(common['fast_mom']))}",
            f"slow_mom_abs:{_absolute_signed_bucket(float(common['slow_mom']))}",
            f"body_abs:{_absolute_signed_bucket(float(common['body']))}",
            f"upper_wick:{'high' if float(common['upper_wick']) >= 0.45 else 'normal'}",
            f"lower_wick:{'high' if float(common['lower_wick']) >= 0.45 else 'normal'}",
            f"bar_range:{'wide' if common['bar_range_atr'] >= 1.2 else 'normal'}",
        }
    )


def _side_features(
    candles: Sequence[BaOhlc],
    idx: int,
    *,
    side: str,
    atr_pips: float,
    common: dict[str, Any],
) -> set[str]:
    direction = 1.0 if side == "LONG" else -1.0
    fast = float(common["fast_mom"]) * direction
    slow = float(common["slow_mom"]) * direction
    body = float(common["body"]) * direction
    range_pos = float(common["range_pos"])
    reward_edge = range_pos <= 0.25 if side == "LONG" else range_pos >= 0.75
    breakout_edge = range_pos >= 0.75 if side == "LONG" else range_pos <= 0.25
    wick_reject = common["lower_wick"] >= 0.45 if side == "LONG" else common["upper_wick"] >= 0.45
    features = {
        f"session:{common['session']}",
        f"atr_regime:{common['atr_regime']}",
        f"spread_regime:{common['spread_regime']}",
        f"range_pos:{_range_pos_bucket(range_pos)}",
        f"side_range:{'reward_edge' if reward_edge else 'breakout_edge' if breakout_edge else 'middle'}",
        f"fast_mom:{_signed_bucket(fast)}",
        f"slow_mom:{_signed_bucket(slow)}",
        f"body:{_signed_bucket(body)}",
        f"wick_reject:{int(wick_reject)}",
        f"bar_range:{'wide' if common['bar_range_atr'] >= 1.2 else 'normal'}",
    }
    if _failed_break(candles, idx, side=side, atr_pips=atr_pips):
        features.add("failed_break:1")
    else:
        features.add("failed_break:0")
    return features


def _opposite_side(side: str) -> str:
    return "SHORT" if str(side).upper() == "LONG" else "LONG"


def _entry_shapes(features: set[str]) -> tuple[str, ...]:
    values: list[str] = []
    reward_edge = "side_range:reward_edge" in features
    breakout_edge = "side_range:breakout_edge" in features
    fast_up = "fast_mom:aligned" in features
    fast_down = "fast_mom:opposed" in features
    slow_up = "slow_mom:aligned" in features
    slow_down = "slow_mom:opposed" in features
    body_up = "body:aligned" in features
    wick = "wick_reject:1" in features
    failed_break = "failed_break:1" in features
    atr_low = "atr_regime:low" in features
    if reward_edge and (fast_down or wick):
        values.append("range_reversion")
    if reward_edge and body_up and (fast_down or wick):
        values.append("range_reclaim")
    if failed_break and reward_edge:
        values.append("failed_break_fade")
    if breakout_edge and fast_up and slow_up:
        values.append("trend_continuation")
    if slow_up and fast_down and "side_range:middle" in features:
        values.append("pullback_continuation")
    if atr_low and breakout_edge and fast_up and body_up:
        values.append("squeeze_breakout")
    if slow_down and breakout_edge and fast_down:
        values.append("exhaustion_chase")
    return tuple(values)


def _score_exit(
    candidate: Candidate,
    candles: Sequence[BaOhlc],
    idx: int,
    *,
    factor: int,
    tp_atr: float,
    sl_atr: float,
    max_hold_bars: int,
    tp_spread_floor: float,
    sl_spread_floor: float,
) -> dict[str, Any]:
    take_profit_pips = max(tp_atr * candidate.atr_pips, tp_spread_floor * candidate.spread_pips)
    stop_loss_pips = max(sl_atr * candidate.atr_pips, sl_spread_floor * candidate.spread_pips)
    for candle in candles[idx + 1: idx + 1 + max_hold_bars]:
        if candidate.side == "LONG":
            favorable = (candle.bid_h - candidate.entry_ask) * factor
            adverse = (candidate.entry_ask - candle.bid_l) * factor
        else:
            favorable = (candidate.entry_bid - candle.ask_l) * factor
            adverse = (candle.ask_h - candidate.entry_bid) * factor
        tp_hit = favorable >= take_profit_pips
        sl_hit = adverse >= stop_loss_pips
        if tp_hit and sl_hit:
            return {
                "outcome": "AMBIGUOUS_SAME_M5",
                "realized_pips": -stop_loss_pips,
                "take_profit_pips": take_profit_pips,
                "stop_loss_pips": stop_loss_pips,
            }
        if tp_hit:
            return {
                "outcome": "TAKE_PROFIT_FIRST",
                "realized_pips": take_profit_pips,
                "take_profit_pips": take_profit_pips,
                "stop_loss_pips": stop_loss_pips,
            }
        if sl_hit:
            return {
                "outcome": "STOP_FIRST",
                "realized_pips": -stop_loss_pips,
                "take_profit_pips": take_profit_pips,
                "stop_loss_pips": stop_loss_pips,
            }
    final = candles[min(idx + max_hold_bars, len(candles) - 1)]
    if candidate.side == "LONG":
        realized = (final.bid_c - candidate.entry_ask) * factor
    else:
        realized = (candidate.entry_bid - final.ask_c) * factor
    return {
        "outcome": "TIMEOUT",
        "realized_pips": realized,
        "take_profit_pips": take_profit_pips,
        "stop_loss_pips": stop_loss_pips,
    }


def _build_report(
    rows: list[dict[str, Any]],
    *,
    inversion_rows: list[dict[str, Any]] | None = None,
    generated_at_utc: datetime,
    history_root: Path,
    files: Sequence[Path],
    exit_shapes: Sequence[tuple[str, float, float]],
    max_hold_bars: int,
    stride_bars: int,
    tp_spread_floor: float,
    sl_spread_floor: float,
    train_fraction: float,
    min_samples: int,
    min_active_days: int,
    min_pair_count: int,
    max_pair_sample_share: float,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
    high_precision_min_win_rate: float,
    high_precision_min_wilson_lower: float,
    multi_confluence_sizes: Sequence[int],
    inversion_selector_sizes: Sequence[int] = DEFAULT_INVERSION_SELECTOR_CONFLUENCE_SIZES,
    top: int,
    load_stats: dict[str, Any],
) -> dict[str, Any]:
    inversion_rows = inversion_rows or []
    shape_rows = _summarize_buckets(
        rows,
        key_fields=("shape", "side", "exit_shape"),
        train_fraction=train_fraction,
        min_samples=min_samples,
        min_active_days=min_active_days,
        min_pair_count=min_pair_count,
        max_pair_sample_share=max_pair_sample_share,
        max_daily_sample_share=max_daily_sample_share,
        min_positive_day_rate=min_positive_day_rate,
        min_validation_expectancy_atr=min_validation_expectancy_atr,
        min_validation_win_rate=min_validation_win_rate,
        min_validation_samples=max(30, min_validation_samples),
        min_profit_factor=min_profit_factor,
    )
    pair_shape_rows = _summarize_buckets(
        rows,
        key_fields=("pair", "shape", "side", "exit_shape"),
        train_fraction=train_fraction,
        min_samples=max(40, min_samples // 4),
        min_active_days=max(5, min_active_days // 3),
        min_pair_count=1,
        max_pair_sample_share=1.0,
        max_daily_sample_share=max(0.35, max_daily_sample_share),
        min_positive_day_rate=min_positive_day_rate,
        min_validation_expectancy_atr=min_validation_expectancy_atr,
        min_validation_win_rate=min_validation_win_rate,
        min_validation_samples=min_validation_samples,
        min_profit_factor=min_profit_factor,
    )
    feature_rows = _summarize_feature_buckets(
        rows,
        train_fraction=train_fraction,
        min_samples=min_samples,
        min_active_days=min_active_days,
        min_pair_count=min_pair_count,
        max_pair_sample_share=max_pair_sample_share,
        max_daily_sample_share=max_daily_sample_share,
        min_positive_day_rate=min_positive_day_rate,
        min_validation_expectancy_atr=min_validation_expectancy_atr,
        min_validation_win_rate=min_validation_win_rate,
        min_validation_samples=max(30, min_validation_samples),
        min_profit_factor=min_profit_factor,
    )
    pair_feature_rows = _summarize_pair_feature_buckets(
        rows,
        train_fraction=train_fraction,
        min_samples=max(30, min_samples // 6),
        min_active_days=max(5, min_active_days // 3),
        max_daily_sample_share=max(0.35, max_daily_sample_share),
        min_positive_day_rate=min_positive_day_rate,
        min_validation_expectancy_atr=min_validation_expectancy_atr,
        min_validation_win_rate=min_validation_win_rate,
        min_validation_samples=min_validation_samples,
        min_profit_factor=min_profit_factor,
    )
    pair_confluence_rows = _summarize_pair_confluence_buckets(
        rows,
        train_fraction=train_fraction,
        min_samples=max(30, min_samples // 8),
        min_active_days=max(5, min_active_days // 3),
        max_daily_sample_share=max(0.35, max_daily_sample_share),
        min_positive_day_rate=min_positive_day_rate,
        min_validation_expectancy_atr=min_validation_expectancy_atr,
        min_validation_win_rate=min_validation_win_rate,
        min_validation_samples=min_validation_samples,
        min_profit_factor=min_profit_factor,
    )
    multi_confluence_rows = []
    multi_confluence_min_samples = max(24, min_samples // 10)
    for confluence_size in multi_confluence_sizes:
        multi_confluence_rows.extend(
            _summarize_pair_multi_confluence_buckets(
                rows,
                confluence_size=confluence_size,
                train_fraction=train_fraction,
                min_samples=multi_confluence_min_samples,
                min_active_days=max(5, min_active_days // 3),
                max_daily_sample_share=max(0.35, max_daily_sample_share),
                min_positive_day_rate=min_positive_day_rate,
                min_validation_expectancy_atr=min_validation_expectancy_atr,
                min_validation_win_rate=min_validation_win_rate,
                min_validation_samples=min_validation_samples,
                min_profit_factor=min_profit_factor,
            )
        )
    multi_confluence_rows.sort(key=_bucket_sort_key)
    directional_selector_rows = _summarize_directional_selector_buckets(
        rows,
        train_fraction=train_fraction,
        min_samples=max(DEFAULT_DIRECTIONAL_SELECTOR_MIN_SAMPLES, min_samples // 4),
        min_active_days=max(5, min_active_days // 3),
        max_daily_sample_share=max(0.35, max_daily_sample_share),
        min_positive_day_rate=min_positive_day_rate,
        min_validation_expectancy_atr=min_validation_expectancy_atr,
        min_validation_win_rate=min_validation_win_rate,
        min_validation_samples=min_validation_samples,
        min_profit_factor=min_profit_factor,
    )
    inversion_selector_rows = _summarize_inversion_selector_buckets(
        inversion_rows,
        confluence_sizes=inversion_selector_sizes,
        train_fraction=train_fraction,
        min_samples=max(DEFAULT_DIRECTIONAL_SELECTOR_MIN_SAMPLES, min_samples // 4),
        min_active_days=max(5, min_active_days // 3),
        max_daily_sample_share=max(0.35, max_daily_sample_share),
        min_positive_day_rate=min_positive_day_rate,
        min_validation_expectancy_atr=min_validation_expectancy_atr,
        min_validation_win_rate=min_validation_win_rate,
        min_validation_samples=min_validation_samples,
        min_profit_factor=min_profit_factor,
    )
    qualified_shapes = [row for row in shape_rows if row["qualification"] == "PASS"]
    qualified_pair_shapes = [row for row in pair_shape_rows if row["qualification"] == "PASS"]
    qualified_features = [row for row in feature_rows if row["qualification"] == "PASS"]
    qualified_pair_features = [row for row in pair_feature_rows if row["qualification"] == "PASS"]
    qualified_pair_confluences = [row for row in pair_confluence_rows if row["qualification"] == "PASS"]
    qualified_multi_confluences = [row for row in multi_confluence_rows if row["qualification"] == "PASS"]
    qualified_directional_selectors = [
        row for row in directional_selector_rows if row["qualification"] == "PASS"
    ]
    qualified_inversion_selectors = [
        row for row in inversion_selector_rows if row["qualification"] == "PASS"
    ]
    high_precision_pair_confluences = [
        row
        for row in qualified_pair_confluences
        if (row.get("validation_win_rate") or 0.0) >= high_precision_min_win_rate
        and (row.get("validation_win_wilson95_lower") or 0.0) >= high_precision_min_wilson_lower
    ]
    high_precision_multi_confluences = [
        row
        for row in qualified_multi_confluences
        if (row.get("validation_win_rate") or 0.0) >= high_precision_min_win_rate
        and (row.get("validation_win_wilson95_lower") or 0.0) >= high_precision_min_wilson_lower
    ]
    high_precision_directional_selectors = [
        row
        for row in qualified_directional_selectors
        if (row.get("validation_win_rate") or 0.0) >= high_precision_min_win_rate
        and (row.get("validation_win_wilson95_lower") or 0.0) >= high_precision_min_wilson_lower
    ]
    high_precision_inversion_selectors = [
        row
        for row in qualified_inversion_selectors
        if (row.get("validation_win_rate") or 0.0) >= high_precision_min_win_rate
        and (row.get("validation_win_wilson95_lower") or 0.0) >= high_precision_min_wilson_lower
    ]
    return {
        "generated_at_utc": _iso(generated_at_utc),
        "source": str(history_root),
        "truth_source": (
            "local OANDA M5 bid/ask candles; LONG entry=ask exit=bid, "
            "SHORT entry=bid exit=ask; same-M5 TP+SL ambiguity counts as stop-first loss"
        ),
        "contract": {
            "prediction_unit": "universal normalized M5 entry shape",
            "volatility_normalization": "TP/SL in current M5 ATR multiples",
            "pair_selection": "same shape is scored across pairs; pair rows show where it currently worked",
            "live_permission": "audit evidence only; no live send gate is waived by this report",
        },
        "config": {
            "exit_shapes": [
                {"name": name, "take_profit_atr": tp, "stop_loss_atr": sl}
                for name, tp, sl in exit_shapes
            ],
            "max_hold_bars": max_hold_bars,
            "stride_bars": stride_bars,
            "tp_spread_floor": tp_spread_floor,
            "sl_spread_floor": sl_spread_floor,
            "train_fraction": train_fraction,
            "min_samples": min_samples,
            "min_active_days": min_active_days,
            "min_pair_count": min_pair_count,
            "max_pair_sample_share": max_pair_sample_share,
            "max_daily_sample_share": max_daily_sample_share,
            "min_positive_day_rate": min_positive_day_rate,
            "min_validation_expectancy_atr": min_validation_expectancy_atr,
            "min_validation_win_rate": min_validation_win_rate,
            "min_validation_samples": min_validation_samples,
            "min_profit_factor": min_profit_factor,
            "high_precision_min_win_rate": high_precision_min_win_rate,
            "high_precision_min_wilson_lower": high_precision_min_wilson_lower,
            "multi_confluence_sizes": list(multi_confluence_sizes),
            "multi_confluence_min_samples": multi_confluence_min_samples,
            "directional_selector_min_samples": max(
                DEFAULT_DIRECTIONAL_SELECTOR_MIN_SAMPLES,
                min_samples // 4,
            ),
            "directional_selector_feature_prefixes": list(DEFAULT_SELECTOR_FEATURE_PREFIXES),
            "inversion_selector_min_samples": max(
                DEFAULT_DIRECTIONAL_SELECTOR_MIN_SAMPLES,
                min_samples // 4,
            ),
            "inversion_selector_confluence_sizes": list(inversion_selector_sizes),
            "inversion_selector_feature_prefixes": list(DEFAULT_SELECTOR_FEATURE_PREFIXES),
        },
        **load_stats,
        "history_files_used": [str(path) for path in files],
        "summary": _summary(rows),
        "qualified_shape_count": len(qualified_shapes),
        "qualified_pair_shape_count": len(qualified_pair_shapes),
        "qualified_feature_count": len(qualified_features),
        "qualified_pair_feature_count": len(qualified_pair_features),
        "qualified_pair_confluence_count": len(qualified_pair_confluences),
        "high_precision_pair_confluence_count": len(high_precision_pair_confluences),
        "qualified_multi_confluence_count": len(qualified_multi_confluences),
        "high_precision_multi_confluence_count": len(high_precision_multi_confluences),
        "qualified_directional_selector_count": len(qualified_directional_selectors),
        "high_precision_directional_selector_count": len(high_precision_directional_selectors),
        "qualified_inversion_selector_count": len(qualified_inversion_selectors),
        "high_precision_inversion_selector_count": len(high_precision_inversion_selectors),
        "qualified_shapes": qualified_shapes[:top],
        "qualified_pair_shapes": qualified_pair_shapes[:top],
        "qualified_features": qualified_features[:top],
        "qualified_pair_features": qualified_pair_features[:top],
        "qualified_pair_confluences": qualified_pair_confluences[:top],
        "high_precision_pair_confluences": high_precision_pair_confluences[:top],
        "qualified_multi_confluences": qualified_multi_confluences[:top],
        "high_precision_multi_confluences": high_precision_multi_confluences[:top],
        "qualified_directional_selectors": qualified_directional_selectors[:top],
        "high_precision_directional_selectors": high_precision_directional_selectors[:top],
        "qualified_inversion_selectors": qualified_inversion_selectors[:top],
        "high_precision_inversion_selectors": high_precision_inversion_selectors[:top],
        "top_shapes": shape_rows[:top],
        "top_pair_shapes": pair_shape_rows[:top],
        "top_features": feature_rows[:top],
        "top_pair_features": pair_feature_rows[:top],
        "top_pair_confluences": pair_confluence_rows[:top],
        "top_multi_confluences": multi_confluence_rows[:top],
        "top_directional_selectors": directional_selector_rows[:top],
        "top_inversion_selectors": inversion_selector_rows[:top],
    }


def _summarize_buckets(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    train_fraction: float,
    min_samples: int,
    min_active_days: int,
    min_pair_count: int,
    max_pair_sample_share: float,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(field) for field in key_fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, values in buckets.items():
        if len(values) < min_samples:
            continue
        row = {field: key[idx] for idx, field in enumerate(key_fields)}
        row.update(
            _bucket_summary(
                values,
                train_fraction=train_fraction,
                min_active_days=min_active_days,
                min_pair_count=min_pair_count,
                max_pair_sample_share=max_pair_sample_share,
                max_daily_sample_share=max_daily_sample_share,
                min_positive_day_rate=min_positive_day_rate,
                min_validation_expectancy_atr=min_validation_expectancy_atr,
                min_validation_win_rate=min_validation_win_rate,
                min_validation_samples=min_validation_samples,
                min_profit_factor=min_profit_factor,
            )
        )
        out.append(row)
    out.sort(key=_bucket_sort_key)
    return out


def _summarize_feature_buckets(
    rows: list[dict[str, Any]],
    *,
    train_fraction: float,
    min_samples: int,
    min_active_days: int,
    min_pair_count: int,
    max_pair_sample_share: float,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        for feature in row.get("features") or ():
            if feature.startswith(("side:", "shape:")):
                continue
            buckets[(str(row.get("shape")), str(row.get("side")), str(row.get("exit_shape")), feature)].append(row)
    out: list[dict[str, Any]] = []
    for key, values in buckets.items():
        if len(values) < min_samples:
            continue
        shape, side, exit_shape, feature = key
        row = {"shape": shape, "side": side, "exit_shape": exit_shape, "feature": feature}
        row.update(
            _bucket_summary(
                values,
                train_fraction=train_fraction,
                min_active_days=min_active_days,
                min_pair_count=min_pair_count,
                max_pair_sample_share=max_pair_sample_share,
                max_daily_sample_share=max_daily_sample_share,
                min_positive_day_rate=min_positive_day_rate,
                min_validation_expectancy_atr=min_validation_expectancy_atr,
                min_validation_win_rate=min_validation_win_rate,
                min_validation_samples=min_validation_samples,
                min_profit_factor=min_profit_factor,
            )
        )
        out.append(row)
    out.sort(key=_bucket_sort_key)
    return out


def _summarize_pair_feature_buckets(
    rows: list[dict[str, Any]],
    *,
    train_fraction: float,
    min_samples: int,
    min_active_days: int,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        for feature in row.get("features") or ():
            if feature.startswith(("side:", "shape:")):
                continue
            buckets[
                (
                    str(row.get("pair")),
                    str(row.get("shape")),
                    str(row.get("side")),
                    str(row.get("exit_shape")),
                    feature,
                )
            ].append(row)
    out: list[dict[str, Any]] = []
    for key, values in buckets.items():
        if len(values) < min_samples:
            continue
        pair, shape, side, exit_shape, feature = key
        row = {
            "pair": pair,
            "shape": shape,
            "side": side,
            "exit_shape": exit_shape,
            "feature": feature,
        }
        row.update(
            _bucket_summary(
                values,
                train_fraction=train_fraction,
                min_active_days=min_active_days,
                min_pair_count=1,
                max_pair_sample_share=1.0,
                max_daily_sample_share=max_daily_sample_share,
                min_positive_day_rate=min_positive_day_rate,
                min_validation_expectancy_atr=min_validation_expectancy_atr,
                min_validation_win_rate=min_validation_win_rate,
                min_validation_samples=min_validation_samples,
                min_profit_factor=min_profit_factor,
            )
        )
        out.append(row)
    out.sort(key=_bucket_sort_key)
    return out


def _summarize_pair_confluence_buckets(
    rows: list[dict[str, Any]],
    *,
    train_fraction: float,
    min_samples: int,
    min_active_days: int,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        features = _pair_confluence_features(row.get("features") or ())
        for first_index, first in enumerate(features):
            for second in features[first_index + 1:]:
                buckets[
                    (
                        str(row.get("pair")),
                        str(row.get("shape")),
                        str(row.get("side")),
                        str(row.get("exit_shape")),
                        first,
                        second,
                    )
                ].append(row)
    out: list[dict[str, Any]] = []
    for key, values in buckets.items():
        if len(values) < min_samples:
            continue
        pair, shape, side, exit_shape, first, second = key
        row = {
            "pair": pair,
            "shape": shape,
            "side": side,
            "exit_shape": exit_shape,
            "feature_a": first,
            "feature_b": second,
            "confluence": f"{first} + {second}",
        }
        row.update(
            _bucket_summary(
                values,
                train_fraction=train_fraction,
                min_active_days=min_active_days,
                min_pair_count=1,
                max_pair_sample_share=1.0,
                max_daily_sample_share=max_daily_sample_share,
                min_positive_day_rate=min_positive_day_rate,
                min_validation_expectancy_atr=min_validation_expectancy_atr,
                min_validation_win_rate=min_validation_win_rate,
                min_validation_samples=min_validation_samples,
                min_profit_factor=min_profit_factor,
            )
        )
        out.append(row)
    out.sort(key=_bucket_sort_key)
    return out


def _summarize_pair_multi_confluence_buckets(
    rows: list[dict[str, Any]],
    *,
    confluence_size: int,
    train_fraction: float,
    min_samples: int,
    min_active_days: int,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, tuple[str, ...]], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        features = _pair_confluence_features(row.get("features") or ())
        for feature_group in itertools.combinations(features, confluence_size):
            buckets[
                (
                    str(row.get("pair")),
                    str(row.get("shape")),
                    str(row.get("side")),
                    str(row.get("exit_shape")),
                    feature_group,
                )
            ].append(row)
    out: list[dict[str, Any]] = []
    for key, values in buckets.items():
        if len(values) < min_samples:
            continue
        pair, shape, side, exit_shape, feature_group = key
        row = {
            "pair": pair,
            "shape": shape,
            "side": side,
            "exit_shape": exit_shape,
            "confluence_size": len(feature_group),
            "confluence": " + ".join(feature_group),
        }
        for index, feature in enumerate(feature_group):
            row[f"feature_{chr(ord('a') + index)}"] = feature
        row.update(
            _bucket_summary(
                values,
                train_fraction=train_fraction,
                min_active_days=min_active_days,
                min_pair_count=1,
                max_pair_sample_share=1.0,
                max_daily_sample_share=max_daily_sample_share,
                min_positive_day_rate=min_positive_day_rate,
                min_validation_expectancy_atr=min_validation_expectancy_atr,
                min_validation_win_rate=min_validation_win_rate,
                min_validation_samples=min_validation_samples,
                min_profit_factor=min_profit_factor,
            )
        )
        out.append(row)
    out.sort(key=_bucket_sort_key)
    return out


def _summarize_directional_selector_buckets(
    rows: list[dict[str, Any]],
    *,
    train_fraction: float,
    min_samples: int,
    min_active_days: int,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        features = _selector_confluence_features(row.get("neutral_features") or ())
        for first_index, first in enumerate(features):
            for second in features[first_index + 1:]:
                buckets[
                    (
                        str(row.get("pair")),
                        str(row.get("shape")),
                        str(row.get("exit_shape")),
                        first,
                        second,
                    )
                ].append(row)
    out: list[dict[str, Any]] = []
    for key, values in buckets.items():
        if len(values) < min_samples:
            continue
        pair, shape, exit_shape, first, second = key
        selected = _train_select_side_summary(
            values,
            train_fraction=train_fraction,
            min_active_days=min_active_days,
            max_daily_sample_share=max_daily_sample_share,
            min_positive_day_rate=min_positive_day_rate,
            min_validation_expectancy_atr=min_validation_expectancy_atr,
            min_validation_win_rate=min_validation_win_rate,
            min_validation_samples=min_validation_samples,
            min_profit_factor=min_profit_factor,
        )
        if selected is None:
            continue
        row = {
            "pair": pair,
            "shape": shape,
            "exit_shape": exit_shape,
            "feature_a": first,
            "feature_b": second,
            "confluence": f"{first} + {second}",
            "selected_side": selected.pop("selected_side"),
            "selection_basis": "train side with highest avg_realized_atr",
        }
        row.update(selected)
        out.append(row)
    out.sort(key=_bucket_sort_key)
    return out


def _summarize_inversion_selector_buckets(
    rows: list[dict[str, Any]],
    *,
    confluence_sizes: Sequence[int],
    train_fraction: float,
    min_samples: int,
    min_active_days: int,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str, tuple[str, ...]], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    sizes = tuple(sorted({int(size) for size in confluence_sizes if int(size) >= 2}))
    for row in rows:
        features = _selector_confluence_features(row.get("neutral_features") or ())
        for confluence_size in sizes:
            if len(features) < confluence_size:
                continue
            for feature_group in itertools.combinations(features, confluence_size):
                buckets[
                    (
                        str(row.get("pair")),
                        str(row.get("shape")),
                        str(row.get("source_side")),
                        str(row.get("selected_side") or row.get("side")),
                        str(row.get("exit_shape")),
                        feature_group,
                    )
                ].append(row)
    out: list[dict[str, Any]] = []
    for key, values in buckets.items():
        if len(values) < min_samples:
            continue
        pair, shape, source_side, selected_side, exit_shape, feature_group = key
        row = {
            "pair": pair,
            "shape": shape,
            "source_shape": shape,
            "source_side": source_side,
            "selected_side": selected_side,
            "side": selected_side,
            "exit_shape": exit_shape,
            "confluence_size": len(feature_group),
            "confluence": " + ".join(feature_group),
            "selection_basis": "same-candle opposite side of fired source shape",
        }
        for index, feature in enumerate(feature_group):
            row[f"feature_{chr(ord('a') + index)}"] = feature
        row.update(
            _inversion_bucket_summary(
                values,
                train_fraction=train_fraction,
                min_active_days=min_active_days,
                max_daily_sample_share=max_daily_sample_share,
                min_positive_day_rate=min_positive_day_rate,
                min_validation_expectancy_atr=min_validation_expectancy_atr,
                min_validation_win_rate=min_validation_win_rate,
                min_validation_samples=min_validation_samples,
                min_profit_factor=min_profit_factor,
            )
        )
        out.append(row)
    out.sort(key=_bucket_sort_key)
    return out


def _selector_confluence_features(features: Iterable[str]) -> list[str]:
    selected = []
    for feature in features:
        if not isinstance(feature, str):
            continue
        if feature.startswith(DEFAULT_SELECTOR_FEATURE_PREFIXES):
            selected.append(feature)
    return sorted(set(selected))


def _train_select_side_summary(
    values: list[dict[str, Any]],
    *,
    train_fraction: float,
    min_active_days: int,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
) -> dict[str, Any] | None:
    ordered = _ensure_chronological(values)
    split = int(len(ordered) * min(max(train_fraction, 0.1), 0.9))
    split = min(max(split, 1), len(ordered) - 1)
    train = ordered[:split]
    validation = ordered[split:]
    side_rows: list[tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = []
    for side in ("LONG", "SHORT"):
        train_side = [row for row in train if row.get("side") == side]
        validation_side = [row for row in validation if row.get("side") == side]
        if len(train_side) < min_validation_samples:
            continue
        train_summary = _metric_summary(train_side)
        side_rows.append((side, train_summary, train_side, validation_side))
    if not side_rows:
        return None
    side_rows.sort(
        key=lambda item: (
            -(item[1].get("avg_realized_atr") or 0.0),
            -(item[1].get("win_rate") or 0.0),
            -len(item[2]),
            item[0],
        )
    )
    side, train_summary, train_side, validation_side = side_rows[0]
    validation_summary = _metric_summary(validation_side)
    daily = _daily_stability(validation_side)
    blockers: list[str] = []
    if train_summary["avg_realized_atr"] <= 0.0:
        blockers.append("TRAIN_SELECTED_SIDE_NOT_POSITIVE")
    if validation_summary["n"] < min_validation_samples:
        blockers.append("INSUFFICIENT_VALIDATION_SAMPLES")
    if daily["active_days"] < min_active_days:
        blockers.append("INSUFFICIENT_ACTIVE_DAYS")
    if daily["max_daily_sample_share"] > max_daily_sample_share:
        blockers.append("DAILY_SAMPLE_CONCENTRATED")
    if daily["positive_day_rate"] < min_positive_day_rate:
        blockers.append("DAILY_EXPECTANCY_UNSTABLE")
    if validation_summary["avg_realized_atr"] <= min_validation_expectancy_atr:
        blockers.append("VALIDATION_EXPECTANCY_TOO_LOW")
    if validation_summary["win_rate"] < min_validation_win_rate:
        blockers.append("VALIDATION_WIN_RATE_TOO_LOW")
    pf = validation_summary["profit_factor"]
    if pf is not None and pf < min_profit_factor:
        blockers.append("VALIDATION_PROFIT_FACTOR_TOO_LOW")
    return {
        "qualification": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "selected_side": side,
        "split_at_utc": validation[0]["timestamp_utc"] if validation else None,
        "pair_count": 1,
        "max_pair_sample_share": 1.0,
        **_prefix("train", train_summary),
        **_prefix("validation", validation_summary),
        **daily,
    }


def _inversion_bucket_summary(
    values: list[dict[str, Any]],
    *,
    train_fraction: float,
    min_active_days: int,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
) -> dict[str, Any]:
    ordered = _ensure_chronological(values)
    split = int(len(ordered) * min(max(train_fraction, 0.1), 0.9))
    split = min(max(split, 1), len(ordered) - 1)
    train = ordered[:split]
    validation = ordered[split:]
    train_summary = _metric_summary(train)
    validation_summary = _metric_summary(validation)
    all_summary = _metric_summary(ordered)
    source_train_summary = _source_metric_summary(train)
    source_validation_summary = _source_metric_summary(validation)
    source_all_summary = _source_metric_summary(ordered)
    daily = _daily_stability(validation)
    inversion_edge_atr = (
        validation_summary["avg_realized_atr"] - source_validation_summary["avg_realized_atr"]
    )
    blockers: list[str] = []
    if validation_summary["n"] <= 0:
        blockers.append("NO_VALIDATION")
    if validation_summary["n"] < min_validation_samples:
        blockers.append("INSUFFICIENT_VALIDATION_SAMPLES")
    if daily["active_days"] < min_active_days:
        blockers.append("INSUFFICIENT_ACTIVE_DAYS")
    if daily["max_daily_sample_share"] > max_daily_sample_share:
        blockers.append("DAILY_SAMPLE_CONCENTRATED")
    if daily["positive_day_rate"] < min_positive_day_rate:
        blockers.append("DAILY_EXPECTANCY_UNSTABLE")
    if train_summary["avg_realized_atr"] <= min_validation_expectancy_atr:
        blockers.append("TRAIN_INVERSION_EXPECTANCY_TOO_LOW")
    if source_train_summary["avg_realized_atr"] >= 0.0:
        blockers.append("TRAIN_SOURCE_NOT_NEGATIVE")
    if validation_summary["avg_realized_atr"] <= min_validation_expectancy_atr:
        blockers.append("VALIDATION_EXPECTANCY_TOO_LOW")
    if validation_summary["win_rate"] < min_validation_win_rate:
        blockers.append("VALIDATION_WIN_RATE_TOO_LOW")
    if source_validation_summary["avg_realized_atr"] >= 0.0:
        blockers.append("VALIDATION_SOURCE_NOT_NEGATIVE")
    if inversion_edge_atr <= min_validation_expectancy_atr:
        blockers.append("VALIDATION_INVERSION_EDGE_TOO_LOW")
    pf = validation_summary["profit_factor"]
    if pf is not None and pf < min_profit_factor:
        blockers.append("VALIDATION_PROFIT_FACTOR_TOO_LOW")
    return {
        "qualification": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "split_at_utc": validation[0]["timestamp_utc"] if validation else None,
        "pair_count": 1,
        "max_pair_sample_share": 1.0,
        "validation_inversion_edge_atr": _round_value(inversion_edge_atr),
        **_prefix("train", train_summary),
        **_prefix("validation", validation_summary),
        **_prefix("all", all_summary),
        **_prefix("source_train", source_train_summary),
        **_prefix("source_validation", source_validation_summary),
        **_prefix("source_all", source_all_summary),
        **daily,
    }


def _source_metric_summary(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    source_rows = [
        {
            "realized_pips": item.get("source_realized_pips"),
            "realized_atr": item.get("source_realized_atr"),
            "win": item.get("source_win"),
            "outcome": item.get("source_outcome"),
        }
        for item in values
    ]
    return _metric_summary(source_rows)


def _pair_confluence_features(features: Iterable[str]) -> list[str]:
    selected = []
    for feature in features:
        if not isinstance(feature, str):
            continue
        if feature.startswith(("side:", "shape:", "side_range:")):
            continue
        if feature.startswith(
            (
                "session:",
                "atr_regime:",
                "spread_regime:",
                "range_pos:",
                "fast_mom:",
                "slow_mom:",
                "body:",
                "wick_reject:",
                "bar_range:",
                "failed_break:",
            )
        ):
            selected.append(feature)
    return sorted(set(selected))


def _bucket_summary(
    values: list[dict[str, Any]],
    *,
    train_fraction: float,
    min_active_days: int,
    min_pair_count: int,
    max_pair_sample_share: float,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
    min_validation_expectancy_atr: float,
    min_validation_win_rate: float,
    min_validation_samples: int,
    min_profit_factor: float,
) -> dict[str, Any]:
    ordered = _ensure_chronological(values)
    split = int(len(ordered) * min(max(train_fraction, 0.1), 0.9))
    split = min(max(split, 1), len(ordered) - 1)
    train = ordered[:split]
    validation = ordered[split:]
    train_summary = _metric_summary(train)
    validation_summary = _metric_summary(validation)
    all_summary = _metric_summary(ordered)
    daily = _daily_stability(validation)
    pair_counts = collections.Counter(item["pair"] for item in validation)
    pair_count = len(pair_counts)
    max_pair_share = max(pair_counts.values()) / len(validation) if validation else 1.0
    blockers: list[str] = []
    if validation_summary["n"] <= 0:
        blockers.append("NO_VALIDATION")
    if validation_summary["n"] < min_validation_samples:
        blockers.append("INSUFFICIENT_VALIDATION_SAMPLES")
    if pair_count < min_pair_count:
        blockers.append("INSUFFICIENT_PAIR_BREADTH")
    if max_pair_share > max_pair_sample_share:
        blockers.append("PAIR_SAMPLE_CONCENTRATED")
    if daily["active_days"] < min_active_days:
        blockers.append("INSUFFICIENT_ACTIVE_DAYS")
    if daily["max_daily_sample_share"] > max_daily_sample_share:
        blockers.append("DAILY_SAMPLE_CONCENTRATED")
    if daily["positive_day_rate"] < min_positive_day_rate:
        blockers.append("DAILY_EXPECTANCY_UNSTABLE")
    if validation_summary["avg_realized_atr"] <= min_validation_expectancy_atr:
        blockers.append("VALIDATION_EXPECTANCY_TOO_LOW")
    if validation_summary["win_rate"] < min_validation_win_rate:
        blockers.append("VALIDATION_WIN_RATE_TOO_LOW")
    pf = validation_summary["profit_factor"]
    if pf is not None and pf < min_profit_factor:
        blockers.append("VALIDATION_PROFIT_FACTOR_TOO_LOW")
    return {
        "qualification": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "split_at_utc": validation[0]["timestamp_utc"] if validation else None,
        "pair_count": pair_count,
        "max_pair_sample_share": round(max_pair_share, 6),
        **_prefix("train", train_summary),
        **_prefix("validation", validation_summary),
        **_prefix("all", all_summary),
        **daily,
        "top_pairs": [
            {"pair": pair, "samples": count}
            for pair, count in pair_counts.most_common(8)
        ],
    }


def _ensure_chronological(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous: Any = None
    for item in values:
        current = item["timestamp_utc"]
        if previous is not None and current < previous:
            return sorted(values, key=lambda row: row["timestamp_utc"])
        previous = current
    return values


def _metric_summary(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(values)
    wins = sum(1 for item in values if item.get("win"))
    pips = [float(item.get("realized_pips") or 0.0) for item in values]
    atrs = [float(item.get("realized_atr") or 0.0) for item in values]
    gains = sum(value for value in pips if value > 0.0)
    losses = -sum(value for value in pips if value < 0.0)
    outcomes = collections.Counter(str(item.get("outcome") or "") for item in values)
    return {
        "n": n,
        "win_rate": wins / n if n else 0.0,
        "win_wilson95_lower": _wilson_lower(wins, n),
        "avg_realized_pips": statistics.fmean(pips) if pips else 0.0,
        "median_realized_pips": statistics.median(pips) if pips else 0.0,
        "avg_realized_atr": statistics.fmean(atrs) if atrs else 0.0,
        "median_realized_atr": statistics.median(atrs) if atrs else 0.0,
        "profit_factor": None if losses <= 0.0 else gains / losses,
        "tp_first_rate": outcomes["TAKE_PROFIT_FIRST"] / n if n else 0.0,
        "stop_first_rate": outcomes["STOP_FIRST"] / n if n else 0.0,
        "timeout_rate": outcomes["TIMEOUT"] / n if n else 0.0,
        "ambiguous_rate": outcomes["AMBIGUOUS_SAME_M5"] / n if n else 0.0,
    }


def _daily_stability(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in values:
        by_day[str(item.get("jst_day"))].append(item)
    day_expectancies = []
    positive_days = 0
    max_share = 1.0
    if values:
        max_share = max(len(items) for items in by_day.values()) / len(values)
    for day, items in sorted(by_day.items()):
        avg = statistics.fmean(float(item.get("realized_atr") or 0.0) for item in items)
        if avg > 0.0:
            positive_days += 1
        day_expectancies.append({"jst_day": day, "n": len(items), "avg_realized_atr": round(avg, 6)})
    active_days = len(by_day)
    return {
        "active_days": active_days,
        "positive_day_rate": positive_days / active_days if active_days else 0.0,
        "max_daily_sample_share": round(max_share, 6),
        "validation_days_tail": day_expectancies[-10:],
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair = collections.Counter(row["pair"] for row in rows)
    by_shape = collections.Counter(row["shape"] for row in rows)
    by_exit = collections.Counter(row["exit_shape"] for row in rows)
    return {
        "scored_outcomes": len(rows),
        "pairs": len(by_pair),
        "shapes": dict(by_shape.most_common()),
        "exit_shapes": dict(by_exit.most_common()),
        "top_pairs": dict(by_pair.most_common(12)),
    }


def _prefix(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": _round_value(value) for key, value in summary.items()}


def _bucket_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["qualification"] != "PASS",
        -(row.get("validation_avg_realized_atr") or 0.0),
        -_profit_factor_sort(row.get("validation_profit_factor")),
        -(row.get("validation_win_wilson95_lower") or 0.0),
        -(row.get("validation_n") or 0),
        row.get("shape") or "",
        row.get("side") or "",
    )


def _profit_factor_sort(value: Any) -> float:
    if value is None:
        return 1_000_000.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _atr_pips(candles: Sequence[BaOhlc], *, factor: int, period: int) -> list[float | None]:
    true_ranges: list[float] = []
    out: list[float | None] = [None] * len(candles)
    prev_close: float | None = None
    for idx, candle in enumerate(candles):
        high = candle.mid_h
        low = candle.mid_l
        if prev_close is None:
            tr = (high - low) * factor
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close)) * factor
        prev_close = candle.mid_c
        true_ranges.append(max(0.0, tr))
        if idx + 1 >= period:
            out[idx] = statistics.fmean(true_ranges[idx + 1 - period: idx + 1])
    return out


def _failed_break(candles: Sequence[BaOhlc], idx: int, *, side: str, atr_pips: float) -> bool:
    if idx < RANGE_LOOKBACK + 1:
        return False
    prev_window = candles[idx - RANGE_LOOKBACK: idx]
    high = max(item.mid_h for item in prev_window)
    low = min(item.mid_l for item in prev_window)
    c = candles[idx]
    width = max(high - low, 1e-12)
    inside_buffer = width * 0.05
    if side == "LONG":
        return c.mid_l < low and c.mid_c > low + inside_buffer
    return c.mid_h > high and c.mid_c < high - inside_buffer


def _range_pos_bucket(value: float) -> str:
    if value <= 0.25:
        return "low"
    if value >= 0.75:
        return "high"
    return "mid"


def _signed_bucket(value: float) -> str:
    if value >= 0.15:
        return "aligned"
    if value <= -0.15:
        return "opposed"
    return "flat"


def _absolute_signed_bucket(value: float) -> str:
    if value >= 0.15:
        return "up"
    if value <= -0.15:
        return "down"
    return "flat"


def _session(timestamp: datetime) -> str:
    hour = timestamp.astimezone(timezone.utc).hour
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "london_open"
    if 12 <= hour < 16:
        return "london_ny_overlap"
    if 16 <= hour < 21:
        return "ny"
    return "rollover"


def _quantile_boundaries(values: Sequence[float], *, lower: float, upper: float) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    return (_quantile(ordered, lower), _quantile(ordered, upper))


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    pos = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _wilson_lower(successes: int, trials: int, *, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0
    p_hat = successes / trials
    denom = 1.0 + z * z / trials
    centre = p_hat + z * z / (2.0 * trials)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * trials)) / trials)
    return max(0.0, min(1.0, (centre - margin) / denom))


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text and text.endswith("+00:00"):
        head, tail = text.split(".", 1)
        fraction, zone = tail.split("+", 1)
        text = f"{head}.{fraction[:6]}+{zone}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    return value


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# OANDA Universal Rotation Mining",
        "",
        f"- generated_at_utc: `{report.get('generated_at_utc')}`",
        f"- history_pairs: `{report.get('history_pairs')}`",
        f"- scored_outcomes: `{report.get('scored_outcomes')}`",
        f"- qualified_shape_count: `{report.get('qualified_shape_count')}`",
        f"- qualified_pair_shape_count: `{report.get('qualified_pair_shape_count')}`",
        f"- qualified_pair_feature_count: `{report.get('qualified_pair_feature_count')}`",
        f"- qualified_pair_confluence_count: `{report.get('qualified_pair_confluence_count')}`",
        f"- high_precision_pair_confluence_count: `{report.get('high_precision_pair_confluence_count')}`",
        f"- qualified_multi_confluence_count: `{report.get('qualified_multi_confluence_count')}`",
        f"- high_precision_multi_confluence_count: `{report.get('high_precision_multi_confluence_count')}`",
        f"- qualified_directional_selector_count: `{report.get('qualified_directional_selector_count')}`",
        f"- high_precision_directional_selector_count: `{report.get('high_precision_directional_selector_count')}`",
        f"- qualified_inversion_selector_count: `{report.get('qualified_inversion_selector_count')}`",
        f"- high_precision_inversion_selector_count: `{report.get('high_precision_inversion_selector_count')}`",
        "",
        "## Qualified Universal Shapes",
        "",
    ]
    lines.extend(_table(report.get("qualified_shapes") or []))
    lines.extend(["", "## Qualified Pair Shapes", ""])
    lines.extend(_table(report.get("qualified_pair_shapes") or []))
    lines.extend(["", "## Qualified Pair Features", ""])
    lines.extend(_table(report.get("qualified_pair_features") or []))
    lines.extend(["", "## High Precision Pair Confluences", ""])
    lines.extend(_table(report.get("high_precision_pair_confluences") or []))
    lines.extend(["", "## High Precision Multi Confluences", ""])
    lines.extend(_table(report.get("high_precision_multi_confluences") or []))
    lines.extend(["", "## High Precision Directional Selectors", ""])
    lines.extend(_table(report.get("high_precision_directional_selectors") or []))
    lines.extend(["", "## High Precision Inversion Selectors", ""])
    lines.extend(_table(report.get("high_precision_inversion_selectors") or []))
    lines.extend(["", "## Qualified Pair Confluences", ""])
    lines.extend(_table(report.get("qualified_pair_confluences") or []))
    lines.extend(["", "## Qualified Multi Confluences", ""])
    lines.extend(_table(report.get("qualified_multi_confluences") or []))
    lines.extend(["", "## Qualified Directional Selectors", ""])
    lines.extend(_table(report.get("qualified_directional_selectors") or []))
    lines.extend(["", "## Qualified Inversion Selectors", ""])
    lines.extend(_table(report.get("qualified_inversion_selectors") or []))
    lines.extend(["", "## Top Shapes", ""])
    lines.extend(_table(report.get("top_shapes") or []))
    lines.extend(["", "## Top Pair Features", ""])
    lines.extend(_table(report.get("top_pair_features") or []))
    lines.extend(["", "## Top Pair Confluences", ""])
    lines.extend(_table(report.get("top_pair_confluences") or []))
    lines.extend(["", "## Top Multi Confluences", ""])
    lines.extend(_table(report.get("top_multi_confluences") or []))
    lines.extend(["", "## Top Directional Selectors", ""])
    lines.extend(_table(report.get("top_directional_selectors") or []))
    lines.extend(["", "## Top Inversion Selectors", ""])
    lines.extend(_table(report.get("top_inversion_selectors") or []))
    lines.append("")
    return "\n".join(lines)


def _table(rows: Sequence[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["none"]
    keys = [
        "pair",
        "shape",
        "side",
        "source_side",
        "selected_side",
        "exit_shape",
        "confluence_size",
        "feature",
        "confluence",
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
        "feature_e",
        "feature_f",
        "feature_g",
        "feature_h",
        "qualification",
        "validation_n",
        "validation_win_rate",
        "validation_avg_realized_pips",
        "validation_avg_realized_atr",
        "validation_profit_factor",
        "source_validation_win_rate",
        "source_validation_avg_realized_atr",
        "validation_inversion_edge_atr",
        "active_days",
        "positive_day_rate",
        "pair_count",
        "blockers",
    ]
    visible = [key for key in keys if any(key in row for row in rows)]
    lines = [
        "| " + " | ".join(visible) + " |",
        "| " + " | ".join("---" for _ in visible) + " |",
    ]
    for row in rows[:30]:
        values = []
        for key in visible:
            value = row.get(key)
            if isinstance(value, list):
                value = ",".join(str(item) for item in value[:3])
            values.append(f"`{value}`")
        lines.append("| " + " | ".join(values) + " |")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())

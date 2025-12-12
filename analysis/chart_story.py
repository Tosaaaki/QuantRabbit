"""
analysis.chart_story
~~~~~~~~~~~~~~~~~~~~~
ローソク足の履歴から複数タイムフレームの「ストーリー」を要約するヘルパ。

現在は factor_cache で取得できる M1 / H4 のローソクを基に、
 M5 / M15 / H1 / H4 / D1 の傾向を簡易推定し、
 エントリー/クローズ時に参照できるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_PIP = 0.01


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_candles(raw: Optional[Iterable[dict]]) -> List[Tuple[float, float, float, float]]:
    candles: List[Tuple[float, float, float, float]] = []
    if not raw:
        return candles
    for candle in raw:
        o = _safe_float(candle.get("open"))
        h = _safe_float(candle.get("high"), o)
        l = _safe_float(candle.get("low"), o)
        c = _safe_float(candle.get("close"), o)
        candles.append((o, h, l, c))
    return candles


def _aggregate(candles: Sequence[Tuple[float, float, float, float]], group: int) -> List[Tuple[float, float, float, float]]:
    if group <= 1 or not candles:
        return list(candles)
    aggregated: List[Tuple[float, float, float, float]] = []
    bucket: List[Tuple[float, float, float, float]] = []
    for candle in candles:
        bucket.append(candle)
        if len(bucket) == group:
            o = bucket[0][0]
            h = max(c[1] for c in bucket)
            l = min(c[2] for c in bucket)
            c = bucket[-1][3]
            aggregated.append((o, h, l, c))
            bucket.clear()
    return aggregated


def _slope(candles: Sequence[Tuple[float, float, float, float]]) -> float:
    if len(candles) < 2:
        return 0.0
    start = candles[0][3]
    end = candles[-1][3]
    return (end - start) / _PIP


def _blended_slope(
    candles: Sequence[Tuple[float, float, float, float]],
    short_count: int,
    long_count: int,
    short_weight: float = 0.6,
) -> float:
    if not candles:
        return 0.0
    short = _slope(candles[-max(short_count, 2):])
    long = _slope(candles[-max(long_count, 3):])
    weight = max(0.0, min(1.0, short_weight))
    return short * weight + long * (1.0 - weight)


def _volatility(candles: Sequence[Tuple[float, float, float, float]]) -> float:
    if not candles:
        return 0.0
    ranges = [(c[1] - c[2]) / _PIP for c in candles]
    return mean(ranges) if ranges else 0.0


def _pivot_levels(
    candles: Sequence[Tuple[float, float, float, float]]
) -> Dict[str, float]:
    if len(candles) < 2:
        return {}
    prev = candles[-2]
    high = prev[1]
    low = prev[2]
    close = prev[3]
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    range_span = max(high - low, 0.0001)
    fib50 = low + range_span * 0.5
    fib618 = low + range_span * 0.618
    return {
        "pivot": round(pivot, 3),
        "r1": round(r1, 3),
        "s1": round(s1, 3),
        "fib50": round(fib50, 3),
        "fib61": round(fib618, 3),
    }


def _recent_extremes(
    candles: Sequence[Tuple[float, float, float, float]],
    count: int,
) -> Dict[str, float]:
    if not candles:
        return {}
    window = candles[-min(count, len(candles)) :]
    highs = [c[1] for c in window]
    lows = [c[2] for c in window]
    return {"recent_high": round(max(highs), 3), "recent_low": round(min(lows), 3)}


def _swing_points(
    candles: Sequence[Tuple[float, float, float, float]],
    lookback: int = 3,
    min_gap_pips: float = 8.0,
) -> List[Dict[str, float]]:
    if len(candles) < lookback * 2 + 1:
        return []
    swings: List[Dict[str, float]] = []
    price_gap = min_gap_pips * _PIP

    def _append(kind: str, price: float, idx: int) -> None:
        if swings:
            prev = swings[-1]
            if prev["kind"] == kind:
                if (kind == "high" and price > prev["price"]) or (
                    kind == "low" and price < prev["price"]
                ):
                    swings[-1] = {"kind": kind, "price": price, "index": idx}
                return
            if abs(price - prev["price"]) < price_gap:
                return
        swings.append({"kind": kind, "price": price, "index": idx})

    for i in range(lookback, len(candles) - lookback):
        segment = candles[i - lookback : i + lookback + 1]
        high = max(c[1] for c in segment)
        low = min(c[2] for c in segment)
        current_high = candles[i][1]
        current_low = candles[i][2]
        if current_high >= high:
            _append("high", current_high, i)
            continue
        if current_low <= low:
            _append("low", current_low, i)
    return swings[-12:]


def _detect_n_wave(
    candles: Sequence[Tuple[float, float, float, float]],
) -> Optional[Dict[str, object]]:
    swings = _swing_points(candles, lookback=3, min_gap_pips=10.0)
    if len(swings) < 4:
        return None
    last = swings[-4:]
    pattern = None
    confidence = 0.0

    def _summarize(sw) -> List[Dict[str, float]]:
        return [
            {"kind": s["kind"], "price": round(s["price"], 3), "index": s["index"]}
            for s in sw
        ]

    bias: Optional[str] = None

    if (
        last[0]["kind"] == "low"
        and last[1]["kind"] == "high"
        and last[2]["kind"] == "low"
        and last[3]["kind"] == "high"
    ):
        higher_low = last[2]["price"] > last[0]["price"]
        higher_high = last[3]["price"] > last[1]["price"]
        if higher_low and higher_high:
            leg_a = (last[1]["price"] - last[0]["price"]) / _PIP
            leg_b = (last[3]["price"] - last[2]["price"]) / _PIP
            swing_depth = (last[1]["price"] - last[2]["price"]) / _PIP
            confidence = min(1.0, max(leg_a, leg_b) / 18.0)
            confidence = max(confidence, min(1.0, swing_depth / 12.0))
            bias = "up"
            pattern = {
                "bias": "up",
                "direction": "up",
                "confidence": round(confidence, 3),
                "structure": {
                    "legs": [round(leg_a, 1), round(swing_depth, 1), round(leg_b, 1)],
                    "higher_high": True,
                    "higher_low": True,
                },
                "swings": _summarize(last),
            }
    elif (
        last[0]["kind"] == "high"
        and last[1]["kind"] == "low"
        and last[2]["kind"] == "high"
        and last[3]["kind"] == "low"
    ):
        lower_high = last[2]["price"] < last[0]["price"]
        lower_low = last[3]["price"] < last[1]["price"]
        if lower_high and lower_low:
            leg_a = (last[0]["price"] - last[1]["price"]) / _PIP
            leg_b = (last[2]["price"] - last[3]["price"]) / _PIP
            swing_depth = (last[2]["price"] - last[1]["price"]) / _PIP
            confidence = min(1.0, max(leg_a, leg_b) / 18.0)
            confidence = max(confidence, min(1.0, swing_depth / 12.0))
            bias = "down"
            pattern = {
                "bias": "down",
                "direction": "down",
                "confidence": round(confidence, 3),
                "structure": {
                    "legs": [round(leg_a, 1), round(swing_depth, 1), round(leg_b, 1)],
                    "lower_high": True,
                    "lower_low": True,
                },
                "swings": _summarize(last),
            }
    if pattern:
        pattern["timeframe"] = "H4"
        if bias and "direction" not in pattern:
            pattern["direction"] = bias
        return pattern
    return None


def _detect_candlestick_pattern(
    candles: Sequence[Tuple[float, float, float, float]],
) -> Optional[Dict[str, object]]:
    if len(candles) < 2:
        return None
    o0, h0, l0, c0 = candles[-2]
    o1, h1, l1, c1 = candles[-1]
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    range1 = max(h1 - l1, _PIP * 0.1)
    upper_wick = h1 - max(o1, c1)
    lower_wick = min(o1, c1) - l1
    pattern_type: Optional[str] = None
    confidence = 0.0

    bias: Optional[str] = None

    if (
        c1 > o1
        and c0 < o0
        and c1 >= max(o0, c0)
        and o1 <= min(o0, c0)
        and body1 > body0
    ):
        pattern_type = "bullish_engulfing"
        confidence = min(1.0, body1 / range1 + 0.3)
        bias = "up"
    elif (
        c1 < o1
        and c0 > o0
        and o1 >= min(o0, c0)
        and c1 <= max(o0, c0)
        and body1 > body0
    ):
        pattern_type = "bearish_engulfing"
        confidence = min(1.0, body1 / range1 + 0.3)
        bias = "down"
    elif lower_wick > body1 * 2.5 and upper_wick <= body1 * 0.6:
        pattern_type = "hammer" if c1 >= o1 else "inverted_hammer"
        confidence = min(1.0, lower_wick / range1 + 0.25)
        bias = "up"
    elif upper_wick > body1 * 2.5 and lower_wick <= body1 * 0.6:
        pattern_type = "shooting_star" if c1 <= o1 else "hanging_man"
        confidence = min(1.0, upper_wick / range1 + 0.25)
        bias = "down"

    if pattern_type:
        return {
            "timeframe": "H1",
            "type": pattern_type,
            "confidence": round(min(confidence, 1.0), 3),
            "body_pips": round(body1 / _PIP, 2),
            "bias": bias,
        }
    return None


def _trend_state(
    slope_pips: float,
    vol_pips: float,
    slope_threshold: float = 4.0,
    *,
    vol_sensitivity: float = 0.0,
) -> str:
    dynamic_threshold = slope_threshold
    if vol_sensitivity > 0.0:
        dynamic_threshold += max(0.0, vol_pips - 6.0) * vol_sensitivity
    if slope_pips >= dynamic_threshold:
        return "up"
    if slope_pips <= -dynamic_threshold:
        return "down"
    if vol_pips <= 3.0:
        return "quiet"
    return "range"


@dataclass(slots=True)
class ChartStorySnapshot:
    macro_trend: str
    micro_trend: str
    higher_trend: str
    structure_bias: float
    volatility_state: str
    summary: Dict[str, str]
    major_levels: Dict[str, Dict[str, float]]
    pattern_summary: Dict[str, object] = field(default_factory=dict)

    def is_aligned(self, pocket: str, action: str) -> bool:
        if pocket == "macro":
            ref = self.macro_trend
        elif pocket == "micro":
            ref = self.micro_trend
        else:
            ref = self.higher_trend
        if not ref or ref in {"range", "quiet"}:
            return True
        if action == "OPEN_LONG":
            return ref == "up"
        if action == "OPEN_SHORT":
            return ref == "down"
        return True


class ChartStory:
    """逐次更新しながらチャートの文脈を保持する。"""

    def __init__(self) -> None:
        self._last_snapshot: Optional[ChartStorySnapshot] = None

    def update(self, fac_m1: Dict[str, object], fac_h4: Dict[str, object]) -> Optional[ChartStorySnapshot]:
        candles_m1 = _extract_candles(fac_m1.get("candles"))
        if len(candles_m1) < 30:
            return self._last_snapshot

        candles_h4 = _extract_candles(fac_h4.get("candles"))
        m5 = _aggregate(candles_m1, 5)
        m15 = _aggregate(candles_m1, 15)
        h1 = _aggregate(candles_m1, 60)

        if not candles_h4:
            candles_h4 = _aggregate(h1, 4)
        d1 = _aggregate(candles_h4, 6)

        slope_m5 = _slope(m5[-12:])
        slope_m15 = _slope(m15[-8:])
        slope_h1 = _blended_slope(h1, short_count=4, long_count=12, short_weight=0.65)
        slope_h4 = _blended_slope(candles_h4, short_count=3, long_count=8, short_weight=0.55)
        slope_d1 = _blended_slope(d1, short_count=3, long_count=8, short_weight=0.5)

        vol_m5 = _volatility(m5[-12:])
        vol_h1 = _volatility(h1[-6:])

        macro_mix = (slope_h1 + slope_h4) / 2.0
        higher_mix = (slope_h4 + slope_d1) / 2.0

        micro_trend = _trend_state(slope_m5, vol_m5, slope_threshold=2.5, vol_sensitivity=0.25)
        macro_trend = _trend_state(macro_mix, vol_h1, slope_threshold=5.0, vol_sensitivity=0.35)
        higher_trend = _trend_state(higher_mix, vol_h1, slope_threshold=6.0, vol_sensitivity=0.35)

        structure_bias = (macro_mix + higher_mix) / 2.0
        volatility_state = "high" if vol_h1 > 8.0 else ("low" if vol_h1 < 3.0 else "normal")

        summary = {
            "M5": micro_trend,
            "M15": _trend_state(slope_m15, vol_m5, slope_threshold=3.5, vol_sensitivity=0.2),
            "H1": _trend_state(slope_h1, vol_h1, slope_threshold=5.0, vol_sensitivity=0.3),
            "H4": _trend_state(slope_h4, vol_h1, slope_threshold=6.0, vol_sensitivity=0.35),
            "D1": _trend_state(slope_d1, vol_h1, slope_threshold=6.5, vol_sensitivity=0.35),
        }

        levels = {
            "h4": {},
            "d1": {},
        }
        h4_levels = _pivot_levels(candles_h4)
        d1_levels = _pivot_levels(d1)
        if h4_levels:
            h4_levels.update(_recent_extremes(candles_h4, 12))
            levels["h4"] = h4_levels
        if d1_levels:
            d1_levels.update(_recent_extremes(d1, 6))
            levels["d1"] = d1_levels

        pattern_summary: Dict[str, object] = {}
        n_wave = _detect_n_wave(candles_h4 or h1)
        if n_wave:
            pattern_summary["n_wave"] = n_wave
        candle_pattern = _detect_candlestick_pattern(h1)
        if candle_pattern:
            pattern_summary["candlestick"] = candle_pattern

        snapshot = ChartStorySnapshot(
            macro_trend=macro_trend,
            micro_trend=micro_trend,
            higher_trend=higher_trend,
            structure_bias=structure_bias,
            volatility_state=volatility_state,
            summary=summary,
            major_levels=levels,
            pattern_summary=pattern_summary,
        )
        self._last_snapshot = snapshot
        return snapshot

    @property
    def last_snapshot(self) -> Optional[ChartStorySnapshot]:
        return self._last_snapshot

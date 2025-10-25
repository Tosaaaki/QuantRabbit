"""
analysis.chart_story
~~~~~~~~~~~~~~~~~~~~~
ローソク足の履歴から複数タイムフレームの「ストーリー」を要約するヘルパ。

現在は factor_cache で取得できる M1 / H4 のローソクを基に、
 M5 / M15 / H1 / H4 / D1 の傾向を簡易推定し、
 エントリー/クローズ時に参照できるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass
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

        snapshot = ChartStorySnapshot(
            macro_trend=macro_trend,
            micro_trend=micro_trend,
            higher_trend=higher_trend,
            structure_bias=structure_bias,
            volatility_state=volatility_state,
            summary=summary,
        )
        self._last_snapshot = snapshot
        return snapshot

    @property
    def last_snapshot(self) -> Optional[ChartStorySnapshot]:
        return self._last_snapshot

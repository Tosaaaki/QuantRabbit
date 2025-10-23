"""
analysis.ma_projection
~~~~~~~~~~~~~~~~~~~~~~
Utility helpers to estimate moving-average and MACD cross dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

PIP = 0.01


@dataclass
class MACrossProjection:
    fast_ma: float
    slow_ma: float
    gap_pips: float
    prev_gap_pips: float
    gap_slope_pips: float
    fast_slope_pips: float
    slow_slope_pips: float
    price_to_fast_pips: float
    price_to_slow_pips: float
    projected_cross_bars: Optional[float]
    projected_cross_minutes: Optional[float]
    macd_pips: Optional[float] = None
    macd_slope_pips: Optional[float] = None
    macd_cross_bars: Optional[float] = None
    macd_cross_minutes: Optional[float] = None


def compute_ma_projection(
    fac: Dict,
    timeframe_minutes: float = 1.0,
    fast_window: int = 10,
    slow_window: int = 20,
) -> Optional[MACrossProjection]:
    """
    Calculate moving-average gap dynamics plus MACD cross projection.

    Parameters
    ----------
    fac:
        Factor dictionary that includes a `candles` deque/list entry.
    timeframe_minutes:
        Duration (in minutes) represented by each candle. Defaults to M1.
    fast_window / slow_window:
        Moving-average lengths that correspond to ma10/ma20 within the system.
    """

    candles = fac.get("candles")
    if not isinstance(candles, Iterable):
        return None

    closes: List[float] = []
    for c in candles:
        try:
            closes.append(float(c["close"]))
        except (TypeError, KeyError, ValueError):
            continue

    need = max(slow_window + 2, 30)
    if len(closes) < need:
        return None

    fast = _simple_ma(closes, fast_window)
    slow = _simple_ma(closes, slow_window)
    prev_fast = _simple_ma(closes[:-1], fast_window)
    prev_slow = _simple_ma(closes[:-1], slow_window)

    if fast is None or slow is None or prev_fast is None or prev_slow is None:
        return None

    gap = fast - slow
    prev_gap = prev_fast - prev_slow
    gap_pips = gap / PIP
    prev_gap_pips = prev_gap / PIP

    fast_slope = (fast - prev_fast) / PIP
    slow_slope = (slow - prev_slow) / PIP
    gap_slope = (gap - prev_gap) / PIP

    price = float(closes[-1])
    price_to_fast = (price - fast) / PIP
    price_to_slow = (price - slow) / PIP

    projected_cross_bars: Optional[float] = None
    projected_cross_minutes: Optional[float] = None
    if gap_slope != 0.0 and gap_pips * gap_slope < 0:
        projected_cross_bars = abs(gap_pips) / abs(gap_slope)
        projected_cross_minutes = projected_cross_bars * max(timeframe_minutes, 1e-6)

    macd_pips: Optional[float] = None
    macd_slope_pips: Optional[float] = None
    macd_cross_bars: Optional[float] = None
    macd_cross_minutes: Optional[float] = None

    macd_curr = _macd(closes)
    macd_prev = _macd(closes[:-1])
    if macd_curr is not None and macd_prev is not None:
        macd_pips = macd_curr / PIP
        macd_prev_pips = macd_prev / PIP
        macd_slope_pips = macd_pips - macd_prev_pips
        if macd_slope_pips != 0.0 and macd_pips * macd_slope_pips < 0:
            macd_cross_bars = abs(macd_pips) / abs(macd_slope_pips)
            macd_cross_minutes = macd_cross_bars * max(timeframe_minutes, 1e-6)

    return MACrossProjection(
        fast_ma=fast,
        slow_ma=slow,
        gap_pips=gap_pips,
        prev_gap_pips=prev_gap_pips,
        gap_slope_pips=gap_slope,
        fast_slope_pips=fast_slope,
        slow_slope_pips=slow_slope,
        price_to_fast_pips=price_to_fast,
        price_to_slow_pips=price_to_slow,
        projected_cross_bars=projected_cross_bars,
        projected_cross_minutes=projected_cross_minutes,
        macd_pips=macd_pips,
        macd_slope_pips=macd_slope_pips,
        macd_cross_bars=macd_cross_bars,
        macd_cross_minutes=macd_cross_minutes,
    )


def _simple_ma(values: Sequence[float], window: int) -> Optional[float]:
    if window <= 0 or len(values) < window:
        return None
    tail = values[-window:]
    return sum(tail) / float(window)


def _macd(values: Sequence[float]) -> Optional[float]:
    if len(values) < 35:
        return None
    ema_fast = _ema(values, 12)
    ema_slow = _ema(values, 26)
    if ema_fast is None or ema_slow is None:
        return None
    return ema_fast - ema_slow


def _ema(values: Sequence[float], span: int) -> Optional[float]:
    if span <= 0 or len(values) < span:
        return None
    alpha = 2.0 / (span + 1.0)
    ema_val = sum(values[:span]) / float(span)
    for v in values[span:]:
        ema_val = alpha * v + (1 - alpha) * ema_val
    return ema_val

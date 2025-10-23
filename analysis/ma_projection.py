"""
analysis.ma_projection
~~~~~~~~~~~~~~~~~~~~~~
Utility helpers to estimate moving-average and MACD cross dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


# ---- Additional projections (RSI / BBW / ADX / ATR / Donchian) ----


@dataclass
class RSIProjection:
    rsi: float
    slope_per_bar: float
    eta_upper_bars: Optional[float]
    eta_lower_bars: Optional[float]
    eta_upper_minutes: Optional[float]
    eta_lower_minutes: Optional[float]


def compute_rsi_projection(
    candles: Iterable[Dict[str, float]],
    timeframe_minutes: float = 1.0,
    period: int = 14,
    upper: float = 70.0,
    lower: float = 30.0,
) -> Optional[RSIProjection]:
    closes = [float(c.get("close", 0.0)) for c in candles if c and "close" in c]
    if len(closes) < period + 2:
        return None
    # compute RSI last 2 values
    def _rsi_seq(vals: List[float]) -> float:
        import math
        diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        gains = [max(0.0, d) for d in diffs]
        losses = [max(0.0, -d) for d in diffs]
        alpha = 1.0 / period
        def _ewm(seq: List[float]) -> float:
            s = seq[0]
            for v in seq[1:]:
                s = alpha * v + (1 - alpha) * s
            return s
        avg_gain = _ewm(gains)
        avg_loss = _ewm(losses)
        rs = avg_gain / (avg_loss if avg_loss != 0 else float("nan"))
        if rs != rs:  # NaN check
            return 50.0
        return 100 - (100 / (1 + rs))

    rsi_prev = _rsi_seq(closes[-(period + 1) : -1])
    rsi_curr = _rsi_seq(closes[-(period + 1) :])
    slope = rsi_curr - rsi_prev

    eta_upper_bars = None
    eta_lower_bars = None
    if slope > 0 and rsi_curr < upper:
        eta_upper_bars = (upper - rsi_curr) / slope
    if slope < 0 and rsi_curr > lower:
        eta_lower_bars = (rsi_curr - lower) / (-slope)
    def _mins(x: Optional[float]) -> Optional[float]:
        return x * timeframe_minutes if x is not None else None
    return RSIProjection(
        rsi=rsi_curr,
        slope_per_bar=slope,
        eta_upper_bars=eta_upper_bars,
        eta_lower_bars=eta_lower_bars,
        eta_upper_minutes=_mins(eta_upper_bars),
        eta_lower_minutes=_mins(eta_lower_bars),
    )


@dataclass
class BBWProjection:
    bbw: float
    slope_per_bar: float
    squeeze_eta_bars: Optional[float]
    squeeze_eta_minutes: Optional[float]


def compute_bbw_projection(
    candles: Iterable[Dict[str, float]],
    timeframe_minutes: float = 1.0,
    period: int = 20,
    std_mult: float = 2.0,
    squeeze_threshold: float = 0.14,
) -> Optional[BBWProjection]:
    import math
    closes = [float(c.get("close", 0.0)) for c in candles if c and "close" in c]
    if len(closes) < period + 2:
        return None
    def _boll(vals: List[float]) -> Tuple[float, float, float]:
        import statistics as st
        seg = vals[-period:]
        mid = sum(seg) / period
        # population std to align with calc_core default ddof=0
        if period <= 1:
            sd = 0.0
        else:
            mean = mid
            var = sum((v - mean) ** 2 for v in seg) / period
            sd = var ** 0.5
        upper = mid + std_mult * sd
        lower = mid - std_mult * sd
        bbw = (upper - lower) / mid if mid != 0 else 0.0
        return upper, mid, lower, bbw  # type: ignore
    _, mid_prev, _, bbw_prev = _boll(closes[:-1])
    _, mid_curr, _, bbw_curr = _boll(closes)
    slope = bbw_curr - bbw_prev
    eta_bars = None
    if slope < 0 and bbw_curr > squeeze_threshold:
        eta_bars = (bbw_curr - squeeze_threshold) / (-slope) if -slope > 0 else None
    return BBWProjection(
        bbw=bbw_curr,
        slope_per_bar=slope,
        squeeze_eta_bars=eta_bars,
        squeeze_eta_minutes=(eta_bars * timeframe_minutes) if eta_bars is not None else None,
    )


@dataclass
class ADXProjection:
    adx: float
    slope_per_bar: float
    eta_to_trend_bars: Optional[float]
    eta_to_trend_minutes: Optional[float]


def compute_adx_projection(
    candles: Iterable[Dict[str, float]],
    timeframe_minutes: float = 1.0,
    period: int = 14,
    trend_threshold: float = 20.0,
) -> Optional[ADXProjection]:
    highs = [float(c.get("high", 0.0)) for c in candles if c and "high" in c]
    lows = [float(c.get("low", 0.0)) for c in candles if c and "low" in c]
    closes = [float(c.get("close", 0.0)) for c in candles if c and "close" in c]
    if not (len(highs) >= period + 2 and len(lows) >= period + 2 and len(closes) >= period + 2):
        return None

    def _atr(h, l, c):
        prev_c = c[:-1] + [c[-1]]
        tr = [max(h[i]-l[i], abs(h[i]-prev_c[i]), abs(l[i]-prev_c[i])) for i in range(1, len(c))]
        alpha = 1.0 / period
        s = tr[0]
        for v in tr[1:]:
            s = alpha * v + (1 - alpha) * s
        return s
    def _adx_last(vals_h, vals_l, vals_c) -> float:
        import numpy as np  # local import is ok
        # Reuse simplified version consistent with calc_core
        # Compute +/-DM
        up = [max(0.0, vals_h[i] - vals_h[i-1]) for i in range(1, len(vals_h))]
        down = [max(0.0, vals_l[i-1] - vals_l[i]) for i in range(1, len(vals_l))]
        alpha = 1.0 / period
        def _ewm(seq: List[float]) -> float:
            s = seq[0]
            for v in seq[1:]:
                s = alpha * v + (1 - alpha) * s
            return s
        tr = _atr(vals_h, vals_l, vals_c)
        plus_di = 100.0 * _ewm(up) / (tr if tr != 0 else 1e-9)
        minus_di = 100.0 * _ewm(down) / (tr if tr != 0 else 1e-9)
        dx = abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-9) * 100.0
        # Smooth dx itself
        s = dx
        return s

    adx_prev = _adx_last(highs[:-1], lows[:-1], closes[:-1])
    adx_curr = _adx_last(highs, lows, closes)
    slope = adx_curr - adx_prev
    eta_bars = None
    if slope > 0 and adx_curr < trend_threshold:
        eta_bars = (trend_threshold - adx_curr) / slope
    return ADXProjection(
        adx=adx_curr,
        slope_per_bar=slope,
        eta_to_trend_bars=eta_bars,
        eta_to_trend_minutes=(eta_bars * timeframe_minutes) if eta_bars is not None else None,
    )


@dataclass
class ATRProjection:
    atr: float
    atr_pips: float
    slope_per_bar_pips: float


def compute_atr_projection(
    candles: Iterable[Dict[str, float]],
    timeframe_minutes: float = 1.0,
    period: int = 14,
) -> Optional[ATRProjection]:
    highs = [float(c.get("high", 0.0)) for c in candles if c and "high" in c]
    lows = [float(c.get("low", 0.0)) for c in candles if c and "low" in c]
    closes = [float(c.get("close", 0.0)) for c in candles if c and "close" in c]
    if not (len(highs) >= period + 2 and len(lows) >= period + 2 and len(closes) >= period + 2):
        return None
    def _true_range(i: int) -> float:
        prev_close = closes[i - 1]
        return max(highs[i] - lows[i], abs(highs[i] - prev_close), abs(lows[i] - prev_close))
    alpha = 1.0 / period
    s_prev = None
    for i in range(1, len(closes) - 1):
        tr = _true_range(i)
        if s_prev is None:
            s_prev = tr
        else:
            s_prev = alpha * tr + (1 - alpha) * s_prev
    tr_last = _true_range(len(closes) - 1)
    atr_prev = s_prev if s_prev is not None else tr_last
    atr_curr = alpha * tr_last + (1 - alpha) * atr_prev
    slope = (atr_curr - atr_prev) / PIP
    return ATRProjection(atr=atr_curr, atr_pips=atr_curr / PIP, slope_per_bar_pips=slope)


@dataclass
class DonchianProjection:
    high: float
    low: float
    dist_high_pips: float
    dist_low_pips: float
    nearest_pips: float


def compute_donchian_projection(
    candles: Iterable[Dict[str, float]],
    lookback: int = 55,
) -> Optional[DonchianProjection]:
    arr = [c for c in candles]
    if len(arr) < lookback + 1:
        return None
    import math
    highs = [float(c.get("high", 0.0)) for c in arr[-(lookback + 1) : -1]]
    lows = [float(c.get("low", 0.0)) for c in arr[-(lookback + 1) : -1]]
    if not highs or not lows:
        return None
    high_val = max(highs)
    low_val = min(lows)
    close = float(arr[-1].get("close", 0.0))
    dist_high = (high_val - close) / PIP
    dist_low = (close - low_val) / PIP
    nearest = min(max(0.0, dist_high), max(0.0, dist_low))
    return DonchianProjection(
        high=high_val,
        low=low_val,
        dist_high_pips=max(0.0, dist_high),
        dist_low_pips=max(0.0, dist_low),
        nearest_pips=nearest,
    )

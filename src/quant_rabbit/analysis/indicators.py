"""Pure-Python technical indicators.

Computes the classical indicator stack the trader needs to read the chart at
each timeframe — moving averages, momentum, volatility, trend strength, range
context, divergence, and Ichimoku cloud position. No pandas/numpy dependency,
so it runs on a stock interpreter without extra installs.

All indicators degrade gracefully on short series: when there is not enough
data they return `None`/`0.0` with a `valid=False` marker rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from quant_rabbit.analysis.candles import Candle, closes, highs, lows


@dataclass(frozen=True)
class IndicatorSet:
    """Indicator readings for a single pair at a single timeframe."""

    pair: str
    granularity: str
    candles_count: int
    pip_size: float
    close: float
    valid: bool

    # Moving averages
    sma_20: float | None
    ema_12: float | None
    ema_20: float | None
    ema_50: float | None
    ema_slope_5: float | None
    ema_slope_20: float | None

    # Momentum
    rsi_14: float | None
    stoch_rsi: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    roc_5: float | None
    roc_10: float | None
    cci_14: float | None

    # Volatility
    atr_14: float | None
    atr_pips: float | None
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    bb_width: float | None
    bb_span_pips: float | None
    keltner_width: float | None
    donchian_high: float | None
    donchian_low: float | None
    donchian_width_pips: float | None

    # Trend strength
    adx_14: float | None
    plus_di_14: float | None
    minus_di_14: float | None

    # Volume / VWAP
    vwap: float | None
    vwap_gap_pips: float | None

    # Range context
    swing_high: float | None
    swing_low: float | None
    swing_distance_high_pips: float | None
    swing_distance_low_pips: float | None

    # Ichimoku
    ichimoku_tenkan: float | None
    ichimoku_kijun: float | None
    ichimoku_span_a: float | None
    ichimoku_span_b: float | None
    ichimoku_cloud_pos: int  # 1=above cloud, -1=below, 0=inside

    def to_dict(self) -> dict[str, object]:
        return {
            "pair": self.pair,
            "granularity": self.granularity,
            "candles_count": self.candles_count,
            "pip_size": self.pip_size,
            "close": self.close,
            "valid": self.valid,
            "sma_20": self.sma_20,
            "ema_12": self.ema_12,
            "ema_20": self.ema_20,
            "ema_50": self.ema_50,
            "ema_slope_5": self.ema_slope_5,
            "ema_slope_20": self.ema_slope_20,
            "rsi_14": self.rsi_14,
            "stoch_rsi": self.stoch_rsi,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "macd_hist": self.macd_hist,
            "roc_5": self.roc_5,
            "roc_10": self.roc_10,
            "cci_14": self.cci_14,
            "atr_14": self.atr_14,
            "atr_pips": self.atr_pips,
            "bb_upper": self.bb_upper,
            "bb_middle": self.bb_middle,
            "bb_lower": self.bb_lower,
            "bb_width": self.bb_width,
            "bb_span_pips": self.bb_span_pips,
            "keltner_width": self.keltner_width,
            "donchian_high": self.donchian_high,
            "donchian_low": self.donchian_low,
            "donchian_width_pips": self.donchian_width_pips,
            "adx_14": self.adx_14,
            "plus_di_14": self.plus_di_14,
            "minus_di_14": self.minus_di_14,
            "vwap": self.vwap,
            "vwap_gap_pips": self.vwap_gap_pips,
            "swing_high": self.swing_high,
            "swing_low": self.swing_low,
            "swing_distance_high_pips": self.swing_distance_high_pips,
            "swing_distance_low_pips": self.swing_distance_low_pips,
            "ichimoku_tenkan": self.ichimoku_tenkan,
            "ichimoku_kijun": self.ichimoku_kijun,
            "ichimoku_span_a": self.ichimoku_span_a,
            "ichimoku_span_b": self.ichimoku_span_b,
            "ichimoku_cloud_pos": self.ichimoku_cloud_pos,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_indicators(
    pair: str, granularity: str, candles: Sequence[Candle]
) -> IndicatorSet:
    """Compute the full indicator set for a pair-timeframe candle series."""

    n = len(candles)
    pip = _pip_size_for(pair)
    last_close = float(candles[-1].close) if n else 0.0
    valid = n >= 30  # need enough bars for the heaviest indicator

    cs = closes(candles)
    hs = highs(candles)
    ls = lows(candles)

    sma_20 = _sma(cs, 20)
    ema_12 = _ema(cs, 12)
    ema_20 = _ema(cs, 20)
    ema_26 = _ema(cs, 26)
    ema_50 = _ema(cs, 50)
    ema_slope_5 = _ema_slope(cs, span=12, lookback=5, pip=pip)
    ema_slope_20 = _ema_slope(cs, span=24, lookback=20, pip=pip)

    rsi_14 = _rsi(cs, 14)
    stoch_rsi = _stoch_rsi(cs, period=14, smooth=3)
    macd_line, macd_signal, macd_hist = _macd(cs, fast=12, slow=26, signal=9)
    roc_5 = _roc(cs, 5)
    roc_10 = _roc(cs, 10)
    cci_14 = _cci(hs, ls, cs, 14)

    atr_14 = _atr(hs, ls, cs, 14)
    atr_pips = (atr_14 / pip) if atr_14 is not None else None
    bb_upper, bb_mid, bb_lower = _bollinger(cs, 20, 2.0)
    bb_width = (bb_upper - bb_lower) / bb_mid if (bb_upper is not None and bb_mid not in (None, 0.0)) else None
    bb_span_pips = ((bb_upper - bb_lower) / pip) if (bb_upper is not None and bb_lower is not None) else None

    keltner_width = _keltner_width(hs, ls, cs, period=20, mult=1.5)
    donchian_high, donchian_low = _donchian(hs, ls, 20)
    donchian_width_pips = ((donchian_high - donchian_low) / pip) if (donchian_high is not None and donchian_low is not None) else None

    adx_14, plus_di_14, minus_di_14 = _adx(hs, ls, cs, 14)

    vwap = _vwap(candles)
    vwap_gap_pips = ((last_close - vwap) / pip) if vwap is not None else None

    swing_high, swing_low = _swing_extrema(hs, ls, lookback=50)
    swing_distance_high_pips = ((swing_high - last_close) / pip) if swing_high is not None else None
    swing_distance_low_pips = ((last_close - swing_low) / pip) if swing_low is not None else None

    tenkan, kijun, span_a, span_b, cloud_pos = _ichimoku(hs, ls, cs, last_close)

    return IndicatorSet(
        pair=pair,
        granularity=granularity,
        candles_count=n,
        pip_size=pip,
        close=last_close,
        valid=valid,
        sma_20=sma_20,
        ema_12=ema_12,
        ema_20=ema_20,
        ema_50=ema_50,
        ema_slope_5=ema_slope_5,
        ema_slope_20=ema_slope_20,
        rsi_14=rsi_14,
        stoch_rsi=stoch_rsi,
        macd=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        roc_5=roc_5,
        roc_10=roc_10,
        cci_14=cci_14,
        atr_14=atr_14,
        atr_pips=atr_pips,
        bb_upper=bb_upper,
        bb_middle=bb_mid,
        bb_lower=bb_lower,
        bb_width=bb_width,
        bb_span_pips=bb_span_pips,
        keltner_width=keltner_width,
        donchian_high=donchian_high,
        donchian_low=donchian_low,
        donchian_width_pips=donchian_width_pips,
        adx_14=adx_14,
        plus_di_14=plus_di_14,
        minus_di_14=minus_di_14,
        vwap=vwap,
        vwap_gap_pips=vwap_gap_pips,
        swing_high=swing_high,
        swing_low=swing_low,
        swing_distance_high_pips=swing_distance_high_pips,
        swing_distance_low_pips=swing_distance_low_pips,
        ichimoku_tenkan=tenkan,
        ichimoku_kijun=kijun,
        ichimoku_span_a=span_a,
        ichimoku_span_b=span_b,
        ichimoku_cloud_pos=cloud_pos,
    )


# ---------------------------------------------------------------------------
# Indicator primitives
# ---------------------------------------------------------------------------


def _pip_size_for(pair: str) -> float:
    return 0.01 if pair.upper().endswith("_JPY") else 0.0001


def _sma(values: Sequence[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None
    return sum(values[-period:]) / period


def _ema(values: Sequence[float], span: int) -> float | None:
    if len(values) < span or span <= 0:
        return None
    k = 2.0 / (span + 1.0)
    seed = sum(values[:span]) / span
    e = seed
    for v in values[span:]:
        e = v * k + e * (1.0 - k)
    return e


def _ema_series(values: Sequence[float], span: int) -> list[float] | None:
    if len(values) < span or span <= 0:
        return None
    k = 2.0 / (span + 1.0)
    seed = sum(values[:span]) / span
    out: list[float] = [None] * (span - 1)  # type: ignore[list-item]
    out.append(seed)
    e = seed
    for v in values[span:]:
        e = v * k + e * (1.0 - k)
        out.append(e)
    return out  # type: ignore[return-value]


def _ema_slope(values: Sequence[float], *, span: int, lookback: int, pip: float) -> float | None:
    series = _ema_series(values, span)
    if series is None or len(series) <= lookback:
        return None
    end = series[-1]
    start = series[-1 - lookback]
    if end is None or start is None:
        return None
    return (end - start) / pip


def _rsi(values: Sequence[float], period: int = 14) -> float | None:
    if len(values) <= period:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _stoch_rsi(values: Sequence[float], *, period: int = 14, smooth: int = 3) -> float | None:
    if len(values) < period * 2:
        return None
    rsi_series: list[float] = []
    for i in range(period, len(values) + 1):
        sub = values[i - period : i + period] if False else values[: i]
        if len(sub) <= period:
            continue
        r = _rsi(sub, period)
        if r is not None:
            rsi_series.append(r)
    if len(rsi_series) < period:
        return None
    window = rsi_series[-period:]
    lo = min(window)
    hi = max(window)
    if hi == lo:
        return 0.0
    return (rsi_series[-1] - lo) / (hi - lo)


def _macd(values: Sequence[float], *, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float | None, float | None, float | None]:
    fast_series = _ema_series(values, fast)
    slow_series = _ema_series(values, slow)
    if fast_series is None or slow_series is None:
        return None, None, None
    macd_line: list[float | None] = []
    for f, s in zip(fast_series, slow_series):
        macd_line.append((f - s) if (f is not None and s is not None) else None)
    macd_clean = [m for m in macd_line if m is not None]
    if len(macd_clean) < signal:
        return macd_clean[-1] if macd_clean else None, None, None
    signal_value = _ema(macd_clean, signal)
    macd_now = macd_clean[-1]
    if signal_value is None:
        return macd_now, None, None
    return macd_now, signal_value, macd_now - signal_value


def _roc(values: Sequence[float], period: int) -> float | None:
    if len(values) <= period or period <= 0:
        return None
    past = values[-period - 1]
    if past == 0:
        return None
    return (values[-1] - past) / past * 100.0


def _cci(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], period: int = 14) -> float | None:
    if len(closes_) < period:
        return None
    tps = [(highs_[i] + lows_[i] + closes_[i]) / 3.0 for i in range(len(closes_))]
    window = tps[-period:]
    sma_tp = sum(window) / period
    mean_dev = sum(abs(x - sma_tp) for x in window) / period
    if mean_dev == 0:
        return 0.0
    return (tps[-1] - sma_tp) / (0.015 * mean_dev)


def _atr(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], period: int = 14) -> float | None:
    n = len(closes_)
    if n <= period:
        return None
    trs: list[float] = []
    for i in range(1, n):
        h = highs_[i]
        low_value = lows_[i]
        prev_close = closes_[i - 1]
        tr = max(h - low_value, abs(h - prev_close), abs(low_value - prev_close))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _bollinger(values: Sequence[float], period: int = 20, std_mult: float = 2.0) -> tuple[float | None, float | None, float | None]:
    if len(values) < period:
        return None, None, None
    window = values[-period:]
    mean = sum(window) / period
    var = sum((v - mean) ** 2 for v in window) / period
    sd = var ** 0.5
    return mean + std_mult * sd, mean, mean - std_mult * sd


def _keltner_width(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], *, period: int = 20, mult: float = 1.5) -> float | None:
    ema = _ema(closes_, period)
    atr = _atr(highs_, lows_, closes_, period)
    if ema is None or atr is None or ema == 0:
        return None
    upper = ema + mult * atr
    lower = ema - mult * atr
    return (upper - lower) / ema


def _donchian(highs_: Sequence[float], lows_: Sequence[float], period: int = 20) -> tuple[float | None, float | None]:
    if len(highs_) < period or len(lows_) < period:
        return None, None
    return max(highs_[-period:]), min(lows_[-period:])


def _adx(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], period: int = 14) -> tuple[float | None, float | None, float | None]:
    n = len(closes_)
    if n <= period * 2:
        return None, None, None
    tr_list: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, n):
        up_move = highs_[i] - highs_[i - 1]
        down_move = lows_[i - 1] - lows_[i]
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0.0)
        h = highs_[i]
        low_value = lows_[i]
        prev_close = closes_[i - 1]
        tr_list.append(max(h - low_value, abs(h - prev_close), abs(low_value - prev_close)))
    if len(tr_list) < period:
        return None, None, None
    tr_smooth = sum(tr_list[:period])
    plus_smooth = sum(plus_dm[:period])
    minus_smooth = sum(minus_dm[:period])
    dx_list: list[float] = []
    for i in range(period, len(tr_list)):
        tr_smooth = tr_smooth - (tr_smooth / period) + tr_list[i]
        plus_smooth = plus_smooth - (plus_smooth / period) + plus_dm[i]
        minus_smooth = minus_smooth - (minus_smooth / period) + minus_dm[i]
        plus_di = (plus_smooth / tr_smooth) * 100.0 if tr_smooth else 0.0
        minus_di = (minus_smooth / tr_smooth) * 100.0 if tr_smooth else 0.0
        di_sum = plus_di + minus_di
        dx = (abs(plus_di - minus_di) / di_sum * 100.0) if di_sum else 0.0
        dx_list.append(dx)
    if len(dx_list) < period:
        return None, None, None
    adx = sum(dx_list[:period]) / period
    for dx in dx_list[period:]:
        adx = (adx * (period - 1) + dx) / period
    plus_di_now = (plus_smooth / tr_smooth) * 100.0 if tr_smooth else 0.0
    minus_di_now = (minus_smooth / tr_smooth) * 100.0 if tr_smooth else 0.0
    return adx, plus_di_now, minus_di_now


def _vwap(candles: Sequence[Candle]) -> float | None:
    if not candles:
        return None
    total_pv = 0.0
    total_v = 0
    for c in candles:
        typical = (c.high + c.low + c.close) / 3.0
        v = max(int(c.volume), 1)
        total_pv += typical * v
        total_v += v
    if total_v == 0:
        return None
    vwap = total_pv / total_v
    return vwap if isfinite(vwap) else None


def _swing_extrema(highs_: Sequence[float], lows_: Sequence[float], lookback: int = 50) -> tuple[float | None, float | None]:
    if not highs_ or not lows_:
        return None, None
    window_highs = highs_[-lookback:]
    window_lows = lows_[-lookback:]
    return max(window_highs), min(window_lows)


def _ichimoku(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], current_price: float) -> tuple[float | None, float | None, float | None, float | None, int]:
    if len(highs_) < 52 or len(lows_) < 52:
        return None, None, None, None, 0
    tenkan = (max(highs_[-9:]) + min(lows_[-9:])) / 2.0
    kijun = (max(highs_[-26:]) + min(lows_[-26:])) / 2.0
    span_a = (tenkan + kijun) / 2.0
    span_b = (max(highs_[-52:]) + min(lows_[-52:])) / 2.0
    cloud_high = max(span_a, span_b)
    cloud_low = min(span_a, span_b)
    if current_price > cloud_high:
        pos = 1
    elif current_price < cloud_low:
        pos = -1
    else:
        pos = 0
    return tenkan, kijun, span_a, span_b, pos

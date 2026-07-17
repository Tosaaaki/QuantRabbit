"""Pure-Python technical indicators.

Computes the full indicator stack the trader needs to read the chart at each
timeframe — moving averages, momentum, volatility, trend strength, range
context, divergence, Ichimoku cloud, and a deep extended layer (Williams%R,
MFI, Supertrend, Parabolic SAR, Aroon, Vortex, Hull/KAMA/ALMA, linear
regression channel, Choppiness, BB squeeze, ATR/BB/ADX percentiles, z-score,
realized volatility, anchored VWAP with ±1σ/±2σ bands, Hurst exponent, mean
reversion half-life, and a quantile-based regime classifier).

No pandas dependency — pure Python with optional numpy for the statistical
primitives (numpy is in the standard system Python on macOS this project
runs against; the heavy stats fall back to None if numpy is absent).

All indicators degrade gracefully on short series: when there is not enough
data they return `None`/`0.0` with a `valid=False` marker rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt
from statistics import stdev
from typing import Sequence

from quant_rabbit.analysis.candles import Candle, closes, highs, lows

try:  # numpy is used by Phase B stats only; fall back to None when absent.
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent
    _np = None  # type: ignore


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

    # ----- Phase A: extended momentum / trend / volatility (defaults=None) ---
    williams_r_14: float | None = None
    mfi_14: float | None = None
    aroon_up_14: float | None = None
    aroon_down_14: float | None = None
    aroon_osc_14: float | None = None
    vortex_plus_14: float | None = None
    vortex_minus_14: float | None = None
    supertrend_value: float | None = None
    supertrend_dir: int | None = None  # +1 long, -1 short
    psar_value: float | None = None
    psar_dir: int | None = None
    hull_ma_20: float | None = None
    kama_10: float | None = None
    alma_20: float | None = None
    linreg_slope_20: float | None = None  # slope per bar in pips
    linreg_r2_20: float | None = None
    linreg_channel_upper: float | None = None
    linreg_channel_lower: float | None = None
    choppiness_14: float | None = None
    bb_squeeze: int | None = None  # 1 if BB inside Keltner, 0 otherwise
    atr_percentile_100: float | None = None  # 0..1 quantile rank of current ATR vs 100-bar window
    bb_width_percentile_100: float | None = None
    adx_percentile_100: float | None = None
    z_score_20: float | None = None
    realized_vol_20: float | None = None  # annualized stdev of log returns

    # Anchored VWAP + sigma bands (anchor = first candle in supplied series)
    avwap_anchor: float | None = None
    avwap_upper_1sd: float | None = None
    avwap_lower_1sd: float | None = None
    avwap_upper_2sd: float | None = None
    avwap_lower_2sd: float | None = None
    avwap_swing_high: float | None = None  # anchored to most recent swing-high
    avwap_swing_low: float | None = None   # anchored to most recent swing-low

    # ----- Phase B: statistics ----------------------------------------------
    hurst_100: float | None = None  # >0.5 trending, <0.5 mean-reverting
    half_life_60: float | None = None  # bars to mean-revert (Ornstein-Uhlenbeck fit)
    regime_quantile: str | None = None  # QUIET / NORMAL / VOLATILE based on ATR percentile

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
            # Phase A extended
            "williams_r_14": self.williams_r_14,
            "mfi_14": self.mfi_14,
            "aroon_up_14": self.aroon_up_14,
            "aroon_down_14": self.aroon_down_14,
            "aroon_osc_14": self.aroon_osc_14,
            "vortex_plus_14": self.vortex_plus_14,
            "vortex_minus_14": self.vortex_minus_14,
            "supertrend_value": self.supertrend_value,
            "supertrend_dir": self.supertrend_dir,
            "psar_value": self.psar_value,
            "psar_dir": self.psar_dir,
            "hull_ma_20": self.hull_ma_20,
            "kama_10": self.kama_10,
            "alma_20": self.alma_20,
            "linreg_slope_20": self.linreg_slope_20,
            "linreg_r2_20": self.linreg_r2_20,
            "linreg_channel_upper": self.linreg_channel_upper,
            "linreg_channel_lower": self.linreg_channel_lower,
            "choppiness_14": self.choppiness_14,
            "bb_squeeze": self.bb_squeeze,
            "atr_percentile_100": self.atr_percentile_100,
            "bb_width_percentile_100": self.bb_width_percentile_100,
            "adx_percentile_100": self.adx_percentile_100,
            "z_score_20": self.z_score_20,
            "realized_vol_20": self.realized_vol_20,
            "avwap_anchor": self.avwap_anchor,
            "avwap_upper_1sd": self.avwap_upper_1sd,
            "avwap_lower_1sd": self.avwap_lower_1sd,
            "avwap_upper_2sd": self.avwap_upper_2sd,
            "avwap_lower_2sd": self.avwap_lower_2sd,
            "avwap_swing_high": self.avwap_swing_high,
            "avwap_swing_low": self.avwap_swing_low,
            # Phase B stats
            "hurst_100": self.hurst_100,
            "half_life_60": self.half_life_60,
            "regime_quantile": self.regime_quantile,
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

    # ---- Phase A: extended indicators ------------------------------------
    williams_r = _williams_r(hs, ls, cs, 14)
    mfi = _mfi(candles, 14)
    aroon_up, aroon_down, aroon_osc = _aroon(hs, ls, 14)
    vortex_plus, vortex_minus = _vortex(hs, ls, cs, 14)
    supertrend_v, supertrend_d = _supertrend(hs, ls, cs, period=10, multiplier=3.0)
    psar_v, psar_d = _parabolic_sar(hs, ls)
    hull_20 = _hull_ma(cs, 20)
    kama_v = _kama(cs, period=10, fast=2, slow=30)
    alma_v = _alma(cs, period=20, offset=0.85, sigma=6.0)
    lr_slope, lr_r2, lr_upper, lr_lower = _linreg_channel(cs, 20, pip)
    chop = _choppiness(hs, ls, cs, 14)
    bb_sq = _bb_squeeze_flag(bb_upper, bb_lower, ema_20, atr_14)
    atr_pct = _percentile_rank(_atr_series(hs, ls, cs, 14), atr_14, lookback=100)
    bb_pct = _percentile_rank(_bb_width_series(cs, 20, 2.0), bb_width, lookback=100)
    adx_pct = _percentile_rank(_adx_series(hs, ls, cs, 14), adx_14, lookback=100)
    z = _z_score(cs, 20)
    rv = _realized_vol(cs, 20)
    a_anchor, a_up1, a_lo1, a_up2, a_lo2 = _anchored_vwap_with_bands(candles)
    a_sh = _anchored_vwap_from_swing(candles, side="high", lookback=50)
    a_sl = _anchored_vwap_from_swing(candles, side="low", lookback=50)

    # ---- Phase B: stats --------------------------------------------------
    hurst = _hurst_exponent(cs, max_lag=20) if len(cs) >= 100 else None
    half_life = _half_life(cs, 60)
    regime_quant = _regime_quantile(atr_pct, adx_pct)

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
        williams_r_14=williams_r,
        mfi_14=mfi,
        aroon_up_14=aroon_up,
        aroon_down_14=aroon_down,
        aroon_osc_14=aroon_osc,
        vortex_plus_14=vortex_plus,
        vortex_minus_14=vortex_minus,
        supertrend_value=supertrend_v,
        supertrend_dir=supertrend_d,
        psar_value=psar_v,
        psar_dir=psar_d,
        hull_ma_20=hull_20,
        kama_10=kama_v,
        alma_20=alma_v,
        linreg_slope_20=lr_slope,
        linreg_r2_20=lr_r2,
        linreg_channel_upper=lr_upper,
        linreg_channel_lower=lr_lower,
        choppiness_14=chop,
        bb_squeeze=bb_sq,
        atr_percentile_100=atr_pct,
        bb_width_percentile_100=bb_pct,
        adx_percentile_100=adx_pct,
        z_score_20=z,
        realized_vol_20=rv,
        avwap_anchor=a_anchor,
        avwap_upper_1sd=a_up1,
        avwap_lower_1sd=a_lo1,
        avwap_upper_2sd=a_up2,
        avwap_lower_2sd=a_lo2,
        avwap_swing_high=a_sh,
        avwap_swing_low=a_sl,
        hurst_100=hurst,
        half_life_60=half_life,
        regime_quantile=regime_quant,
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


def compute_rsi_series(values: Sequence[float], *, period: int = 14, count: int = 30) -> tuple[float, ...]:
    """Return the LAST `count` RSI values as a tuple.

    Public companion to `_rsi`. Internally walks the Wilder-smoothing
    accumulator once and emits a tuple of the most recent `count`
    values. Returns () when there aren't enough bars to seed the
    average. Used by pattern_signals to detect true RSI divergence
    against price swings.
    """
    if len(values) <= period:
        return tuple()
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    series: list[float] = []
    if avg_loss == 0:
        series.append(100.0)
    else:
        rs = avg_gain / avg_loss
        series.append(100.0 - (100.0 / (1.0 + rs)))
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            series.append(100.0)
        else:
            rs = avg_gain / avg_loss
            series.append(100.0 - (100.0 / (1.0 + rs)))
    return tuple(series[-count:])


def compute_macd_hist_series(values: Sequence[float], *, fast: int = 12, slow: int = 26, signal: int = 9, count: int = 30) -> tuple[float, ...]:
    """Return the LAST `count` MACD histogram values as a tuple.

    Built from `_ema_series` for both legs, then a signal EMA across
    the MACD line. Used for divergence detection — comparing histogram
    extrema at the same swing points as the price extrema.
    """
    fast_series = _ema_series(values, fast)
    slow_series = _ema_series(values, slow)
    if fast_series is None or slow_series is None:
        return tuple()
    macd_line: list[float | None] = []
    for f, s in zip(fast_series, slow_series):
        if f is None or s is None:
            macd_line.append(None)
        else:
            macd_line.append(f - s)
    # Drop None prefix
    macd_clean = [m for m in macd_line if m is not None]
    if len(macd_clean) < signal:
        return tuple()
    signal_series = _ema_series(macd_clean, signal)
    if signal_series is None:
        return tuple()
    hist: list[float] = []
    for m, sv in zip(macd_clean, signal_series):
        if sv is None:
            continue
        hist.append(m - sv)
    return tuple(hist[-count:])


def compute_adx_series(
    highs_: Sequence[float],
    lows_: Sequence[float],
    closes_: Sequence[float],
    *,
    period: int = 14,
    count: int = 30,
) -> tuple[float, ...]:
    """Return the last ``count`` Wilder ADX values.

    ADX is directionless, so a single high reading cannot distinguish a
    strengthening trend from an exhausted one.  Publishing a bounded series
    lets causal shadow policies measure its slope while ``plus_di_14`` and
    ``minus_di_14`` continue to supply direction.
    """

    if count <= 0:
        return tuple()
    series = _adx_series(highs_, lows_, closes_, period)
    if not series:
        return tuple()
    finite = [float(value) for value in series if value is not None and isfinite(value)]
    return tuple(finite[-count:])


def compute_atr_pips_series(
    highs_: Sequence[float],
    lows_: Sequence[float],
    closes_: Sequence[float],
    *,
    pip_size: float,
    period: int = 14,
    count: int = 30,
) -> tuple[float, ...]:
    """Return the bounded ATR history in executable pip units."""

    if count <= 0 or not isfinite(pip_size) or pip_size <= 0.0:
        return tuple()
    series = _atr_series(highs_, lows_, closes_, period)
    if not series:
        return tuple()
    finite = [
        float(value) / pip_size
        for value in series
        if value is not None and isfinite(value)
    ]
    return tuple(finite[-count:])


def compute_ema_spread_pips_series(
    values: Sequence[float],
    *,
    pip_size: float,
    fast: int = 12,
    slow: int = 50,
    count: int = 30,
) -> tuple[float, ...]:
    """Return fast-minus-slow EMA history in pips.

    A zero crossing records when a golden/death cross actually occurred;
    the current EMA ordering alone cannot distinguish a fresh cross from a
    stale, already extended trend.
    """

    if count <= 0 or not isfinite(pip_size) or pip_size <= 0.0 or fast >= slow:
        return tuple()
    fast_series = _ema_series(values, fast)
    slow_series = _ema_series(values, slow)
    if fast_series is None or slow_series is None:
        return tuple()
    spread = [
        (float(fast_value) - float(slow_value)) / pip_size
        for fast_value, slow_value in zip(fast_series, slow_series)
        if fast_value is not None
        and slow_value is not None
        and isfinite(fast_value)
        and isfinite(slow_value)
    ]
    return tuple(spread[-count:])


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


# ===========================================================================
# Phase A: extended technical primitives
# ===========================================================================


def _williams_r(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], period: int = 14) -> float | None:
    if len(closes_) < period:
        return None
    hh = max(highs_[-period:])
    ll = min(lows_[-period:])
    if hh == ll:
        return -50.0
    return -100.0 * (hh - closes_[-1]) / (hh - ll)


def _mfi(candles: Sequence[Candle], period: int = 14) -> float | None:
    """Money Flow Index using tick volume (FX has no real volume; tick-volume is the standard proxy)."""
    n = len(candles)
    if n <= period:
        return None
    pos_flow = 0.0
    neg_flow = 0.0
    for i in range(n - period, n):
        if i == 0:
            continue
        tp = (candles[i].high + candles[i].low + candles[i].close) / 3.0
        prev_tp = (candles[i - 1].high + candles[i - 1].low + candles[i - 1].close) / 3.0
        flow = tp * max(int(candles[i].volume), 1)
        if tp > prev_tp:
            pos_flow += flow
        elif tp < prev_tp:
            neg_flow += flow
    if neg_flow == 0:
        return 100.0 if pos_flow > 0 else 50.0
    mr = pos_flow / neg_flow
    return 100.0 - (100.0 / (1.0 + mr))


def _aroon(highs_: Sequence[float], lows_: Sequence[float], period: int = 14) -> tuple[float | None, float | None, float | None]:
    if len(highs_) < period + 1 or len(lows_) < period + 1:
        return None, None, None
    window_h = highs_[-(period + 1):]
    window_l = lows_[-(period + 1):]
    # bars since highest/lowest in the window
    hi_idx = max(range(len(window_h)), key=lambda i: window_h[i])
    lo_idx = max(range(len(window_l)), key=lambda i: -window_l[i])  # argmin
    bars_since_hi = period - hi_idx
    bars_since_lo = period - lo_idx
    aroon_up = 100.0 * (period - bars_since_hi) / period
    aroon_down = 100.0 * (period - bars_since_lo) / period
    return aroon_up, aroon_down, aroon_up - aroon_down


def _vortex(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], period: int = 14) -> tuple[float | None, float | None]:
    n = len(closes_)
    if n <= period:
        return None, None
    vm_plus = []
    vm_minus = []
    trs = []
    for i in range(1, n):
        vm_plus.append(abs(highs_[i] - lows_[i - 1]))
        vm_minus.append(abs(lows_[i] - highs_[i - 1]))
        trs.append(max(
            highs_[i] - lows_[i],
            abs(highs_[i] - closes_[i - 1]),
            abs(lows_[i] - closes_[i - 1]),
        ))
    if len(trs) < period:
        return None, None
    sum_tr = sum(trs[-period:])
    if sum_tr == 0:
        return None, None
    vi_plus = sum(vm_plus[-period:]) / sum_tr
    vi_minus = sum(vm_minus[-period:]) / sum_tr
    return vi_plus, vi_minus


def _supertrend(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], *, period: int = 10, multiplier: float = 3.0) -> tuple[float | None, int | None]:
    """Supertrend (Olivier Seban). Returns (band_value, direction +1/-1)."""
    n = len(closes_)
    if n <= period + 1:
        return None, None
    atr_series = _atr_series(highs_, lows_, closes_, period)
    if atr_series is None:
        return None, None
    final_upper: list[float | None] = [None] * n
    final_lower: list[float | None] = [None] * n
    direction: list[int] = [1] * n
    for i in range(period, n):
        atr_i = atr_series[i]
        if atr_i is None:
            continue
        hl2 = (highs_[i] + lows_[i]) / 2.0
        basic_u = hl2 + multiplier * atr_i
        basic_l = hl2 - multiplier * atr_i
        prev_u = final_upper[i - 1] if final_upper[i - 1] is not None else basic_u
        prev_l = final_lower[i - 1] if final_lower[i - 1] is not None else basic_l
        final_upper[i] = basic_u if (basic_u < prev_u or closes_[i - 1] > prev_u) else prev_u
        final_lower[i] = basic_l if (basic_l > prev_l or closes_[i - 1] < prev_l) else prev_l
        prev_dir = direction[i - 1]
        if prev_dir == 1 and closes_[i] < (final_lower[i] or basic_l):
            direction[i] = -1
        elif prev_dir == -1 and closes_[i] > (final_upper[i] or basic_u):
            direction[i] = 1
        else:
            direction[i] = prev_dir
    last_dir = direction[-1]
    band = final_lower[-1] if last_dir == 1 else final_upper[-1]
    return band, last_dir


def _parabolic_sar(highs_: Sequence[float], lows_: Sequence[float], *, af_step: float = 0.02, af_max: float = 0.2) -> tuple[float | None, int | None]:
    n = len(highs_)
    if n < 5:
        return None, None
    sar = lows_[0]
    ep = highs_[0]
    af = af_step
    direction = 1  # +1 long, -1 short
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if direction == 1:
            sar = min(sar, lows_[i - 1], lows_[i - 2] if i >= 2 else lows_[i - 1])
            if lows_[i] < sar:
                direction = -1
                sar = ep
                ep = lows_[i]
                af = af_step
            else:
                if highs_[i] > ep:
                    ep = highs_[i]
                    af = min(af + af_step, af_max)
        else:
            sar = max(sar, highs_[i - 1], highs_[i - 2] if i >= 2 else highs_[i - 1])
            if highs_[i] > sar:
                direction = 1
                sar = ep
                ep = highs_[i]
                af = af_step
            else:
                if lows_[i] < ep:
                    ep = lows_[i]
                    af = min(af + af_step, af_max)
    return sar, direction


def _hull_ma(values: Sequence[float], period: int) -> float | None:
    if len(values) < period:
        return None
    half = max(1, period // 2)
    sqrt_p = max(1, int(round(sqrt(period))))
    wma_half = _wma(values, half)
    wma_full = _wma(values, period)
    if wma_half is None or wma_full is None:
        return None
    diff_series: list[float] = []
    for i in range(period, len(values) + 1):
        wh = _wma(values[:i], half)
        wf = _wma(values[:i], period)
        if wh is None or wf is None:
            continue
        diff_series.append(2.0 * wh - wf)
    if len(diff_series) < sqrt_p:
        return None
    return _wma(diff_series, sqrt_p)


def _wma(values: Sequence[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None
    weights = list(range(1, period + 1))
    window = values[-period:]
    return sum(v * w for v, w in zip(window, weights)) / sum(weights)


def _kama(values: Sequence[float], *, period: int = 10, fast: int = 2, slow: int = 30) -> float | None:
    n = len(values)
    if n <= period:
        return None
    kama = values[period]
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    for i in range(period + 1, n):
        change = abs(values[i] - values[i - period])
        volatility = sum(abs(values[j] - values[j - 1]) for j in range(i - period + 1, i + 1))
        er = change / volatility if volatility else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = kama + sc * (values[i] - kama)
    return kama


def _alma(values: Sequence[float], *, period: int = 20, offset: float = 0.85, sigma: float = 6.0) -> float | None:
    if len(values) < period:
        return None
    m = offset * (period - 1)
    s = period / sigma
    weights: list[float] = []
    for i in range(period):
        weights.append(pow(2.71828182846, -((i - m) ** 2) / (2 * s * s)))
    norm = sum(weights)
    window = values[-period:]
    return sum(v * w for v, w in zip(window, weights)) / norm if norm else None


def _linreg_channel(values: Sequence[float], period: int, pip: float) -> tuple[float | None, float | None, float | None, float | None]:
    """Linear regression slope (per-bar in pips), R², upper/lower 2σ channel."""
    if len(values) < period:
        return None, None, None, None
    window = values[-period:]
    xs = list(range(period))
    n = period
    mean_x = sum(xs) / n
    mean_y = sum(window) / n
    cov = sum((xs[i] - mean_x) * (window[i] - mean_y) for i in range(n))
    var_x = sum((xs[i] - mean_x) ** 2 for i in range(n))
    var_y = sum((window[i] - mean_y) ** 2 for i in range(n))
    if var_x == 0:
        return None, None, None, None
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    fitted = [intercept + slope * x for x in xs]
    residuals = [window[i] - fitted[i] for i in range(n)]
    rss = sum(r * r for r in residuals)
    r2 = 1.0 - (rss / var_y) if var_y else 0.0
    sd = (rss / max(n - 2, 1)) ** 0.5
    last_fit = fitted[-1]
    upper = last_fit + 2.0 * sd
    lower = last_fit - 2.0 * sd
    return slope / pip, r2, upper, lower


def _choppiness(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], period: int = 14) -> float | None:
    """Choppiness Index — 0..100, higher means choppy/range, lower means trend."""
    n = len(closes_)
    if n <= period:
        return None
    trs = []
    for i in range(1, n):
        trs.append(max(
            highs_[i] - lows_[i],
            abs(highs_[i] - closes_[i - 1]),
            abs(lows_[i] - closes_[i - 1]),
        ))
    if len(trs) < period:
        return None
    sum_tr = sum(trs[-period:])
    hh = max(highs_[-period:])
    ll = min(lows_[-period:])
    if hh == ll or sum_tr == 0:
        return None
    import math as _m
    return 100.0 * _m.log10(sum_tr / (hh - ll)) / _m.log10(period)


def _bb_squeeze_flag(bb_upper: float | None, bb_lower: float | None, ema_value: float | None, atr: float | None, *, mult: float = 1.5) -> int | None:
    """1 if Bollinger Band span is inside Keltner Channel (Carter squeeze), 0 if not."""
    if bb_upper is None or bb_lower is None or ema_value is None or atr is None:
        return None
    kelt_upper = ema_value + mult * atr
    kelt_lower = ema_value - mult * atr
    return 1 if (bb_upper < kelt_upper and bb_lower > kelt_lower) else 0


def _atr_series(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], period: int = 14) -> list[float | None] | None:
    n = len(closes_)
    if n <= period:
        return None
    trs: list[float] = []
    for i in range(1, n):
        trs.append(max(
            highs_[i] - lows_[i],
            abs(highs_[i] - closes_[i - 1]),
            abs(lows_[i] - closes_[i - 1]),
        ))
    out: list[float | None] = [None] * n
    if len(trs) < period:
        return out
    atr = sum(trs[:period]) / period
    out[period] = atr
    for i, tr in enumerate(trs[period:], start=period + 1):
        atr = (atr * (period - 1) + tr) / period
        if i < n:
            out[i] = atr
    return out


def _bb_width_series(values: Sequence[float], period: int = 20, mult: float = 2.0) -> list[float | None] | None:
    n = len(values)
    if n < period:
        return None
    out: list[float | None] = [None] * n
    for i in range(period - 1, n):
        window = values[i - period + 1 : i + 1]
        m = sum(window) / period
        var = sum((v - m) ** 2 for v in window) / period
        sd = var ** 0.5
        if m:
            out[i] = (2.0 * mult * sd) / m
    return out


def _adx_series(highs_: Sequence[float], lows_: Sequence[float], closes_: Sequence[float], period: int = 14) -> list[float | None] | None:
    n = len(closes_)
    if n <= period * 2:
        return None
    tr_list: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, n):
        up = highs_[i] - highs_[i - 1]
        down = lows_[i - 1] - lows_[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        tr_list.append(max(
            highs_[i] - lows_[i],
            abs(highs_[i] - closes_[i - 1]),
            abs(lows_[i] - closes_[i - 1]),
        ))
    if len(tr_list) < period * 2:
        return None
    out: list[float | None] = [None] * n
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
        return out
    adx = sum(dx_list[:period]) / period
    out[period * 2] = adx
    for j, dx in enumerate(dx_list[period:], start=period * 2 + 1):
        adx = (adx * (period - 1) + dx) / period
        if j < n:
            out[j] = adx
    return out


def _percentile_rank(series: list[float | None] | None, value: float | None, *, lookback: int = 100) -> float | None:
    if series is None or value is None:
        return None
    cleaned = [s for s in series[-lookback:] if s is not None]
    if len(cleaned) < 5:
        return None
    below = sum(1 for s in cleaned if s <= value)
    return below / len(cleaned)


def _z_score(values: Sequence[float], period: int = 20) -> float | None:
    if len(values) < period:
        return None
    window = values[-period:]
    m = sum(window) / period
    var = sum((v - m) ** 2 for v in window) / period
    sd = var ** 0.5
    if sd == 0:
        return 0.0
    return (window[-1] - m) / sd


def _realized_vol(values: Sequence[float], period: int = 20) -> float | None:
    if len(values) <= period:
        return None
    rets: list[float] = []
    for i in range(len(values) - period, len(values)):
        if i == 0 or values[i - 1] <= 0 or values[i] <= 0:
            continue
        rets.append(log(values[i] / values[i - 1]))
    if len(rets) < 5:
        return None
    try:
        sd = stdev(rets)
    except Exception:
        return None
    # Annualization assumes 252 trading days × bars-per-day; we report per-bar stdev
    # multiplied by sqrt(252) as a regime indicator (not a precise IV equivalent).
    return sd * (252.0 ** 0.5)


def _anchored_vwap_with_bands(candles: Sequence[Candle]) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """VWAP anchored to first candle of supplied series, plus ±1σ/±2σ deviation bands."""
    if not candles:
        return None, None, None, None, None
    cum_pv = 0.0
    cum_v = 0
    cum_pv2 = 0.0
    for c in candles:
        tp = (c.high + c.low + c.close) / 3.0
        v = max(int(c.volume), 1)
        cum_pv += tp * v
        cum_v += v
        cum_pv2 += tp * tp * v
    if cum_v == 0:
        return None, None, None, None, None
    vwap = cum_pv / cum_v
    var = max(cum_pv2 / cum_v - vwap * vwap, 0.0)
    sd = var ** 0.5
    return vwap, vwap + sd, vwap - sd, vwap + 2 * sd, vwap - 2 * sd


def _anchored_vwap_from_swing(candles: Sequence[Candle], *, side: str, lookback: int = 50) -> float | None:
    """Anchor VWAP to the most recent swing-high (side='high') or swing-low ('low')."""
    if not candles:
        return None
    n = len(candles)
    window = candles[-lookback:] if n > lookback else candles
    if side == "high":
        idx = max(range(len(window)), key=lambda i: window[i].high)
    else:
        idx = min(range(len(window)), key=lambda i: window[i].low)
    anchor = window[idx:]
    if not anchor:
        return None
    cum_pv = 0.0
    cum_v = 0
    for c in anchor:
        tp = (c.high + c.low + c.close) / 3.0
        v = max(int(c.volume), 1)
        cum_pv += tp * v
        cum_v += v
    return (cum_pv / cum_v) if cum_v else None


# ===========================================================================
# Phase B: statistics
# ===========================================================================


def _hurst_exponent(values: Sequence[float], *, max_lag: int = 20) -> float | None:
    """R/S Hurst estimate. >0.5 trending, ~0.5 random walk, <0.5 mean reverting."""
    n = len(values)
    if n < 100 or _np is None:
        return None
    arr = _np.asarray(values, dtype=float)
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = arr[lag:] - arr[:-lag]
        if diff.size == 0:
            continue
        sd = float(_np.std(diff))
        if sd <= 0:
            continue
        tau.append(sd)
    if len(tau) < 4:
        return None
    log_lags = _np.log(_np.array(list(lags))[: len(tau)])
    log_tau = _np.log(_np.array(tau))
    poly = _np.polyfit(log_lags, log_tau, 1)
    return float(poly[0])


def _half_life(values: Sequence[float], period: int = 60) -> float | None:
    """Mean-reversion half-life via Ornstein-Uhlenbeck regression: Δp = α + β·p_{t-1}."""
    if _np is None:
        return None
    if len(values) < period + 2:
        return None
    arr = _np.asarray(values[-(period + 1):], dtype=float)
    lagged = arr[:-1]
    delta = arr[1:] - lagged
    # OLS: delta = α + β·lagged
    x_mean = float(_np.mean(lagged))
    y_mean = float(_np.mean(delta))
    cov = float(_np.sum((lagged - x_mean) * (delta - y_mean)))
    var = float(_np.sum((lagged - x_mean) ** 2))
    if var == 0:
        return None
    beta = cov / var
    if beta >= 0:
        return None  # not mean-reverting
    return float(-log(2.0) / beta)


def _regime_quantile(atr_pct: float | None, adx_pct: float | None) -> str | None:
    """Coarse regime classifier from ATR/ADX percentiles. QUIET / NORMAL / VOLATILE."""
    if atr_pct is None and adx_pct is None:
        return None
    a = atr_pct if atr_pct is not None else 0.5
    d = adx_pct if adx_pct is not None else 0.5
    score = (a + d) / 2.0
    if score < 0.33:
        return "QUIET"
    if score > 0.66:
        return "VOLATILE"
    return "NORMAL"

"""Core indicator calculations without pandas-ta."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class IndicatorEngine:
    """Compute technical indicators from OHLC DataFrame."""

    @staticmethod
    def compute(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {
                "ma10": 0.0,
                "ma20": 0.0,
                "ema12": 0.0,
                "ema20": 0.0,
                "ema24": 0.0,
                "rsi": 0.0,
                "atr": 0.0,
                "atr_pips": 0.0,
                "adx": 0.0,
                "bbw": 0.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_hist": 0.0,
                "roc5": 0.0,
                "roc10": 0.0,
                "cci": 0.0,
                "stoch_rsi": 0.0,
                "plus_di": 0.0,
                "minus_di": 0.0,
                "ema_slope_5": 0.0,
                "ema_slope_10": 0.0,
                "ema_slope_20": 0.0,
                "kc_width": 0.0,
                "donchian_width": 0.0,
                "chaikin_vol": 0.0,
                "vwap_gap": 0.0,
                "swing_dist_high": 0.0,
                "swing_dist_low": 0.0,
                "ichimoku_span_a_gap": 0.0,
                "ichimoku_span_b_gap": 0.0,
                "ichimoku_cloud_pos": 0.0,
                "cluster_high_gap": 0.0,
                "cluster_low_gap": 0.0,
                "upper_wick_avg_pips": 0.0,
                "lower_wick_avg_pips": 0.0,
                "high_hits": 0.0,
                "low_hits": 0.0,
                "high_hit_interval": 0.0,
                "low_hit_interval": 0.0,
            }

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ma10 = close.rolling(window=10, min_periods=10).mean()
        ma20 = close.rolling(window=20, min_periods=20).mean()
        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
        ema24 = close.ewm(span=24, adjust=False, min_periods=24).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()

        # pip単位のボラ（5本平均）
        vol_5m = close.diff().abs().rolling(window=5, min_periods=5).mean() / 0.01

        rsi = _rsi(close, period=14)
        atr = _atr(high, low, close, period=14)
        adx, plus_di, minus_di = _adx(high, low, close, period=14, with_di=True)

        upper, middle, lower = _bollinger(close, period=20, std_mult=2.0)
        bbw_series = np.where(middle != 0, (upper - lower) / middle, 0.0)

        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        macd_hist = macd_line - macd_signal
        roc5 = _roc(close, period=5)
        roc10 = _roc(close, period=10)
        cci = _cci(high, low, close, period=14)
        stoch_rsi = _stoch_rsi(close, period=14)
        ema_slope_5 = _slope(ema12, window=5)
        ema_slope_10 = _slope(ema20, window=10)
        ema_slope_20 = _slope(ema24, window=20)
        kc_width = _keltner_width(close, high, low, period=20, mult=1.5)
        donchian_width = _donchian_width(high, low, period=20)
        chaikin_vol = _chaikin_vol(high, low, period=10, slow=20)
        vwap_gap = _vwap_gap(df)
        swing_high_dist, swing_low_dist = _swing_distance(close, high, low, lookback=50)
        ichimoku_a_gap, ichimoku_b_gap, ichimoku_pos = _ichimoku_position(high, low, close)
        cluster_high_gap, cluster_low_gap = _cluster_distance(high, low, close)
        upper_wick_avg, lower_wick_avg = _wick_ratios(df, window=20)
        high_hits, low_hits, high_int, low_int = _hit_stats(high, low, window=30, band=0.0008)

        out: Dict[str, float] = {
            "ma10": float(ma10.iloc[-1]) if not ma10.empty else 0.0,
            "ma20": float(ma20.iloc[-1]) if not ma20.empty else 0.0,
            "ema12": float(ema12.iloc[-1]) if not ema12.empty else 0.0,
            "ema20": float(ema20.iloc[-1]) if not ema20.empty else 0.0,
            "ema24": float(ema24.iloc[-1]) if not ema24.empty else 0.0,
            "rsi": float(rsi.iloc[-1]) if not rsi.empty else 0.0,
            "atr": float(atr.iloc[-1]) if not atr.empty else 0.0,
            "atr_pips": (float(atr.iloc[-1]) / 0.01) if not atr.empty else 0.0,
            "adx": float(adx.iloc[-1]) if not adx.empty else 0.0,
            "bbw": float(bbw_series[-1]) if bbw_series.size else 0.0,
            "vol_5m": float(vol_5m.iloc[-1]) if not vol_5m.empty else 0.0,
            "macd": float(macd_line.iloc[-1]) if not macd_line.empty else 0.0,
            "macd_signal": float(macd_signal.iloc[-1]) if not macd_signal.empty else 0.0,
            "macd_hist": float(macd_hist.iloc[-1]) if not macd_hist.empty else 0.0,
            "roc5": float(roc5.iloc[-1]) if not roc5.empty else 0.0,
            "roc10": float(roc10.iloc[-1]) if not roc10.empty else 0.0,
            "cci": float(cci.iloc[-1]) if not cci.empty else 0.0,
            "stoch_rsi": float(stoch_rsi.iloc[-1]) if not stoch_rsi.empty else 0.0,
            "plus_di": float(plus_di.iloc[-1]) if not plus_di.empty else 0.0,
            "minus_di": float(minus_di.iloc[-1]) if not minus_di.empty else 0.0,
            "ema_slope_5": float(ema_slope_5) if np.isfinite(ema_slope_5) else 0.0,
            "ema_slope_10": float(ema_slope_10) if np.isfinite(ema_slope_10) else 0.0,
            "ema_slope_20": float(ema_slope_20) if np.isfinite(ema_slope_20) else 0.0,
            "kc_width": float(kc_width) if np.isfinite(kc_width) else 0.0,
            "donchian_width": float(donchian_width) if np.isfinite(donchian_width) else 0.0,
            "chaikin_vol": float(chaikin_vol) if np.isfinite(chaikin_vol) else 0.0,
            "vwap_gap": float(vwap_gap) if np.isfinite(vwap_gap) else 0.0,
            "swing_dist_high": float(swing_high_dist) if np.isfinite(swing_high_dist) else 0.0,
            "swing_dist_low": float(swing_low_dist) if np.isfinite(swing_low_dist) else 0.0,
            "ichimoku_span_a_gap": float(ichimoku_a_gap) if np.isfinite(ichimoku_a_gap) else 0.0,
            "ichimoku_span_b_gap": float(ichimoku_b_gap) if np.isfinite(ichimoku_b_gap) else 0.0,
            "ichimoku_cloud_pos": float(ichimoku_pos) if np.isfinite(ichimoku_pos) else 0.0,
            "cluster_high_gap": float(cluster_high_gap) if np.isfinite(cluster_high_gap) else 0.0,
            "cluster_low_gap": float(cluster_low_gap) if np.isfinite(cluster_low_gap) else 0.0,
            "upper_wick_avg_pips": float(upper_wick_avg) if np.isfinite(upper_wick_avg) else 0.0,
            "lower_wick_avg_pips": float(lower_wick_avg) if np.isfinite(lower_wick_avg) else 0.0,
            "high_hits": float(high_hits),
            "low_hits": float(low_hits),
            "high_hit_interval": float(high_int),
            "low_hit_interval": float(low_int),
        }

        for k, v in out.items():
            if not np.isfinite(v):
                out[k] = 0.0
        return out


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0.0)


def _dm(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    up = high.diff().clip(lower=0.0)
    # Down move is the magnitude of lower lows (positive when price moves down)
    down = (-low.diff()).clip(lower=0.0)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    return pd.Series(plus_dm, index=high.index), pd.Series(minus_dm, index=high.index)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int, with_di: bool = False):
    atr = _atr(high, low, close, period)
    plus_dm, minus_dm = _dm(high, low)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0.0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0.0, np.nan)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs()) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0.0)
    if with_di:
        return adx, plus_di.fillna(0.0), minus_di.fillna(0.0)
    return adx


def _roc(close: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        return pd.Series([0.0])
    return close.pct_change(periods=period).fillna(0.0) * 100


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    typical = (high + low + close) / 3.0
    sma = typical.rolling(window=period, min_periods=period).mean()
    mad = (typical - sma).abs().rolling(window=period, min_periods=period).mean()
    cci = (typical - sma) / (0.015 * mad.replace(0.0, np.nan))
    return cci.fillna(0.0)


def _stoch_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    rsi = _rsi(close, period)
    min_rsi = rsi.rolling(window=period, min_periods=period).min()
    max_rsi = rsi.rolling(window=period, min_periods=period).max()
    denom = (max_rsi - min_rsi).replace(0.0, np.nan)
    stoch = (rsi - min_rsi) / denom
    return stoch.fillna(0.0)


def _slope(series: pd.Series, window: int) -> float:
    if len(series) < window or window <= 1:
        return 0.0
    y = series.iloc[-window:]
    x = np.arange(window)
    if np.ptp(x) == 0:
        return 0.0
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]


def _keltner_width(close: pd.Series, high: pd.Series, low: pd.Series, period: int, mult: float) -> float:
    ema = close.ewm(span=period, adjust=False, min_periods=period).mean()
    atr = _atr(high, low, close, period)
    if ema.empty or atr.empty or ema.iloc[-1] == 0:
        return 0.0
    upper = ema + mult * atr
    lower = ema - mult * atr
    return float((upper.iloc[-1] - lower.iloc[-1]) / ema.iloc[-1]) if ema.iloc[-1] != 0 else 0.0


def _donchian_width(high: pd.Series, low: pd.Series, period: int) -> float:
    if len(high) < period or len(low) < period:
        return 0.0
    upper = high.rolling(window=period, min_periods=period).max().iloc[-1]
    lower = low.rolling(window=period, min_periods=period).min().iloc[-1]
    mid = (upper + lower) / 2.0
    if mid == 0:
        return 0.0
    return float((upper - lower) / mid)


def _chaikin_vol(high: pd.Series, low: pd.Series, period: int = 10, slow: int = 20) -> float:
    if len(high) < slow or len(low) < slow:
        return 0.0
    hl = high - low
    ema_fast = hl.ewm(span=period, adjust=False, min_periods=period).mean()
    ema_slow = hl.ewm(span=slow, adjust=False, min_periods=slow).mean()
    if ema_slow.empty or ema_slow.iloc[-1] == 0:
        return 0.0
    return float((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1])


def _vwap_gap(df: pd.DataFrame) -> float:
    if df.empty or "close" not in df:
        return 0.0
    try:
        price = (df["high"] + df["low"] + df["close"]) / 3.0
        # ボリュームなしのため時間重みで近似
        cum_price = (price * np.arange(1, len(price) + 1)).sum()
        vwap = cum_price / (np.arange(1, len(price) + 1).sum())
        latest = price.iloc[-1]
        if vwap == 0:
            return 0.0
        return float((latest - vwap) / 0.01)
    except Exception:
        return 0.0


def _swing_distance(close: pd.Series, high: pd.Series, low: pd.Series, lookback: int = 50) -> tuple[float, float]:
    if len(close) < max(5, lookback):
        return (0.0, 0.0)
    segment_high = high.iloc[-lookback:]
    segment_low = low.iloc[-lookback:]
    ch = segment_high.max()
    cl = segment_low.min()
    last = close.iloc[-1]
    return ((ch - last) / 0.01, (last - cl) / 0.01)


def _bollinger(close: pd.Series, period: int, std_mult: float):
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper.fillna(0.0), middle.fillna(0.0), lower.fillna(0.0)


def _wick_ratios(df: pd.DataFrame, window: int = 20) -> tuple[float, float]:
    if df.empty:
        return (0.0, 0.0)
    segment = df.iloc[-window:]
    uppers = []
    lowers = []
    for _, row in segment.iterrows():
        try:
            o = float(row["open"])
            c = float(row["close"])
            h = float(row["high"])
            l = float(row["low"])
        except Exception:
            continue
        upper = max(0.0, h - max(o, c)) / 0.01
        lower = max(0.0, min(o, c) - l) / 0.01
        uppers.append(upper)
        lowers.append(lower)
    if not uppers:
        return (0.0, 0.0)
    return (float(np.mean(uppers)), float(np.mean(lowers)))


def _hit_stats(high: pd.Series, low: pd.Series, window: int = 30, band: float = 0.0008) -> tuple[int, int, float, float]:
    """
    Count how many times recent highs/lows retest the extreme within a band, and average interval (bars) between hits.
    """
    if len(high) < window or len(low) < window:
        return (0, 0, 0.0, 0.0)
    segment_high = high.iloc[-window:]
    segment_low = low.iloc[-window:]
    recent_high = segment_high.max()
    recent_low = segment_low.min()
    hit_high_idx = [i for i, val in enumerate(segment_high) if recent_high - val <= band]
    hit_low_idx = [i for i, val in enumerate(segment_low) if val - recent_low <= band]

    def _avg_interval(idxs: list[int]) -> float:
        if len(idxs) < 2:
            return 0.0
        diffs = [idxs[i] - idxs[i - 1] for i in range(1, len(idxs))]
        return float(np.mean(diffs))

    return (len(hit_high_idx), len(hit_low_idx), _avg_interval(hit_high_idx), _avg_interval(hit_low_idx))


def _ichimoku_position(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[float, float, float]:
    """
    Price vs Ichimoku cloud.
    Returns (spanA_gap_pips, spanB_gap_pips, cloud_position)
    cloud_position: >0 above cloud top, <0 below cloud bottom, 0 inside.
    """
    if len(close) < 52:
        return (0.0, 0.0, 0.0)
    tenkan = (high.rolling(window=9, min_periods=9).max() + low.rolling(window=9, min_periods=9).min()) / 2.0
    kijun = (high.rolling(window=26, min_periods=26).max() + low.rolling(window=26, min_periods=26).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = ((high.rolling(window=52, min_periods=52).max() + low.rolling(window=52, min_periods=52).min()) / 2.0).shift(26)
    span_a = senkou_a.iloc[-1] if not senkou_a.empty else np.nan
    span_b = senkou_b.iloc[-1] if not senkou_b.empty else np.nan
    price = close.iloc[-1]
    if not np.isfinite(span_a) or not np.isfinite(span_b) or not np.isfinite(price):
        return (0.0, 0.0, 0.0)
    span_a_gap = (price - span_a) / 0.01
    span_b_gap = (price - span_b) / 0.01
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    if price > cloud_top:
        pos = (price - cloud_top) / 0.01
    elif price < cloud_bottom:
        pos = (price - cloud_bottom) / 0.01
    else:
        pos = 0.0
    return (float(span_a_gap), float(span_b_gap), float(pos))


def _cluster_distance(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 120, bin_size: float = 0.02) -> tuple[float, float]:
    """
    Approximate distance (pips) to nearest high/low price clusters within lookback.
    bin_size in price units (0.02 ~= 2 pips).
    """
    if len(close) < max(10, lookback):
        return (0.0, 0.0)
    last = float(close.iloc[-1])

    def _cluster_gap(series: pd.Series, above: bool) -> float:
        seg = series.iloc[-lookback:]
        if seg.empty:
            return 0.0
        binned = (seg / bin_size).round().astype(int)
        counts = binned.value_counts()
        if counts.empty:
            return 0.0
        center_bin = counts.idxmax()
        center = float(center_bin) * bin_size
        if above:
            gap = max(0.0, center - last)
        else:
            gap = max(0.0, last - center)
        return gap / 0.01

    return (_cluster_gap(high, above=True), _cluster_gap(low, above=False))

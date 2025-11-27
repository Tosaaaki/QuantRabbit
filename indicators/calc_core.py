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
                "adx": 0.0,
                "bbw": 0.0,
            }

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ma10 = close.rolling(window=10, min_periods=10).mean()
        ma20 = close.rolling(window=20, min_periods=20).mean()
        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
        ema24 = close.ewm(span=24, adjust=False, min_periods=24).mean()

        rsi = _rsi(close, period=14)
        atr = _atr(high, low, close, period=14)
        adx = _adx(high, low, close, period=14)

        upper, middle, lower = _bollinger(close, period=20, std_mult=2.0)
        bbw_series = np.where(middle != 0, (upper - lower) / middle, 0.0)

        out: Dict[str, float] = {
            "ma10": float(ma10.iloc[-1]) if not ma10.empty else 0.0,
            "ma20": float(ma20.iloc[-1]) if not ma20.empty else 0.0,
            "ema12": float(ema12.iloc[-1]) if not ema12.empty else 0.0,
            "ema20": float(ema20.iloc[-1]) if not ema20.empty else 0.0,
            "ema24": float(ema24.iloc[-1]) if not ema24.empty else 0.0,
            "rsi": float(rsi.iloc[-1]) if not rsi.empty else 0.0,
            "atr": float(atr.iloc[-1]) if not atr.empty else 0.0,
            "adx": float(adx.iloc[-1]) if not adx.empty else 0.0,
            "bbw": float(bbw_series[-1]) if bbw_series.size else 0.0,
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


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    atr = _atr(high, low, close, period)
    plus_dm, minus_dm = _dm(high, low)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0.0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0.0, np.nan)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs()) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


def _bollinger(close: pd.Series, period: int, std_mult: float):
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper.fillna(0.0), middle.fillna(0.0), lower.fillna(0.0)

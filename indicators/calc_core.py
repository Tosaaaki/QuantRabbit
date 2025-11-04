"""
indicators.calc_core
~~~~~~~~~~~~~~~~~~~~

ローソク足 DataFrame（open/high/low/close）から主要テクニカル指標を算出する。
外部依存に頼らず pandas だけで再計算する実装。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _ensure_length(series: pd.Series, length: int) -> bool:
    return len(series.dropna()) >= length


def _sma(series: pd.Series, length: int) -> float:
    if not _ensure_length(series, length):
        return 0.0
    return float(series.rolling(window=length).mean().iloc[-1])


def _ema(series: pd.Series, span: int) -> float:
    if not _ensure_length(series, span):
        return 0.0
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _rsi(series: pd.Series, period: int) -> float:
    if not _ensure_length(series, period + 1):
        return 0.0
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    if avg_loss.iloc[-1] == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    return float(100.0 - (100.0 / (1.0 + rs)))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
    if len(close) < period + 1:
        return 0.0
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return float(atr.iloc[-1])


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
    if len(close) < period * 2:
        return 0.0

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    atr = atr.replace(0, np.nan)

    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean() / atr

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100

    adx = dx.rolling(window=period, min_periods=period).mean()
    return float(adx.iloc[-1]) if not adx.empty else 0.0


def _bollinger_width(series: pd.Series, length: int, std_mult: float) -> tuple[float, float, float, float]:
    if not _ensure_length(series, length):
        return 0.0, 0.0, 0.0, 0.0
    middle = series.rolling(window=length).mean().iloc[-1]
    std = series.rolling(window=length).std(ddof=0).iloc[-1]
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bbw = float((upper - lower) / middle) if middle else 0.0
    return float(upper), float(middle), float(lower), bbw


def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> tuple[float, float, float]:
    if len(series.dropna()) < slow:
        return 0.0, 0.0, 0.0
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return (
        float(macd_line.iloc[-1]),
        float(signal_line.iloc[-1]),
        float(hist.iloc[-1]),
    )


class IndicatorEngine:
    """ローソク足 DataFrame から指標を計算するユーティリティ."""

    @staticmethod
    def compute(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {
                "ma10": 0.0,
                "ma20": 0.0,
                "ema20": 0.0,
                "rsi": 0.0,
                "atr": 0.0,
                "adx": 0.0,
                "bbw": 0.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_hist": 0.0,
            }

        df = df.astype(float).copy()
        closes = df["close"]
        highs = df["high"]
        lows = df["low"]

        ma10 = _sma(closes, 10)
        ma20 = _sma(closes, 20)
        ema20 = _ema(closes, 20)
        rsi = _rsi(closes, 14)
        atr = _atr(highs, lows, closes, 14)
        adx = _adx(highs, lows, closes, 14)
        upper, middle, lower, bbw = _bollinger_width(closes, 20, 2.0)
        macd_line, macd_signal, macd_hist = _macd(closes, 12, 26, 9)

        out: Dict[str, float] = {
            "ma10": ma10,
            "ma20": ma20,
            "ema20": ema20,
            "rsi": rsi,
            "atr": atr,
            "adx": adx,
            "bbw": bbw,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        }

        # fallback for NaNs
        for k, v in out.items():
            if pd.isna(v) or np.isinf(v):
                out[k] = 0.0

        return out

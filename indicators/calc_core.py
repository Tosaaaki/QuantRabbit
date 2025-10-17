"""
indicators.calc_core
~~~~~~~~~~~~~~~~~~~~

ta‑lib をラップし、DataFrame（open/high/low/close）から
主要テクニカル指標を 1 行 dict で返すエンジン。

・終値ベース SMA10/20, EMA20
・RSI14, ATR14, ADX14
・Bollinger 幅 (upper-lower)/middle
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd


class IndicatorEngine:
    """ローソク足 DataFrame から指標を計算."""

    @staticmethod
    def compute(df: pd.DataFrame) -> Dict[str, float]:
        """
        Parameters
        ----------
        df :  columns = ['open','high','low','close']
              index は時間順 (昇順)

        Returns
        -------
        Dict[str, float]
            {
              "ma10": 157.25,
              "ma20": ...,
              "ema20": ...,
              "rsi":  63.2,
              "atr":  0.35,
              "adx":  28.4,
              "bbw":  0.26
            }
        """
        df = df.astype(float).copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        out: Dict[str, float] = {}

        # Moving averages
        out["ma10"] = close.rolling(window=10).mean().iloc[-1]
        out["ma20"] = close.rolling(window=20).mean().iloc[-1]
        out["ema20"] = close.ewm(span=20, adjust=False).mean().iloc[-1]

        # RSI (Wilder's smoothing)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        out["rsi"] = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 0.0

        # True range & ATR
        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = true_range.ewm(alpha=1 / 14, adjust=False).mean()
        out["atr"] = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0

        # ADX
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
        out["adx"] = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0

        # Bollinger Bands (20, 2)
        ma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        middle = ma20
        if middle.iloc[-1] != 0 and not np.isnan(middle.iloc[-1]):
            out["bbw"] = float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1])
        else:
            out["bbw"] = 0.0

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        out["macd"] = float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else 0.0
        out["macd_signal"] = (
            float(macd_signal.iloc[-1]) if not np.isnan(macd_signal.iloc[-1]) else 0.0
        )
        out["macd_hist"] = (
            float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0.0
        )

        # Normalize NaNs to 0.0
        for k, v in out.items():
            if pd.isna(v) or np.isnan(v):
                out[k] = 0.0

        return out

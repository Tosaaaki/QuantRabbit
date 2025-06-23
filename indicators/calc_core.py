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
import talib


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
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        out: Dict[str, float] = {}

        out["ma10"] = float(talib.SMA(close, 10)[-1])
        out["ma20"] = float(talib.SMA(close, 20)[-1])
        out["ema20"] = float(talib.EMA(close, 20)[-1])

        out["rsi"] = float(talib.RSI(close, 14)[-1])
        out["atr"] = float(talib.ATR(high, low, close, 14)[-1])
        out["adx"] = float(talib.ADX(high, low, close, 14)[-1])

        upper, middle, lower = talib.BBANDS(close, 20, 2, 2)
        if middle[-1] != 0:
            out["bbw"] = float((upper[-1] - lower[-1]) / middle[-1])
        else:
            out["bbw"] = 0.0

        # 数値が nan の場合は 0.0 で埋める
        for k, v in out.items():
            if v is None or np.isnan(v):
                out[k] = 0.0

        return out
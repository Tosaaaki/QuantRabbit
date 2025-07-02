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
import pandas_ta as pta


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
        # pandas_ta は DataFrame に直接アクセスできる
        # df.ta.sma() のように呼び出す
        df.ta.sma(length=10, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        out: Dict[str, float] = {}

        # 最新の値を抽出
        out["ma10"] = df["SMA_10"].iloc[-1]
        out["ma20"] = df["SMA_20"].iloc[-1]
        out["ema20"] = df["EMA_20"].iloc[-1]

        out["rsi"] = df["RSI_14"].iloc[-1]
        out["atr"] = df["ATR_14"].iloc[-1]
        out["adx"] = df["ADX_14"].iloc[-1]

        # Bollinger Bands の計算
        upper = df["BBU_20_2.0"].iloc[-1]
        middle = df["BBM_20_2.0"].iloc[-1]
        lower = df["BBL_20_2.0"].iloc[-1]

        if middle != 0:
            out["bbw"] = float((upper - lower) / middle)
        else:
            out["bbw"] = 0.0

        # 数値が nan の場合は 0.0 で埋める
        for k, v in out.items():
            if pd.isna(v):
                out[k] = 0.0

        return out
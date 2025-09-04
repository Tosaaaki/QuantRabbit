import pandas as pd
from typing import Dict

from analysis.regime_classifier import THRESH_ADX_TREND, THRESH_BBW_RANGE


class Donchian55:
    name = "Donchian55"
    pocket = "macro"

    @staticmethod
    def check(fac_m1: Dict, fac_h4: Dict) -> Dict | None:
        """
        H4 の 55 本ドンチャンブレイクで順張り。ADX/BBW でブレイク質をフィルタ。
        M1 は使用しない（今はフィルタ無し）。
        """
        candles = fac_h4.get("candles")
        if candles is None or len(candles) < 56:
            return None

        df = pd.DataFrame(candles)[-56:]
        high55 = df["high"][:-1].max()
        low55 = df["low"][:-1].min()
        close = df["close"].iloc[-1]

        adx_h4 = float(fac_h4.get("adx", 0.0) or 0.0)
        bbw_h4 = float(fac_h4.get("bbw", 1.0) or 1.0)
        # トレンド立ち上がり（ADX しきい値近辺以上）かつ BB 幅拡大でのブレイクのみ採用
        if adx_h4 < (THRESH_ADX_TREND["H4"] - 2) or bbw_h4 <= THRESH_BBW_RANGE["H4"]:
            return None

        # ドンチャンはボラに合わせてATR連動の方が安定しやすい
        if close > high55:
            return {"action": "buy", "sl_atr_mult": 2.0, "tp_atr_mult": 4.0, "be_rr": 1.0, "trail_at_rr": 1.5, "trail_atr_mult": 2.5}
        if close < low55:
            return {"action": "sell", "sl_atr_mult": 2.0, "tp_atr_mult": 4.0, "be_rr": 1.0, "trail_at_rr": 1.5, "trail_atr_mult": 2.5}
        return None

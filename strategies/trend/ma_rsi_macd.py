from __future__ import annotations
from typing import Dict

from analysis.regime_classifier import THRESH_ADX_TREND, THRESH_MA_SLOPE


class MaRsiMacd:
    name = "MA_RSI_MACD"
    pocket = "macro"

    @staticmethod
    def check(fac_m1: Dict, fac_h4: Dict) -> Dict | None:
        """
        H4: トレンド方向・強さの確認（ADX/MA傾き）
        M1: タイミング判定（RSI + MACD方向）

        参照:
          - 20/50 のクロス相当は H4 の MA10/20 で代替の方向判定
          - RSI(14) をフィルタ
          - MACD(12,26,9) のヒストグラム向きで勢い確認
        """
        ma10_h4 = fac_h4.get("ma10")
        ma20_h4 = fac_h4.get("ma20")
        adx_h4 = float(fac_h4.get("adx", 0.0) or 0.0)
        if not ma10_h4 or not ma20_h4:
            return None

        slope_h4 = abs(ma20_h4 - ma10_h4) / ma10_h4 if ma10_h4 else 0.0
        if not (adx_h4 >= THRESH_ADX_TREND["H4"] and slope_h4 >= THRESH_MA_SLOPE["H4"]):
            return None

        rsi_m1 = float(fac_m1.get("rsi", 50.0) or 50.0)
        macd_hist = float(fac_m1.get("macd_hist", 0.0) or 0.0)
        price = float(fac_m1.get("close", 0.0) or 0.0)
        ma20_m1 = float(fac_m1.get("ma20", 0.0) or 0.0)

        # ATR連動のSL/TP（ボラ対応）: SL=1.5ATR, TP=3ATR
        if ma10_h4 > ma20_h4:
            if rsi_m1 >= 55 and macd_hist > 0 and price >= ma20_m1:
                return {"action": "buy", "sl_atr_mult": 1.5, "tp_atr_mult": 3.0, "be_rr": 1.0, "trail_at_rr": 1.5}
        else:
            if rsi_m1 <= 45 and macd_hist < 0 and (ma20_m1 and price <= ma20_m1):
                return {"action": "sell", "sl_atr_mult": 1.5, "tp_atr_mult": 3.0, "be_rr": 1.0, "trail_at_rr": 1.5}

        return None


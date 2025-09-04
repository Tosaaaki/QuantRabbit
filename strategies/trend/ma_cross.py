from __future__ import annotations
from typing import Dict

from analysis.regime_classifier import THRESH_ADX_TREND, THRESH_MA_SLOPE


class MovingAverageCross:
    name = "TrendMA"
    pocket = "macro"

    @staticmethod
    def check(fac_m1: Dict, fac_h4: Dict) -> Dict | None:
        """
        H4 でトレンド方向と強さを判定し、M1 はエントリーの簡易フィルタに使用。
        条件:
          - H4: MA10 と MA20 の方向が明確、かつ ADX/傾きが閾値以上
          - M1: 方向合致 + RSI フィルタで順張りエントリーを厳選
        """
        ma10_h4 = fac_h4.get("ma10")
        ma20_h4 = fac_h4.get("ma20")
        adx_h4 = fac_h4.get("adx", 0.0)
        if not ma10_h4 or not ma20_h4:
            return None

        slope_h4 = abs(ma20_h4 - ma10_h4) / ma10_h4 if ma10_h4 else 0.0
        if not (adx_h4 >= THRESH_ADX_TREND["H4"] and slope_h4 >= THRESH_MA_SLOPE["H4"]):
            return None

        # M1 フィルタ: 方向一致 + 軽いモメンタム
        ma20_m1 = fac_m1.get("ma20") or 0.0
        rsi_m1 = fac_m1.get("rsi") or 50.0

        # ATR連動: 初期SL/TPは ATR の倍率で動的設定（sl=1.5ATR, tp=3ATR 相当）
        if ma10_h4 > ma20_h4 and (ma20_m1 and fac_m1.get("close", 0) >= ma20_m1) and rsi_m1 >= 55:
            return {"action": "buy", "sl_atr_mult": 1.5, "tp_atr_mult": 3.0}
        if ma10_h4 < ma20_h4 and (ma20_m1 and fac_m1.get("close", 0) <= ma20_m1) and rsi_m1 <= 45:
            return {"action": "sell", "sl_atr_mult": 1.5, "tp_atr_mult": 3.0}
        return None

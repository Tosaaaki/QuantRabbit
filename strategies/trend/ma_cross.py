from __future__ import annotations
from typing import Dict

from analysis.regime_classifier import THRESH_ADX_TREND, THRESH_MA_SLOPE


MAX_ATR_EXT_RATIO = 0.45  # M1 close が MA20 から乖離できる ATR 比率上限
MAX_RSI_EXTREME = 72.0    # 過熱感フィルタ（順張りでも買われ過ぎは回避）


class MovingAverageCross:
    name = "TrendMA"
    pocket = "macro"
    requires_h4 = True

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
        adx_th = THRESH_ADX_TREND["H4"]
        slope_th = THRESH_MA_SLOPE["H4"]
        trend_confirmed = adx_h4 >= adx_th and slope_h4 >= slope_th
        trend_soft = adx_h4 >= (adx_th - 3.0) and slope_h4 >= (slope_th * 0.7)
        if not (trend_confirmed or trend_soft):
            return None

        # M1 フィルタ: 方向一致 + 軽いモメンタム
        ma20_m1 = fac_m1.get("ma20") or 0.0
        close_m1 = fac_m1.get("close") or 0.0
        rsi_m1 = fac_m1.get("rsi") or 50.0
        atr_m1 = float(fac_m1.get("atr") or 0.0)
        if atr_m1 <= 0.0:
            atr_m1 = 0.01  # 乖離判定のゼロ除算回避（USD/JPY 1pip ≒0.01）

        # 伸び切りを避けるため、MA20 からの距離を ATR 比率で制限
        dist_ratio = abs(close_m1 - ma20_m1) / atr_m1 if atr_m1 else 0.0

        # ATR連動: 初期SL/TPは ATR の倍率で動的設定（sl≈1.2ATR, tp≈2.0ATR）
        dist_limit = MAX_ATR_EXT_RATIO * (1.4 if trend_confirmed else 1.7)

        if (
            ma10_h4 > ma20_h4
            and ma20_m1
            and close_m1 >= ma20_m1
            and rsi_m1 >= 48
            and rsi_m1 <= MAX_RSI_EXTREME + 5.0
            and dist_ratio <= dist_limit
        ):
            return {
                "action": "buy",
                "sl_atr_mult": 1.1,
                "tp_atr_mult": 1.9,
                "sl_cap_pips": 24.0,
                "sl_floor_pips": 8.0,
                "tp_cap_pips": 40.0,
            }
        if (
            ma10_h4 < ma20_h4
            and ma20_m1
            and close_m1 <= ma20_m1
            and rsi_m1 <= 55
            and rsi_m1 >= 42.0
            and dist_ratio <= dist_limit
        ):
            return {
                "action": "sell",
                "sl_atr_mult": 1.1,
                "tp_atr_mult": 1.9,
                "sl_cap_pips": 24.0,
                "sl_floor_pips": 8.0,
                "tp_cap_pips": 40.0,
            }
        return None

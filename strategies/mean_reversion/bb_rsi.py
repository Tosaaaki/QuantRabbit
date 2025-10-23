from __future__ import annotations
from typing import Dict

PIP = 0.01


class BBRsi:
    name = "BB_RSI"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        rsi = fac.get("rsi")
        bbw = fac.get("bbw")
        ma = fac.get("ma20")
        if not all([rsi, bbw, ma]):
            return None

        price = fac.get("close", ma)
        upper = ma + (ma * bbw / 2)
        lower = ma - (ma * bbw / 2)
        band_width = upper - lower if upper is not None and lower is not None else 0.0
        if band_width <= 0:
            return None

        atr_hint = fac.get("atr_pips") or (fac.get("atr") or 0.0) * 100 or 6.0
        tp_dynamic = max(1.4, min(atr_hint * 1.35, 6.0))
        sl_dynamic = max(1.1, min(atr_hint * 1.05, tp_dynamic * 1.02))

        rsi_eta_up = fac.get("rsi_eta_upper_min")
        rsi_eta_dn = fac.get("rsi_eta_lower_min")
        bbw_eta = fac.get("bbw_squeeze_eta_min")
        bbw_slope = fac.get("bbw_slope_per_bar", 0.0) or 0.0
        # 逆張り：バンド端＋RSI極端＋（可能なら）RSIが反転方向へ傾いていること
        if price < lower and rsi < 30:
            distance = (lower - price) / band_width if band_width else 0.0
            rsi_gap = max(0.0, 30 - rsi) / 30
            eta_bonus = 0.0
            if rsi_eta_up is not None:
                eta_bonus = max(0.0, min(12.0, (8.0 - min(8.0, rsi_eta_up)) * 1.4))
            squeeze_bias = 0.0 if bbw_eta is None else max(0.0, 6.0 - min(6.0, bbw_eta)) * 0.6
            if bbw_slope > 0:
                squeeze_bias *= 0.5
            confidence = int(max(35.0, min(95.0, 45.0 + distance * 70.0 + rsi_gap * 35.0 + eta_bonus + squeeze_bias)))
            return {
                "action": "OPEN_LONG",
                "sl_pips": sl_dynamic,
                "tp_pips": tp_dynamic,
                "confidence": confidence,
                "tag": f"{BBRsi.name}-long",
            }
        if price > upper and rsi > 70:
            distance = (price - upper) / band_width if band_width else 0.0
            rsi_gap = max(0.0, rsi - 70) / 30
            eta_bonus = 0.0
            if rsi_eta_dn is not None:
                eta_bonus = max(0.0, min(12.0, (8.0 - min(8.0, rsi_eta_dn)) * 1.4))
            squeeze_bias = 0.0 if bbw_eta is None else max(0.0, 6.0 - min(6.0, bbw_eta)) * 0.6
            if bbw_slope > 0:
                squeeze_bias *= 0.5
            confidence = int(max(35.0, min(95.0, 45.0 + distance * 70.0 + rsi_gap * 35.0 + eta_bonus + squeeze_bias)))
            return {
                "action": "OPEN_SHORT",
                "sl_pips": sl_dynamic,
                "tp_pips": tp_dynamic,
                "confidence": confidence,
                "tag": f"{BBRsi.name}-short",
            }
        return None

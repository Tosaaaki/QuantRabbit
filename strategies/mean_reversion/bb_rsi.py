from __future__ import annotations
from typing import Dict


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

        if price < lower and rsi < 30:
            distance = (lower - price) / band_width if band_width else 0.0
            rsi_gap = max(0.0, 30 - rsi) / 30
            confidence = int(
                max(35.0, min(95.0, 45.0 + distance * 80.0 + rsi_gap * 40.0))
            )
            return {
                "action": "OPEN_LONG",
                "sl_pips": 10,
                "tp_pips": 15,
                "confidence": confidence,
                "tag": f"{BBRsi.name}-long",
            }
        if price > upper and rsi > 70:
            distance = (price - upper) / band_width if band_width else 0.0
            rsi_gap = max(0.0, rsi - 70) / 30
            confidence = int(
                max(35.0, min(95.0, 45.0 + distance * 80.0 + rsi_gap * 40.0))
            )
            return {
                "action": "OPEN_SHORT",
                "sl_pips": 10,
                "tp_pips": 15,
                "confidence": confidence,
                "tag": f"{BBRsi.name}-short",
            }
        return None

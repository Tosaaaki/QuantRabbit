from __future__ import annotations
from typing import Dict


class MovingAverageCross:
    name = "TrendMA"
    pocket = "macro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        ma10 = fac.get("ma10")
        ma20 = fac.get("ma20")
        adx = fac.get("adx", 0.0)
        if not ma10 or not ma20:
            return None
        spread = abs(ma10 - ma20) / ma20 if ma20 else 0.0
        base_conf = min(50.0 + spread * 8000.0, 95.0)
        trend_bonus = max(0.0, adx - 20.0) * 1.5
        confidence = int(max(40.0, min(95.0, base_conf + trend_bonus)))
        if ma10 > ma20:
            return {
                "action": "OPEN_LONG",
                "sl_pips": 30,
                "tp_pips": 60,
                "confidence": confidence,
                "tag": f"{MovingAverageCross.name}-bull",
            }
        if ma10 < ma20:
            return {
                "action": "OPEN_SHORT",
                "sl_pips": 30,
                "tp_pips": 60,
                "confidence": confidence,
                "tag": f"{MovingAverageCross.name}-bear",
            }
        return None

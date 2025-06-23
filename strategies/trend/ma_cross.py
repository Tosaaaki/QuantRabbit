from __future__ import annotations
from typing import Dict
from indicators.factor_cache import get

class MovingAverageCross:
    name = "TrendMA"
    pocket = "macro"

    @staticmethod
    def check() -> Dict | None:
        ma10 = get("ma10"); ma20 = get("ma20")
        if not ma10 or not ma20:
            return None
        if ma10 > ma20:        # ゴールデン
            return {"action":"buy","sl_pips":30,"tp_pips":60}
        if ma10 < ma20:        # デッド
            return {"action":"sell","sl_pips":30,"tp_pips":60}
        return None
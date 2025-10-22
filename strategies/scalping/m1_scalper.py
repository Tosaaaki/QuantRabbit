from __future__ import annotations
from typing import Dict


class M1Scalper:
    name = "M1Scalper"
    pocket = "scalp"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = fac.get("close")
        ema20 = fac.get("ema20")
        rsi = fac.get("rsi")
        atr = fac.get("atr", 0.02)
        if close is None or ema20 is None or rsi is None:
            return None

        momentum = close - ema20
        atr_pips = atr * 100

        if atr_pips < 3:
            return None

        if momentum < -0.005 and rsi < 52:
            return {"action": "buy", "sl_pips": 6, "tp_pips": 9}
        if momentum > 0.005 and rsi > 48:
            return {"action": "sell", "sl_pips": 6, "tp_pips": 9}
        return None

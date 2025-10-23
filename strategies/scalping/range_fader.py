from __future__ import annotations

from typing import Dict


class RangeFader:
    name = "RangeFader"
    pocket = "scalp"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = fac.get("close")
        ema20 = fac.get("ema20")
        rsi = fac.get("rsi")
        atr_pips = fac.get("atr_pips")
        vol_5m = fac.get("vol_5m", 1.0)
        if close is None or ema20 is None or rsi is None:
            return None

        if atr_pips is None:
            atr = fac.get("atr")
            atr_pips = (atr or 0.0) * 100

        if atr_pips < 1.6 or vol_5m < 0.8:
            return None

        momentum = close - ema20
        if abs(momentum) > 0.008:
            return None

        if rsi <= 34:
            sl = max(3.2, min(4.8, atr_pips * 1.6))
            tp = max(3.0, min(4.4, atr_pips * 1.5))
            confidence = int(min(90, max(45, (36 - rsi) * 3.2 + vol_5m * 5)))
            return {
                "action": "OPEN_LONG",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "tag": f"{RangeFader.name}-buy-fade",
            }

        if rsi >= 66:
            sl = max(3.2, min(4.8, atr_pips * 1.6))
            tp = max(3.0, min(4.4, atr_pips * 1.5))
            confidence = int(min(90, max(45, (rsi - 64) * 3.2 + vol_5m * 5)))
            return {
                "action": "OPEN_SHORT",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "tag": f"{RangeFader.name}-sell-fade",
            }

        return None

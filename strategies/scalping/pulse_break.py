from __future__ import annotations

from typing import Dict


class PulseBreak:
    name = "PulseBreak"
    pocket = "scalp"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = fac.get("close")
        ema20 = fac.get("ema20")
        ema50 = fac.get("ema50") or fac.get("ma20")
        atr_pips = fac.get("atr_pips")
        vol_5m = fac.get("vol_5m", 1.0)
        if close is None or ema20 is None or ema50 is None:
            return None

        if atr_pips is None:
            atr = fac.get("atr")
            atr_pips = (atr or 0.0) * 100

        if atr_pips < 2.4 or vol_5m < 1.1:
            return None

        momentum = close - ema20
        bias = ema20 - ema50
        if abs(momentum) < 0.003 or abs(bias) < 0.002:
            return None

        tp = max(4.8, min(7.5, atr_pips * 2.4))
        sl = max(3.6, min(tp * 0.75, atr_pips * 1.8))

        if momentum > 0 and bias > 0:
            confidence = int(min(95, max(55, (momentum + bias) * 7500 + vol_5m * 6)))
            return {
                "action": "OPEN_LONG",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "tag": f"{PulseBreak.name}-momentum-up",
            }

        if momentum < 0 and bias < 0:
            confidence = int(min(95, max(55, abs(momentum + bias) * 7500 + vol_5m * 6)))
            return {
                "action": "OPEN_SHORT",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "tag": f"{PulseBreak.name}-momentum-down",
            }

        return None

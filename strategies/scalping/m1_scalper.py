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

        if atr_pips < 2:
            return None

        if momentum < -0.0025 and rsi < 54:
            speed = abs(momentum) / max(0.0005, atr)
            rsi_gap = max(0.0, 54 - rsi) / 10
            confidence = int(
                max(40.0, min(95.0, 45.0 + speed * 30.0 + rsi_gap * 25.0))
            )
            return {
                "action": "OPEN_LONG",
                "sl_pips": 6,
                "tp_pips": 9,
                "confidence": confidence,
                "tag": f"{M1Scalper.name}-buy-dip",
            }
        if momentum > 0.0025 and rsi > 46:
            speed = abs(momentum) / max(0.0005, atr)
            rsi_gap = max(0.0, rsi - 46) / 10
            confidence = int(
                max(40.0, min(95.0, 45.0 + speed * 30.0 + rsi_gap * 25.0))
            )
            return {
                "action": "OPEN_SHORT",
                "sl_pips": 6,
                "tp_pips": 9,
                "confidence": confidence,
                "tag": f"{M1Scalper.name}-sell-rally",
            }
        return None

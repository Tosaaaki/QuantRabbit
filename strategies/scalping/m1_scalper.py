from __future__ import annotations
from typing import Dict
import os


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
        # Prefer explicit atr_pips if provided; otherwise convert ATR (price units) to pips
        atr_pips = fac.get("atr_pips")
        if atr_pips is None:
            atr_pips = (atr or 0.0) * 100

        if atr_pips < 2:
            return None

        # Dynamic TP/SL (pips) tuned to recent volatility
        # Tactical scalp mode prefers narrower targets with a small emergency SL.
        scalp_tactical = os.getenv("SCALP_TACTICAL", "0").strip().lower() not in {"", "0", "false", "no"}
        if scalp_tactical:
            tp_dyn = max(2.2, min(3.6, atr_pips * 1.3))
            sl_dyn = max(2.0, min(3.4, atr_pips * 1.2, tp_dyn * 0.95))
        else:
            # - TP ≈ 3x ATR (pips) within [5, 9]
            # - SL ≈ min(2x ATR, 0.95*TP) with a floor of 4, keeping RR >= ~1.05
            tp_dyn = max(5.0, min(9.0, atr_pips * 3.0))
            sl_dyn = max(4.0, min(atr_pips * 2.0, tp_dyn * 0.95))
        tp_dyn = round(tp_dyn, 2)
        sl_dyn = round(sl_dyn, 2)

        if momentum < -0.0025 and rsi < 54:
            speed = abs(momentum) / max(0.0005, atr)
            rsi_gap = max(0.0, 54 - rsi) / 10
            confidence = int(
                max(40.0, min(95.0, 45.0 + speed * 30.0 + rsi_gap * 25.0))
            )
            return {
                "action": "OPEN_LONG",
                "sl_pips": sl_dyn,
                "tp_pips": tp_dyn,
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
                "sl_pips": sl_dyn,
                "tp_pips": tp_dyn,
                "confidence": confidence,
                "tag": f"{M1Scalper.name}-sell-rally",
            }
        return None

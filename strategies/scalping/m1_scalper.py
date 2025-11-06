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
        adx = fac.get("adx", 0.0) or 0.0
        vol5 = fac.get("vol_5m", 0.0) or 0.0
        bbw = fac.get("bbw") or 0.0
        if close is None or ema20 is None or rsi is None:
            return None

        momentum = close - ema20
        # Prefer explicit atr_pips if provided; otherwise convert ATR (price units) to pips
        atr_pips = fac.get("atr_pips")
        if atr_pips is None:
            atr_pips = (atr or 0.0) * 100

        if atr_pips < 2.5:
            return None
        # Avoid tight range compression; prefer moderate activity
        if bbw and bbw <= 0.20:
            return None
        if vol5 < 1.2:
            return None
        if adx < 18.0:
            return None

        # Dynamic TP/SL (pips) tuned to recent volatility
        # - TP ≈ 3x ATR (pips) within [5, 9]
        # - SL ≈ min(2x ATR, 0.95*TP) with a floor of 4, keeping RR >= ~1.05
        tp_dyn = max(5.0, min(9.0, atr_pips * 3.0))
        sl_dyn = max(4.0, min(atr_pips * 2.0, tp_dyn * 0.95))
        tp_dyn = round(tp_dyn, 2)
        sl_dyn = round(sl_dyn, 2)

        if momentum < -0.0030 and rsi < 54:
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
        if momentum > 0.0030 and rsi > 46:
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

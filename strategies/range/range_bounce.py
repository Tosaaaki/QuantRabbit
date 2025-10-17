from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

PIP = 0.01  # USD/JPY


@dataclass
class RangeSignal:
    action: str
    sl_pips: float
    tp_pips: float


class RangeBounce:
    name = "RangeBounce"
    pocket = "micro"

    @staticmethod
    def check(
        fac_m1: Dict[str, float],
        fac_h4: Dict[str, float],
        micro_regime: str,
        macro_regime: str,
    ) -> Dict[str, float] | None:
        if micro_regime not in ("Range", "Mixed"):
            return None

        close = fac_m1.get("close")
        ma20 = fac_m1.get("ma20")
        bbw = fac_m1.get("bbw")
        atr = fac_m1.get("atr")
        if None in (close, ma20, bbw, atr):
            return None

        close = float(close)
        ma20 = float(ma20)
        bbw = float(bbw)
        atr = float(atr)
        atr_pips = max(atr / PIP, 0.1)

        # ブレイク仕掛け中はスキップ
        bbw_h4 = float(fac_h4.get("bbw", 1.0) or 1.0)
        if macro_regime == "Breakout" or bbw_h4 > 0.6:
            return None

        deviation_pips = (close - ma20) / PIP
        threshold = max(0.35, atr_pips * 0.2)
        far_threshold = threshold * 1.1

        if deviation_pips <= -threshold:
            sl = max(1.5, atr_pips * 0.5)
            tp = max(2.2, sl * 1.35)
            return {"action": "buy", "sl_pips": sl, "tp_pips": tp}
        if deviation_pips >= threshold:
            sl = max(1.5, atr_pips * 0.5)
            tp = max(2.2, sl * 1.35)
            return {"action": "sell", "sl_pips": sl, "tp_pips": tp}
        return None

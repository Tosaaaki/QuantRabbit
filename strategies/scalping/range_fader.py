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

        if atr_pips < 1.2 or atr_pips > 3.6:
            return None
        if vol_5m < 0.6 or vol_5m > 2.3:
            return None

        bbw = fac.get("bbw", 0.0) or 0.0
        bbw_eta = fac.get("bbw_squeeze_eta_min")
        # レンジ確度：BBW が小さい／またはさらに縮小方向
        if bbw > 0.32:
            return None
        if bbw_eta is not None and bbw_eta > 10.0:
            return None

        momentum_pips = abs(close - ema20) / 0.01
        drift_cap = max(2.0, min(4.5, atr_pips * 1.08))
        if momentum_pips > drift_cap:
            return None

        if rsi <= 36:
            sl = max(3.0, min(4.8, atr_pips * 1.5))
            tp = max(2.8, min(4.2, atr_pips * 1.35))
            eta_bonus = 0.0
            rsi_eta_up = fac.get("rsi_eta_upper_min")
            if rsi_eta_up is not None:
                eta_bonus = max(0.0, min(8.0, (6.0 - min(6.0, rsi_eta_up)) * 1.0))
            confidence = int(min(90, max(45, (38 - rsi) * 2.6 + vol_5m * 5.5 + eta_bonus)))
            return {
                "action": "OPEN_LONG",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "tag": f"{RangeFader.name}-buy-fade",
            }

        if rsi >= 64:
            sl = max(3.0, min(4.8, atr_pips * 1.5))
            tp = max(2.8, min(4.2, atr_pips * 1.35))
            eta_bonus = 0.0
            rsi_eta_dn = fac.get("rsi_eta_lower_min")
            if rsi_eta_dn is not None:
                eta_bonus = max(0.0, min(8.0, (6.0 - min(6.0, rsi_eta_dn)) * 1.0))
            confidence = int(min(90, max(45, (rsi - 62) * 2.6 + vol_5m * 5.5 + eta_bonus)))
            return {
                "action": "OPEN_SHORT",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "tag": f"{RangeFader.name}-sell-fade",
            }

        return None

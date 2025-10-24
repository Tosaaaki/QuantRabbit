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
        ema100 = fac.get("ema100") or fac.get("ma50")
        atr_pips = fac.get("atr_pips")
        vol_5m = fac.get("vol_5m", 1.0) or 0.0
        adx = fac.get("adx", 0.0) or 0.0
        if close is None or ema20 is None or ema50 is None:
            return None

        if atr_pips is None:
            atr = fac.get("atr")
            atr_pips = (atr or 0.0) * 100

        if atr_pips < 3.6 or vol_5m < 1.45:
            return None

        momentum = close - ema20
        bias = ema20 - ema50
        if abs(momentum) < 0.0055:
            return None

        if adx < 24.5:
            return None

        adx_slope = fac.get("adx_slope_per_bar", 0.0) or 0.0
        atr_slope = fac.get("atr_slope_pips", 0.0) or 0.0
        def _build_payload(action: str, slope_bonus: float, tag_suffix: str) -> Dict:
            base_conf = 52.0 + abs(momentum + bias) * 6200 + vol_5m * 5.0 + slope_bonus
            confidence = int(max(50.0, min(92.0, base_conf)))
            tp = max(4.8, min(6.4, atr_pips * 1.9))
            sl = max(3.1, min(tp * 0.7, atr_pips * 1.4))
            return {
                "action": action,
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "tag": f"{PulseBreak.name}-{tag_suffix}",
            }

        if momentum > 0 and bias > 0.06:
            if ema100 is not None and ema20 < ema100:
                return None
            if adx_slope < 0.05:
                return None
            slope_bonus = max(0.0, min(7.0, adx_slope * 35.0 + max(0.0, atr_slope) * 2.0))
            return _build_payload("OPEN_LONG", slope_bonus, "momentum-up")

        if momentum < 0 and bias < -0.06:
            # 2024Q4 環境ではショートの期待値が低いため、厳格な条件が整うまで見送り。
            return None

        return None

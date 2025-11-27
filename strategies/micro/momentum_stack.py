from __future__ import annotations

from typing import Dict

PIP = 0.01


class MicroMomentumStack:
    """Trend継続時に小さな押し目を拾って積み増す戦略。"""

    name = "MicroMomentumStack"
    pocket = "micro"

    _BASE_PULLBACK = 0.45  # pips
    _MAX_PULLBACK = 1.05
    _MAX_SKEW = 0.35
    _MIN_ATR = 1.2

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        price = fac.get("close")
        ma10 = fac.get("ma10")
        ma20 = fac.get("ma20")
        ema20 = fac.get("ema20") or ma20
        if price is None or ma10 is None or ma20 is None or ema20 is None:
            return None
        try:
            price = float(price)
            ma10 = float(ma10)
            ma20 = float(ma20)
            ema20 = float(ema20)
        except (TypeError, ValueError):
            return None

        try:
            gap_dyn = float(fac.get("trend_gap_dynamic") or 0.0)
        except (TypeError, ValueError):
            gap_dyn = 0.0
        try:
            adx_dyn = float(fac.get("trend_adx_dynamic") or 0.0)
        except (TypeError, ValueError):
            adx_dyn = 0.0
        try:
            adx = float(fac.get("adx") or 0.0)
        except (TypeError, ValueError):
            adx = 0.0

        if gap_dyn <= 0.0 or adx_dyn <= 0.0:
            return None

        gap = (ma10 - ma20) / PIP
        direction = None
        if gap >= gap_dyn and adx >= adx_dyn:
            direction = "OPEN_LONG"
        elif gap <= -gap_dyn and adx >= adx_dyn:
            direction = "OPEN_SHORT"
        if direction is None:
            return None

        stack_ready = bool(fac.get("momentum_stack_ready"))
        if not stack_ready:
            return None
        bias = fac.get("momentum_stack_bias")
        if bias in {"long", "short"}:
            desired = "OPEN_LONG" if bias == "long" else "OPEN_SHORT"
            if direction != desired:
                return None

        pullback = (price - ma10) / PIP
        if direction == "OPEN_LONG":
            if pullback > MicroMomentumStack._MAX_SKEW:
                return None
            if pullback < -MicroMomentumStack._MAX_PULLBACK:
                return None
        else:
            if pullback < -MicroMomentumStack._MAX_SKEW:
                return None
            if pullback > MicroMomentumStack._MAX_PULLBACK:
                return None

        ema_gap = (ma10 - ema20) / PIP
        if direction == "OPEN_LONG" and ema_gap < gap_dyn * 0.5:
            return None
        if direction == "OPEN_SHORT" and ema_gap > -gap_dyn * 0.5:
            return None

        atr = fac.get("atr_pips") or (fac.get("atr") or 0.0) * 100.0
        try:
            atr = float(atr or 0.0)
        except (TypeError, ValueError):
            atr = 0.0
        if atr < MicroMomentumStack._MIN_ATR:
            return None
        atr = min(atr, 12.0)

        dist = abs(pullback)
        pullback_req = MicroMomentumStack._BASE_PULLBACK
        if gap_dyn > 0.7:
            pullback_req *= 0.8
        if dist < pullback_req * 0.35:
            return None

        sl_pips = max(0.95, atr * 0.58)
        tp_pips = max(sl_pips * 1.45, sl_pips + atr * 0.85)

        vol = fac.get("vol_5m")
        try:
            vol = float(vol or 0.0)
        except (TypeError, ValueError):
            vol = 0.0

        confidence = 55 + int(min(20.0, abs(gap)) + max(0.0, (adx - adx_dyn) * 0.8))
        if vol > 0.9:
            confidence += 5
        if dist > pullback_req:
            confidence += 4
        confidence = max(48, min(98, confidence))

        tag = f"{MicroMomentumStack.name}-{'long' if direction == 'OPEN_LONG' else 'short'}"
        return {
            "action": direction,
            "sl_pips": round(sl_pips, 2),
            "tp_pips": round(tp_pips, 2),
            "confidence": confidence,
            "profile": "momentum_stack",
            "loss_guard_pips": round(sl_pips, 2),
            "target_tp_pips": round(tp_pips, 2),
            "min_hold_sec": round(tp_pips * 30.0, 1),
            "tag": tag,
        }

from __future__ import annotations

from typing import Dict

PIP = 0.01


class MicroPullbackEMA:
    name = "MicroPullbackEMA"
    pocket = "micro"

    _MIN_GAP_PIPS = 0.45
    _MIN_ADX = 20.0
    _PULLBACK_MIN = 0.35
    _PULLBACK_MAX = 1.25

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
            adx = float(fac.get("adx") or 0.0)
        except (TypeError, ValueError):
            adx = 0.0
        try:
            dyn_gap = float(fac.get("trend_gap_dynamic") or 0.0)
        except (TypeError, ValueError):
            dyn_gap = 0.0
        gap_threshold = dyn_gap if dyn_gap > 0 else MicroPullbackEMA._MIN_GAP_PIPS
        try:
            dyn_adx = float(fac.get("trend_adx_dynamic") or 0.0)
        except (TypeError, ValueError):
            dyn_adx = 0.0
        adx_threshold = dyn_adx if dyn_adx > 0 else MicroPullbackEMA._MIN_ADX

        gap = (ma10 - ma20) / PIP
        direction = None
        if gap >= gap_threshold and adx >= adx_threshold:
            direction = "OPEN_LONG"
        elif gap <= -gap_threshold and adx >= adx_threshold:
            direction = "OPEN_SHORT"
        if direction is None:
            return None

        pullback = (price - ma10) / PIP
        if direction == "OPEN_LONG":
            if not (-MicroPullbackEMA._PULLBACK_MAX <= pullback <= -MicroPullbackEMA._PULLBACK_MIN):
                return None
        else:
            if not (MicroPullbackEMA._PULLBACK_MIN <= pullback <= MicroPullbackEMA._PULLBACK_MAX):
                return None

        atr_hint = fac.get("atr_pips") or (fac.get("atr") or 0.0) * 100.0 or 5.0
        try:
            atr_hint = float(atr_hint)
        except (TypeError, ValueError):
            atr_hint = 5.0
        atr_hint = max(1.2, min(atr_hint, 10.0))
        sl_pips = max(1.0, atr_hint * 0.6)
        tp_pips = max(sl_pips * 1.5, sl_pips + atr_hint * 0.85)

        rsi = fac.get("rsi")
        try:
            rsi = float(rsi) if rsi is not None else 50.0
        except (TypeError, ValueError):
            rsi = 50.0
        confidence = 60 + int(min(14.0, abs(gap)) + min(10.0, max(0.0, adx - MicroPullbackEMA._MIN_ADX)))
        if direction == "OPEN_LONG" and rsi < 45:
            confidence += 4
        if direction == "OPEN_SHORT" and rsi > 55:
            confidence += 4
        confidence = max(48, min(96, confidence))

        return {
            "action": direction,
            "sl_pips": round(sl_pips, 2),
            "tp_pips": round(tp_pips, 2),
            "confidence": confidence,
            "profile": "trend_pullback",
            "loss_guard_pips": round(sl_pips, 2),
            "target_tp_pips": round(tp_pips, 2),
            "min_hold_sec": round(tp_pips * 32.0, 1),
            "tag": f"{MicroPullbackEMA.name}-{'long' if direction == 'OPEN_LONG' else 'short'}",
        }

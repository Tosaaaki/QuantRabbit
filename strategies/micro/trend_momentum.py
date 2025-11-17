from __future__ import annotations

from typing import Dict

PIP = 0.01


class TrendMomentumMicro:
    """Lightweight trend-follow entry for the micro pocket."""

    name = "TrendMomentumMicro"
    pocket = "micro"

    _MIN_GAP_PIPS = 0.45
    _MIN_ADX = 20.0
    _MIN_SLOPE = 0.06
    _MAX_PULLBACK = 1.2

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        price = fac.get("close")
        ma10 = fac.get("ma10")
        ma20 = fac.get("ma20")
        adx_raw = fac.get("adx", 0.0)
        ema20 = fac.get("ema20") or ma20
        if price is None or ma10 is None or ma20 is None or ema20 is None:
            return None
        try:
            adx = float(adx_raw)
        except (TypeError, ValueError):
            adx = 0.0
        try:
            dyn_gap = float(fac.get("trend_gap_dynamic") or 0.0)
        except (TypeError, ValueError):
            dyn_gap = 0.0
        gap_threshold = dyn_gap if dyn_gap > 0 else TrendMomentumMicro._MIN_GAP_PIPS
        try:
            dyn_adx = float(fac.get("trend_adx_dynamic") or 0.0)
        except (TypeError, ValueError):
            dyn_adx = 0.0
        adx_threshold = dyn_adx if dyn_adx > 0 else TrendMomentumMicro._MIN_ADX

        diff = (ma10 - ma20) / PIP
        direction = None
        if diff >= gap_threshold and adx >= adx_threshold:
            direction = "OPEN_LONG"
        elif diff <= -gap_threshold and adx >= adx_threshold:
            direction = "OPEN_SHORT"
        if direction is None:
            return None

        # ensure slope agrees with direction
        ema_gap = (ma10 - ema20) / PIP
        if direction == "OPEN_LONG" and ema_gap < TrendMomentumMicro._MIN_SLOPE:
            return None
        if direction == "OPEN_SHORT" and ema_gap > -TrendMomentumMicro._MIN_SLOPE:
            return None

        pullback = (price - ma10) / PIP
        if direction == "OPEN_LONG" and pullback < -TrendMomentumMicro._MAX_PULLBACK:
            return None
        if direction == "OPEN_SHORT" and pullback > TrendMomentumMicro._MAX_PULLBACK:
            return None

        atr = fac.get("atr_pips")
        try:
            atr_pips = float(atr) if atr is not None else None
        except (TypeError, ValueError):
            atr_pips = None
        if atr_pips is None:
            atr_raw = fac.get("atr")
            try:
                atr_pips = float(atr_raw or 0.0) * 100.0
            except (TypeError, ValueError):
                atr_pips = 6.0
        atr_pips = max(1.6, min(atr_pips, 12.0))

        sl_pips = round(max(1.2, atr_pips * 0.75), 2)
        tp_pips = round(max(sl_pips * 1.45, sl_pips + atr_pips * 0.9), 2)
        confidence = 50 + int(min(18.0, abs(diff)) + max(0.0, (adx - TrendMomentumMicro._MIN_ADX) * 0.7))
        confidence = max(45, min(95, confidence))

        return {
            "action": direction,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "confidence": confidence,
            "profile": "trend_momentum_micro",
            "loss_guard_pips": sl_pips,
            "target_tp_pips": tp_pips,
            "min_hold_sec": round(tp_pips * 35.0, 1),
            "tag": f"{TrendMomentumMicro.name}-{'long' if direction == 'OPEN_LONG' else 'short'}",
        }

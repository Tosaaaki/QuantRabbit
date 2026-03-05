from __future__ import annotations

from typing import Dict

PIP = 0.01


class MicroPullbackEMA:
    name = "MicroPullbackEMA"
    pocket = "micro"

    _MIN_GAP_PIPS = 0.45
    _MIN_ADX = 20.0
    _GAP_ATR_RATIO_MIN = 0.18
    _PULLBACK_MIN = 0.35
    _PULLBACK_MAX = 1.25
    _PULLBACK_ATR_MIN_RATIO = 0.12
    _PULLBACK_ATR_MAX_RATIO = 0.55
    _PULLBACK_OVER_GAP_BUFFER_MIN_PIPS = 0.10
    _PULLBACK_OVER_GAP_BUFFER_ATR_RATIO = 0.08
    _RANGE_BIAS_SCORE = 0.30
    _RANGE_BIAS_ADX_BONUS = 3.0
    _SPREAD_PIPS_MAX = 1.2
    _SPREAD_ATR_RATIO_MAX = 0.30

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        price = fac.get("close")
        ma10 = fac.get("ma10")
        ma20 = fac.get("ma20")
        ema20 = fac.get("ema20") or ma20
        if price is None or ma10 is None or ma20 is None or ema20 is None:
            return None

        # Spread filter: skip when spread is wide relative to ATR.
        try:
            spread_pips = float(fac.get("spread_pips") or 0.0)
        except (TypeError, ValueError):
            spread_pips = 0.0
        try:
            atr_pips = float(fac.get("atr_pips") or 0.0)
        except (TypeError, ValueError):
            atr_pips = 0.0
        if atr_pips <= 0.0:
            try:
                atr_pips = float(fac.get("atr") or 0.0) * 100.0
            except (TypeError, ValueError):
                atr_pips = 0.0
        if spread_pips > 0 and atr_pips > 0:
            spread_cap = max(MicroPullbackEMA._SPREAD_PIPS_MAX, atr_pips * MicroPullbackEMA._SPREAD_ATR_RATIO_MAX)
            if spread_pips > spread_cap:
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
        try:
            range_score = float(fac.get("range_score") or 0.0)
        except (TypeError, ValueError):
            range_score = 0.0
        if range_score >= MicroPullbackEMA._RANGE_BIAS_SCORE:
            adx_threshold = max(adx_threshold, MicroPullbackEMA._MIN_ADX + MicroPullbackEMA._RANGE_BIAS_ADX_BONUS)

        gap = (ma10 - ma20) / PIP
        direction = None
        if atr_pips > 0.0:
            gap_threshold = max(gap_threshold, atr_pips * MicroPullbackEMA._GAP_ATR_RATIO_MIN)
        if gap >= gap_threshold and adx >= adx_threshold:
            direction = "OPEN_LONG"
        elif gap <= -gap_threshold and adx >= adx_threshold:
            direction = "OPEN_SHORT"
        if direction is None:
            return None

        plus_di = fac.get("plus_di")
        minus_di = fac.get("minus_di")
        try:
            plus_di = float(plus_di) if plus_di is not None else None
        except (TypeError, ValueError):
            plus_di = None
        try:
            minus_di = float(minus_di) if minus_di is not None else None
        except (TypeError, ValueError):
            minus_di = None
        if plus_di is not None and minus_di is not None:
            if direction == "OPEN_LONG" and plus_di <= minus_di:
                return None
            if direction == "OPEN_SHORT" and minus_di <= plus_di:
                return None

        pullback = (price - ma10) / PIP
        pull_min = MicroPullbackEMA._PULLBACK_MIN
        pull_max = MicroPullbackEMA._PULLBACK_MAX
        if atr_pips > 0.0:
            pull_min = max(pull_min, atr_pips * MicroPullbackEMA._PULLBACK_ATR_MIN_RATIO)
            pull_max = min(pull_max, atr_pips * MicroPullbackEMA._PULLBACK_ATR_MAX_RATIO)
            if pull_max < pull_min:
                pull_max = pull_min
        if direction == "OPEN_LONG":
            if not (-pull_max <= pullback <= -pull_min):
                return None
        else:
            if not (pull_min <= pullback <= pull_max):
                return None

        buffer_pips = MicroPullbackEMA._PULLBACK_OVER_GAP_BUFFER_MIN_PIPS
        if atr_pips > 0.0:
            buffer_pips = max(buffer_pips, atr_pips * MicroPullbackEMA._PULLBACK_OVER_GAP_BUFFER_ATR_RATIO)
        if abs(pullback) > abs(gap) + buffer_pips:
            return None

        atr_hint = atr_pips or 5.0
        atr_hint = max(1.2, min(atr_hint, 10.0))
        # Previous: sl = max(1.0, atr * 0.6) was ~1.2-2.4p -- too tight for pullback entries
        # that need room for the pullback to complete before trending.
        # New: ATR-based with higher floor and multiplier.
        sl_pips = max(2.5, atr_hint * 1.05)
        tp_pips = max(sl_pips * 1.6, sl_pips + atr_hint * 0.95)

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

from __future__ import annotations

from typing import Dict
import os

PIP = 0.01
_ENV_PREFIX = os.getenv("MICRO_MULTI_ENV_PREFIX", "")


def _to_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: str, default: bool) -> bool:
    raw = str(value).strip().lower() if value is not None else ""
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


class MicroRangeBreak:
    name = "MicroRangeBreak"
    pocket = "micro"

    _MIN_RANGE_SCORE = _to_float(os.getenv("MICRO_RANGEBREAK_MIN_RANGE_SCORE", "0.45"), 0.45)
    _MAX_ADX = _to_float(os.getenv("MICRO_RANGEBREAK_REVERSION_MAX_ADX", "23.0"), 23.0)
    _ENTRY_RATIO = _to_float(os.getenv("MICRO_RANGEBREAK_ENTRY_RATIO", "0.28"), 0.28)  # distance to band edge
    _MAX_DISTANCE_RATIO = 0.65

    _BREAKOUT_ENABLED = _to_bool(os.getenv("MICRO_RANGEBREAK_BREAKOUT_ENABLED", "1"), True)
    _BREAKOUT_MIN_ADX = _to_float(os.getenv("MICRO_RANGEBREAK_BREAKOUT_MIN_ADX", "20.0"), 20.0)
    _BREAKOUT_MIN_RANGE_SCORE = _to_float(os.getenv("MICRO_RANGEBREAK_BREAKOUT_MIN_RANGE_SCORE", "0.46"), 0.46)
    _BREAKOUT_BAND_RATIO = _to_float(os.getenv("MICRO_RANGEBREAK_BREAKOUT_BAND_RATIO", "0.22"), 0.22)
    _BREAKOUT_MIN_ATR = _to_float(os.getenv("MICRO_RANGEBREAK_BREAKOUT_MIN_ATR", "1.2"), 1.2)

    # Spread filter: skip entry when spread eats too much of expected profit
    _MAX_SPREAD_PIPS = _to_float(os.getenv("MICRO_RANGEBREAK_MAX_SPREAD_PIPS", "1.0"), 1.0)
    _MAX_SPREAD_ATR_RATIO = _to_float(os.getenv("MICRO_RANGEBREAK_MAX_SPREAD_ATR_RATIO", "0.30"), 0.30)

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        range_score = fac.get("range_score") or 0.0
        try:
            range_score = float(range_score)
        except (TypeError, ValueError):
            range_score = 0.0
        try:
            adx = float(fac.get("adx") or 0.0)
        except (TypeError, ValueError):
            adx = 0.0
        ma = fac.get("ma20")
        bbw = fac.get("bbw")
        price = fac.get("close", ma)
        if ma is None or bbw is None or price is None:
            return None
        try:
            ma = float(ma)
            bbw = float(bbw)
            price = float(price)
        except (TypeError, ValueError):
            return None

        half_band = (ma * bbw) / 2.0
        if half_band <= 0:
            return None
        upper = ma + half_band
        lower = ma - half_band
        band_width = upper - lower
        if band_width <= 0:
            return None
        breakout_margin = half_band * MicroRangeBreak._BREAKOUT_BAND_RATIO
        breakout_long = price >= upper + breakout_margin
        breakout_short = price <= lower - breakout_margin

        # Spread filter for breakout entries
        try:
            _spread_pips = float(fac.get("spread_pips") or 0.0)
        except (TypeError, ValueError):
            _spread_pips = 0.0

        if MicroRangeBreak._BREAKOUT_ENABLED and (
            range_score >= MicroRangeBreak._BREAKOUT_MIN_RANGE_SCORE
        ) and (adx >= MicroRangeBreak._BREAKOUT_MIN_ADX):
            atr_hint = fac.get("atr_pips") or (fac.get("atr") or 0.0) * 100.0 or 0.0
            try:
                atr_hint = float(atr_hint)
            except (TypeError, ValueError):
                atr_hint = 0.0
            atr_hint = max(MicroRangeBreak._BREAKOUT_MIN_ATR, min(atr_hint, 12.0))
            # Skip breakout if spread is too wide
            if _spread_pips > 0:
                breakout_spread_cap = max(MicroRangeBreak._MAX_SPREAD_PIPS, atr_hint * MicroRangeBreak._MAX_SPREAD_ATR_RATIO)
                if _spread_pips > breakout_spread_cap:
                    return None
            if breakout_long:
                sl_pips = max(atr_hint * 0.82, 1.5)
                tp_pips = max(sl_pips * 1.8, sl_pips + atr_hint * 1.05)
                confidence = int(52 + range_score * 28 + min(6.0, (adx - MicroRangeBreak._BREAKOUT_MIN_ADX) * 0.9))
                confidence = max(40, min(94, confidence))
                return {
                    "action": "OPEN_LONG",
                    "sl_pips": round(sl_pips, 2),
                    "tp_pips": round(tp_pips, 2),
                    "confidence": confidence,
                    "profile": "range_breakout",
                    "loss_guard_pips": round(sl_pips, 2),
                    "target_tp_pips": round(tp_pips, 2),
                    "min_hold_sec": round(tp_pips * 32.0, 1),
                    "tag": f"{MicroRangeBreak.name}-breakout-long",
                    "signal_mode": "trend",
                }
            if breakout_short:
                sl_pips = max(atr_hint * 0.82, 1.5)
                tp_pips = max(sl_pips * 1.8, sl_pips + atr_hint * 1.05)
                confidence = int(52 + range_score * 28 + min(6.0, (adx - MicroRangeBreak._BREAKOUT_MIN_ADX) * 0.9))
                confidence = max(40, min(94, confidence))
                return {
                    "action": "OPEN_SHORT",
                    "sl_pips": round(sl_pips, 2),
                    "tp_pips": round(tp_pips, 2),
                    "confidence": confidence,
                    "profile": "range_breakout",
                    "loss_guard_pips": round(sl_pips, 2),
                    "target_tp_pips": round(tp_pips, 2),
                    "min_hold_sec": round(tp_pips * 32.0, 1),
                    "tag": f"{MicroRangeBreak.name}-breakout-short",
                    "signal_mode": "trend",
                }

        if range_score < MicroRangeBreak._MIN_RANGE_SCORE or adx > MicroRangeBreak._MAX_ADX:
            return None

        # Spread filter: reject entry when spread is wide relative to ATR
        spread_pips = 0.0
        try:
            spread_pips = float(fac.get("spread_pips") or 0.0)
        except (TypeError, ValueError):
            spread_pips = 0.0
        atr_hint = fac.get("atr_pips") or (fac.get("atr") or 0.0) * 100.0 or 4.0
        try:
            atr_hint = float(atr_hint)
        except (TypeError, ValueError):
            atr_hint = 4.0
        atr_hint = max(1.0, min(atr_hint, 8.0))
        if spread_pips > 0:
            spread_cap = max(MicroRangeBreak._MAX_SPREAD_PIPS, atr_hint * MicroRangeBreak._MAX_SPREAD_ATR_RATIO)
            if spread_pips > spread_cap:
                return None

        distance_top = (upper - price) / band_width
        distance_bottom = (price - lower) / band_width
        action = None
        distance_ratio = 0.0
        if distance_bottom <= MicroRangeBreak._ENTRY_RATIO:
            action = "OPEN_LONG"
            distance_ratio = distance_bottom
        elif distance_top <= MicroRangeBreak._ENTRY_RATIO:
            action = "OPEN_SHORT"
            distance_ratio = distance_top
        else:
            return None
        if distance_ratio > MicroRangeBreak._MAX_DISTANCE_RATIO:
            return None

        # SL widened: ATR-based with higher floor to survive normal noise.
        # Previous: sl = max(0.8, atr * 0.65) which was ~1.3-2.6p -- too tight.
        # New: sl = max(2.2, atr * 1.05) giving ~2.2-4.2p -- room to breathe.
        sl_pips = max(2.2, atr_hint * 1.05)
        tp_pips = max(sl_pips * 1.5, sl_pips + atr_hint * 0.9)
        if action == "OPEN_SHORT":
            sl_pips = max(2.4, atr_hint * 1.10)
            tp_pips = max(sl_pips * 1.5, sl_pips + atr_hint * 0.95)

        confidence = int(55 + range_score * 35 - distance_ratio * 20)
        confidence = max(40, min(94, confidence))

        return {
            "action": action,
            "sl_pips": round(sl_pips, 2),
            "tp_pips": round(tp_pips, 2),
            "confidence": confidence,
            "profile": "range_scalp",
            "loss_guard_pips": round(sl_pips, 2),
            "target_tp_pips": round(tp_pips, 2),
            "min_hold_sec": round(tp_pips * 28.0, 1),
            "tag": f"{MicroRangeBreak.name}-{'long' if action == 'OPEN_LONG' else 'short'}",
            "signal_mode": "reversion",
        }

from __future__ import annotations

from typing import Dict, Optional

from .common import (
    atr_pips,
    candle_body_pips,
    clamp,
    latest_candles,
    price_delta_pips,
    to_float,
    typical_price,
)
from utils.tuning_loader import get_tuning_value


def _tuned_float(keys: tuple[str, ...], default: float) -> float:
    raw = get_tuning_value(keys)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


class MicroVWAPRevert:
    name = "MicroVWAPRevert"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = to_float(fac.get("close"))
        ema20 = to_float(fac.get("ema20"))
        ma10 = to_float(fac.get("ma10"), ema20)
        adx = to_float(fac.get("adx"))
        vol_5m = to_float(fac.get("vol_5m"))
        bbw = to_float(fac.get("bbw"))

        if None in (close, ema20, ma10):
            return None
        if adx is not None and adx > 24.0:
            return None
        if vol_5m is not None and vol_5m > 1.1:
            return None

        atr = atr_pips(fac)
        if atr <= 0.5 or atr > 2.8:
            return None
        if bbw is not None and bbw > 0.30:
            return None

        candles = latest_candles(fac, 9)
        if len(candles) < 4:
            return None

        typicals = [typical_price(c) for c in candles[-7:]]
        typicals = [t for t in typicals if t is not None]
        if not typicals:
            return None
        vwap = sum(typicals) / len(typicals)
        deviation = price_delta_pips(close, vwap)

        drift = price_delta_pips(ema20, ma10)
        body_bias_raw = candle_body_pips(candles[-1]) if candles else 0.0
        body_bias = float(body_bias_raw or 0.0)

        threshold = max(1.05, atr * 0.55)
        tuned_min = get_tuning_value(("strategies", "MicroVWAPRevert", "vwap_z_min"))
        if tuned_min is not None:
            try:
                threshold = max(threshold, float(tuned_min))
            except (TypeError, ValueError):
                pass
        if abs(deviation) < threshold:
            return None

        prev_close = to_float(candles[-2].get("close")) if len(candles) >= 2 else None
        if prev_close is None:
            return None
        prev_deviation = price_delta_pips(prev_close, vwap)
        retrace_min = max(
            0.0,
            _tuned_float(("strategies", "MicroVWAPRevert", "retrace_min_pips"), 0.18),
        )
        extension_mult = max(
            1.0,
            _tuned_float(("strategies", "MicroVWAPRevert", "extension_mult"), 1.08),
        )
        max_counter_drift = max(
            0.4,
            _tuned_float(("strategies", "MicroVWAPRevert", "max_counter_drift_pips"), 1.45),
        )
        confirm_body_min = max(
            0.0,
            _tuned_float(("strategies", "MicroVWAPRevert", "confirm_body_min_pips"), 0.08),
        )

        retrace_from_prev = 0.0
        if deviation <= 0.0 and prev_deviation <= 0.0:
            retrace_from_prev = deviation - prev_deviation
        elif deviation >= 0.0 and prev_deviation >= 0.0:
            retrace_from_prev = prev_deviation - deviation

        direction: Optional[str] = None
        if deviation <= -threshold:
            if prev_deviation > -(threshold * extension_mult):
                return None
            if retrace_from_prev < retrace_min:
                return None
            if body_bias < confirm_body_min:
                return None
            if drift > max_counter_drift:
                return None
            direction = "OPEN_LONG"
        elif deviation >= threshold:
            if prev_deviation < threshold * extension_mult:
                return None
            if retrace_from_prev < retrace_min:
                return None
            if body_bias > -confirm_body_min:
                return None
            if drift < -max_counter_drift:
                return None
            direction = "OPEN_SHORT"
        else:
            return None

        conf_base = 50.0
        conf_base += clamp(abs(deviation) - threshold, 0.0, 3.0) * 2.6
        conf_base += clamp(retrace_from_prev, 0.0, 2.0) * 2.0
        conf_base += clamp(max(0.0, 1.3 - (vol_5m or 1.3)), 0.0, 1.1) * 4.5
        conf_base += clamp(max(0.0, 0.9 - abs(drift)), 0.0, 0.9) * 3.2
        conf_base -= clamp(max(0.0, abs(body_bias) - 0.8), 0.0, 2.0) * 2.4
        confidence = int(clamp(conf_base, 44.0, 84.0))

        sl = clamp(atr * 1.18, 1.2, 2.4)
        tp = clamp(sl * 0.9, 0.9, 2.2)

        tag_suffix = "long" if direction == "OPEN_LONG" else "short"
        return {
            "action": direction,
            "sl_pips": round(sl, 2),
            "tp_pips": round(tp, 2),
            "confidence": confidence,
            "tag": f"{MicroVWAPRevert.name}-{tag_suffix}",
            "notes": {
                "deviation": round(deviation, 2),
                "threshold": round(threshold, 2),
                "vwap": round(vwap, 3),
                "atr": round(atr, 2),
                "vol5m": round(vol_5m or 0.0, 2),
                "drift": round(drift, 2),
                "prev_deviation": round(prev_deviation, 2),
                "retrace": round(retrace_from_prev, 2),
            },
        }

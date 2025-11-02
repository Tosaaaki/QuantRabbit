from __future__ import annotations

from typing import Dict, Optional

from .common import (
    PIP,
    atr_pips,
    candle_close,
    clamp,
    latest_candles,
    price_delta_pips,
    to_float,
)


class MomentumPulse:
    name = "MomentumPulse"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = to_float(fac.get("close"))
        ema12 = to_float(fac.get("ema12"))
        ema20 = to_float(fac.get("ema20"))
        ma10 = to_float(fac.get("ma10"))
        adx = to_float(fac.get("adx"))
        vol_5m = to_float(fac.get("vol_5m"))
        bbw = to_float(fac.get("bbw"))

        if None in (close, ema12, ema20, ma10):
            return None

        atr = atr_pips(fac)
        if atr <= 0.6 or atr >= 4.8:
            return None
        if vol_5m is not None and vol_5m > 1.25:
            return None
        if bbw is not None and bbw > 0.30:
            return None
        if adx is not None and adx > 28.0:
            return None

        mom_pips = price_delta_pips(ema12, ema20)
        slope_ma = price_delta_pips(ma10, ema20)
        bias = price_delta_pips(close, ema12)

        candles = latest_candles(fac, 4)
        last_close = close
        prev_close: Optional[float] = None
        prev_prev_close: Optional[float] = None

        if candles:
            last_close_c = candle_close(candles[-1])
            if last_close_c is not None:
                last_close = last_close_c
            if len(candles) >= 2:
                prev_close = candle_close(candles[-2])
            if len(candles) >= 3:
                prev_prev_close = candle_close(candles[-3])

        direction: Optional[str] = None
        if mom_pips >= 0.35 and bias >= -0.2:
            higher_lows = prev_close is None or last_close >= (prev_close or last_close)
            if prev_prev_close is not None:
                higher_lows = higher_lows and last_close >= prev_prev_close
            if higher_lows:
                direction = "OPEN_LONG"
        elif mom_pips <= -0.35 and bias <= 0.2:
            lower_highs = prev_close is None or last_close <= (prev_close or last_close)
            if prev_prev_close is not None:
                lower_highs = lower_highs and last_close <= prev_prev_close
            if lower_highs:
                direction = "OPEN_SHORT"

        if not direction:
            return None

        vol_term = max(0.0, min(1.0, (1.15 - (vol_5m or 1.15)) * 0.8))
        slope_term = clamp(abs(slope_ma), 0.0, 3.0) * 1.6
        bias_penalty = max(0.0, abs(bias) - 1.2) * 3.0
        base_conf = 52.0 + clamp(abs(mom_pips), 0.0, 2.4) * 3.2 + vol_term * 14.0 + slope_term
        conf = int(clamp(base_conf - bias_penalty, 44.0, 88.0))

        sl = clamp(atr * 1.05, 1.3, 2.7)
        tp = clamp(sl * 0.92, 0.9, 2.4)

        notes = {
            "mom_pips": round(mom_pips, 2),
            "bias": round(bias, 2),
            "atr": round(atr, 2),
            "vol5m": round(vol_5m or 0.0, 2),
            "slope": round(slope_ma, 2),
        }
        tag_suffix = "long" if direction == "OPEN_LONG" else "short"
        return {
            "action": direction,
            "sl_pips": round(sl, 2),
            "tp_pips": round(tp, 2),
            "confidence": conf,
            "tag": f"{MomentumPulse.name}-{tag_suffix}",
            "notes": notes,
        }

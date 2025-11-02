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
        body_bias = candle_body_pips(candles[-1]) if candles else 0.0

        threshold = max(1.05, atr * 0.55)
        if abs(deviation) < threshold:
            return None

        direction: Optional[str] = None
        if deviation <= -threshold:
            direction = "OPEN_LONG"
        elif deviation >= threshold:
            direction = "OPEN_SHORT"
        else:
            return None

        conf_base = 50.0
        conf_base += clamp(abs(deviation) - threshold, 0.0, 3.0) * 2.6
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
                "atr": round(atr, 2),
                "vol5m": round(vol_5m or 0.0, 2),
                "drift": round(drift, 2),
            },
        }

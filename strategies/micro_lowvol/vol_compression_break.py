from __future__ import annotations

from typing import Dict, Optional

from .common import (
    atr_pips,
    candle_close,
    candle_high,
    candle_low,
    clamp,
    latest_candles,
    price_delta_pips,
    to_float,
)


class VolCompressionBreak:
    name = "VolCompressionBreak"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = to_float(fac.get("close"))
        ema20 = to_float(fac.get("ema20"))
        ema12 = to_float(fac.get("ema12"))
        bbw = to_float(fac.get("bbw"))
        adx = to_float(fac.get("adx"))
        vol_5m = to_float(fac.get("vol_5m"))

        if None in (close, ema20, ema12):
            return None

        atr = atr_pips(fac)
        if atr <= 0.55 or atr >= 4.2:
            return None
        if bbw is not None and bbw > 0.22:
            return None
        if adx is not None and adx > 24.0:
            return None

        candles = latest_candles(fac, 6)
        if len(candles) < 4:
            return None

        highs = [candle_high(c) for c in candles[:-1]]
        lows = [candle_low(c) for c in candles[:-1]]
        last_candle = candles[-1]
        last_close = candle_close(last_candle)
        prev_close = candle_close(candles[-2]) if len(candles) >= 2 else None
        if last_close is None:
            last_close = close
        if prev_close is None:
            prev_close = last_close

        highs = [h for h in highs if h is not None]
        lows = [l for l in lows if l is not None]
        if not highs or not lows:
            return None

        range_high = max(highs)
        range_low = min(lows)
        range_span = (range_high - range_low) / 0.01
        compression = clamp(max(0.0, 6.0 - range_span), 0.0, 6.0)

        breakout_up = last_close > range_high + 0.0006
        breakout_dn = last_close < range_low - 0.0006

        move_pips = price_delta_pips(last_close, prev_close)
        bias_ema = price_delta_pips(last_close, ema12)

        direction: Optional[str] = None
        if breakout_up and move_pips >= 0.25:
            direction = "OPEN_LONG"
        elif breakout_dn and move_pips <= -0.25:
            direction = "OPEN_SHORT"
        else:
            return None

        vol_term = max(0.0, min(1.0, (1.05 - (vol_5m or 1.05)) * 1.2))
        bias_penalty = max(0.0, abs(bias_ema) - 1.8) * 1.5
        conf_base = 58.0 + compression * 2.4 + abs(move_pips) * 2.8 + vol_term * 10.0
        confidence = int(clamp(conf_base - bias_penalty, 46.0, 92.0))

        sl = clamp(max(1.2, atr * 1.05), 1.2, 2.35)
        tp = clamp(sl * 1.05, 1.1, 2.6)

        tag_suffix = "long" if direction == "OPEN_LONG" else "short"
        notes = {
            "range_span": round(range_span, 2),
            "compression": round(compression, 2),
            "move": round(move_pips, 2),
            "atr": round(atr, 2),
            "vol5m": round(vol_5m or 0.0, 2),
        }
        return {
            "action": direction,
            "sl_pips": round(sl, 2),
            "tp_pips": round(tp, 2),
            "confidence": confidence,
            "tag": f"{VolCompressionBreak.name}-{tag_suffix}",
            "notes": notes,
        }

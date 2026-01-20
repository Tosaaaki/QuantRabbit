from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .common import (
    PIP,
    atr_pips,
    candle_body_pips,
    clamp,
    latest_candles,
    to_float,
)


def _band_edges(ma: Optional[float], bbw: Optional[float]) -> Optional[Tuple[float, float, float]]:
    if ma is None or bbw is None:
        return None
    try:
        basis = float(ma)
        width = float(bbw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(basis) or not math.isfinite(width):
        return None
    half = abs(basis) * width / 2.0
    lower = basis - half
    upper = basis + half
    span = upper - lower
    if not math.isfinite(span) or span <= 0.0:
        return None
    return lower, basis, upper


MIN_RANGE_SCORE = 0.30


class BBRsiFast:
    name = "BB_RSI_Fast"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        rsi = to_float(fac.get("rsi"))
        bbw = to_float(fac.get("bbw"))
        ma20 = to_float(fac.get("ma20"))
        ema20 = to_float(fac.get("ema20"), ma20)
        close = to_float(fac.get("close"), ema20)
        adx = to_float(fac.get("adx"))
        vol_5m = to_float(fac.get("vol_5m"))

        if None in (rsi, bbw, ma20, close):
            return None
        if "range_score" in fac or "range_active" in fac:
            range_score = to_float(fac.get("range_score"), 0.0) or 0.0
            range_score = clamp(range_score, 0.0, 1.0)
            range_active = bool(fac.get("range_active"))
            if not range_active and range_score < MIN_RANGE_SCORE:
                return None
        if bbw is not None and bbw > 0.28:
            return None
        if adx is not None and adx > 25.0:
            return None

        atr = atr_pips(fac)
        if atr <= 0.65 or atr > 3.1:
            return None
        if vol_5m is not None and vol_5m > 1.25:
            return None

        band = _band_edges(ma20, bbw)
        if band is None:
            return None
        lower, basis, upper = band
        span_pips = (upper - lower) / PIP

        dist_lower = (close - lower) / PIP
        dist_upper = (upper - close) / PIP

        candles = latest_candles(fac, 3)
        last_body = candle_body_pips(candles[-1]) if len(candles) >= 1 else 0.0
        prev_body = candle_body_pips(candles[-2]) if len(candles) >= 2 else 0.0

        long_ready = rsi <= 39.0 and dist_lower <= max(2.1, span_pips * 0.22)
        short_ready = rsi >= 61.0 and dist_upper <= max(2.1, span_pips * 0.22)

        if long_ready and prev_body is not None and last_body is not None:
            if prev_body is not None and prev_body >= 0.6 and (last_body or 0.0) >= 0.2:
                long_ready = False
        if short_ready and prev_body is not None and last_body is not None:
            if prev_body is not None and prev_body <= -0.6 and (last_body or 0.0) <= -0.2:
                short_ready = False

        if not long_ready and not short_ready:
            return None

        def _confidence_base() -> float:
            conf = 48.0
            conf += clamp(abs(span_pips) - 3.0, 0.0, 6.0) * 0.9
            conf += clamp(max(0.0, 2.2 - atr), 0.0, 1.6) * 3.5
            conf += clamp(max(0.0, 1.4 - (vol_5m or 1.4)), 0.0, 1.2) * 4.0
            return conf

        sl = clamp(atr * 1.25, 1.3, 2.05)
        tp = clamp(sl * 1.12, 1.0, 2.3)

        if long_ready:
            conf = _confidence_base()
            conf += clamp(max(0.0, 42.0 - rsi), 0.0, 12.0) * 0.6
            conf -= clamp(max(0.0, dist_lower - 2.4), 0.0, 3.0) * 1.4
            confidence = int(clamp(conf, 42.0, 86.0))
            return {
                "action": "OPEN_LONG",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "tag": f"{BBRsiFast.name}-long",
                "notes": {
                    "rsi": round(rsi, 2),
                    "bbw": round(bbw or 0.0, 3),
                    "dist_lower": round(dist_lower, 2),
                    "atr": round(atr, 2),
                },
            }

        conf = _confidence_base()
        conf += clamp(max(0.0, rsi - 58.0), 0.0, 12.0) * 0.6
        conf -= clamp(max(0.0, dist_upper - 2.4), 0.0, 3.0) * 1.4
        confidence = int(clamp(conf, 42.0, 86.0))
        return {
            "action": "OPEN_SHORT",
            "sl_pips": round(sl, 2),
            "tp_pips": round(tp, 2),
            "confidence": confidence,
            "tag": f"{BBRsiFast.name}-short",
            "notes": {
                "rsi": round(rsi, 2),
                "bbw": round(bbw or 0.0, 3),
                "dist_upper": round(dist_upper, 2),
                "atr": round(atr, 2),
            },
        }

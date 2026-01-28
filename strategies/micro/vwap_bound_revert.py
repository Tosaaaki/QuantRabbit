from __future__ import annotations

import statistics
from typing import Dict, Optional, Sequence

from strategies.micro_lowvol.common import (
    clamp,
    candle_close,
    latest_candles,
    price_delta_pips,
    to_float,
    typical_price,
)


class MicroVWAPBound:
    """Range-mode VWAP bound reversion for the micro pocket."""

    name = "MicroVWAPBound"
    pocket = "micro"

    VWAP_WINDOW = 20
    MIN_Z = 1.7
    MAX_ADX = 14.0
    MAX_BBW = 0.28
    MIN_RANGE_SCORE = 0.72
    MAX_ATR_PIPS = 3.2
    TREND_GAP_PIPS = 1.8

    @staticmethod
    def _vwap_and_sigma(candles: Sequence[Dict[str, object]]) -> tuple[Optional[float], Optional[float]]:
        closes = []
        for candle in candles[-MicroVWAPBound.VWAP_WINDOW :]:
            tp = typical_price(candle)
            if tp is None:
                continue
            closes.append(tp)
        if len(closes) < 8:
            return None, None
        vwap = sum(closes) / len(closes)
        sigma = statistics.pstdev(closes)
        if sigma <= 0:
            return vwap, None
        return vwap, sigma

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = to_float(fac.get("close"))
        adx = to_float(fac.get("adx"), 0.0)
        bbw = to_float(fac.get("bbw"), 0.0) or 0.0
        range_score = to_float(fac.get("range_score"), 0.0) or 0.0
        atr_pips = to_float(fac.get("atr_pips"), 0.0) or 0.0
        if close is None:
            return None
        if not fac.get("range_active"):
            return None
        if range_score < MicroVWAPBound.MIN_RANGE_SCORE:
            return None
        if adx is None or adx >= MicroVWAPBound.MAX_ADX:
            return None
        if bbw > MicroVWAPBound.MAX_BBW:
            return None
        if MicroVWAPBound.MAX_ATR_PIPS > 0 and atr_pips > MicroVWAPBound.MAX_ATR_PIPS:
            return None

        candles = latest_candles(fac, MicroVWAPBound.VWAP_WINDOW + 4)
        vwap, sigma = MicroVWAPBound._vwap_and_sigma(candles)
        if vwap is None or sigma is None or sigma <= 0:
            return None

        z_score = (close - vwap) / sigma
        if abs(z_score) < MicroVWAPBound.MIN_Z:
            return None

        ma10 = to_float(fac.get("ma10"))
        ma20 = to_float(fac.get("ma20"))
        if ma10 is not None and ma20 is not None:
            gap_pips = abs(price_delta_pips(ma10, ma20))
            if gap_pips >= MicroVWAPBound.TREND_GAP_PIPS:
                if z_score > 0 and ma10 > ma20:
                    return None
                if z_score < 0 and ma10 < ma20:
                    return None

        candles = latest_candles(fac, 3)
        if len(candles) >= 2:
            prev_close = candle_close(candles[-2])
            last_close = candle_close(candles[-1])
            if prev_close is not None and last_close is not None:
                if z_score > 0 and last_close >= prev_close:
                    return None
                if z_score < 0 and last_close <= prev_close:
                    return None

        direction = "OPEN_SHORT" if z_score > 0 else "OPEN_LONG"
        deviation_pips = abs(price_delta_pips(close, vwap))

        # SL/TP aligned to spec (6-8p sl, 8-10p tp with ~1.2 RR)
        sl_pips = clamp(max(6.0, deviation_pips * 0.9), 6.0, 8.0)
        tp_pips = clamp(sl_pips * 1.22, 8.0, 10.0)

        z_bonus = clamp(abs(z_score) - MicroVWAPBound.MIN_Z, 0.0, 3.0) * 10.0
        bbw_bonus = clamp(MicroVWAPBound.MAX_BBW - bbw, 0.0, 0.25) * 70.0
        confidence = clamp(52.0 + z_bonus + bbw_bonus + clamp(range_score, 0.0, 1.0) * 6.0, 48.0, 90.0)

        tag_suffix = "short" if direction == "OPEN_SHORT" else "long"
        return {
            "action": direction,
            "sl_pips": round(sl_pips, 2),
            "tp_pips": round(tp_pips, 2),
            "confidence": int(confidence),
            "tag": f"{MicroVWAPBound.name}-{tag_suffix}",
            "profile": "micro_vwap_bound",
            "loss_guard_pips": sl_pips,
            "target_tp_pips": tp_pips,
            "notes": {
                "z": round(z_score, 2),
                "bbw": round(bbw, 3),
                "vwap": round(vwap, 3),
            },
        }

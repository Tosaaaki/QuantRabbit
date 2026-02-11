from __future__ import annotations

from typing import Dict, Optional

from .common import PIP, atr_pips, to_float


class MicroCompressionRevert:
    """Range compression fade strategy (micro pocket)."""

    name = "MicroCompressionRevert"
    pocket = "micro"

    _ADX_MAX = 22.0
    _BBW_MAX = 0.21
    _ATR_MAX = 5.5
    _TOUCH_PIPS = 0.7
    _SPAN_MIN_PIPS = 1.4
    _RSI_LONG_MAX = 40.0
    _RSI_SHORT_MIN = 60.0
    _TREND_GAP_BLOCK_PIPS = 1.8
    _MID_GAP_BLOCK_PIPS = 3.2

    @staticmethod
    def _bb_levels(fac: Dict[str, object]) -> Optional[tuple[float, float, float, float, float]]:
        upper = to_float(fac.get("bb_upper"))
        lower = to_float(fac.get("bb_lower"))
        mid = to_float(fac.get("bb_mid"))
        if mid is None:
            mid = to_float(fac.get("ma20")) or to_float(fac.get("ma10"))
        bbw = to_float(fac.get("bbw")) or 0.0
        if upper is None or lower is None:
            if mid is None or bbw <= 0.0:
                return None
            half = abs(mid) * bbw / 2.0
            upper = mid + half
            lower = mid - half
        span = upper - lower
        if span <= 0:
            return None
        span_pips = span / PIP
        return upper, mid if mid is not None else (upper + lower) / 2.0, lower, span, span_pips

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        price = to_float(fac.get("close"))
        if price is None or price <= 0:
            return None

        adx = to_float(fac.get("adx"), 99.0) or 99.0
        bbw = to_float(fac.get("bbw"), 1.0) or 1.0
        atr = atr_pips(fac, default=2.0)
        if adx > MicroCompressionRevert._ADX_MAX:
            return None
        if bbw > MicroCompressionRevert._BBW_MAX:
            return None
        if atr > MicroCompressionRevert._ATR_MAX:
            return None

        levels = MicroCompressionRevert._bb_levels(fac)
        if not levels:
            return None
        upper, mid, lower, span, span_pips = levels
        if span_pips < MicroCompressionRevert._SPAN_MIN_PIPS:
            return None
        touch_pips = max(MicroCompressionRevert._TOUCH_PIPS, span_pips * 0.12)

        rsi = to_float(fac.get("rsi"), 50.0) or 50.0
        side = None
        if price >= upper - touch_pips * PIP and rsi >= MicroCompressionRevert._RSI_SHORT_MIN:
            side = "OPEN_SHORT"
        elif price <= lower + touch_pips * PIP and rsi <= MicroCompressionRevert._RSI_LONG_MAX:
            side = "OPEN_LONG"
        if side is None:
            return None

        ma_fast = to_float(fac.get("ma10")) or to_float(fac.get("ema10"))
        ma_slow = to_float(fac.get("ma20")) or to_float(fac.get("ema20"))
        trend_gap_pips = 0.0
        if ma_fast is not None and ma_slow is not None:
            trend_gap_pips = (ma_fast - ma_slow) / PIP
        mid_gap_pips = (price - mid) / PIP
        if side == "OPEN_SHORT":
            if trend_gap_pips >= MicroCompressionRevert._TREND_GAP_BLOCK_PIPS:
                return None
            if mid_gap_pips >= MicroCompressionRevert._MID_GAP_BLOCK_PIPS:
                return None
        else:
            if trend_gap_pips <= -MicroCompressionRevert._TREND_GAP_BLOCK_PIPS:
                return None
            if mid_gap_pips <= -MicroCompressionRevert._MID_GAP_BLOCK_PIPS:
                return None

        sl_pips = max(1.0, min(2.2, atr * 0.70))
        tp_pips = max(1.2, min(2.6, atr * 0.90))

        compression_score = max(0.0, min(1.0, (MicroCompressionRevert._BBW_MAX - bbw) / max(MicroCompressionRevert._BBW_MAX, 1e-6)))
        confidence = 56 + int(min(18.0, compression_score * 24.0 + max(0.0, (MicroCompressionRevert._ATR_MAX - atr)) * 1.6))
        confidence = max(45, min(92, confidence))

        return {
            "action": side,
            "sl_pips": round(sl_pips, 2),
            "tp_pips": round(tp_pips, 2),
            "confidence": confidence,
            "profile": "compression_revert",
            "tag": f"{MicroCompressionRevert.name}-{'long' if side == 'OPEN_LONG' else 'short'}",
        }

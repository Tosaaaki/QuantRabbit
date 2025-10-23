from __future__ import annotations
from typing import Dict, Optional

from analysis.ma_projection import MACrossProjection, compute_ma_projection


class MovingAverageCross:
    name = "TrendMA"
    pocket = "macro"

    _MIN_GAP_PIPS = 0.30
    _MAX_FAST_DISTANCE = 7.0
    _CROSS_MINUTES_STOP = 6.0

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        projection = compute_ma_projection(fac, timeframe_minutes=1.0)
        if not projection:
            return None

        ma10 = projection.fast_ma
        ma20 = projection.slow_ma
        adx = fac.get("adx", 0.0)
        if not ma10 or not ma20:
            return None
        if abs(projection.gap_pips) < MovingAverageCross._MIN_GAP_PIPS:
            return None
        if (
            abs(projection.price_to_fast_pips) > MovingAverageCross._MAX_FAST_DISTANCE
            and abs(projection.price_to_fast_pips) > abs(projection.price_to_slow_pips)
        ):
            return None

        direction = "long" if ma10 > ma20 else "short" if ma10 < ma20 else None
        if direction is None:
            return None

        macd_adjust = MovingAverageCross._macd_adjust(projection, direction)
        if macd_adjust is None:
            return None

        if projection.projected_cross_minutes and projection.projected_cross_minutes < MovingAverageCross._CROSS_MINUTES_STOP:
            return None

        confidence = MovingAverageCross._confidence(projection, direction, adx, macd_adjust)
        sl_pips, tp_pips = MovingAverageCross._targets(projection, direction, macd_adjust)

        if tp_pips <= 0 or sl_pips <= 0:
            return None
        tag_suffix = "bull" if direction == "long" else "bear"
        action = "OPEN_LONG" if direction == "long" else "OPEN_SHORT"
        return {
            "action": action,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "confidence": confidence,
            "tag": f"{MovingAverageCross.name}-{tag_suffix}",
        }

    @staticmethod
    def _confidence(
        projection: MACrossProjection,
        direction: str,
        adx: float,
        macd_adjust: float,
    ) -> int:
        gap_term = max(0.0, min(24.0, (abs(projection.gap_pips) - 0.5) * 3.4))
        slope = projection.gap_slope_pips if direction == "long" else -projection.gap_slope_pips
        slope_term = max(-12.0, min(18.0, slope * 6.5))
        stretch_penalty = max(0.0, abs(projection.price_to_fast_pips) - 2.5) * 1.2

        cross_penalty = 0.0
        if projection.projected_cross_minutes:
            cross_penalty = min(18.0, max(0.0, 24.0 - projection.projected_cross_minutes) * 0.7)

        adx_bonus = max(0.0, adx - 20.0) * 0.8
        confidence = 52.0 + gap_term + slope_term + adx_bonus + macd_adjust - stretch_penalty - cross_penalty
        return int(max(38.0, min(95.0, confidence)))

    @staticmethod
    def _targets(
        projection: MACrossProjection,
        direction: str,
        macd_adjust: float,
    ) -> tuple[float, float]:
        slope = projection.gap_slope_pips if direction == "long" else -projection.gap_slope_pips
        slope = max(-3.5, min(3.5, slope))
        trend_strength = min(6.0, max(0.0, abs(projection.gap_pips) - 0.5))

        stretch = abs(projection.price_to_fast_pips)
        base_sl = 23.5
        base_sl += max(0.0, stretch - 1.5) * 0.55
        base_sl -= slope * 1.4
        base_sl -= macd_adjust * 0.15
        if projection.projected_cross_minutes:
            base_sl += max(0.0, 18.0 - projection.projected_cross_minutes) * 0.08
        base_sl = max(18.0, min(36.0, base_sl))

        rr = 1.55 + min(0.65, trend_strength * 0.09) + slope * 0.05 + macd_adjust * 0.02
        if projection.projected_cross_minutes:
            rr -= min(0.35, max(0.0, 22.0 - projection.projected_cross_minutes) * 0.012)
        rr = max(1.25, min(2.45, rr))
        tp = base_sl * rr
        sl_rounded = round(base_sl, 2)
        tp_rounded = round(max(sl_rounded * 1.1, tp), 2)
        return sl_rounded, tp_rounded

    @staticmethod
    def _macd_adjust(projection: MACrossProjection, direction: str) -> Optional[float]:
        macd = projection.macd_pips
        slope = projection.macd_slope_pips
        if macd is None or slope is None:
            return 0.0

        cross_minutes = projection.macd_cross_minutes
        adjust = 0.0

        if direction == "long":
            if macd < -1.2 and slope <= 0:
                return None
            adjust += min(6.0, max(-6.0, macd * 0.7))
            adjust += slope * 3.2
            if slope < 0 and cross_minutes is not None:
                adjust -= min(8.0, max(0.0, 24.0 - cross_minutes) * 0.5)
        else:
            if macd > 1.2 and slope >= 0:
                return None
            adjust -= min(6.0, max(-6.0, macd * 0.7))
            adjust -= slope * 3.2
            if slope > 0 and cross_minutes is not None:
                adjust -= min(8.0, max(0.0, 24.0 - cross_minutes) * 0.5)

        return max(-12.0, min(12.0, adjust))


from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

PIP = 0.01
# Treat MA10/MA20 gaps >=0.5 pips combined with ADX>=21 as strong trend
MAX_TREND_GAP_PIPS = 0.5
MAX_ADX = 21.0
MIN_VOL_5M = 0.6
PROFILE_NAME = "bb_range_reversion"
MIN_DISTANCE_RANGE = 0.05
MIN_DISTANCE_TREND = 0.12


class BBRsi:
    name = "BB_RSI"
    pocket = "micro"
    _MIN_HOLD_SEC_RANGE = (80.0, 420.0)

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        if fac.get("bb_suppress_strong"):
            return None
        rsi = fac.get("rsi")
        bbw = fac.get("bbw")
        ma = fac.get("ma20")
        if not all([rsi, bbw, ma]):
            return None

        price = fac.get("close", ma)
        ma10 = fac.get("ma10")
        ma20 = fac.get("ma20") or ma
        adx = float(fac.get("adx") or 0.0)
        vol_5m = float(fac.get("vol_5m") or 0.0)
        if vol_5m < MIN_VOL_5M:
            return None
        try:
            ma_gap_pips = abs((ma10 - ma20) / PIP) if ma10 is not None and ma20 is not None else 0.0
        except (TypeError, ValueError):
            ma_gap_pips = 0.0
        if ma_gap_pips >= MAX_TREND_GAP_PIPS and adx >= MAX_ADX:
            return None

        upper = ma + (ma * bbw / 2)
        lower = ma - (ma * bbw / 2)
        band_width = upper - lower if upper is not None and lower is not None else 0.0
        if band_width <= 0:
            return None

        atr_hint = fac.get("atr_pips") or (fac.get("atr") or 0.0) * 100 or 6.0
        atr_hint = max(1.4, min(atr_hint, 9.0))
        spread_pips = float(fac.get("spread_pips") or 0.0)
        spread_buffer = max(0.5, spread_pips * 1.6)

        def _distance_ratio(boundary: float | None) -> float:
            if boundary is None or price is None or band_width <= 0:
                return 0.0
            return abs(price - boundary) / band_width

        def _calc_sl_tp(distance_ratio: float) -> tuple[float, float]:
            dist_bonus = max(0.8, distance_ratio * 3.5)
            sl_dynamic = max(
                spread_buffer + 0.6,
                min(atr_hint * 1.1, dist_bonus * atr_hint),
            )
            tp_dynamic = max(
                sl_dynamic * 1.45,
                min(atr_hint * 2.4, sl_dynamic + dist_bonus * atr_hint * 0.9),
            )
            return round(sl_dynamic, 2), round(tp_dynamic, 2)

        def _apply_range_bias(value: float, *, kind: str, score: float) -> float:
            score = max(0.0, min(1.0, score))
            if kind == "sl":
                scaled = value * (1.0 - 0.18 * score)
                return round(max(spread_buffer + 0.55, scaled), 2)
            scaled = value * (1.0 + 0.38 * score)
            return round(max(value, scaled), 2)

        range_score = fac.get("range_score") or 0.0
        try:
            range_score = max(0.0, min(1.0, float(range_score)))
        except (TypeError, ValueError):
            range_score = 0.0
        range_active = bool(fac.get("range_active"))

        def _coerce(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        range_floor = _coerce(fac.get("bb_min_distance_range"))
        trend_floor = _coerce(fac.get("bb_min_distance_trend"))

        def _min_distance(score: float, active: bool) -> float:
            if active or score >= 0.5:
                base = range_floor if range_floor is not None else MIN_DISTANCE_RANGE
                return max(MIN_DISTANCE_RANGE, base)
            penalty = max(0.0, (0.5 - score)) * 0.12
            base = trend_floor if trend_floor is not None else MIN_DISTANCE_TREND
            return max(MIN_DISTANCE_RANGE, base - penalty)

        min_distance_req = _min_distance(range_score, range_active)

        rsi_eta_up = fac.get("rsi_eta_upper_min")
        rsi_eta_dn = fac.get("rsi_eta_lower_min")
        bbw_eta = fac.get("bbw_squeeze_eta_min")
        bbw_slope = fac.get("bbw_slope_per_bar", 0.0) or 0.0

        if price < lower and rsi < 30:
            distance = (lower - price) / band_width if band_width else 0.0
            if distance < min_distance_req:
                return None
            sl_dynamic, tp_dynamic = _calc_sl_tp(distance)
            sl_dynamic = _apply_range_bias(sl_dynamic, kind="sl", score=range_score)
            tp_dynamic = _apply_range_bias(tp_dynamic, kind="tp", score=range_score)
            if tp_dynamic <= sl_dynamic * 1.3:
                tp_dynamic = round(sl_dynamic * 1.35, 2)
            rsi_gap = max(0.0, 30 - rsi) / 30
            eta_bonus = 0.0
            if rsi_eta_up is not None:
                eta_bonus = max(0.0, min(12.0, (8.0 - min(8.0, rsi_eta_up)) * 1.4))
            squeeze_bias = 0.0 if bbw_eta is None else max(0.0, 6.0 - min(6.0, bbw_eta)) * 0.6
            if bbw_slope > 0:
                squeeze_bias *= 0.5
            confidence = int(
                max(35.0, min(95.0, 45.0 + distance * 70.0 + rsi_gap * 35.0 + eta_bonus + squeeze_bias))
            )
            confidence = min(100, confidence + int(range_score * 20.0))
            return {
                "action": "OPEN_LONG",
                "sl_pips": sl_dynamic,
                "tp_pips": tp_dynamic,
                "confidence": confidence,
                "profile": PROFILE_NAME,
                "loss_guard_pips": sl_dynamic,
                "target_tp_pips": tp_dynamic,
                "min_hold_sec": BBRsi._min_hold_seconds(tp_dynamic),
                "tag": f"{BBRsi.name}-long",
            }
        if price > upper and rsi > 70:
            distance = (price - upper) / band_width if band_width else 0.0
            if distance < min_distance_req:
                return None
            sl_dynamic, tp_dynamic = _calc_sl_tp(distance)
            sl_dynamic = _apply_range_bias(sl_dynamic, kind="sl", score=range_score)
            tp_dynamic = _apply_range_bias(tp_dynamic, kind="tp", score=range_score)
            if tp_dynamic <= sl_dynamic * 1.3:
                tp_dynamic = round(sl_dynamic * 1.35, 2)
            rsi_gap = max(0.0, rsi - 70) / 30
            eta_bonus = 0.0
            if rsi_eta_dn is not None:
                eta_bonus = max(0.0, min(12.0, (8.0 - min(8.0, rsi_eta_dn)) * 1.4))
            squeeze_bias = 0.0 if bbw_eta is None else max(0.0, 6.0 - min(6.0, bbw_eta)) * 0.6
            if bbw_slope > 0:
                squeeze_bias *= 0.5
            confidence = int(
                max(35.0, min(95.0, 45.0 + distance * 70.0 + rsi_gap * 35.0 + eta_bonus + squeeze_bias))
            )
            confidence = min(100, confidence + int(range_score * 20.0))
            return {
                "action": "OPEN_SHORT",
                "sl_pips": sl_dynamic,
                "tp_pips": tp_dynamic,
                "confidence": confidence,
                "profile": PROFILE_NAME,
                "loss_guard_pips": sl_dynamic,
                "target_tp_pips": tp_dynamic,
                "min_hold_sec": BBRsi._min_hold_seconds(tp_dynamic),
                "tag": f"{BBRsi.name}-short",
            }
        return None

    @staticmethod
    def _min_hold_seconds(tp_pips: float) -> float:
        floor, ceiling = BBRsi._MIN_HOLD_SEC_RANGE
        return round(max(floor, min(ceiling, tp_pips * 42.0)), 1)

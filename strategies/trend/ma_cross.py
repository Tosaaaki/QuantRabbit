from __future__ import annotations
from typing import Dict, Optional

from analysis.ma_projection import MACrossProjection, compute_ma_projection
from typing import List, Dict, Optional as _Optional


class MovingAverageCross:
    name = "TrendMA"
    pocket = "macro"

    # Entry quality baselines (pips are 0.01 for USD/JPY)
    _MIN_GAP_PIPS = 0.90            # require >= ~1 pip MA10/MA20 separation
    _MIN_TREND_ADX = 25.0           # trend baseline stronger than 24
    _MIN_GAP_IN_WEAK_TREND = 1.40   # if ADX weak, demand clearer gap
    _MIN_SLOPE_IN_WEAK_TREND = 0.12 # and a minimal positive slope (pips/bar)
    _NARROW_BBW_LIMIT = 0.20        # align with range_guard threshold
    _MAX_FAST_DISTANCE = 7.0        # avoid stretched entries far from fast MA
    # Use bars-based threshold to work across timeframes (M1/H4)
    _CROSS_BARS_STOP = 6.0          # avoid entering too close to a cross (in bars)
    _MIN_ATR_PIPS = 1.20            # avoid illiquid minutes
    _MIN_GAP_ATR_RATIO = 0.55       # require separation vs ATR

    # Pullback gating: avoid buying too far below fast MA or selling too far above
    _PULLBACK_LIMIT = 0.80          # pips relative to fast MA

    # Simple M1-only range suppression to avoid trend entries under compression
    _RANGE_ADX_CUTOFF = 22.0
    _RANGE_BBW_CUTOFF = 0.20
    _RANGE_ATR_MAX = 6.0

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        # Derive timeframe minutes from candle timestamps if available
        tf_minutes = 1.0
        candles = fac.get("candles") or []
        try:
            if isinstance(candles, list) and len(candles) >= 2:
                import datetime as _dt
                def _parse(ts: str) -> _dt.datetime:
                    t = ts.replace("Z", "+00:00")
                    return _dt.datetime.fromisoformat(t)
                t2 = _parse(str(candles[-1]["timestamp"]))
                t1 = _parse(str(candles[-2]["timestamp"]))
                delta_min = max(1.0, (t2 - t1).total_seconds() / 60.0)
                # Clamp to common frames to avoid drift
                if abs(delta_min - 240.0) < 30.0:
                    tf_minutes = 240.0
                elif abs(delta_min - 60.0) < 10.0:
                    tf_minutes = 60.0
                else:
                    tf_minutes = max(1.0, min(240.0, delta_min))
        except Exception:
            tf_minutes = 1.0

        projection = compute_ma_projection(fac, timeframe_minutes=tf_minutes)
        if not projection:
            return None

        ma10 = projection.fast_ma
        ma20 = projection.slow_ma
        adx = fac.get("adx", 0.0)
        try:
            adx = float(adx)
        except (TypeError, ValueError):
            adx = 0.0
        if not ma10 or not ma20:
            return None
        bbw = fac.get("bbw")
        if isinstance(bbw, str):
            try:
                bbw = float(bbw)
            except ValueError:
                bbw = None

        # Lightweight range suppression when classic compression is present
        atr_pips_val: Optional[float]
        atr_pips_raw = fac.get("atr_pips")
        try:
            atr_pips_val = float(atr_pips_raw) if atr_pips_raw is not None else None
        except (TypeError, ValueError):
            atr_pips_val = None
        if atr_pips_val is None:
            try:
                # fallback from raw ATR (price units) to pips
                atr_pips_val = float(fac.get("atr") or 0.0) * 100.0
            except (TypeError, ValueError):
                atr_pips_val = None

        if (
            isinstance(bbw, (int, float))
            and atr_pips_val is not None
            and float(adx or 0.0) <= MovingAverageCross._RANGE_ADX_CUTOFF
            and float(bbw) <= MovingAverageCross._RANGE_BBW_CUTOFF
            and float(atr_pips_val) <= MovingAverageCross._RANGE_ATR_MAX
        ):
            # Defer to range-mode strategies; TrendMA stands down
            return None

        weak_trend = adx < MovingAverageCross._MIN_TREND_ADX
        if weak_trend:
            if abs(projection.gap_pips) < MovingAverageCross._MIN_GAP_IN_WEAK_TREND:
                return None
            if abs(projection.gap_slope_pips) < MovingAverageCross._MIN_SLOPE_IN_WEAK_TREND:
                return None
            if isinstance(bbw, (int, float)) and bbw <= MovingAverageCross._NARROW_BBW_LIMIT:
                return None
        if abs(projection.gap_pips) < MovingAverageCross._MIN_GAP_PIPS:
            return None
        if (
            abs(projection.price_to_fast_pips) > MovingAverageCross._MAX_FAST_DISTANCE
            and abs(projection.price_to_fast_pips) > abs(projection.price_to_slow_pips)
        ):
            return None

        # ATR/strength checks
        strength_ratio = None
        if atr_pips_val is not None and atr_pips_val > 0:
            strength_ratio = abs(projection.gap_pips) / max(atr_pips_val, 0.01)
            if atr_pips_val < MovingAverageCross._MIN_ATR_PIPS:
                return None
            if strength_ratio < MovingAverageCross._MIN_GAP_ATR_RATIO:
                return None

        direction = "long" if ma10 > ma20 else "short" if ma10 < ma20 else None
        if direction is None:
            return None

        # Align slope with the intended direction; avoid entering against a rolling flattening
        slope = projection.gap_slope_pips
        if direction == "long" and slope < -0.05:
            return None
        if direction == "short" and slope > 0.05:
            return None

        # Avoid chasing when price sits on the wrong side of the fast MA too far
        if direction == "long" and projection.price_to_fast_pips < -MovingAverageCross._PULLBACK_LIMIT:
            return None
        if direction == "short" and projection.price_to_fast_pips > MovingAverageCross._PULLBACK_LIMIT:
            return None

        macd_adjust = MovingAverageCross._macd_adjust(projection, direction)
        if macd_adjust is None:
            return None

        # Guard: avoid entries when too close to a cross in terms of bars
        if (
            projection.projected_cross_bars is not None
            and projection.projected_cross_bars < MovingAverageCross._CROSS_BARS_STOP
        ):
            return None

        # Multi-timeframe confluence: use H1/M10/M5 slopes to shape confidence.
        mtf = fac.get("mtf") if isinstance(fac, dict) else None
        conf_adj = 0
        sl_adj = 0.0
        tp_scale = 1.0
        if isinstance(mtf, dict):
            def _proj_from_candles(candles: List[Dict[str, float]], minutes: float) -> _Optional[MACrossProjection]:
                if not candles:
                    return None
                return compute_ma_projection({"candles": candles}, timeframe_minutes=minutes)
            p_h1 = _proj_from_candles(mtf.get("candles_h1") or [], 60.0)
            p_m10 = _proj_from_candles(mtf.get("candles_m10") or [], 10.0)
            p_m5 = _proj_from_candles(mtf.get("candles_m5") or [], 5.0)
            dir_sign = 1.0 if direction == "long" else -1.0
            support = 0
            oppose = 0
            for p, thr in ((p_h1, 0.04), (p_m10, 0.06), (p_m5, 0.08)):
                if not p or p.gap_slope_pips is None:
                    continue
                s = p.gap_slope_pips * dir_sign
                if s > thr:
                    support += 1
                elif s < -thr:
                    oppose += 1
            # Hard gate: if H1 opposes and either M10 or M5 opposes, stand down
            if (p_h1 and p_h1.gap_slope_pips is not None and p_h1.gap_slope_pips * dir_sign < -0.05) and (oppose >= 2):
                return None
            # Adjust confidence and risk: reward broad alignment, penalize near-term opposition
            conf_adj += max(0, support - 1) * 6  # +6..+12 when 2–3 agree
            conf_adj -= max(0, oppose - 0) * 5   # -5..-10 when 1–2 oppose
            if support >= 2:
                tp_scale = 1.06
            if oppose >= 1:
                sl_adj += 2.0

        confidence = MovingAverageCross._confidence(
            projection, direction, adx, macd_adjust
        )
        confidence = int(max(40.0, min(96.0, confidence + conf_adj)))
        sl_pips, tp_pips = MovingAverageCross._targets(
            projection, direction, macd_adjust, atr_pips_val
        )
        if sl_adj:
            sl_pips = round(max(10.0, sl_pips + sl_adj), 2)
        if tp_scale != 1.0:
            tp_pips = round(max(sl_pips * 1.08, tp_pips * tp_scale), 2)

        if tp_pips <= 0 or sl_pips <= 0:
            return None
        tag_suffix = "bull" if direction == "long" else "bear"
        action = "OPEN_LONG" if direction == "long" else "OPEN_SHORT"
        meta = {
            "gap_pips": projection.gap_pips,
            "gap_slope_pips": projection.gap_slope_pips,
            "price_to_fast_pips": projection.price_to_fast_pips,
            "strength_ratio": strength_ratio,
            "atr_pips": atr_pips_val,
            "adx": adx,
            "mtf_support": int(locals().get("support", 0)) if isinstance(mtf, dict) else 0,
            "mtf_oppose": int(locals().get("oppose", 0)) if isinstance(mtf, dict) else 0,
        }
        return {
            "action": action,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "confidence": confidence,
            "tag": f"{MovingAverageCross.name}-{tag_suffix}",
            "_meta": meta,
        }

    @staticmethod
    def _confidence(
        projection: MACrossProjection,
        direction: str,
        adx: float,
        macd_adjust: float,
    ) -> int:
        # Emphasize separation and directional slope; penalize stretch and near-cross risk
        sep = max(0.0, abs(projection.gap_pips) - 0.8)
        gap_term = min(26.0, sep * 3.2)
        signed_slope = projection.gap_slope_pips if direction == "long" else -projection.gap_slope_pips
        slope_term = max(-10.0, min(20.0, signed_slope * 7.5))
        stretch_penalty = max(0.0, abs(projection.price_to_fast_pips) - 2.0) * 1.4

        cross_penalty = 0.0
        if projection.projected_cross_bars is not None:
            cross_penalty = min(16.0, max(0.0, 6.0 - projection.projected_cross_bars) * 2.5)

        adx_bonus = max(0.0, adx - 22.0) * 0.85
        raw = 50.0 + gap_term + slope_term + adx_bonus + macd_adjust - stretch_penalty - cross_penalty
        return int(max(40.0, min(96.0, raw)))

    @staticmethod
    def _targets(
        projection: MACrossProjection,
        direction: str,
        macd_adjust: float,
        atr_pips: Optional[float],
    ) -> tuple[float, float]:
        slope = projection.gap_slope_pips if direction == "long" else -projection.gap_slope_pips
        slope = max(-3.5, min(3.5, slope))
        trend_strength = min(6.0, max(0.0, abs(projection.gap_pips) - 0.8))

        stretch = abs(projection.price_to_fast_pips)
        # ATR-aware SL sizing; tighten slightly when near a cross or stretched
        atr = atr_pips or 0.0
        base_sl = max(14.0, atr * 3.2)
        base_sl += max(0.0, stretch - 1.2) * 0.6
        base_sl -= slope * 1.2
        base_sl -= macd_adjust * 0.12
        if projection.projected_cross_bars is not None:
            base_sl += max(0.0, 4.0 - projection.projected_cross_bars) * 0.6
        base_sl = max(16.0, min(32.0, base_sl))

        rr = 1.45 + min(0.55, trend_strength * 0.08) + slope * 0.04 + macd_adjust * 0.02
        if projection.projected_cross_bars is not None:
            rr -= min(0.30, max(0.0, 4.0 - projection.projected_cross_bars) * 0.06)
        rr = max(1.30, min(2.20, rr))
        tp = base_sl * rr
        sl_rounded = round(base_sl, 2)
        tp_rounded = round(max(sl_rounded * 1.08, tp), 2)
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

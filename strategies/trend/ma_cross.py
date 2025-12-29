from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from analysis.ma_projection import MACrossProjection, compute_ma_projection


class MovingAverageCross:
    name = "TrendMA"
    pocket = "macro"
    profile = "macro_trend_ma"

    # Entry quality baselines (pips are 0.01 for USD/JPY)
    _MIN_GAP_PIPS = 0.28            # allow smaller MA10/MA20 separation
    _MIN_TREND_ADX = 12.5           # allow weaker trend
    _MIN_GAP_IN_WEAK_TREND = 0.60   # lower gap requirement when ADX is weak
    _MIN_SLOPE_IN_WEAK_TREND = 0.03 # allow gentler slope
    _NARROW_BBW_LIMIT = 0.14        # stand down only in ultra compression
    _MAX_FAST_DISTANCE = 8.0        # avoid stretched entries far from fast MA
    # Use bars-based threshold to work across timeframes (M1/H4)
    _CROSS_BARS_STOP = 1.5          # allow entries closer to a cross
    _MIN_ATR_PIPS = 0.55            # allow lower ATR
    _MIN_GAP_ATR_RATIO = 0.14       # require separation vs ATR

    # Pullback gating: avoid buying too far below fast MA or selling too far above
    _PULLBACK_LIMIT = 1.60          # pips relative to fast MA

    # Simple M1-only range suppression to avoid trend entries under compression
    _RANGE_ADX_CUTOFF = 12.0  # さらに緩和して弱トレンドでも拾う
    _RANGE_BBW_CUTOFF = 0.12  # 締まり気味でも拾う
    _RANGE_ATR_MAX = 10.0     # 高ATRでも抑制しすぎない
    _MACRO_TREND_ADX_OVERRIDE = 24.0
    _MACRO_TREND_GAP_PIPS = 1.2
    _OPPOSITE_FLIP_COOLDOWN_SEC = 1800  # 30min flip guard for macro direction changes

    @staticmethod
    def _swing_levels(candles: List[Dict], close_val: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        candles_list = list(candles or [])
        highs: List[float] = []
        lows: List[float] = []
        for c in candles_list[-30:]:
            h = c.get("high") or c.get("h")
            l = c.get("low") or c.get("l")
            try:
                if h is not None:
                    highs.append(float(h))
                if l is not None:
                    lows.append(float(l))
            except (TypeError, ValueError):
                continue
        swing_high = max(highs) if highs else close_val
        swing_low = min(lows) if lows else close_val
        return swing_high, swing_low

    @staticmethod
    def _adjust_targets_with_structure(
        *,
        tp_pips: float,
        sl_pips: float,
        direction: str,
        close_price: Optional[float],
        candles: List[Dict],
        projection: MACrossProjection,
        fac: Dict,
    ) -> Tuple[float, float]:
        """スイング高安・VWAP・MA傾きに基づき TP/SL を補正する。"""
        if close_price is None:
            return tp_pips, sl_pips
        swing_high, swing_low = MovingAverageCross._swing_levels(candles, close_price)
        vwap_gap = None
        try:
            vwap = fac.get("vwap")
            vwap_gap = abs(close_price - float(vwap)) / 0.01 if vwap is not None else None
        except Exception:
            vwap_gap = None
        slope_boost = abs(projection.gap_slope_pips or 0.0) * 9.0  # ~0.1 slope -> +0.9pips

        tp_out = tp_pips
        sl_out = sl_pips
        if direction == "long":
            swing_room = (swing_high - close_price) / 0.01 if swing_high and swing_high > close_price else None
            if swing_room is not None:
                tp_out = max(tp_out, min(tp_out * 1.6, swing_room * 0.9 + slope_boost))
            swing_buffer = (close_price - swing_low) / 0.01 * 0.9 if swing_low and swing_low < close_price else None
            if swing_buffer:
                sl_out = max(sl_out, min(sl_out * 1.6, swing_buffer))
        else:
            swing_room = (close_price - swing_low) / 0.01 if swing_low and swing_low < close_price else None
            if swing_room is not None:
                tp_out = max(tp_out, min(tp_out * 1.6, swing_room * 0.9 + slope_boost))
            swing_buffer = (swing_high - close_price) / 0.01 * 0.9 if swing_high and swing_high > close_price else None
            if swing_buffer:
                sl_out = max(sl_out, min(sl_out * 1.6, swing_buffer))
        if vwap_gap is not None and tp_out > 0:
            tp_out = min(tp_out * 1.05, tp_out + max(0.0, (vwap_gap - 5.0) * 0.1))
        return round(max(tp_out, tp_pips), 2), round(max(sl_out, sl_pips), 2)

    @staticmethod
    def _log_skip(reason: str, **kwargs) -> None:
        extras = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
        logging.info("[STRAT_SKIP_DETAIL] TrendMA reason=%s %s", reason, extras)

    @staticmethod
    def _parse_dt(value: object) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            return None

    @staticmethod
    def _opposite_flip_blocked(
        direction: str, now: Optional[datetime] = None
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        cooldown = MovingAverageCross._OPPOSITE_FLIP_COOLDOWN_SEC
        if cooldown <= 0 or not direction:
            return False, None, None
        path = Path("logs/trades.db")
        if not path.exists():
            return False, None, None
        try:
            with sqlite3.connect(path) as con:
                con.row_factory = sqlite3.Row
                row = con.execute(
                    """
                    SELECT close_time, units
                    FROM trades
                    WHERE state='CLOSED'
                      AND strategy_tag LIKE 'TrendMA%'
                      AND close_time IS NOT NULL
                    ORDER BY close_time DESC
                    LIMIT 1
                    """
                ).fetchone()
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("[TrendMA] flip_guard_lookup_failed err=%s", exc)
            return False, None, None
        if not row:
            return False, None, None
        try:
            units = float(row["units"] or 0.0)
        except Exception:
            return False, None, None
        if units == 0:
            return False, None, None
        last_dir = "long" if units > 0 else "short"
        if last_dir == direction:
            return False, None, last_dir
        closed_at = MovingAverageCross._parse_dt(row["close_time"])
        if not closed_at:
            return False, None, last_dir
        now_ts = now or datetime.now(timezone.utc)
        diff_sec = (now_ts - closed_at).total_seconds()
        if diff_sec < 0:
            return False, None, last_dir
        if diff_sec < cooldown:
            return True, diff_sec / 60.0, last_dir
        return False, diff_sec / 60.0, last_dir

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

        try:
            spread_pips = float(fac.get("spread_pips") or 0.0)
        except (TypeError, ValueError):
            spread_pips = 0.0

        projection = compute_ma_projection(fac, timeframe_minutes=tf_minutes)
        if not projection:
            MovingAverageCross._log_skip("projection_missing", tf_minutes=tf_minutes)
            return None

        ma10 = projection.fast_ma
        ma20 = projection.slow_ma
        adx = fac.get("adx", 0.0)
        try:
            adx = float(adx)
        except (TypeError, ValueError):
            adx = 0.0
        if not ma10 or not ma20:
            MovingAverageCross._log_skip("ma_missing", ma10=ma10, ma20=ma20)
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

        trend_override = False
        try:
            trend_override = abs(projection.gap_pips) >= 1.0 and abs(projection.gap_slope_pips or 0.0) >= 0.05
        except Exception:
            trend_override = False

        direction = "long" if ma10 > ma20 else "short" if ma10 < ma20 else None
        macro_trend_override = False
        h4_adx = None
        h4_gap_pips = None
        h4_dir = None
        if direction:
            try:
                from indicators.factor_cache import all_factors
                fac_h4 = all_factors().get("H4") or {}
                h4_adx = float(fac_h4.get("adx") or 0.0)
                ma10_h4 = float(fac_h4.get("ma10") or 0.0)
                ma20_h4 = float(fac_h4.get("ma20") or 0.0)
                if ma10_h4 and ma20_h4:
                    h4_dir = "long" if ma10_h4 > ma20_h4 else "short" if ma10_h4 < ma20_h4 else None
                    h4_gap_pips = abs(ma10_h4 - ma20_h4) / 0.01
                if (
                    h4_dir
                    and h4_dir == direction
                    and h4_adx is not None
                    and h4_adx >= MovingAverageCross._MACRO_TREND_ADX_OVERRIDE
                    and h4_gap_pips is not None
                    and h4_gap_pips >= MovingAverageCross._MACRO_TREND_GAP_PIPS
                ):
                    macro_trend_override = True
            except Exception:
                macro_trend_override = False

        if h4_dir and direction and h4_dir != direction and not macro_trend_override:
            MovingAverageCross._log_skip(
                "h4_conflict",
                h4_dir=h4_dir,
                h4_adx=round(float(h4_adx or 0.0), 2) if h4_adx is not None else None,
                h4_gap=round(float(h4_gap_pips or 0.0), 3) if h4_gap_pips is not None else None,
                dir=direction,
            )
            return None

        range_block = (
            isinstance(bbw, (int, float))
            and atr_pips_val is not None
            and float(adx or 0.0) <= MovingAverageCross._RANGE_ADX_CUTOFF
            and float(bbw) <= MovingAverageCross._RANGE_BBW_CUTOFF
            and float(atr_pips_val) <= MovingAverageCross._RANGE_ATR_MAX
        )
        if range_block and macro_trend_override:
            logging.info(
                "[STRAT_GUARD] TrendMA bypass range_suppression h4_adx=%.2f h4_gap=%.2f h4_dir=%s",
                float(h4_adx or 0.0),
                float(h4_gap_pips or 0.0),
                h4_dir,
            )
        elif range_block and not trend_override:
            # Defer to range-mode strategies; TrendMA stands down
            MovingAverageCross._log_skip(
                "range_suppression",
                adx=round(float(adx or 0.0), 2),
                bbw=round(float(bbw), 4),
                atr_pips=round(float(atr_pips_val), 3),
            )
            return None
        if trend_override:
            MovingAverageCross._log_skip(
                "range_override_strong_trend",
                gap_pips=round(projection.gap_pips, 4),
                gap_slope=round(projection.gap_slope_pips or 0.0, 5),
                adx=round(float(adx or 0.0), 2),
                bbw=round(float(bbw), 4) if isinstance(bbw, (int, float)) else None,
                atr_pips=round(float(atr_pips_val or 0.0), 3) if atr_pips_val is not None else None,
            )

        # 短期ドリフトが逆行しているときは方向を捨てる
        drift_keys = ("drift_pips_15m", "drift_15m", "return_15m_pips", "drift_pips_30m", "return_30m_pips")
        drift_pips = 0.0
        for k in drift_keys:
            val = fac.get(k)
            if val is None:
                continue
            try:
                drift_pips = float(val)
                break
            except (TypeError, ValueError):
                continue
        if direction == "long" and drift_pips < -1.5:
            MovingAverageCross._log_skip("drift_against_long", drift_pips=drift_pips)
            return None
        if direction == "short" and drift_pips > 1.5:
            MovingAverageCross._log_skip("drift_against_short", drift_pips=drift_pips)
            return None

        weak_trend = adx < MovingAverageCross._MIN_TREND_ADX
        if weak_trend:
            if abs(projection.gap_pips) < MovingAverageCross._MIN_GAP_IN_WEAK_TREND:
                MovingAverageCross._log_skip(
                    "weak_trend_gap_small",
                    adx=round(adx, 2),
                    gap_pips=round(projection.gap_pips, 4),
                )
                return None
            if abs(projection.gap_slope_pips) < MovingAverageCross._MIN_SLOPE_IN_WEAK_TREND:
                MovingAverageCross._log_skip(
                    "weak_trend_slope_small",
                    adx=round(adx, 2),
                    gap_slope=round(projection.gap_slope_pips or 0.0, 5),
                )
                return None
            if isinstance(bbw, (int, float)) and bbw <= MovingAverageCross._NARROW_BBW_LIMIT:
                MovingAverageCross._log_skip(
                    "weak_trend_bbw_narrow",
                    adx=round(adx, 2),
                    bbw=round(bbw, 4),
                )
                return None
        if abs(projection.gap_pips) < MovingAverageCross._MIN_GAP_PIPS:
            MovingAverageCross._log_skip(
                "gap_small",
                gap_pips=round(projection.gap_pips, 4),
                min_gap=MovingAverageCross._MIN_GAP_PIPS,
            )
            return None
        if (
            abs(projection.price_to_fast_pips) > MovingAverageCross._MAX_FAST_DISTANCE
            and abs(projection.price_to_fast_pips) > abs(projection.price_to_slow_pips)
        ):
            MovingAverageCross._log_skip(
                "price_far_from_fast",
                price_to_fast=round(projection.price_to_fast_pips, 4),
                price_to_slow=round(projection.price_to_slow_pips or 0.0, 4),
                max_fast=MovingAverageCross._MAX_FAST_DISTANCE,
            )
            return None

        # ATR/strength checks
        strength_ratio = None
        if atr_pips_val is not None and atr_pips_val > 0:
            strength_ratio = abs(projection.gap_pips) / max(atr_pips_val, 0.01)
            if atr_pips_val < MovingAverageCross._MIN_ATR_PIPS:
                MovingAverageCross._log_skip(
                    "atr_low",
                    atr_pips=round(atr_pips_val, 3),
                    min_atr=MovingAverageCross._MIN_ATR_PIPS,
                )
                return None
            if strength_ratio < MovingAverageCross._MIN_GAP_ATR_RATIO:
                MovingAverageCross._log_skip(
                    "strength_ratio_low",
                    strength_ratio=round(strength_ratio, 4),
                    gap_pips=round(projection.gap_pips, 4),
                    atr_pips=round(atr_pips_val, 3),
                )
                return None

        if direction is None:
            MovingAverageCross._log_skip("direction_flat", ma10=ma10, ma20=ma20)
            return None

        # Align slope with the intended direction; avoid entering against a rolling flattening
        slope = projection.gap_slope_pips
        if direction == "long" and slope < -0.05:
            MovingAverageCross._log_skip(
                "slope_against_long", slope=round(slope or 0.0, 5)
            )
            return None
        if direction == "short" and slope > 0.05:
            MovingAverageCross._log_skip(
                "slope_against_short", slope=round(slope or 0.0, 5)
            )
            return None

        blocked, since_min, last_dir = MovingAverageCross._opposite_flip_blocked(direction)
        if blocked:
            MovingAverageCross._log_skip(
                "opposite_flip_cooldown",
                last_dir=last_dir,
                minutes=round(since_min or 0.0, 2),
                cooldown_min=round(MovingAverageCross._OPPOSITE_FLIP_COOLDOWN_SEC / 60.0, 1),
            )
            return None

        # Avoid chasing when price sits on the wrong side of the fast MA too far
        if direction == "long" and projection.price_to_fast_pips < -MovingAverageCross._PULLBACK_LIMIT:
            MovingAverageCross._log_skip(
                "long_pullback_too_far",
                price_to_fast=round(projection.price_to_fast_pips, 4),
                limit=MovingAverageCross._PULLBACK_LIMIT,
            )
            return None
        if direction == "short" and projection.price_to_fast_pips > MovingAverageCross._PULLBACK_LIMIT:
            MovingAverageCross._log_skip(
                "short_pullback_too_far",
                price_to_fast=round(projection.price_to_fast_pips, 4),
                limit=MovingAverageCross._PULLBACK_LIMIT,
            )
            return None

        macd_adjust = MovingAverageCross._macd_adjust(projection, direction)
        if macd_adjust is None:
            MovingAverageCross._log_skip(
                "macd_filter",
                macd=fac.get("macd"),
                macd_sig=fac.get("macd_signal"),
                direction=direction,
            )
            return None

        # Guard: avoid entries when too close to a cross in terms of bars
        if (
            projection.projected_cross_bars is not None
            and projection.projected_cross_bars < MovingAverageCross._CROSS_BARS_STOP
        ):
            MovingAverageCross._log_skip(
                "too_close_to_cross",
                projected_cross=round(projection.projected_cross_bars, 3),
                stop=MovingAverageCross._CROSS_BARS_STOP,
            )
            return None

        # Multi-timeframe confluence: use H1/M10/M5 slopes to shape confidence.
        mtf = fac.get("mtf") if isinstance(fac, dict) else None
        conf_adj = 0
        sl_adj = 0.0
        tp_scale = 1.0
        # Session bias: London/NY slightly looser, Asia slightly stricter (no hard gate)
        hour = datetime.utcnow().hour
        if 7 <= hour < 17 or 17 <= hour < 23:
            conf_adj += 3
        else:
            conf_adj -= 2
        if isinstance(mtf, dict):
            def _proj_from_candles(candles: List[Dict[str, float]], minutes: float) -> Optional[MACrossProjection]:
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
                MovingAverageCross._log_skip(
                    "mtf_opposition",
                    h1_slope=round(p_h1.gap_slope_pips, 5) if p_h1 and p_h1.gap_slope_pips is not None else None,
                    oppose=oppose,
                    support=support,
                    dir=direction,
                )
                return None
            # Adjust confidence and risk: reward broad alignment, penalize near-term opposition
            conf_adj += max(0, support - 1) * 6  # +6..+12 when 2–3 agree
            conf_adj -= max(0, oppose - 0) * 5   # -5..-10 when 1–2 oppose
            if support >= 2:
                tp_scale = 1.06
            if oppose >= 1:
                sl_adj += 2.0

        # Cluster / VWAP context (favor breakouts when room, penalize when tight)
        try:
            cluster_gap = float(fac.get("cluster_high_gap") or fac.get("cluster_low_gap") or 0.0)
        except Exception:
            cluster_gap = 0.0
        try:
            vwap_gap = float(fac.get("vwap_gap") or 0.0)
        except Exception:
            vwap_gap = 0.0
        if cluster_gap > 0:
            if cluster_gap < 3.0:
                conf_adj -= 4
            elif cluster_gap > 7.0:
                conf_adj += 4
        if abs(vwap_gap) >= 3.0:
            conf_adj += 2

        confidence = MovingAverageCross._confidence(
            projection, direction, adx, macd_adjust
        )
        confidence = int(max(40.0, min(96.0, confidence + conf_adj)))
        sl_pips, tp_pips = MovingAverageCross._targets(
            projection, direction, macd_adjust, atr_pips_val, spread_pips
        )
        if sl_adj:
            sl_pips = round(max(10.0, sl_pips + sl_adj), 2)
        if tp_scale != 1.0:
            tp_pips = round(max(sl_pips * 1.08, tp_pips * tp_scale), 2)

        entry_type: Optional[str] = None
        entry_price: Optional[float] = None
        entry_tolerance: Optional[float] = None
        close_price = fac.get("close")
        try:
            close_val = float(close_price) if close_price is not None else None
        except (TypeError, ValueError):
            close_val = None
        # Treat stretched distance from fast MA as a momentum proxy for pullback sizing
        try:
            strong_momentum = abs(float(projection.price_to_fast_pips or 0.0)) >= 3.0
        except Exception:
            strong_momentum = False
        atr_hint = fac.get("atr_pips")
        if atr_hint is None:
            atr_hint = (fac.get("atr") or 0.0) * 100
        try:
            atr_value = float(atr_hint or 0.0)
        except (TypeError, ValueError):
            atr_value = 0.0
        pullback = None
        if close_val is not None:
            pullback = 1.1 + atr_value * 0.35
            if atr_value <= 1.6:
                pullback = 1.05 + atr_value * 0.32
            elif atr_value <= 3.0:
                pullback = 1.35 + atr_value * 0.38
            elif atr_value <= 5.0:
                pullback = 1.55 + atr_value * 0.4
            else:
                pullback = 1.85 + atr_value * 0.42
            if strong_momentum:
                pullback *= 0.85
            pullback = max(1.2, min(6.5, round(pullback, 2)))
            tolerance = max(0.35, min(1.2, pullback * 0.32 + (0.18 if atr_value <= 2.0 else 0.1)))
            if direction == "long":
                entry_price = round(close_val - pullback * 0.01, 3)
            else:
                entry_price = round(close_val + pullback * 0.01, 3)
            entry_type = "limit"
            entry_tolerance = tolerance

        if tp_pips <= 0 or sl_pips <= 0:
            return None
        tp_pips, sl_pips = MovingAverageCross._adjust_targets_with_structure(
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            direction=direction,
            close_price=close_val,
            candles=candles,
            projection=projection,
            fac=fac,
        )
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
            "profile": MovingAverageCross.profile,
            "loss_guard_pips": round(sl_pips * 0.75, 2),
            "target_tp_pips": tp_pips,
            "min_hold_sec": MovingAverageCross._min_hold_seconds(tp_pips, tf_minutes),
            "tag": f"{MovingAverageCross.name}-{tag_suffix}",
            "_meta": meta,
        }
        if entry_type and entry_price is not None:
            payload["entry_type"] = entry_type
            payload["entry_price"] = entry_price
            payload["entry_tolerance_pips"] = entry_tolerance or 0.6
            payload["meta"] = {
                "pullback_pips": pullback,
                "gap_slope_pips": projection.gap_slope_pips,
                "gap_pips": projection.gap_pips,
                "momentum_gap": momentum_gap,
                "macd_slope": macd_slope,
            }
        return payload

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
        spread_pips: float = 0.0,
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
        spread_floor = max(10.0, spread_pips * 2.4 + 6.0)
        sl_rounded = max(sl_rounded, spread_floor)
        tp_floor = sl_rounded * 1.2 + spread_pips * 0.8
        tp_rounded = round(max(sl_rounded * 1.08, tp, tp_floor), 2)
        try:
            adx_val = float(fac.get("adx") or 0.0)
        except Exception:
            adx_val = 0.0
        if adx_val >= 28.0:
            tp_rounded = round(min(tp_rounded * 1.15, sl_rounded * 2.4), 2)
        elif adx_val <= 14.0:
            tp_rounded = round(max(tp_rounded * 0.9, sl_rounded * 1.15), 2)
        return sl_rounded, tp_rounded

    @staticmethod
    def _min_hold_seconds(tp_pips: float, timeframe_minutes: float) -> float:
        base = 300.0 if timeframe_minutes <= 5 else 480.0
        scale = max(1.0, timeframe_minutes / 5.0)
        hold = max(base, tp_pips * 5.5 * scale)
        return round(min(2400.0, hold), 1)

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

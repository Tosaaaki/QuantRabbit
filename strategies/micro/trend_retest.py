from __future__ import annotations

from typing import Dict, Iterable, Optional

PIP = 0.01


class MicroTrendRetest:
    """Trend breakout + retest catcher on micro timeframe."""

    name = "MicroTrendRetest"
    pocket = "micro"

    _LOOKBACK = 20
    _MIN_GAP_PIPS = 0.6
    _MIN_ADX = 20.0
    _RETEST_BUFFER_PIPS = 0.8
    _BREAKOUT_BUFFER_PIPS = 0.3
    _MAX_RETEST_DIST_PIPS = 3.2
    _SPREAD_PIPS_MAX = 1.2
    _SPREAD_ATR_RATIO_MAX = 0.30
    _QUALITY_GAP_ATR_MIN = 0.75
    _QUALITY_ADX_MIN = 24.0
    _QUALITY_TREND_RSI_BIAS_MAX = 4.0
    _TREND_SNAPSHOT_GAP_MIN = 8.0
    _TREND_SNAPSHOT_ADX_MIN = 22.0
    _LOW_ATR_PIPS = 3.4
    _RETEST_CLOSE_RECOVERY_MIN = 0.20
    _RETEST_CLOSE_RECOVERY_LOW_ATR_MIN = 0.55
    _RECLAIM_EXHAUSTION_LONG_RSI_MIN = 62.0
    _RECLAIM_EXHAUSTION_SHORT_RSI_MAX = 38.0
    _RECLAIM_SMALL_BODY_RATIO_MAX = 0.38
    _RECLAIM_LONG_BEAR_CLOSE_POS_MAX = 0.35
    _RECLAIM_SHORT_BULL_CLOSE_POS_MIN = 0.65
    _CHASE_EXHAUSTION_GAP_ATR_MIN = 0.35
    _CHASE_EXHAUSTION_ADX_MIN = 24.0
    _CHASE_EXHAUSTION_SNAPSHOT_GAP_MIN = 12.0
    _CHASE_EXHAUSTION_SNAPSHOT_ADX_MIN = 24.0
    _CHASE_EXHAUSTION_LONG_RSI_MIN = 58.0
    _CHASE_EXHAUSTION_SHORT_RSI_MAX = 42.0
    _LOW_ATR_CHASE_BREAKOUT_ATR_MIN = 0.55
    _LOW_ATR_CHASE_BREAKOUT_ATR_STEP = 0.10
    _LOW_ATR_CHASE_RETEST_ATR_MAX = 0.30
    _LOW_ATR_CHASE_RETEST_ATR_STEP = 0.06
    _LOW_ATR_CHASE_RECOVERY_MIN = 0.62
    _LOW_ATR_CHASE_RECOVERY_STEP = 0.06
    _LOW_ATR_CHASE_BODY_MIN_PIPS = 0.10
    _LOW_ATR_CHASE_WICK_EDGE_PIPS = 0.15

    @staticmethod
    def _to_float(value: object, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _candles(fac: Dict[str, object], count: int) -> list[dict]:
        candles = fac.get("candles") or []
        if not isinstance(candles, Iterable):
            return []
        tail: list[dict] = []
        for candle in list(candles)[-count:]:
            if isinstance(candle, dict):
                tail.append(candle)
        return tail

    @staticmethod
    def _indicator_quality_ok(
        *,
        direction: str,
        gap_pips: float,
        adx: float,
        atr_pips: float,
        rsi: float,
    ) -> bool:
        atr_norm = max(1.0, atr_pips)
        gap_atr_ratio = abs(gap_pips) / atr_norm
        if gap_atr_ratio < MicroTrendRetest._QUALITY_GAP_ATR_MIN:
            return True
        if adx < MicroTrendRetest._QUALITY_ADX_MIN:
            return True
        direction_sign = 1.0 if direction == "OPEN_LONG" else -1.0
        trend_rsi_bias = (rsi - 50.0) * direction_sign
        return trend_rsi_bias <= MicroTrendRetest._QUALITY_TREND_RSI_BIAS_MAX

    @staticmethod
    def _trend_snapshot_supports(direction: str, fac: Dict[str, object]) -> bool:
        snapshot = fac.get("trend_snapshot")
        if not isinstance(snapshot, dict):
            return True
        snap_direction = str(snapshot.get("direction") or "").strip().lower()
        if snap_direction not in {"long", "short"}:
            return True
        if (direction == "OPEN_LONG" and snap_direction == "long") or (
            direction == "OPEN_SHORT" and snap_direction == "short"
        ):
            return True
        tf = str(snapshot.get("tf") or "").strip().upper()
        if tf not in {"H1", "H4"}:
            return True
        gap_pips = MicroTrendRetest._to_float(snapshot.get("gap_pips"), 0.0) or 0.0
        adx = MicroTrendRetest._to_float(snapshot.get("adx"), 0.0) or 0.0
        return not (
            abs(gap_pips) >= MicroTrendRetest._TREND_SNAPSHOT_GAP_MIN
            and adx >= MicroTrendRetest._TREND_SNAPSHOT_ADX_MIN
        )

    @staticmethod
    def _retest_close_supports(
        *,
        direction: str,
        last_high: float,
        last_low: float,
        last_close: float,
        atr_pips: float,
    ) -> bool:
        candle_range = max(last_high - last_low, 0.0)
        if candle_range <= 0.0:
            return True
        threshold = (
            MicroTrendRetest._RETEST_CLOSE_RECOVERY_LOW_ATR_MIN
            if atr_pips <= MicroTrendRetest._LOW_ATR_PIPS
            else MicroTrendRetest._RETEST_CLOSE_RECOVERY_MIN
        )
        if direction == "OPEN_LONG":
            recovery = (last_close - last_low) / candle_range
        else:
            recovery = (last_high - last_close) / candle_range
        return recovery >= threshold

    @staticmethod
    def _reclaim_exhaustion_ok(
        *,
        direction: str,
        last_open: float,
        last_high: float,
        last_low: float,
        last_close: float,
        rsi: float,
    ) -> bool:
        candle_range = max(last_high - last_low, 0.0)
        if candle_range <= 0.0:
            return True
        body_ratio = abs(last_close - last_open) / candle_range
        close_pos = (last_close - last_low) / candle_range
        if direction == "OPEN_LONG":
            return not (
                rsi >= MicroTrendRetest._RECLAIM_EXHAUSTION_LONG_RSI_MIN
                and last_close <= last_open
                and (
                    body_ratio <= MicroTrendRetest._RECLAIM_SMALL_BODY_RATIO_MAX
                    or close_pos <= MicroTrendRetest._RECLAIM_LONG_BEAR_CLOSE_POS_MAX
                )
            )
        return not (
            rsi <= MicroTrendRetest._RECLAIM_EXHAUSTION_SHORT_RSI_MAX
            and last_close >= last_open
            and (
                body_ratio <= MicroTrendRetest._RECLAIM_SMALL_BODY_RATIO_MAX
                or close_pos >= MicroTrendRetest._RECLAIM_SHORT_BULL_CLOSE_POS_MIN
            )
        )

    @staticmethod
    def _same_direction_chase_pressure(
        *,
        direction: str,
        fac: Dict[str, object],
        gap_pips: float,
        adx: float,
        atr_pips: float,
    ) -> int:
        pressure = 0
        atr_norm = max(1.0, atr_pips)
        if (
            abs(gap_pips) / atr_norm >= MicroTrendRetest._CHASE_EXHAUSTION_GAP_ATR_MIN
            and adx >= MicroTrendRetest._CHASE_EXHAUSTION_ADX_MIN
        ):
            pressure += 1
        snapshot = fac.get("trend_snapshot")
        if not isinstance(snapshot, dict):
            return pressure
        snap_direction = str(snapshot.get("direction") or "").strip().lower()
        expected = "long" if direction == "OPEN_LONG" else "short"
        snap_gap = abs(MicroTrendRetest._to_float(snapshot.get("gap_pips"), 0.0) or 0.0)
        snap_adx = MicroTrendRetest._to_float(snapshot.get("adx"), 0.0) or 0.0
        if (
            snap_direction == expected
            and snap_gap >= MicroTrendRetest._CHASE_EXHAUSTION_SNAPSHOT_GAP_MIN
            and snap_adx >= MicroTrendRetest._CHASE_EXHAUSTION_SNAPSHOT_ADX_MIN
        ):
            pressure += 1
        return pressure

    @staticmethod
    def _chase_reset_ok(
        *,
        direction: str,
        fac: Dict[str, object],
        gap_pips: float,
        adx: float,
        atr_pips: float,
        rsi: float,
        level: float,
        prev_close: float,
        last_open: float,
        last_high: float,
        last_low: float,
        last_close: float,
    ) -> bool:
        pressure = MicroTrendRetest._same_direction_chase_pressure(
            direction=direction,
            fac=fac,
            gap_pips=gap_pips,
            adx=adx,
            atr_pips=atr_pips,
        )
        if pressure <= 0:
            return True
        candle_range = max(last_high - last_low, 0.0)
        if candle_range <= 0.0:
            return True
        close_pos = (last_close - last_low) / candle_range
        atr_norm = max(1.0, atr_pips)
        pressure_step = max(0, pressure - 1)
        if direction == "OPEN_LONG":
            if (
                rsi
                < MicroTrendRetest._CHASE_EXHAUSTION_LONG_RSI_MIN - pressure_step * 3.0
            ):
                return True
            breakout_stretch_pips = max(0.0, (prev_close - level) / PIP)
            retest_depth_pips = max(0.0, (level - last_low) / PIP)
            overshoot_pips = max(0.0, (last_high - level) / PIP)
            min_reset_depth = max(0.5, atr_norm * (0.18 + pressure_step * 0.07))
            max_overshoot = max(0.25, atr_norm * (0.10 - pressure_step * 0.03))
            max_close_pos = 0.68 - pressure_step * 0.08
            if atr_pips <= MicroTrendRetest._LOW_ATR_PIPS:
                shallow_breakout_min = atr_norm * (
                    MicroTrendRetest._LOW_ATR_CHASE_BREAKOUT_ATR_MIN
                    + pressure_step * MicroTrendRetest._LOW_ATR_CHASE_BREAKOUT_ATR_STEP
                )
                shallow_retest_max = atr_norm * (
                    MicroTrendRetest._LOW_ATR_CHASE_RETEST_ATR_MAX
                    + pressure_step * MicroTrendRetest._LOW_ATR_CHASE_RETEST_ATR_STEP
                )
                if (
                    breakout_stretch_pips >= shallow_breakout_min
                    and retest_depth_pips <= shallow_retest_max
                ):
                    body_pips = (last_close - last_open) / PIP
                    upper_wick_pips = max(
                        0.0, (last_high - max(last_open, last_close)) / PIP
                    )
                    lower_wick_pips = max(
                        0.0, (min(last_open, last_close) - last_low) / PIP
                    )
                    recovery_min = (
                        MicroTrendRetest._LOW_ATR_CHASE_RECOVERY_MIN
                        + pressure_step * MicroTrendRetest._LOW_ATR_CHASE_RECOVERY_STEP
                    )
                    if (
                        body_pips <= MicroTrendRetest._LOW_ATR_CHASE_BODY_MIN_PIPS
                        or close_pos < recovery_min
                        or lower_wick_pips
                        <= upper_wick_pips
                        + MicroTrendRetest._LOW_ATR_CHASE_WICK_EDGE_PIPS
                    ):
                        return False
            return (
                retest_depth_pips >= min_reset_depth
                and overshoot_pips <= max_overshoot
                and close_pos <= max_close_pos
                and last_close <= prev_close
            )
        if rsi > MicroTrendRetest._CHASE_EXHAUSTION_SHORT_RSI_MAX + pressure_step * 3.0:
            return True
        breakout_stretch_pips = max(0.0, (level - prev_close) / PIP)
        retest_depth_pips = max(0.0, (last_high - level) / PIP)
        overshoot_pips = max(0.0, (level - last_low) / PIP)
        min_reset_depth = max(0.5, atr_norm * (0.18 + pressure_step * 0.07))
        max_overshoot = max(0.25, atr_norm * (0.10 - pressure_step * 0.03))
        min_close_pos = 0.32 + pressure_step * 0.08
        if atr_pips <= MicroTrendRetest._LOW_ATR_PIPS:
            shallow_breakout_min = atr_norm * (
                MicroTrendRetest._LOW_ATR_CHASE_BREAKOUT_ATR_MIN
                + pressure_step * MicroTrendRetest._LOW_ATR_CHASE_BREAKOUT_ATR_STEP
            )
            shallow_retest_max = atr_norm * (
                MicroTrendRetest._LOW_ATR_CHASE_RETEST_ATR_MAX
                + pressure_step * MicroTrendRetest._LOW_ATR_CHASE_RETEST_ATR_STEP
            )
            if (
                breakout_stretch_pips >= shallow_breakout_min
                and retest_depth_pips <= shallow_retest_max
            ):
                body_pips = (last_open - last_close) / PIP
                upper_wick_pips = max(
                    0.0, (last_high - max(last_open, last_close)) / PIP
                )
                lower_wick_pips = max(
                    0.0, (min(last_open, last_close) - last_low) / PIP
                )
                recovery = 1.0 - close_pos
                recovery_min = (
                    MicroTrendRetest._LOW_ATR_CHASE_RECOVERY_MIN
                    + pressure_step * MicroTrendRetest._LOW_ATR_CHASE_RECOVERY_STEP
                )
                if (
                    body_pips <= MicroTrendRetest._LOW_ATR_CHASE_BODY_MIN_PIPS
                    or recovery < recovery_min
                    or upper_wick_pips
                    <= lower_wick_pips + MicroTrendRetest._LOW_ATR_CHASE_WICK_EDGE_PIPS
                ):
                    return False
        return (
            retest_depth_pips >= min_reset_depth
            and overshoot_pips <= max_overshoot
            and close_pos >= min_close_pos
            and last_close >= prev_close
        )

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        price = MicroTrendRetest._to_float(fac.get("close"))
        ma10 = MicroTrendRetest._to_float(fac.get("ma10"))
        ma20 = MicroTrendRetest._to_float(fac.get("ma20"))
        if price is None or ma10 is None or ma20 is None:
            return None

        # Spread filter
        spread_pips = MicroTrendRetest._to_float(fac.get("spread_pips"), 0.0) or 0.0
        atr_check = MicroTrendRetest._to_float(fac.get("atr_pips"), 0.0) or 0.0
        if spread_pips > 0 and atr_check > 0:
            spread_cap = max(
                MicroTrendRetest._SPREAD_PIPS_MAX,
                atr_check * MicroTrendRetest._SPREAD_ATR_RATIO_MAX,
            )
            if spread_pips > spread_cap:
                return None

        try:
            adx = float(fac.get("adx") or 0.0)
        except (TypeError, ValueError):
            adx = 0.0
        if adx < MicroTrendRetest._MIN_ADX:
            return None
        rsi = MicroTrendRetest._to_float(fac.get("rsi"), 50.0) or 50.0

        gap = (ma10 - ma20) / PIP
        if gap >= MicroTrendRetest._MIN_GAP_PIPS:
            direction = "OPEN_LONG"
        elif gap <= -MicroTrendRetest._MIN_GAP_PIPS:
            direction = "OPEN_SHORT"
        else:
            return None
        if not MicroTrendRetest._indicator_quality_ok(
            direction=direction,
            gap_pips=gap,
            adx=adx,
            atr_pips=atr_check,
            rsi=rsi,
        ):
            return None
        if not MicroTrendRetest._trend_snapshot_supports(direction, fac):
            return None

        candles = MicroTrendRetest._candles(fac, MicroTrendRetest._LOOKBACK + 2)
        if len(candles) < MicroTrendRetest._LOOKBACK + 2:
            return None

        recent = candles[-(MicroTrendRetest._LOOKBACK + 2) :]
        history = recent[:-2]
        prev = recent[-2]
        last = recent[-1]

        highs = [MicroTrendRetest._to_float(c.get("high"), 0.0) or 0.0 for c in history]
        lows = [MicroTrendRetest._to_float(c.get("low"), 0.0) or 0.0 for c in history]
        if not highs or not lows:
            return None
        level_high = max(highs)
        level_low = min(lows)

        prev_close = MicroTrendRetest._to_float(prev.get("close"), 0.0) or 0.0
        last_open = (
            MicroTrendRetest._to_float(last.get("open"), prev_close) or prev_close
        )
        last_close = MicroTrendRetest._to_float(last.get("close"), 0.0) or 0.0
        last_low = MicroTrendRetest._to_float(last.get("low"), 0.0) or 0.0
        last_high = MicroTrendRetest._to_float(last.get("high"), 0.0) or 0.0

        if direction == "OPEN_LONG":
            if prev_close < level_high + MicroTrendRetest._BREAKOUT_BUFFER_PIPS * PIP:
                return None
            if last_low > level_high + MicroTrendRetest._RETEST_BUFFER_PIPS * PIP:
                return None
            if (
                abs(last_close - level_high)
                > MicroTrendRetest._RETEST_BUFFER_PIPS * PIP
            ):
                return None
            if not MicroTrendRetest._retest_close_supports(
                direction=direction,
                last_high=last_high,
                last_low=last_low,
                last_close=last_close,
                atr_pips=atr_check,
            ):
                return None
            if not MicroTrendRetest._reclaim_exhaustion_ok(
                direction=direction,
                last_open=last_open,
                last_high=last_high,
                last_low=last_low,
                last_close=last_close,
                rsi=rsi,
            ):
                return None
            if not MicroTrendRetest._chase_reset_ok(
                direction=direction,
                fac=fac,
                gap_pips=gap,
                adx=adx,
                atr_pips=atr_check,
                rsi=rsi,
                level=level_high,
                prev_close=prev_close,
                last_open=last_open,
                last_high=last_high,
                last_low=last_low,
                last_close=last_close,
            ):
                return None
            retest_dist = abs(price - level_high) / PIP
            if retest_dist > MicroTrendRetest._MAX_RETEST_DIST_PIPS:
                return None
            if last_close > prev_close + 0.1 * PIP:
                return None
        else:
            if prev_close > level_low - MicroTrendRetest._BREAKOUT_BUFFER_PIPS * PIP:
                return None
            if last_high < level_low - MicroTrendRetest._RETEST_BUFFER_PIPS * PIP:
                return None
            if abs(last_close - level_low) > MicroTrendRetest._RETEST_BUFFER_PIPS * PIP:
                return None
            if not MicroTrendRetest._retest_close_supports(
                direction=direction,
                last_high=last_high,
                last_low=last_low,
                last_close=last_close,
                atr_pips=atr_check,
            ):
                return None
            if not MicroTrendRetest._reclaim_exhaustion_ok(
                direction=direction,
                last_open=last_open,
                last_high=last_high,
                last_low=last_low,
                last_close=last_close,
                rsi=rsi,
            ):
                return None
            if not MicroTrendRetest._chase_reset_ok(
                direction=direction,
                fac=fac,
                gap_pips=gap,
                adx=adx,
                atr_pips=atr_check,
                rsi=rsi,
                level=level_low,
                prev_close=prev_close,
                last_open=last_open,
                last_high=last_high,
                last_low=last_low,
                last_close=last_close,
            ):
                return None
            retest_dist = abs(price - level_low) / PIP
            if retest_dist > MicroTrendRetest._MAX_RETEST_DIST_PIPS:
                return None
            if last_close < prev_close - 0.1 * PIP:
                return None

        atr_hint = fac.get("atr_pips") or (fac.get("atr") or 0.0) * 100.0 or 5.0
        try:
            atr_hint = float(atr_hint)
        except (TypeError, ValueError):
            atr_hint = 5.0
        atr_hint = max(1.2, min(atr_hint, 10.0))
        # Previous: sl = max(1.2, atr * 0.7) was ~1.4-2.8p -- too tight for retest entries
        # that need to survive the retest probe before continuation.
        sl_pips = max(2.5, atr_hint * 1.10)
        tp_pips = max(sl_pips * 1.5, sl_pips + atr_hint * 0.95)

        confidence = 58 + int(
            min(16.0, abs(gap)) + min(10.0, max(0.0, adx - MicroTrendRetest._MIN_ADX))
        )
        if direction == "OPEN_LONG" and rsi < 52:
            confidence += 4
        if direction == "OPEN_SHORT" and rsi > 48:
            confidence += 4
        confidence = max(48, min(96, confidence))

        return {
            "action": direction,
            "sl_pips": round(sl_pips, 2),
            "tp_pips": round(tp_pips, 2),
            "confidence": confidence,
            "profile": "trend_retest",
            "tag": f"{MicroTrendRetest.name}-{'long' if direction == 'OPEN_LONG' else 'short'}",
        }

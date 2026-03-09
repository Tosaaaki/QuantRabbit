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
            spread_cap = max(MicroTrendRetest._SPREAD_PIPS_MAX, atr_check * MicroTrendRetest._SPREAD_ATR_RATIO_MAX)
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

        recent = candles[-(MicroTrendRetest._LOOKBACK + 2):]
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
        last_close = MicroTrendRetest._to_float(last.get("close"), 0.0) or 0.0
        last_low = MicroTrendRetest._to_float(last.get("low"), 0.0) or 0.0
        last_high = MicroTrendRetest._to_float(last.get("high"), 0.0) or 0.0

        if direction == "OPEN_LONG":
            if prev_close < level_high + MicroTrendRetest._BREAKOUT_BUFFER_PIPS * PIP:
                return None
            if last_low > level_high + MicroTrendRetest._RETEST_BUFFER_PIPS * PIP:
                return None
            if abs(last_close - level_high) > MicroTrendRetest._RETEST_BUFFER_PIPS * PIP:
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

        confidence = 58 + int(min(16.0, abs(gap)) + min(10.0, max(0.0, adx - MicroTrendRetest._MIN_ADX)))
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

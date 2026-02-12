from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

PIP = 0.01


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _candle_value(candle: Dict, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        val = _to_float(candle.get(key))
        if val is not None:
            return val
    return None


@dataclass(frozen=True)
class _SignalInputs:
    close: float
    ema10: float
    ema20: float
    ema50: float
    adx: float
    rsi: float
    atr_pips: float
    h1_close: float
    h1_ema20: float
    h1_adx: float
    m5_close: float
    m5_ema20: float
    vol_5m: float


class TrendReclaimLong:
    """
    Trend continuation idea:
    - Only long entries.
    - Require H1 uptrend and M1 EMA stack.
    - Wait for a pullback near EMA20, then enter on reclaim (close back above EMA10).
    """

    name = "TrendReclaimLong"
    pocket = "scalp"

    @staticmethod
    def _parse_inputs(fac: Dict) -> _SignalInputs | None:
        close = _to_float(fac.get("close"))
        ema10 = _to_float(fac.get("ema10") or fac.get("ma10"))
        ema20 = _to_float(fac.get("ema20") or fac.get("ma20"))
        ema50 = _to_float(fac.get("ema50") or fac.get("ma50"))
        adx = _to_float(fac.get("adx"), 0.0) or 0.0
        rsi = _to_float(fac.get("rsi"), 50.0) or 50.0
        atr_pips = _to_float(fac.get("atr_pips"))
        if atr_pips is None:
            atr_pips = (_to_float(fac.get("atr"), 0.0) or 0.0) * 100.0

        h1_close = _to_float(fac.get("h1_close"))
        h1_ema20 = _to_float(fac.get("h1_ema20") or fac.get("h1_ma20"))
        h1_adx = _to_float(fac.get("h1_adx"), 0.0) or 0.0

        m5_close = _to_float(fac.get("m5_close"), close)
        m5_ema20 = _to_float(fac.get("m5_ema20") or fac.get("m5_ma20"), ema20)
        vol_5m = _to_float(fac.get("vol_5m"), 0.0) or 0.0

        if None in (close, ema10, ema20, ema50, h1_close, h1_ema20):
            return None
        return _SignalInputs(
            close=close,
            ema10=ema10,
            ema20=ema20,
            ema50=ema50,
            adx=adx,
            rsi=rsi,
            atr_pips=atr_pips or 0.0,
            h1_close=h1_close,
            h1_ema20=h1_ema20,
            h1_adx=h1_adx,
            m5_close=m5_close or close,
            m5_ema20=m5_ema20 or ema20,
            vol_5m=vol_5m,
        )

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        inp = TrendReclaimLong._parse_inputs(fac)
        if inp is None:
            return None

        min_atr = _env_float("TR_LONG_MIN_ATR_PIPS", 0.8)
        max_atr = _env_float("TR_LONG_MAX_ATR_PIPS", 5.5)
        if inp.atr_pips <= min_atr or inp.atr_pips > max_atr:
            return None

        min_adx = _env_float("TR_LONG_MIN_ADX", 16.0)
        if inp.adx < min_adx:
            return None

        min_h1_adx = _env_float("TR_LONG_H1_MIN_ADX", 18.0)
        min_h1_gap = _env_float("TR_LONG_H1_MIN_GAP_PIPS", 3.0)
        h1_gap_pips = (inp.h1_close - inp.h1_ema20) / PIP
        if inp.h1_adx < min_h1_adx or h1_gap_pips < min_h1_gap:
            return None

        min_m5_gap = _env_float("TR_LONG_M5_MIN_GAP_PIPS", 0.5)
        m5_gap_pips = (inp.m5_close - inp.m5_ema20) / PIP
        if m5_gap_pips < min_m5_gap:
            return None

        if not (inp.ema10 > inp.ema20 > inp.ema50):
            return None

        rsi_min = _env_float("TR_LONG_RSI_MIN", 50.0)
        rsi_max = _env_float("TR_LONG_RSI_MAX", 73.0)
        if inp.rsi < rsi_min or inp.rsi > rsi_max:
            return None

        min_vol_5m = _env_float("TR_LONG_MIN_VOL_5M", 0.35)
        if inp.vol_5m < min_vol_5m:
            return None

        candles = fac.get("candles") or []
        if not isinstance(candles, list) or len(candles) < 12:
            return None

        lookback = max(8, _env_int("TR_LONG_IMPULSE_LOOKBACK", 14))
        pb_lookback = max(3, _env_int("TR_LONG_PULLBACK_LOOKBACK", 5))
        recent = candles[-lookback:]
        pullback_window = candles[-pb_lookback:]

        highs = [
            _candle_value(c, ("high", "h"))
            for c in recent
        ]
        lows = [
            _candle_value(c, ("low", "l"))
            for c in pullback_window
        ]
        highs = [h for h in highs if h is not None]
        lows = [l for l in lows if l is not None]
        if not highs or not lows:
            return None

        impulse_high = max(highs)
        pullback_low = min(lows)

        impulse_pips = (impulse_high - inp.ema20) / PIP
        pullback_depth = (impulse_high - pullback_low) / PIP
        pullback_touch = abs(pullback_low - inp.ema20) / PIP

        min_impulse = _env_float("TR_LONG_MIN_IMPULSE_PIPS", 2.2)
        min_pb_depth = _env_float("TR_LONG_PULLBACK_DEPTH_MIN_PIPS", 0.6)
        max_pb_depth = _env_float("TR_LONG_PULLBACK_DEPTH_MAX_PIPS", 5.0)
        max_touch = _env_float("TR_LONG_PULLBACK_TOUCH_MAX_PIPS", 1.0)
        if impulse_pips < min_impulse:
            return None
        if pullback_depth < min_pb_depth or pullback_depth > max_pb_depth:
            return None
        if pullback_touch > max_touch:
            return None

        reclaim_min = _env_float("TR_LONG_RECLAIM_MIN_PIPS", 0.10)
        reclaim_pips = (inp.close - inp.ema10) / PIP
        if reclaim_pips < reclaim_min:
            return None

        max_ema10_ext = _env_float("TR_LONG_MAX_EMA10_EXT_PIPS", 2.2)
        if reclaim_pips > max_ema10_ext:
            return None

        prev_high = _candle_value(candles[-2], ("high", "h"))
        if prev_high is not None:
            break_prev_high = (inp.close - prev_high) / PIP
            min_break = _env_float("TR_LONG_MIN_BREAK_PREV_HIGH_PIPS", -0.05)
            if break_prev_high < min_break:
                return None

        bb_upper = _to_float(fac.get("bb_upper"))
        if bb_upper is not None:
            bb_room = (bb_upper - inp.close) / PIP
            min_bb_room = _env_float("TR_LONG_MIN_BB_ROOM_PIPS", 0.15)
            if bb_room < min_bb_room:
                return None

        trend_score = _clamp((h1_gap_pips - min_h1_gap) / 8.0, 0.0, 1.0)
        reclaim_score = _clamp(reclaim_pips / 0.8, 0.0, 1.0)
        depth_score = _clamp((pullback_depth - min_pb_depth) / 3.0, 0.0, 1.0)
        adx_score = _clamp((inp.adx - min_adx) / 12.0, 0.0, 1.0)

        confidence = 53.0
        confidence += trend_score * 16.0
        confidence += reclaim_score * 10.0
        confidence += depth_score * 8.0
        confidence += adx_score * 7.0
        if inp.rsi > 68.0:
            confidence -= min(6.0, (inp.rsi - 68.0) * 1.5)
        confidence = _clamp(confidence, 45.0, 94.0)

        sl_atr_mult = _env_float("TR_LONG_SL_ATR_MULT", 1.2)
        sl_pb_mult = _env_float("TR_LONG_SL_PULLBACK_MULT", 0.65)
        sl_min = _env_float("TR_LONG_SL_MIN_PIPS", 1.8)
        sl_max = _env_float("TR_LONG_SL_MAX_PIPS", 7.5)
        sl_pips = max(inp.atr_pips * sl_atr_mult, pullback_depth * sl_pb_mult)
        sl_pips = _clamp(sl_pips, sl_min, sl_max)

        rr_min = _env_float("TR_LONG_RR_MIN", 1.25)
        rr_max = _env_float("TR_LONG_RR_MAX", 2.2)
        rr = rr_min + (rr_max - rr_min) * (0.55 * trend_score + 0.45 * reclaim_score)
        tp_min = _env_float("TR_LONG_TP_MIN_PIPS", 2.6)
        tp_max = _env_float("TR_LONG_TP_MAX_PIPS", 11.5)
        tp_pips = _clamp(sl_pips * rr, tp_min, tp_max)

        fast_cut = _clamp(inp.atr_pips * 0.9, 1.2, 6.0)
        fast_cut_time = int(_clamp(70.0 + inp.atr_pips * 22.0, 70.0, 260.0))

        return {
            "action": "OPEN_LONG",
            "sl_pips": round(sl_pips, 2),
            "tp_pips": round(tp_pips, 2),
            "confidence": int(round(confidence)),
            "fast_cut_pips": round(fast_cut, 2),
            "fast_cut_time_sec": fast_cut_time,
            "tag": f"{TrendReclaimLong.name}-pb-reclaim",
            "profile": "trend_reclaim_long",
            "notes": {
                "h1_gap_pips": round(h1_gap_pips, 2),
                "m5_gap_pips": round(m5_gap_pips, 2),
                "impulse_pips": round(impulse_pips, 2),
                "pullback_depth": round(pullback_depth, 2),
                "pullback_touch": round(pullback_touch, 2),
                "reclaim_pips": round(reclaim_pips, 2),
                "adx": round(inp.adx, 2),
                "rsi": round(inp.rsi, 2),
            },
        }

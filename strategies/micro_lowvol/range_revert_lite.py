from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

from .common import (
    PIP,
    atr_pips,
    candle_body_pips,
    clamp,
    latest_candles,
    price_delta_pips,
    to_float,
)
from utils.tuning_loader import get_tuning_value


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _tuned_float(keys: tuple[str, ...], default: float) -> float:
    val = get_tuning_value(keys)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _bb_levels(fac: Dict) -> Optional[Tuple[float, float, float, float]]:
    upper = to_float(fac.get("bb_upper"))
    lower = to_float(fac.get("bb_lower"))
    mid = to_float(fac.get("bb_mid")) or to_float(fac.get("ma20")) or to_float(fac.get("ma10"))
    bbw = to_float(fac.get("bbw"))
    if upper is None or lower is None:
        if mid is None or bbw is None or bbw <= 0:
            return None
        half = abs(mid) * bbw / 2.0
        upper = mid + half
        lower = mid - half
    span = upper - lower
    if span <= 0:
        return None
    span_pips = span / PIP
    return upper, mid if mid is not None else (upper + lower) / 2.0, lower, span_pips


class RangeRevertLite:
    name = "RangeRevertLite"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        close = to_float(fac.get("close"))
        rsi = to_float(fac.get("rsi"))
        stoch_rsi = to_float(fac.get("stoch_rsi"))
        adx = to_float(fac.get("adx"))
        bbw = to_float(fac.get("bbw"))
        macd_hist = to_float(fac.get("macd_hist"), 0.0) or 0.0
        vwap_gap = to_float(fac.get("vwap_gap"), 0.0) or 0.0
        ema_slope = to_float(fac.get("ema_slope_10"), 0.0) or 0.0
        div_score = to_float(fac.get("div_score"), 0.0) or 0.0

        if None in (close, rsi, stoch_rsi, adx, bbw):
            return None

        range_score = to_float(fac.get("range_score"), 0.0) or 0.0
        range_active = bool(fac.get("range_active"))

        range_score_min = _env_float("RRL_RANGE_SCORE_MIN", 0.40)
        range_score_min = _tuned_float(("strategies", "RangeRevertLite", "range_score_min"), range_score_min)
        require_range = _env_bool("RRL_REQUIRE_RANGE_ACTIVE", False)
        require_score = _env_bool("RRL_REQUIRE_RANGE_SCORE", True)
        if require_score and range_score < range_score_min:
            return None
        if require_range and not range_active:
            return None

        atr = atr_pips(fac)
        atr_min = _env_float("RRL_ATR_MIN", 0.6)
        atr_max = _env_float("RRL_ATR_MAX", 2.8)
        atr_min = _tuned_float(("strategies", "RangeRevertLite", "atr_min"), atr_min)
        atr_max = _tuned_float(("strategies", "RangeRevertLite", "atr_max"), atr_max)
        if atr <= atr_min or atr > atr_max:
            return None

        adx_max = _env_float("RRL_ADX_MAX", 24.0)
        bbw_max = _env_float("RRL_BBW_MAX", 0.26)
        adx_max = _tuned_float(("strategies", "RangeRevertLite", "adx_max"), adx_max)
        bbw_max = _tuned_float(("strategies", "RangeRevertLite", "bbw_max"), bbw_max)
        if adx > adx_max:
            return None
        if bbw > bbw_max:
            return None

        macd_max = _env_float("RRL_MACD_HIST_MAX", 0.35)
        macd_max = _tuned_float(("strategies", "RangeRevertLite", "macd_hist_max"), macd_max)
        if abs(macd_hist) > macd_max:
            return None

        ema_slope_max = _env_float("RRL_EMA_SLOPE_MAX", 0.45)
        ema_slope_max = _tuned_float(("strategies", "RangeRevertLite", "ema_slope_max"), ema_slope_max)
        if abs(ema_slope) / PIP > ema_slope_max:
            return None

        levels = _bb_levels(fac)
        if not levels:
            return None
        upper, mid, lower, span_pips = levels
        dist_lower = price_delta_pips(close, lower)
        dist_upper = price_delta_pips(upper, close)

        band_dist = _env_float("RRL_BAND_DIST_PIPS", 2.0)
        band_ratio = _env_float("RRL_BAND_DIST_RATIO", 0.22)
        band_dist = _tuned_float(("strategies", "RangeRevertLite", "band_dist_pips"), band_dist)
        band_ratio = _tuned_float(("strategies", "RangeRevertLite", "band_dist_ratio"), band_ratio)
        band_threshold = max(band_dist, span_pips * band_ratio)

        rsi_long = _env_float("RRL_RSI_LONG", 35.0)
        rsi_short = _env_float("RRL_RSI_SHORT", 65.0)
        rsi_long = _tuned_float(("strategies", "RangeRevertLite", "rsi_long"), rsi_long)
        rsi_short = _tuned_float(("strategies", "RangeRevertLite", "rsi_short"), rsi_short)

        stoch_long = _env_float("RRL_STOCH_LONG", 0.20)
        stoch_short = _env_float("RRL_STOCH_SHORT", 0.80)
        stoch_long = _tuned_float(("strategies", "RangeRevertLite", "stoch_long"), stoch_long)
        stoch_short = _tuned_float(("strategies", "RangeRevertLite", "stoch_short"), stoch_short)

        vwap_gap_min = _env_float("RRL_VWAP_GAP_MIN", 0.9)
        vwap_gap_min = _tuned_float(("strategies", "RangeRevertLite", "vwap_gap_min"), vwap_gap_min)

        div_min = _env_float("RRL_DIV_MIN", 0.12)
        div_min = _tuned_float(("strategies", "RangeRevertLite", "div_min"), div_min)
        min_hits = _env_int("RRL_MIN_HITS", 2)
        min_hits = max(1, min(4, min_hits))

        candles = latest_candles(fac, 2)
        last_body = candle_body_pips(candles[-1]) if len(candles) >= 1 else None
        body_cap = _env_float("RRL_BODY_CAP", 1.4)
        body_cap = _tuned_float(("strategies", "RangeRevertLite", "body_cap"), body_cap)

        if dist_lower <= band_threshold:
            rsi_ok = rsi <= rsi_long
            stoch_ok = stoch_rsi <= stoch_long
            vwap_ok = vwap_gap <= -vwap_gap_min
            div_ok = div_score >= div_min
            hit_long = sum(1 for flag in (rsi_ok, stoch_ok, vwap_ok, div_ok) if flag)
            long_ok = hit_long >= min_hits
        else:
            long_ok = False
        if dist_upper <= band_threshold:
            rsi_ok = rsi >= rsi_short
            stoch_ok = stoch_rsi >= stoch_short
            vwap_ok = vwap_gap >= vwap_gap_min
            div_ok = div_score <= -div_min
            hit_short = sum(1 for flag in (rsi_ok, stoch_ok, vwap_ok, div_ok) if flag)
            short_ok = hit_short >= min_hits
        else:
            short_ok = False

        if long_ok and last_body is not None and last_body > body_cap:
            long_ok = False
        if short_ok and last_body is not None and last_body < -body_cap:
            short_ok = False

        if not long_ok and not short_ok:
            return None

        def _confidence_base() -> float:
            conf = 48.0
            conf += clamp(max(0.0, span_pips - 2.0), 0.0, 6.0) * 0.9
            conf += clamp(max(0.0, 2.3 - atr), 0.0, 1.6) * 3.2
            conf += clamp(max(0.0, band_threshold - min(dist_lower, dist_upper)), 0.0, 2.0) * 2.5
            conf -= clamp(abs(ema_slope) / PIP, 0.0, 0.6) * 1.8
            return conf

        sl_mult = _env_float("RRL_SL_MULT", 1.2)
        tp_mult = _env_float("RRL_TP_MULT", 0.9)
        sl_min = _env_float("RRL_SL_MIN", 1.2)
        sl_max = _env_float("RRL_SL_MAX", 2.8)
        tp_min = _env_float("RRL_TP_MIN", 0.9)
        tp_max = _env_float("RRL_TP_MAX", 2.2)

        sl = clamp(atr * sl_mult, sl_min, sl_max)
        tp = clamp(sl * tp_mult, tp_min, tp_max)

        if long_ok:
            conf = _confidence_base()
            conf += clamp(max(0.0, rsi_long - rsi), 0.0, 12.0) * 0.45
            conf += clamp(max(0.0, stoch_long - stoch_rsi), 0.0, 0.4) * 7.0
            conf += clamp(max(0.0, abs(vwap_gap) - vwap_gap_min), 0.0, 2.5) * 1.3
            if div_score > 0:
                conf += min(8.0, abs(div_score) * 10.0)
            if macd_hist > 0:
                conf += min(2.0, macd_hist * 6.0)
            confidence = int(clamp(conf, 42.0, 88.0))
            return {
                "action": "OPEN_LONG",
                "sl_pips": round(sl, 2),
                "tp_pips": round(tp, 2),
                "confidence": confidence,
                "tag": f"{RangeRevertLite.name}-long",
                "profile": "range_revert_lite",
                "notes": {
                    "rsi": round(rsi, 2),
                    "stoch": round(stoch_rsi, 2),
                    "bbw": round(bbw, 3),
                    "adx": round(adx, 2),
                    "dist_lower": round(dist_lower, 2),
                    "vwap_gap": round(vwap_gap, 2),
                    "div": round(div_score, 2),
                },
            }

        conf = _confidence_base()
        conf += clamp(max(0.0, rsi - rsi_short), 0.0, 12.0) * 0.45
        conf += clamp(max(0.0, stoch_rsi - stoch_short), 0.0, 0.4) * 7.0
        conf += clamp(max(0.0, abs(vwap_gap) - vwap_gap_min), 0.0, 2.5) * 1.3
        if div_score < 0:
            conf += min(8.0, abs(div_score) * 10.0)
        if macd_hist < 0:
            conf += min(2.0, abs(macd_hist) * 6.0)
        confidence = int(clamp(conf, 42.0, 88.0))
        return {
            "action": "OPEN_SHORT",
            "sl_pips": round(sl, 2),
            "tp_pips": round(tp, 2),
            "confidence": confidence,
            "tag": f"{RangeRevertLite.name}-short",
            "profile": "range_revert_lite",
            "notes": {
                "rsi": round(rsi, 2),
                "stoch": round(stoch_rsi, 2),
                "bbw": round(bbw, 3),
                "adx": round(adx, 2),
                "dist_upper": round(dist_upper, 2),
                "vwap_gap": round(vwap_gap, 2),
                "div": round(div_score, 2),
            },
        }

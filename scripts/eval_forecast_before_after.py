#!/usr/bin/env python3
"""Evaluate forecast technical formulas (before vs after) on the same period.

This script compares:
- `before`: technical forecast formula before trendline/support-resistance expansion
- `after`: current technical forecast formula

It also reports directional consistency of `breakout_bias_20` against realized future
move direction for each horizon step.

In addition, this script compares quantile-range quality:
- band coverage (realized move inside forecast upper/lower band)
- average band width (narrower is better if coverage is preserved)

`after` formula includes adaptive breakout-bias weighting that learns directional
skill from past samples only (no lookahead within each step evaluation loop).
"""

from __future__ import annotations

import argparse
from collections import deque
import glob
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Any
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.forecast_sklearn import compute_feature_frame

PIP = 0.01
TECH_SCORE_GAIN = 1.0
TECH_PROB_STRENGTH = 0.75
TECH_PROJECTION_WEIGHT = 0.38
DEFAULT_FEATURE_EXPANSION_GAIN = 0.35
DEFAULT_BREAKOUT_ADAPTIVE_WEIGHT = 0.22
DEFAULT_BREAKOUT_ADAPTIVE_WEIGHT_MAP = "1m=0.16,5m=0.22,10m=0.30"
DEFAULT_BREAKOUT_ADAPTIVE_MIN_SAMPLES = 80
DEFAULT_BREAKOUT_ADAPTIVE_LOOKBACK = 360
DEFAULT_SESSION_BIAS_WEIGHT = 0.12
DEFAULT_SESSION_BIAS_WEIGHT_MAP = "1m=0.0,5m=0.26,10m=0.38"
DEFAULT_SESSION_BIAS_MIN_SAMPLES = 24
DEFAULT_SESSION_BIAS_LOOKBACK = 720
DEFAULT_REBOUND_WEIGHT = 0.06
DEFAULT_REBOUND_WEIGHT_MAP = "1m=0.10,5m=0.04,10m=0.02"
RANGE_BAND_LOWER_Q = 0.20
RANGE_BAND_UPPER_Q = 0.80
RANGE_SIGMA_FLOOR_PIPS = 0.35
RANGE_NORMAL = NormalDist()

TECH_HORIZON_CFG = {
    "1m": {"trend_w": 0.70, "mr_w": 0.30, "temp": 0.9},
    "5m": {"trend_w": 0.40, "mr_w": 0.60, "temp": 1.0},
    "10m": {"trend_w": 0.40, "mr_w": 0.60, "temp": 1.04},
    "1h": {"trend_w": 0.56, "mr_w": 0.44, "temp": 1.35},
    "8h": {"trend_w": 0.68, "mr_w": 0.32, "temp": 1.25},
    "1d": {"trend_w": 0.76, "mr_w": 0.24, "temp": 1.10},
    "1w": {"trend_w": 0.82, "mr_w": 0.18, "temp": 1.00},
}


@dataclass(frozen=True)
class EvalRow:
    step: int
    horizon: str
    n: int
    hit_before: float
    hit_after: float
    mae_before: float
    mae_after: float
    hit_delta: float
    mae_delta: float
    breakout_hit: float
    breakout_hit_filtered: float
    breakout_filtered_coverage: float
    range_coverage_before: float
    range_coverage_after: float
    range_coverage_delta: float
    range_width_before: float
    range_width_after: float
    range_width_delta: float


@dataclass(frozen=True)
class StepPrediction:
    p_up: float
    expected_pips: float
    range_low_pips: float
    range_high_pips: float


@dataclass(frozen=True)
class Summary:
    bars: int
    from_ts: str
    to_ts: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sigmoid(x: float) -> float:
    x = _clamp(float(x), -40.0, 40.0)
    return 1.0 / (1.0 + math.exp(-x))


def _estimate_directional_skill(
    *,
    signal_values: list[float],
    target_values: list[float],
    min_samples: int,
    lookback: int,
) -> tuple[float, float, int]:
    min_samples = max(1, int(min_samples))
    usable: list[tuple[float, float]] = []
    for signal, target in zip(signal_values, target_values):
        s = _safe_float(signal, math.nan)
        t = _safe_float(target, math.nan)
        if not math.isfinite(s) or not math.isfinite(t):
            continue
        if abs(s) < 1e-12 or abs(t) < 1e-12:
            continue
        usable.append((float(s), float(t)))
    if lookback > 0 and len(usable) > lookback:
        usable = usable[-lookback:]
    count = len(usable)
    if count < min_samples:
        return 0.0, 0.5, count
    hits = sum(1 for signal, target in usable if signal * target > 0.0)
    hit_rate = hits / float(count)
    skill = _clamp((hit_rate - 0.5) * 2.0, -1.0, 1.0)
    confidence = _clamp(count / max(float(min_samples * 2), 1.0), 0.0, 1.0)
    return float(skill * confidence), float(hit_rate), int(count)


def _extract_jst_hour(value: Any) -> Optional[int]:
    ts: Optional[pd.Timestamp]
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int((int(ts.hour) + 9) % 24)


def _estimate_session_hour_bias(
    *,
    timestamp_values: list[Any],
    target_values: list[float],
    current_timestamp: Any = None,
    min_samples: int,
    lookback: int,
) -> tuple[float, float, int, Optional[int]]:
    min_samples = max(1, int(min_samples))
    usable: list[tuple[int, float]] = []
    for ts_value, target in zip(timestamp_values, target_values):
        hour = _extract_jst_hour(ts_value)
        move = _safe_float(target, math.nan)
        if hour is None or not math.isfinite(move):
            continue
        if abs(move) < 1e-12:
            continue
        usable.append((int(hour), float(move)))
    if lookback > 0 and len(usable) > lookback:
        usable = usable[-lookback:]
    if not usable:
        return 0.0, 0.0, 0, None

    current_hour = _extract_jst_hour(current_timestamp)
    if current_hour is None:
        current_hour = usable[-1][0]

    hour_moves = [move for hour, move in usable if hour == current_hour]
    sample_count = len(hour_moves)
    if sample_count <= 0:
        return 0.0, 0.0, 0, current_hour
    mean_move = sum(hour_moves) / float(sample_count)
    if sample_count < min_samples:
        return 0.0, float(mean_move), sample_count, current_hour

    abs_all = sorted(abs(move) for _, move in usable)
    median_abs = abs_all[len(abs_all) // 2] if abs_all else 0.0
    scale = max(0.35, float(median_abs))
    raw_bias = _clamp(mean_move / scale, -1.0, 1.0)
    confidence = _clamp(sample_count / max(float(min_samples * 3), 1.0), 0.0, 1.0)
    return float(raw_bias * confidence), float(mean_move), sample_count, current_hour


def _parse_horizon_weight_map(raw: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    text = str(raw or "").strip()
    if not text:
        return out
    for token in text.split(","):
        part = str(token).strip()
        if not part or "=" not in part:
            continue
        key_raw, value_raw = part.split("=", 1)
        key = str(key_raw).strip().lower()
        if not key:
            continue
        try:
            value = float(value_raw)
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        out[key] = _clamp(value, 0.0, 1.0)
    return out


def _normalized_breakout_bias(row: pd.Series) -> float:
    atr = max(0.45, abs(_safe_float(row.get("atr_pips_14"), 0.0)))
    breakout_up = _safe_float(row.get("breakout_up_pips_20"), 0.0) / max(0.70, atr)
    breakout_down = _safe_float(row.get("breakout_down_pips_20"), 0.0) / max(0.70, atr)
    return float(breakout_up - breakout_down)


def _extract_ohlc(row: pd.Series) -> Optional[tuple[float, float, float, float]]:
    o = _safe_float(row.get("open"), math.nan)
    h = _safe_float(row.get("high"), math.nan)
    l = _safe_float(row.get("low"), math.nan)
    c = _safe_float(row.get("close"), math.nan)
    if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l) and math.isfinite(c)):
        return None
    if h < l:
        h, l = l, h
    if h <= l:
        h = l + 1e-6
    return float(o), float(h), float(l), float(c)


def _rebound_bias_signal(
    *,
    ret1: float,
    ret3: float,
    ret12: float,
    rsi: float,
    range_pos: float,
    sr_balance: float,
    trend_accel: float,
    trend_pullback: float,
    breakout_bias: float,
    trend_strength: float,
    ohlc: Optional[tuple[float, float, float, float]],
) -> float:
    drop_score = _clamp(
        0.44 * _clamp(math.tanh(-ret1 * 1.35), 0.0, 1.0)
        + 0.34 * _clamp(math.tanh(-ret3 * 1.05), 0.0, 1.0)
        + 0.22 * _clamp(math.tanh(-ret12 * 0.72), 0.0, 1.0),
        0.0,
        1.0,
    )
    oversold_score = _clamp(
        0.40 * _clamp(math.tanh(-rsi * 1.15), 0.0, 1.0)
        + 0.32 * _clamp(math.tanh(-range_pos * 1.25), 0.0, 1.0)
        + 0.28 * _clamp(math.tanh(-sr_balance * 1.10), 0.0, 1.0),
        0.0,
        1.0,
    )
    decel_score = _clamp(
        0.58 * _clamp(math.tanh(trend_accel * 1.20), 0.0, 1.0)
        + 0.42 * _clamp(math.tanh(-trend_pullback * 0.95), 0.0, 1.0),
        0.0,
        1.0,
    )
    wick_score = 0.0
    if ohlc is not None:
        o, h, l, c = ohlc
        span = max(1e-6, h - l)
        lower_wick = max(0.0, min(o, c) - l)
        upper_wick = max(0.0, h - max(o, c))
        body = abs(c - o)
        lower_ratio = lower_wick / span
        upper_ratio = upper_wick / span
        close_pos = _clamp((c - l) / span, 0.0, 1.0)
        body_ratio = _clamp(body / span, 0.0, 1.0)
        wick_score = _clamp(
            0.62 * max(lower_ratio - upper_ratio, 0.0)
            + 0.28 * _clamp((close_pos - 0.52) / 0.48, 0.0, 1.0)
            + 0.10 * _clamp((0.40 - body_ratio) / 0.40, 0.0, 1.0),
            0.0,
            1.0,
        )
    breakout_drag = _clamp(math.tanh(max(0.0, -breakout_bias) * 1.05), 0.0, 1.0)
    trend_drag = _clamp((trend_strength - 0.72) / 0.28, 0.0, 1.0)
    core = _clamp(
        0.42 * oversold_score + 0.36 * decel_score + 0.22 * wick_score,
        0.0,
        1.0,
    )
    rebound_signal = drop_score * core
    rebound_signal *= (1.0 - 0.42 * breakout_drag)
    rebound_signal *= (1.0 - 0.48 * trend_drag)
    if drop_score < 0.20:
        rebound_signal *= 0.35
    if oversold_score < 0.15:
        rebound_signal *= 0.50
    return _clamp(rebound_signal, 0.0, 1.0)


def _normal_quantile_z(quantile: float) -> float:
    q = _clamp(float(quantile), 0.01, 0.99)
    try:
        return float(RANGE_NORMAL.inv_cdf(q))
    except Exception:
        return 0.0


def _build_range_band(
    *,
    expected_pips: float,
    expected_move: float,
    projection_confidence: float,
) -> tuple[float, float]:
    sigma = max(
        RANGE_SIGMA_FLOOR_PIPS,
        float(expected_move) * (0.62 + 0.38 * (1.0 - _clamp(float(projection_confidence), 0.0, 1.0))),
    )
    low = float(expected_pips) + _normal_quantile_z(RANGE_BAND_LOWER_Q) * sigma
    high = float(expected_pips) + _normal_quantile_z(RANGE_BAND_UPPER_Q) * sigma
    if low > high:
        low, high = high, low
    return float(low), float(high)


def _parse_ts(value: Any) -> Optional[pd.Timestamp]:
    text = str(value or "").strip()
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _to_datetime_utc(values: Any) -> Any:
    """Parse datetimes robustly for mixed ISO8601 precision inputs.

    Some live feeds emit timestamps with fractional seconds while others do not.
    Pandas may coerce one format to NaT when inferring from a mixed series.
    """
    try:
        return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, utc=True, errors="coerce")


def _extract_price(candle: dict[str, Any], key: str) -> Optional[float]:
    if key in candle:
        return _safe_float(candle.get(key), math.nan)
    mid = candle.get("mid")
    if isinstance(mid, dict):
        return _safe_float(mid.get(key[0]), math.nan)
    return None


def _normalize_candle(candle: dict[str, Any]) -> Optional[dict[str, Any]]:
    ts = _parse_ts(candle.get("time") or candle.get("timestamp") or candle.get("ts"))
    if ts is None:
        return None

    def _pick(*keys: str) -> Optional[float]:
        for key in keys:
            if key in candle:
                value = _safe_float(candle.get(key), math.nan)
                if math.isfinite(value):
                    return value
        mid = candle.get("mid")
        if isinstance(mid, dict):
            for k in keys:
                mkey = k[0]
                value = _safe_float(mid.get(mkey), math.nan)
                if math.isfinite(value):
                    return value
        return None

    o = _pick("open", "o")
    h = _pick("high", "h")
    l = _pick("low", "l")
    c = _pick("close", "c")
    if None in {o, h, l, c}:
        return None
    return {
        "timestamp": ts.isoformat(),
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
    }


def _load_candles(patterns: list[str]) -> list[dict[str, Any]]:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))

    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        candles = payload.get("candles") if isinstance(payload, dict) else None
        if not isinstance(candles, list):
            continue
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            normalized = _normalize_candle(candle)
            if normalized is not None:
                rows.append(normalized)

    if not rows:
        return []

    # Keep the last candle for each timestamp.
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup[str(row["timestamp"])] = row
    out = sorted(dedup.values(), key=lambda r: str(r["timestamp"]))
    return out


def _horizon_from_step(step: int) -> str:
    if step == 1:
        return "1m"
    if step == 5:
        return "5m"
    if step == 10:
        return "10m"
    if step == 12:
        return "1h"
    if step >= 24:
        return "1d"
    return "8h"


def _prediction_before(
    row: pd.Series,
    *,
    step: int,
    sample_count: int,
    projection_score: float = 0.0,
    projection_confidence: float = 0.0,
) -> StepPrediction:
    atr = max(0.45, abs(_safe_float(row.get("atr_pips_14"), 0.0)))
    vol = max(0.2, abs(_safe_float(row.get("vol_pips_20"), 0.0)))
    ret1 = _safe_float(row.get("ret_pips_1"), 0.0) / atr
    ret3 = _safe_float(row.get("ret_pips_3"), 0.0) / atr
    ret12 = _safe_float(row.get("ret_pips_12"), 0.0) / max(0.7, atr * 1.2)
    ma_gap = _safe_float(row.get("ma_gap_pips_10_20"), 0.0) / atr
    close_ma20 = _safe_float(row.get("close_ma20_pips"), 0.0) / max(0.6, atr)
    close_ma50 = _safe_float(row.get("close_ma50_pips"), 0.0) / max(0.8, atr)
    rsi = (_safe_float(row.get("rsi_14"), 50.0) - 50.0) / 16.0
    range_pos = (_safe_float(row.get("range_pos"), 0.5) - 0.5) * 2.0

    trend_score = (
        0.82 * math.tanh(ret1 * 1.2)
        + 1.02 * math.tanh(ret3 * 1.0)
        + 0.68 * math.tanh(ret12 * 0.7)
        + 0.74 * math.tanh(ma_gap * 1.1)
        + 0.46 * math.tanh(close_ma20 * 1.0)
        + 0.33 * math.tanh(close_ma50 * 0.9)
        + 0.24 * math.tanh(rsi)
    )
    mean_revert_score = (
        -0.68 * math.tanh(close_ma20 * 1.2)
        - 0.58 * math.tanh(range_pos * 1.4)
        - 0.42 * math.tanh(rsi * 1.1)
    )

    trend_boost = 0.0
    range_boost = 0.0
    trend_strength = _clamp(
        0.18
        + 0.44 * abs(math.tanh(ma_gap * 1.4))
        + 0.33 * abs(math.tanh(ret3 * 1.1))
        + 0.18 * _clamp((vol / max(atr, 1e-6)) - 0.55, 0.0, 1.0),
        0.0,
        1.0,
    )
    trend_strength = _clamp(
        trend_strength + trend_boost - 0.35 * range_boost + 0.05 * abs(projection_score),
        0.0,
        1.0,
    )
    range_pressure = _clamp(1.0 - trend_strength, 0.0, 1.0)
    range_pressure = _clamp(range_pressure + 0.45 * range_boost - 0.15 * trend_boost, 0.0, 1.0)

    horizon = _horizon_from_step(step)
    cfg = TECH_HORIZON_CFG.get(horizon, TECH_HORIZON_CFG["8h"])
    combo = (
        cfg["trend_w"] * trend_score * (1.0 - 0.55 * range_pressure)
        + cfg["mr_w"] * mean_revert_score * range_pressure
        + TECH_PROJECTION_WEIGHT * projection_score
    )
    raw_prob = _sigmoid(float(cfg["temp"]) * TECH_SCORE_GAIN * combo)

    sample_strength = _clamp((sample_count - 40.0) / 120.0, 0.25, 1.0)
    damp = TECH_PROB_STRENGTH * sample_strength
    p_up = _clamp(0.5 + (raw_prob - 0.5) * damp, 0.02, 0.98)

    step_scale = math.sqrt(max(1.0, float(step)) / 12.0)
    expected_move = max(0.35, vol * step_scale * (0.92 + 0.25 * projection_confidence))
    expected_pips = (p_up - 0.5) * 2.0 * expected_move
    range_low_pips, range_high_pips = _build_range_band(
        expected_pips=expected_pips,
        expected_move=expected_move,
        projection_confidence=projection_confidence,
    )
    return StepPrediction(
        p_up=float(p_up),
        expected_pips=float(expected_pips),
        range_low_pips=float(range_low_pips),
        range_high_pips=float(range_high_pips),
    )


def _prediction_after(
    row: pd.Series,
    *,
    step: int,
    sample_count: int,
    feature_expansion_gain: float,
    breakout_skill: float = 0.0,
    breakout_adaptive_weight: float = 0.0,
    session_bias: float = 0.0,
    session_bias_weight: float = 0.0,
    rebound_weight: float = 0.0,
    projection_score: float = 0.0,
    projection_confidence: float = 0.0,
) -> StepPrediction:
    atr = max(0.45, abs(_safe_float(row.get("atr_pips_14"), 0.0)))
    vol = max(0.2, abs(_safe_float(row.get("vol_pips_20"), 0.0)))
    ret1 = _safe_float(row.get("ret_pips_1"), 0.0) / atr
    ret3 = _safe_float(row.get("ret_pips_3"), 0.0) / atr
    ret12 = _safe_float(row.get("ret_pips_12"), 0.0) / max(0.7, atr * 1.2)
    ma_gap = _safe_float(row.get("ma_gap_pips_10_20"), 0.0) / atr
    close_ma20 = _safe_float(row.get("close_ma20_pips"), 0.0) / max(0.6, atr)
    close_ma50 = _safe_float(row.get("close_ma50_pips"), 0.0) / max(0.8, atr)
    rsi = (_safe_float(row.get("rsi_14"), 50.0) - 50.0) / 16.0
    range_pos = (_safe_float(row.get("range_pos"), 0.5) - 0.5) * 2.0

    trend_slope20 = _safe_float(row.get("trend_slope_pips_20"), 0.0) / max(0.55, atr * 0.42)
    trend_slope50 = _safe_float(row.get("trend_slope_pips_50"), 0.0) / max(0.55, atr * 0.46)
    trend_accel = _safe_float(row.get("trend_accel_pips"), 0.0) / max(0.40, atr * 0.34)
    sr_balance = _clamp(_safe_float(row.get("sr_balance_20"), 0.0), -1.0, 1.0)
    breakout_up = _safe_float(row.get("breakout_up_pips_20"), 0.0) / max(0.70, atr)
    breakout_down = _safe_float(row.get("breakout_down_pips_20"), 0.0) / max(0.70, atr)
    breakout_bias = breakout_up - breakout_down
    breakout_adaptive = math.tanh(breakout_bias * 0.9) * _clamp(float(breakout_skill), -1.0, 1.0)
    donchian_width = max(0.25, abs(_safe_float(row.get("donchian_width_pips_20"), 0.0)))
    range_compression = _safe_float(row.get("range_compression_20"), 0.0)
    trend_pullback = _safe_float(row.get("trend_pullback_norm_20"), 0.0)
    width_ratio = donchian_width / max(0.45, atr)
    squeeze_score = _clamp(1.0 - (width_ratio / 8.0), 0.0, 1.0)

    trend_score = (
        0.82 * math.tanh(ret1 * 1.2)
        + 1.02 * math.tanh(ret3 * 1.0)
        + 0.68 * math.tanh(ret12 * 0.7)
        + 0.74 * math.tanh(ma_gap * 1.1)
        + 0.46 * math.tanh(close_ma20 * 1.0)
        + 0.33 * math.tanh(close_ma50 * 0.9)
        + 0.24 * math.tanh(rsi)
        + feature_expansion_gain
        * (
            0.58 * math.tanh(trend_slope20 * 1.05)
            + 0.34 * math.tanh(trend_slope50 * 0.95)
            + 0.22 * math.tanh(trend_accel * 1.2)
            + 0.30 * math.tanh(breakout_bias * 0.85)
            + 0.18 * math.tanh(sr_balance * 0.9)
            + 0.16 * math.tanh(trend_pullback * 0.8)
        )
    )
    mean_revert_score = (
        -0.68 * math.tanh(close_ma20 * 1.2)
        - 0.58 * math.tanh(range_pos * 1.4)
        - 0.42 * math.tanh(rsi * 1.1)
        + feature_expansion_gain
        * (
            -0.44 * math.tanh(sr_balance * 1.2)
            - 0.26 * math.tanh(breakout_bias * 0.9)
            - 0.20 * math.tanh(trend_pullback * 0.95)
            + 0.22 * squeeze_score
            + 0.08 * math.tanh((0.20 - range_compression) * 3.5)
        )
    )

    trend_boost = 0.0
    range_boost = 0.0
    trend_strength = _clamp(
        0.18
        + 0.44 * abs(math.tanh(ma_gap * 1.4))
        + 0.33 * abs(math.tanh(ret3 * 1.1))
        + feature_expansion_gain
        * (
            0.18 * abs(math.tanh(trend_slope20 * 1.0))
            + 0.10 * abs(math.tanh(trend_slope50 * 1.0))
            + 0.09 * abs(math.tanh(breakout_bias * 0.85))
        )
        + 0.18 * _clamp((vol / max(atr, 1e-6)) - 0.55, 0.0, 1.0),
        0.0,
        1.0,
    )
    trend_strength = _clamp(
        trend_strength + trend_boost - 0.35 * range_boost + 0.05 * abs(projection_score),
        0.0,
        1.0,
    )
    range_pressure = _clamp(1.0 - trend_strength, 0.0, 1.0)
    range_pressure = _clamp(
        range_pressure
        + 0.45 * range_boost
        - 0.15 * trend_boost
        + feature_expansion_gain
        * (
            0.16 * squeeze_score
            + 0.10 * _clamp(1.0 - abs(sr_balance), 0.0, 1.0)
            - 0.12 * abs(math.tanh(breakout_bias * 0.8))
        ),
        0.0,
        1.0,
    )

    horizon = _horizon_from_step(step)
    cfg = TECH_HORIZON_CFG.get(horizon, TECH_HORIZON_CFG["8h"])
    session_weight = (
        0.0
        if str(horizon).strip().lower() == "1m"
        else _clamp(float(session_bias_weight), 0.0, 1.0)
    )
    rebound_signal = 0.0
    rebound_weight_eff = _clamp(float(rebound_weight), 0.0, 1.0)
    if rebound_weight_eff > 0.0:
        rebound_signal = _rebound_bias_signal(
            ret1=ret1,
            ret3=ret3,
            ret12=ret12,
            rsi=rsi,
            range_pos=range_pos,
            sr_balance=sr_balance,
            trend_accel=trend_accel,
            trend_pullback=trend_pullback,
            breakout_bias=breakout_bias,
            trend_strength=trend_strength,
            ohlc=_extract_ohlc(row),
        )
    combo = (
        cfg["trend_w"] * trend_score * (1.0 - 0.55 * range_pressure)
        + cfg["mr_w"] * mean_revert_score * range_pressure
        + TECH_PROJECTION_WEIGHT * projection_score
        + _clamp(float(breakout_adaptive_weight), 0.0, 1.0) * breakout_adaptive
        + session_weight * _clamp(float(session_bias), -1.0, 1.0)
        + rebound_weight_eff * rebound_signal
    )
    raw_prob = _sigmoid(float(cfg["temp"]) * TECH_SCORE_GAIN * combo)

    sample_strength = _clamp((sample_count - 40.0) / 120.0, 0.25, 1.0)
    damp = TECH_PROB_STRENGTH * sample_strength
    p_up = _clamp(0.5 + (raw_prob - 0.5) * damp, 0.02, 0.98)

    step_scale = math.sqrt(max(1.0, float(step)) / 12.0)
    expected_move = max(0.35, vol * step_scale * (0.92 + 0.25 * projection_confidence))
    expected_pips = (p_up - 0.5) * 2.0 * expected_move
    range_low_pips, range_high_pips = _build_range_band(
        expected_pips=expected_pips,
        expected_move=expected_move,
        projection_confidence=projection_confidence,
    )
    return StepPrediction(
        p_up=float(p_up),
        expected_pips=float(expected_pips),
        range_low_pips=float(range_low_pips),
        range_high_pips=float(range_high_pips),
    )


def _evaluate_step(
    *,
    candles_df: pd.DataFrame,
    feats_df: pd.DataFrame,
    step: int,
    min_abs_breakout_bias: float,
    feature_expansion_gain: float,
    breakout_adaptive_weight: float,
    breakout_adaptive_weight_map: dict[str, float],
    breakout_adaptive_min_samples: int,
    breakout_adaptive_lookback: int,
    session_bias_weight: float,
    session_bias_weight_map: dict[str, float],
    session_bias_min_samples: int,
    session_bias_lookback: int,
    rebound_weight: float,
    rebound_weight_map: dict[str, float],
) -> EvalRow:
    merged = feats_df.join(candles_df[["open", "high", "low", "close"]], how="inner")
    merged["future_pips"] = (merged["close"].shift(-step) - merged["close"]) / PIP

    required = [
        "future_pips",
        "atr_pips_14",
        "vol_pips_20",
        "ret_pips_1",
        "ret_pips_3",
        "ret_pips_12",
        "ma_gap_pips_10_20",
        "close_ma20_pips",
        "close_ma50_pips",
        "rsi_14",
        "range_pos",
        "trend_slope_pips_20",
        "trend_slope_pips_50",
        "trend_accel_pips",
        "sr_balance_20",
        "breakout_up_pips_20",
        "breakout_down_pips_20",
        "donchian_width_pips_20",
        "range_compression_20",
        "trend_pullback_norm_20",
    ]
    merged = merged.dropna(subset=required)

    horizon_name = _horizon_from_step(step)
    breakout_adaptive_weight_eff = _clamp(
        float(breakout_adaptive_weight_map.get(horizon_name, breakout_adaptive_weight)),
        0.0,
        1.0,
    )
    session_bias_weight_eff = _clamp(
        float(session_bias_weight_map.get(horizon_name, session_bias_weight)),
        0.0,
        1.0,
    )
    rebound_weight_eff = _clamp(
        float(rebound_weight_map.get(horizon_name, rebound_weight)),
        0.0,
        1.0,
    )

    hit_before: list[int] = []
    hit_after: list[int] = []
    mae_before: list[float] = []
    mae_after: list[float] = []
    range_coverage_before: list[int] = []
    range_coverage_after: list[int] = []
    range_width_before: list[float] = []
    range_width_after: list[float] = []
    breakout_hits: list[int] = []
    breakout_filtered_hits: list[int] = []
    breakout_filtered_count = 0
    breakout_signal_hist: list[float] = []
    realized_hist: list[float] = []
    session_window: deque[tuple[int, float]] = deque()
    session_hour_sum = [0.0] * 24
    session_hour_count = [0] * 24
    session_abs_sum = 0.0
    session_total_count = 0

    for i, (_, row) in enumerate(merged.iterrows(), start=1):
        realized = float(row["future_pips"])
        if abs(realized) < 1e-12:
            continue

        pred_before = _prediction_before(row, step=step, sample_count=i)
        breakout_skill = 0.0
        if breakout_adaptive_weight_eff > 0.0:
            signal_hist = breakout_signal_hist
            target_hist = realized_hist
            if breakout_adaptive_lookback > 0 and len(breakout_signal_hist) > breakout_adaptive_lookback:
                signal_hist = breakout_signal_hist[-breakout_adaptive_lookback:]
                target_hist = realized_hist[-breakout_adaptive_lookback:]
            breakout_skill, _, _ = _estimate_directional_skill(
                signal_values=signal_hist,
                target_values=target_hist,
                min_samples=breakout_adaptive_min_samples,
                lookback=0,
            )
        session_bias = 0.0
        if session_bias_weight_eff > 0.0:
            current_hour = _extract_jst_hour(row.name)
            if current_hour is not None:
                hour_count = session_hour_count[current_hour]
                if hour_count >= session_bias_min_samples:
                    mean_move = session_hour_sum[current_hour] / float(hour_count)
                    scale = max(
                        0.35,
                        session_abs_sum / float(max(session_total_count, 1)),
                    )
                    raw = _clamp(mean_move / scale, -1.0, 1.0)
                    confidence = _clamp(
                        hour_count / max(float(session_bias_min_samples * 3), 1.0),
                        0.0,
                        1.0,
                    )
                    session_bias = float(raw * confidence)
        pred_after = _prediction_after(
            row,
            step=step,
            sample_count=i,
            feature_expansion_gain=feature_expansion_gain,
            breakout_skill=breakout_skill,
            breakout_adaptive_weight=breakout_adaptive_weight_eff,
            session_bias=session_bias,
            session_bias_weight=session_bias_weight_eff,
            rebound_weight=rebound_weight_eff,
        )
        expected_before = pred_before.expected_pips
        expected_after = pred_after.expected_pips

        hit_before.append(1 if expected_before * realized > 0.0 else 0)
        hit_after.append(1 if expected_after * realized > 0.0 else 0)
        mae_before.append(abs(expected_before - realized))
        mae_after.append(abs(expected_after - realized))
        range_coverage_before.append(
            1 if pred_before.range_low_pips <= realized <= pred_before.range_high_pips else 0
        )
        range_coverage_after.append(
            1 if pred_after.range_low_pips <= realized <= pred_after.range_high_pips else 0
        )
        range_width_before.append(pred_before.range_high_pips - pred_before.range_low_pips)
        range_width_after.append(pred_after.range_high_pips - pred_after.range_low_pips)

        breakout_bias = float(row["breakout_up_pips_20"] - row["breakout_down_pips_20"])
        breakout_hits.append(1 if breakout_bias * realized > 0.0 else 0)
        if abs(breakout_bias) >= float(min_abs_breakout_bias):
            breakout_filtered_count += 1
            breakout_filtered_hits.append(1 if breakout_bias * realized > 0.0 else 0)
        breakout_signal_hist.append(_normalized_breakout_bias(row))
        realized_hist.append(realized)
        if session_bias_weight_eff > 0.0:
            hist_hour = _extract_jst_hour(row.name)
            if hist_hour is not None:
                session_window.append((hist_hour, realized))
                session_hour_sum[hist_hour] += realized
                session_hour_count[hist_hour] += 1
                session_abs_sum += abs(realized)
                session_total_count += 1
                if session_bias_lookback > 0:
                    while len(session_window) > session_bias_lookback:
                        old_hour, old_move = session_window.popleft()
                        session_hour_sum[old_hour] -= old_move
                        session_hour_count[old_hour] = max(
                            0,
                            session_hour_count[old_hour] - 1,
                        )
                        session_abs_sum = max(0.0, session_abs_sum - abs(old_move))
                        session_total_count = max(0, session_total_count - 1)

    n = len(hit_before)
    if n <= 0:
        return EvalRow(
            step=step,
            horizon=_horizon_from_step(step),
            n=0,
            hit_before=0.0,
            hit_after=0.0,
            mae_before=0.0,
            mae_after=0.0,
            hit_delta=0.0,
            mae_delta=0.0,
            breakout_hit=0.0,
            breakout_hit_filtered=0.0,
            breakout_filtered_coverage=0.0,
            range_coverage_before=0.0,
            range_coverage_after=0.0,
            range_coverage_delta=0.0,
            range_width_before=0.0,
            range_width_after=0.0,
            range_width_delta=0.0,
        )

    hit_b = sum(hit_before) / float(n)
    hit_a = sum(hit_after) / float(n)
    mae_b = sum(mae_before) / float(n)
    mae_a = sum(mae_after) / float(n)
    breakout_hit = sum(breakout_hits) / float(len(breakout_hits)) if breakout_hits else 0.0
    breakout_hit_filtered = (
        sum(breakout_filtered_hits) / float(len(breakout_filtered_hits))
        if breakout_filtered_hits
        else 0.0
    )
    coverage = breakout_filtered_count / float(n)
    range_cov_before = sum(range_coverage_before) / float(len(range_coverage_before))
    range_cov_after = sum(range_coverage_after) / float(len(range_coverage_after))
    range_width_b = sum(range_width_before) / float(len(range_width_before))
    range_width_a = sum(range_width_after) / float(len(range_width_after))

    return EvalRow(
        step=step,
        horizon=horizon_name,
        n=n,
        hit_before=hit_b,
        hit_after=hit_a,
        mae_before=mae_b,
        mae_after=mae_a,
        hit_delta=hit_a - hit_b,
        mae_delta=mae_a - mae_b,
        breakout_hit=breakout_hit,
        breakout_hit_filtered=breakout_hit_filtered,
        breakout_filtered_coverage=coverage,
        range_coverage_before=range_cov_before,
        range_coverage_after=range_cov_after,
        range_coverage_delta=range_cov_after - range_cov_before,
        range_width_before=range_width_b,
        range_width_after=range_width_a,
        range_width_delta=range_width_a - range_width_b,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--patterns",
        default="logs/candles_M1*.json,logs/candles_USDJPY_M1*.json,logs/oanda/candles_M1_latest.json",
        help="Comma-separated glob patterns for M1 candle JSON files.",
    )
    ap.add_argument("--steps", default="1,5,10", help="Comma-separated step bars.")
    ap.add_argument(
        "--max-bars",
        type=int,
        default=0,
        help="If >0, keep only the latest N bars before evaluation.",
    )
    ap.add_argument(
        "--min-abs-breakout-bias",
        type=float,
        default=0.2,
        help="Minimum abs breakout_bias_20 to count as filtered signal.",
    )
    ap.add_argument(
        "--feature-expansion-gain",
        type=float,
        default=DEFAULT_FEATURE_EXPANSION_GAIN,
        help="Gain for added trendline/support-resistance terms in after formula.",
    )
    ap.add_argument(
        "--breakout-adaptive-weight",
        type=float,
        default=DEFAULT_BREAKOUT_ADAPTIVE_WEIGHT,
        help="Weight for adaptive breakout_bias directional term in after formula.",
    )
    ap.add_argument(
        "--breakout-adaptive-weight-map",
        default=DEFAULT_BREAKOUT_ADAPTIVE_WEIGHT_MAP,
        help="Comma-separated horizon weights, e.g. '1m=0.16,5m=0.22,10m=0.30'.",
    )
    ap.add_argument(
        "--breakout-adaptive-min-samples",
        type=int,
        default=DEFAULT_BREAKOUT_ADAPTIVE_MIN_SAMPLES,
        help="Minimum historical samples required to activate breakout adaptive skill.",
    )
    ap.add_argument(
        "--breakout-adaptive-lookback",
        type=int,
        default=DEFAULT_BREAKOUT_ADAPTIVE_LOOKBACK,
        help="Lookback window for breakout adaptive skill estimation.",
    )
    ap.add_argument(
        "--session-bias-weight",
        type=float,
        default=DEFAULT_SESSION_BIAS_WEIGHT,
        help="Weight for JST hour-of-day directional bias term in after formula.",
    )
    ap.add_argument(
        "--session-bias-weight-map",
        default=DEFAULT_SESSION_BIAS_WEIGHT_MAP,
        help="Comma-separated horizon weights, e.g. '1m=0.0,5m=0.14,10m=0.30'.",
    )
    ap.add_argument(
        "--session-bias-min-samples",
        type=int,
        default=DEFAULT_SESSION_BIAS_MIN_SAMPLES,
        help="Minimum per-hour samples required for JST session bias activation.",
    )
    ap.add_argument(
        "--session-bias-lookback",
        type=int,
        default=DEFAULT_SESSION_BIAS_LOOKBACK,
        help="Lookback window for JST session bias estimation.",
    )
    ap.add_argument(
        "--rebound-weight",
        type=float,
        default=DEFAULT_REBOUND_WEIGHT,
        help="Weight for rebound signal term in after formula.",
    )
    ap.add_argument(
        "--rebound-weight-map",
        default=DEFAULT_REBOUND_WEIGHT_MAP,
        help="Comma-separated horizon weights, e.g. '1m=0.10,5m=0.04,10m=0.02'.",
    )
    ap.add_argument("--json-out", default="", help="Optional output JSON path.")
    args = ap.parse_args()

    patterns = [token.strip() for token in str(args.patterns).split(",") if token.strip()]
    candles = _load_candles(patterns)
    if not candles:
        print("no candles loaded")
        return 1

    if args.max_bars and args.max_bars > 0:
        candles = candles[-int(args.max_bars) :]

    candles_df = pd.DataFrame(candles)
    candles_df["timestamp"] = _to_datetime_utc(candles_df["timestamp"])
    candles_df = candles_df.dropna(subset=["timestamp", "close"]).set_index("timestamp")
    candles_df = candles_df[~candles_df.index.duplicated(keep="last")]
    candles_df = candles_df.sort_index()

    feats_df = compute_feature_frame(candles)
    if feats_df.empty:
        print("feature frame empty")
        return 1

    steps: list[int] = []
    for token in [t.strip() for t in str(args.steps).split(",") if t.strip()]:
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0:
            steps.append(value)
    steps = sorted(set(steps))
    if not steps:
        print("no valid steps")
        return 1

    feature_expansion_gain = _clamp(float(args.feature_expansion_gain), 0.0, 1.0)
    breakout_adaptive_weight = _clamp(float(args.breakout_adaptive_weight), 0.0, 1.0)
    breakout_adaptive_weight_map = _parse_horizon_weight_map(args.breakout_adaptive_weight_map)
    breakout_adaptive_min_samples = max(1, int(args.breakout_adaptive_min_samples))
    breakout_adaptive_lookback = max(
        breakout_adaptive_min_samples,
        int(args.breakout_adaptive_lookback),
    )
    session_bias_weight = _clamp(float(args.session_bias_weight), 0.0, 1.0)
    session_bias_weight_map = _parse_horizon_weight_map(args.session_bias_weight_map)
    session_bias_min_samples = max(1, int(args.session_bias_min_samples))
    session_bias_lookback = max(
        session_bias_min_samples,
        int(args.session_bias_lookback),
    )
    rebound_weight = _clamp(float(args.rebound_weight), 0.0, 1.0)
    rebound_weight_map = _parse_horizon_weight_map(args.rebound_weight_map)

    rows = [
        _evaluate_step(
            candles_df=candles_df,
            feats_df=feats_df,
            step=step,
            min_abs_breakout_bias=float(args.min_abs_breakout_bias),
            feature_expansion_gain=feature_expansion_gain,
            breakout_adaptive_weight=breakout_adaptive_weight,
            breakout_adaptive_weight_map=breakout_adaptive_weight_map,
            breakout_adaptive_min_samples=breakout_adaptive_min_samples,
            breakout_adaptive_lookback=breakout_adaptive_lookback,
            session_bias_weight=session_bias_weight,
            session_bias_weight_map=session_bias_weight_map,
            session_bias_min_samples=session_bias_min_samples,
            session_bias_lookback=session_bias_lookback,
            rebound_weight=rebound_weight,
            rebound_weight_map=rebound_weight_map,
        )
        for step in steps
    ]

    summary = Summary(
        bars=len(candles_df),
        from_ts=str(candles_df.index.min().isoformat()),
        to_ts=str(candles_df.index.max().isoformat()),
    )

    print(
        f"bars={summary.bars} from={summary.from_ts} to={summary.to_ts} "
        f"min_abs_breakout_bias={float(args.min_abs_breakout_bias):.4f} "
        f"feature_expansion_gain={feature_expansion_gain:.4f} "
        f"breakout_adaptive_weight={breakout_adaptive_weight:.4f} "
        f"breakout_adaptive_weight_map={json.dumps(breakout_adaptive_weight_map, ensure_ascii=False)} "
        f"breakout_adaptive_min_samples={breakout_adaptive_min_samples} "
        f"breakout_adaptive_lookback={breakout_adaptive_lookback} "
        f"session_bias_weight={session_bias_weight:.4f} "
        f"session_bias_weight_map={json.dumps(session_bias_weight_map, ensure_ascii=False)} "
        f"session_bias_min_samples={session_bias_min_samples} "
        f"session_bias_lookback={session_bias_lookback} "
        f"rebound_weight={rebound_weight:.4f} "
        f"rebound_weight_map={json.dumps(rebound_weight_map, ensure_ascii=False)}"
    )
    print(
        "step horizon n "
        "hit_before hit_after hit_delta "
        "mae_before mae_after mae_delta "
        "breakout_hit breakout_hit_filtered breakout_filtered_coverage "
        "range_cov_before range_cov_after range_cov_delta "
        "range_width_before range_width_after range_width_delta"
    )
    for row in rows:
        print(
            f"{row.step:>4} {row.horizon:>7} {row.n:>5} "
            f"{row.hit_before:>10.4f} {row.hit_after:>9.4f} {row.hit_delta:>9.4f} "
            f"{row.mae_before:>10.4f} {row.mae_after:>9.4f} {row.mae_delta:>9.4f} "
            f"{row.breakout_hit:>11.4f} {row.breakout_hit_filtered:>20.4f} {row.breakout_filtered_coverage:>26.4f} "
            f"{row.range_coverage_before:>16.4f} {row.range_coverage_after:>15.4f} {row.range_coverage_delta:>14.4f} "
            f"{row.range_width_before:>18.4f} {row.range_width_after:>17.4f} {row.range_width_delta:>16.4f}"
        )

    if args.json_out:
        payload = {
            "summary": asdict(summary),
            "config": {
                "feature_expansion_gain": feature_expansion_gain,
                "min_abs_breakout_bias": float(args.min_abs_breakout_bias),
                "breakout_adaptive_weight": breakout_adaptive_weight,
                "breakout_adaptive_weight_map": breakout_adaptive_weight_map,
                "breakout_adaptive_min_samples": breakout_adaptive_min_samples,
                "breakout_adaptive_lookback": breakout_adaptive_lookback,
                "session_bias_weight": session_bias_weight,
                "session_bias_weight_map": session_bias_weight_map,
                "session_bias_min_samples": session_bias_min_samples,
                "session_bias_lookback": session_bias_lookback,
                "rebound_weight": rebound_weight,
                "rebound_weight_map": rebound_weight_map,
            },
            "rows": [asdict(r) for r in rows],
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json_out={out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

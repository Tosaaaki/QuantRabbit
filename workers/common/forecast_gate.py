"""
workers.common.forecast_gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Probabilistic forecast gate for entry sizing / blocking.

The gate uses one of the following sources:
- sklearn bundle prediction (`analysis/forecast_sklearn.py`)
- deterministic technical forecast fallback computed from live OHLC
- blend of both (default, when bundle is available)

This keeps trading deterministic (no LLM) and allows predictive decisions even
when an offline model bundle is not mounted yet.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from statistics import NormalDist
import time
from typing import Any
from typing import Optional

from utils.metrics_logger import log_metric

LOG = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_set(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


_ENABLED = _env_bool("FORECAST_GATE_ENABLED", True)
_BUNDLE_PATH = os.getenv(
    "FORECAST_BUNDLE_PATH",
    "config/forecast_models/USD_JPY_bundle.joblib",
).strip()
_TTL_SEC = max(1.0, _env_float("FORECAST_GATE_TTL_SEC", 15.0))
_SOURCE = os.getenv("FORECAST_GATE_SOURCE", "auto").strip().lower()  # auto|bundle|technical
if _SOURCE not in {"auto", "bundle", "technical"}:
    _SOURCE = "auto"
_AUTO_BLEND_TECH = max(0.0, min(1.0, _env_float("FORECAST_GATE_AUTO_BLEND_TECH", 0.35)))
_TECH_ENABLED = _env_bool("FORECAST_TECH_ENABLED", True)
_TECH_SCORE_GAIN = max(0.1, _env_float("FORECAST_TECH_SCORE_GAIN", 1.0))
_TECH_PROB_STRENGTH = max(0.0, min(1.0, _env_float("FORECAST_TECH_PROB_STRENGTH", 0.75)))
_TECH_PROJECTION_WEIGHT = max(
    0.0,
    min(1.2, _env_float("FORECAST_TECH_PROJECTION_WEIGHT", 0.38)),
)
_TECH_PROJECTION_GAIN = max(0.1, _env_float("FORECAST_TECH_PROJECTION_GAIN", 1.0))
_TECH_FEATURE_EXPANSION_GAIN = max(
    0.0,
    min(1.0, _env_float("FORECAST_TECH_FEATURE_EXPANSION_GAIN", 0.0)),
)
_TECH_BREAKOUT_ADAPTIVE_ENABLED = _env_bool(
    "FORECAST_TECH_BREAKOUT_ADAPTIVE_ENABLED",
    True,
)
_TECH_BREAKOUT_ADAPTIVE_WEIGHT = max(
    0.0,
    min(0.8, _env_float("FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT", 0.22)),
)
_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES = max(
    16,
    int(_env_float("FORECAST_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES", 80)),
)
_TECH_BREAKOUT_ADAPTIVE_LOOKBACK = max(
    _TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES,
    int(_env_float("FORECAST_TECH_BREAKOUT_ADAPTIVE_LOOKBACK", 360)),
)
_STYLE_GUARD_ENABLED = _env_bool("FORECAST_GATE_STYLE_GUARD_ENABLED", True)
_STYLE_TREND_MIN_STRENGTH = max(
    0.0, min(1.0, _env_float("FORECAST_GATE_STYLE_TREND_MIN_STRENGTH", 0.52))
)
_STYLE_RANGE_MIN_PRESSURE = max(
    0.0, min(1.0, _env_float("FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE", 0.52))
)

_EDGE_BLOCK = max(0.0, min(1.0, _env_float("FORECAST_GATE_EDGE_BLOCK", 0.38)))
_EDGE_BLOCK_TREND = max(
    0.0, min(1.0, _env_float("FORECAST_GATE_EDGE_BLOCK_TREND", max(_EDGE_BLOCK, 0.41)))
)
_EDGE_BLOCK_RANGE = max(
    0.0, min(1.0, _env_float("FORECAST_GATE_EDGE_BLOCK_RANGE", max(0.2, _EDGE_BLOCK - 0.03)))
)
_EDGE_BAD = max(0.0, min(1.0, _env_float("FORECAST_GATE_EDGE_BAD", 0.45)))
_EDGE_REF = max(0.0, min(1.0, _env_float("FORECAST_GATE_EDGE_REF", 0.55)))
_SCALE_MIN = max(0.0, min(1.0, _env_float("FORECAST_GATE_SCALE_MIN", 0.5)))
_EDGE_PROJECTION_BONUS = max(
    0.0,
    min(0.3, _env_float("FORECAST_GATE_PROJECTION_EDGE_BONUS", 0.08)),
)
_EDGE_PROJECTION_PENALTY = max(
    0.0,
    min(0.4, _env_float("FORECAST_GATE_PROJECTION_EDGE_PENALTY", 0.12)),
)
_TF_CONFLUENCE_ENABLED = _env_bool("FORECAST_GATE_TF_CONFLUENCE_ENABLED", True)
_TF_CONFLUENCE_BONUS = max(
    0.0,
    min(0.25, _env_float("FORECAST_GATE_TF_CONFLUENCE_BONUS", 0.04)),
)
_TF_CONFLUENCE_PENALTY = max(
    0.0,
    min(0.35, _env_float("FORECAST_GATE_TF_CONFLUENCE_PENALTY", 0.12)),
)
_TF_CONFLUENCE_MIN_CONFIRM = max(
    1,
    min(3, int(_env_float("FORECAST_GATE_TF_CONFLUENCE_MIN_CONFIRM", 1))),
)
_VOL_REGIME_LOW = max(0.0, min(1.0, _env_float("FORECAST_GATE_VOL_REGIME_LOW", 0.32)))
_VOL_REGIME_HIGH = max(0.0, min(1.0, _env_float("FORECAST_GATE_VOL_REGIME_HIGH", 0.62)))
_VOL_REGIME_EXTREME = max(0.0, min(1.0, _env_float("FORECAST_GATE_VOL_REGIME_EXTREME", 0.82)))

_REQUIRE_FRESH = _env_bool("FORECAST_GATE_REQUIRE_FRESH", False)
_MAX_AGE_SEC = max(1.0, _env_float("FORECAST_GATE_MAX_AGE_SEC", 120.0))
_M1_STALE_SEC = max(30.0, _env_float("FORECAST_GATE_M1_STALE_SEC", 150.0))
_M1_FALLBACK_PATH = Path(
    os.getenv("FORECAST_GATE_M1_FALLBACK_PATH", "logs/oanda/candles_M1_latest.json")
).expanduser()

_STRATEGY_ALLOWLIST = _env_set("FORECAST_GATE_STRATEGY_ALLOWLIST")
_POCKET_ALLOWLIST = _env_set("FORECAST_GATE_POCKET_ALLOWLIST")

_HORIZON_FORCE = os.getenv("FORECAST_GATE_HORIZON", "").strip()
_HORIZON_SCALP_FAST = os.getenv("FORECAST_GATE_HORIZON_SCALP_FAST", "1m").strip()
_HORIZON_SCALP = os.getenv("FORECAST_GATE_HORIZON_SCALP", "5m").strip()
_HORIZON_MICRO = os.getenv("FORECAST_GATE_HORIZON_MICRO", "8h").strip()
_HORIZON_MACRO = os.getenv("FORECAST_GATE_HORIZON_MACRO", "1d").strip()
_PIP_SIZE = max(1e-8, _env_float("FORECAST_GATE_PIP_SIZE", 0.01))
_RANGE_LOWER_Q = max(
    0.01,
    min(0.49, _env_float("FORECAST_RANGE_BAND_LOWER_Q", 0.20)),
)
_RANGE_UPPER_Q = max(
    0.51,
    min(0.99, _env_float("FORECAST_RANGE_BAND_UPPER_Q", 0.80)),
)
if _RANGE_UPPER_Q <= _RANGE_LOWER_Q:
    _RANGE_LOWER_Q = 0.20
    _RANGE_UPPER_Q = 0.80
_RANGE_SIGMA_FLOOR_PIPS = max(0.05, _env_float("FORECAST_RANGE_SIGMA_FLOOR_PIPS", 0.35))
_RANGE_NORMAL = NormalDist()
_TECH_PREFERRED_HORIZONS = _env_set("FORECAST_GATE_TECH_PREFERRED_HORIZONS")
if not _TECH_PREFERRED_HORIZONS:
    _TECH_PREFERRED_HORIZONS = {"1m", "5m", "10m"}
_TF_CONFLUENCE_DEFAULT = {
    "1m": ("5m", "10m"),
    "5m": ("1m", "10m"),
    "10m": ("5m", "1h"),
    "1h": ("10m", "8h"),
    "8h": ("1h", "1d"),
    "1d": ("8h", "1w"),
    "1w": ("1d",),
}


@dataclass(frozen=True)
class ForecastDecision:
    allowed: bool
    scale: float
    reason: str
    horizon: str
    edge: float
    p_up: float
    expected_pips: Optional[float] = None
    anchor_price: Optional[float] = None
    target_price: Optional[float] = None
    range_low_pips: Optional[float] = None
    range_high_pips: Optional[float] = None
    range_sigma_pips: Optional[float] = None
    range_low_price: Optional[float] = None
    range_high_price: Optional[float] = None
    tp_pips_hint: Optional[float] = None
    sl_pips_cap: Optional[float] = None
    rr_floor: Optional[float] = None
    feature_ts: Optional[str] = None
    source: Optional[str] = None
    style: Optional[str] = None
    trend_strength: Optional[float] = None
    range_pressure: Optional[float] = None
    future_flow: Optional[str] = None
    volatility_state: Optional[str] = None
    trend_state: Optional[str] = None
    range_state: Optional[str] = None
    volatility_rank: Optional[float] = None
    regime_score: Optional[float] = None
    leading_indicator: Optional[str] = None
    leading_indicator_strength: Optional[float] = None
    tf_confluence_score: Optional[float] = None
    tf_confluence_count: Optional[int] = None
    tf_confluence_horizons: Optional[str] = None


def _future_flow_plan(
    *,
    p_up: float,
    trend_strength: float,
    range_pressure: float,
    edge: float,
    side: str,
    style: Optional[str],
    horizon: str,
) -> str:
    p = _clamp(float(p_up), 0.0, 1.0)
    t = _clamp(float(trend_strength), 0.0, 1.0)
    r = _clamp(float(range_pressure), 0.0, 1.0)
    e = _clamp(float(edge), 0.0, 1.0)
    s = str(side or "").strip().lower()
    is_buy = s == "buy"
    flow_mode = "trend" if style == "trend" else "range" if style == "range" else "mixed"
    if t >= 0.62:
        core = "上昇トレンド継続" if is_buy else "下落トレンド継続"
    elif r >= 0.7:
        if is_buy and p > 0.5:
            core = "上値圧力強め（天井警戒）"
        elif (not is_buy) and p < 0.5:
            core = "下値圧力強め（底警戒）"
        else:
            core = "レンジ継続"
    elif 0.45 < p < 0.55:
        core = "レンジ寄り（方向未確定）"
    else:
        core = "上振れ寄り" if p >= 0.55 else "下振れ寄り"
    strength = "強い" if e >= 0.62 else "中程度" if e >= 0.52 else "弱い"
    return f"{horizon}:{flow_mode}:{core}:{strength}"


def _regime_profile_from_row(
    row: dict[str, Any],
    *,
    p_up: float,
    edge: float,
    trend_strength: float,
    range_pressure: float,
) -> dict[str, Any]:
    p = _clamp(float(p_up), 0.0, 1.0)
    edge_norm = _clamp(float(edge), 0.0, 1.0)
    trend_strength = _clamp(float(trend_strength), 0.0, 1.0)
    range_pressure = _clamp(float(range_pressure), 0.0, 1.0)
    proj = _safe_float(row.get("projection_score"), (p - 0.5) * 2.0)
    proj_conf = _clamp(_safe_float(row.get("projection_confidence"), 0.0), 0.0, 1.0)
    expected_pips = _safe_float(row.get("expected_pips"), 0.0)
    min_move = _safe_float(row.get("min_move_pips"), 0.0)
    step_bars = _safe_float(row.get("step_bars"), 12.0)
    if min_move > 0.0:
        vol_base = abs(expected_pips) / max(0.45, min_move * 1.55)
    else:
        vol_base = abs(expected_pips) / max(0.35, 0.26 * math.sqrt(max(1.0, step_bars)))
    volatility_rank = _clamp(
        0.55 * _clamp(vol_base, 0.0, 1.0)
        + 0.28 * proj_conf
        + 0.17 * _clamp(abs(proj), 0.0, 1.0),
        0.0,
        1.0,
    )

    if volatility_rank >= _VOL_REGIME_EXTREME:
        volatility_state = "very_high"
    elif volatility_rank >= _VOL_REGIME_HIGH:
        volatility_state = "high"
    elif volatility_rank <= _VOL_REGIME_LOW / 2.0:
        volatility_state = "very_low"
    elif volatility_rank <= _VOL_REGIME_LOW:
        volatility_state = "low"
    else:
        volatility_state = "normal"

    momentum = _clamp(edge_norm * 2.0 - 1.0, -1.0, 1.0)
    if abs(proj) > 0.04:
        if proj < 0:
            momentum = _clamp((0.35 * proj + 0.65 * momentum), -1.0, 1.0)
        else:
            momentum = _clamp((0.35 * proj + 0.65 * momentum), -1.0, 1.0)

    if abs(momentum) < 0.12 or trend_strength < 0.30:
        trend_state = "flat"
    else:
        direction = "up" if momentum >= 0.0 else "down"
        if trend_strength >= 0.72:
            trend_state = f"strong_{direction}"
        elif trend_strength >= 0.58:
            trend_state = f"{direction}"
        else:
            trend_state = f"weak_{direction}"

    if range_pressure >= 0.75 and trend_strength <= 0.38:
        range_state = "squeeze"
    elif range_pressure >= 0.65 and trend_strength <= 0.54:
        range_state = "range_hold"
    elif range_pressure <= 0.34 and trend_strength >= 0.62:
        range_state = "breakout_ready"
    elif range_pressure <= 0.40 and abs(momentum) >= 0.24:
        range_state = "trend_extension"
    else:
        range_state = "mixed"

    if proj_conf >= 0.60 and abs(proj) >= 0.25:
        leading_indicator = "projection"
    elif range_state == "squeeze" and abs(proj) <= 0.12:
        leading_indicator = "squeeze_watch"
    elif trend_state.startswith("strong"):
        leading_indicator = "strong_momentum"
    elif range_state in {"range_hold", "mixed"} and p <= 0.45:
        leading_indicator = "reversion_tilt_down"
    elif range_state in {"range_hold", "mixed"} and p >= 0.55:
        leading_indicator = "reversion_tilt_up"
    else:
        leading_indicator = "neutral"

    leading_indicator_strength = _clamp(
        0.45 * proj_conf + 0.35 * _clamp(abs(proj), 0.0, 1.0) + 0.20 * abs(momentum),
        0.0,
        1.0,
    )

    trend_term = (2.0 * trend_strength - 1.0) * (1.0 if momentum >= 0.0 else -1.0)
    regime_score = _clamp(0.66 * momentum + 0.34 * trend_term, -1.0, 1.0)

    return {
        "volatility_state": volatility_state,
        "trend_state": trend_state,
        "range_state": range_state,
        "volatility_rank": round(float(volatility_rank), 6),
        "regime_score": round(float(regime_score), 6),
        "leading_indicator": leading_indicator,
        "leading_indicator_strength": round(float(leading_indicator_strength), 6),
    }


_HORIZON_META_DEFAULT = {
    "1h": {"timeframe": "M5", "step_bars": 12},
    "8h": {"timeframe": "M5", "step_bars": 96},
    "1d": {"timeframe": "H1", "step_bars": 24},
    "1w": {"timeframe": "D1", "step_bars": 5},
    "1m": {"timeframe": "M1", "step_bars": 1},
    "5m": {"timeframe": "M1", "step_bars": 5},
    "10m": {"timeframe": "M1", "step_bars": 10},
}
_TF_MINUTES = {"M1": 1.0, "M5": 5.0, "H1": 60.0, "H4": 240.0, "D1": 1440.0}
_TECH_HORIZON_CFG = {
    # shorter horizons allow more mean-reversion influence
    "1m": {"trend_w": 0.70, "mr_w": 0.30, "temp": 0.9},
    "5m": {"trend_w": 0.40, "mr_w": 0.60, "temp": 1.0},
    "10m": {"trend_w": 0.40, "mr_w": 0.60, "temp": 1.04},
    "1h": {"trend_w": 0.56, "mr_w": 0.44, "temp": 1.35},
    "8h": {"trend_w": 0.68, "mr_w": 0.32, "temp": 1.25},
    "1d": {"trend_w": 0.76, "mr_w": 0.24, "temp": 1.10},
    "1w": {"trend_w": 0.82, "mr_w": 0.18, "temp": 1.00},
}
_TECH_MIN_FEATURE_ROWS = max(24, int(_env_float("FORECAST_GATE_TECH_MIN_FEATURE_ROWS", 50)))


_BUNDLE_CACHE = None
_BUNDLE_MTIME = 0.0
_PRED_CACHE: dict | None = None
_PRED_CACHE_TS = 0.0

_TREND_STYLE_HINTS = (
    "trend",
    "momentum",
    "break",
    "burst",
    "impulse",
    "runner",
    "spike",
    "h1",
)
_RANGE_STYLE_HINTS = (
    "range",
    "revert",
    "reversion",
    "fader",
    "mean",
    "vwap",
    "bb_rsi",
    "bb-rsi",
)
_STYLE_BY_STRATEGY_BASE = {
    # trend / breakout family
    "trendma": "trend",
    "h1momentum": "trend",
    "trendmomentummicro": "trend",
    "micromomentumstack": "trend",
    "momentumburst": "trend",
    "micropullbackema": "trend",
    "microrangebreak": "trend",
    "momentumpulse": "trend",
    "volcompressionbreak": "trend",
    # range / mean-revert family
    "bbrsi": "range",
    "bbrsifast": "range",
    "microvwaprevert": "range",
    "microlevelreactor": "range",
    "microvwapbound": "range",
    "rangefader": "range",
}
_HORIZON_BY_STRATEGY_BASE = {
    "trendma": "1d",
    "scalpping5s": "1m",
    "scalpping5sb": "1m",
    "scalpm1scalper": "5m",
    "scalptickimbalance": "5m",
    "scalptickimbalancerrplus": "5m",
    "scalplevelreject": "5m",
    "scalplevelrejectplus": "5m",
    "scalpmacdrsidiv": "10m",
    "scalpwickreversalblend": "10m",
    "scalpwickreversalpro": "10m",
    "scalpsqueezepulsebreak": "10m",
    "techfusion": "10m",
    "macrotechfusion": "10m",
    "sessionopen": "10m",
    "londonmomentum": "10m",
    "macroh1momentum": "10m",
    "h1momentumswing": "10m",
    "trendh1": "10m",
    "micromultistrat": "10m",
    "microadaptiverevert": "10m",
}
_HORIZON_HINTS = (
    ("scalpping5sb", "1m"),
    ("scalpping5s", "1m"),
    ("m1scalper", "5m"),
    ("tickimbalance", "5m"),
    ("levelreject", "5m"),
    ("reversalnwave", "5m"),
    ("trendreclaim", "5m"),
    ("volspikerider", "5m"),
    ("micro", "10m"),
    ("momentumburst", "10m"),
    ("techfusion", "10m"),
    ("sessionopen", "10m"),
    ("macdrsidiv", "10m"),
    ("wickreversal", "10m"),
    ("squeezepulse", "10m"),
    ("h1momentum", "10m"),
    ("londonmomentum", "10m"),
    ("trendh1", "10m"),
    ("trendma", "1d"),
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        value_float = float(value)
    except Exception:
        return default
    if not math.isfinite(value_float):
        return default
    return value_float


def _safe_abs_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        value_float = float(value)
    except Exception:
        return default
    if not math.isfinite(value_float):
        return default
    return abs(value_float)


def _extract_candle_close(candle: Any) -> Optional[float]:
    if not isinstance(candle, dict):
        return None
    for key in ("close", "c", "mid", "price", "last", "close_price"):
        raw = candle.get(key)
        if isinstance(raw, dict):
            raw = raw.get("c")
        value = _safe_optional_float(raw)
        if value is not None:
            return value
    return None


def _future_pips_from_candles_tail(
    *,
    candles: list[dict],
    aligned_length: int,
    step_bars: int,
) -> list[float]:
    if aligned_length <= 0 or step_bars <= 0:
        return []
    closes: list[float] = []
    for candle in candles or []:
        close = _extract_candle_close(candle)
        if close is None:
            continue
        closes.append(float(close))
    if len(closes) < aligned_length:
        return []
    aligned = closes[-aligned_length:]
    out: list[float] = []
    denom = float(_PIP_SIZE)
    for idx, current in enumerate(aligned):
        future_idx = idx + int(step_bars)
        if future_idx >= len(aligned):
            out.append(float("nan"))
            continue
        out.append((float(aligned[future_idx]) - float(current)) / denom)
    return out


def _estimate_directional_skill(
    *,
    signal_values: list[float],
    target_values: list[float],
    min_samples: int,
    lookback: int,
) -> tuple[float, float, int]:
    if min_samples <= 0:
        min_samples = 1
    usable: list[tuple[float, float]] = []
    for signal, target in zip(signal_values, target_values):
        s = _safe_optional_float(signal)
        t = _safe_optional_float(target)
        if s is None or t is None:
            continue
        if not math.isfinite(s) or not math.isfinite(t):
            continue
        if abs(t) < 1e-12 or abs(s) < 1e-12:
            continue
        usable.append((float(s), float(t)))
    if lookback > 0 and len(usable) > lookback:
        usable = usable[-lookback:]
    count = len(usable)
    if count < int(min_samples):
        return 0.0, 0.5, count
    hits = 0
    for signal, target in usable:
        if signal * target > 0.0:
            hits += 1
    hit_rate = hits / float(count)
    skill = _clamp((hit_rate - 0.5) * 2.0, -1.0, 1.0)
    confidence = _clamp(count / max(float(min_samples * 2), 1.0), 0.0, 1.0)
    return float(skill * confidence), float(hit_rate), count


def _sigmoid(x: float) -> float:
    x = _clamp(float(x), -40.0, 40.0)
    return 1.0 / (1.0 + math.exp(-x))


def _normal_quantile_z(quantile: float) -> float:
    q = _clamp(float(quantile), 0.01, 0.99)
    try:
        return float(_RANGE_NORMAL.inv_cdf(q))
    except Exception:
        return 0.0


def _estimate_range_sigma_pips(
    row: dict[str, Any],
    *,
    fallback: Optional[float] = None,
) -> float:
    candidates: tuple[Any, ...] = (
        row.get("range_sigma_pips"),
        row.get("dispersion_pips"),
        fallback,
    )
    for candidate in candidates:
        sigma = _safe_optional_float(candidate)
        if sigma is not None and sigma > 0.0:
            return max(_RANGE_SIGMA_FLOOR_PIPS, float(sigma))

    expected = abs(_safe_float(row.get("expected_pips"), 0.0))
    step_bars = max(1.0, _safe_float(row.get("step_bars"), 12.0))
    min_move = max(0.2, _safe_float(row.get("min_move_pips"), 0.6))
    projection_conf = _clamp(_safe_float(row.get("projection_confidence"), 0.0), 0.0, 1.0)
    sigma = (
        0.42 * max(0.25, expected)
        + 0.35 * min_move
        + 0.18 * math.sqrt(step_bars)
        + 0.22 * (1.0 - projection_conf)
    )
    if not math.isfinite(sigma):
        sigma = _RANGE_SIGMA_FLOOR_PIPS
    return max(_RANGE_SIGMA_FLOOR_PIPS, float(sigma))


def _attach_range_band(
    row: dict[str, Any],
    *,
    fallback_sigma: Optional[float] = None,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row

    expected_pips = _safe_optional_float(row.get("expected_pips"))
    if expected_pips is None:
        row.setdefault("range_low_pips", None)
        row.setdefault("range_high_pips", None)
        row.setdefault("range_sigma_pips", None)
        row.setdefault("range_low_price", None)
        row.setdefault("range_high_price", None)
        return row

    current_low = _safe_optional_float(row.get("range_low_pips"))
    current_high = _safe_optional_float(row.get("range_high_pips"))
    if current_low is not None and current_high is not None and current_low < current_high:
        row["range_low_pips"] = round(float(current_low), 4)
        row["range_high_pips"] = round(float(current_high), 4)
        sigma_existing = _safe_optional_float(row.get("range_sigma_pips"))
        if sigma_existing is None or sigma_existing <= 0.0:
            span = max(1e-6, _normal_quantile_z(_RANGE_UPPER_Q) - _normal_quantile_z(_RANGE_LOWER_Q))
            sigma_existing = max(_RANGE_SIGMA_FLOOR_PIPS, (current_high - current_low) / span)
        row["range_sigma_pips"] = round(float(max(_RANGE_SIGMA_FLOOR_PIPS, sigma_existing)), 4)
        row.setdefault("q10_pips", round(expected_pips + _normal_quantile_z(0.10) * row["range_sigma_pips"], 4))
        row.setdefault("q50_pips", round(expected_pips, 4))
        row.setdefault("q90_pips", round(expected_pips + _normal_quantile_z(0.90) * row["range_sigma_pips"], 4))
        return row

    sigma = _estimate_range_sigma_pips(row, fallback=fallback_sigma)
    lower_z = _normal_quantile_z(_RANGE_LOWER_Q)
    upper_z = _normal_quantile_z(_RANGE_UPPER_Q)
    low = expected_pips + lower_z * sigma
    high = expected_pips + upper_z * sigma
    if low > high:
        low, high = high, low

    row["range_low_pips"] = round(float(low), 4)
    row["range_high_pips"] = round(float(high), 4)
    row["range_sigma_pips"] = round(float(sigma), 4)

    # Optional broader quantiles for VM-side monitoring.
    row.setdefault("q10_pips", round(expected_pips + _normal_quantile_z(0.10) * sigma, 4))
    row.setdefault("q50_pips", round(expected_pips, 4))
    row.setdefault("q90_pips", round(expected_pips + _normal_quantile_z(0.90) * sigma, 4))
    return row


def _attach_range_prices(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    anchor = _safe_optional_float(row.get("anchor_price"))
    low = _safe_optional_float(row.get("range_low_pips"))
    high = _safe_optional_float(row.get("range_high_pips"))
    if anchor is None or low is None or high is None:
        existing_low = _safe_optional_float(row.get("range_low_price"))
        existing_high = _safe_optional_float(row.get("range_high_price"))
        if existing_low is not None and existing_high is not None:
            row["range_low_price"] = round(float(existing_low), 5)
            row["range_high_price"] = round(float(existing_high), 5)
            return row
        row["range_low_price"] = None
        row["range_high_price"] = None
        return row
    row["range_low_price"] = round(anchor + low * float(_PIP_SIZE), 5)
    row["range_high_price"] = round(anchor + high * float(_PIP_SIZE), 5)
    return row


def _normalize_strategy_key(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _to_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _normalize_forecast_timeframe(value: Any) -> Optional[str]:
    candidate = str(value or "").strip().upper()
    if candidate in _TF_MINUTES:
        return candidate
    return None


def _normalize_forecast_horizon_profile(profile_raw: Any) -> dict[str, Any]:
    if not isinstance(profile_raw, dict):
        return {}
    out: dict[str, Any] = {}
    timeframe = _normalize_forecast_timeframe(
        profile_raw.get("timeframe") or profile_raw.get("forecast_timeframe")
    )
    if timeframe:
        out["timeframe"] = timeframe
    step_bars = _to_positive_int(
        profile_raw.get("step_bars")
        if profile_raw.get("step_bars") is not None
        else profile_raw.get("forecast_step_bars")
    )
    if step_bars is not None:
        out["step_bars"] = step_bars
    for key in ("blend_with_bundle", "technical_only", "blend"):
        if key in profile_raw:
            if key == "blend":
                out["blend_with_bundle"] = bool(profile_raw.get(key))
            else:
                out[key] = bool(profile_raw.get(key))
    if "horizon" in profile_raw:
        value = str(profile_raw.get("horizon")).strip().lower()
        if value:
            out["horizon"] = value
    if out:
        return out
    return {}


def _resolve_forecast_profile(
    entry_thesis: Optional[dict],
    meta: Optional[dict],
) -> dict[str, Any]:
    source: list[dict[str, Any]] = []
    if isinstance(meta, dict):
        source.append(dict(meta))
    if isinstance(entry_thesis, dict):
        source.append(dict(entry_thesis))

    profile: dict[str, Any] = {}
    for container in source:
        normalized = _normalize_forecast_horizon_profile(container.get("forecast_profile"))
        if normalized:
            profile.update(normalized)

        for key in ("forecast_timeframe", "timeframe"):
            if key in container:
                tf = _normalize_forecast_timeframe(container.get(key))
                if tf:
                    profile["timeframe"] = tf

        if "forecast_step_bars" in container:
            step = _to_positive_int(container.get("forecast_step_bars"))
            if step is not None:
                profile["step_bars"] = step

        for key in ("horizon", "forecast_horizon"):
            if key in container:
                value = str(container.get(key)).strip().lower()
                if value:
                    profile["horizon"] = value

        for key in ("blend_with_bundle", "technical_only", "forecast_blend_with_bundle", "forecast_technical_only"):
            if key in container:
                normalized_key = "technical_only" if key == "forecast_technical_only" else "blend_with_bundle" if key == "forecast_blend_with_bundle" else key
                profile[normalized_key] = bool(container.get(key))

    return profile


def _infer_horizon_from_profile(
    profile: dict[str, Any],
) -> Optional[str]:
    horizon = profile.get("horizon")
    if isinstance(horizon, str):
        hint = str(horizon).strip().lower()
        if hint:
            return hint
    timeframe = _normalize_forecast_timeframe(profile.get("timeframe"))
    step_bars = _to_positive_int(profile.get("step_bars"))
    if not timeframe:
        return None
    candidates: list[tuple[int, str]] = []
    try:
        meta = dict(_HORIZON_META)
    except Exception:
        meta = {}
    for candidate_horizon, cfg in meta.items():
        if not isinstance(cfg, dict):
            continue
        candidate_tf = _normalize_forecast_timeframe(cfg.get("timeframe"))
        if candidate_tf != timeframe:
            continue
        candidate_step = _to_positive_int(cfg.get("step_bars"))
        if not candidate_step:
            continue
        distance = 0 if step_bars is None else abs(candidate_step - step_bars)
        candidates.append((distance, str(candidate_horizon)))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    fallback = {"M1": "1m", "M5": "5m", "H1": "1h", "H4": "4h", "D1": "1d"}
    return fallback.get(timeframe)


def _parse_horizon_list(value: Any) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    tokens: list[Any]
    if isinstance(value, (list, tuple, set)):
        tokens = list(value)
    else:
        tokens = str(value).replace("|", ",").split(",")
    for token in tokens:
        text = str(token or "").strip().lower()
        if not text:
            continue
        if text not in out:
            out.append(text)
    return out


def _resolve_tf_confluence_horizons(
    *,
    horizon: str,
    entry_thesis: Optional[dict],
    meta: Optional[dict],
) -> list[str]:
    containers: tuple[Optional[dict], ...] = (entry_thesis, meta)
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in (
            "forecast_support_horizons",
            "forecast_confirm_horizons",
            "support_horizons",
            "confirm_horizons",
        ):
            parsed = _parse_horizon_list(container.get(key))
            if parsed:
                return [h for h in parsed if h != horizon]
    fallback = list(_TF_CONFLUENCE_DEFAULT.get(str(horizon).strip().lower(), ()))
    return [h for h in fallback if h != horizon]


def _strategy_base(strategy_tag: Optional[str]) -> str:
    if not strategy_tag:
        return ""
    raw = str(strategy_tag).strip()
    if not raw:
        return ""
    return raw.split("-", 1)[0].strip()


def _strategy_horizon(strategy_tag: Optional[str]) -> Optional[str]:
    key = _normalize_strategy_key(_strategy_base(strategy_tag))
    if not key:
        return None
    horizon = _HORIZON_BY_STRATEGY_BASE.get(key)
    if horizon:
        return str(horizon).strip().lower()
    for token, hinted in _HORIZON_HINTS:
        if token in key:
            return str(hinted).strip().lower()
    return None


def _horizon_meta() -> dict[str, dict[str, object]]:
    meta = dict(_HORIZON_META_DEFAULT)
    try:
        from analysis.forecast_sklearn import DEFAULT_HORIZONS

        for spec in DEFAULT_HORIZONS:
            name = str(getattr(spec, "name", "") or "").strip()
            timeframe = str(getattr(spec, "timeframe", "") or "").strip()
            step_bars = int(getattr(spec, "step_bars", 0) or 0)
            if name and timeframe and step_bars > 0:
                if name not in meta:
                    meta[name] = {"timeframe": timeframe, "step_bars": step_bars}
    except Exception:
        pass
    return meta


_HORIZON_META = _horizon_meta()


def _strategy_style(strategy_tag: Optional[str]) -> Optional[str]:
    if not strategy_tag:
        return None
    key = str(strategy_tag).strip().lower()
    if not key:
        return None
    base = _normalize_strategy_key(_strategy_base(strategy_tag))
    if base and base in _STYLE_BY_STRATEGY_BASE:
        return _STYLE_BY_STRATEGY_BASE[base]
    if any(h in key for h in _RANGE_STYLE_HINTS):
        return "range"
    if any(h in key for h in _TREND_STYLE_HINTS):
        return "trend"
    return None


def _strategy_env_float(name: str, strategy_tag: Optional[str], default: float) -> float:
    value = _env_float(name, default)
    base = _strategy_base(strategy_tag)
    if not base:
        return value
    suffix = _normalize_strategy_key(base).upper()
    if not suffix:
        return value
    raw = os.getenv(f"{name}_STRATEGY_{suffix}")
    if raw is None:
        return value
    try:
        return float(raw)
    except Exception:
        return value


def _strategy_env_bool(name: str, strategy_tag: Optional[str], default: bool) -> bool:
    value = _env_bool(name, default)
    base = _strategy_base(strategy_tag)
    if not base:
        return value
    suffix = _normalize_strategy_key(base).upper()
    if not suffix:
        return value
    raw = os.getenv(f"{name}_STRATEGY_{suffix}")
    if raw is None:
        return value
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _should_use(strategy_tag: Optional[str], pocket: Optional[str]) -> bool:
    if not _ENABLED:
        return False
    if not pocket:
        return False
    if pocket.strip().lower() == "manual":
        return False
    if _POCKET_ALLOWLIST and pocket.strip().lower() not in _POCKET_ALLOWLIST:
        return False
    if _STRATEGY_ALLOWLIST:
        if not strategy_tag:
            return False
        key = strategy_tag.strip().lower()
        base = key.split("-", 1)[0]
        if key not in _STRATEGY_ALLOWLIST and base not in _STRATEGY_ALLOWLIST:
            return False
    return True


def _horizon_for(
    pocket: str,
    strategy_tag: Optional[str],
    entry_thesis: Optional[dict],
    meta: Optional[dict] = None,
) -> str | None:
    if not pocket:
        return None
    profile = _resolve_forecast_profile(entry_thesis, meta)
    inferred_horizon = _infer_horizon_from_profile(profile)
    if inferred_horizon:
        return inferred_horizon
    if isinstance(entry_thesis, dict):
        hinted = entry_thesis.get("forecast_horizon") or entry_thesis.get("horizon")
        if hinted:
            return str(hinted).strip()
    if isinstance(meta, dict):
        hinted = meta.get("forecast_horizon") or meta.get("horizon")
        if hinted:
            return str(hinted).strip()
    if _HORIZON_FORCE:
        return _HORIZON_FORCE
    by_strategy = _strategy_horizon(strategy_tag)
    if by_strategy:
        return by_strategy
    p = (pocket or "").strip().lower()
    if p == "scalp_fast":
        return _HORIZON_SCALP_FAST
    if p == "scalp":
        return _HORIZON_SCALP
    if p == "micro":
        return _HORIZON_MICRO
    if p == "macro":
        return _HORIZON_MACRO
    return None


def _latest_close(candles: list[dict]) -> float | None:
    for candle in reversed(candles):
        if not isinstance(candle, dict):
            continue
        for key in ("close", "mid", "price", "last", "close_price"):
            value = candle.get(key)
            if value is None:
                continue
            try:
                price = float(value)
                if math.isfinite(price):
                    return price
            except Exception:
                continue
    return None


def _attach_price_target(row: dict[str, Any], candles: list[dict]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    _attach_range_band(row)
    anchor = _latest_close(candles)
    if isinstance(anchor, float) and math.isfinite(anchor):
        row["anchor_price"] = round(anchor, 5)
        expected = row.get("expected_pips")
        if isinstance(expected, (int, float)) and math.isfinite(float(expected)):
            row["target_price"] = round(anchor + float(expected) * float(_PIP_SIZE), 5)
        else:
            row["target_price"] = None
    else:
        row["anchor_price"] = None
        row["target_price"] = None
    return _attach_range_prices(row)


def _attach_regime_profile(
    row: dict[str, Any],
    *,
    edge: float | None = None,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    p_up = _safe_float(row.get("p_up"), 0.5)
    trend_strength = _safe_float(row.get("trend_strength"), 0.5)
    range_pressure = _safe_float(row.get("range_pressure"), 0.5)
    row_profile = _regime_profile_from_row(
        row,
        p_up=p_up,
        edge=0.5 if edge is None else edge,
        trend_strength=trend_strength,
        range_pressure=range_pressure,
    )
    row.update(row_profile)
    return row


def _load_bundle_cached():
    global _BUNDLE_CACHE, _BUNDLE_MTIME, _PRED_CACHE, _PRED_CACHE_TS
    if not _BUNDLE_PATH:
        return None
    path = Path(_BUNDLE_PATH)
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0
    if _BUNDLE_CACHE is not None and mtime > 0 and abs(mtime - _BUNDLE_MTIME) < 1e-3:
        return _BUNDLE_CACHE
    try:
        from analysis.forecast_sklearn import load_bundle
    except Exception as exc:
        LOG.debug("[FORECAST] sklearn bundle import failed: %s", exc)
        return None
    try:
        bundle = load_bundle(path)
    except Exception as exc:
        LOG.warning("[FORECAST] failed to load bundle path=%s err=%s", path, exc)
        return None
    _BUNDLE_CACHE = bundle
    _BUNDLE_MTIME = mtime
    _PRED_CACHE = None
    _PRED_CACHE_TS = 0.0
    return bundle


def _scale_from_edge(edge: float) -> float:
    e = max(0.0, min(1.0, float(edge)))
    lo = max(0.0, min(1.0, float(_EDGE_BAD)))
    hi = max(0.0, min(1.0, float(_EDGE_REF)))
    if hi <= lo + 1e-9:
        return 1.0 if e >= hi else float(_SCALE_MIN)
    if e <= lo:
        return float(_SCALE_MIN)
    if e >= hi:
        return 1.0
    score = (e - lo) / (hi - lo)
    return float(_SCALE_MIN) + score * (1.0 - float(_SCALE_MIN))


def _apply_tf_confluence(
    *,
    edge: float,
    side_key: str,
    units: int,
    horizon: str,
    preds: dict[str, dict],
    entry_thesis: Optional[dict],
    meta: Optional[dict],
) -> tuple[float, Optional[float], int, str]:
    if not _TF_CONFLUENCE_ENABLED:
        return edge, None, 0, ""
    horizons = _resolve_tf_confluence_horizons(
        horizon=horizon,
        entry_thesis=entry_thesis,
        meta=meta,
    )
    if not horizons:
        return edge, None, 0, ""

    is_buy = side_key == "buy" or units > 0
    confirm_edges: list[float] = []
    used_horizons: list[str] = []
    for h in horizons:
        row = preds.get(h)
        if not isinstance(row, dict):
            continue
        p_up = _safe_optional_float(row.get("p_up"))
        if p_up is None:
            continue
        confirm_edges.append(_clamp(p_up if is_buy else (1.0 - p_up), 0.0, 1.0))
        used_horizons.append(str(h))

    confirm_count = len(confirm_edges)
    if confirm_count <= 0:
        return edge, None, 0, ""

    mean_edge = sum(confirm_edges) / float(confirm_count)
    confluence_score = _clamp((mean_edge - 0.5) / 0.5, -1.0, 1.0)
    adjusted = float(edge)
    if confirm_count >= _TF_CONFLUENCE_MIN_CONFIRM:
        if confluence_score >= 0.0:
            adjusted = _clamp(adjusted + _TF_CONFLUENCE_BONUS * confluence_score, 0.0, 1.0)
        else:
            adjusted = _clamp(adjusted - _TF_CONFLUENCE_PENALTY * abs(confluence_score), 0.0, 1.0)
    return adjusted, confluence_score, confirm_count, ",".join(used_horizons)


def _fetch_candles_by_tf() -> dict[str, list[dict]]:
    from indicators.factor_cache import get_candles_snapshot

    candles_by_tf = {
        "M1": get_candles_snapshot("M1", limit=400, include_live=True),
        "M5": get_candles_snapshot("M5", limit=1200, include_live=True),
        "H1": get_candles_snapshot("H1", limit=1000, include_live=True),
        "D1": get_candles_snapshot("D1", limit=500, include_live=True),
    }
    m1 = candles_by_tf.get("M1") or []
    if _M1_STALE_SEC > 0.0:
        age = _candles_age_sec(m1)
        if age is None or age > _M1_STALE_SEC:
            fallback = _load_candle_rows(_M1_FALLBACK_PATH, limit=400)
            if fallback:
                candles_by_tf["M1"] = fallback
    return candles_by_tf


def _parse_candle_ts(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        ts = datetime.fromisoformat(text)
    except Exception:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _candles_last_ts(candles: list[dict]) -> Optional[datetime]:
    for candle in reversed(candles or []):
        if not isinstance(candle, dict):
            continue
        ts = _parse_candle_ts(candle.get("time") or candle.get("timestamp"))
        if ts is not None:
            return ts
    return None


def _candles_age_sec(candles: list[dict]) -> Optional[float]:
    ts = _candles_last_ts(candles)
    if ts is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


def _load_candle_rows(path: Path, *, limit: int) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    rows = payload.get("candles") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(dict(row))
    if limit > 0:
        out = out[-limit:]
    return out


def _predict_bundle_latest(bundle, candles_by_tf: dict[str, list[dict]]) -> dict | None:  # noqa: ANN001
    if bundle is None:
        return None
    try:
        from analysis.forecast_sklearn import predict_latest
    except Exception:
        return None
    try:
        out = predict_latest(bundle, candles_by_tf)
    except Exception as exc:
        LOG.debug("[FORECAST] predict_latest failed: %s", exc)
        return None
    if not isinstance(out, dict):
        return None
    for row in out.values():
        if isinstance(row, dict):
            row.setdefault("source", "bundle")
            tf = str(row.get("timeframe") or "").strip().upper()
            _attach_price_target(row, candles_by_tf.get(tf, []))
            _attach_regime_profile(row)
    return out


def _timeframe_minutes(timeframe: str) -> float:
    return float(_TF_MINUTES.get(str(timeframe or "").strip().upper(), 5.0))


def _normalize_short_horizon_profile_to_m1(
    *,
    horizon: str,
    timeframe: str,
    step_bars: int,
) -> tuple[str, int, Optional[str]]:
    horizon_key = str(horizon or "").strip().lower()
    if horizon_key not in {"1m", "5m", "10m"}:
        return timeframe, step_bars, None
    tf = str(timeframe or "").strip().upper()
    if tf == "M1":
        return "M1", int(step_bars), None
    minutes = _timeframe_minutes(tf)
    if minutes <= 0.0:
        return timeframe, step_bars, None
    converted_step = max(1, int(round(float(step_bars) * minutes)))
    detail = f"{tf}x{int(step_bars)}->M1x{converted_step}"
    return "M1", converted_step, detail


def _projection_bias_from_candles(
    candles: list[dict],
    *,
    timeframe_minutes: float,
    step_bars: int,
    atr_pips: float,
    trend_hint: float,
) -> dict[str, Any]:
    try:
        from analysis.ma_projection import (
            compute_adx_projection,
            compute_bbw_projection,
            compute_ma_projection,
            compute_rsi_projection,
        )
    except Exception:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "trend_boost": 0.0,
            "range_boost": 0.0,
            "components": {},
        }

    if not candles or len(candles) < 30:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "trend_boost": 0.0,
            "range_boost": 0.0,
            "components": {},
        }

    try:
        ma = compute_ma_projection({"candles": candles}, timeframe_minutes=timeframe_minutes)
        rsi = compute_rsi_projection(candles, timeframe_minutes=timeframe_minutes)
        adx = compute_adx_projection(
            candles,
            timeframe_minutes=timeframe_minutes,
            trend_threshold=20.0,
        )
        bbw = compute_bbw_projection(
            candles,
            timeframe_minutes=timeframe_minutes,
            squeeze_threshold=0.16,
        )
    except Exception as exc:
        LOG.debug("[FORECAST] projection build failed: %s", exc)
        return {
            "score": 0.0,
            "confidence": 0.0,
            "trend_boost": 0.0,
            "range_boost": 0.0,
            "components": {},
        }

    score = 0.0
    trend_boost = 0.0
    range_boost = 0.0
    components: dict[str, float] = {}
    norm_atr = max(0.6, float(atr_pips or 0.0))

    if ma is not None:
        gap_norm = ma.gap_pips / max(1.0, norm_atr)
        slope_norm = ma.gap_slope_pips / max(0.6, norm_atr * 0.45)
        ma_score = 0.58 * math.tanh(gap_norm * 1.2) + 0.42 * math.tanh(slope_norm * 1.3)
        cross_score = 0.0
        if ma.projected_cross_bars is not None and ma.projected_cross_bars > 0.0:
            horizon_eta = max(1.5, float(step_bars) * 0.28)
            eta_scale = _clamp(1.0 - (float(ma.projected_cross_bars) / horizon_eta), 0.0, 1.0)
            if ma.gap_pips < 0.0 and ma.gap_slope_pips > 0.0:
                cross_score = 0.28 * eta_scale
            elif ma.gap_pips > 0.0 and ma.gap_slope_pips < 0.0:
                cross_score = -0.28 * eta_scale
        ma_total = ma_score + cross_score
        score += ma_total
        trend_boost += 0.12 * abs(math.tanh(gap_norm * 1.1))
        components["ma"] = round(ma_total, 6)

    if rsi is not None:
        rsi_center = (rsi.rsi - 50.0) / 17.0
        rsi_slope = math.tanh(rsi.slope_per_bar / 1.9)
        rsi_score = 0.55 * math.tanh(rsi_center) + 0.45 * rsi_slope
        horizon_eta = max(2.0, float(step_bars) * 0.35)
        exhaustion = 0.0
        if rsi.eta_upper_bars is not None and rsi.eta_upper_bars <= horizon_eta:
            exhaustion -= 0.18 * _clamp(1.0 - (rsi.eta_upper_bars / horizon_eta), 0.0, 1.0)
        if rsi.eta_lower_bars is not None and rsi.eta_lower_bars <= horizon_eta:
            exhaustion += 0.18 * _clamp(1.0 - (rsi.eta_lower_bars / horizon_eta), 0.0, 1.0)
        rsi_total = 0.7 * (rsi_score + exhaustion)
        score += rsi_total
        components["rsi"] = round(rsi_total, 6)

    if adx is not None:
        adx_level = _clamp((adx.adx - 16.0) / 18.0, 0.0, 1.0)
        adx_slope = math.tanh(adx.slope_per_bar / 2.2)
        adx_total = 0.34 * adx_slope
        score += adx_total
        trend_boost += 0.18 * adx_level + 0.1 * max(0.0, adx_slope)
        components["adx"] = round(adx_total, 6)

    if bbw is not None:
        squeeze = _clamp((0.16 - bbw.bbw) / 0.10, 0.0, 1.0)
        expansion = max(0.0, math.tanh(bbw.slope_per_bar / 0.015))
        quiet = _clamp(1.0 - expansion, 0.0, 1.0)
        hint_sign = 0.0
        if trend_hint > 0.05:
            hint_sign = 1.0
        elif trend_hint < -0.05:
            hint_sign = -1.0
        else:
            hint_sign = 1.0 if score >= 0.0 else -1.0
        bbw_total = 0.26 * hint_sign * squeeze * expansion
        score += bbw_total
        trend_boost += 0.1 * squeeze * expansion
        range_boost += 0.16 * squeeze * quiet
        components["bbw"] = round(bbw_total, 6)

    score = math.tanh(score * float(_TECH_PROJECTION_GAIN))
    trend_boost = _clamp(trend_boost, 0.0, 0.45)
    range_boost = _clamp(range_boost, 0.0, 0.45)
    confidence = _clamp(
        0.25 + 0.55 * abs(score) + 0.2 * max(trend_boost, range_boost),
        0.0,
        1.0,
    )
    return {
        "score": round(score, 6),
        "confidence": round(confidence, 6),
        "trend_boost": round(trend_boost, 6),
        "range_boost": round(range_boost, 6),
        "components": components,
    }


def _technical_missing_row(
    *,
    horizon: str,
    timeframe: str,
    step_bars: int,
    available_candles: int,
    required_candles: int,
    reason: str,
    detail: str | None = None,
    candles: list[dict] | None = None,
) -> dict[str, Any]:
    row = _attach_price_target(
        {
            "horizon": horizon,
            "source": "technical",
            "status": "insufficient_history",
            "forecast_ready": False,
            "p_up": None,
            "expected_pips": None,
            "feature_ts": None,
            "trend_strength": 0.5,
            "range_pressure": 0.5,
            "projection_score": 0.0,
            "projection_confidence": 0.0,
            "projection_components": {},
            "timeframe": timeframe,
            "step_bars": int(step_bars) if step_bars else 0,
            "available_candles": int(available_candles) if available_candles else 0,
            "required_candles": int(required_candles),
            "reason": reason,
            "detail": detail,
            "remediation": f"need >= {required_candles} candles on {timeframe}; run backfill for recent {timeframe} candles",
        },
        candles or [],
    )
    return _attach_regime_profile(row)


def _technical_prediction_for_horizon(
    candles: list[dict],
    *,
    horizon: str,
    step_bars: int,
    timeframe: str = "M5",
) -> dict | None:
    available = len(candles)
    required = max(_TECH_MIN_FEATURE_ROWS, int(step_bars) * 3, 24)
    if available == 0:
        LOG.info(
            "[FORECAST] technical prediction skipped (empty history): horizon=%s timeframe=%s required=%s",
            horizon,
            timeframe,
            required,
        )
        return _technical_missing_row(
            horizon=horizon,
            timeframe=timeframe,
            step_bars=step_bars,
            available_candles=0,
            required_candles=required,
            reason="empty_history",
            candles=candles,
        )
    if available < required:
        LOG.info(
            "[FORECAST] technical prediction skipped (history too short): horizon=%s timeframe=%s available=%s required=%s",
            horizon,
            timeframe,
            available,
            required,
        )
        return _technical_missing_row(
            horizon=horizon,
            timeframe=timeframe,
            step_bars=step_bars,
            available_candles=available,
            required_candles=required,
            reason="short_history",
            candles=candles,
        )

    try:
        from analysis.forecast_sklearn import compute_feature_frame
    except Exception:
        return _technical_missing_row(
            horizon=horizon,
            timeframe=timeframe,
            step_bars=step_bars,
            available_candles=available,
            required_candles=required,
            reason="feature_builder_unavailable",
            candles=candles,
        )

    try:
        feats = compute_feature_frame(candles)
    except Exception as exc:
        LOG.debug("[FORECAST] technical feature build failed horizon=%s err=%s", horizon, exc)
        return _technical_missing_row(
            horizon=horizon,
            timeframe=timeframe,
            step_bars=step_bars,
            available_candles=available,
            required_candles=required,
            reason="feature_build_failed",
            detail=str(exc),
            candles=candles,
        )
    if feats.empty:
        return _technical_missing_row(
            horizon=horizon,
            timeframe=timeframe,
            step_bars=step_bars,
            available_candles=available,
            required_candles=required,
            reason="feature_frame_empty",
            candles=candles,
        )

    try:
        last = feats.iloc[-1]
        required_keys = (
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
        )
        for key in required_keys:
            fv = last.get(key)
            if fv is None:
                raise ValueError(f"missing_feature_{key}")
            fv_f = float(fv)
            if not math.isfinite(fv_f):
                raise ValueError(f"nonfinite_feature_{key}:{fv_f}")
    except Exception as exc:
        LOG.debug("[FORECAST] technical feature row invalid horizon=%s err=%s", horizon, exc)
        return _technical_missing_row(
            horizon=horizon,
            timeframe=timeframe,
            step_bars=step_bars,
            available_candles=available,
            required_candles=required,
            reason="feature_row_incomplete",
            detail=str(exc),
            candles=candles,
        )

    atr = max(0.45, abs(_safe_float(last.get("atr_pips_14"), 0.0)))
    vol = max(0.2, abs(_safe_float(last.get("vol_pips_20"), 0.0)))
    ret1 = _safe_float(last.get("ret_pips_1"), 0.0) / atr
    ret3 = _safe_float(last.get("ret_pips_3"), 0.0) / atr
    ret12 = _safe_float(last.get("ret_pips_12"), 0.0) / max(0.7, atr * 1.2)
    ma_gap = _safe_float(last.get("ma_gap_pips_10_20"), 0.0) / atr
    close_ma20 = _safe_float(last.get("close_ma20_pips"), 0.0) / max(0.6, atr)
    close_ma50 = _safe_float(last.get("close_ma50_pips"), 0.0) / max(0.8, atr)
    rsi = (_safe_float(last.get("rsi_14"), 50.0) - 50.0) / 16.0
    range_pos = (_safe_float(last.get("range_pos"), 0.5) - 0.5) * 2.0
    trend_slope20 = _safe_float(last.get("trend_slope_pips_20"), 0.0) / max(0.55, atr * 0.42)
    trend_slope50 = _safe_float(last.get("trend_slope_pips_50"), 0.0) / max(0.55, atr * 0.46)
    trend_accel = _safe_float(last.get("trend_accel_pips"), 0.0) / max(0.40, atr * 0.34)
    sr_balance = _clamp(_safe_float(last.get("sr_balance_20"), 0.0), -1.0, 1.0)
    breakout_up = _safe_float(last.get("breakout_up_pips_20"), 0.0) / max(0.70, atr)
    breakout_down = _safe_float(last.get("breakout_down_pips_20"), 0.0) / max(0.70, atr)
    breakout_bias = breakout_up - breakout_down
    breakout_skill = 0.0
    breakout_hit_rate = 0.5
    breakout_samples = 0
    if _TECH_BREAKOUT_ADAPTIVE_ENABLED:
        signal_values: list[float] = []
        try:
            signal_series = feats["breakout_up_pips_20"] - feats["breakout_down_pips_20"]
            signal_values = [float(v) for v in signal_series.tolist()]
        except Exception:
            signal_values = []
        future_values = _future_pips_from_candles_tail(
            candles=candles,
            aligned_length=len(signal_values),
            step_bars=step_bars,
        )
        breakout_skill, breakout_hit_rate, breakout_samples = _estimate_directional_skill(
            signal_values=signal_values,
            target_values=future_values,
            min_samples=_TECH_BREAKOUT_ADAPTIVE_MIN_SAMPLES,
            lookback=_TECH_BREAKOUT_ADAPTIVE_LOOKBACK,
        )
    breakout_adaptive = math.tanh(breakout_bias * 0.9) * breakout_skill
    donchian_width = max(0.25, abs(_safe_float(last.get("donchian_width_pips_20"), 0.0)))
    range_compression = _safe_float(last.get("range_compression_20"), 0.0)
    trend_pullback = _safe_float(last.get("trend_pullback_norm_20"), 0.0)
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
        + _TECH_FEATURE_EXPANSION_GAIN
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
        + _TECH_FEATURE_EXPANSION_GAIN
        * (
            -0.44 * math.tanh(sr_balance * 1.2)
            - 0.26 * math.tanh(breakout_bias * 0.9)
            - 0.20 * math.tanh(trend_pullback * 0.95)
            + 0.22 * squeeze_score
            + 0.08 * math.tanh((0.20 - range_compression) * 3.5)
        )
    )
    trend_hint = math.tanh(trend_score * 0.8)
    proj = _projection_bias_from_candles(
        candles,
        timeframe_minutes=_timeframe_minutes(timeframe),
        step_bars=step_bars,
        atr_pips=atr,
        trend_hint=trend_hint,
    )
    projection_score = _safe_float(proj.get("score"), 0.0)
    projection_confidence = _safe_float(proj.get("confidence"), 0.0)
    trend_boost = _safe_float(proj.get("trend_boost"), 0.0)
    range_boost = _safe_float(proj.get("range_boost"), 0.0)

    trend_strength = _clamp(
        0.18
        + 0.44 * abs(math.tanh(ma_gap * 1.4))
        + 0.33 * abs(math.tanh(ret3 * 1.1))
        + _TECH_FEATURE_EXPANSION_GAIN
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
        + _TECH_FEATURE_EXPANSION_GAIN
        * (
            0.16 * squeeze_score
            + 0.10 * _clamp(1.0 - abs(sr_balance), 0.0, 1.0)
            - 0.12 * abs(math.tanh(breakout_bias * 0.8))
        ),
        0.0,
        1.0,
    )

    cfg = _TECH_HORIZON_CFG.get(horizon, _TECH_HORIZON_CFG["8h"])
    combo = (
        cfg["trend_w"] * trend_score * (1.0 - 0.55 * range_pressure)
        + cfg["mr_w"] * mean_revert_score * range_pressure
        + _TECH_PROJECTION_WEIGHT * projection_score
        + _TECH_BREAKOUT_ADAPTIVE_WEIGHT * breakout_adaptive
    )
    raw_prob = _sigmoid(float(cfg["temp"]) * float(_TECH_SCORE_GAIN) * combo)

    sample_count = len(feats.index)
    sample_strength = _clamp((sample_count - 40.0) / 120.0, 0.25, 1.0)
    damp = _TECH_PROB_STRENGTH * sample_strength
    p_up = 0.5 + (raw_prob - 0.5) * damp
    p_up = _clamp(p_up, 0.02, 0.98)

    step_scale = math.sqrt(max(1.0, float(step_bars)) / 12.0)
    expected_move = max(0.35, vol * step_scale * (0.92 + 0.25 * projection_confidence))
    expected_pips = (p_up - 0.5) * 2.0 * expected_move
    range_sigma_pips = max(
        _RANGE_SIGMA_FLOOR_PIPS,
        expected_move * (0.62 + 0.38 * (1.0 - projection_confidence)),
    )

    feature_ts = None
    try:
        feature_ts = feats.index[-1].isoformat()
    except Exception:
        feature_ts = None

    row = _attach_price_target(
        {
            "horizon": horizon,
            "p_up": round(float(p_up), 6),
            "expected_pips": round(float(expected_pips), 4),
            "range_sigma_pips": round(float(range_sigma_pips), 4),
            "feature_ts": feature_ts,
            "source": "technical",
            "status": "ready",
            "forecast_ready": True,
            "trend_strength": round(float(trend_strength), 6),
            "range_pressure": round(float(range_pressure), 6),
            "projection_score": round(float(projection_score), 6),
            "projection_confidence": round(float(projection_confidence), 6),
            "projection_components": proj.get("components"),
            "trend_slope_pips_20": round(float(_safe_float(last.get("trend_slope_pips_20"), 0.0)), 6),
            "trend_slope_pips_50": round(float(_safe_float(last.get("trend_slope_pips_50"), 0.0)), 6),
            "trend_accel_pips": round(float(_safe_float(last.get("trend_accel_pips"), 0.0)), 6),
            "sr_balance_20": round(float(sr_balance), 6),
            "breakout_bias_20": round(float(breakout_bias), 6),
            "breakout_skill_20": round(float(breakout_skill), 6),
            "breakout_hit_rate_20": round(float(breakout_hit_rate), 6),
            "breakout_samples_20": int(breakout_samples),
            "squeeze_score_20": round(float(squeeze_score), 6),
            "timeframe": timeframe,
            "step_bars": int(step_bars) if step_bars else 0,
            "available_candles": available,
            "required_candles": required,
        },
        candles,
    )
    return _attach_regime_profile(row, edge=p_up)


def _predict_technical_latest(candles_by_tf: dict[str, list[dict]]) -> dict[str, dict]:
    preds: dict[str, dict] = {}
    for horizon, meta in _HORIZON_META.items():
        timeframe = str(meta.get("timeframe") or "").strip().upper()
        step_bars = int(meta.get("step_bars") or 0)
        if not timeframe or step_bars <= 0:
            continue
        candles = candles_by_tf.get(timeframe) or []
        row = _technical_prediction_for_horizon(
            candles,
            horizon=horizon,
            step_bars=step_bars,
            timeframe=timeframe,
        )
        if isinstance(row, dict):
            preds[horizon] = row
    return preds


def _technical_row_for_forecast_profile(
    *,
    horizon: str,
    profile: dict[str, Any],
) -> dict[str, Any] | None:
    timeframe = _normalize_forecast_timeframe(profile.get("timeframe"))
    step_bars = _to_positive_int(profile.get("step_bars"))
    if (not timeframe or not step_bars) and isinstance(horizon, str):
        horizon_key = str(horizon).strip().lower()
        horizon_spec = _HORIZON_META.get(horizon_key)
        if isinstance(horizon_spec, dict):
            if not timeframe:
                timeframe = _normalize_forecast_timeframe(horizon_spec.get("timeframe"))
            if not step_bars:
                step_bars = _to_positive_int(horizon_spec.get("step_bars"))
    if not timeframe or not step_bars:
        return None
    timeframe, step_bars, normalized_detail = _normalize_short_horizon_profile_to_m1(
        horizon=horizon,
        timeframe=timeframe,
        step_bars=step_bars,
    )
    try:
        candles_by_tf = _fetch_candles_by_tf()
    except Exception:
        return None
    candles = candles_by_tf.get(timeframe) or []
    row = _technical_prediction_for_horizon(
        candles,
        horizon=horizon,
        step_bars=step_bars,
        timeframe=timeframe,
    )
    if isinstance(row, dict):
        if normalized_detail:
            row["profile_normalization"] = normalized_detail
    return row


def _blend_prediction_rows(base_row: dict, tech_row: dict) -> dict:
    alpha = _AUTO_BLEND_TECH
    if alpha <= 0.0:
        return base_row
    _attach_range_band(base_row)
    _attach_range_band(tech_row)
    p_base = _safe_float(base_row.get("p_up"), 0.5)
    p_tech = _safe_float(tech_row.get("p_up"), p_base)
    e_base = _safe_float(base_row.get("expected_pips"), 0.0)
    e_tech = _safe_float(tech_row.get("expected_pips"), e_base)
    sigma_base = _estimate_range_sigma_pips(base_row)
    sigma_tech = _estimate_range_sigma_pips(tech_row, fallback=sigma_base)
    proj_base = _safe_float(
        base_row.get("projection_score"),
        _safe_float(tech_row.get("projection_score"), 0.0),
    )
    proj_tech = _safe_float(tech_row.get("projection_score"), proj_base)
    proj_conf_base = _safe_float(
        base_row.get("projection_confidence"),
        _safe_float(tech_row.get("projection_confidence"), 0.0),
    )
    proj_conf_tech = _safe_float(tech_row.get("projection_confidence"), proj_conf_base)
    blended_expected_pips = round((1.0 - alpha) * e_base + alpha * e_tech, 4)
    blended_sigma = (1.0 - alpha) * sigma_base + alpha * sigma_tech
    anchor_price = base_row.get("anchor_price")
    if anchor_price is None:
        anchor_price = tech_row.get("anchor_price")
    if isinstance(anchor_price, (int, float)) and math.isfinite(float(anchor_price)):
        blended_target_price = round(float(anchor_price) + float(blended_expected_pips) * float(_PIP_SIZE), 5)
    else:
        blended_target_price = None
    row = {
        **base_row,
        "p_up": round((1.0 - alpha) * p_base + alpha * p_tech, 6),
        "expected_pips": blended_expected_pips,
        "anchor_price": anchor_price if anchor_price is not None else None,
        "target_price": blended_target_price,
        "range_sigma_pips": round(float(blended_sigma), 4),
        "feature_ts": base_row.get("feature_ts") or tech_row.get("feature_ts"),
        "source": "blend",
        "projection_score": round((1.0 - alpha) * proj_base + alpha * proj_tech, 6),
        "projection_confidence": round(
            (1.0 - alpha) * proj_conf_base + alpha * proj_conf_tech,
            6,
        ),
        "projection_components": tech_row.get("projection_components")
        or base_row.get("projection_components"),
        "trend_strength": round(
            (1.0 - alpha) * _safe_float(base_row.get("trend_strength"), _safe_float(tech_row.get("trend_strength"), 0.5))
            + alpha * _safe_float(tech_row.get("trend_strength"), 0.5),
            6,
        ),
        "range_pressure": round(
            (1.0 - alpha) * _safe_float(base_row.get("range_pressure"), _safe_float(tech_row.get("range_pressure"), 0.5))
            + alpha * _safe_float(tech_row.get("range_pressure"), 0.5),
            6,
        ),
    }
    _attach_range_band(row, fallback_sigma=blended_sigma)
    _attach_range_prices(row)
    return _attach_regime_profile(
        row,
        edge=_safe_float(row.get("p_up"), 0.5),
    )


def _ensure_predictions(bundle) -> dict | None:  # noqa: ANN001 - sklearn bundle type
    global _PRED_CACHE, _PRED_CACHE_TS
    now = time.time()
    if _PRED_CACHE is not None and now - _PRED_CACHE_TS < _TTL_SEC:
        return _PRED_CACHE

    try:
        candles_by_tf = _fetch_candles_by_tf()
    except Exception:
        return None

    preds: dict[str, dict] = {}
    if _SOURCE in {"auto", "bundle"} and bundle is not None:
        bpred = _predict_bundle_latest(bundle, candles_by_tf)
        if isinstance(bpred, dict):
            for key, row in bpred.items():
                if isinstance(row, dict):
                    preds[str(key)] = dict(row)

    if _TECH_ENABLED and _SOURCE in {"auto", "technical"}:
        tpred = _predict_technical_latest(candles_by_tf)
        if _SOURCE == "technical":
            preds = tpred
        else:
            for key, trow in tpred.items():
                if key in _TECH_PREFERRED_HORIZONS:
                    preds[key] = trow
                    continue
                brow = preds.get(key)
                if isinstance(brow, dict):
                    preds[key] = _blend_prediction_rows(brow, trow)
                else:
                    preds[key] = trow

    if preds:
        _PRED_CACHE = preds
        _PRED_CACHE_TS = now
        return preds
    return None


def decide(
    *,
    strategy_tag: Optional[str],
    pocket: str,
    side: str,
    units: int,
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> ForecastDecision | None:
    if not _should_use(strategy_tag, pocket):
        return None

    # Only gate known instrument (avoid accidental use on other pairs).
    instrument = None
    if isinstance(meta, dict):
        instrument = meta.get("instrument")
    if instrument and str(instrument).strip().upper() != "USD_JPY":
        return None

    profile = _resolve_forecast_profile(entry_thesis, meta)
    horizon = _horizon_for(
        pocket,
        strategy_tag,
        entry_thesis,
        meta,
    )
    if not horizon:
        return None

    bundle = _load_bundle_cached()
    preds = _ensure_predictions(bundle)
    if not isinstance(preds, dict):
        return None

    row = preds.get(horizon)
    technical_row = None
    technical_only = bool(profile.get("technical_only"))
    blend_with_bundle = None
    if "blend_with_bundle" in profile:
        blend_with_bundle = bool(profile.get("blend_with_bundle"))
    elif "blend" in profile:
        blend_with_bundle = bool(profile.get("blend"))
    if isinstance(horizon, str) and isinstance(profile, dict):
        try:
            technical_row = _technical_row_for_forecast_profile(
                horizon=horizon,
                profile=profile,
            )
        except Exception:
            technical_row = None

    if technical_row is not None and technical_only:
        row = technical_row
    elif (
        technical_row is not None
        and blend_with_bundle is True
        and isinstance(row, dict)
    ):
        row = _blend_prediction_rows(row, technical_row)
    elif row is None and technical_row is not None:
        row = technical_row
    if not isinstance(row, dict):
        return None
    _attach_range_band(row)
    _attach_range_prices(row)

    try:
        p_up = float(row.get("p_up"))
    except Exception:
        return None
    p_up = max(0.0, min(1.0, p_up))
    source = row.get("source")
    style = _strategy_style(strategy_tag)
    trend_strength = _safe_float(row.get("trend_strength"), 0.5)
    range_pressure = _safe_float(row.get("range_pressure"), 0.5)
    style_guard_enabled = _strategy_env_bool(
        "FORECAST_GATE_STYLE_GUARD_ENABLED", strategy_tag, _STYLE_GUARD_ENABLED
    )
    trend_min_strength = _clamp(
        _strategy_env_float(
            "FORECAST_GATE_STYLE_TREND_MIN_STRENGTH", strategy_tag, _STYLE_TREND_MIN_STRENGTH
        ),
        0.0,
        1.0,
    )
    range_min_pressure = _clamp(
        _strategy_env_float(
            "FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE", strategy_tag, _STYLE_RANGE_MIN_PRESSURE
        ),
        0.0,
        1.0,
    )
    edge_block_trend = _clamp(
        _strategy_env_float("FORECAST_GATE_EDGE_BLOCK_TREND", strategy_tag, _EDGE_BLOCK_TREND),
        0.0,
        1.0,
    )
    edge_block_range = _clamp(
        _strategy_env_float("FORECAST_GATE_EDGE_BLOCK_RANGE", strategy_tag, _EDGE_BLOCK_RANGE),
        0.0,
        1.0,
    )
    side_key = (side or "").strip().lower()
    edge = p_up if side_key == "buy" or units > 0 else 1.0 - p_up
    edge = max(0.0, min(1.0, float(edge)))
    projection_score = _safe_float(row.get("projection_score"), 0.0)
    if projection_score != 0.0:
        signed_alignment = projection_score if (side_key == "buy" or units > 0) else -projection_score
        if signed_alignment > 0.0:
            edge = _clamp(edge + _EDGE_PROJECTION_BONUS * min(1.0, signed_alignment), 0.0, 1.0)
        elif signed_alignment < 0.0:
            edge = _clamp(
                edge - _EDGE_PROJECTION_PENALTY * min(1.0, abs(signed_alignment)),
                0.0,
                1.0,
            )
    (
        edge,
        tf_confluence_score,
        tf_confluence_count,
        tf_confluence_horizons,
    ) = _apply_tf_confluence(
        edge=edge,
        side_key=side_key,
        units=units,
        horizon=horizon,
        preds=preds,
        entry_thesis=entry_thesis if isinstance(entry_thesis, dict) else None,
        meta=meta if isinstance(meta, dict) else None,
    )
    future_flow = _future_flow_plan(
        p_up=p_up,
        trend_strength=trend_strength,
        range_pressure=range_pressure,
        edge=edge,
        side=side_key if side_key else ("buy" if units > 0 else "sell"),
        style=style,
        horizon=horizon,
    )
    regime_profile = _regime_profile_from_row(
        row,
        p_up=p_up,
        edge=edge,
        trend_strength=trend_strength,
        range_pressure=range_pressure,
    )

    if style_guard_enabled and style == "trend" and trend_strength < trend_min_strength:
        log_metric(
            "forecast_gate_block",
            1.0,
            tags={
                "pocket": pocket,
                "strategy": str(strategy_tag or "unknown"),
                "horizon": horizon,
                "reason": "style_mismatch_trend",
                "source": str(source or "unknown"),
                "style_threshold": f"{trend_min_strength:.3f}",
            },
        )
        return ForecastDecision(
            allowed=False,
            scale=0.0,
            reason="style_mismatch_trend",
            horizon=horizon,
            edge=edge,
            p_up=p_up,
            expected_pips=row.get("expected_pips"),
            anchor_price=row.get("anchor_price"),
            target_price=row.get("target_price"),
            range_low_pips=_safe_optional_float(row.get("range_low_pips")),
            range_high_pips=_safe_optional_float(row.get("range_high_pips")),
            range_sigma_pips=_safe_optional_float(row.get("range_sigma_pips")),
            range_low_price=_safe_optional_float(row.get("range_low_price")),
            range_high_price=_safe_optional_float(row.get("range_high_price")),
            tf_confluence_score=(
                round(float(tf_confluence_score), 6)
                if tf_confluence_score is not None
                else None
            ),
            tf_confluence_count=int(tf_confluence_count),
            tf_confluence_horizons=tf_confluence_horizons or None,
            tp_pips_hint=_safe_abs_float(row.get("tp_pips_hint"), _safe_abs_float(row.get("expected_pips"))),
            sl_pips_cap=_safe_abs_float(row.get("sl_pips_cap")),
            rr_floor=_safe_optional_float(row.get("rr_floor")),
            feature_ts=row.get("feature_ts"),
            source=str(source) if source is not None else None,
            style=style,
            trend_strength=trend_strength,
            range_pressure=range_pressure,
            future_flow=future_flow,
            **regime_profile,
        )
    if style_guard_enabled and style == "range" and range_pressure < range_min_pressure:
        log_metric(
            "forecast_gate_block",
            1.0,
            tags={
                "pocket": pocket,
                "strategy": str(strategy_tag or "unknown"),
                "horizon": horizon,
                "reason": "style_mismatch_range",
                "source": str(source or "unknown"),
                "style_threshold": f"{range_min_pressure:.3f}",
            },
        )
        return ForecastDecision(
            allowed=False,
            scale=0.0,
            reason="style_mismatch_range",
            horizon=horizon,
            edge=edge,
            p_up=p_up,
            expected_pips=row.get("expected_pips"),
            anchor_price=row.get("anchor_price"),
            target_price=row.get("target_price"),
            range_low_pips=_safe_optional_float(row.get("range_low_pips")),
            range_high_pips=_safe_optional_float(row.get("range_high_pips")),
            range_sigma_pips=_safe_optional_float(row.get("range_sigma_pips")),
            range_low_price=_safe_optional_float(row.get("range_low_price")),
            range_high_price=_safe_optional_float(row.get("range_high_price")),
            tf_confluence_score=(
                round(float(tf_confluence_score), 6)
                if tf_confluence_score is not None
                else None
            ),
            tf_confluence_count=int(tf_confluence_count),
            tf_confluence_horizons=tf_confluence_horizons or None,
            tp_pips_hint=_safe_abs_float(row.get("tp_pips_hint"), _safe_abs_float(row.get("expected_pips"))),
            sl_pips_cap=_safe_abs_float(row.get("sl_pips_cap")),
            rr_floor=_safe_optional_float(row.get("rr_floor")),
            feature_ts=row.get("feature_ts"),
            source=str(source) if source is not None else None,
            style=style,
            trend_strength=trend_strength,
            range_pressure=range_pressure,
            future_flow=future_flow,
            **regime_profile,
        )

    edge_block = float(_EDGE_BLOCK)
    if style == "trend":
        edge_block = max(edge_block, float(edge_block_trend))
    elif style == "range":
        edge_block = max(0.0, min(1.0, float(edge_block_range)))

    if edge < edge_block:
        log_metric(
            "forecast_gate_block",
            1.0,
            tags={
                "pocket": pocket,
                "strategy": str(strategy_tag or "unknown"),
                "horizon": horizon,
                "reason": "edge_block",
                "source": str(source or "unknown"),
                "style": style or "n/a",
            },
        )
        return ForecastDecision(
            allowed=False,
            scale=0.0,
            reason="edge_block",
            horizon=horizon,
            edge=edge,
            p_up=p_up,
            expected_pips=row.get("expected_pips"),
            anchor_price=row.get("anchor_price"),
            target_price=row.get("target_price"),
            range_low_pips=_safe_optional_float(row.get("range_low_pips")),
            range_high_pips=_safe_optional_float(row.get("range_high_pips")),
            range_sigma_pips=_safe_optional_float(row.get("range_sigma_pips")),
            range_low_price=_safe_optional_float(row.get("range_low_price")),
            range_high_price=_safe_optional_float(row.get("range_high_price")),
            tf_confluence_score=(
                round(float(tf_confluence_score), 6)
                if tf_confluence_score is not None
                else None
            ),
            tf_confluence_count=int(tf_confluence_count),
            tf_confluence_horizons=tf_confluence_horizons or None,
            tp_pips_hint=_safe_abs_float(row.get("tp_pips_hint"), _safe_abs_float(row.get("expected_pips"))),
            sl_pips_cap=_safe_abs_float(row.get("sl_pips_cap")),
            rr_floor=_safe_optional_float(row.get("rr_floor")),
            feature_ts=row.get("feature_ts"),
            source=str(source) if source is not None else None,
            style=style,
            trend_strength=trend_strength,
            range_pressure=range_pressure,
            future_flow=future_flow,
            **regime_profile,
        )

    feature_ts = row.get("feature_ts")
    if _REQUIRE_FRESH and feature_ts:
        try:
            from datetime import datetime, timezone

            ts = datetime.fromisoformat(str(feature_ts).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age > _MAX_AGE_SEC:
                log_metric(
                    "forecast_gate_block",
                    1.0,
                    tags={
                        "pocket": pocket,
                        "strategy": str(strategy_tag or "unknown"),
                        "horizon": horizon,
                        "reason": "stale",
                        "source": str(source or "unknown"),
                    },
                )
                return ForecastDecision(
                    allowed=False,
                    scale=0.0,
                    reason="stale",
                    horizon=horizon,
                    edge=edge,
                    p_up=p_up,
                    expected_pips=row.get("expected_pips"),
                    anchor_price=row.get("anchor_price"),
                    target_price=row.get("target_price"),
                    range_low_pips=_safe_optional_float(row.get("range_low_pips")),
                    range_high_pips=_safe_optional_float(row.get("range_high_pips")),
                    range_sigma_pips=_safe_optional_float(row.get("range_sigma_pips")),
                    range_low_price=_safe_optional_float(row.get("range_low_price")),
                    range_high_price=_safe_optional_float(row.get("range_high_price")),
                    tf_confluence_score=(
                        round(float(tf_confluence_score), 6)
                        if tf_confluence_score is not None
                        else None
                    ),
                    tf_confluence_count=int(tf_confluence_count),
                    tf_confluence_horizons=tf_confluence_horizons or None,
                    tp_pips_hint=_safe_abs_float(row.get("tp_pips_hint"), _safe_abs_float(row.get("expected_pips"))),
                    sl_pips_cap=_safe_abs_float(row.get("sl_pips_cap")),
                    rr_floor=_safe_optional_float(row.get("rr_floor")),
                    feature_ts=str(feature_ts),
                    source=str(source) if source is not None else None,
                    style=style,
                    trend_strength=trend_strength,
                    range_pressure=range_pressure,
                    future_flow=future_flow,
                    **regime_profile,
                )
        except Exception:
            # If timestamp parsing fails, do not block; behave as no-op.
            pass

    scale = _scale_from_edge(edge)
    if scale >= 0.999:
        return None
    log_metric(
        "forecast_gate_scale",
        1.0,
        tags={
            "pocket": pocket,
            "strategy": str(strategy_tag or "unknown"),
            "horizon": horizon,
            "reason": "edge_scale",
            "source": str(source or "unknown"),
            "style": style or "n/a",
        },
    )
    return ForecastDecision(
        allowed=True,
        scale=scale,
        reason="edge_scale",
        horizon=horizon,
        edge=edge,
        p_up=p_up,
        expected_pips=row.get("expected_pips"),
        anchor_price=row.get("anchor_price"),
        target_price=row.get("target_price"),
        range_low_pips=_safe_optional_float(row.get("range_low_pips")),
        range_high_pips=_safe_optional_float(row.get("range_high_pips")),
        range_sigma_pips=_safe_optional_float(row.get("range_sigma_pips")),
        range_low_price=_safe_optional_float(row.get("range_low_price")),
        range_high_price=_safe_optional_float(row.get("range_high_price")),
        tf_confluence_score=(
            round(float(tf_confluence_score), 6)
            if tf_confluence_score is not None
            else None
        ),
        tf_confluence_count=int(tf_confluence_count),
        tf_confluence_horizons=tf_confluence_horizons or None,
        tp_pips_hint=_safe_abs_float(row.get("tp_pips_hint"), _safe_abs_float(row.get("expected_pips"))),
        sl_pips_cap=_safe_abs_float(row.get("sl_pips_cap")),
        rr_floor=_safe_optional_float(row.get("rr_floor")),
        feature_ts=row.get("feature_ts"),
        source=str(source) if source is not None else None,
        style=style,
        trend_strength=trend_strength,
        range_pressure=range_pressure,
        future_flow=future_flow,
        **regime_profile,
    )


__all__ = [
    "ForecastDecision",
    "decide",
]

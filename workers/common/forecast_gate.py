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

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
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

_REQUIRE_FRESH = _env_bool("FORECAST_GATE_REQUIRE_FRESH", False)
_MAX_AGE_SEC = max(1.0, _env_float("FORECAST_GATE_MAX_AGE_SEC", 120.0))

_STRATEGY_ALLOWLIST = _env_set("FORECAST_GATE_STRATEGY_ALLOWLIST")
_POCKET_ALLOWLIST = _env_set("FORECAST_GATE_POCKET_ALLOWLIST")

_HORIZON_FORCE = os.getenv("FORECAST_GATE_HORIZON", "").strip()
_HORIZON_SCALP_FAST = os.getenv("FORECAST_GATE_HORIZON_SCALP_FAST", "1h").strip()
_HORIZON_SCALP = os.getenv("FORECAST_GATE_HORIZON_SCALP", "1h").strip()
_HORIZON_MICRO = os.getenv("FORECAST_GATE_HORIZON_MICRO", "8h").strip()
_HORIZON_MACRO = os.getenv("FORECAST_GATE_HORIZON_MACRO", "1d").strip()


@dataclass(frozen=True)
class ForecastDecision:
    allowed: bool
    scale: float
    reason: str
    horizon: str
    edge: float
    p_up: float
    expected_pips: Optional[float] = None
    feature_ts: Optional[str] = None
    source: Optional[str] = None
    style: Optional[str] = None
    trend_strength: Optional[float] = None
    range_pressure: Optional[float] = None


_HORIZON_META_DEFAULT = {
    "1h": {"timeframe": "M5", "step_bars": 12},
    "8h": {"timeframe": "M5", "step_bars": 96},
    "1d": {"timeframe": "H1", "step_bars": 24},
    "1w": {"timeframe": "D1", "step_bars": 5},
    "1m": {"timeframe": "D1", "step_bars": 21},
}
_TF_MINUTES = {"M1": 1.0, "M5": 5.0, "H1": 60.0, "H4": 240.0, "D1": 1440.0}
_TECH_HORIZON_CFG = {
    # shorter horizons allow more mean-reversion influence
    "1h": {"trend_w": 0.56, "mr_w": 0.44, "temp": 1.35},
    "8h": {"trend_w": 0.68, "mr_w": 0.32, "temp": 1.25},
    "1d": {"trend_w": 0.76, "mr_w": 0.24, "temp": 1.10},
    "1w": {"trend_w": 0.82, "mr_w": 0.18, "temp": 1.00},
    "1m": {"trend_w": 0.88, "mr_w": 0.12, "temp": 0.92},
}


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


def _normalize_strategy_key(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _strategy_base(strategy_tag: Optional[str]) -> str:
    if not strategy_tag:
        return ""
    raw = str(strategy_tag).strip()
    if not raw:
        return ""
    return raw.split("-", 1)[0].strip()


def _horizon_meta() -> dict[str, dict[str, object]]:
    meta = dict(_HORIZON_META_DEFAULT)
    try:
        from analysis.forecast_sklearn import DEFAULT_HORIZONS

        for spec in DEFAULT_HORIZONS:
            name = str(getattr(spec, "name", "") or "").strip()
            timeframe = str(getattr(spec, "timeframe", "") or "").strip()
            step_bars = int(getattr(spec, "step_bars", 0) or 0)
            if name and timeframe and step_bars > 0:
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


def _horizon_for(pocket: str, entry_thesis: Optional[dict]) -> str | None:
    if isinstance(entry_thesis, dict):
        hinted = entry_thesis.get("forecast_horizon") or entry_thesis.get("horizon")
        if hinted:
            return str(hinted).strip()
    if _HORIZON_FORCE:
        return _HORIZON_FORCE
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


def _fetch_candles_by_tf() -> dict[str, list[dict]]:
    from indicators.factor_cache import get_candles_snapshot

    return {
        "M5": get_candles_snapshot("M5", limit=1200, include_live=True),
        "H1": get_candles_snapshot("H1", limit=1000, include_live=True),
        "D1": get_candles_snapshot("D1", limit=500, include_live=True),
    }


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
    return out


def _timeframe_minutes(timeframe: str) -> float:
    return float(_TF_MINUTES.get(str(timeframe or "").strip().upper(), 5.0))


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


def _technical_prediction_for_horizon(
    candles: list[dict],
    *,
    horizon: str,
    step_bars: int,
    timeframe: str = "M5",
) -> dict | None:
    if not candles:
        return None
    try:
        from analysis.forecast_sklearn import compute_feature_frame
    except Exception:
        return None

    try:
        feats = compute_feature_frame(candles)
    except Exception as exc:
        LOG.debug("[FORECAST] technical feature build failed horizon=%s err=%s", horizon, exc)
        return None
    if feats.empty:
        return None

    last = feats.iloc[-1]
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

    cfg = _TECH_HORIZON_CFG.get(horizon, _TECH_HORIZON_CFG["8h"])
    combo = (
        cfg["trend_w"] * trend_score * (1.0 - 0.55 * range_pressure)
        + cfg["mr_w"] * mean_revert_score * range_pressure
        + _TECH_PROJECTION_WEIGHT * projection_score
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

    feature_ts = None
    try:
        feature_ts = feats.index[-1].isoformat()
    except Exception:
        feature_ts = None

    return {
        "horizon": horizon,
        "p_up": round(float(p_up), 6),
        "expected_pips": round(float(expected_pips), 4),
        "feature_ts": feature_ts,
        "source": "technical",
        "trend_strength": round(float(trend_strength), 6),
        "range_pressure": round(float(range_pressure), 6),
        "projection_score": round(float(projection_score), 6),
        "projection_confidence": round(float(projection_confidence), 6),
        "projection_components": proj.get("components"),
    }


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


def _blend_prediction_rows(base_row: dict, tech_row: dict) -> dict:
    alpha = _AUTO_BLEND_TECH
    if alpha <= 0.0:
        return base_row
    p_base = _safe_float(base_row.get("p_up"), 0.5)
    p_tech = _safe_float(tech_row.get("p_up"), p_base)
    e_base = _safe_float(base_row.get("expected_pips"), 0.0)
    e_tech = _safe_float(tech_row.get("expected_pips"), e_base)
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
    return {
        **base_row,
        "p_up": round((1.0 - alpha) * p_base + alpha * p_tech, 6),
        "expected_pips": round((1.0 - alpha) * e_base + alpha * e_tech, 4),
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

    horizon = _horizon_for(pocket, entry_thesis)
    if not horizon:
        return None

    bundle = _load_bundle_cached()
    preds = _ensure_predictions(bundle)
    if not isinstance(preds, dict):
        return None

    row = preds.get(horizon)
    if not isinstance(row, dict):
        return None

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
            feature_ts=row.get("feature_ts"),
            source=str(source) if source is not None else None,
            style=style,
            trend_strength=trend_strength,
            range_pressure=range_pressure,
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
            feature_ts=row.get("feature_ts"),
            source=str(source) if source is not None else None,
            style=style,
            trend_strength=trend_strength,
            range_pressure=range_pressure,
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
            feature_ts=row.get("feature_ts"),
            source=str(source) if source is not None else None,
            style=style,
            trend_strength=trend_strength,
            range_pressure=range_pressure,
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
                    feature_ts=str(feature_ts),
                    source=str(source) if source is not None else None,
                    style=style,
                    trend_strength=trend_strength,
                    range_pressure=range_pressure,
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
        feature_ts=row.get("feature_ts"),
        source=str(source) if source is not None else None,
        style=style,
        trend_strength=trend_strength,
        range_pressure=range_pressure,
    )


__all__ = [
    "ForecastDecision",
    "decide",
]

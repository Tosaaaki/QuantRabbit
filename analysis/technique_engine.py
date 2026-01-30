from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from analysis.patterns import detect_latest_n_wave
from analysis.range_model import RangeSnapshot, compute_range_snapshot
from indicators.factor_cache import all_factors, get_candles_snapshot

PIP = 0.01

_TF_ALIASES = {
    "1m": "M1",
    "m1": "M1",
    "5m": "M5",
    "m5": "M5",
    "1h": "H1",
    "h1": "H1",
    "4h": "H4",
    "h4": "H4",
    "1d": "D1",
    "d1": "D1",
}
_VALID_TFS = {"M1", "M5", "H1", "H4", "D1"}

_BASE_TF_BY_POCKET = {
    "macro": "H4",
    "micro": "H1",
    "scalp": "M1",
    "scalp_fast": "M1",
}
_SIGNAL_TF_BY_POCKET = {
    "macro": "H1",
    "micro": "M5",
    "scalp": "M1",
    "scalp_fast": "M1",
}
_LOOKBACK_BY_TF = {
    "M1": 80,
    "M5": 60,
    "H1": 80,
    "H4": 120,
    "D1": 200,
}
_TF_RANK = {"M1": 1, "M5": 2, "H1": 3, "H4": 4, "D1": 5}
_DEFAULT_MTF_TFS = {
    "macro": {
        "fib": ("H4", "H1"),
        "median": ("H4", "H1"),
        "nwave": ("H1", "M5"),
        "candle": ("H1", "M5"),
    },
    "micro": {
        "fib": ("H1", "M5"),
        "median": ("H1", "M5"),
        "nwave": ("M5", "M1"),
        "candle": ("M5", "M1"),
    },
    "scalp": {
        "fib": ("M5", "M1"),
        "median": ("M5", "M1"),
        "nwave": ("M1", "M5"),
        "candle": ("M1", "M5"),
    },
    "scalp_fast": {
        "fib": ("M5", "M1"),
        "median": ("M5", "M1"),
        "nwave": ("M1", "M5"),
        "candle": ("M1", "M5"),
    },
}

_REVERSAL_HINTS = {
    "bbrsi",
    "bb_rsi",
    "range",
    "reversal",
    "revert",
    "fader",
    "fade",
    "vwapbound",
    "vwap_bound",
    "level",
    "mirror",
    "spike",
}
_TREND_HINTS = {
    "trend",
    "momentum",
    "donchian",
    "break",
    "impulse",
    "squeeze",
    "runner",
    "session_open",
    "mtf",
}
_SCALP_HINTS = {"scalp", "m1scalper"}

_REQUIRE_MEDIAN_EPS = float(os.getenv("TECH_REQUIRE_MEDIAN_EPS", "0.03"))

_STRATEGY_POLICY_OVERRIDES: dict[str, dict[str, object]] = {
    # macro trend
    "trendma": {
        "mode": "trend",
        "fib_tf": "H4",
        "median_tf": "H4",
        "nwave_tf": "H1",
        "candle_tf": "H1",
        "min_score": 0.12,
        "min_coverage": 0.6,
        "mid_distance_pips": 8.0,
        "weight_fib": 0.3,
        "weight_median": 0.25,
        "weight_nwave": 0.25,
        "weight_candle": 0.2,
    },
    "donchian55": {
        "mode": "trend",
        "fib_tf": "H4",
        "median_tf": "H4",
        "nwave_tf": "H1",
        "candle_tf": "H1",
        "min_score": 0.12,
        "min_coverage": 0.6,
        "mid_distance_pips": 10.0,
        "weight_fib": 0.3,
        "weight_median": 0.25,
        "weight_nwave": 0.25,
        "weight_candle": 0.2,
    },
    "h1momentum": {
        "mode": "trend",
        "fib_tf": "H1",
        "median_tf": "H1",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.1,
        "min_coverage": 0.55,
        "mid_distance_pips": 5.0,
        "weight_fib": 0.25,
        "weight_median": 0.2,
        "weight_nwave": 0.35,
        "weight_candle": 0.2,
    },
    "trendh1": {
        "mode": "trend",
        "fib_tf": "H1",
        "median_tf": "H1",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.1,
        "min_coverage": 0.55,
        "mid_distance_pips": 5.0,
        "weight_fib": 0.25,
        "weight_median": 0.2,
        "weight_nwave": 0.35,
        "weight_candle": 0.2,
    },
    "londonmomentum": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.55,
        "mid_distance_pips": 2.5,
        "weight_fib": 0.2,
        "weight_median": 0.2,
        "weight_nwave": 0.4,
        "weight_candle": 0.2,
    },
    # scalp
    "m1scalper": {
        "mode": "trend",
        "fib_tf": "M1",
        "median_tf": "M1",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.03,
        "min_coverage": 0.5,
        "weight_fib": 0.15,
        "weight_median": 0.15,
        "weight_nwave": 0.5,
        "weight_candle": 0.2,
        "size_scale": 0.45,
    },
    "pulsebreak": {
        "mode": "trend",
        "fib_tf": "M1",
        "median_tf": "M1",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.06,
        "min_coverage": 0.6,
        "weight_fib": 0.15,
        "weight_median": 0.15,
        "weight_nwave": 0.5,
        "weight_candle": 0.2,
        "size_scale": 0.4,
    },
    "rangefader": {
        "mode": "reversal",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.06,
        "min_coverage": 0.55,
        "mid_distance_pips": 1.8,
        "weight_fib": 0.3,
        "weight_median": 0.25,
        "weight_nwave": 0.15,
        "weight_candle": 0.3,
        "require_median": False,
    },
    "impulseretrace": {
        "mode": "reversal",
        "fib_tf": "M1",
        "median_tf": "M1",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.04,
        "min_coverage": 0.5,
        "weight_fib": 0.3,
        "weight_median": 0.25,
        "weight_nwave": 0.15,
        "weight_candle": 0.3,
        "require_median": False,
    },
    "fastscalp": {
        "mode": "scalp",
        "fib_tf": "M1",
        "median_tf": "M1",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.05,
        "min_coverage": 0.6,
        "weight_fib": 0.2,
        "weight_median": 0.1,
        "weight_nwave": 0.45,
        "weight_candle": 0.25,
        "size_scale": 0.45,
    },
    # micro trend / momentum
    "momentumburst": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.55,
        "mid_distance_pips": 3.5,
        "weight_fib": 0.25,
        "weight_median": 0.2,
        "weight_nwave": 0.35,
        "weight_candle": 0.2,
    },
    "micromomentumstack": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.55,
        "mid_distance_pips": 3.5,
        "weight_fib": 0.25,
        "weight_median": 0.2,
        "weight_nwave": 0.35,
        "weight_candle": 0.2,
    },
    "micropullbackema": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.07,
        "min_coverage": 0.55,
        "mid_distance_pips": 3.5,
        "weight_fib": 0.3,
        "weight_median": 0.2,
        "weight_nwave": 0.3,
        "weight_candle": 0.2,
    },
    "trendmomentummicro": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.55,
        "mid_distance_pips": 3.5,
        "weight_fib": 0.25,
        "weight_median": 0.2,
        "weight_nwave": 0.35,
        "weight_candle": 0.2,
    },
    "microrangebreak": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.55,
        "mid_distance_pips": 3.5,
        "weight_fib": 0.25,
        "weight_median": 0.2,
        "weight_nwave": 0.35,
        "weight_candle": 0.2,
    },
    "momentumpulse": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.55,
        "mid_distance_pips": 3.5,
        "weight_fib": 0.25,
        "weight_median": 0.2,
        "weight_nwave": 0.35,
        "weight_candle": 0.2,
    },
    "volcompressionbreak": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.55,
        "mid_distance_pips": 3.5,
        "weight_fib": 0.25,
        "weight_median": 0.2,
        "weight_nwave": 0.35,
        "weight_candle": 0.2,
    },
    # micro mean-reversion
    "bbrsi": {
        "mode": "reversal",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "mid_distance_pips": 1.8,
        "weight_fib": 0.35,
        "weight_median": 0.3,
        "weight_nwave": 0.15,
        "weight_candle": 0.2,
        "require_median": True,
    },
    "bbrsifast": {
        "mode": "reversal",
        "fib_tf": "M1",
        "median_tf": "M1",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.06,
        "min_coverage": 0.6,
        "weight_fib": 0.35,
        "weight_median": 0.25,
        "weight_nwave": 0.1,
        "weight_candle": 0.3,
        "require_median": True,
    },
    "microvwapbound": {
        "mode": "reversal",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "mid_distance_pips": 1.8,
        "weight_fib": 0.35,
        "weight_median": 0.3,
        "weight_nwave": 0.15,
        "weight_candle": 0.2,
        "require_fib": True,
        "require_median": True,
    },
    "microvwaprevert": {
        "mode": "reversal",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "mid_distance_pips": 1.8,
        "weight_fib": 0.35,
        "weight_median": 0.3,
        "weight_nwave": 0.15,
        "weight_candle": 0.2,
        "require_fib": True,
        "require_median": True,
    },
    "microlevelreactor": {
        "mode": "reversal",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "mid_distance_pips": 1.8,
        "weight_fib": 0.35,
        "weight_median": 0.3,
        "weight_nwave": 0.15,
        "weight_candle": 0.2,
        "require_median": True,
    },
    # impulse / breakout
    "impulsebreaks5": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "weight_fib": 0.15,
        "weight_median": 0.15,
        "weight_nwave": 0.5,
        "weight_candle": 0.2,
        "size_scale": 0.4,
    },
    "impulseretests5": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "weight_fib": 0.15,
        "weight_median": 0.15,
        "weight_nwave": 0.5,
        "weight_candle": 0.2,
        "size_scale": 0.4,
    },
    "impulsemomentums5": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "weight_fib": 0.15,
        "weight_median": 0.15,
        "weight_nwave": 0.5,
        "weight_candle": 0.2,
        "size_scale": 0.4,
    },
    "squeezebreaks5": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "weight_fib": 0.15,
        "weight_median": 0.15,
        "weight_nwave": 0.5,
        "weight_candle": 0.2,
        "size_scale": 0.4,
    },
    "pullbacks5": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "weight_fib": 0.15,
        "weight_median": 0.15,
        "weight_nwave": 0.5,
        "weight_candle": 0.2,
        "size_scale": 0.4,
    },
    "pullbackrunners5": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "weight_fib": 0.15,
        "weight_median": 0.15,
        "weight_nwave": 0.5,
        "weight_candle": 0.2,
        "size_scale": 0.4,
    },
    "pullbackscalp": {
        "mode": "trend",
        "fib_tf": "M1",
        "median_tf": "M1",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.05,
        "min_coverage": 0.6,
        "weight_fib": 0.15,
        "weight_median": 0.15,
        "weight_nwave": 0.5,
        "weight_candle": 0.2,
        "size_scale": 0.4,
    },
    "vwapmagnets5": {
        "mode": "reversal",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "weight_fib": 0.3,
        "weight_median": 0.25,
        "weight_nwave": 0.15,
        "weight_candle": 0.3,
    },
    # reversal spikes
    "mirrorspike": {
        "mode": "reversal",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.1,
        "min_coverage": 0.6,
        "weight_fib": 0.3,
        "weight_median": 0.25,
        "weight_nwave": 0.15,
        "weight_candle": 0.3,
        "require_candle": True,
    },
    "mirrorspikes5": {
        "mode": "reversal",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.1,
        "min_coverage": 0.6,
        "weight_fib": 0.3,
        "weight_median": 0.25,
        "weight_nwave": 0.15,
        "weight_candle": 0.3,
        "require_candle": True,
    },
    "mirrorspiketight": {
        "mode": "reversal",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M5",
        "candle_tf": "M5",
        "min_score": 0.1,
        "min_coverage": 0.6,
        "weight_fib": 0.3,
        "weight_median": 0.25,
        "weight_nwave": 0.15,
        "weight_candle": 0.3,
        "require_candle": True,
    },
    "stoprunreversal": {
        "mode": "reversal",
        "fib_tf": "M1",
        "median_tf": "M1",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.6,
        "weight_fib": 0.3,
        "weight_median": 0.25,
        "weight_nwave": 0.15,
        "weight_candle": 0.3,
        "require_median": True,
    },
    # misc trend
    "volsqueeze": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.55,
    },
    "mtfbreakout": {
        "mode": "trend",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.08,
        "min_coverage": 0.55,
        "mid_distance_pips": 2.5,
        "weight_fib": 0.2,
        "weight_median": 0.2,
        "weight_nwave": 0.4,
        "weight_candle": 0.2,
    },
    "mmlite": {
        "mode": "balanced",
        "fib_tf": "M5",
        "median_tf": "M5",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.05,
        "min_coverage": 0.5,
    },
    "manualswing": {
        "mode": "trend",
        "fib_tf": "H4",
        "median_tf": "H4",
        "nwave_tf": "H1",
        "candle_tf": "H1",
        "min_score": 0.1,
        "min_coverage": 0.55,
    },
}


# Keep __dict__ available for downstream policy introspection.
@dataclass
class TechniquePolicy:
    mode: str
    fib_tf: str
    median_tf: str
    nwave_tf: str
    candle_tf: str
    lookback: int
    method: str
    hi_pct: float
    lo_pct: float
    fib_trigger: float
    fib_deep: float
    mid_distance_pips: float
    min_score: float
    min_coverage: float
    min_positive: int
    require_fib: bool
    require_median: bool
    require_nwave: bool
    require_candle: bool
    weight_fib: float
    weight_median: float
    weight_nwave: float
    weight_candle: float
    size_scale: float
    size_min: float
    size_max: float
    candle_min_conf: float
    nwave_min_quality: float
    nwave_min_leg_pips: float
    exit_min_neg_pips: float
    exit_max_neg_pips: float
    exit_return_score: float


@dataclass(slots=True)
class TechniqueExitDecision:
    should_exit: bool
    reason: Optional[str]
    allow_negative: bool
    debug: Dict[str, object]


@dataclass(slots=True)
class TechniqueEntryDecision:
    allowed: bool
    reason: Optional[str]
    score: Optional[float]
    coverage: Optional[float]
    size_mult: float
    debug: Dict[str, object]


def _normalize_tf(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    alias = _TF_ALIASES.get(text.lower())
    if alias:
        return alias
    upper = text.upper()
    return upper if upper in _VALID_TFS else None


def _normalize_tf_list(values: object) -> list[str]:
    if not values:
        return []
    items: list[object]
    if isinstance(values, str):
        parts = [p.strip() for p in values.replace(";", ",").split(",")]
        items = [p for p in parts if p]
    elif isinstance(values, (list, tuple, set)):
        items = list(values)
    else:
        return []
    tfs: list[str] = []
    for item in items:
        norm = _normalize_tf(str(item))
        if norm and norm not in tfs:
            tfs.append(norm)
    return tfs


def _tf_weight(tf: str, mode: str, *, prefer_lower: bool = False) -> float:
    rank = _TF_RANK.get(tf, 3)
    if prefer_lower:
        return 1.1 - 0.05 * rank
    if mode == "trend":
        return 0.85 + 0.05 * rank
    if mode in {"reversal", "scalp"}:
        return 1.1 - 0.05 * rank
    return 1.0


def _tf_length_scale(tf: str) -> float:
    rank = _TF_RANK.get(tf, 3)
    return 0.8 + 0.2 * rank


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_str(name: str) -> Optional[str]:
    raw = os.getenv(name)
    return raw.strip() if raw is not None else None


def _env_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() not in {"", "0", "false", "no"}


def _common_candle_enabled() -> bool:
    enabled = _env_bool("TECH_COMMON_CANDLE_ENABLED")
    return False if enabled is None else enabled


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        try:
            return int(float(raw))
        except ValueError:
            return None


def _infer_mode(strategy_tag: Optional[str], pocket: str) -> str:
    tag = (strategy_tag or "").strip().lower()
    if any(hint in tag for hint in _SCALP_HINTS):
        return "scalp"
    if any(hint in tag for hint in _REVERSAL_HINTS):
        return "reversal"
    if any(hint in tag for hint in _TREND_HINTS):
        return "trend"
    if pocket in {"scalp", "scalp_fast"}:
        return "scalp"
    if pocket == "macro":
        return "trend"
    if pocket == "micro":
        return "balanced"
    return "balanced"


def _base_policy(mode: str, pocket: str) -> TechniquePolicy:
    base_tf = _BASE_TF_BY_POCKET.get(pocket, "H1")
    signal_tf = _SIGNAL_TF_BY_POCKET.get(pocket, "M5")
    lookback = _LOOKBACK_BY_TF.get(base_tf, 80)
    weights = {
        "trend": (0.2, 0.25, 0.35, 0.2),
        "reversal": (0.35, 0.25, 0.15, 0.25),
        "scalp": (0.2, 0.2, 0.4, 0.2),
        "balanced": (0.25, 0.25, 0.25, 0.25),
    }
    w_fib, w_mid, w_nw, w_cndl = weights.get(mode, weights["balanced"])
    min_score = 0.1 if mode in {"trend", "reversal"} else 0.0
    mid_distance = 1.2 if pocket in {"scalp", "scalp_fast"} else 2.5
    if pocket == "macro":
        mid_distance = 6.0
    min_positive = 2 if pocket in {"macro", "micro"} else 1
    exit_min_neg_pips = 2.0
    if pocket == "scalp_fast":
        exit_min_neg_pips = 1.3
    elif pocket == "scalp":
        exit_min_neg_pips = 1.8
    elif pocket == "micro":
        exit_min_neg_pips = 3.5
    elif pocket == "macro":
        exit_min_neg_pips = 6.0
    exit_max_neg_pips = 0.0
    exit_return_score = -0.25
    if pocket in {"scalp", "scalp_fast"}:
        exit_return_score = -0.35
    elif pocket == "micro":
        exit_return_score = -0.3
    elif pocket == "macro":
        exit_return_score = -0.2
    return TechniquePolicy(
        mode=mode,
        fib_tf=base_tf,
        median_tf=base_tf,
        nwave_tf=signal_tf,
        candle_tf=signal_tf,
        lookback=lookback,
        method="percentile",
        hi_pct=95.0,
        lo_pct=5.0,
        fib_trigger=0.382,
        fib_deep=0.236,
        mid_distance_pips=mid_distance,
        min_score=min_score,
        min_coverage=0.5,
        min_positive=min_positive,
        require_fib=False,
        require_median=False,
        require_nwave=False,
        require_candle=False,
        weight_fib=w_fib,
        weight_median=w_mid,
        weight_nwave=w_nw,
        weight_candle=w_cndl,
        size_scale=0.35,
        size_min=0.6,
        size_max=1.25,
        candle_min_conf=0.35,
        nwave_min_quality=0.18,
        nwave_min_leg_pips=3.0,
        exit_min_neg_pips=exit_min_neg_pips,
        exit_max_neg_pips=exit_max_neg_pips,
        exit_return_score=exit_return_score,
    )


def _normalize_tag_key(tag: str) -> str:
    base = tag.split("-", 1)[0].strip().lower()
    return "".join(ch for ch in base if ch.isalnum())


def _entry_thesis_hint_tfs(label: str, entry_thesis: Optional[dict]) -> list[str]:
    if not isinstance(entry_thesis, dict):
        return []
    env_tf = _normalize_tf(entry_thesis.get("env_tf"))
    struct_tf = _normalize_tf(entry_thesis.get("struct_tf"))
    entry_tf = _normalize_tf(entry_thesis.get("entry_tf"))
    tfs: list[str] = []
    if label in {"fib", "median"}:
        for tf in (env_tf, struct_tf, entry_tf):
            if tf and tf not in tfs:
                tfs.append(tf)
    else:
        for tf in (struct_tf, entry_tf, env_tf):
            if tf and tf not in tfs:
                tfs.append(tf)
    return tfs


def _tag_hint_tfs(strategy_tag: Optional[str]) -> list[str]:
    if not strategy_tag:
        return []
    tag = _normalize_tag_key(str(strategy_tag))
    tfs: list[str] = []
    if "d1" in tag or "daily" in tag:
        tfs.append("D1")
    if "h4" in tag:
        tfs.append("H4")
    if "h1" in tag:
        tfs.append("H1")
    if "m5" in tag or "s5" in tag:
        tfs.append("M5")
    if "m1" in tag or "scalp" in tag:
        tfs.append("M1")
    return tfs


def _resolve_mtf_tfs(
    label: str,
    *,
    policy: TechniquePolicy,
    pocket: str,
    strategy_tag: Optional[str],
    entry_thesis: Optional[dict],
) -> list[str]:
    if label == "candle" and not _common_candle_enabled():
        if not (isinstance(entry_thesis, dict) and entry_thesis.get("tech_allow_candle")):
            return []
    mtf_enabled = _env_bool("TECH_MTF_ENABLED")
    if mtf_enabled is False:
        return [getattr(policy, f"{label}_tf")]

    if isinstance(entry_thesis, dict):
        if entry_thesis.get("tech_tf") or entry_thesis.get(f"tech_tf_{label}"):
            return [getattr(policy, f"{label}_tf")]
        tfs = None
        blob = entry_thesis.get("tech_tfs")
        if isinstance(blob, dict):
            tfs = blob.get(label)
        if tfs is None:
            tfs = entry_thesis.get(f"tech_tfs_{label}") or entry_thesis.get(f"{label}_tfs")
        norm = _normalize_tf_list(tfs)
        if norm:
            return norm

    env_name = f"TECH_MTF_{label.upper()}_TFS"
    specific = None
    if strategy_tag:
        key = _normalize_tag_key(str(strategy_tag)).upper()
        if key:
            specific = _env_str(f"{env_name}_{key}")
    if specific is None:
        specific = _env_str(f"{env_name}_{pocket.upper()}") or _env_str(env_name)
    norm = _normalize_tf_list(specific)
    if norm:
        return norm

    hint = _entry_thesis_hint_tfs(label, entry_thesis)
    if hint:
        return hint

    hinted = _tag_hint_tfs(strategy_tag)
    if hinted:
        return hinted

    defaults = _DEFAULT_MTF_TFS.get(pocket, {})
    tfs = list(defaults.get(label, (getattr(policy, f"{label}_tf"),)))
    base_tf = getattr(policy, f"{label}_tf")
    if base_tf and base_tf not in tfs:
        tfs.insert(0, base_tf)
    return tfs


def _strategy_overrides(strategy_tag: Optional[str]) -> dict[str, object]:
    if not strategy_tag:
        return {}
    key = _normalize_tag_key(str(strategy_tag))
    return dict(_STRATEGY_POLICY_OVERRIDES.get(key, {}))


def _apply_override(policy: TechniquePolicy, overrides: Mapping[str, object]) -> TechniquePolicy:
    data = {k: v for k, v in overrides.items() if v is not None}
    if not data:
        return policy
    updated = replace(policy)
    for key, val in data.items():
        if not hasattr(updated, key):
            continue
        if key.endswith("_tf"):
            norm = _normalize_tf(str(val))
            if norm:
                setattr(updated, key, norm)
            continue
        if isinstance(getattr(updated, key), bool):
            try:
                setattr(updated, key, bool(val))
            except Exception:
                pass
            continue
        try:
            setattr(updated, key, type(getattr(updated, key))(val))
        except Exception:
            continue
    return updated


def _resolve_policy(
    *,
    strategy_tag: Optional[str],
    pocket: str,
    entry_thesis: Optional[dict],
    allow_candle: Optional[bool] = None,
) -> TechniquePolicy:
    mode = _infer_mode(strategy_tag, pocket)
    policy = _base_policy(mode, pocket)

    strategy_override = _strategy_overrides(strategy_tag)
    if strategy_override:
        policy = _apply_override(policy, strategy_override)

    thesis = entry_thesis if isinstance(entry_thesis, dict) else {}
    thesis_policy = thesis.get("tech_policy") if isinstance(thesis, dict) else None
    if isinstance(thesis_policy, dict):
        policy = _apply_override(policy, thesis_policy)

    pocket_upper = pocket.upper()
    if strategy_tag:
        key = "".join(ch for ch in strategy_tag if ch.isalnum()).upper()
    else:
        key = None

    env_overrides = {}
    for field in (
        "mode",
        "fib_tf",
        "median_tf",
        "nwave_tf",
        "candle_tf",
        "lookback",
        "fib_trigger",
        "fib_deep",
        "mid_distance_pips",
        "min_score",
        "min_coverage",
        "min_positive",
        "weight_fib",
        "weight_median",
        "weight_nwave",
        "weight_candle",
        "size_scale",
        "size_min",
        "size_max",
        "candle_min_conf",
        "nwave_min_quality",
        "nwave_min_leg_pips",
        "exit_min_neg_pips",
        "exit_max_neg_pips",
        "exit_return_score",
    ):
        env_name = f"TECH_{field.upper()}"
        specific = None
        if key:
            specific = _env_str(f"{env_name}_{key}")
        if specific is None:
            specific = _env_str(f"{env_name}_{pocket_upper}") or _env_str(env_name)
        if specific is None:
            continue
        env_overrides[field] = specific

    for flag in ("require_fib", "require_median", "require_nwave", "require_candle"):
        env_name = f"TECH_{flag.upper()}"
        specific = None
        if key:
            specific = _env_bool(f"{env_name}_{key}")
        if specific is None:
            specific = _env_bool(f"{env_name}_{pocket_upper}")
        if specific is None:
            specific = _env_bool(env_name)
        if specific is not None:
            env_overrides[flag] = specific

    if env_overrides:
        policy = _apply_override(policy, env_overrides)

    if isinstance(thesis, dict):
        for k in ("tech_tf", "tech_tf_fib", "tech_tf_median", "tech_tf_nwave", "tech_tf_candle"):
            val = thesis.get(k)
            if val:
                norm = _normalize_tf(val)
                if norm:
                    if k == "tech_tf":
                        policy.fib_tf = norm
                        policy.median_tf = norm
                        policy.nwave_tf = norm
                        policy.candle_tf = norm
                    elif k == "tech_tf_fib":
                        policy.fib_tf = norm
                    elif k == "tech_tf_median":
                        policy.median_tf = norm
                    elif k == "tech_tf_nwave":
                        policy.nwave_tf = norm
                    elif k == "tech_tf_candle":
                        policy.candle_tf = norm
    if allow_candle is None:
        allow_candle = _common_candle_enabled()
    if not allow_candle:
        policy = replace(policy, weight_candle=0.0, require_candle=False)
    return policy




def evaluate_entry_techniques(
    *,
    entry_price: float,
    side: str,
    pocket: str,
    strategy_tag: Optional[str] = None,
    entry_thesis: Optional[dict] = None,
    allow_candle: Optional[bool] = None,
) -> TechniqueEntryDecision:
    price_dbg = round(float(entry_price), 5) if entry_price else 0.0
    policy = _resolve_policy(
        strategy_tag=strategy_tag,
        pocket=pocket,
        entry_thesis=entry_thesis,
        allow_candle=allow_candle,
    )

    fib_score = fib_debug = None
    fib_items: list[tuple[str, float, Dict[str, object]]] = []
    fib_tfs = _resolve_mtf_tfs(
        "fib",
        policy=policy,
        pocket=pocket,
        strategy_tag=strategy_tag,
        entry_thesis=entry_thesis,
    )
    axis_override = _axis_from_thesis(entry_thesis or {}) if isinstance(entry_thesis, dict) else None
    for tf in fib_tfs:
        axis = axis_override or _range_from_policy(policy, tf=tf, entry_thesis=entry_thesis)
        if axis:
            score, detail = _score_fib(
                entry_price=entry_price,
                axis=axis,
                side=side,
                mode=policy.mode,
                fib_trigger=policy.fib_trigger,
            )
            if score is not None:
                fib_items.append((tf, score, detail))
    if fib_items:
        fib_score, fib_debug = _blend_tf_scores(fib_items, mode=policy.mode)

    median_score = median_debug = None
    median_items: list[tuple[str, float, Dict[str, object]]] = []
    median_tfs = _resolve_mtf_tfs(
        "median",
        policy=policy,
        pocket=pocket,
        strategy_tag=strategy_tag,
        entry_thesis=entry_thesis,
    )
    for tf in median_tfs:
        axis = axis_override or _range_from_policy(policy, tf=tf, entry_thesis=entry_thesis)
        if axis:
            dist_scale = _tf_length_scale(tf)
            score, detail = _score_median(
                entry_price=entry_price,
                axis=axis,
                side=side,
                mode=policy.mode,
                mid_distance_pips=policy.mid_distance_pips * dist_scale,
            )
            if score is not None:
                median_items.append((tf, score, detail))
    if median_items:
        median_score, median_debug = _blend_tf_scores(median_items, mode=policy.mode)

    nwave_score = nwave_debug = None
    nwave_items: list[tuple[str, float, Dict[str, object]]] = []
    nwave_tfs = _resolve_mtf_tfs(
        "nwave",
        policy=policy,
        pocket=pocket,
        strategy_tag=strategy_tag,
        entry_thesis=entry_thesis,
    )
    for tf in nwave_tfs:
        lookback = _LOOKBACK_BY_TF.get(tf, policy.lookback)
        nwave_candles = get_candles_snapshot(tf, limit=lookback)
        if nwave_candles:
            leg_scale = _tf_length_scale(tf)
            score, detail = _score_nwave(
                candles=nwave_candles,
                side=side,
                min_quality=policy.nwave_min_quality,
                min_leg_pips=policy.nwave_min_leg_pips * leg_scale,
            )
            if score is not None:
                nwave_items.append((tf, score, detail))
    if nwave_items:
        nwave_score, nwave_debug = _blend_tf_scores(
            nwave_items,
            mode=policy.mode,
            prefer_lower=True,
        )

    candle_score = candle_debug = None
    candle_items: list[tuple[str, float, Dict[str, object]]] = []
    if policy.weight_candle > 0:
        candle_tfs = _resolve_mtf_tfs(
            "candle",
            policy=policy,
            pocket=pocket,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis,
        )
        for tf in candle_tfs:
            candle_candles = get_candles_snapshot(tf, limit=4)
            if candle_candles:
                score, detail = _score_candle(
                    candles=candle_candles,
                    side=side,
                    min_conf=policy.candle_min_conf,
                )
                if score is not None:
                    candle_items.append((tf, score, detail))
        if candle_items:
            candle_score, candle_debug = _blend_tf_scores(
                candle_items,
                mode=policy.mode,
                prefer_lower=True,
            )

    weights = [
        ("fib", policy.weight_fib, fib_score),
        ("median", policy.weight_median, median_score),
        ("nwave", policy.weight_nwave, nwave_score),
        ("candle", policy.weight_candle, candle_score),
    ]
    weight_sum = 0.0
    score_sum = 0.0
    pos_count = 0
    neg_count = 0
    for _, weight, score in weights:
        if score is None:
            continue
        weight_sum += weight
        score_sum += weight * score
        if score > 0:
            pos_count += 1
        elif score < 0:
            neg_count += 1

    debug = {
        "price": price_dbg,
        "pos_count": pos_count,
        "neg_count": neg_count,
    }
    if fib_debug:
        debug["fib"] = fib_debug
    if median_debug:
        debug["median"] = median_debug
    if nwave_debug:
        debug["nwave"] = nwave_debug
    if candle_debug:
        debug["candle"] = candle_debug

    score = None
    coverage = None
    if weight_sum > 0:
        score = _clamp(score_sum / weight_sum, -1.0, 1.0)
        coverage = weight_sum / max(
            policy.weight_fib + policy.weight_median + policy.weight_nwave + policy.weight_candle,
            1e-6,
        )
        debug["score"] = round(score, 3)
        debug["coverage"] = round(coverage, 3)

    allowed = True
    reason = None
    if score is None:
        allowed = False
        reason = "no_score"
    elif score < policy.min_score:
        allowed = False
        reason = "low_score"
    if allowed and coverage is not None and coverage < policy.min_coverage:
        allowed = False
        reason = "low_coverage"

    def _entry_require_failed(required: bool, score_val: Optional[float], label: str) -> bool:
        if not required:
            return False
        if score_val is None:
            debug["entry_guard"] = f"{label}_missing"
            return True
        if score_val <= 0:
            debug["entry_guard"] = f"{label}_weak"
            return True
        return False

    if allowed:
        if _entry_require_failed(policy.require_fib, fib_score, "fib"):
            allowed = False
            reason = "fib_required"
        elif _entry_require_failed(policy.require_median, median_score, "median"):
            allowed = False
            reason = "median_required"
        elif _entry_require_failed(policy.require_nwave, nwave_score, "nwave"):
            allowed = False
            reason = "nwave_required"
        elif _entry_require_failed(policy.require_candle, candle_score, "candle"):
            allowed = False
            reason = "candle_required"

    size_mult = 1.0
    if score is not None:
        size_mult = 1.0 + max(0.0, score) * policy.size_scale
        size_mult = max(policy.size_min, min(policy.size_max, size_mult))
    debug["size_mult"] = round(size_mult, 3)

    return TechniqueEntryDecision(
        allowed=allowed,
        reason=reason,
        score=score,
        coverage=coverage,
        size_mult=size_mult,
        debug=debug,
    )
def _axis_from_thesis(thesis: dict) -> Optional[RangeSnapshot]:
    if not isinstance(thesis, dict):
        return None
    axis = thesis.get("section_axis")
    if isinstance(axis, dict):
        try:
            high = float(axis.get("high"))
            low = float(axis.get("low"))
            mid = float(axis.get("mid"))
            if high > low:
                return RangeSnapshot(
                    high=high,
                    low=low,
                    mid=mid,
                    method=str(axis.get("method") or "percentile"),
                    lookback=int(axis.get("lookback") or 0),
                    hi_pct=float(axis.get("hi_pct") or 95.0),
                    lo_pct=float(axis.get("lo_pct") or 5.0),
                    end_time=str(axis.get("end_time") or ""),
                )
        except Exception:
            pass
    rng = thesis.get("range_snapshot")
    if isinstance(rng, dict):
        try:
            high = float(rng.get("high"))
            low = float(rng.get("low"))
            mid = float(rng.get("mid"))
            if high > low:
                return RangeSnapshot(
                    high=high,
                    low=low,
                    mid=mid,
                    method=str(rng.get("method") or "percentile"),
                    lookback=int(rng.get("lookback") or 0),
                    hi_pct=float(rng.get("hi_pct") or 95.0),
                    lo_pct=float(rng.get("lo_pct") or 5.0),
                    end_time=str(rng.get("end_time") or ""),
                )
        except Exception:
            pass
    return None


def _extract_candles(raw: Iterable[dict]) -> list[tuple[float, float, float, float]]:
    candles: list[tuple[float, float, float, float]] = []
    for candle in raw:
        try:
            o = float(candle.get("open"))
            h = float(candle.get("high"))
            l = float(candle.get("low"))
            c = float(candle.get("close"))
        except Exception:
            continue
        candles.append((o, h, l, c))
    return candles


def _detect_candlestick_pattern(candles: Sequence[tuple[float, float, float, float]]) -> Optional[Dict[str, object]]:
    if len(candles) < 2:
        return None
    o0, h0, l0, c0 = candles[-2]
    o1, h1, l1, c1 = candles[-1]
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    range1 = max(h1 - l1, PIP * 0.1)
    upper_wick = h1 - max(o1, c1)
    lower_wick = min(o1, c1) - l1

    if body1 <= range1 * 0.1:
        return {
            "type": "doji",
            "confidence": round(min(1.0, (range1 - body1) / range1), 3),
            "bias": None,
        }

    if (
        c1 > o1
        and c0 < o0
        and c1 >= max(o0, c0)
        and o1 <= min(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bullish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "up",
        }
    if (
        c1 < o1
        and c0 > o0
        and o1 >= min(o0, c0)
        and c1 <= max(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bearish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "down",
        }
    if lower_wick > body1 * 2.5 and upper_wick <= body1 * 0.6:
        return {
            "type": "hammer" if c1 >= o1 else "inverted_hammer",
            "confidence": round(min(1.0, lower_wick / range1 + 0.25), 3),
            "bias": "up",
        }
    if upper_wick > body1 * 2.5 and lower_wick <= body1 * 0.6:
        return {
            "type": "shooting_star" if c1 <= o1 else "hanging_man",
            "confidence": round(min(1.0, upper_wick / range1 + 0.25), 3),
            "bias": "down",
        }
    return None


def _range_from_policy(
    policy: TechniquePolicy,
    *,
    tf: str,
    entry_thesis: Optional[dict],
    axis_override: Optional[RangeSnapshot] = None,
) -> Optional[RangeSnapshot]:
    if axis_override is not None:
        return axis_override
    if isinstance(entry_thesis, dict):
        axis = _axis_from_thesis(entry_thesis)
        if axis is not None:
            return axis
    lookback = _LOOKBACK_BY_TF.get(tf, policy.lookback)
    candles = get_candles_snapshot(tf, limit=lookback)
    if not candles:
        return None
    return compute_range_snapshot(
        candles,
        lookback=lookback,
        method=policy.method,
        hi_pct=policy.hi_pct,
        lo_pct=policy.lo_pct,
    )


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _blend_tf_scores(
    items: Sequence[tuple[str, float, Dict[str, object]]],
    *,
    mode: str,
    prefer_lower: bool = False,
) -> tuple[Optional[float], Dict[str, object]]:
    if not items:
        return None, {}
    total_weight = 0.0
    weighted_sum = 0.0
    signed_sum = 0.0
    details: list[dict] = []
    for tf, score, detail in items:
        weight = _tf_weight(tf, mode, prefer_lower=prefer_lower)
        total_weight += weight
        weighted_sum += score * weight
        if score > 0:
            signed_sum += weight
        elif score < 0:
            signed_sum -= weight
        details.append(
            {
                "tf": tf,
                "score": round(score, 3),
                "detail": detail,
            }
        )
    if total_weight <= 0:
        return None, {}
    avg = weighted_sum / total_weight
    alignment = abs(signed_sum / total_weight)
    blended = avg * alignment
    debug = {
        "blend": {
            "score": round(blended, 3),
            "alignment": round(alignment, 3),
        },
        "tfs": details,
    }
    return _clamp(blended, -1.0, 1.0), debug


def _score_fib(
    *,
    entry_price: float,
    axis: RangeSnapshot,
    side: str,
    mode: str,
    fib_trigger: float,
) -> tuple[Optional[float], Dict[str, object]]:
    span = axis.high - axis.low
    if span <= 0:
        return None, {}
    rel = _clamp((entry_price - axis.low) / span, 0.0, 1.0)
    if mode == "reversal":
        if side == "long":
            score = (fib_trigger - rel) / fib_trigger
        else:
            score = (rel - (1.0 - fib_trigger)) / fib_trigger
    else:
        if side == "long":
            score = (rel - 0.5) / 0.5
        else:
            score = (0.5 - rel) / 0.5
    return _clamp(score, -1.0, 1.0), {
        "rel": round(rel, 3),
        "mode": mode,
        "trigger": round(fib_trigger, 3),
    }


def _score_median(
    *,
    entry_price: float,
    axis: RangeSnapshot,
    side: str,
    mode: str,
    mid_distance_pips: float,
) -> tuple[Optional[float], Dict[str, object]]:
    mid = axis.mid
    dist = abs(entry_price - mid) / PIP
    dist_score = _clamp(dist / max(mid_distance_pips, 0.1), 0.0, 1.0)
    if side == "long":
        preferred = entry_price >= mid if mode == "trend" else entry_price <= mid
    else:
        preferred = entry_price <= mid if mode == "trend" else entry_price >= mid
    score = dist_score if preferred else -dist_score
    return _clamp(score, -1.0, 1.0), {
        "dist_pips": round(dist, 2),
        "mid": round(mid, 5),
        "preferred": preferred,
    }


def _score_nwave(
    *,
    candles: Sequence[dict],
    side: str,
    min_quality: float,
    min_leg_pips: float,
) -> tuple[Optional[float], Dict[str, object]]:
    nwave = detect_latest_n_wave(
        candles,
        min_leg_pips=min_leg_pips,
        min_quality=min_quality,
    )
    if not nwave:
        return None, {}
    match = (side == "long" and nwave.direction == "long") or (
        side == "short" and nwave.direction == "short"
    )
    score = _clamp(nwave.quality, 0.0, 1.0)
    if not match:
        score = -score * 0.7
    return score, {
        "direction": nwave.direction,
        "quality": round(nwave.quality, 3),
    }


def _score_candle(
    *,
    candles: Sequence[dict],
    side: str,
    min_conf: float,
) -> tuple[Optional[float], Dict[str, object]]:
    pattern = _detect_candlestick_pattern(_extract_candles(candles))
    if not pattern:
        return None, {}
    bias = pattern.get("bias")
    conf = float(pattern.get("confidence") or 0.0)
    if conf < min_conf:
        return None, {"type": pattern.get("type"), "confidence": round(conf, 3)}
    if bias is None:
        return 0.0, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": None}
    match = (side == "long" and bias == "up") or (side == "short" and bias == "down")
    score = conf if match else -conf * 0.7
    return _clamp(score, -1.0, 1.0), {
        "type": pattern.get("type"),
        "confidence": round(conf, 3),
        "bias": bias,
    }


def _pivot_levels(candles: Sequence[dict]) -> Optional[Dict[str, float]]:
    if len(candles) < 2:
        return None
    prev = candles[-2]
    try:
        high = float(prev.get("high"))
        low = float(prev.get("low"))
        close = float(prev.get("close"))
    except (TypeError, ValueError):
        return None
    if high <= 0 or low <= 0 or close <= 0:
        return None
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return {"pivot": pivot, "r1": r1, "s1": s1}


def _exit_pivot_tf(strategy_tag: Optional[str], pocket: str, policy: TechniquePolicy) -> Optional[str]:
    key = _normalize_tag_key(str(strategy_tag)).upper() if strategy_tag else None
    pocket_upper = pocket.upper()
    candidate = None
    if key:
        candidate = _env_str(f"TECH_EXIT_PIVOT_TF_{key}")
    if candidate is None:
        candidate = _env_str(f"TECH_EXIT_PIVOT_TF_{pocket_upper}") or _env_str("TECH_EXIT_PIVOT_TF")
    if not candidate:
        candidate = policy.median_tf or policy.fib_tf
    norm = _normalize_tf(str(candidate))
    return norm or policy.median_tf or policy.fib_tf


def _exit_pivot_min_pips(strategy_tag: Optional[str], pocket: str) -> float:
    key = _normalize_tag_key(str(strategy_tag)).upper() if strategy_tag else None
    pocket_upper = pocket.upper()
    value = None
    if key:
        value = _env_float(f"TECH_EXIT_PIVOT_MIN_PIPS_{key}")
    if value is None:
        value = _env_float(f"TECH_EXIT_PIVOT_MIN_PIPS_{pocket_upper}") or _env_float(
            "TECH_EXIT_PIVOT_MIN_PIPS"
        )
    if value is None:
        if pocket in {"scalp", "scalp_fast"}:
            value = 0.8
        elif pocket == "micro":
            value = 1.5
        else:
            value = 3.0
    return max(0.0, float(value))


def _exit_momentum_tf(strategy_tag: Optional[str], pocket: str, policy: TechniquePolicy) -> Optional[str]:
    key = _normalize_tag_key(str(strategy_tag)).upper() if strategy_tag else None
    pocket_upper = pocket.upper()
    candidate = None
    if key:
        candidate = _env_str(f"TECH_EXIT_MOMENTUM_TF_{key}")
    if candidate is None:
        candidate = _env_str(f"TECH_EXIT_MOMENTUM_TF_{pocket_upper}") or _env_str(
            "TECH_EXIT_MOMENTUM_TF"
        )
    if not candidate:
        candidate = policy.median_tf or policy.fib_tf
    norm = _normalize_tf(str(candidate))
    return norm or policy.median_tf or policy.fib_tf


def _exit_momentum_min_score(strategy_tag: Optional[str], pocket: str) -> float:
    key = _normalize_tag_key(str(strategy_tag)).upper() if strategy_tag else None
    pocket_upper = pocket.upper()
    value = None
    if key:
        value = _env_float(f"TECH_EXIT_MOMENTUM_MIN_SCORE_{key}")
    if value is None:
        value = _env_float(f"TECH_EXIT_MOMENTUM_MIN_SCORE_{pocket_upper}") or _env_float(
            "TECH_EXIT_MOMENTUM_MIN_SCORE"
        )
    if value is None:
        if pocket in {"scalp", "scalp_fast"}:
            value = 0.34
        elif pocket == "micro":
            value = 0.34
        else:
            value = 0.3
    return max(0.0, float(value))


def _exit_momentum_rsi_bounds(strategy_tag: Optional[str], pocket: str) -> tuple[float, float]:
    key = _normalize_tag_key(str(strategy_tag)).upper() if strategy_tag else None
    pocket_upper = pocket.upper()
    upper = None
    lower = None
    if key:
        upper = _env_float(f"TECH_EXIT_RSI_UPPER_{key}")
        lower = _env_float(f"TECH_EXIT_RSI_LOWER_{key}")
    if upper is None:
        upper = _env_float(f"TECH_EXIT_RSI_UPPER_{pocket_upper}") or _env_float("TECH_EXIT_RSI_UPPER")
    if lower is None:
        lower = _env_float(f"TECH_EXIT_RSI_LOWER_{pocket_upper}") or _env_float("TECH_EXIT_RSI_LOWER")
    if upper is None:
        upper = 55.0
    if lower is None:
        lower = 45.0
    if upper < lower:
        upper, lower = lower, upper
    return float(upper), float(lower)


def _exit_require_flag(
    flag: str,
    *,
    strategy_tag: Optional[str],
    pocket: str,
    policy_default: bool,
) -> bool:
    key = _normalize_tag_key(str(strategy_tag)).upper() if strategy_tag else None
    pocket_upper = pocket.upper()
    value = None
    if key:
        value = _env_bool(f"TECH_EXIT_REQUIRE_{flag}_{key}")
    if value is None:
        value = _env_bool(f"TECH_EXIT_REQUIRE_{flag}_{pocket_upper}") or _env_bool(
            f"TECH_EXIT_REQUIRE_{flag}"
        )
    if value is None:
        return policy_default
    return bool(value)


def _score_momentum_exit(
    *,
    factors: Dict[str, object],
    side: str,
    rsi_upper: float,
    rsi_lower: float,
) -> tuple[Optional[float], Dict[str, object]]:
    rsi = _to_float(factors.get("rsi"))
    ma10 = _to_float(factors.get("ma10"))
    ma20 = _to_float(factors.get("ma20"))
    macd_hist = _to_float(factors.get("macd_hist"))
    macd = _to_float(factors.get("macd"))
    macd_signal = _to_float(factors.get("macd_signal"))

    scores: list[float] = []
    detail: Dict[str, object] = {
        "rsi": round(rsi, 3) if rsi is not None else None,
        "ma10": round(ma10, 5) if ma10 is not None else None,
        "ma20": round(ma20, 5) if ma20 is not None else None,
        "macd_hist": round(macd_hist, 5) if macd_hist is not None else None,
        "macd": round(macd, 5) if macd is not None else None,
        "macd_signal": round(macd_signal, 5) if macd_signal is not None else None,
    }

    ma_score = None
    if ma10 is not None and ma20 is not None and ma10 != ma20:
        ma_score = 1.0 if ma10 > ma20 else -1.0
    rsi_score = None
    if rsi is not None:
        if rsi >= rsi_upper:
            rsi_score = 1.0
        elif rsi <= rsi_lower:
            rsi_score = -1.0
        else:
            rsi_score = 0.0
    macd_score = None
    if macd_hist is not None:
        if macd_hist > 0:
            macd_score = 1.0
        elif macd_hist < 0:
            macd_score = -1.0
        else:
            macd_score = 0.0
    elif macd is not None and macd_signal is not None:
        if macd > macd_signal:
            macd_score = 1.0
        elif macd < macd_signal:
            macd_score = -1.0
        else:
            macd_score = 0.0

    if ma_score is not None:
        scores.append(ma_score)
    if rsi_score is not None:
        scores.append(rsi_score)
    if macd_score is not None:
        scores.append(macd_score)

    if not scores:
        return None, detail

    support_score = sum(scores) / float(len(scores))
    if side == "short":
        support_score = -support_score
    detail["support_score"] = round(support_score, 3)
    return support_score, detail


def evaluate_exit_techniques(
    *,
    trade: dict,
    current_price: float,
    side: str,
    pocket: str,
) -> TechniqueExitDecision:
    price_dbg = round(float(current_price), 5) if current_price else 0.0
    strategy_tag = None
    if isinstance(trade.get("entry_thesis"), dict):
        strategy_tag = trade.get("entry_thesis", {}).get("strategy_tag")
    if not strategy_tag:
        strategy_tag = trade.get("strategy_tag")
    tag_key = _normalize_tag_key(strategy_tag) if strategy_tag else ""
    policy = _resolve_policy(strategy_tag=strategy_tag, pocket=pocket, entry_thesis=trade.get("entry_thesis"))

    allow_negative = _env_bool("TECH_EXIT_ALLOW_NEGATIVE") or False
    entry_thesis = trade.get("entry_thesis") if isinstance(trade.get("entry_thesis"), dict) else None
    axis_override = _axis_from_thesis(entry_thesis) if isinstance(entry_thesis, dict) else None
    entry_price = None
    try:
        entry_price = float(trade.get("price") or trade.get("entry_price") or 0.0)
    except (TypeError, ValueError):
        entry_price = None
    pnl_pips = None
    if entry_price and entry_price > 0 and current_price:
        pnl_pips = (current_price - entry_price) / PIP if side == "long" else (entry_price - current_price) / PIP

    fib_score = fib_debug = None
    fib_items: list[tuple[str, float, Dict[str, object]]] = []
    fib_tfs = (
        [policy.fib_tf]
        if axis_override
        else _resolve_mtf_tfs(
            "fib",
            policy=policy,
            pocket=pocket,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis,
        )
    )
    for tf in fib_tfs:
        axis = axis_override or _range_from_policy(policy, tf=tf, entry_thesis=entry_thesis)
        if axis:
            score, detail = _score_fib(
                entry_price=current_price,
                axis=axis,
                side=side,
                mode=policy.mode,
                fib_trigger=policy.fib_trigger,
            )
            if score is not None:
                fib_items.append((tf, score, detail))
    if fib_items:
        fib_score, fib_debug = _blend_tf_scores(fib_items, mode=policy.mode)

    median_score = median_debug = None
    median_items: list[tuple[str, float, Dict[str, object]]] = []
    median_tfs = (
        [policy.median_tf]
        if axis_override
        else _resolve_mtf_tfs(
            "median",
            policy=policy,
            pocket=pocket,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis,
        )
    )
    for tf in median_tfs:
        axis = axis_override or _range_from_policy(policy, tf=tf, entry_thesis=entry_thesis)
        if axis:
            dist_scale = _tf_length_scale(tf)
            score, detail = _score_median(
                entry_price=current_price,
                axis=axis,
                side=side,
                mode=policy.mode,
                mid_distance_pips=policy.mid_distance_pips * dist_scale,
            )
            if score is not None:
                median_items.append((tf, score, detail))
    if median_items:
        median_score, median_debug = _blend_tf_scores(median_items, mode=policy.mode)

    nwave_score = nwave_debug = None
    nwave_items: list[tuple[str, float, Dict[str, object]]] = []
    nwave_tfs = _resolve_mtf_tfs(
        "nwave",
        policy=policy,
        pocket=pocket,
        strategy_tag=strategy_tag,
        entry_thesis=entry_thesis,
    )
    for tf in nwave_tfs:
        lookback = _LOOKBACK_BY_TF.get(tf, policy.lookback)
        nwave_candles = get_candles_snapshot(tf, limit=lookback)
        if nwave_candles:
            leg_scale = _tf_length_scale(tf)
            score, detail = _score_nwave(
                candles=nwave_candles,
                side=side,
                min_quality=policy.nwave_min_quality,
                min_leg_pips=policy.nwave_min_leg_pips * leg_scale,
            )
            if score is not None:
                nwave_items.append((tf, score, detail))
    if nwave_items:
        nwave_score, nwave_debug = _blend_tf_scores(
            nwave_items,
            mode=policy.mode,
            prefer_lower=True,
        )

    candle_score = candle_debug = None
    candle_items: list[tuple[str, float, Dict[str, object]]] = []
    if _common_candle_enabled() and policy.weight_candle > 0:
        candle_tfs = _resolve_mtf_tfs(
            "candle",
            policy=policy,
            pocket=pocket,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis,
        )
        for tf in candle_tfs:
            candle_candles = get_candles_snapshot(tf, limit=4)
            if candle_candles:
                score, detail = _score_candle(
                    candles=candle_candles,
                    side=side,
                    min_conf=policy.candle_min_conf,
                )
                if score is not None:
                    candle_items.append((tf, score, detail))
        if candle_items:
            candle_score, candle_debug = _blend_tf_scores(
                candle_items,
                mode=policy.mode,
                prefer_lower=True,
            )

    weights = [
        ("fib", policy.weight_fib, fib_score),
        ("median", policy.weight_median, median_score),
        ("nwave", policy.weight_nwave, nwave_score),
        ("candle", policy.weight_candle, candle_score),
    ]
    weight_sum = 0.0
    score_sum = 0.0
    pos_count = 0
    neg_count = 0
    for _, weight, score in weights:
        if score is None:
            continue
        weight_sum += weight
        score_sum += weight * score
        if score > 0:
            pos_count += 1
        elif score < 0:
            neg_count += 1

    debug = {
        "price": price_dbg,
        "pnl_pips": round(pnl_pips, 3) if pnl_pips is not None else None,
        "exit_min_neg_pips": round(policy.exit_min_neg_pips, 3),
        "exit_return_score": round(policy.exit_return_score, 3),
        "pos_count": pos_count,
        "neg_count": neg_count,
    }
    if fib_debug:
        debug["fib"] = fib_debug
    if median_debug:
        debug["median"] = median_debug
    if nwave_debug:
        debug["nwave"] = nwave_debug
    if candle_debug:
        debug["candle"] = candle_debug

    return_score = None
    coverage = None
    if weight_sum > 0:
        return_score = _clamp(score_sum / weight_sum, -1.0, 1.0)
        coverage = weight_sum / max(
            policy.weight_fib + policy.weight_median + policy.weight_nwave + policy.weight_candle,
            1e-6,
        )
        debug["return_score"] = round(return_score, 3)
        debug["coverage"] = round(coverage, 3)

    exit_strict = _env_bool("TECH_EXIT_STRICT")
    if exit_strict is None:
        exit_strict = True
    exit_require_coverage = _env_bool("TECH_EXIT_REQUIRE_COVERAGE")
    if exit_require_coverage is None:
        exit_require_coverage = True
    exit_min_coverage = _env_float("TECH_EXIT_MIN_COVERAGE")
    if exit_min_coverage is None:
        exit_min_coverage = policy.min_coverage

    reversal_signal = (candle_score is not None and candle_score < 0) or (
        nwave_score is not None and nwave_score < 0
    )
    reversal_combo = (candle_score is not None and candle_score < 0) and (
        nwave_score is not None and nwave_score < 0
    )
    reversal_confirmed = False
    if reversal_signal:
        if reversal_combo:
            reversal_confirmed = True
        elif return_score is not None and return_score <= policy.exit_return_score:
            reversal_confirmed = True
        elif neg_count >= 2 and pos_count == 0:
            reversal_confirmed = True
    debug["reversal_combo"] = reversal_combo
    debug["reversal_confirmed"] = reversal_confirmed
    if tag_key == "m1scalper" and pocket in {"scalp", "scalp_fast"} and reversal_signal:
        coverage_floor = _env_float("M1SCALP_EXIT_MIN_COVERAGE")
        if coverage_floor is None:
            coverage_floor = 0.7
        coverage_pnl = _env_float("M1SCALP_EXIT_COVERAGE_PNL")
        if coverage_pnl is None:
            coverage_pnl = -3.5
        if pnl_pips is not None and pnl_pips <= coverage_pnl and exit_min_coverage > coverage_floor:
            exit_min_coverage = coverage_floor
            debug["exit_min_coverage"] = round(exit_min_coverage, 3)

    allow_negative_reversal = allow_negative
    if pnl_pips is not None and pnl_pips <= -policy.exit_min_neg_pips:
        if return_score is not None and return_score <= policy.exit_return_score:
            allow_negative_reversal = True
        elif neg_count >= 2 and pos_count == 0:
            allow_negative_reversal = True
    if reversal_signal:
        breakeven_guard = max(0.2, policy.exit_min_neg_pips * 0.2)
        if pnl_pips is None:
            allow_negative_reversal = True
        elif pnl_pips <= breakeven_guard:
            allow_negative_reversal = True
        elif reversal_confirmed:
            allow_negative_reversal = True
    if exit_require_coverage:
        if coverage is None or coverage < exit_min_coverage:
            allow_negative_reversal = False
            debug["exit_guard"] = "low_coverage"
    if exit_strict and reversal_signal:
        if not reversal_confirmed and not (
            return_score is not None and return_score <= policy.exit_return_score
        ):
            allow_negative_reversal = False
            debug["exit_guard"] = "reversal_unconfirmed"
    require_mtf_triad = _env_bool("TECH_EXIT_REQUIRE_MTF_TRIAD")
    if require_mtf_triad is None:
        require_mtf_triad = False
    if require_mtf_triad and allow_negative_reversal:
        triad_ok = (
            median_score is not None
            and median_score < 0
            and nwave_score is not None
            and nwave_score < 0
            and candle_score is not None
            and candle_score < 0
        )
        debug["mtf_triad_ok"] = triad_ok
        if not triad_ok:
            allow_negative_reversal = False
            debug["exit_guard"] = "mtf_triad"
    if allow_negative_reversal:
        req_fib = _exit_require_flag(
            "FIB",
            strategy_tag=strategy_tag,
            pocket=pocket,
            policy_default=policy.require_fib,
        )
        req_median = _exit_require_flag(
            "MEDIAN",
            strategy_tag=strategy_tag,
            pocket=pocket,
            policy_default=policy.require_median,
        )
        req_nwave = _exit_require_flag(
            "NWAVE",
            strategy_tag=strategy_tag,
            pocket=pocket,
            policy_default=policy.require_nwave,
        )
        req_candle = _exit_require_flag(
            "CANDLE",
            strategy_tag=strategy_tag,
            pocket=pocket,
            policy_default=policy.require_candle,
        )

        def _exit_require_failed(required: bool, score_val: Optional[float], label: str) -> bool:
            if not required:
                return False
            if score_val is None:
                debug["exit_guard"] = f"{label}_missing"
                return True
            if score_val >= 0:
                debug["exit_guard"] = f"{label}_not_reversal"
                return True
            return False

        if _exit_require_failed(req_fib, fib_score, "fib"):
            allow_negative_reversal = False
        elif _exit_require_failed(req_median, median_score, "median"):
            allow_negative_reversal = False
        elif _exit_require_failed(req_nwave, nwave_score, "nwave"):
            allow_negative_reversal = False
        elif _exit_require_failed(req_candle, candle_score, "candle"):
            allow_negative_reversal = False

    pivot_guard = _env_bool("TECH_EXIT_PIVOT_GUARD")
    if pivot_guard is None:
        pivot_guard = True
    pivot_failopen = _env_bool("TECH_EXIT_PIVOT_FAILOPEN")
    if pivot_failopen is None:
        pivot_failopen = True
    if pivot_guard and allow_negative_reversal:
        pivot_tf = _exit_pivot_tf(strategy_tag, pocket, policy)
        pivot_levels = None
        if pivot_tf:
            pivot_levels = _pivot_levels(get_candles_snapshot(pivot_tf, limit=2))
        if pivot_levels:
            min_pips = _exit_pivot_min_pips(strategy_tag, pocket)
            buffer = min_pips * PIP
            pivot = pivot_levels["pivot"]
            if side == "long":
                pivot_ok = current_price <= pivot - buffer
            else:
                pivot_ok = current_price >= pivot + buffer
            debug["pivot"] = {
                "tf": pivot_tf,
                "pivot": round(pivot, 5),
                "r1": round(pivot_levels["r1"], 5),
                "s1": round(pivot_levels["s1"], 5),
                "min_pips": round(min_pips, 3),
                "ok": pivot_ok,
            }
            if not pivot_ok:
                allow_negative_reversal = False
                debug["exit_guard"] = "pivot_block"
        elif not pivot_failopen:
            allow_negative_reversal = False
            debug["exit_guard"] = "pivot_missing"

    skip_momentum_guard = False
    if allow_negative_reversal and reversal_combo and reversal_confirmed:
        tag_key = _normalize_tag_key(strategy_tag) if strategy_tag else ""
        if tag_key == "m1scalper" and pocket in {"scalp", "scalp_fast"}:
            bypass = _env_bool("TECH_EXIT_MOMENTUM_BYPASS_M1SCALPER")
            if bypass is None:
                bypass = True
            if bypass:
                skip_momentum_guard = True
                debug["momentum_guard"] = "bypass_m1scalper"

    momentum_guard = _env_bool("TECH_EXIT_MOMENTUM_GUARD")
    if momentum_guard is None:
        momentum_guard = True
    momentum_failopen = _env_bool("TECH_EXIT_MOMENTUM_FAILOPEN")
    if momentum_failopen is None:
        momentum_failopen = True
    if momentum_guard and allow_negative_reversal and not skip_momentum_guard:
        momentum_tf = _exit_momentum_tf(strategy_tag, pocket, policy)
        fac = {}
        try:
            fac = (all_factors().get(momentum_tf) or {}) if momentum_tf else {}
        except Exception:
            fac = {}
        rsi_upper, rsi_lower = _exit_momentum_rsi_bounds(strategy_tag, pocket)
        score, detail = _score_momentum_exit(
            factors=fac,
            side=side,
            rsi_upper=rsi_upper,
            rsi_lower=rsi_lower,
        )
        min_score = _exit_momentum_min_score(strategy_tag, pocket)
        if score is not None:
            against_ok = score <= -min_score
            debug["momentum"] = {
                **detail,
                "tf": momentum_tf,
                "min_score": round(min_score, 3),
                "ok": against_ok,
            }
            if not against_ok:
                allow_negative_reversal = False
                debug["exit_guard"] = "momentum_block"
        elif not momentum_failopen:
            allow_negative_reversal = False
            debug["exit_guard"] = "momentum_missing"

    if (
        policy.exit_max_neg_pips
        and pnl_pips is not None
        and pnl_pips < 0
        and abs(pnl_pips) > policy.exit_max_neg_pips
    ):
        allow_negative_reversal = False
        debug["exit_guard"] = "max_neg_pips"
        debug["exit_max_neg_pips"] = round(policy.exit_max_neg_pips, 3)

    debug["reversal_allow_negative"] = allow_negative_reversal

    def _neg_blocked() -> bool:
        return pnl_pips is not None and pnl_pips < 0 and not allow_negative_reversal

    if reversal_signal and reversal_confirmed:
        reason = "tech_candle_reversal" if candle_score is not None and candle_score < 0 else "tech_nwave_flip"
        if reversal_combo:
            reason = "tech_reversal_combo"
        if _neg_blocked():
            return TechniqueExitDecision(False, None, False, debug)
        return TechniqueExitDecision(True, reason, allow_negative_reversal, debug)

    if pnl_pips is None or pnl_pips > -policy.exit_min_neg_pips:
        return TechniqueExitDecision(False, None, False, {})

    if return_score is not None and return_score <= policy.exit_return_score:
        if _neg_blocked():
            return TechniqueExitDecision(False, None, False, debug)
        return TechniqueExitDecision(True, "tech_return_fail", allow_negative_reversal, debug)

    return TechniqueExitDecision(False, None, False, {})

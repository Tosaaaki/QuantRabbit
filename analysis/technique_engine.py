from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from analysis.patterns import detect_latest_n_wave
from analysis.range_model import RangeSnapshot, compute_range_snapshot
from indicators.factor_cache import get_candles_snapshot

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
_SCALP_HINTS = {"scalp", "m1scalper", "onepip"}

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
        "min_score": 0.05,
        "min_coverage": 0.6,
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
        "min_score": 0.08,
        "min_coverage": 0.6,
        "mid_distance_pips": 1.8,
        "weight_fib": 0.3,
        "weight_median": 0.25,
        "weight_nwave": 0.15,
        "weight_candle": 0.3,
        "require_median": True,
    },
    "impulseretrace": {
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
    "onepipmakers1": {
        "mode": "reversal",
        "fib_tf": "M1",
        "median_tf": "M1",
        "nwave_tf": "M1",
        "candle_tf": "M1",
        "min_score": 0.03,
        "min_coverage": 0.55,
        "weight_fib": 0.3,
        "weight_median": 0.3,
        "weight_nwave": 0.1,
        "weight_candle": 0.3,
        "size_scale": 0.2,
        "require_median": True,
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
    exit_return_score: float


@dataclass(slots=True)
class TechniqueDecision:
    allowed: bool
    score: float
    size_multiplier: float
    reasons: Sequence[str]
    debug: Dict[str, object]


@dataclass(slots=True)
class TechniqueExitDecision:
    should_exit: bool
    reason: Optional[str]
    allow_negative: bool
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


def _env_str(name: str) -> Optional[str]:
    raw = os.getenv(name)
    return raw.strip() if raw is not None else None


def _env_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() not in {"", "0", "false", "no"}


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return float(raw)
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
        exit_min_neg_pips = 1.0
    elif pocket == "scalp":
        exit_min_neg_pips = 1.5
    elif pocket == "micro":
        exit_min_neg_pips = 3.0
    elif pocket == "macro":
        exit_min_neg_pips = 6.0
    exit_return_score = -0.25
    if pocket in {"scalp", "scalp_fast"}:
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
    if "m1" in tag or "scalp" in tag or "onepip" in tag:
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
    return policy


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


def evaluate_entry_techniques(
    *,
    entry_price: float,
    side: str,
    pocket: str,
    strategy_tag: Optional[str],
    entry_thesis: Optional[dict],
) -> TechniqueDecision:
    policy = _resolve_policy(strategy_tag=strategy_tag, pocket=pocket, entry_thesis=entry_thesis)
    debug: Dict[str, object] = {"mode": policy.mode}
    reasons: list[str] = []

    axis_override = _axis_from_thesis(entry_thesis) if isinstance(entry_thesis, dict) else None
    axis_cache: dict[str, Optional[RangeSnapshot]] = {}

    def _axis_for(tf: str) -> Optional[RangeSnapshot]:
        if axis_override is not None:
            return axis_override
        if tf not in axis_cache:
            axis_cache[tf] = _range_from_policy(
                policy,
                tf=tf,
                entry_thesis=entry_thesis,
            )
        return axis_cache[tf]

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
        axis = _axis_for(tf)
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
    if fib_debug:
        debug["fib"] = fib_debug

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
        axis = _axis_for(tf)
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
    if median_debug:
        debug["median"] = median_debug

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
    if nwave_debug:
        debug["nwave"] = nwave_debug

    candle_score = candle_debug = None
    candle_items: list[tuple[str, float, Dict[str, object]]] = []
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
    if candle_debug:
        debug["candle"] = candle_debug

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
    for name, weight, score in weights:
        if score is None:
            continue
        weight_sum += weight
        score_sum += weight * score
        if score > 0:
            pos_count += 1
            reasons.append(f"{name}_ok")
        elif score < 0:
            neg_count += 1
            reasons.append(f"{name}_ng")

    coverage = weight_sum / max(policy.weight_fib + policy.weight_median + policy.weight_nwave + policy.weight_candle, 1e-6)
    debug["coverage"] = round(coverage, 3)

    def _require_failed(required: bool, score_val: Optional[float], label: str) -> bool:
        if not required:
            return False
        if score_val is None or score_val <= 0:
            reasons.append(f"require_{label}_fail")
            return True
        return False

    if _require_failed(policy.require_fib, fib_score, "fib"):
        return TechniqueDecision(False, -1.0, 1.0, reasons, debug)
    if _require_failed(policy.require_median, median_score, "median"):
        return TechniqueDecision(False, -1.0, 1.0, reasons, debug)
    if _require_failed(policy.require_nwave, nwave_score, "nwave"):
        return TechniqueDecision(False, -1.0, 1.0, reasons, debug)
    if _require_failed(policy.require_candle, candle_score, "candle"):
        return TechniqueDecision(False, -1.0, 1.0, reasons, debug)

    if weight_sum <= 0:
        return TechniqueDecision(True, 0.0, 1.0, reasons, debug)

    score = _clamp(score_sum / weight_sum, -1.0, 1.0)
    debug["score"] = round(score, 3)
    debug["pos_count"] = pos_count
    debug["neg_count"] = neg_count
    debug["min_positive"] = policy.min_positive
    if coverage < policy.min_coverage:
        reasons.append("low_coverage")
        return TechniqueDecision(True, score, 1.0, reasons, debug)
    hard_block_score = -0.35
    hard_block_neg = 3

    def _hard_block(score_val: float, neg_count_val: int, pos_count_val: int) -> bool:
        if neg_count_val >= hard_block_neg:
            return True
        if pos_count_val == 0 and score_val <= hard_block_score:
            return True
        return False

    if pos_count < policy.min_positive:
        if _hard_block(score, neg_count, pos_count):
            reasons.append("min_positive_block")
            return TechniqueDecision(False, score, 1.0, reasons, debug)
        reasons.append("min_positive_soft")
        return TechniqueDecision(True, score, policy.size_min, reasons, debug)
    if score < policy.min_score:
        if _hard_block(score, neg_count, pos_count):
            reasons.append("min_score_block")
            return TechniqueDecision(False, score, 1.0, reasons, debug)
        reasons.append("min_score_soft")
        return TechniqueDecision(True, score, policy.size_min, reasons, debug)

    multiplier = _clamp(1.0 + score * policy.size_scale, policy.size_min, policy.size_max)
    return TechniqueDecision(True, score, multiplier, reasons, debug)


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

    reversal_signal = (candle_score is not None and candle_score < 0) or (
        nwave_score is not None and nwave_score < 0
    )
    allow_negative_reversal = allow_negative
    if pnl_pips is not None and pnl_pips <= -policy.exit_min_neg_pips:
        if return_score is not None and return_score <= policy.exit_return_score:
            allow_negative_reversal = True
        elif neg_count >= 2 and pos_count == 0:
            allow_negative_reversal = True

    if reversal_signal:
        reason = "tech_candle_reversal" if candle_score is not None and candle_score < 0 else "tech_nwave_flip"
        if (candle_score is not None and candle_score < 0) and (nwave_score is not None and nwave_score < 0):
            reason = "tech_reversal_combo"
        return TechniqueExitDecision(True, reason, allow_negative_reversal, debug)

    if pnl_pips is None or pnl_pips > -policy.exit_min_neg_pips:
        return TechniqueExitDecision(False, None, False, {})

    if return_score is not None and return_score <= policy.exit_return_score:
        return TechniqueExitDecision(True, "tech_return_fail", True, debug)

    return TechniqueExitDecision(False, None, False, {})

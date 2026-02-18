"""Strategy-facing entry helpers.

Strategies submit orders through this module. Before dispatching, it coordinates
entry intent via order_manager's blackboard to keep strategy-level intent and avoid
duplicate cross-strategy overexposure.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import os
from typing import Any, Iterable, Literal, Optional

from analysis import strategy_feedback
from indicators.factor_cache import all_factors
from execution.order_manager import cancel_order, close_trade, set_trade_protections
from execution import order_manager
from workers.common import forecast_gate, pattern_gate


def get_last_order_status_by_client_id(
    client_order_id: Optional[str],
) -> Optional[dict[str, object]]:
    """Compatibility wrapper retained for existing strategy imports."""
    return order_manager.get_last_order_status_by_client_id(client_order_id)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


_ENTRY_TECH_CONTEXT_ENABLED = _env_bool("ENTRY_TECH_CONTEXT_ENABLED", True)
_ENTRY_TECH_CONTEXT_STRATEGY_REQUIREMENTS = _env_bool(
    "ENTRY_TECH_CONTEXT_STRATEGY_REQUIREMENTS",
    False,
)
_STRATEGY_PATTERN_GATE_ENABLED = _env_bool("STRATEGY_PATTERN_GATE_ENABLED", True)
_STRATEGY_PATTERN_GATE_AUTO_OPT_IN = _env_bool(
    "STRATEGY_PATTERN_GATE_AUTO_OPT_IN",
    False,
)
_ORDER_PATTERN_GATE_SCALE_TO_MIN_UNITS = _env_bool(
    "ORDER_PATTERN_GATE_SCALE_TO_MIN_UNITS",
    False,
)
_STRATEGY_FORECAST_CONTEXT_ENABLED = _env_bool(
    "STRATEGY_FORECAST_CONTEXT_ENABLED",
    True,
)
_STRATEGY_FORECAST_FUSION_ENABLED = _env_bool(
    "STRATEGY_FORECAST_FUSION_ENABLED",
    True,
)
_STRATEGY_FORECAST_FUSION_UNITS_CUT_MAX = max(
    0.0,
    min(0.95, _env_float("STRATEGY_FORECAST_FUSION_UNITS_CUT_MAX", 0.65)),
)
_STRATEGY_FORECAST_FUSION_UNITS_BOOST_MAX = max(
    0.0,
    min(0.5, _env_float("STRATEGY_FORECAST_FUSION_UNITS_BOOST_MAX", 0.20)),
)
_STRATEGY_FORECAST_FUSION_UNITS_MIN_SCALE = max(
    0.05,
    min(1.0, _env_float("STRATEGY_FORECAST_FUSION_UNITS_MIN_SCALE", 0.25)),
)
_STRATEGY_FORECAST_FUSION_UNITS_MAX_SCALE = max(
    _STRATEGY_FORECAST_FUSION_UNITS_MIN_SCALE,
    min(2.0, _env_float("STRATEGY_FORECAST_FUSION_UNITS_MAX_SCALE", 1.20)),
)
_STRATEGY_FORECAST_FUSION_DISALLOW_UNITS_MULT = max(
    0.0,
    min(1.0, _env_float("STRATEGY_FORECAST_FUSION_DISALLOW_UNITS_MULT", 0.65)),
)
_STRATEGY_FORECAST_FUSION_PROB_GAIN = max(
    0.0,
    min(0.5, _env_float("STRATEGY_FORECAST_FUSION_PROB_GAIN", 0.22)),
)
_STRATEGY_FORECAST_FUSION_DISALLOW_PROB_MULT = max(
    0.0,
    min(1.0, _env_float("STRATEGY_FORECAST_FUSION_DISALLOW_PROB_MULT", 0.70)),
)
_STRATEGY_FORECAST_FUSION_SYNTH_PROB_IF_MISSING = _env_bool(
    "STRATEGY_FORECAST_FUSION_SYNTH_PROB_IF_MISSING",
    True,
)
_STRATEGY_FORECAST_FUSION_TP_BLEND_ENABLED = _env_bool(
    "STRATEGY_FORECAST_FUSION_TP_BLEND_ENABLED",
    True,
)
_STRATEGY_FORECAST_FUSION_TP_BLEND_BASE = max(
    0.0,
    min(1.0, _env_float("STRATEGY_FORECAST_FUSION_TP_BLEND_BASE", 0.20)),
)
_STRATEGY_FORECAST_FUSION_TP_BLEND_EDGE_GAIN = max(
    0.0,
    min(1.0, _env_float("STRATEGY_FORECAST_FUSION_TP_BLEND_EDGE_GAIN", 0.40)),
)
_STRATEGY_FORECAST_FUSION_TF_CUT_MAX = max(
    0.0,
    min(0.9, _env_float("STRATEGY_FORECAST_FUSION_TF_CUT_MAX", 0.35)),
)
_STRATEGY_FORECAST_FUSION_TF_BOOST_MAX = max(
    0.0,
    min(0.5, _env_float("STRATEGY_FORECAST_FUSION_TF_BOOST_MAX", 0.12)),
)
_STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED = _env_bool(
    "STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED",
    True,
)
_STRATEGY_FORECAST_FUSION_STRONG_CONTRA_PROB_MAX = max(
    0.0,
    min(0.5, _env_float("STRATEGY_FORECAST_FUSION_STRONG_CONTRA_PROB_MAX", 0.22)),
)
_STRATEGY_FORECAST_FUSION_STRONG_CONTRA_EDGE_MIN = max(
    0.0,
    min(1.0, _env_float("STRATEGY_FORECAST_FUSION_STRONG_CONTRA_EDGE_MIN", 0.65)),
)
_PATTERN_GATE_META_KEYS = ("pattern_gate_opt_in", "use_pattern_gate", "pattern_gate_enabled")


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        raw = default
    out: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(token.upper())
    return out


_TECH_DEFAULT_TFS_BY_POCKET = {
    "macro": ("D1", "H4", "H1", "M5", "M1"),
    "micro": ("H4", "H1", "M5", "M1"),
    "scalp": ("H1", "M5", "M1"),
    "scalp_fast": ("H1", "M5", "M1"),
    "manual": ("H4", "H1", "M5", "M1"),
}
_TECH_ALL_KNOWN_TFS = ("D1", "H4", "H1", "M5", "M1")
_DEFAULT_ENTRY_TECH_TFS = _env_csv(
    "ENTRY_TECH_DEFAULT_TFS",
    ",".join(_TECH_ALL_KNOWN_TFS),
)

_TECH_POLICY_REQUIRE_ALL = {}
_FORECAST_SUPPORT_DEFAULT_BY_HORIZON: dict[str, tuple[str, ...]] = {
    "1m": ("5m", "10m"),
    "5m": ("1m", "10m"),
    "10m": ("5m", "1h"),
    "1h": ("10m", "8h"),
    "8h": ("1h", "1d"),
    "1d": ("8h", "1w"),
}


def _coerce_bool(value: object, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 0.0):
            return False
        if value in (1, 1.0):
            return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _has_pattern_gate_optin(entry_thesis: Optional[dict], meta: Optional[dict]) -> bool:
    for container in (entry_thesis, meta):
        if not isinstance(container, dict):
            continue
        if any(key in container for key in _PATTERN_GATE_META_KEYS):
            return True
    return False


def _normalize_env_prefix(value: object) -> Optional[str]:
    if value is None:
        return None
    value_text = str(value).strip()
    if not value_text:
        return None
    return value_text.upper()


def _infer_env_prefix_from_strategy_tag(strategy_tag: Optional[str]) -> Optional[str]:
    tag = str(strategy_tag or "").strip().upper()
    if not tag:
        return None
    if tag.startswith("SCALP_PING_5S_B"):
        return "SCALP_PING_5S_B"
    if tag.startswith("SCALP_PING_5S"):
        return "SCALP_PING_5S"
    return None


def _normalize_pattern_gate_meta(
    entry_thesis: Optional[dict], meta: Optional[dict]
) -> Optional[dict]:
    if _has_pattern_gate_optin(entry_thesis, meta):
        return meta if isinstance(meta, dict) else None
    if not _STRATEGY_PATTERN_GATE_AUTO_OPT_IN:
        return meta if isinstance(meta, dict) else None
    if isinstance(meta, dict):
        injected = dict(meta)
    else:
        injected = {}
    injected["pattern_gate_opt_in"] = True
    return injected


def _coalesce_env_prefix(
    entry_thesis: Optional[dict],
    meta: Optional[dict],
    strategy_tag: Optional[str] = None,
) -> Optional[str]:
    inferred_env_prefix = _infer_env_prefix_from_strategy_tag(strategy_tag)
    if inferred_env_prefix is not None:
        return inferred_env_prefix

    for container in (entry_thesis, meta):
        if not isinstance(container, dict):
            continue
        value = _normalize_env_prefix(container.get("env_prefix") or container.get("ENV_PREFIX"))
        if not value:
            continue
        return value
    return None


def _inject_env_prefix_context(
    entry_thesis: Optional[dict],
    meta: Optional[dict],
    strategy_tag: Optional[str] = None,
) -> tuple[dict, dict]:
    env_prefix = _coalesce_env_prefix(entry_thesis, meta, strategy_tag)
    normalized_entry_thesis: dict = dict(entry_thesis) if isinstance(entry_thesis, dict) else {}
    normalized_meta: dict = dict(meta) if isinstance(meta, dict) else {}
    if env_prefix:
        normalized_entry_thesis["env_prefix"] = env_prefix
        normalized_entry_thesis["ENV_PREFIX"] = env_prefix
        normalized_meta["env_prefix"] = env_prefix
        normalized_meta["ENV_PREFIX"] = env_prefix
    return normalized_entry_thesis, normalized_meta


_STRATEGY_TECH_CONTEXT_REQUIREMENTS: dict[str, dict[str, object]] = {
    "SCALP_PING_5S": {
        "technical_context_tfs": ["M1", "M5", "H1", "H4"],
        "technical_context_fields": ["ma10", "ma20", "rsi", "atr", "atr_pips", "adx", "bbw", "macd", "ema12", "ema20", "ema24"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
        "technical_context_candle_counts": {"M1": 140, "M5": 90, "H1": 70, "H4": 40},
        "forecast_horizon": "1m",
        "forecast_profile": {"timeframe": "M1", "step_bars": 1},
        "forecast_support_horizons": ["5m", "10m"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
    "SCALP_M1SCALPER": {
        "technical_context_tfs": ["M1", "M5", "H1"],
        "technical_context_fields": ["ma10", "ma20", "ema12", "ema20", "ema24", "rsi", "atr", "atr_pips", "adx", "bbw", "macd"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
        "technical_context_candle_counts": {"M1": 120, "M5": 80, "H1": 60},
        "forecast_horizon": "5m",
        "forecast_profile": {"timeframe": "M1", "step_bars": 5},
        "forecast_support_horizons": ["1m", "10m"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
    "SCALP_MACD_RSI_DIV": {
        "technical_context_tfs": ["M1", "M5", "H1", "H4"],
        "technical_context_fields": ["ma10", "ma20", "ema12", "ema24", "rsi", "atr", "atr_pips", "adx", "bbw", "macd", "macd_hist"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
        "technical_context_candle_counts": {"M1": 160, "M5": 100, "H1": 70, "H4": 40},
        "forecast_horizon": "10m",
        "forecast_profile": {"timeframe": "M5", "step_bars": 2},
        "forecast_support_horizons": ["5m", "1h"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
    "SCALP_TICK_IMBALANCE": {
        "technical_context_tfs": ["M1", "M5", "H1"],
        "technical_context_fields": ["ma10", "ma20", "ema12", "ema20", "rsi", "atr", "atr_pips", "adx", "bbw", "macd"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips", "tick_rate"],
        "technical_context_candle_counts": {"M1": 160, "M5": 90, "H1": 50},
        "forecast_horizon": "5m",
        "forecast_profile": {"timeframe": "M1", "step_bars": 5},
        "forecast_support_horizons": ["1m", "10m"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
    "SCALP_PING_5S_B": {
        "technical_context_tfs": ["M1", "M5", "H1", "H4"],
        "technical_context_fields": ["ma10", "ma20", "rsi", "atr", "atr_pips", "adx", "bbw", "macd", "ema12", "ema20", "ema24"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
        "technical_context_candle_counts": {"M1": 140, "M5": 90, "H1": 70, "H4": 40},
        "forecast_horizon": "1m",
        "forecast_profile": {"timeframe": "M1", "step_bars": 1},
        "forecast_support_horizons": ["5m", "10m"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
    "SCALP_WICK_REVERSAL_BLEND": {
        "technical_context_tfs": ["M1", "M5", "H1", "H4"],
        "technical_context_fields": ["ma10", "ma20", "rsi", "atr", "atr_pips", "adx", "bbw", "macd", "ema12", "ema20", "ema24"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
        "technical_context_candle_counts": {"M1": 140, "M5": 90, "H1": 70, "H4": 40},
        "forecast_horizon": "10m",
        "forecast_profile": {"timeframe": "M5", "step_bars": 2},
        "forecast_support_horizons": ["5m", "1h"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
    "SCALP_WICK_REVERSAL_PRO": {
        "technical_context_tfs": ["M1", "M5", "H1", "H4"],
        "technical_context_fields": ["ma10", "ma20", "rsi", "atr", "atr_pips", "adx", "bbw", "macd", "ema12", "ema20", "ema24"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
        "technical_context_candle_counts": {"M1": 140, "M5": 90, "H1": 70, "H4": 40},
        "forecast_horizon": "10m",
        "forecast_profile": {"timeframe": "M5", "step_bars": 2},
        "forecast_support_horizons": ["5m", "1h"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
    "SCALP_SQUEEZE_PULSE_BREAK": {
        "technical_context_tfs": ["M1", "M5", "H1", "H4"],
        "technical_context_fields": ["ma10", "ma20", "rsi", "atr", "atr_pips", "adx", "bbw", "macd", "ema12", "ema20", "ema24"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
        "technical_context_candle_counts": {"M1": 140, "M5": 90, "H1": 70, "H4": 40},
        "forecast_horizon": "10m",
        "forecast_profile": {"timeframe": "M5", "step_bars": 2},
        "forecast_support_horizons": ["5m", "1h"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
    "MICRO_ADAPTIVE_REVERT": {
        "technical_context_tfs": ["M1", "M5", "H1"],
        "technical_context_fields": ["ma10", "ma20", "ema12", "ema20", "rsi", "atr", "atr_pips", "adx", "bbw", "macd"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
        "technical_context_candle_counts": {"M5": 120, "M1": 80, "H1": 50},
        "forecast_horizon": "10m",
        "forecast_profile": {"timeframe": "M5", "step_bars": 2},
        "forecast_support_horizons": ["5m", "1h"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
    "MICRO_MULTISTRAT": {
        "technical_context_tfs": ["M5", "M1", "H1"],
        "technical_context_fields": ["ma10", "ma20", "ema12", "ema20", "rsi", "atr", "atr_pips", "adx", "bbw", "macd", "volume"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
        "technical_context_candle_counts": {"M5": 120, "M1": 140, "H1": 60},
        "forecast_horizon": "10m",
        "forecast_profile": {"timeframe": "M5", "step_bars": 2},
        "forecast_support_horizons": ["5m", "1h"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
    "SESSION_OPEN": {
        "technical_context_tfs": ["M1", "M5", "H1"],
        "technical_context_fields": ["ma10", "ma20", "ema12", "ema24", "atr", "atr_pips", "adx", "bbw", "rsi", "macd"],
        "technical_context_ticks": ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
        "technical_context_candle_counts": {"M1": 120, "M5": 90, "H1": 60},
        "forecast_horizon": "10m",
        "forecast_profile": {"timeframe": "M5", "step_bars": 2},
        "forecast_support_horizons": ["5m", "1h"],
        "forecast_technical_only": True,
        "tech_policy": dict(_TECH_POLICY_REQUIRE_ALL),
    },
}

_STRATEGY_TECH_CONTEXT_REQUIREMENTS.update(
    {
        "SCALP_TICK_IMBALANCE_RRPLUS": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_TICK_IMBALANCE"]),
        "SCALP_TICK_WICK_REVERSAL": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_WICK_REVERSAL_BLEND"]),
        "SCALP_WICK_REVERSAL": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_WICK_REVERSAL_BLEND"]),
        "SCALP_WICK_REVERSAL_HF": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_WICK_REVERSAL_BLEND"]),
        "SCALP_TICK_WICK_REVERSAL_HF": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_WICK_REVERSAL_BLEND"]),
        "SCALP_LEVEL_REJECT": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_TICK_IMBALANCE"]),
        "SCALP_LEVEL_REJECT_PLUS": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_TICK_IMBALANCE"]),
        "MICRO_RANGEBREAK": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_VWAPBOUND": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_VWAPREVERT": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_MOMENTUMBURST": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_MOMENTUMSTACK": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_PULLBACKEMA": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_LEVELREACTOR": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_TRENDMOMENTUM": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_TRENDRETEST": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_COMPRESSIONREVERT": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_MOMENTUMPULSE": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "TECH_FUSION": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MACRO_TECH_FUSION": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_MACD_RSI_DIV"]),
        "RANGE_COMPRESSION_BREAK": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "MICRO_PULLBACK_FIB": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"]),
        "SCALP_REVERSAL_NWAVE": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_M1SCALPER"]),
        "TREND_RECLAIM_LONG": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_M1SCALPER"]),
        "VOL_SPIKE_RIDER": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_M1SCALPER"]),
        "LONDON_MOMENTUM": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_MACD_RSI_DIV"]),
        "MACRO_H1MOMENTUM": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_MACD_RSI_DIV"]),
        "TREND_H1": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_MACD_RSI_DIV"]),
        "H1_MOMENTUMSWING": dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_MACD_RSI_DIV"]),
    }
)

def _strategy_key(value: Optional[str]) -> str:
    if not value:
        return ""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _normalize_strategy_requirements(
    requirements: dict[str, dict[str, object]]
) -> dict[str, dict[str, object]]:
    return {_strategy_key(key): dict(value) for key, value in requirements.items()}


_NORMALIZED_STRATEGY_TECH_CONTEXT_REQUIREMENTS: dict[str, dict[str, object]] = (
    _normalize_strategy_requirements(_STRATEGY_TECH_CONTEXT_REQUIREMENTS)
)


def _resolve_strategy_technical_context_contract(
    strategy_tag: Optional[str],
    pocket: str,
) -> dict[str, object]:
    key = _strategy_key(strategy_tag)
    if not key:
        return {}
    if key in _NORMALIZED_STRATEGY_TECH_CONTEXT_REQUIREMENTS:
        return dict(_NORMALIZED_STRATEGY_TECH_CONTEXT_REQUIREMENTS[key])
    if "sessionopen" in key:
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SESSION_OPEN"])
    if key.startswith("techfusion"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["TECH_FUSION"])
    if key.startswith("macrotechfusion"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MACRO_TECH_FUSION"])
    if key.startswith("micropullbackfib"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_PULLBACK_FIB"])
    if key.startswith("rangecompressionbreak"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["RANGE_COMPRESSION_BREAK"])
    if key.startswith("scalpreversalnwave"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_REVERSAL_NWAVE"])
    if key.startswith("trendreclaim"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["TREND_RECLAIM_LONG"])
    if key.startswith("volspikerider"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["VOL_SPIKE_RIDER"])
    if key.startswith("macroh1momentum") or key.startswith("h1momentumswing") or key.startswith("trendh1"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MACRO_H1MOMENTUM"])
    if key.startswith("londonmomentum"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["LONDON_MOMENTUM"])
    if "ping" in key and key.startswith("scalp"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_PING_5S"])
    if ("m1" in key and key.startswith("scalp")) or key.startswith("scalpm1"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_M1SCALPER"])
    if "macd" in key and key.startswith("scalp"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_MACD_RSI_DIV"])
    if "tickimbalance" in key:
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_TICK_IMBALANCE"])
    if "tickwick" in key:
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_TICK_WICK_REVERSAL"])
    if "levelreject" in key:
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_LEVEL_REJECT"])
    if "tick" in key or "imbalance" in key:
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_TICK_IMBALANCE"])
    if "squeeze" in key and "pulse" in key:
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_SQUEEZE_PULSE_BREAK"])
    if "wick" in key and "reversal" in key:
        if "pro" in key:
            return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_WICK_REVERSAL_PRO"])
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_WICK_REVERSAL_BLEND"])
    if "micro" in key and key.startswith("micro"):
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"])
    if "micro" in key:
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"])
    if pocket == "micro":
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["MICRO_MULTISTRAT"])
    if pocket == "scalp":
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_M1SCALPER"])
    if pocket == "macro":
        return dict(_STRATEGY_TECH_CONTEXT_REQUIREMENTS["SCALP_MACD_RSI_DIV"])
    return {}


def _attach_strategy_technical_context_requirements(
    entry_thesis: Optional[dict],
    strategy_tag: Optional[str],
    pocket: str,
) -> Optional[dict]:
    if not isinstance(entry_thesis, dict):
        return entry_thesis
    if not isinstance(strategy_tag, str) and not pocket:
        return entry_thesis
    contract = _resolve_strategy_technical_context_contract(strategy_tag, pocket)
    if not contract:
        return entry_thesis
    if (
        not _ENTRY_TECH_CONTEXT_STRATEGY_REQUIREMENTS
        and not _has_explicit_technical_context_requirements(entry_thesis)
    ):
        return entry_thesis
    for key, raw_value in contract.items():
        if key in entry_thesis:
            if key == "tech_policy":
                existing_policy = entry_thesis.get("tech_policy")
                if isinstance(existing_policy, dict) and isinstance(raw_value, dict):
                    merged_policy = dict(existing_policy)
                    for policy_key, policy_value in raw_value.items():
                        if policy_key in {"require_fib", "require_nwave", "require_candle"}:
                            co = _coerce_bool(policy_value, default=False)
                            if co is not None:
                                merged_policy[policy_key] = co
                        elif policy_key == "tech_policy_locked":
                            if _coerce_bool(policy_value, default=False):
                                merged_policy[policy_key] = True
                        elif policy_key not in merged_policy:
                            merged_policy[policy_key] = policy_value
                    entry_thesis["tech_policy"] = merged_policy
            continue
        if key == "tech_policy" and isinstance(raw_value, dict) and not raw_value:
            continue
        if isinstance(raw_value, dict):
            entry_thesis[key] = dict(raw_value)
        elif isinstance(raw_value, (list, tuple, set)):
            entry_thesis[key] = list(raw_value)
        else:
            entry_thesis[key] = raw_value
    return entry_thesis


def _has_explicit_technical_context_requirements(entry_thesis: Optional[dict]) -> bool:
    if not isinstance(entry_thesis, dict):
        return False
    for key in (
        "technical_context",
        "technical_context_tfs",
        "technical_context_fields",
        "technical_context_ticks",
        "technical_context_candle_counts",
    ):
        if key in entry_thesis:
            return True
    return False


def _resolve_strategy_tag(
    strategy_tag: Optional[str],
    client_order_id: Optional[str],
    entry_thesis: Optional[dict],
) -> Optional[str]:
    if isinstance(entry_thesis, dict):
        thesis_strategy_tag = order_manager._strategy_tag_from_thesis(entry_thesis)
        if thesis_strategy_tag:
            return thesis_strategy_tag
    resolved = strategy_tag
    if not resolved:
        resolved = order_manager._strategy_tag_from_client_id(client_order_id)
    if not resolved and isinstance(entry_thesis, dict):
        resolved = order_manager._strategy_tag_from_thesis(entry_thesis)
    return resolved


def _resolve_entry_probability(
    entry_thesis: Optional[dict],
    confidence: Optional[float],
) -> Optional[float]:
    if not isinstance(entry_thesis, dict):
        return _entry_probability_value(confidence) if confidence is not None else None
    for key in ("entry_probability", "confidence"):
        if key not in entry_thesis:
            continue
        raw = entry_thesis.get(key)
        try:
            probability = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(probability) or math.isinf(probability):
            continue
        if probability <= 1.0:
            return max(0.0, min(1.0, probability))
        return max(0.0, min(1.0, probability / 100.0))
    return (
        _entry_probability_value(confidence)
        if confidence is not None
        else None
    )


def _entry_probability_value(raw: Optional[float]) -> Optional[float]:
    if raw is None:
        return None
    try:
        probability = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(probability) or math.isinf(probability):
        return None
    if probability <= 1.0:
        return max(0.0, min(1.0, probability))
    return max(0.0, min(1.0, probability / 100.0))


def _to_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _to_positive_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _normalize_timeframe(value: object) -> Optional[str]:
    candidate = str(value or "").strip().upper() if value is not None else ""
    if candidate in {"M1", "M5", "H1", "H4", "D1"}:
        return candidate
    return None


def _parse_horizon_minutes(value: object) -> Optional[int]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    unit = text[-1:]
    raw = text[:-1]
    try:
        n = int(raw)
    except Exception:
        return None
    if n <= 0:
        return None
    if unit == "m":
        return n
    if unit == "h":
        return n * 60
    if unit == "d":
        return n * 24 * 60
    if unit == "w":
        return n * 7 * 24 * 60
    return None


def _normalize_horizon_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        tokens = list(value)
    else:
        tokens = str(value).replace("|", ",").split(",")
    out: list[str] = []
    for token in tokens:
        text = str(token or "").strip().lower()
        if not text:
            continue
        if _parse_horizon_minutes(text) is None:
            continue
        if text not in out:
            out.append(text)
    return out


def _normalize_forecast_horizon_profile(profile_raw: object) -> dict[str, object]:
    if not isinstance(profile_raw, dict):
        return {}
    out: dict[str, object] = {}
    timeframe = _normalize_timeframe(profile_raw.get("timeframe"))
    if timeframe is None:
        timeframe = _normalize_timeframe(profile_raw.get("forecast_timeframe"))
    if timeframe is not None:
        out["timeframe"] = timeframe
    step_bars = _to_positive_int(profile_raw.get("step_bars"))
    if step_bars is None:
        step_bars = _to_positive_int(profile_raw.get("forecast_step_bars"))
    if step_bars is not None:
        out["step_bars"] = int(step_bars)
    for key in (
        "support_horizons",
        "confirm_horizons",
        "forecast_support_horizons",
        "forecast_confirm_horizons",
    ):
        horizons = _normalize_horizon_list(profile_raw.get(key))
        if horizons:
            out["support_horizons"] = horizons
            break
    for key in ("blend", "blend_with_bundle", "technical_only"):
        if key in profile_raw:
            out[key] = bool(profile_raw.get(key))
    if out:
        return out
    return {}


def _build_entry_forecast_profile(
    entry_thesis: Optional[dict],
    *,
    strategy_tag: Optional[str] = None,
    pocket: str = "",
    meta: Optional[dict] = None,
) -> Optional[dict[str, object]]:
    if not isinstance(entry_thesis, dict):
        entry_thesis = {}

    contract = _resolve_strategy_technical_context_contract(
        strategy_tag=strategy_tag,
        pocket=pocket,
    )
    contract_profile = _normalize_forecast_horizon_profile(
        contract.get("forecast_profile") if isinstance(contract, dict) else None
    )
    profile = dict(contract_profile or {})
    if isinstance(contract, dict):
        if "forecast_timeframe" in contract:
            tf = _normalize_timeframe(contract.get("forecast_timeframe"))
            if tf is not None:
                profile["timeframe"] = tf
        if "forecast_step_bars" in contract:
            step = _to_positive_int(contract.get("forecast_step_bars"))
            if step is not None:
                profile["step_bars"] = int(step)
        if "forecast_horizon" in contract and str(contract.get("forecast_horizon")).strip():
            profile["horizon"] = str(contract.get("forecast_horizon")).strip().lower()
        if "forecast_blend_with_bundle" in contract:
            profile["blend_with_bundle"] = bool(contract.get("forecast_blend_with_bundle"))
        if "forecast_technical_only" in contract:
            profile["technical_only"] = bool(contract.get("forecast_technical_only"))
        if "blend_with_bundle" in contract:
            profile["blend_with_bundle"] = bool(contract.get("blend_with_bundle"))
        if "technical_only" in contract:
            profile["technical_only"] = bool(contract.get("technical_only"))
        support_horizons = _normalize_horizon_list(
            contract.get("forecast_support_horizons")
            or contract.get("forecast_confirm_horizons")
            or contract.get("support_horizons")
        )
        if support_horizons:
            profile["support_horizons"] = support_horizons

    direct_profile = _normalize_forecast_horizon_profile(entry_thesis.get("forecast_profile"))
    if direct_profile:
        profile.update(direct_profile)
    explicit_timeframe = _normalize_timeframe(entry_thesis.get("forecast_timeframe"))
    explicit_step = _to_positive_int(entry_thesis.get("forecast_step_bars"))
    if explicit_timeframe is not None:
        profile["timeframe"] = explicit_timeframe
    if explicit_step is not None:
        profile["step_bars"] = int(explicit_step)
    if "forecast_blend_with_bundle" in entry_thesis:
        profile["blend_with_bundle"] = bool(entry_thesis.get("forecast_blend_with_bundle"))
    if "forecast_technical_only" in entry_thesis:
        profile["technical_only"] = bool(entry_thesis.get("forecast_technical_only"))
    explicit_support_horizons = _normalize_horizon_list(
        entry_thesis.get("forecast_support_horizons")
        or entry_thesis.get("forecast_confirm_horizons")
        or entry_thesis.get("support_horizons")
    )
    if explicit_support_horizons:
        profile["support_horizons"] = explicit_support_horizons
    elif isinstance(meta, dict):
        meta_support_horizons = _normalize_horizon_list(
            meta.get("forecast_support_horizons")
            or meta.get("forecast_confirm_horizons")
            or meta.get("support_horizons")
        )
        if meta_support_horizons:
            profile["support_horizons"] = meta_support_horizons

    horizon = (
        entry_thesis.get("forecast_horizon")
        or entry_thesis.get("horizon")
        or profile.get("horizon")
    )
    if not horizon:
        try:
            horizon = forecast_gate._infer_horizon_from_profile(profile)  # noqa: SLF001
        except Exception:
            horizon = None
    if not horizon:
        return None

    horizon_text = str(horizon).strip().lower()
    horizon_minutes = _parse_horizon_minutes(horizon_text)
    default_meta = {}
    try:
        default_meta = dict(forecast_gate._HORIZON_META)
    except Exception:
        default_meta = {}
    default_meta = default_meta.get(horizon_text) if isinstance(default_meta, dict) else None
    if isinstance(default_meta, dict):
        tf = _normalize_timeframe(default_meta.get("timeframe"))
        if tf is not None:
            profile["timeframe"] = tf
        step_bars = _to_positive_int(default_meta.get("step_bars"))
        if step_bars is not None:
            profile["step_bars"] = int(step_bars)

    counts = contract.get("technical_context_candle_counts")
    if not isinstance(counts, dict):
        counts = entry_thesis.get("technical_context_candle_counts")
    if isinstance(counts, dict):
        requested_tf = profile.get("timeframe")
        if requested_tf is None:
            requested_tf = _normalize_timeframe(next(iter(counts.keys()), ""))
        if requested_tf is not None:
            count = _to_positive_int(counts.get(requested_tf))
            if count is not None:
                # Use contract-defined candle depth as a directional hint for forecast profile.
                # Keep it in a practical range for technical fallback latency.
                profile["timeframe"] = str(requested_tf)
                profile["step_bars"] = min(24, max(6, max(1, int(round(count / 12)))))

    if profile.get("timeframe") is None and profile.get("step_bars") is None:
        if horizon_minutes is not None:
            if horizon_minutes <= 1:
                profile["timeframe"] = "M1"
            elif horizon_minutes <= 10:
                profile["timeframe"] = "M5"
            else:
                profile["timeframe"] = "H1"

    if profile.get("step_bars") is None and horizon_minutes is not None:
        if horizon_minutes <= 1:
            profile["step_bars"] = 12
        elif horizon_minutes <= 10:
            profile["step_bars"] = 6
        elif horizon_minutes <= 180:
            profile["step_bars"] = 9
        else:
            profile["step_bars"] = 12
    profile["horizon"] = horizon_text
    support_horizons = _normalize_horizon_list(profile.get("support_horizons"))
    if not support_horizons:
        support_horizons = [
            candidate
            for candidate in _FORECAST_SUPPORT_DEFAULT_BY_HORIZON.get(horizon_text, ())
            if candidate != horizon_text
        ]
    else:
        support_horizons = [
            candidate for candidate in support_horizons if candidate != horizon_text
        ]
    if support_horizons:
        profile["support_horizons"] = support_horizons
    elif "support_horizons" in profile:
        profile.pop("support_horizons", None)
    if profile.get("blend_with_bundle") in {"", None}:
        profile.pop("blend_with_bundle", None)
    if profile.get("technical_only") in {"", None}:
        profile.pop("technical_only", None)
    if not profile:
        return None
    return profile


def _scale_price_by_entry_distance(
    *,
    entry_price: Optional[float],
    price: Optional[float],
    multiplier: float,
) -> Optional[float]:
    if entry_price is None or price is None:
        return price
    anchor = _to_float(entry_price)
    target = _to_float(price)
    if anchor is None or target is None:
        return price
    distance = target - anchor
    if distance == 0:
        return price
    scaled = anchor + (distance) * max(0.05, min(5.0, float(multiplier)))
    return round(float(scaled), 3)


def _to_float_or_bool(value: object) -> object | None:
    if isinstance(value, bool):
        return value
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(float_value) or math.isinf(float_value):
        return None
    return float_value


def _normalize_tf_list(values: Iterable[object] | None) -> list[str]:
    if values is None:
        return []
    out: list[str] = []
    for value in values:
        text = str(value).strip().upper()
        if not text:
            continue
        if text not in out:
            out.append(text)
    return out


def _to_tfs(entry_thesis: Optional[dict], requested: list[str], pocket: str) -> list[str]:
    output: list[str] = []

    if isinstance(entry_thesis, dict):
        candidate = entry_thesis.get("technical_context_tfs")
        if isinstance(candidate, str):
            parsed = _normalize_tf_list(token.strip() for token in candidate.split(","))
            if parsed:
                return parsed
        elif isinstance(candidate, (tuple, list)):
            parsed = _normalize_tf_list(candidate)
            if parsed:
                return parsed
        tech_tfs = entry_thesis.get("tech_tfs")
        if isinstance(tech_tfs, dict):
            for key in ("fib", "median", "nwave", "candle", "default"):
                parsed = _normalize_tf_list(tech_tfs.get(key))
                if parsed:
                    output.extend(parsed)
            if output:
                return output

    if requested:
        output.extend(requested)
    if not output:
        output.extend(_TECH_DEFAULT_TFS_BY_POCKET.get(pocket, ()))
    if not output:
        output.extend(_DEFAULT_ENTRY_TECH_TFS or ("D1", "H4", "H1", "M5", "M1"))
    if not output:
        return []
    deduped: list[str] = []
    for tf in output:
        if tf not in deduped:
            deduped.append(tf)
    return deduped


def _to_fields(entry_thesis: Optional[dict]) -> list[str]:
    if not isinstance(entry_thesis, dict):
        return []
    candidate = entry_thesis.get("technical_context_fields")
    if isinstance(candidate, str):
        return [token.strip() for token in candidate.split(",") if token.strip()]
    if not isinstance(candidate, (tuple, list)):
        return []
    return [str(token).strip() for token in candidate if str(token).strip()]


def _to_tick_requirements(entry_thesis: Optional[dict]) -> list[str]:
    if not isinstance(entry_thesis, dict):
        return ["latest_bid", "latest_ask", "latest_mid", "spread_pips"]
    candidate = entry_thesis.get("technical_context_ticks")
    if isinstance(candidate, str):
        parsed = [token.strip() for token in candidate.split(",") if token.strip()]
        if parsed:
            return parsed
    elif isinstance(candidate, (tuple, list)):
        parsed = [str(token).strip() for token in candidate if str(token).strip()]
        if parsed:
            return parsed
    return ["latest_bid", "latest_ask", "latest_mid", "spread_pips"]


def _to_candle_counts(entry_thesis: Optional[dict]) -> dict[str, int]:
    if not isinstance(entry_thesis, dict):
        return {}
    candidate = entry_thesis.get("technical_context_candle_counts")
    if isinstance(candidate, str):
        counts: dict[str, int] = {}
        for item in candidate.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            tf, raw = item.split(":", 1)
            tf = tf.strip().upper()
            if not tf:
                continue
            try:
                value = int(raw.strip())
            except (TypeError, ValueError):
                continue
            if value > 0:
                counts[tf] = value
        return counts
    if not isinstance(candidate, dict):
        return {}
    counts: dict[str, int] = {}
    for tf_raw, raw in candidate.items():
        tf = str(tf_raw).strip().upper()
        if not tf:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            counts[tf] = value
    return counts


def _collect_strategy_tick_context(
    *,
    requested_fields: list[str],
) -> dict[str, object]:
    if not requested_fields:
        return {}
    try:
        from market_data import tick_window
    except Exception:
        return {}
    latest = {}
    try:
        latest = tick_window.summarize(seconds=4.0) or {}
    except Exception:
        return {}
    out: dict[str, object] = {}
    for key in requested_fields:
        if key in latest:
            value = _to_float_or_bool(latest.get(key))
            if value is not None:
                out[key] = value
    return out


def _collect_strategy_technical_context(
    *,
    strategy_tag: Optional[str],
    requested_tfs: list[str],
    requested_fields: list[str],
    requested_candle_counts: dict[str, int],
) -> dict[str, object]:
    fields_set = set(
        field for field in requested_fields if field not in {"", "candles"}
    )
    include_all_fields = not fields_set
    snapshot: dict[str, object] = {}
    factors = all_factors()
    candle_requirements = dict(requested_candle_counts or {})
    for tf in requested_tfs:
        tf_data = factors.get(tf)
        if not isinstance(tf_data, dict):
            continue
        payload: dict[str, object] = {}
        for key, raw in tf_data.items():
            if key in {"candles", "timestamp", "last_closed_timestamp", "live_updated_ts"}:
                continue
            if not include_all_fields and key not in fields_set:
                continue
            value = _to_float_or_bool(raw)
            if value is None:
                continue
            payload[key] = value
        candle_count = candle_requirements.get(tf, 0)
        if candle_count > 0:
            raw_candles = tf_data.get("candles")
            if isinstance(raw_candles, list):
                selected = raw_candles[-int(candle_count) :]
                parsed_candles: list[dict[str, object]] = []
                for raw in selected:
                    if not isinstance(raw, dict):
                        continue
                    candle_payload = {}
                    for key in ("timestamp", "time", "open", "high", "low", "close"):
                        value = raw.get(key)
                        if key in {"timestamp", "time"}:
                            if isinstance(value, str) and value:
                                candle_payload[key] = value
                            continue
                        value_float = _to_float(value)
                        if value_float is None:
                            continue
                        candle_payload[key] = value_float
                    if candle_payload:
                        parsed_candles.append(candle_payload)
                payload["candles"] = parsed_candles
        if payload:
            if strategy_tag:
                payload["_strategy_tag"] = strategy_tag
            snapshot[tf] = payload
    return snapshot


def _resolve_entry_side(units: int) -> str:
    return "long" if units >= 0 else "short"


def _resolve_entry_price(
    units: int,
    entry_thesis: Optional[dict],
    *,
    limit_price: Optional[float] = None,
) -> Optional[float]:
    if limit_price is not None:
        resolved = _to_float(limit_price)
        if resolved is not None:
            return resolved
    if isinstance(entry_thesis, dict):
        for key in ("entry_price", "price", "mid", "current_mid"):
            value = _to_float(entry_thesis.get(key))
            if value and value > 0:
                return value
        side = _resolve_entry_side(units)
        for key in ("current_ask", "ask", "current_bid", "bid"):
            if side == "long" and key in ("current_bid", "bid"):
                continue
            value = _to_float(entry_thesis.get(key))
            if value is not None and value > 0:
                return value
        for key in ("current_bid", "bid", "current_ask", "ask"):
            if side == "short" and key in ("current_ask", "ask"):
                continue
            value = _to_float(entry_thesis.get(key))
            if value is not None and value > 0:
                return value
    try:
        from market_data import tick_window
    except Exception:
        return None
    try:
        latest = tick_window.summarize(seconds=2.5)
        if not latest:
            return None
        side = _resolve_entry_side(units)
        if side == "long":
            value = _to_float(latest.get("latest_ask"))
            if value is not None and value > 0:
                return value
        value = _to_float(latest.get("latest_bid"))
        if value is not None and value > 0:
            return value
        return _to_float(latest.get("latest_mid"))
    except Exception:
        return None


def _format_forecast_context(decision: Any) -> dict[str, object]:
    if not decision:
        return {
            "forecast_source": "strategy_entry",
            "horizon": None,
            "reason": "not_applicable",
        }
    return {
        "forecast_source": "strategy_entry",
        "horizon": str(decision.horizon) if decision.horizon is not None else None,
        "allowed": bool(getattr(decision, "allowed", True)),
        "scale": _to_float(decision.scale),
        "reason": str(decision.reason or ""),
        "edge": _to_float(decision.edge),
        "p_up": _to_float(decision.p_up),
        "expected_pips": _to_float(decision.expected_pips),
        "anchor_price": _to_float(decision.anchor_price),
        "target_price": _to_float(decision.target_price),
        "range_low_pips": _to_float(decision.range_low_pips),
        "range_high_pips": _to_float(decision.range_high_pips),
        "range_sigma_pips": _to_float(decision.range_sigma_pips),
        "range_low_price": _to_float(decision.range_low_price),
        "range_high_price": _to_float(decision.range_high_price),
        "tp_pips_hint": _to_float(decision.tp_pips_hint),
        "target_reach_prob": _to_float(decision.target_reach_prob),
        "sl_pips_cap": _to_float(decision.sl_pips_cap),
        "rr_floor": _to_float(decision.rr_floor),
        "trend_strength": _to_float(decision.trend_strength),
        "range_pressure": _to_float(decision.range_pressure),
        "future_flow": str(decision.future_flow) if decision.future_flow is not None else None,
        "volatility_state": str(decision.volatility_state)
        if decision.volatility_state is not None
        else None,
        "trend_state": str(decision.trend_state) if decision.trend_state is not None else None,
        "range_state": str(decision.range_state) if decision.range_state is not None else None,
        "volatility_rank": _to_float(decision.volatility_rank),
        "regime_score": _to_float(decision.regime_score),
        "leading_indicator": (
            str(decision.leading_indicator) if decision.leading_indicator is not None else None
        ),
        "leading_indicator_strength": _to_float(decision.leading_indicator_strength),
        "tf_confluence_score": _to_float(decision.tf_confluence_score),
        "tf_confluence_count": (
            int(decision.tf_confluence_count)
            if decision.tf_confluence_count is not None
            else None
        ),
        "tf_confluence_horizons": (
            str(decision.tf_confluence_horizons)
            if decision.tf_confluence_horizons is not None
            else None
        ),
        "source": str(decision.source) if decision.source is not None else None,
        "style": str(decision.style) if decision.style is not None else None,
        "feature_ts": str(decision.feature_ts) if decision.feature_ts is not None else None,
    }


def _inject_entry_forecast_context(
    *,
    instrument: str,
    strategy_tag: Optional[str],
    pocket: str,
    units: int,
    entry_thesis: Optional[dict],
    meta: Optional[dict],
) -> tuple[Optional[dict], Optional[dict[str, object]]]:
    if not _STRATEGY_FORECAST_CONTEXT_ENABLED:
        return entry_thesis, None
    if not isinstance(entry_thesis, dict):
        entry_thesis = {}
    if not strategy_tag:
        return (
            entry_thesis,
            {
                "forecast_source": "strategy_entry",
                "enabled": False,
                "reason": "missing_strategy_tag",
            },
        )
    if not units:
        return (
            entry_thesis,
            {
                "forecast_source": "strategy_entry",
                "enabled": True,
                "horizon": None,
                "reason": "zero_units",
            },
        )
    side = "buy" if units > 0 else "sell"
    forecast_profile = _build_entry_forecast_profile(
        entry_thesis,
        strategy_tag=strategy_tag,
        pocket=pocket,
        meta=meta if isinstance(meta, dict) else None,
    )
    if isinstance(forecast_profile, dict) and forecast_profile:
        entry_thesis["forecast_profile"] = forecast_profile
        if "horizon" in forecast_profile and isinstance(forecast_profile.get("horizon"), str):
            entry_thesis["forecast_horizon"] = str(forecast_profile.get("horizon"))
        if "timeframe" in forecast_profile:
            entry_thesis["forecast_timeframe"] = forecast_profile.get("timeframe")
        if "step_bars" in forecast_profile:
            entry_thesis["forecast_step_bars"] = forecast_profile.get("step_bars")
        if "blend_with_bundle" in forecast_profile:
            entry_thesis["forecast_blend_with_bundle"] = bool(
                forecast_profile.get("blend_with_bundle")
            )
        if "technical_only" in forecast_profile:
            entry_thesis["forecast_technical_only"] = bool(
                forecast_profile.get("technical_only")
            )
        if "support_horizons" in forecast_profile:
            support = _normalize_horizon_list(forecast_profile.get("support_horizons"))
            if support:
                entry_thesis["forecast_support_horizons"] = support
    forecast_meta: dict[str, object] = {"instrument": instrument}
    if isinstance(meta, dict):
        forecast_meta.update(meta)
    try:
        decision = forecast_gate.decide(
            strategy_tag=strategy_tag,
            pocket=pocket,
            side=side,
            units=units,
            entry_thesis=entry_thesis,
            meta=forecast_meta,
        )
        forecast_context = _format_forecast_context(decision)
    except Exception as exc:
        forecast_context = {
            "forecast_source": "strategy_entry",
            "enabled": True,
            "horizon": None,
            "reason": "error",
            "error": str(exc),
        }
    current_forecast = entry_thesis.get("forecast")
    merged = dict(current_forecast) if isinstance(current_forecast, dict) else {}
    merged.update(forecast_context)
    if merged:
        entry_thesis["forecast"] = merged
    return entry_thesis, forecast_context


def _inject_entry_technical_context(
    *,
    units: int,
    pocket: str,
    strategy_tag: Optional[str],
    entry_thesis: Optional[dict],
    entry_price: Optional[float],
) -> Optional[dict]:
    if not _ENTRY_TECH_CONTEXT_ENABLED:
        return entry_thesis
    if entry_price is None:
        if not isinstance(entry_thesis, dict):
            return entry_thesis
    if not isinstance(entry_thesis, dict):
        return entry_thesis
    entry_thesis = _attach_strategy_technical_context_requirements(
        entry_thesis,
        strategy_tag=strategy_tag,
        pocket=pocket,
    )
    if (
        not _ENTRY_TECH_CONTEXT_STRATEGY_REQUIREMENTS
        and not _has_explicit_technical_context_requirements(entry_thesis)
    ):
        return entry_thesis
    requested_tfs = _to_tfs(entry_thesis, list(_DEFAULT_ENTRY_TECH_TFS), pocket)
    requested_fields = _to_fields(entry_thesis)
    requested_ticks = _to_tick_requirements(entry_thesis)
    requested_candle_counts = _to_candle_counts(entry_thesis)
    requested_ticks = requested_ticks or ["latest_bid", "latest_ask", "latest_mid", "spread_pips"]
    technical_snapshot = _collect_strategy_technical_context(
        strategy_tag=strategy_tag,
        requested_tfs=requested_tfs,
        requested_fields=requested_fields,
        requested_candle_counts=requested_candle_counts,
    )
    tick_snapshot = _collect_strategy_tick_context(
        requested_fields=requested_ticks,
    )
    existing_context = entry_thesis.get("technical_context")
    if isinstance(existing_context, dict):
        existing_result = existing_context.get("result") if isinstance(existing_context.get("result"), dict) else None
    else:
        existing_context = {}
        existing_result = None
    context = {
        "enabled": True,
        "entry_price": entry_price,
        "side": _resolve_entry_side(units),
        "entry_side": _resolve_entry_side(units),
        "requested_timeframes": requested_tfs,
        "requested_fields": requested_fields or ["all"],
        "requested_ticks": requested_ticks,
        "requested_candle_counts": requested_candle_counts,
        "indicators": technical_snapshot,
        "ticks": tick_snapshot,
        "requested": {
            "timeframes": requested_tfs,
            "fields": requested_fields or ["all"],
            "ticks": requested_ticks,
            "candle_counts": requested_candle_counts,
        },
    }
    try:
        if isinstance(existing_result, dict):
            context["result"] = existing_result
            if isinstance(existing_context.get("debug"), dict):
                context["debug"] = dict(existing_context.get("debug"))
        else:
            context["result"] = {
                "allowed": True,
                "reason": "strategy_local_only",
                "score": 0.0,
                "coverage": 0.0,
                "size_mult": 1.0,
            }
            context["debug"] = {
                "strategy_local_only": True,
            }
    except Exception as exc:
        context["debug"] = {"error": str(exc)}
        context["result"] = {
            "allowed": True,
            "reason": "entry_technical_eval_failed",
            "score": 0.0,
            "coverage": 0.0,
            "size_mult": 1.0,
        }
    prev = existing_context
    if isinstance(prev, dict):
        merged = dict(prev)
        merged.update(context)
        entry_thesis["technical_context"] = merged
    else:
        entry_thesis["technical_context"] = context
    return entry_thesis


def _apply_strategy_feedback(
    strategy_tag: Optional[str],
    *,
    pocket: str,
    units: int,
    entry_probability: Optional[float],
    entry_price: Optional[float],
    sl_price: Optional[float],
    tp_price: Optional[float],
    entry_thesis: Optional[dict],
) -> tuple[int, Optional[float], Optional[float], Optional[float], dict[str, object]]:
    try:
        advice = strategy_feedback.current_advice(strategy_tag, pocket=pocket)
    except Exception:
        return units, entry_probability, sl_price, tp_price, {}
    if not advice:
        return units, entry_probability, sl_price, tp_price, {}
    if not isinstance(entry_thesis, dict):
        entry_thesis = None

    orig_units = int(units)
    adjusted_units = orig_units
    adjusted_probability = entry_probability
    adjusted_sl_price = sl_price
    adjusted_tp_price = tp_price
    applied: dict[str, object] = {}

    units_multiplier = _to_float(advice.get("entry_units_multiplier", 1.0))
    if units_multiplier is None:
        units_multiplier = 1.0
    units_multiplier = max(0.0, min(5.0, units_multiplier))
    if units_multiplier != 1.0 and orig_units:
        sign = 1 if orig_units >= 0 else -1
        adjusted_units = sign * max(0, int(round(abs(orig_units) * units_multiplier)))
        applied["entry_units_multiplier"] = round(units_multiplier, 6)
        if adjusted_units != orig_units:
            applied["entry_units"] = {
                "before": orig_units,
                "after": adjusted_units,
            }

    units_min_raw = advice.get("entry_units_min")
    units_min = _to_float(units_min_raw)
    if units_min is not None:
        units_min = max(0.0, units_min)
        if adjusted_units:
            if abs(adjusted_units) < units_min:
                sign = 1 if adjusted_units >= 0 else -1
                adjusted_units = sign * int(round(units_min))
                applied["entry_units_min"] = int(round(units_min))

    units_max_raw = advice.get("entry_units_max")
    units_max = _to_float(units_max_raw)
    if units_max is not None:
        units_max = max(0.0, units_max)
        if adjusted_units and abs(adjusted_units) > units_max:
            sign = 1 if adjusted_units >= 0 else -1
            adjusted_units = sign * int(round(units_max))
            applied["entry_units_max"] = int(round(units_max))

    if adjusted_probability is not None:
        probability_multiplier = _to_float(advice.get("entry_probability_multiplier", 1.0))
        if probability_multiplier is None:
            probability_multiplier = 1.0
        probability_delta = _to_float(advice.get("entry_probability_delta", 0.0))
        if probability_delta is None:
            probability_delta = 0.0
        if probability_multiplier != 1.0 or probability_delta != 0.0:
            adjusted_probability = adjusted_probability * probability_multiplier + probability_delta
            adjusted_probability = max(0.0, min(1.0, adjusted_probability))
            if adjusted_probability != entry_probability:
                applied["entry_probability"] = {
                    "before": entry_probability,
                    "after": adjusted_probability,
                }

    sl_multiplier = _to_float(advice.get("sl_distance_multiplier", 1.0))
    if sl_multiplier is None:
        sl_multiplier = 1.0
    if sl_multiplier != 1.0:
        candidate = _scale_price_by_entry_distance(
            entry_price=entry_price,
            price=adjusted_sl_price,
            multiplier=sl_multiplier,
        )
        if candidate != adjusted_sl_price:
            adjusted_sl_price = candidate
            applied["sl_distance_multiplier"] = round(sl_multiplier, 6)

    tp_multiplier = _to_float(advice.get("tp_distance_multiplier", 1.0))
    if tp_multiplier is None:
        tp_multiplier = 1.0
    if tp_multiplier != 1.0:
        candidate = _scale_price_by_entry_distance(
            entry_price=entry_price,
            price=adjusted_tp_price,
            multiplier=tp_multiplier,
        )
        if candidate != adjusted_tp_price:
            adjusted_tp_price = candidate
            applied["tp_distance_multiplier"] = round(tp_multiplier, 6)

    if adjusted_probability is not None and entry_thesis is not None:
        entry_thesis["entry_probability"] = adjusted_probability
    strategy_params = advice.get("strategy_params")
    configured_params = advice.get("configured_params")
    analysis_feedback: dict[str, object] = {
        "source": advice.get("_meta", {}),
        "applied": applied,
    }
    notes = advice.get("notes")
    if notes is not None:
        analysis_feedback["notes"] = notes
    if isinstance(strategy_params, dict) and strategy_params:
        analysis_feedback["strategy_params"] = strategy_params
    if isinstance(configured_params, dict) and configured_params:
        analysis_feedback["configured_params"] = configured_params
    if (applied or analysis_feedback.get("strategy_params") or analysis_feedback.get("configured_params")) and entry_thesis is not None:
        # Keep both the new key (analysis_feedback) and historical key
        # (analysis_advice) for compatibility with any downstream consumers.
        entry_thesis["analysis_feedback"] = analysis_feedback
        entry_thesis["analysis_advice"] = analysis_feedback
    return adjusted_units, adjusted_probability, adjusted_sl_price, adjusted_tp_price, applied


def _apply_forecast_fusion(
    *,
    strategy_tag: Optional[str],
    pocket: str,
    units: int,
    entry_probability: Optional[float],
    entry_thesis: Optional[dict],
    forecast_context: Optional[dict[str, object]],
) -> tuple[int, Optional[float], dict[str, object]]:
    if not _STRATEGY_FORECAST_FUSION_ENABLED:
        return units, entry_probability, {}
    if not units:
        return units, entry_probability, {}
    if (pocket or "").lower() == "manual":
        return units, entry_probability, {}
    if not isinstance(forecast_context, dict):
        return units, entry_probability, {}

    p_up_raw = _to_float(forecast_context.get("p_up"))
    if p_up_raw is None:
        return units, entry_probability, {}
    p_up = max(0.0, min(1.0, float(p_up_raw)))
    side_sign = 1 if units > 0 else -1
    direction_prob = p_up if side_sign > 0 else (1.0 - p_up)
    direction_prob = max(0.0, min(1.0, direction_prob))
    direction_bias = max(-1.0, min(1.0, (direction_prob - 0.5) * 2.0))

    edge_raw = _to_float(forecast_context.get("edge"))
    if edge_raw is None:
        edge_strength = max(0.0, min(1.0, abs(direction_bias)))
    else:
        edge_clamped = max(0.0, min(1.0, float(edge_raw)))
        # Strength is distance from neutral (0.5), not only bullish-side excess.
        # This allows strong bearish forecasts (edge<<0.5) to trigger contra rejects.
        edge_strength = max(0.0, min(1.0, abs(edge_clamped - 0.5) / 0.5))

    allowed_flag = _coerce_bool(forecast_context.get("allowed"), None)
    tf_confluence_score_raw = _to_float(forecast_context.get("tf_confluence_score"))
    tf_confluence_score = (
        max(-1.0, min(1.0, float(tf_confluence_score_raw)))
        if tf_confluence_score_raw is not None
        else None
    )
    tf_confluence_count = _to_positive_int(forecast_context.get("tf_confluence_count"))
    if tf_confluence_count is None:
        tf_confluence_weight = 0.5
    else:
        tf_confluence_weight = max(0.35, min(1.0, float(tf_confluence_count) / 3.0))

    units_scale = 1.0
    if direction_bias >= 0.0:
        units_scale += (
            _STRATEGY_FORECAST_FUSION_UNITS_BOOST_MAX * direction_bias * edge_strength
        )
    else:
        units_scale -= (
            _STRATEGY_FORECAST_FUSION_UNITS_CUT_MAX
            * abs(direction_bias)
            * (0.5 + 0.5 * edge_strength)
        )
    if tf_confluence_score is not None:
        if tf_confluence_score >= 0.0:
            units_scale *= (
                1.0
                + _STRATEGY_FORECAST_FUSION_TF_BOOST_MAX
                * tf_confluence_score
                * tf_confluence_weight
            )
        else:
            units_scale *= (
                1.0
                - _STRATEGY_FORECAST_FUSION_TF_CUT_MAX
                * abs(tf_confluence_score)
                * tf_confluence_weight
            )
    if allowed_flag is False:
        units_scale *= _STRATEGY_FORECAST_FUSION_DISALLOW_UNITS_MULT
    units_scale = max(
        _STRATEGY_FORECAST_FUSION_UNITS_MIN_SCALE,
        min(_STRATEGY_FORECAST_FUSION_UNITS_MAX_SCALE, units_scale),
    )
    adjusted_units = int((1 if side_sign > 0 else -1) * round(abs(units) * units_scale))

    adjusted_probability = entry_probability
    if adjusted_probability is None and _STRATEGY_FORECAST_FUSION_SYNTH_PROB_IF_MISSING:
        adjusted_probability = direction_prob
    if adjusted_probability is not None:
        probability_shift = (
            direction_bias
            * _STRATEGY_FORECAST_FUSION_PROB_GAIN
            * (0.35 + 0.65 * edge_strength)
        )
        adjusted_probability = max(
            0.0,
            min(1.0, float(adjusted_probability) + probability_shift),
        )
        if tf_confluence_score is not None:
            adjusted_probability = max(
                0.0,
                min(
                    1.0,
                    float(adjusted_probability)
                    + tf_confluence_score
                    * _STRATEGY_FORECAST_FUSION_PROB_GAIN
                    * 0.35
                    * tf_confluence_weight,
                ),
            )
        if allowed_flag is False:
            adjusted_probability = max(
                0.0,
                min(
                    1.0,
                    float(adjusted_probability)
                    * _STRATEGY_FORECAST_FUSION_DISALLOW_PROB_MULT,
                ),
            )

    strong_contra = (
        _STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED
        and direction_prob <= _STRATEGY_FORECAST_FUSION_STRONG_CONTRA_PROB_MAX
        and edge_strength >= _STRATEGY_FORECAST_FUSION_STRONG_CONTRA_EDGE_MIN
        and (allowed_flag is False or direction_bias < 0.0)
    )
    if strong_contra:
        adjusted_units = 0
        if adjusted_probability is not None:
            adjusted_probability = max(0.0, min(float(adjusted_probability), direction_prob))

    applied: dict[str, object] = {
        "strategy_tag": str(strategy_tag or ""),
        "pocket": str(pocket or ""),
        "units_before": int(units),
        "units_after": int(adjusted_units),
        "units_scale": round(float(units_scale), 6),
        "entry_probability_before": (
            round(float(entry_probability), 6)
            if entry_probability is not None
            else None
        ),
        "entry_probability_after": (
            round(float(adjusted_probability), 6)
            if adjusted_probability is not None
            else None
        ),
        "direction_prob": round(float(direction_prob), 6),
        "direction_bias": round(float(direction_bias), 6),
        "edge_strength": round(float(edge_strength), 6),
        "forecast_horizon": forecast_context.get("horizon"),
        "forecast_allowed": allowed_flag,
        "forecast_reason": forecast_context.get("reason"),
        "tf_confluence_score": (
            round(float(tf_confluence_score), 6) if tf_confluence_score is not None else None
        ),
        "tf_confluence_count": tf_confluence_count,
        "strong_contra_reject": bool(strong_contra),
    }
    if strong_contra:
        applied["reject_reason"] = "strong_contra_forecast"

    if isinstance(entry_thesis, dict):
        entry_thesis["forecast_fusion"] = applied
        if adjusted_probability is not None:
            entry_thesis["entry_probability"] = adjusted_probability
        if adjusted_units != 0 and _STRATEGY_FORECAST_FUSION_TP_BLEND_ENABLED:
            tp_hint = _to_float(forecast_context.get("tp_pips_hint"))
            if tp_hint is not None and tp_hint > 0.0 and direction_prob >= 0.5:
                base_tp = _to_float(
                    entry_thesis.get("tp_pips") or entry_thesis.get("target_tp_pips")
                )
                if base_tp is None or base_tp <= 0.0:
                    merged_tp = tp_hint
                    tp_blend = 1.0
                else:
                    tp_blend = max(
                        0.0,
                        min(
                            1.0,
                            _STRATEGY_FORECAST_FUSION_TP_BLEND_BASE
                            + _STRATEGY_FORECAST_FUSION_TP_BLEND_EDGE_GAIN
                            * edge_strength
                            * max(0.0, direction_bias),
                        ),
                    )
                    merged_tp = (1.0 - tp_blend) * base_tp + tp_blend * tp_hint
                entry_thesis["tp_pips"] = round(max(0.1, float(merged_tp)), 3)
                applied["tp_hint_pips"] = round(float(tp_hint), 4)
                applied["tp_blend"] = round(float(tp_blend), 4)
                applied["tp_after"] = round(float(entry_thesis["tp_pips"]), 4)

        if adjusted_units != 0:
            sl_cap = _to_float(forecast_context.get("sl_pips_cap"))
            current_sl = _to_float(entry_thesis.get("sl_pips"))
            if (
                sl_cap is not None
                and sl_cap > 0.0
                and current_sl is not None
                and current_sl > sl_cap
            ):
                entry_thesis["sl_pips"] = round(float(sl_cap), 3)
                applied["sl_cap_applied"] = round(float(sl_cap), 4)

    return adjusted_units, adjusted_probability, applied


async def _coordinate_entry_units(
    *,
    instrument: str,
    pocket: str,
    strategy_tag: Optional[str],
    units: int,
    reduce_only: bool,
    entry_probability: Optional[float],
    client_order_id: Optional[str],
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
    forecast_context: Optional[dict[str, object]] = None,
) -> int:
    if not units:
        return units
    if reduce_only:
        return units
    if not strategy_tag:
        if (pocket or "").lower() != "manual":
            return 0
        return units
    min_units = order_manager.min_units_for_strategy(strategy_tag, pocket=pocket)
    if _STRATEGY_PATTERN_GATE_ENABLED and not reduce_only and pocket != "manual":
        if isinstance(entry_thesis, dict):
            gate_meta = _normalize_pattern_gate_meta(entry_thesis, meta)
            try:
                pattern_decision = pattern_gate.decide(
                    strategy_tag=strategy_tag,
                    pocket=pocket,
                    side=1 if units > 0 else -1,
                    units=units,
                    entry_thesis=entry_thesis,
                    meta=gate_meta,
                )
            except Exception:
                pattern_decision = None
            if pattern_decision is not None:
                entry_thesis["pattern_gate"] = pattern_decision.to_payload()
                if not pattern_decision.allowed:
                    return 0
                if pattern_decision.scale != 1.0:
                    scaled_units = int(round(abs(units) * pattern_decision.scale))
                    if scaled_units < min_units:
                        if _ORDER_PATTERN_GATE_SCALE_TO_MIN_UNITS and min_units > 0:
                            scaled_units = min_units
                        else:
                            return 0
                    if scaled_units > 0:
                        units = int((1 if units > 0 else -1) * scaled_units)
    final_units, reason, _ = await order_manager.coordinate_entry_intent(
        instrument=instrument,
        pocket=pocket,
        strategy_tag=strategy_tag,
        side=1 if units > 0 else -1,
        raw_units=units,
        entry_probability=entry_probability,
        client_order_id=client_order_id,
        min_units=min_units,
        forecast_context=forecast_context,
    )
    if not final_units and reason in {"reject", "scaled", "rejected", None}:
        return 0
    return int(final_units)


async def market_order(
    instrument: str,
    units: int,
    sl_price: Optional[float],
    tp_price: Optional[float],
    pocket: Literal["micro", "macro", "scalp", "scalp_fast", "manual"],
    *,
    client_order_id: Optional[str] = None,
    strategy_tag: Optional[str] = None,
    reduce_only: bool = False,
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
    confidence: Optional[int] = None,
    stage_index: Optional[int] = None,
    arbiter_final: bool = False,
) -> Optional[str]:
    resolved_strategy_tag = _resolve_strategy_tag(strategy_tag, client_order_id, entry_thesis)
    if entry_thesis is None:
        entry_thesis = {}
    entry_thesis = _inject_entry_technical_context(
        units=units,
        pocket=pocket,
        strategy_tag=resolved_strategy_tag,
        entry_thesis=entry_thesis,
        entry_price=_resolve_entry_price(units, entry_thesis),
    )
    entry_probability = _resolve_entry_probability(entry_thesis, confidence)
    entry_thesis = dict(entry_thesis) if isinstance(entry_thesis, dict) else None
    entry_thesis, forecast_context = _inject_entry_forecast_context(
        instrument=instrument,
        strategy_tag=resolved_strategy_tag,
        pocket=pocket,
        units=units,
        entry_thesis=entry_thesis,
        meta=meta,
    )
    units, entry_probability, sl_price, tp_price, _ = _apply_strategy_feedback(
        resolved_strategy_tag,
        pocket=pocket,
        units=units,
        entry_probability=entry_probability,
        entry_price=_resolve_entry_price(units, entry_thesis),
        sl_price=sl_price,
        tp_price=tp_price,
        entry_thesis=entry_thesis,
    )
    units, entry_probability, _ = _apply_forecast_fusion(
        strategy_tag=resolved_strategy_tag,
        pocket=pocket,
        units=units,
        entry_probability=entry_probability,
        entry_thesis=entry_thesis,
        forecast_context=forecast_context,
    )
    entry_thesis, meta = _inject_env_prefix_context(
        entry_thesis, meta, resolved_strategy_tag
    )
    coordinated_units = await _coordinate_entry_units(
        instrument=instrument,
        pocket=pocket,
        strategy_tag=resolved_strategy_tag,
        units=units,
        reduce_only=reduce_only,
        entry_probability=entry_probability,
        client_order_id=client_order_id,
        entry_thesis=entry_thesis,
        meta=meta,
        forecast_context=forecast_context,
    )
    if not coordinated_units:
        return None
    if coordinated_units != units:
        units = coordinated_units
    if isinstance(entry_thesis, dict):
        entry_thesis["entry_units_intent"] = abs(int(units))
    return await order_manager.market_order(
        instrument=instrument,
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        pocket=pocket,
        client_order_id=client_order_id,
        strategy_tag=resolved_strategy_tag,
        reduce_only=reduce_only,
        entry_thesis=entry_thesis,
        meta=meta,
        confidence=confidence,
        stage_index=stage_index,
        arbiter_final=arbiter_final,
    )


async def limit_order(
    instrument: str,
    units: int,
    price: float,
    sl_price: Optional[float],
    tp_price: Optional[float],
    pocket: Literal["micro", "macro", "scalp"],
    *,
    current_bid: Optional[float] = None,
    current_ask: Optional[float] = None,
    require_passive: bool = True,
    client_order_id: Optional[str] = None,
    strategy_tag: Optional[str] = None,
    reduce_only: bool = False,
    ttl_ms: float = 800.0,
    entry_thesis: Optional[dict] = None,
    confidence: Optional[int] = None,
    meta: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    resolved_strategy_tag = _resolve_strategy_tag(strategy_tag, client_order_id, entry_thesis)
    if entry_thesis is None:
        entry_thesis = {}
    entry_thesis = _inject_entry_technical_context(
        units=units,
        pocket=pocket,
        strategy_tag=resolved_strategy_tag,
        entry_thesis=entry_thesis,
        entry_price=_resolve_entry_price(units, entry_thesis, limit_price=price),
    )
    entry_probability = _resolve_entry_probability(entry_thesis, confidence)
    entry_thesis = dict(entry_thesis) if isinstance(entry_thesis, dict) else None
    entry_thesis, forecast_context = _inject_entry_forecast_context(
        instrument=instrument,
        strategy_tag=resolved_strategy_tag,
        pocket=pocket,
        units=units,
        entry_thesis=entry_thesis,
        meta=meta,
    )
    units, entry_probability, sl_price, tp_price, _ = _apply_strategy_feedback(
        resolved_strategy_tag,
        pocket=pocket,
        units=units,
        entry_probability=entry_probability,
        entry_price=_resolve_entry_price(units, entry_thesis, limit_price=price),
        sl_price=sl_price,
        tp_price=tp_price,
        entry_thesis=entry_thesis,
    )
    units, entry_probability, _ = _apply_forecast_fusion(
        strategy_tag=resolved_strategy_tag,
        pocket=pocket,
        units=units,
        entry_probability=entry_probability,
        entry_thesis=entry_thesis,
        forecast_context=forecast_context,
    )
    entry_thesis, meta = _inject_env_prefix_context(
        entry_thesis, meta, resolved_strategy_tag
    )
    if isinstance(entry_thesis, dict):
        entry_thesis["entry_units_intent"] = abs(int(units))
    coordinated_units = await _coordinate_entry_units(
        instrument=instrument,
        pocket=pocket,
        strategy_tag=resolved_strategy_tag,
        units=units,
        reduce_only=reduce_only,
        entry_probability=entry_probability,
        client_order_id=client_order_id,
        entry_thesis=entry_thesis,
        meta=meta,
        forecast_context=forecast_context,
    )
    if not coordinated_units:
        return None, None
    if coordinated_units != units:
        units = coordinated_units
    return await order_manager.limit_order(
        instrument=instrument,
        units=units,
        price=price,
        sl_price=sl_price,
        tp_price=tp_price,
        pocket=pocket,
        current_bid=current_bid,
        current_ask=current_ask,
        require_passive=require_passive,
        client_order_id=client_order_id,
        reduce_only=reduce_only,
        ttl_ms=ttl_ms,
        entry_thesis=entry_thesis,
        confidence=confidence,
        meta=meta,
    )

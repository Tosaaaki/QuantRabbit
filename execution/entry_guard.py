"""Entry guard based on median/fib section axis."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Dict, Optional

from analysis.range_model import compute_range_snapshot
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

_DEFAULT_TF_BY_POCKET = {
    "scalp": "M1",
    "scalp_fast": "M1",
    "micro": "H1",
    "macro": "H1",
    "manual": "H1",
}
_DEFAULT_LOOKBACK_BY_POCKET = {
    "scalp": 24,
    "scalp_fast": 20,
    "micro": 30,
    "macro": 60,
    "manual": 30,
}
_DEFAULT_MIN_RANGE_PIPS = {
    "scalp": 3.0,
    "scalp_fast": 2.5,
    "micro": 6.0,
    "macro": 18.0,
    "manual": 10.0,
}
_DEFAULT_FIB_EXTREME = {
    "scalp": 0.214,
    "scalp_fast": 0.214,
    "micro": 0.236,
    "macro": 0.236,
    "manual": 0.236,
}
_DEFAULT_MID_DISTANCE_PIPS = {
    "scalp": 1.6,
    "scalp_fast": 1.3,
    "micro": 2.8,
    "macro": 10.0,
    "manual": 6.0,
}
_DEFAULT_MID_DISTANCE_FRAC = {
    "scalp": 0.25,
    "scalp_fast": 0.25,
    "micro": 0.2,
    "macro": 0.2,
    "manual": 0.2,
}
_DEFAULT_ADX_BYPASS = {
    "scalp": 28.0,
    "scalp_fast": 28.0,
    "micro": 26.0,
    "macro": 30.0,
    "manual": 30.0,
}
_DEFAULT_TREND_ADX_MIN = {
    "scalp": 20.0,
    "scalp_fast": 20.0,
    "micro": 18.0,
    "macro": 22.0,
    "manual": 22.0,
}
_DEFAULT_TREND_MA_DIFF_PIPS = {
    "scalp": 0.8,
    "scalp_fast": 0.6,
    "micro": 1.2,
    "macro": 2.0,
    "manual": 2.0,
}


@dataclass(slots=True)
class EntryGuardDecision:
    allowed: bool
    reason: Optional[str]
    debug: Dict[str, object]


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
        return None


def _env_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() not in {"", "0", "false", "no"}


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


def _coerce_thesis(meta: object) -> dict:
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _strategy_key_candidates(tag: Optional[str]) -> tuple[str, ...]:
    if not tag:
        return ()
    raw = "".join(ch for ch in str(tag) if ch.isalnum()).upper()
    if not raw:
        return ()
    keys = [raw]
    if len(raw) > 9:
        keys.append(raw[:9])
    return tuple(dict.fromkeys(keys))


def _trend_hint(strategy_tag: Optional[str], thesis: dict) -> bool:
    if thesis.get("entry_guard_trend") is True:
        return True
    if thesis.get("entry_guard_trend") is False:
        return False
    if thesis.get("trend_bias") or thesis.get("trend_score"):
        return True
    profile = thesis.get("profile") or thesis.get("strategy_profile") or ""
    parts = " ".join([str(strategy_tag or ""), str(profile or "")]).lower()
    for token in ("trend", "break", "breakout", "donchian", "momentum", "impulse", "pulse", "trendma"):
        if token in parts:
            return True
    return False


def _resolve_tf(thesis: dict, pocket: str, strategy_keys: tuple[str, ...]) -> str:
    for key in ("entry_tf", "timeframe", "entry_timeframe", "tf"):
        tf = _normalize_tf(thesis.get(key))
        if tf:
            return tf
    for key in strategy_keys:
        tf = _normalize_tf(os.getenv(f"ENTRY_GUARD_TF_{key}"))
        if tf:
            return tf
    pocket_upper = pocket.upper()
    tf = _normalize_tf(os.getenv(f"ENTRY_GUARD_TF_{pocket_upper}") or os.getenv("ENTRY_GUARD_TF"))
    if tf:
        return tf
    return _DEFAULT_TF_BY_POCKET.get(pocket, "H1")


def _resolve_bool(name: str, pocket: str, default: bool, strategy_keys: tuple[str, ...]) -> bool:
    for key in strategy_keys:
        specific = _env_bool(f"{name}_{key}")
        if specific is not None:
            return specific
    pocket_upper = pocket.upper()
    specific = _env_bool(f"{name}_{pocket_upper}")
    if specific is not None:
        return specific
    common = _env_bool(name)
    if common is not None:
        return common
    return default


def _resolve_float(
    name: str, pocket: str, defaults: Dict[str, float], strategy_keys: tuple[str, ...]
) -> float:
    for key in strategy_keys:
        specific = _env_float(f"{name}_{key}")
        if specific is not None:
            return specific
    pocket_upper = pocket.upper()
    specific = _env_float(f"{name}_{pocket_upper}")
    if specific is not None:
        return specific
    common = _env_float(name)
    if common is not None:
        return common
    return float(defaults.get(pocket, defaults.get("default", 0.0)))


def _resolve_int(
    name: str, pocket: str, defaults: Dict[str, int], strategy_keys: tuple[str, ...]
) -> int:
    for key in strategy_keys:
        specific = _env_int(f"{name}_{key}")
        if specific is not None:
            return specific
    pocket_upper = pocket.upper()
    specific = _env_int(f"{name}_{pocket_upper}")
    if specific is not None:
        return specific
    common = _env_int(name)
    if common is not None:
        return common
    return int(defaults.get(pocket, defaults.get("default", 0)))


def evaluate_entry_guard(
    *,
    entry_price: float,
    side: str,
    pocket: str,
    strategy_tag: Optional[str] = None,
    entry_thesis: Optional[dict] = None,
) -> EntryGuardDecision:
    if entry_price <= 0.0:
        return EntryGuardDecision(True, None, {})
    if side not in {"long", "short"}:
        return EntryGuardDecision(True, None, {})
    thesis = _coerce_thesis(entry_thesis)
    if thesis.get("entry_guard") is False or thesis.get("entry_guard_disabled") is True:
        return EntryGuardDecision(True, "disabled", {})

    if not strategy_tag:
        raw_tag = thesis.get("strategy_tag") or thesis.get("strategy") or thesis.get("tag")
        if raw_tag:
            strategy_tag = str(raw_tag)

    strategy_keys = _strategy_key_candidates(strategy_tag)
    if not _resolve_bool("ENTRY_GUARD_ENABLED", pocket, True, strategy_keys):
        return EntryGuardDecision(True, None, {})

    tf = _resolve_tf(thesis, pocket, strategy_keys)
    lookback = _resolve_int("ENTRY_GUARD_LOOKBACK", pocket, _DEFAULT_LOOKBACK_BY_POCKET, strategy_keys)
    hi_pct = _resolve_float("ENTRY_GUARD_HI_PCT", pocket, {"default": 95.0}, strategy_keys)
    lo_pct = _resolve_float("ENTRY_GUARD_LO_PCT", pocket, {"default": 5.0}, strategy_keys)
    candles = get_candles_snapshot(tf, limit=max(lookback, 6))
    axis = compute_range_snapshot(
        candles,
        lookback=max(10, lookback),
        method="percentile",
        hi_pct=hi_pct,
        lo_pct=lo_pct,
    )
    if axis is None:
        return EntryGuardDecision(True, None, {})

    range_span = float(axis.high) - float(axis.low)
    if range_span <= 0.0:
        return EntryGuardDecision(True, None, {})
    range_pips = range_span / PIP

    min_range_pips = _resolve_float(
        "ENTRY_GUARD_MIN_RANGE_PIPS",
        pocket,
        _DEFAULT_MIN_RANGE_PIPS,
        strategy_keys,
    )
    if range_pips < min_range_pips:
        return EntryGuardDecision(True, None, {})

    factors = all_factors()
    fac = factors.get(tf) or factors.get("H1") or {}
    try:
        adx = float(fac.get("adx") or 0.0)
    except Exception:
        adx = 0.0
    adx_bypass = _resolve_float("ENTRY_GUARD_ADX_BYPASS", pocket, _DEFAULT_ADX_BYPASS, strategy_keys)
    if adx >= adx_bypass:
        return EntryGuardDecision(True, None, {})

    trend_hint = _trend_hint(strategy_tag, thesis)
    trend_bypass_enabled = _resolve_bool("ENTRY_GUARD_TREND_BYPASS", pocket, True, strategy_keys)
    trend_bypass = False
    trend_dir = 0
    ma_diff_pips = 0.0
    if trend_hint and trend_bypass_enabled:
        try:
            ma10 = float(fac.get("ma10") or 0.0)
            ma20 = float(fac.get("ma20") or 0.0)
        except Exception:
            ma10 = 0.0
            ma20 = 0.0
        if ma10 > 0.0 and ma20 > 0.0:
            if ma10 > ma20:
                trend_dir = 1
            elif ma10 < ma20:
                trend_dir = -1
            ma_diff_pips = abs(ma10 - ma20) / PIP
        trend_adx_min = _resolve_float(
            "ENTRY_GUARD_TREND_ADX_MIN", pocket, _DEFAULT_TREND_ADX_MIN, strategy_keys
        )
        trend_ma_min = _resolve_float(
            "ENTRY_GUARD_TREND_MA_DIFF_PIPS", pocket, _DEFAULT_TREND_MA_DIFF_PIPS, strategy_keys
        )
        side_dir = 1 if side == "long" else -1
        if trend_dir == side_dir and (adx >= trend_adx_min or ma_diff_pips >= trend_ma_min):
            trend_bypass = True

    fib_extreme = _resolve_float("ENTRY_GUARD_FIB_EXTREME", pocket, _DEFAULT_FIB_EXTREME, strategy_keys)
    fib_extreme = max(0.05, min(0.45, fib_extreme))
    mid_distance_pips = _resolve_float(
        "ENTRY_GUARD_MID_DISTANCE_PIPS",
        pocket,
        _DEFAULT_MID_DISTANCE_PIPS,
        strategy_keys,
    )
    mid_distance_frac = _resolve_float(
        "ENTRY_GUARD_MID_DISTANCE_FRAC",
        pocket,
        _DEFAULT_MID_DISTANCE_FRAC,
        strategy_keys,
    )
    mid_distance_pips = max(mid_distance_pips, range_pips * mid_distance_frac)

    upper_guard = float(axis.high) - range_span * fib_extreme
    lower_guard = float(axis.low) + range_span * fib_extreme
    mid = float(axis.mid)
    mid_distance = abs(entry_price - mid) / PIP

    if side == "long":
        if entry_price >= upper_guard:
            if trend_bypass:
                return EntryGuardDecision(
                    True,
                    "entry_guard_trend_bypass",
                    {
                        "entry": entry_price,
                        "upper": upper_guard,
                        "mid": mid,
                        "range_pips": round(range_pips, 3),
                        "adx": round(adx, 2),
                        "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
                        "trend_ma_pips": round(ma_diff_pips, 2),
                    },
                )
            return EntryGuardDecision(
                False,
                "entry_guard_extreme_long",
                {
                    "entry": entry_price,
                    "upper": upper_guard,
                    "mid": mid,
                    "range_pips": round(range_pips, 3),
                    "adx": round(adx, 2),
                },
            )
        if entry_price > mid and mid_distance >= mid_distance_pips:
            if trend_bypass:
                return EntryGuardDecision(
                    True,
                    "entry_guard_trend_bypass",
                    {
                        "entry": entry_price,
                        "mid": mid,
                        "distance_pips": round(mid_distance, 2),
                        "range_pips": round(range_pips, 3),
                        "adx": round(adx, 2),
                        "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
                        "trend_ma_pips": round(ma_diff_pips, 2),
                    },
                )
            return EntryGuardDecision(
                False,
                "entry_guard_mid_far_long",
                {
                    "entry": entry_price,
                    "mid": mid,
                    "distance_pips": round(mid_distance, 2),
                    "range_pips": round(range_pips, 3),
                    "adx": round(adx, 2),
                },
            )
    else:
        if entry_price <= lower_guard:
            if trend_bypass:
                return EntryGuardDecision(
                    True,
                    "entry_guard_trend_bypass",
                    {
                        "entry": entry_price,
                        "lower": lower_guard,
                        "mid": mid,
                        "range_pips": round(range_pips, 3),
                        "adx": round(adx, 2),
                        "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
                        "trend_ma_pips": round(ma_diff_pips, 2),
                    },
                )
            return EntryGuardDecision(
                False,
                "entry_guard_extreme_short",
                {
                    "entry": entry_price,
                    "lower": lower_guard,
                    "mid": mid,
                    "range_pips": round(range_pips, 3),
                    "adx": round(adx, 2),
                },
            )
        if entry_price < mid and mid_distance >= mid_distance_pips:
            if trend_bypass:
                return EntryGuardDecision(
                    True,
                    "entry_guard_trend_bypass",
                    {
                        "entry": entry_price,
                        "mid": mid,
                        "distance_pips": round(mid_distance, 2),
                        "range_pips": round(range_pips, 3),
                        "adx": round(adx, 2),
                        "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
                        "trend_ma_pips": round(ma_diff_pips, 2),
                    },
                )
            return EntryGuardDecision(
                False,
                "entry_guard_mid_far_short",
                {
                    "entry": entry_price,
                    "mid": mid,
                    "distance_pips": round(mid_distance, 2),
                    "range_pips": round(range_pips, 3),
                    "adx": round(adx, 2),
                },
            )

    return EntryGuardDecision(
        True,
        None,
        {
            "entry": entry_price,
            "mid": mid,
            "range_pips": round(range_pips, 3),
            "adx": round(adx, 2),
            "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
            "trend_ma_pips": round(ma_diff_pips, 2),
        },
    )

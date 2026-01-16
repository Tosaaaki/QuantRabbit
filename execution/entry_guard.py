"""Entry guard based on median/fib section axis."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
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
_TF_ORDER = {"M1": 1, "M5": 2, "H1": 3, "H4": 4, "D1": 5}

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
    "scalp": 0.12,
    "scalp_fast": 0.12,
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
_DEFAULT_PULLBACK_BUFFER_PIPS = {
    "scalp": 0.8,
    "scalp_fast": 0.6,
    "micro": 1.2,
    "macro": 2.0,
    "manual": 2.0,
}
_DEFAULT_SOFT_MIN_CHECKS = {
    "default": 2,
}
_DEFAULT_SOFT_MIN_PENALTIES = {
    "default": 2,
}
_DEFAULT_MTF_MIN_BLOCKS = {
    "default": 2,
}
_DEFAULT_ALIGN_MIN = {
    "default": 0,
}
_DEFAULT_ALIGN_ADX_MIN = {
    "default": 0.0,
}
_DEFAULT_ALIGN_MA_DIFF_PIPS = {
    "default": 0.0,
}
_DEFAULT_ADX_RANGE_MIN = {
    "default": 0.0,
}
_DEFAULT_ADX_RANGE_MAX = {
    "default": 0.0,
}
_DEFAULT_MA20_GAP_ATR_MAX = {
    "default": 0.0,
}
_DEFAULT_MOMENTUM_MIN = {
    "scalp": 0.004,
    "scalp_fast": 0.004,
    "micro": 0.003,
    "macro": 0.002,
    "manual": 0.002,
}
_DEFAULT_VOL_5M_MIN = {
    "scalp": 0.8,
    "scalp_fast": 0.8,
    "micro": 0.7,
    "macro": 0.6,
    "manual": 0.6,
}
_DEFAULT_RSI_HIGH = {
    "default": 70.0,
}
_DEFAULT_RSI_LOW = {
    "default": 30.0,
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


def _parse_tf_list(value: object) -> list[str]:
    if value is None:
        return []
    items = []
    if isinstance(value, (list, tuple, set)):
        items.extend(value)
    else:
        text = str(value)
        for token in re.split(r"[\\s,|/]+", text):
            if token:
                items.append(token)
    tfs: list[str] = []
    for item in items:
        tf = _normalize_tf(item)
        if tf:
            tfs.append(tf)
    return tfs


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _tf_rank(tf: Optional[str]) -> int:
    if not tf:
        return 99
    return _TF_ORDER.get(tf, 99)


def _pick_tf_extremes(tfs: list[str]) -> tuple[Optional[str], Optional[str]]:
    if not tfs:
        return None, None
    ordered = sorted(tfs, key=_tf_rank)
    return ordered[0], ordered[-1]


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


def _factor_from_thesis(thesis: dict, key: str, tf: Optional[str]) -> Optional[float]:
    ctx = thesis.get("context") if isinstance(thesis, dict) else None
    if isinstance(ctx, dict) and key in ctx:
        try:
            return float(ctx[key])
        except Exception:
            return None
    factors = thesis.get("factors") if isinstance(thesis, dict) else None
    if isinstance(factors, dict):
        tf_key = (tf or "").lower()
        if tf_key:
            sub = factors.get(tf_key)
            if isinstance(sub, dict) and key in sub:
                try:
                    return float(sub[key])
                except Exception:
                    return None
        for sub in factors.values():
            if isinstance(sub, dict) and key in sub:
                try:
                    return float(sub[key])
                except Exception:
                    return None
    return None


def _resolve_factor(thesis: dict, fac: dict, key: str, tf: str) -> Optional[float]:
    val = _factor_from_thesis(thesis, key, tf)
    if val is not None:
        return val
    if isinstance(fac, dict) and key in fac:
        try:
            return float(fac[key])
        except Exception:
            return None
    return None


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
    for token in (
        "trend",
        "break",
        "breakout",
        "donchian",
        "momentum",
        "impulse",
        "pulse",
        "trendma",
        "pullback",
        "retest",
        "runner",
        "stack",
        "retrace",
    ):
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


def _resolve_tfs(thesis: dict, pocket: str, strategy_keys: tuple[str, ...]) -> list[str]:
    primary_tf = _resolve_tf(thesis, pocket, strategy_keys)
    tfs = [primary_tf]

    explicit: list[str] = []
    explicit.extend(_parse_tf_list(thesis.get("entry_guard_tfs")))
    if not explicit:
        for key in strategy_keys:
            explicit.extend(_parse_tf_list(os.getenv(f"ENTRY_GUARD_TFS_{key}")))
        pocket_upper = pocket.upper()
        explicit.extend(_parse_tf_list(os.getenv(f"ENTRY_GUARD_TFS_{pocket_upper}")))
        explicit.extend(_parse_tf_list(os.getenv("ENTRY_GUARD_TFS")))

    if explicit:
        tfs.extend(explicit)
    else:
        for key in ("entry_tf", "struct_tf", "env_tf", "tf", "timeframe", "entry_timeframe"):
            tfs.extend(_parse_tf_list(thesis.get(key)))
        section_axis = thesis.get("section_axis")
        if isinstance(section_axis, dict):
            tfs.extend(_parse_tf_list(section_axis.get("tf")))

    return _dedupe_keep_order([tf for tf in tfs if tf])


def _resolve_metric_tf(
    thesis: dict,
    thesis_key: str,
    env_key: str,
    pocket: str,
    strategy_keys: tuple[str, ...],
    fallback: Optional[str],
) -> Optional[str]:
    tf = _normalize_tf(thesis.get(thesis_key))
    if tf:
        return tf
    for key in strategy_keys:
        tf = _normalize_tf(os.getenv(f"{env_key}_{key}"))
        if tf:
            return tf
    pocket_upper = pocket.upper()
    tf = _normalize_tf(os.getenv(f"{env_key}_{pocket_upper}") or os.getenv(env_key))
    return tf or fallback


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


def _resolve_trend_metrics(
    thesis: dict, fac: dict, tf: str
) -> tuple[float, int, float]:
    adx = _resolve_factor(thesis, fac, "adx", tf) or 0.0
    ma10 = _resolve_factor(thesis, fac, "ma10", tf) or 0.0
    ma20 = _resolve_factor(thesis, fac, "ma20", tf) or 0.0
    trend_dir = 0
    ma_diff_pips = 0.0
    if ma10 > 0.0 and ma20 > 0.0:
        if ma10 > ma20:
            trend_dir = 1
        elif ma10 < ma20:
            trend_dir = -1
        ma_diff_pips = abs(ma10 - ma20) / PIP
    return float(adx), trend_dir, float(ma_diff_pips)


def _resolve_momentum_metrics(
    thesis: dict, fac: dict, tf: str
) -> tuple[Optional[float], Optional[float]]:
    momentum = _resolve_factor(thesis, fac, "momentum", tf)
    vol_5m = _resolve_factor(thesis, fac, "vol_5m", tf)
    return momentum, vol_5m


def _compute_alignment(
    tfs: list[str],
    side: str,
    thesis: dict,
    factors: Dict[str, Dict[str, float]],
    adx_min: float,
    ma_min: float,
) -> tuple[int, int, Dict[str, object]]:
    if not tfs:
        return 0, 0, {}
    side_dir = 1 if side == "long" else -1
    align_count = 0
    valid_count = 0
    details: Dict[str, object] = {}
    for tf in tfs:
        fac = factors.get(tf) or factors.get("H1") or {}
        adx, trend_dir, ma_diff_pips = _resolve_trend_metrics(thesis, fac, tf)
        aligned = trend_dir == side_dir
        strong = aligned and (
            (adx_min <= 0.0 and ma_min <= 0.0) or adx >= adx_min or ma_diff_pips >= ma_min
        )
        if trend_dir != 0:
            valid_count += 1
        if strong:
            align_count += 1
        details[tf] = {
            "adx": round(adx, 2),
            "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
            "trend_ma_pips": round(ma_diff_pips, 2),
            "aligned": aligned,
            "strong": strong,
        }
    return align_count, valid_count, details


def _pick_block_reason(blocked: list[EntryGuardDecision], side: str) -> Optional[str]:
    if not blocked:
        return None
    if side == "short":
        priority = ("entry_guard_extreme_short", "entry_guard_mid_far_short")
    else:
        priority = ("entry_guard_extreme_long", "entry_guard_mid_far_long")
    for reason in priority:
        for decision in blocked:
            if decision.reason == reason:
                return reason
    return blocked[0].reason


def _merge_mtf_debug(
    base_debug: Dict[str, object],
    tfs: list[str],
    decisions: list[EntryGuardDecision],
    min_blocks: int,
) -> Dict[str, object]:
    debug = dict(base_debug or {})
    mtf_decisions: Dict[str, object] = {}
    for tf, decision in zip(tfs, decisions):
        mtf_decisions[tf] = {
            "allowed": decision.allowed,
            "reason": decision.reason,
            "debug": decision.debug,
        }
    debug["mtf"] = {
        "primary_tf": tfs[0] if tfs else None,
        "tfs": list(tfs),
        "min_blocks": int(min_blocks),
        "blocked": sum(1 for decision in decisions if not decision.allowed),
        "decisions": mtf_decisions,
    }
    return debug


def evaluate_entry_guard(
    *,
    entry_price: float,
    side: str,
    pocket: str,
    strategy_tag: Optional[str] = None,
    entry_thesis: Optional[dict] = None,
    _tf_override: Optional[str] = None,
    _factors: Optional[Dict[str, Dict[str, float]]] = None,
    _mtf: bool = True,
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

    mtf_enabled = _resolve_bool("ENTRY_GUARD_MTF_ENABLED", pocket, True, strategy_keys)
    factors = _factors or all_factors()
    tfs: list[str] = []
    lower_tf: Optional[str] = None
    higher_tf: Optional[str] = None
    align_min = 0
    align_count: Optional[int] = None
    align_valid: Optional[int] = None
    align_details: Optional[Dict[str, object]] = None

    def _base_guard_debug() -> Dict[str, object]:
        debug: Dict[str, object] = {}
        if tfs:
            mtf_debug: Dict[str, object] = {
                "primary_tf": tfs[0],
                "tfs": list(tfs),
            }
            if align_count is not None:
                mtf_debug["align_min"] = align_min
                mtf_debug["align_count"] = align_count
                mtf_debug["align_valid"] = align_valid
                mtf_debug["align_details"] = align_details
            debug["mtf"] = mtf_debug
        return debug

    if _tf_override is None:
        tfs = _resolve_tfs(thesis, pocket, strategy_keys)
        lower_tf, higher_tf = _pick_tf_extremes(tfs)

        align_min = _resolve_int("ENTRY_GUARD_ALIGN_MIN", pocket, _DEFAULT_ALIGN_MIN, strategy_keys)
        align_min = max(0, min(int(align_min), len(tfs))) if tfs else 0
        align_adx_min = _resolve_float(
            "ENTRY_GUARD_ALIGN_ADX_MIN", pocket, _DEFAULT_ALIGN_ADX_MIN, strategy_keys
        )
        align_ma_min = _resolve_float(
            "ENTRY_GUARD_ALIGN_MA_DIFF_PIPS", pocket, _DEFAULT_ALIGN_MA_DIFF_PIPS, strategy_keys
        )
        if tfs:
            align_count, align_valid, align_details = _compute_alignment(
                tfs,
                side,
                thesis,
                factors,
                align_adx_min,
                align_ma_min,
            )
        if align_min > 0 and align_count is not None:
            if align_valid is not None and align_valid >= align_min and align_count < align_min:
                debug = _base_guard_debug()
                return EntryGuardDecision(False, "entry_guard_align_min", debug)

        overheat_enabled = _resolve_bool(
            "ENTRY_GUARD_OVERHEAT_BLOCK", pocket, False, strategy_keys
        )
        if overheat_enabled:
            rsi_tf = _resolve_metric_tf(
                thesis,
                "entry_guard_rsi_tf",
                "ENTRY_GUARD_RSI_TF",
                pocket,
                strategy_keys,
                lower_tf or (tfs[0] if tfs else None),
            )
            rsi_high = _resolve_float("ENTRY_GUARD_RSI_HIGH", pocket, _DEFAULT_RSI_HIGH, strategy_keys)
            rsi_low = _resolve_float("ENTRY_GUARD_RSI_LOW", pocket, _DEFAULT_RSI_LOW, strategy_keys)
            if rsi_tf:
                fac = factors.get(rsi_tf) or factors.get("H1") or {}
                rsi = _resolve_factor(thesis, fac, "rsi", rsi_tf)
                if rsi is not None:
                    if side == "long" and rsi >= rsi_high:
                        debug = _base_guard_debug()
                        debug["overheat"] = {
                            "tf": rsi_tf,
                            "rsi": round(rsi, 2),
                            "threshold": rsi_high,
                        }
                        return EntryGuardDecision(False, "entry_guard_overheat_long", debug)
                    if side == "short" and rsi <= rsi_low:
                        debug = _base_guard_debug()
                        debug["overheat"] = {
                            "tf": rsi_tf,
                            "rsi": round(rsi, 2),
                            "threshold": rsi_low,
                        }
                        return EntryGuardDecision(False, "entry_guard_overheat_short", debug)

        adx_min = _resolve_float(
            "ENTRY_GUARD_ADX_RANGE_MIN", pocket, _DEFAULT_ADX_RANGE_MIN, strategy_keys
        )
        adx_max = _resolve_float(
            "ENTRY_GUARD_ADX_RANGE_MAX", pocket, _DEFAULT_ADX_RANGE_MAX, strategy_keys
        )
        if adx_min > 0.0 or adx_max > 0.0:
            adx_tf = _resolve_metric_tf(
                thesis,
                "entry_guard_adx_tf",
                "ENTRY_GUARD_ADX_TF",
                pocket,
                strategy_keys,
                higher_tf or (tfs[0] if tfs else None),
            )
            if adx_tf:
                fac = factors.get(adx_tf) or factors.get("H1") or {}
                adx_value = _resolve_factor(thesis, fac, "adx", adx_tf)
                if adx_value is not None:
                    if adx_min > 0.0 and adx_value < adx_min:
                        debug = _base_guard_debug()
                        debug["adx_range"] = {
                            "tf": adx_tf,
                            "adx": round(adx_value, 2),
                            "min": adx_min,
                            "max": adx_max if adx_max > 0.0 else None,
                        }
                        return EntryGuardDecision(False, "entry_guard_adx_low", debug)
                    if adx_max > 0.0 and adx_value > adx_max:
                        debug = _base_guard_debug()
                        debug["adx_range"] = {
                            "tf": adx_tf,
                            "adx": round(adx_value, 2),
                            "min": adx_min if adx_min > 0.0 else None,
                            "max": adx_max,
                        }
                        return EntryGuardDecision(False, "entry_guard_adx_high", debug)

        gap_atr_max = _resolve_float(
            "ENTRY_GUARD_MA20_GAP_ATR_MAX", pocket, _DEFAULT_MA20_GAP_ATR_MAX, strategy_keys
        )
        if gap_atr_max > 0.0:
            gap_tf = _resolve_metric_tf(
                thesis,
                "entry_guard_ma20_gap_tf",
                "ENTRY_GUARD_MA20_GAP_TF",
                pocket,
                strategy_keys,
                higher_tf or (tfs[0] if tfs else None),
            )
            if gap_tf:
                fac = factors.get(gap_tf) or factors.get("H1") or {}
                ma20 = _resolve_factor(thesis, fac, "ma20", gap_tf)
                atr_pips = _resolve_factor(thesis, fac, "atr_pips", gap_tf)
                if ma20 is not None and atr_pips is not None and atr_pips > 0.0:
                    gap_atr = abs(entry_price - ma20) / (atr_pips * PIP)
                    if gap_atr > gap_atr_max:
                        debug = _base_guard_debug()
                        debug["ma20_gap_atr"] = {
                            "tf": gap_tf,
                            "entry": round(entry_price, 3),
                            "ma20": round(ma20, 3),
                            "atr_pips": round(atr_pips, 2),
                            "gap_atr": round(gap_atr, 2),
                            "max": gap_atr_max,
                        }
                        return EntryGuardDecision(False, "entry_guard_ma20_gap", debug)

    if _mtf and _tf_override is None and mtf_enabled:
        tfs = tfs or _resolve_tfs(thesis, pocket, strategy_keys)
        if len(tfs) > 1:
            decisions = [
                evaluate_entry_guard(
                    entry_price=entry_price,
                    side=side,
                    pocket=pocket,
                    strategy_tag=strategy_tag,
                    entry_thesis=thesis,
                    _tf_override=tf,
                    _factors=factors,
                    _mtf=False,
                )
                for tf in tfs
            ]
            primary = decisions[0]
            blocked = [decision for decision in decisions if not decision.allowed]
            extreme_reason = (
                "entry_guard_extreme_short" if side == "short" else "entry_guard_extreme_long"
            )
            extreme_blocked = [
                decision
                for decision in decisions
                if (not decision.allowed) and decision.reason == extreme_reason
            ]
            min_blocks = _resolve_int(
                "ENTRY_GUARD_MTF_MIN_BLOCKS", pocket, _DEFAULT_MTF_MIN_BLOCKS, strategy_keys
            )
            min_blocks = max(1, min(int(min_blocks), len(decisions)))
            base_debug = primary.debug if decisions else {}
            debug = _merge_mtf_debug(base_debug, tfs, decisions, min_blocks)
            trend_adx_min = _resolve_float(
                "ENTRY_GUARD_TREND_ADX_MIN", pocket, _DEFAULT_TREND_ADX_MIN, strategy_keys
            )
            trend_ma_min = _resolve_float(
                "ENTRY_GUARD_TREND_MA_DIFF_PIPS", pocket, _DEFAULT_TREND_MA_DIFF_PIPS, strategy_keys
            )
            momentum_min = _resolve_float(
                "ENTRY_GUARD_MOMENTUM_MIN", pocket, _DEFAULT_MOMENTUM_MIN, strategy_keys
            )
            vol_5m_min = _resolve_float(
                "ENTRY_GUARD_VOL_5M_MIN", pocket, _DEFAULT_VOL_5M_MIN, strategy_keys
            )
            if lower_tf is None or higher_tf is None:
                lower_tf, higher_tf = _pick_tf_extremes(tfs)
            trend_ok = False
            trend_metrics = None
            if higher_tf:
                fac = factors.get(higher_tf) or factors.get("H1") or {}
                adx, trend_dir, ma_diff_pips = _resolve_trend_metrics(thesis, fac, higher_tf)
                side_dir = 1 if side == "long" else -1
                trend_ok = trend_dir == side_dir and (
                    adx >= trend_adx_min or ma_diff_pips >= trend_ma_min
                )
                trend_metrics = {
                    "tf": higher_tf,
                    "adx": round(adx, 2),
                    "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
                    "trend_ma_pips": round(ma_diff_pips, 2),
                }
            momentum_ok = False
            momentum_metrics = None
            if lower_tf:
                fac = factors.get(lower_tf) or factors.get("H1") or {}
                momentum, vol_5m = _resolve_momentum_metrics(thesis, fac, lower_tf)
                if momentum is not None:
                    if side == "long":
                        momentum_ok = momentum >= momentum_min
                    else:
                        momentum_ok = momentum <= -momentum_min
                    if vol_5m is not None:
                        momentum_ok = momentum_ok and vol_5m >= vol_5m_min
                momentum_metrics = {
                    "tf": lower_tf,
                    "momentum": round(momentum, 5) if momentum is not None else None,
                    "vol_5m": round(vol_5m, 3) if vol_5m is not None else None,
                }
            if "mtf" not in debug:
                debug["mtf"] = {}
            if align_count is not None:
                debug["mtf"]["align_min"] = align_min
                debug["mtf"]["align_count"] = align_count
                debug["mtf"]["align_valid"] = align_valid
                debug["mtf"]["align_details"] = align_details
            debug["mtf"]["trend_ok"] = trend_ok
            debug["mtf"]["momentum_ok"] = momentum_ok
            debug["mtf"]["extreme_blocked"] = len(extreme_blocked)
            debug["mtf"]["trend_metrics"] = trend_metrics
            debug["mtf"]["momentum_metrics"] = momentum_metrics
            if primary.allowed:
                return EntryGuardDecision(True, primary.reason, debug)
            if trend_ok or momentum_ok:
                debug["mtf"]["allow_reason"] = "trend_ok" if trend_ok else "momentum_ok"
                return EntryGuardDecision(True, "entry_guard_mtf_allow", debug)
            if len(extreme_blocked) >= min_blocks:
                reason = _pick_block_reason(blocked, side)
                return EntryGuardDecision(False, reason, debug)
            return EntryGuardDecision(True, "entry_guard_mtf_allow", debug)

    tf = _tf_override or _resolve_tf(thesis, pocket, strategy_keys)
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

    fac = factors.get(tf) or factors.get("H1") or {}
    adx_value = None
    try:
        if fac.get("adx") is not None:
            adx_value = float(fac.get("adx"))
    except Exception:
        adx_value = None
    adx = adx_value or 0.0
    adx_bypass = _resolve_float("ENTRY_GUARD_ADX_BYPASS", pocket, _DEFAULT_ADX_BYPASS, strategy_keys)
    if adx >= adx_bypass:
        return EntryGuardDecision(True, None, {})

    try:
        ma10 = float(fac.get("ma10") or 0.0)
        ma20 = float(fac.get("ma20") or 0.0)
    except Exception:
        ma10 = 0.0
        ma20 = 0.0
    trend_dir = 0
    ma_diff_pips = 0.0
    if ma10 > 0.0 and ma20 > 0.0:
        if ma10 > ma20:
            trend_dir = 1
        elif ma10 < ma20:
            trend_dir = -1
        ma_diff_pips = abs(ma10 - ma20) / PIP

    trend_hint = _trend_hint(strategy_tag, thesis)
    trend_bypass_enabled = _resolve_bool("ENTRY_GUARD_TREND_BYPASS", pocket, True, strategy_keys)
    if thesis.get("entry_guard_pullback_only") is True:
        trend_bypass_enabled = False
    trend_bypass = False
    pullback_bypass = False
    trend_ok = False
    trend_adx_min = _resolve_float(
        "ENTRY_GUARD_TREND_ADX_MIN", pocket, _DEFAULT_TREND_ADX_MIN, strategy_keys
    )
    trend_ma_min = _resolve_float(
        "ENTRY_GUARD_TREND_MA_DIFF_PIPS", pocket, _DEFAULT_TREND_MA_DIFF_PIPS, strategy_keys
    )
    if trend_hint:
        side_dir = 1 if side == "long" else -1
        trend_ok = trend_dir == side_dir and (adx >= trend_adx_min or ma_diff_pips >= trend_ma_min)
        if trend_bypass_enabled and trend_ok:
            trend_bypass = True

    pullback_flag = thesis.get("entry_guard_pullback")
    if pullback_flag is True:
        pullback_enabled = True
    elif pullback_flag is False:
        pullback_enabled = False
    else:
        pullback_enabled = _resolve_bool("ENTRY_GUARD_PULLBACK", pocket, False, strategy_keys)
    pullback_buffer_pips = None
    pullback_dist_pips = None
    if pullback_enabled and trend_hint and trend_ok:
        buffer_pips = _resolve_float(
            "ENTRY_GUARD_PULLBACK_BUFFER_PIPS",
            pocket,
            _DEFAULT_PULLBACK_BUFFER_PIPS,
            strategy_keys,
        )
        buffer_pips = max(0.1, buffer_pips)
        buffer = buffer_pips * PIP
        if ma10 > 0.0:
            dist = abs(entry_price - ma10)
            if dist <= buffer:
                pullback_bypass = True
                pullback_dist_pips = dist / PIP
        if (not pullback_bypass) and ma20 > 0.0:
            dist = abs(entry_price - ma20)
            if dist <= buffer:
                pullback_bypass = True
                pullback_dist_pips = dist / PIP
        if pullback_bypass:
            pullback_buffer_pips = buffer_pips

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

    soft_enabled = _resolve_bool("ENTRY_GUARD_SOFT_ENABLED", pocket, True, strategy_keys)
    soft_min_checks = _resolve_int(
        "ENTRY_GUARD_SOFT_MIN_CHECKS", pocket, _DEFAULT_SOFT_MIN_CHECKS, strategy_keys
    )
    soft_min_penalties = _resolve_int(
        "ENTRY_GUARD_SOFT_MIN_PENALTIES",
        pocket,
        _DEFAULT_SOFT_MIN_PENALTIES,
        strategy_keys,
    )
    momentum_min = _resolve_float(
        "ENTRY_GUARD_MOMENTUM_MIN",
        pocket,
        _DEFAULT_MOMENTUM_MIN,
        strategy_keys,
    )
    vol_5m_min = _resolve_float(
        "ENTRY_GUARD_VOL_5M_MIN",
        pocket,
        _DEFAULT_VOL_5M_MIN,
        strategy_keys,
    )
    rsi_high = _resolve_float("ENTRY_GUARD_RSI_HIGH", pocket, _DEFAULT_RSI_HIGH, strategy_keys)
    rsi_low = _resolve_float("ENTRY_GUARD_RSI_LOW", pocket, _DEFAULT_RSI_LOW, strategy_keys)
    momentum = _resolve_factor(thesis, fac, "momentum", tf)
    vol_5m = _resolve_factor(thesis, fac, "vol_5m", tf)
    rsi = _resolve_factor(thesis, fac, "rsi", tf)

    soft_checks = []
    soft_penalties = []

    def _soft_check(name: str, ok: bool) -> None:
        soft_checks.append(name)
        if not ok:
            soft_penalties.append(name)

    if adx_value is not None:
        _soft_check("adx", adx >= trend_adx_min)
    if ma_diff_pips > 0.0:
        _soft_check("ma_gap", ma_diff_pips >= trend_ma_min)
    if trend_dir != 0:
        _soft_check("trend_dir", trend_dir == (1 if side == "long" else -1))
    if momentum is not None:
        if side == "long":
            _soft_check("momentum", momentum >= momentum_min)
        else:
            _soft_check("momentum", momentum <= -momentum_min)
    if vol_5m is not None:
        _soft_check("vol_5m", vol_5m >= vol_5m_min)
    if rsi is not None:
        if side == "long":
            _soft_check("rsi", rsi <= rsi_high)
        else:
            _soft_check("rsi", rsi >= rsi_low)

    def _soft_allow(block_reason: str, base_debug: Dict[str, object]) -> Optional[EntryGuardDecision]:
        if not soft_enabled:
            return None
        if len(soft_checks) < max(0, soft_min_checks):
            return None
        if len(soft_penalties) >= max(1, soft_min_penalties):
            return None
        debug = dict(base_debug)
        debug["soft_block_reason"] = block_reason
        debug["soft_checks"] = list(soft_checks)
        debug["soft_penalties"] = list(soft_penalties)
        debug["soft_metrics"] = {
            "adx": round(adx, 2),
            "trend_ma_pips": round(ma_diff_pips, 2),
            "momentum": round(momentum, 5) if momentum is not None else None,
            "vol_5m": round(vol_5m, 3) if vol_5m is not None else None,
            "rsi": round(rsi, 2) if rsi is not None else None,
        }
        return EntryGuardDecision(True, "entry_guard_soft_allow", debug)

    if side == "long":
        if entry_price >= upper_guard:
            if pullback_bypass:
                return EntryGuardDecision(
                    True,
                    "entry_guard_pullback",
                    {
                        "entry": entry_price,
                        "upper": upper_guard,
                        "mid": mid,
                        "range_pips": round(range_pips, 3),
                        "adx": round(adx, 2),
                        "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
                        "trend_ma_pips": round(ma_diff_pips, 2),
                        "ma10": round(ma10, 3) if ma10 > 0 else None,
                        "ma20": round(ma20, 3) if ma20 > 0 else None,
                        "pullback_buffer_pips": pullback_buffer_pips,
                        "pullback_dist_pips": round(pullback_dist_pips, 3)
                        if pullback_dist_pips is not None
                        else None,
                    },
                )
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
            soft = _soft_allow(
                "entry_guard_extreme_long",
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
            if soft:
                return soft
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
            if pullback_bypass:
                return EntryGuardDecision(
                    True,
                    "entry_guard_pullback",
                    {
                        "entry": entry_price,
                        "mid": mid,
                        "distance_pips": round(mid_distance, 2),
                        "range_pips": round(range_pips, 3),
                        "adx": round(adx, 2),
                        "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
                        "trend_ma_pips": round(ma_diff_pips, 2),
                        "ma10": round(ma10, 3) if ma10 > 0 else None,
                        "ma20": round(ma20, 3) if ma20 > 0 else None,
                        "pullback_buffer_pips": pullback_buffer_pips,
                        "pullback_dist_pips": round(pullback_dist_pips, 3)
                        if pullback_dist_pips is not None
                        else None,
                    },
                )
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
            soft = _soft_allow(
                "entry_guard_mid_far_long",
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
            if soft:
                return soft
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
            if pullback_bypass:
                return EntryGuardDecision(
                    True,
                    "entry_guard_pullback",
                    {
                        "entry": entry_price,
                        "lower": lower_guard,
                        "mid": mid,
                        "range_pips": round(range_pips, 3),
                        "adx": round(adx, 2),
                        "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
                        "trend_ma_pips": round(ma_diff_pips, 2),
                        "ma10": round(ma10, 3) if ma10 > 0 else None,
                        "ma20": round(ma20, 3) if ma20 > 0 else None,
                        "pullback_buffer_pips": pullback_buffer_pips,
                        "pullback_dist_pips": round(pullback_dist_pips, 3)
                        if pullback_dist_pips is not None
                        else None,
                    },
                )
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
            soft = _soft_allow(
                "entry_guard_extreme_short",
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
            if soft:
                return soft
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
            if pullback_bypass:
                return EntryGuardDecision(
                    True,
                    "entry_guard_pullback",
                    {
                        "entry": entry_price,
                        "mid": mid,
                        "distance_pips": round(mid_distance, 2),
                        "range_pips": round(range_pips, 3),
                        "adx": round(adx, 2),
                        "trend_dir": "up" if trend_dir > 0 else "down" if trend_dir < 0 else "flat",
                        "trend_ma_pips": round(ma_diff_pips, 2),
                        "ma10": round(ma10, 3) if ma10 > 0 else None,
                        "ma20": round(ma20, 3) if ma20 > 0 else None,
                        "pullback_buffer_pips": pullback_buffer_pips,
                        "pullback_dist_pips": round(pullback_dist_pips, 3)
                        if pullback_dist_pips is not None
                        else None,
                    },
                )
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
            soft = _soft_allow(
                "entry_guard_mid_far_short",
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
            if soft:
                return soft
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

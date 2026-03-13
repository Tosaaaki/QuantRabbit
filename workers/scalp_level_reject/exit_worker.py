from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence, Set

from analysis.range_guard import detect_range_mode
from execution.strategy_entry import set_trade_protections
from execution.position_manager import PositionManager
from indicators.factor_cache import all_factors
from market_data import tick_window
from utils.metrics_logger import log_metric
from utils.oanda_account import get_account_snapshot
from .exit_utils import close_trade, mark_pnl_pips
from .rollout_gate import load_rollout_start_ts, trade_passes_rollout
from .reentry_decider import decide_reentry
from .pro_stop import maybe_close_pro_stop
from .exit_forecast import (
    apply_exit_forecast_to_loss_cut,
    apply_exit_forecast_to_targets,
    build_exit_forecast_adjustment,
)

try:  # optional config
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None


from . import config
from utils.env_utils import env_bool, env_float

_EXIT_ENV_PREFIX = "SCALP_LEVEL_REJECT_EXIT"


def _map_exit_env_prefix() -> None:
    source = f"{_EXIT_ENV_PREFIX}_"
    source_len = len(source)
    for key, value in list(os.environ.items()):
        if not key.startswith(source):
            continue
        target = f"SCALP_LEVEL_REJECT_{key[source_len:]}"
        os.environ.setdefault(target, value)

    os.environ.setdefault("SCALP_LEVEL_REJECT_EXIT_TAGS", os.getenv(f"{_EXIT_ENV_PREFIX}_TAGS", "LevelReject"))
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_EXIT_LOG_PREFIX",
        os.getenv(f"{_EXIT_ENV_PREFIX}_LOG_PREFIX", "[ScalpExit:LevelReject]"),
    )
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_POCKET",
        os.getenv(f"{_EXIT_ENV_PREFIX}_POCKET", "scalp_fast"),
    )
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_EXIT_PROFILE_ENABLED",
        os.getenv(f"{_EXIT_ENV_PREFIX}_PROFILE_ENABLED", os.getenv("SCALP_LEVEL_REJECT_EXIT_PROFILE_ENABLED", "1")),
    )
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_EXIT_PROFILE_PATH",
        os.getenv(
            f"{_EXIT_ENV_PREFIX}_PROFILE_PATH",
            os.getenv("STRATEGY_PROTECTION_PATH", "config/strategy_exit_protections.yaml"),
        ),
    )
    os.environ.setdefault(
        "SCALP_LEVEL_REJECT_EXIT_PROFILE_TTL_SEC",
        os.getenv(f"{_EXIT_ENV_PREFIX}_PROFILE_TTL_SEC", "12.0"),
    )


_map_exit_env_prefix()

_BB_ENV_PREFIX = getattr(config, "ENV_PREFIX", "")
_BB_EXIT_ENABLED = env_bool("BB_EXIT_ENABLED", True, prefix=_BB_ENV_PREFIX)
_BB_EXIT_REVERT_PIPS = env_float("BB_EXIT_REVERT_PIPS", 2.0, prefix=_BB_ENV_PREFIX)
_BB_EXIT_REVERT_RATIO = env_float("BB_EXIT_REVERT_RATIO", 0.20, prefix=_BB_ENV_PREFIX)
_BB_EXIT_TREND_EXT_PIPS = env_float("BB_EXIT_TREND_EXT_PIPS", 3.0, prefix=_BB_ENV_PREFIX)
_BB_EXIT_TREND_EXT_RATIO = env_float("BB_EXIT_TREND_EXT_RATIO", 0.35, prefix=_BB_ENV_PREFIX)
_BB_EXIT_SCALP_REVERT_PIPS = env_float("BB_EXIT_SCALP_REVERT_PIPS", 1.6, prefix=_BB_ENV_PREFIX)
_BB_EXIT_SCALP_REVERT_RATIO = env_float("BB_EXIT_SCALP_REVERT_RATIO", 0.18, prefix=_BB_ENV_PREFIX)
_BB_EXIT_SCALP_EXT_PIPS = env_float("BB_EXIT_SCALP_EXT_PIPS", 2.0, prefix=_BB_ENV_PREFIX)
_BB_EXIT_SCALP_EXT_RATIO = env_float("BB_EXIT_SCALP_EXT_RATIO", 0.28, prefix=_BB_ENV_PREFIX)
_BB_EXIT_MID_BUFFER_PIPS = env_float("BB_EXIT_MID_BUFFER_PIPS", 0.4, prefix=_BB_ENV_PREFIX)
_BB_EXIT_BYPASS_TOKENS = {
    "hard_stop",
    "structure",
    "time_stop",
    "timeout",
    "max_hold",
    "max_adverse",
    "force",
    "margin",
    "risk",
    "health",
    "event",
    "session",
    "halt",
    "liquid",
}
_BB_EXIT_TF = "M1"
_BB_PIP = 0.01

_FALSEY = {"", "0", "false", "no", "off"}
_STRATEGY_PROTECTION_PATH = pathlib.Path(
    os.getenv("SCALP_LEVEL_REJECT_EXIT_PROFILE_PATH", os.getenv("STRATEGY_PROTECTION_PATH", "config/strategy_exit_protections.yaml"))
)
_STRATEGY_PROTECTION_ENABLED = os.getenv("SCALP_LEVEL_REJECT_EXIT_PROFILE_ENABLED", "1").strip().lower() not in _FALSEY
_STRATEGY_PROTECTION_TTL_SEC = max(
    2.0, float(os.getenv("SCALP_LEVEL_REJECT_EXIT_PROFILE_TTL_SEC", "12.0") or 12.0)
)
_STRATEGY_PROTECTION_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}
_STRATEGY_ALIAS_BASE = {
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "donchian55": "Donchian55",
    "h1momentum": "H1Momentum",
    "microlevelreactor": "MicroLevelReactor",
    "microrangebreak": "MicroRangeBreak",
    "microvwapbound": "MicroVWAPBound",
    "momentumburst": "MomentumBurst",
    "techfusion": "TechFusion",
    "macrotechfusion": "MacroTechFusion",
    "micropullbackfib": "MicroPullbackFib",
    "scalpreversalnwave": "ScalpReversalNWave",
    "rangecompressionbreak": "RangeCompressionBreak",
}


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _FALSEY:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return default


def _base_strategy_tag(tag: Optional[str]) -> str:
    if not tag:
        return ""
    text = str(tag).strip()
    if not text:
        return ""
    base = text.split("-", 1)[0].strip() or text
    alias = _STRATEGY_ALIAS_BASE.get(base.lower())
    return alias or base


def _load_strategy_protection_config() -> dict:
    if not _STRATEGY_PROTECTION_ENABLED:
        return {"defaults": {}, "strategies": {}}
    now = time.monotonic()
    cached_ts = float(_STRATEGY_PROTECTION_CACHE.get("ts") or 0.0)
    if (now - cached_ts) < _STRATEGY_PROTECTION_TTL_SEC and isinstance(
        _STRATEGY_PROTECTION_CACHE.get("data"), dict
    ):
        return _STRATEGY_PROTECTION_CACHE["data"]  # type: ignore[return-value]
    payload: dict[str, Any] = {"defaults": {}, "strategies": {}}
    if yaml is not None and _STRATEGY_PROTECTION_PATH.exists():
        try:
            loaded = yaml.safe_load(_STRATEGY_PROTECTION_PATH.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {"defaults": {}, "strategies": {}}
    _STRATEGY_PROTECTION_CACHE["ts"] = now
    _STRATEGY_PROTECTION_CACHE["data"] = payload
    return payload


def _strategy_override(config: dict, strategy_tag: Optional[str]) -> dict:
    if not isinstance(config, dict):
        return {}
    strategies = config.get("strategies")
    if not isinstance(strategies, dict) or not strategy_tag:
        return {}
    base = _base_strategy_tag(strategy_tag)
    candidates = [
        strategy_tag,
        base,
        strategy_tag.lower(),
        base.lower(),
    ]
    for key in candidates:
        if not key:
            continue
        override = strategies.get(key)
        if isinstance(override, dict):
            return override
    return {}


def _merge_profile(base: Optional[dict], override: Optional[dict]) -> dict:
    merged = dict(base) if isinstance(base, dict) else {}
    if isinstance(override, dict):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged.get(key) or {})
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
    return merged


def _exit_profile_for_tag(strategy_tag: Optional[str]) -> dict:
    cfg = _load_strategy_protection_config()
    defaults = cfg.get("defaults") if isinstance(cfg, dict) else {}
    defaults_profile = defaults.get("exit_profile") if isinstance(defaults, dict) else None
    override = _strategy_override(cfg, strategy_tag)
    override_profile = override.get("exit_profile") if isinstance(override, dict) else None
    return _merge_profile(defaults_profile, override_profile)


def _be_profile_for_tag(strategy_tag: Optional[str], *, pocket: str) -> dict:
    cfg = _load_strategy_protection_config()
    defaults = cfg.get("defaults") if isinstance(cfg, dict) else {}
    be_defaults = defaults.get("be_profile") if isinstance(defaults, dict) else None
    pocket_defaults = be_defaults.get(pocket) if isinstance(be_defaults, dict) else None
    if not isinstance(pocket_defaults, dict):
        pocket_defaults = {}
    override = _strategy_override(cfg, strategy_tag)
    override_profile = override.get("be_profile") if isinstance(override, dict) else None
    if not isinstance(override_profile, dict):
        return {}
    return _merge_profile(pocket_defaults, override_profile)


def _tp_move_profile_for_tag(strategy_tag: Optional[str], *, pocket: str) -> dict:
    cfg = _load_strategy_protection_config()
    defaults = cfg.get("defaults") if isinstance(cfg, dict) else {}
    tp_defaults = defaults.get("tp_move") if isinstance(defaults, dict) else None
    pocket_defaults = tp_defaults.get(pocket) if isinstance(tp_defaults, dict) else None
    if not isinstance(pocket_defaults, dict):
        pocket_defaults = {}
    pocket_defaults = dict(pocket_defaults)
    min_gap_default = defaults.get("tp_move_min_gap_pips") if isinstance(defaults, dict) else None
    try:
        min_gap_value = float(min_gap_default) if min_gap_default is not None else None
    except (TypeError, ValueError):
        min_gap_value = None
    if min_gap_value is not None and "min_gap_pips" not in pocket_defaults:
        pocket_defaults["min_gap_pips"] = min_gap_value
    override = _strategy_override(cfg, strategy_tag)
    override_profile = override.get("tp_move") if isinstance(override, dict) else None
    if not isinstance(override_profile, dict):
        return {}
    return _merge_profile(pocket_defaults, override_profile)


def _trade_stop_loss_price(trade: dict) -> Optional[float]:
    sl = trade.get("stop_loss")
    if not isinstance(sl, dict):
        return None
    try:
        price = sl.get("price")
        return float(price) if price is not None else None
    except Exception:
        return None


def _trade_take_profit_price(trade: dict) -> Optional[float]:
    tp = trade.get("take_profit")
    if not isinstance(tp, dict):
        return None
    try:
        price = tp.get("price")
        return float(price) if price is not None else None
    except Exception:
        return None


def _protection_spread_pips(*, bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    try:
        spread = (float(ask) - float(bid)) / _BB_PIP
    except Exception:
        return None
    if not math.isfinite(spread) or spread < 0.0:
        return None
    return spread


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _inventory_stress_exit_trigger(
    *,
    exit_profile: dict,
    thesis: dict,
    pnl: float,
    hold_sec: float,
    atr_pips: Optional[float],
) -> tuple[Optional[str], Optional[str]]:
    if pnl >= 0.0 or not isinstance(exit_profile, dict):
        return None, None
    profile = exit_profile.get("inventory_stress")
    if not isinstance(profile, dict):
        return None, None
    if not _coerce_bool(profile.get("enabled"), False):
        return None, None

    def _pick_float(value: object, default: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(out):
            return default
        return out

    min_hold_sec = max(0.0, _pick_float(profile.get("min_hold_sec"), 0.0))
    if hold_sec < min_hold_sec:
        return None, None

    base_loss_pips = max(0.0, _pick_float(profile.get("loss_pips"), 0.0))
    loss_sl_mult = max(0.0, _pick_float(profile.get("loss_sl_mult"), 0.0))
    loss_atr_mult = max(0.0, _pick_float(profile.get("loss_atr_mult"), 0.0))
    loss_cap_pips = max(0.0, _pick_float(profile.get("loss_cap_pips"), 0.0))
    max_hold_sec = max(0.0, _pick_float(profile.get("max_hold_sec"), 0.0))
    time_loss_pips = max(0.0, _pick_float(profile.get("time_loss_pips"), 0.4))
    cache_ttl_sec = max(0.5, _pick_float(profile.get("cache_ttl_sec"), 2.0))

    adverse_pips = abs(float(pnl))
    threshold_pips = base_loss_pips
    thesis_sl_pips = max(0.0, _pick_float(thesis.get("sl_pips"), 0.0))
    if loss_sl_mult > 0.0 and thesis_sl_pips > 0.0:
        threshold_pips = max(threshold_pips, thesis_sl_pips * loss_sl_mult)
    if loss_atr_mult > 0.0 and atr_pips is not None and atr_pips > 0.0:
        threshold_pips = max(threshold_pips, float(atr_pips) * loss_atr_mult)
    if loss_cap_pips > 0.0 and threshold_pips > 0.0:
        threshold_pips = min(threshold_pips, loss_cap_pips)

    loss_trigger = threshold_pips > 0.0 and adverse_pips >= threshold_pips
    time_trigger = max_hold_sec > 0.0 and hold_sec >= max_hold_sec and adverse_pips >= time_loss_pips
    if not loss_trigger and not time_trigger:
        return None, None

    try:
        snapshot = get_account_snapshot(cache_ttl_sec=cache_ttl_sec)
    except Exception:
        return None, None

    def _snapshot_float(value: object) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    nav = _snapshot_float(getattr(snapshot, "nav", None)) or 0.0
    health_buffer = _snapshot_float(getattr(snapshot, "health_buffer", None))
    free_margin_ratio = _snapshot_float(getattr(snapshot, "free_margin_ratio", None))
    margin_used = _snapshot_float(getattr(snapshot, "margin_used", None))
    unrealized_pl = _snapshot_float(getattr(snapshot, "unrealized_pl", None)) or 0.0
    margin_usage_ratio = (margin_used / nav) if nav > 0.0 and margin_used is not None else None
    unrealized_dd_ratio = abs(unrealized_pl) / nav if nav > 0.0 and unrealized_pl < 0.0 else None

    hb_limit = max(0.0, _pick_float(profile.get("health_buffer"), 0.0))
    free_limit = max(0.0, _pick_float(profile.get("free_margin_ratio"), 0.0))
    usage_limit = max(0.0, _pick_float(profile.get("margin_usage_ratio"), 0.0))
    dd_limit = max(0.0, _pick_float(profile.get("unrealized_dd_ratio"), 0.0))

    if hb_limit > 0.0 and health_buffer is not None and health_buffer <= hb_limit:
        return str(profile.get("reason_health") or "margin_health").strip() or "margin_health", (
            "time" if time_trigger and not loss_trigger else "loss"
        )
    if free_limit > 0.0 and free_margin_ratio is not None and free_margin_ratio <= free_limit:
        return str(profile.get("reason_free_margin") or "free_margin_low").strip() or "free_margin_low", (
            "time" if time_trigger and not loss_trigger else "loss"
        )
    if usage_limit > 0.0 and margin_usage_ratio is not None and margin_usage_ratio >= usage_limit:
        return str(profile.get("reason_margin_usage") or "margin_usage_high").strip() or "margin_usage_high", (
            "time" if time_trigger and not loss_trigger else "loss"
        )
    if dd_limit > 0.0 and unrealized_dd_ratio is not None and unrealized_dd_ratio >= dd_limit:
        return str(profile.get("reason_drawdown") or "drawdown").strip() or "drawdown", (
            "time" if time_trigger and not loss_trigger else "loss"
        )
    return None, None


def _level_reject_live_protection_adjustments(
    *,
    thesis: dict,
    fac_m1: Optional[dict],
    range_active: bool,
    side: str,
    bid: Optional[float],
    ask: Optional[float],
) -> dict[str, float]:
    fac_m1 = fac_m1 if isinstance(fac_m1, dict) else {}

    def _pick_float(value: object) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    atr_pips = _pick_float(fac_m1.get("atr_pips"))
    if atr_pips is None or atr_pips <= 0.0:
        atr_pips = _pick_float(thesis.get("atr_pips"))
    spread_pips = _protection_spread_pips(bid=bid, ask=ask)
    spread_ratio = None
    if spread_pips is not None and atr_pips is not None and atr_pips > 0.0:
        spread_ratio = spread_pips / max(atr_pips, 0.8)

    range_score = _pick_float(thesis.get("range_score"))
    range_reason = str(thesis.get("range_reason") or "").strip().lower()
    extrema = thesis.get("extrema")
    extrema = extrema if isinstance(extrema, dict) else {}
    supportive = bool(extrema.get(f"supportive_{side}", False))
    tick_strength = _pick_float(extrema.get("tick_strength"))
    bounce_pips = _pick_float(
        extrema.get("long_bounce_pips" if side == "long" else "short_bounce_pips")
    )
    dist_edge_pips = _pick_float(
        extrema.get("dist_low_pips" if side == "long" else "dist_high_pips")
    )
    setup_pressure = extrema.get(f"{side}_setup_pressure")
    setup_pressure = setup_pressure if isinstance(setup_pressure, dict) else {}
    sl_rate = _pick_float(setup_pressure.get("sl_rate"))
    fast_sl_rate = _pick_float(setup_pressure.get("fast_sl_rate"))
    net_jpy = _pick_float(setup_pressure.get("net_jpy"))
    ma_gap_pips = _pick_float(extrema.get("ma_gap_pips"))
    weak_probe = any(
        bool(extrema.get(key, False))
        for key in (
            f"{side}_shallow_probe_block",
            f"{side}_mid_rsi_probe_block",
            f"{side}_drift_probe_block",
            f"{side}_setup_pressure_block",
            f"{side}_positive_gap_probe_block",
        )
    )

    stress = 0.0
    support = 0.0

    if spread_ratio is not None and spread_ratio > 0.40:
        stress += min(0.14, (spread_ratio - 0.40) * 0.45)
    elif spread_ratio is not None and spread_ratio < 0.28:
        support += min(0.04, (0.28 - spread_ratio) * 0.20)

    if not supportive and extrema:
        stress += 0.06
    elif supportive:
        support += 0.08

    if tick_strength is not None and tick_strength < 0.55:
        stress += min(0.08, (0.55 - tick_strength) * 0.25)
    elif tick_strength is not None and tick_strength > 0.75:
        support += min(0.06, (tick_strength - 0.75) * 0.24)

    if range_score is not None and range_score < 0.55:
        stress += min(0.06, (0.55 - range_score) * 0.15)
    elif range_score is not None and range_score > 0.68:
        support += min(0.05, (range_score - 0.68) * 0.16)

    if bounce_pips is not None and bounce_pips > 0.9:
        support += min(0.05, (bounce_pips - 0.9) * 0.04)
    if dist_edge_pips is not None and dist_edge_pips <= 0.8:
        support += 0.03

    if sl_rate is not None and sl_rate > 0.42:
        stress += min(0.08, (sl_rate - 0.42) * 0.28)
    if fast_sl_rate is not None and fast_sl_rate > 0.18:
        stress += min(0.06, (fast_sl_rate - 0.18) * 0.22)
    if net_jpy is not None and net_jpy < 0.0:
        stress += min(0.05, abs(net_jpy) / 120.0)
    elif net_jpy is not None and net_jpy > 20.0:
        support += min(0.03, net_jpy / 400.0)

    if ma_gap_pips is not None and atr_pips is not None and abs(ma_gap_pips) > max(1.0, atr_pips * 1.2):
        stress += 0.04
    if weak_probe:
        stress += 0.08
    if range_active and range_reason == "volatility_compression" and not supportive:
        stress += 0.04

    stress = min(0.28, max(0.0, stress))
    support = min(0.22, max(0.0, support))

    return {
        "trigger_mult": round(_clamp_float(1.0 - stress + support, 0.65, 1.22), 3),
        "lock_ratio_mult": round(_clamp_float(1.0 + stress * 0.85 - support * 0.55, 0.76, 1.28), 3),
        "min_lock_mult": round(_clamp_float(1.0 + stress * 0.35 - support * 0.10, 0.90, 1.15), 3),
        "cooldown_mult": round(_clamp_float(1.0 - stress * 0.40 + support * 0.15, 0.65, 1.10), 3),
        "buffer_mult": round(_clamp_float(1.0 - stress * 0.55 + support * 0.80, 0.62, 1.28), 3),
        "min_gap_mult": round(
            _clamp_float(
                1.0 - stress * 0.15 + min(0.20, (spread_ratio or 0.0) * 0.12),
                0.85,
                1.18,
            ),
            3,
        ),
        "stress": round(stress, 3),
        "support": round(support, 3),
        "atr_pips": round(atr_pips, 3) if atr_pips is not None else 0.0,
        "spread_pips": round(spread_pips, 3) if spread_pips is not None else 0.0,
    }


async def _maybe_move_profit_protection(
    *,
    trade: dict,
    trade_id: str,
    strategy_tag: str,
    state: "_TradeState",
    side: str,
    entry: float,
    pnl: float,
    mid: Optional[float],
    profit_take: float,
    now: datetime,
    thesis: Optional[dict] = None,
    fac_m1: Optional[dict] = None,
    range_active: bool = False,
) -> None:
    if pnl <= 0.0 or pnl >= profit_take:
        return
    be_profile = _be_profile_for_tag(strategy_tag, pocket=POCKET)
    tp_move_profile = _tp_move_profile_for_tag(strategy_tag, pocket=POCKET)
    if not be_profile and not tp_move_profile:
        return

    def _pick_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    current_sl = _trade_stop_loss_price(trade)
    current_tp = _trade_take_profit_price(trade)
    desired_sl = current_sl
    desired_tp = current_tp
    sl_updated = False
    tp_updated = False
    lock_metric_pips: Optional[float] = None
    now_mono = time.monotonic()
    bid, ask = _latest_bid_ask()
    side_ref = bid if side == "long" else ask
    if side_ref is None:
        side_ref = mid
    live_adj = _level_reject_live_protection_adjustments(
        thesis=thesis if isinstance(thesis, dict) else {},
        fac_m1=fac_m1,
        range_active=range_active,
        side=side,
        bid=bid,
        ask=ask,
    )

    if isinstance(be_profile, dict):
        trigger_pips = max(
            0.0,
            _pick_float(be_profile.get("trigger_pips"), 0.0) * float(live_adj.get("trigger_mult", 1.0)),
        )
        lock_ratio = max(
            0.0,
            min(
                1.0,
                _pick_float(be_profile.get("lock_ratio"), 0.0) * float(live_adj.get("lock_ratio_mult", 1.0)),
            ),
        )
        min_lock_pips = max(
            0.0,
            _pick_float(be_profile.get("min_lock_pips"), 0.0) * float(live_adj.get("min_lock_mult", 1.0)),
        )
        cooldown_sec = max(
            0.0,
            _pick_float(be_profile.get("cooldown_sec"), 0.0) * float(live_adj.get("cooldown_mult", 1.0)),
        )
        if (
            trigger_pips > 0.0
            and pnl >= trigger_pips
            and lock_ratio > 0.0
            and (cooldown_sec <= 0.0 or (now_mono - float(state.be_last_ts or 0.0)) >= cooldown_sec)
        ):
            lock_pips = max(min_lock_pips, float(pnl) * lock_ratio)
            candidate_sl = entry + lock_pips * _BB_PIP if side == "long" else entry - lock_pips * _BB_PIP
            if side_ref is not None:
                if side == "long":
                    candidate_sl = min(candidate_sl, float(side_ref) - 0.003)
                else:
                    candidate_sl = max(candidate_sl, float(side_ref) + 0.003)
            if side == "long":
                if (
                    candidate_sl > entry + 1e-6
                    and (current_sl is None or candidate_sl > current_sl + 1e-6)
                ):
                    desired_sl = round(candidate_sl, 3)
                    sl_updated = True
                    lock_metric_pips = lock_pips
            else:
                if (
                    candidate_sl < entry - 1e-6
                    and (current_sl is None or candidate_sl < current_sl - 1e-6)
                ):
                    desired_sl = round(candidate_sl, 3)
                    sl_updated = True
                    lock_metric_pips = lock_pips

    effective_sl = desired_sl if sl_updated else current_sl
    locked_profit = False
    if effective_sl is not None:
        locked_profit = (
            (side == "long" and effective_sl > entry + 1e-6)
            or (side == "short" and effective_sl < entry - 1e-6)
        )

    if isinstance(tp_move_profile, dict) and locked_profit:
        enabled = _coerce_bool(tp_move_profile.get("enabled"), True)
        trigger_pips = max(
            0.0,
            _pick_float(tp_move_profile.get("trigger_pips"), 0.0) * float(live_adj.get("trigger_mult", 1.0)),
        )
        buffer_pips = max(
            0.0,
            _pick_float(tp_move_profile.get("buffer_pips"), 0.0) * float(live_adj.get("buffer_mult", 1.0)),
        )
        min_gap_pips = max(
            0.3,
            _pick_float(tp_move_profile.get("min_gap_pips"), 0.3) * float(live_adj.get("min_gap_mult", 1.0)),
        )
        tp_ref = mid if mid is not None else side_ref
        if enabled and trigger_pips > 0.0 and pnl >= trigger_pips and tp_ref is not None:
            if side == "long":
                candidate_tp = max(
                    entry + min_gap_pips * _BB_PIP,
                    float(tp_ref) + buffer_pips * _BB_PIP,
                )
                if current_tp is None or current_tp - candidate_tp > 1e-6:
                    desired_tp = round(candidate_tp, 3)
                    tp_updated = True
            else:
                candidate_tp = min(
                    entry - min_gap_pips * _BB_PIP,
                    float(tp_ref) - buffer_pips * _BB_PIP,
                )
                if current_tp is None or candidate_tp - current_tp > 1e-6:
                    desired_tp = round(candidate_tp, 3)
                    tp_updated = True

    if not sl_updated and not tp_updated:
        return

    state.be_last_ts = now_mono
    ok = await set_trade_protections(
        trade_id,
        sl_price=desired_sl if (sl_updated or current_sl is not None) else None,
        tp_price=desired_tp if (tp_updated or current_tp is not None) else None,
    )
    if not ok:
        return

    strategy_label = _base_strategy_tag(strategy_tag) or strategy_tag or "unknown"
    if sl_updated:
        state.be_last_sl = desired_sl
        if lock_metric_pips is not None:
            log_metric(
                "scalp_level_reject_profit_lock_sl",
                lock_metric_pips,
                tags={"strategy": strategy_label, "side": side},
                ts=now,
            )
    if tp_updated:
        log_metric(
            "scalp_level_reject_tp_move",
            pnl,
            tags={"strategy": strategy_label, "side": side},
            ts=now,
        )
    LOG.info(
        "[exit-rangefader] protection_move trade=%s tag=%s pnl=%.2fp sl=%s tp=%s stress=%.3f support=%.3f atr=%.2f spread=%.2f",
        trade_id,
        strategy_label,
        pnl,
        f"{desired_sl:.3f}" if desired_sl is not None else "-",
        f"{desired_tp:.3f}" if desired_tp is not None else "-",
        float(live_adj.get("stress", 0.0)),
        float(live_adj.get("support", 0.0)),
        float(live_adj.get("atr_pips", 0.0)),
        float(live_adj.get("spread_pips", 0.0)),
    )


def _reentry_prefix_for_tag(tag: str) -> Optional[str]:
    raw = str(tag or "").strip()
    if not raw:
        return None
    base = raw.split("-", 1)[0].strip()
    if not base:
        return None
    out = []
    for ch in base:
        if ch.isalnum():
            out.append(ch.upper())
        else:
            out.append("_")
    prefix = "".join(out).strip("_")
    return prefix or None


def _reentry_enabled_for_tag(prefix: Optional[str]) -> bool:
    if not prefix:
        return False
    raw = os.getenv(f"{prefix}_REENTRY_ENABLE")
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes"}


def _bb_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bb_levels(fac):
    if not fac:
        return None
    upper = _bb_float(fac.get("bb_upper"))
    lower = _bb_float(fac.get("bb_lower"))
    mid = _bb_float(fac.get("bb_mid")) or _bb_float(fac.get("ma20"))
    bbw = _bb_float(fac.get("bbw")) or 0.0
    if upper is None or lower is None:
        if mid is None or bbw <= 0:
            return None
        half = abs(mid) * bbw / 2.0
        upper = mid + half
        lower = mid - half
    span = upper - lower
    if span <= 0:
        return None
    return upper, mid if mid is not None else (upper + lower) / 2.0, lower, span, span / _BB_PIP


def _bb_exit_price(fac):
    price = None
    latest_mid = globals().get("_latest_mid")
    if callable(latest_mid):
        try:
            price = latest_mid()
        except Exception:
            price = None
    if price is None:
        latest_quote = globals().get("_latest_quote")
        if callable(latest_quote):
            try:
                _, _, mid = latest_quote()
                price = mid
            except Exception:
                price = None
    if price is None or price <= 0:
        try:
            price = float(fac.get("close") or 0.0)
        except Exception:
            price = None
    return price


def _bb_exit_should_bypass(reason, pnl, allow_negative):
    if allow_negative:
        return True
    if pnl is not None:
        try:
            if float(pnl) <= 0:
                return True
        except Exception:
            pass
    if not reason:
        return False
    reason_key = str(reason).lower()
    for token in _BB_EXIT_BYPASS_TOKENS:
        if token in reason_key:
            return True
    return False


def _bb_exit_allowed(style, side, price, fac, *, range_active=None):
    if not _BB_EXIT_ENABLED:
        return True
    if price is None or price <= 0:
        return True
    levels = _bb_levels(fac)
    if not levels:
        return True
    upper, mid, lower, span, span_pips = levels
    side_key = str(side or "").lower()
    if side_key in {"buy", "long", "open_long"}:
        direction = "long"
    else:
        direction = "short"
    orig_style = style
    if style == "scalp" and range_active:
        style = "reversion"
    mid_buffer = max(_BB_EXIT_MID_BUFFER_PIPS, span_pips * 0.05)
    if style == "reversion":
        base_pips = _BB_EXIT_SCALP_REVERT_PIPS if orig_style == "scalp" else _BB_EXIT_REVERT_PIPS
        base_ratio = _BB_EXIT_SCALP_REVERT_RATIO if orig_style == "scalp" else _BB_EXIT_REVERT_RATIO
        threshold = max(base_pips, span_pips * base_ratio)
        if direction == "long":
            dist = (price - lower) / _BB_PIP
        else:
            dist = (upper - price) / _BB_PIP
        return dist >= threshold
    band_buffer = max(_BB_EXIT_TREND_EXT_PIPS, span_pips * _BB_EXIT_TREND_EXT_RATIO)
    if orig_style == "scalp":
        band_buffer = max(_BB_EXIT_SCALP_EXT_PIPS, span_pips * _BB_EXIT_SCALP_EXT_RATIO)
    if direction == "long":
        if price <= mid - mid_buffer * _BB_PIP:
            return True
        return price >= (upper - band_buffer * _BB_PIP)
    if price >= mid + mid_buffer * _BB_PIP:
        return True
    return price <= (lower + band_buffer * _BB_PIP)

BB_STYLE = "scalp"
LOG = logging.getLogger(__name__)
LOG_PREFIX = os.getenv("SCALP_LEVEL_REJECT_EXIT_LOG_PREFIX", "[ScalpPrecisionExit]")


# Tags that scalp_level_reject exit_worker is allowed to manage. Missing tags effectively disable
# exit protections (loss cuts, max-hold, etc.) for those strategies, so keep this list complete.
ALLOWED_TAGS: Set[str] = {
    "SpreadRangeRevert",
    "RangeFaderPro",
    "VwapRevertS",
    "StochBollBounce",
    "DivergenceRevert",
    "CompressionRetest",
    "HTFPullbackS",
    "MacdTrendRide",
    "EmaSlopePull",
    "TickImbalance",
    "TickImbalanceRRPlus",
    "LevelReject",
    "LiquiditySweep",
    "WickReversal",
    "WickReversalHF",
    "WickReversalPro",
    "TickWickReversal",
    "SessionEdge",
    "SqueezePulseBreak",
    "FalseBreakFade",
}


def _tags_env(key: str, default: Set[str]) -> Set[str]:
    raw = os.getenv(key)
    if raw is None:
        return default
    tags = {t.strip() for t in raw.replace(";", ",").split(",") if t.strip()}
    return tags or default


def _pocket_env(key: str, default: str) -> str:
    raw = os.getenv(key)
    return raw.strip().lower() if raw else default


POCKET = _pocket_env("SCALP_LEVEL_REJECT_POCKET", "scalp")
ALLOWED_TAGS = _tags_env("SCALP_LEVEL_REJECT_EXIT_TAGS", ALLOWED_TAGS)


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _latest_mid() -> Optional[float]:
    tick = tick_window.recent_ticks(seconds=2.0, limit=1)
    if tick:
        try:
            return float(tick[-1]["mid"])
        except Exception:
            pass
    try:
        return float(all_factors().get("M1", {}).get("close"))
    except Exception:
        return None


def _latest_bid_ask() -> tuple[Optional[float], Optional[float]]:
    tick = tick_window.recent_ticks(seconds=2.0, limit=1)
    if not tick:
        return None, None
    last = tick[-1] or {}
    try:
        bid = float(last.get("bid")) if last.get("bid") is not None else None
    except Exception:
        bid = None
    try:
        ask = float(last.get("ask")) if last.get("ask") is not None else None
    except Exception:
        ask = None
    return bid, ask


def _filter_trades(trades: Sequence[dict], tags: Set[str]) -> list[dict]:
    if not tags:
        return []
    filtered: list[dict] = []
    for tr in trades:
        thesis = tr.get("entry_thesis") or {}
        tag = (
            thesis.get("strategy_tag")
            or thesis.get("strategy_tag_raw")
            or thesis.get("strategy")
            or thesis.get("tag")
            or tr.get("strategy_tag")
            or tr.get("strategy")
        )
        if not tag:
            continue
        tag_str = str(tag)
        base_tag = tag_str.split("-", 1)[0]
        if tag_str in tags or base_tag in tags:
            filtered.append(tr)
    return filtered


@dataclass
class _TradeState:
    peak: float
    lock_floor: Optional[float] = None
    partial_done: bool = False
    be_moved: bool = False
    be_floor_price: Optional[float] = None
    be_last_ts: float = 0.0
    be_last_sl: Optional[float] = None

    def update(self, pnl: float, lock_buffer: float) -> None:
        if pnl > self.peak:
            self.peak = pnl
        if pnl > 0:
            floor = max(0.0, pnl - lock_buffer)
            self.lock_floor = floor if self.lock_floor is None else max(self.lock_floor, floor)


class RangeFaderExitWorker:
    def __init__(self) -> None:
        self.loop_interval = max(
            0.5,
            _float_env("RANGEFADER_EXIT_LOOP_INTERVAL_SEC", 0.7),
        )
        # When no relevant open trades exist, poll much less frequently to avoid
        # N exit workers hammering PositionManager/OANDA in parallel.
        self.idle_interval = max(
            self.loop_interval,
            _float_env("RANGEFADER_EXIT_IDLE_INTERVAL_SEC", 3.0),
        )
        self.min_hold_sec = max(
            5.0,
            _float_env("RANGEFADER_EXIT_MIN_HOLD_SEC", 10.0),
        )
        self.profit_take = max(
            0.5,
            _float_env("RANGEFADER_EXIT_PROFIT_PIPS", 1.5),
        )
        self.trail_start = max(
            0.5,
            _float_env("RANGEFADER_EXIT_TRAIL_START_PIPS", 2.1),
        )
        self.trail_backoff = max(
            0.1,
            _float_env("RANGEFADER_EXIT_TRAIL_BACKOFF_PIPS", 0.7),
        )
        self.lock_buffer = max(
            0.05,
            _float_env("RANGEFADER_EXIT_LOCK_BUFFER_PIPS", 0.35),
        )
        self.range_profit_take = max(
            0.4,
            _float_env("RANGEFADER_EXIT_RANGE_PROFIT_PIPS", 1.2),
        )
        self.range_trail_start = max(
            0.4,
            _float_env("RANGEFADER_EXIT_RANGE_TRAIL_START_PIPS", 1.7),
        )
        self.range_trail_backoff = max(
            0.1,
            _float_env("RANGEFADER_EXIT_RANGE_TRAIL_BACKOFF_PIPS", 0.55),
        )
        self.range_lock_buffer = max(
            0.05,
            _float_env("RANGEFADER_EXIT_RANGE_LOCK_BUFFER_PIPS", 0.25),
        )
        self.range_max_hold_sec = max(
            60.0,
            _float_env("RANGEFADER_EXIT_RANGE_MAX_HOLD_SEC", 600.0),
        )

        self.loss_cut_enabled = _bool_env("RANGEFADER_EXIT_LOSS_CUT_ENABLED", False)
        self.loss_cut_require_sl = _bool_env("RANGEFADER_EXIT_LOSS_CUT_REQUIRE_SL", True)
        self.loss_cut_soft_pips = max(0.0, _float_env("RANGEFADER_EXIT_LOSS_CUT_SOFT_PIPS", 12.0))
        self.loss_cut_hard_pips = max(
            self.loss_cut_soft_pips,
            _float_env("RANGEFADER_EXIT_LOSS_CUT_HARD_PIPS", 30.0),
        )
        self.loss_cut_max_hold_sec = max(0.0, _float_env("RANGEFADER_EXIT_LOSS_CUT_MAX_HOLD_SEC", 1200.0))
        self.loss_cut_cooldown_sec = max(0.0, _float_env("RANGEFADER_EXIT_LOSS_CUT_COOLDOWN_SEC", 8.0))
        self.loss_cut_reason_soft = (
            os.getenv("RANGEFADER_EXIT_LOSS_CUT_REASON_SOFT", "m1_structure_break").strip()
            or "m1_structure_break"
        )
        self.loss_cut_reason_hard = (
            os.getenv("RANGEFADER_EXIT_LOSS_CUT_REASON_HARD", "max_adverse").strip() or "max_adverse"
        )
        self.loss_cut_reason_time = (
            os.getenv("RANGEFADER_EXIT_LOSS_CUT_REASON_TIME", "time_stop").strip() or "time_stop"
        )

        self.tick_imb_tags = _tags_env("TICK_IMB_EXIT_TAGS", {"TickImbalance", "TickImbalanceRRPlus"})
        self.tick_imb_partial_enabled = _bool_env("TICK_IMB_EXIT_PARTIAL_ENABLED", True)
        self.tick_imb_partial_trigger = max(0.4, _float_env("TICK_IMB_EXIT_PARTIAL_TRIGGER_PIPS", 1.0))
        self.tick_imb_partial_fraction = min(
            0.8, max(0.1, _float_env("TICK_IMB_EXIT_PARTIAL_FRACTION", 0.4))
        )
        self.tick_imb_partial_min_units = max(20, int(_float_env("TICK_IMB_EXIT_PARTIAL_MIN_UNITS", 1000)))
        self.tick_imb_partial_min_remaining = max(
            20, int(_float_env("TICK_IMB_EXIT_PARTIAL_MIN_REMAIN", 1000))
        )
        self.tick_imb_be_buffer_pips = max(0.05, _float_env("TICK_IMB_EXIT_BE_BUFFER_PIPS", 0.2))
        self.tick_imb_min_hold_sec = max(0.0, _float_env("TICK_IMB_EXIT_MIN_HOLD_SEC", 0.0))
        # NOTE: these envs default to 0.0 and are intended as opt-in overrides during tuning.
        # If the key is not set, config/strategy_exit_protections.yaml remains authoritative.
        self._tick_imb_profit_take_override = os.getenv("TICK_IMB_EXIT_PROFIT_PIPS") is not None
        self.tick_imb_profit_take = max(0.0, _float_env("TICK_IMB_EXIT_PROFIT_PIPS", 0.0))
        self._tick_imb_trail_start_override = os.getenv("TICK_IMB_EXIT_TRAIL_START_PIPS") is not None
        self.tick_imb_trail_start = max(0.0, _float_env("TICK_IMB_EXIT_TRAIL_START_PIPS", 0.0))
        self._tick_imb_trail_backoff_override = os.getenv("TICK_IMB_EXIT_TRAIL_BACKOFF_PIPS") is not None
        self.tick_imb_trail_backoff = max(0.0, _float_env("TICK_IMB_EXIT_TRAIL_BACKOFF_PIPS", 0.0))
        self._tick_imb_lock_buffer_override = os.getenv("TICK_IMB_EXIT_LOCK_BUFFER_PIPS") is not None
        self.tick_imb_lock_buffer = max(0.0, _float_env("TICK_IMB_EXIT_LOCK_BUFFER_PIPS", 0.0))
        self._tick_imb_max_hold_sec_override = os.getenv("TICK_IMB_EXIT_MAX_HOLD_SEC") is not None
        self.tick_imb_max_hold_sec = max(0.0, _float_env("TICK_IMB_EXIT_MAX_HOLD_SEC", 0.0))
        self.tick_imb_max_adverse_pips = max(0.0, _float_env("TICK_IMB_EXIT_MAX_ADVERSE_PIPS", 0.0))
        # Safety knob: cap config-driven max_adverse without editing YAML (useful for rollouts/tests).
        self.tick_imb_max_adverse_cap_pips = max(
            0.0, _float_env("TICK_IMB_EXIT_MAX_ADVERSE_CAP_PIPS", 0.0)
        )
        # When hard SL is disabled, cap tail losses mechanically using the intended entry SL as a reference.
        # These are used only when max_adverse_pips is not explicitly configured.
        self.tick_imb_max_adverse_from_sl_mult = max(
            0.0, _float_env("TICK_IMB_EXIT_MAX_ADVERSE_FROM_SL_MULT", 1.6)
        )
        self.tick_imb_max_adverse_from_sl_min = max(
            0.0, _float_env("TICK_IMB_EXIT_MAX_ADVERSE_FROM_SL_MIN_PIPS", 2.0)
        )
        self.tick_imb_max_adverse_from_sl_max = max(
            0.0, _float_env("TICK_IMB_EXIT_MAX_ADVERSE_FROM_SL_MAX_PIPS", 6.0)
        )
        self.exit_policy_start_ts = load_rollout_start_ts("SCALP_LEVEL_REJECT_EXIT_POLICY_START_TS")

        self._pos_manager = PositionManager()
        self._states: dict[str, _TradeState] = {}
        self._loss_cut_last_ts: dict[str, float] = {}
        self._rollout_skip_log_ts: dict[str, float] = {}

    def _unrealized_pnl_pips(
        self,
        trade: dict,
        *,
        entry: float,
        units: int,
        mid: Optional[float],
    ) -> Optional[float]:
        """Return current PnL in pips for EXIT decisions.

        Prefer OANDA-provided unrealized PnL (already normalized to pips by PositionManager) so the exit loop
        doesn't stall when local tick data is stale/missing.
        """
        raw = trade.get("unrealized_pl_pips")
        if raw is not None:
            try:
                value = float(raw)
                if math.isfinite(value):
                    return value
            except Exception:
                pass
        raw_pl = trade.get("unrealized_pl")
        if raw_pl is not None:
            try:
                unrealized_pl = float(raw_pl)
                pip_value = abs(int(units)) * 0.01
                if pip_value > 0:
                    value = unrealized_pl / pip_value
                    if math.isfinite(value):
                        return value
            except Exception:
                pass
        if mid is not None:
            return mark_pnl_pips(entry, units, mid=mid)
        return None

    def _context(self) -> tuple[Optional[float], bool]:
        fac_m1 = all_factors().get("M1") or {}
        fac_h4 = all_factors().get("H4") or {}
        range_active = False
        try:
            range_active = bool(detect_range_mode(fac_m1, fac_h4).active)
        except Exception:
            range_active = False
        return _latest_mid(), range_active

    def _trade_has_stop_loss(self, trade: dict) -> bool:
        sl = trade.get("stop_loss")
        if isinstance(sl, dict):
            price = sl.get("price")
            try:
                return float(price) > 0.0
            except (TypeError, ValueError):
                return False
        return False

    def _loss_cut_eligible(
        self,
        trade: dict,
        *,
        enabled: Optional[bool] = None,
        require_sl: Optional[bool] = None,
    ) -> bool:
        loss_cut_enabled = self.loss_cut_enabled if enabled is None else bool(enabled)
        if not loss_cut_enabled:
            return False
        loss_cut_require_sl = self.loss_cut_require_sl if require_sl is None else bool(require_sl)
        if not loss_cut_require_sl:
            return True
        return self._trade_has_stop_loss(trade)

    def _maybe_log_rollout_skip(
        self,
        *,
        trade_id: str,
        side: str,
        pnl: float,
        hold_sec: float,
        reason: str,
    ) -> None:
        now_mono = time.monotonic()
        last = float(self._rollout_skip_log_ts.get(trade_id) or 0.0)
        if now_mono - last < 60.0:
            return
        self._rollout_skip_log_ts[trade_id] = now_mono
        log_metric(
            "scalp_level_reject_rollout_skip",
            float(pnl),
            tags={
                "reason": reason,
                "side": side,
            },
        )
        LOG.info(
            "[exit-rangefader] rollout skip trade=%s reason=%s pnl=%.2fp hold=%.0fs",
            trade_id,
            reason,
            pnl,
            hold_sec,
        )

    async def _close(self, trade_id: str, units: int, reason: str, pnl: float, client_order_id: Optional[str], allow_negative: bool = False) -> None:
        if _BB_EXIT_ENABLED:
            allow_neg = bool(locals().get("allow_negative"))
            pnl_val = locals().get("pnl")
            if not _bb_exit_should_bypass(reason, pnl_val, allow_neg):
                fac = all_factors().get(_BB_EXIT_TF) or {}
                price = _bb_exit_price(fac)
                side = "long" if units > 0 else "short"
                if not _bb_exit_allowed(BB_STYLE, side, price, fac):
                    LOG.info("[exit-bb] trade=%s reason=%s price=%.3f", trade_id, reason, price or 0.0)
                    return
        if pnl <= 0:
            allow_negative = True
        ok = await close_trade(
            trade_id,
            units,
            client_order_id=client_order_id,
            allow_negative=allow_negative,
            exit_reason=reason,
            env_prefix=_BB_ENV_PREFIX,
        )
        if ok:
            LOG.info("[exit-rangefader] trade=%s units=%s reason=%s pnl=%.2fp", trade_id, units, reason, pnl)
        else:
            LOG.error("[exit-rangefader] close failed trade=%s units=%s reason=%s", trade_id, units, reason)

    async def _review_trade(self, trade: dict, now: datetime, mid: Optional[float], range_active: bool) -> None:
        trade_id = str(trade.get("trade_id"))
        if not trade_id:
            return
        units = int(trade.get("units", 0) or 0)
        if units == 0:
            return
        entry = float(trade.get("price") or 0.0)
        if entry <= 0.0:
            return

        side = "long" if units > 0 else "short"
        pnl = self._unrealized_pnl_pips(trade, entry=entry, units=units, mid=mid)
        if pnl is None:
            return
        opened_at = _parse_time(trade.get("open_time"))
        hold_sec = (now - opened_at).total_seconds() if opened_at else 0.0
        policy_active = trade_passes_rollout(
            opened_at,
            self.exit_policy_start_ts,
            unknown_is_new=False,
        )

        thesis = trade.get("entry_thesis") or {}
        if isinstance(thesis, str):
            try:
                thesis = json.loads(thesis) or {}
            except Exception:
                thesis = {}
        if not isinstance(thesis, dict):
            thesis = {}
        strategy_tag = (
            thesis.get("strategy_tag")
            or thesis.get("strategy_tag_raw")
            or thesis.get("strategy")
            or thesis.get("tag")
            or trade.get("strategy_tag")
            or trade.get("strategy")
            or ""
        )
        base_tag = str(strategy_tag).split("-", 1)[0] if strategy_tag else ""
        is_tick_imb = bool(base_tag and base_tag in self.tick_imb_tags)
        exit_profile = _exit_profile_for_tag(base_tag or strategy_tag)
        fac_m1 = all_factors().get("M1") or {}

        def _pick_float(value: object, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _pick_int(value: object, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        min_hold_sec = _pick_float(exit_profile.get("min_hold_sec"), self.min_hold_sec)
        profit_take = _pick_float(exit_profile.get("profit_pips"), self.profit_take)
        trail_start = _pick_float(exit_profile.get("trail_start_pips"), self.trail_start)
        trail_backoff = _pick_float(exit_profile.get("trail_backoff_pips"), self.trail_backoff)
        lock_buffer = _pick_float(exit_profile.get("lock_buffer_pips"), self.lock_buffer)
        lock_floor_min_hold_sec = max(
            0.0, _pick_float(exit_profile.get("lock_floor_min_hold_sec"), 0.0)
        )
        range_profit_take = _pick_float(exit_profile.get("range_profit_pips"), self.range_profit_take)
        range_trail_start = _pick_float(exit_profile.get("range_trail_start_pips"), self.range_trail_start)
        range_trail_backoff = _pick_float(exit_profile.get("range_trail_backoff_pips"), self.range_trail_backoff)
        range_lock_buffer = _pick_float(exit_profile.get("range_lock_buffer_pips"), self.range_lock_buffer)
        range_max_hold_sec = _pick_float(exit_profile.get("range_max_hold_sec"), self.range_max_hold_sec)
        loss_cut_enabled = _coerce_bool(exit_profile.get("loss_cut_enabled"), self.loss_cut_enabled)
        loss_cut_require_sl = _coerce_bool(exit_profile.get("loss_cut_require_sl"), self.loss_cut_require_sl)
        loss_cut_soft_pips = _pick_float(exit_profile.get("loss_cut_soft_pips"), self.loss_cut_soft_pips)
        loss_cut_hard_pips = max(
            loss_cut_soft_pips,
            _pick_float(exit_profile.get("loss_cut_hard_pips"), self.loss_cut_hard_pips),
        )
        loss_cut_max_hold_sec = _pick_float(exit_profile.get("loss_cut_max_hold_sec"), self.loss_cut_max_hold_sec)
        loss_cut_cooldown_sec = _pick_float(
            exit_profile.get("loss_cut_cooldown_sec"), self.loss_cut_cooldown_sec
        )
        loss_cut_reason_soft = (
            str(exit_profile.get("loss_cut_reason_soft") or self.loss_cut_reason_soft).strip()
            or self.loss_cut_reason_soft
        )
        loss_cut_reason_hard = (
            str(exit_profile.get("loss_cut_reason_hard") or self.loss_cut_reason_hard).strip()
            or self.loss_cut_reason_hard
        )
        loss_cut_reason_time = (
            str(exit_profile.get("loss_cut_reason_time") or self.loss_cut_reason_time).strip()
            or self.loss_cut_reason_time
        )
        loss_cut_hard_atr_mult = max(
            0.0, _pick_float(exit_profile.get("loss_cut_hard_atr_mult"), 0.0)
        )
        loss_cut_hard_cap_pips = max(
            0.0, _pick_float(exit_profile.get("loss_cut_hard_cap_pips"), 0.0)
        )
        loss_cut_reversion_enabled = _coerce_bool(
            exit_profile.get("loss_cut_reversion_enabled"), False
        )
        loss_cut_reversion_hard_pips = max(
            0.0, _pick_float(exit_profile.get("loss_cut_reversion_hard_pips"), 0.0)
        )
        loss_cut_reversion_vwap_gap_min = max(
            0.0, _pick_float(exit_profile.get("loss_cut_reversion_vwap_gap_min"), 0.0)
        )
        loss_cut_reversion_adx_max = max(
            0.0, _pick_float(exit_profile.get("loss_cut_reversion_adx_max"), 0.0)
        )
        loss_cut_reversion_rsi_short_min = _pick_float(
            exit_profile.get("loss_cut_reversion_rsi_short_min"), 100.0
        )
        loss_cut_reversion_rsi_long_max = _pick_float(
            exit_profile.get("loss_cut_reversion_rsi_long_max"), 0.0
        )
        if str(base_tag).lower() == "levelreject":
            # LevelReject is tuned via config/strategy_exit_protections.yaml.
            # Keep that configuration authoritative (avoid VWAP-gap-based overrides that
            # tended to cause premature max_adverse closes and occasional tail risk).
            loss_cut_enabled = True
            loss_cut_require_sl = False
        forecast_adj = build_exit_forecast_adjustment(
            side=side,
            entry_thesis=thesis,
            env_prefix=_BB_ENV_PREFIX,
        )
        profit_take, trail_start, trail_backoff, lock_buffer = apply_exit_forecast_to_targets(
            profit_take=profit_take,
            trail_start=trail_start,
            trail_backoff=trail_backoff,
            lock_buffer=lock_buffer,
            adjustment=forecast_adj,
            profit_take_floor=0.5,
            trail_start_floor=0.5,
            trail_backoff_floor=0.05,
            lock_buffer_floor=0.05,
        )
        range_profit_take, range_trail_start, range_trail_backoff, range_lock_buffer = apply_exit_forecast_to_targets(
            profit_take=range_profit_take,
            trail_start=range_trail_start,
            trail_backoff=range_trail_backoff,
            lock_buffer=range_lock_buffer,
            adjustment=forecast_adj,
            profit_take_floor=0.5,
            trail_start_floor=0.5,
            trail_backoff_floor=0.05,
            lock_buffer_floor=0.05,
        )
        loss_cut_soft_pips, loss_cut_hard_pips, loss_cut_max_hold_sec = apply_exit_forecast_to_loss_cut(
            soft_pips=loss_cut_soft_pips,
            hard_pips=loss_cut_hard_pips,
            max_hold_sec=loss_cut_max_hold_sec,
            adjustment=forecast_adj,
            floor_pips=0.1,
        )
        if "non_range_max_hold_sec" in locals() and non_range_max_hold_sec > 0.0 and forecast_adj.enabled:
            non_range_max_hold_sec = max(min_hold_sec, non_range_max_hold_sec * forecast_adj.max_hold_mult)
        tick_imb_profile = exit_profile.get("tick_imb")
        if not isinstance(tick_imb_profile, dict):
            tick_imb_profile = {}
        inventory_stress_profile = exit_profile.get("inventory_stress")
        if not isinstance(inventory_stress_profile, dict):
            inventory_stress_profile = {}
        tick_imb_partial_enabled = _coerce_bool(
            tick_imb_profile.get("partial_enabled"), self.tick_imb_partial_enabled
        )
        tick_imb_partial_trigger = _pick_float(
            tick_imb_profile.get("partial_trigger_pips"), self.tick_imb_partial_trigger
        )
        tick_imb_partial_fraction = min(
            0.8, max(0.1, _pick_float(tick_imb_profile.get("partial_fraction"), self.tick_imb_partial_fraction))
        )
        tick_imb_partial_min_units = max(
            20, _pick_int(tick_imb_profile.get("partial_min_units"), self.tick_imb_partial_min_units)
        )
        tick_imb_partial_min_remaining = max(
            20, _pick_int(tick_imb_profile.get("partial_min_remaining"), self.tick_imb_partial_min_remaining)
        )
        tick_imb_be_buffer_pips = _pick_float(
            tick_imb_profile.get("be_buffer_pips"), self.tick_imb_be_buffer_pips
        )
        tick_imb_min_hold_sec = max(
            0.0, _pick_float(tick_imb_profile.get("min_hold_sec"), self.tick_imb_min_hold_sec)
        )
        tick_imb_profit_take = max(
            0.0, _pick_float(tick_imb_profile.get("profit_pips"), self.tick_imb_profit_take)
        )
        tick_imb_trail_start = max(
            0.0, _pick_float(tick_imb_profile.get("trail_start_pips"), self.tick_imb_trail_start)
        )
        tick_imb_trail_backoff = max(
            0.0, _pick_float(tick_imb_profile.get("trail_backoff_pips"), self.tick_imb_trail_backoff)
        )
        tick_imb_lock_buffer = max(
            0.0, _pick_float(tick_imb_profile.get("lock_buffer_pips"), self.tick_imb_lock_buffer)
        )
        tick_imb_max_hold_sec = max(
            0.0, _pick_float(tick_imb_profile.get("max_hold_sec"), self.tick_imb_max_hold_sec)
        )
        if is_tick_imb:
            # Opt-in overrides to enable fast replay sweeps without editing YAML.
            if self._tick_imb_profit_take_override:
                tick_imb_profit_take = float(self.tick_imb_profit_take or 0.0)
            if self._tick_imb_trail_start_override:
                tick_imb_trail_start = float(self.tick_imb_trail_start or 0.0)
            if self._tick_imb_trail_backoff_override:
                tick_imb_trail_backoff = float(self.tick_imb_trail_backoff or 0.0)
            if self._tick_imb_lock_buffer_override:
                tick_imb_lock_buffer = float(self.tick_imb_lock_buffer or 0.0)
            if self._tick_imb_max_hold_sec_override:
                tick_imb_max_hold_sec = float(self.tick_imb_max_hold_sec or 0.0)
        cfg_tick_imb_max_adverse_pips = _pick_float(tick_imb_profile.get("max_adverse_pips"), 0.0)
        # Treat non-positive config values as "unset" so env-derived defaults still work.
        # (config/strategy_exit_protections.yaml uses 0 for "no override" in most fields.)
        tick_imb_max_adverse_pips = (
            cfg_tick_imb_max_adverse_pips
            if cfg_tick_imb_max_adverse_pips > 0.0
            else max(0.0, float(self.tick_imb_max_adverse_pips or 0.0))
        )
        if is_tick_imb and tick_imb_max_adverse_pips <= 0.0:
            thesis_sl_hint = _pick_float(thesis.get("sl_pips"), 0.0)
            mult = float(self.tick_imb_max_adverse_from_sl_mult or 0.0)
            if thesis_sl_hint > 0.0 and mult > 0.0:
                derived = thesis_sl_hint * mult
                lo = float(self.tick_imb_max_adverse_from_sl_min or 0.0)
                hi = float(self.tick_imb_max_adverse_from_sl_max or 0.0)
                if lo > 0.0:
                    derived = max(lo, derived)
                if hi > 0.0:
                    derived = min(hi, derived)
                tick_imb_max_adverse_pips = derived
        if is_tick_imb and tick_imb_max_adverse_pips > 0.0:
            cap = float(self.tick_imb_max_adverse_cap_pips or 0.0)
            if cap > 0.0:
                tick_imb_max_adverse_pips = min(tick_imb_max_adverse_pips, cap)

        client_ext = trade.get("clientExtensions")
        client_id = trade.get("client_order_id")
        if not client_id and isinstance(client_ext, dict):
            client_id = client_ext.get("id")
        if not client_id:
            LOG.warning("[exit-rangefader] missing client_id trade=%s skip close", trade_id)
            return
        # For TickImbalance, enforce max-adverse loss cap before any other pro-stop logic.
        # This is critical when hard SL is disabled; otherwise a structure-based stop can realize
        # much larger losses than the intended entry SL.
        if (
            policy_active
            and is_tick_imb
            and tick_imb_max_adverse_pips > 0.0
            and pnl <= -tick_imb_max_adverse_pips
        ):
            await self._close(trade_id, -units, "max_adverse", pnl, client_id, allow_negative=True)
            self._states.pop(trade_id, None)
            return
        if await maybe_close_pro_stop(trade, now=now):
            return

        if is_tick_imb and tick_imb_min_hold_sec > 0:
            min_hold_sec = max(min_hold_sec, tick_imb_min_hold_sec)
        if hold_sec < min_hold_sec:
            return
        candle_reason = _exit_candle_reversal("long" if units > 0 else "short")
        if candle_reason and pnl >= 0:
            candle_client_id = trade.get("client_order_id")
            if not candle_client_id:
                client_ext = trade.get("clientExtensions")
                if isinstance(client_ext, dict):
                    candle_client_id = client_ext.get("id")
            if candle_client_id:
                await self._close(trade_id, -units, candle_reason, pnl, candle_client_id)
                if hasattr(self, "_states"):
                    self._states.pop(trade_id, None)
                return
        if is_tick_imb:
            state = self._states.get(trade_id)
            if state is None:
                state = _TradeState(peak=pnl)
                self._states[trade_id] = state
            # When hard SL is disabled, TickImbalance still needs a deterministic break-even stop
            # after partial take. In live this is typically enforced via set_trade_protections();
            # in replay/offline modes we enforce it internally using bid/ask.
            if state.partial_done and state.be_floor_price is not None:
                bid, ask = _latest_bid_ask()
                if side == "long" and bid is not None and bid <= state.be_floor_price:
                    await self._close(trade_id, -units, "be_stop", pnl, client_id, allow_negative=True)
                    self._states.pop(trade_id, None)
                    return
                if side == "short" and ask is not None and ask >= state.be_floor_price:
                    await self._close(trade_id, -units, "be_stop", pnl, client_id, allow_negative=True)
                    self._states.pop(trade_id, None)
                    return
            if (
                tick_imb_partial_enabled
                and pnl > 0
                and not state.partial_done
                and pnl >= tick_imb_partial_trigger
            ):
                reduce_units = int(abs(units) * tick_imb_partial_fraction)
                remaining = abs(units) - reduce_units
                if reduce_units >= tick_imb_partial_min_units and remaining >= tick_imb_partial_min_remaining:
                    ok = await close_trade(
                        trade_id,
                        reduce_units,
                        client_order_id=client_id,
                        allow_negative=False,
                        exit_reason="partial_take",
                        env_prefix=_BB_ENV_PREFIX,
                    )
                    if ok:
                        state.partial_done = True
                        log_metric(
                            "scalp_level_reject_tick_imb_partial_take",
                            pnl,
                            tags={"side": side},
                            ts=now,
                        )
                        be = entry + (tick_imb_be_buffer_pips * 0.01) if side == "long" else entry - (
                            tick_imb_be_buffer_pips * 0.01
                        )
                        # Always arm the internal BE stop, even if broker protections are disabled/ignored.
                        state.be_floor_price = round(be, 3)
                        be_ok = await set_trade_protections(trade_id, sl_price=round(be, 3), tp_price=None)
                        if be_ok:
                            state.be_moved = True
                        return
            if (
                policy_active
                and tick_imb_max_adverse_pips > 0
                and pnl <= -tick_imb_max_adverse_pips
            ):
                await self._close(trade_id, -units, "max_adverse", pnl, client_id, allow_negative=True)
                self._states.pop(trade_id, None)
                return
            if (
                policy_active
                and tick_imb_max_hold_sec > 0
                and hold_sec >= tick_imb_max_hold_sec
                and pnl <= 0
            ):
                await self._close(trade_id, -units, "time_stop", pnl, client_id, allow_negative=True)
                self._states.pop(trade_id, None)
                return
        if pnl <= 0:
            if not policy_active:
                self._maybe_log_rollout_skip(
                    trade_id=trade_id,
                    side=side,
                    pnl=float(pnl),
                    hold_sec=hold_sec,
                    reason="negative_exit",
                )
                return
            rsi = _bb_float(fac_m1.get("rsi"))
            adx = _bb_float(fac_m1.get("adx"))
            atr_pips = _bb_float(fac_m1.get("atr_pips"))
            bbw = _bb_float(fac_m1.get("bbw"))
            vwap_gap = _bb_float(fac_m1.get("vwap_gap"))
            ma10 = _bb_float(fac_m1.get("ma10"))
            ma20 = _bb_float(fac_m1.get("ma20"))
            inventory_stress_reason, inventory_stress_trigger = _inventory_stress_exit_trigger(
                exit_profile=exit_profile,
                thesis=thesis,
                pnl=float(pnl),
                hold_sec=float(hold_sec),
                atr_pips=atr_pips,
            )
            if inventory_stress_reason:
                now_mono = time.monotonic()
                inventory_stress_cooldown_sec = max(
                    0.0,
                    _pick_float(
                        inventory_stress_profile.get("cooldown_sec"),
                        loss_cut_cooldown_sec,
                    ),
                )
                last = self._loss_cut_last_ts.get(trade_id, 0.0)
                if inventory_stress_cooldown_sec <= 0.0 or (now_mono - last) >= inventory_stress_cooldown_sec:
                    self._loss_cut_last_ts[trade_id] = now_mono
                    log_metric(
                        "scalp_level_reject_inventory_stress_exit",
                        abs(float(pnl)),
                        tags={
                            "strategy": base_tag or "unknown",
                            "reason": inventory_stress_reason,
                            "trigger": inventory_stress_trigger or "loss",
                            "side": side,
                        },
                        ts=now,
                    )
                    await self._close(
                        trade_id,
                        -units,
                        inventory_stress_reason,
                        pnl,
                        client_id,
                        allow_negative=True,
                    )
                    return
            ma_pair = (ma10, ma20) if ma10 is not None and ma20 is not None else None
            reentry_prefix = _reentry_prefix_for_tag(base_tag)
            reentry = None
            if _reentry_enabled_for_tag(reentry_prefix):
                reentry = decide_reentry(
                    prefix=reentry_prefix or "RANGEFADER",
                    side=side,
                    pnl_pips=pnl,
                    rsi=rsi,
                    adx=adx,
                    atr_pips=atr_pips,
                    bbw=bbw,
                    vwap_gap=vwap_gap,
                    ma_pair=ma_pair,
                    range_active=range_active,
                    log_tags={"trade": trade_id},
                )
            if reentry and reentry.action == "hold":
                return
            if reentry and reentry.action == "exit_reentry" and not reentry.shadow:
                await self._close(trade_id, -units, "reentry_reset", pnl, client_id)
                self._states.pop(trade_id, None)
                return

            eff_loss_cut_soft_pips = max(0.0, float(loss_cut_soft_pips))
            eff_loss_cut_hard_pips = max(
                eff_loss_cut_soft_pips, float(loss_cut_hard_pips)
            )
            if (
                atr_pips is not None
                and atr_pips > 0.0
                and loss_cut_hard_atr_mult > 0.0
            ):
                eff_loss_cut_hard_pips = max(
                    eff_loss_cut_hard_pips, atr_pips * loss_cut_hard_atr_mult
                )
            if (
                loss_cut_reversion_enabled
                and loss_cut_reversion_hard_pips > 0.0
                and vwap_gap is not None
            ):
                signed_gap = vwap_gap if side == "short" else -vwap_gap
                if signed_gap >= loss_cut_reversion_vwap_gap_min:
                    adx_ok = (
                        loss_cut_reversion_adx_max <= 0.0
                        or (adx is not None and adx <= loss_cut_reversion_adx_max)
                    )
                    if side == "short":
                        rsi_ok = (
                            rsi is not None and rsi >= loss_cut_reversion_rsi_short_min
                        )
                    else:
                        rsi_ok = (
                            rsi is not None and rsi <= loss_cut_reversion_rsi_long_max
                        )
                    if adx_ok and rsi_ok:
                        eff_loss_cut_hard_pips = max(
                            eff_loss_cut_hard_pips, loss_cut_reversion_hard_pips
                        )
            if loss_cut_hard_cap_pips > 0.0:
                eff_loss_cut_hard_pips = min(
                    eff_loss_cut_hard_pips, loss_cut_hard_cap_pips
                )
            eff_loss_cut_hard_pips = max(
                eff_loss_cut_hard_pips, eff_loss_cut_soft_pips
            )

            if not self._loss_cut_eligible(trade, enabled=loss_cut_enabled, require_sl=loss_cut_require_sl):
                return
            now_mono = time.monotonic()
            last = self._loss_cut_last_ts.get(trade_id, 0.0)
            if loss_cut_cooldown_sec > 0 and (now_mono - last) < loss_cut_cooldown_sec:
                return

            adverse_pips = abs(float(pnl))
            reason = None
            if loss_cut_max_hold_sec > 0 and hold_sec >= loss_cut_max_hold_sec:
                reason = loss_cut_reason_time
            elif eff_loss_cut_hard_pips > 0 and adverse_pips >= eff_loss_cut_hard_pips:
                reason = loss_cut_reason_hard
            elif eff_loss_cut_soft_pips > 0 and adverse_pips >= eff_loss_cut_soft_pips:
                reason = loss_cut_reason_soft
            if not reason:
                return
            self._loss_cut_last_ts[trade_id] = now_mono
            await self._close(trade_id, -units, reason, pnl, client_id, allow_negative=True)
            return

        lock_buffer = range_lock_buffer if range_active else lock_buffer
        profit_take = range_profit_take if range_active else profit_take
        trail_start = range_trail_start if range_active else trail_start
        trail_backoff = range_trail_backoff if range_active else trail_backoff
        if is_tick_imb:
            if tick_imb_lock_buffer > 0:
                lock_buffer = tick_imb_lock_buffer
            if tick_imb_profit_take > 0:
                profit_take = tick_imb_profit_take
            if tick_imb_trail_start > 0:
                trail_start = tick_imb_trail_start
            if tick_imb_trail_backoff > 0:
                trail_backoff = tick_imb_trail_backoff

        profit_take, trail_start, trail_backoff, lock_buffer = apply_exit_forecast_to_targets(
            profit_take=profit_take,
            trail_start=trail_start,
            trail_backoff=trail_backoff,
            lock_buffer=lock_buffer,
            adjustment=forecast_adj,
            profit_take_floor=0.5,
            trail_start_floor=0.5,
            trail_backoff_floor=0.05,
            lock_buffer_floor=0.05,
        )

        state = self._states.get(trade_id)
        if state is None:
            state = _TradeState(peak=pnl)
            self._states[trade_id] = state
        state.update(pnl, lock_buffer)

        if pnl >= trail_start:
            candidate = max(0.0, pnl - trail_backoff)
            state.lock_floor = candidate if state.lock_floor is None else max(state.lock_floor, candidate)

        await _maybe_move_profit_protection(
            trade=trade,
            trade_id=trade_id,
            strategy_tag=base_tag or str(strategy_tag or ""),
            state=state,
            side=side,
            entry=entry,
            pnl=float(pnl),
            mid=mid,
            profit_take=float(profit_take),
            now=now,
            thesis=thesis,
            fac_m1=fac_m1,
            range_active=range_active,
        )

        if pnl >= profit_take:
            await self._close(trade_id, -units, "take_profit", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        if (
            state.lock_floor is not None
            and hold_sec >= lock_floor_min_hold_sec
            and pnl <= state.lock_floor
        ):
            await self._close(trade_id, -units, "lock_floor", pnl, client_id)
            self._states.pop(trade_id, None)
            return

        if range_active and hold_sec >= range_max_hold_sec:
            await self._close(trade_id, -units, "range_timeout", pnl, client_id)
            self._states.pop(trade_id, None)

    async def run(self) -> None:
        LOG.info(
            "[exit-rangefader] exit worker start interval=%.2fs idle=%.2fs tags=%s pocket=%s policy_start_ts=%.3f",
            self.loop_interval,
            self.idle_interval,
            ",".join(sorted(ALLOWED_TAGS)) if ALLOWED_TAGS else "none",
            POCKET,
            float(self.exit_policy_start_ts or 0.0),
        )
        if not ALLOWED_TAGS:
            LOG.info("[exit-rangefader] no allowed tags configured; idle")
            try:
                while True:
                    await asyncio.sleep(3600.0)
            except asyncio.CancelledError:
                return
        try:
            had_trades = False
            while True:
                await asyncio.sleep(self.loop_interval if had_trades else self.idle_interval)
                positions = self._pos_manager.get_open_positions()
                pocket_info = positions.get(POCKET) or {}
                trades = _filter_trades(pocket_info.get("open_trades") or [], ALLOWED_TAGS)
                had_trades = bool(trades)
                active_ids = {str(tr.get("trade_id")) for tr in trades if tr.get("trade_id")}
                for tid in list(self._states.keys()):
                    if tid not in active_ids:
                        self._states.pop(tid, None)
                for tid in list(self._loss_cut_last_ts.keys()):
                    if tid not in active_ids:
                        self._loss_cut_last_ts.pop(tid, None)
                for tid in list(self._rollout_skip_log_ts.keys()):
                    if tid not in active_ids:
                        self._rollout_skip_log_ts.pop(tid, None)
                if not trades:
                    continue

                mid, range_active = self._context()
                now = datetime.now(timezone.utc)
                for tr in trades:
                    try:
                        await self._review_trade(tr, now, mid, range_active)
                    except Exception:
                        LOG.exception("[exit-rangefader] review failed trade=%s", tr.get("trade_id"))
        except asyncio.CancelledError:
            LOG.info("[exit-rangefader] worker cancelled")
            raise
        finally:
            try:
                self._pos_manager.close()
            except Exception:
                LOG.exception("[exit-rangefader] failed to close PositionManager")


async def scalp_level_reject_exit_worker() -> None:
    worker = RangeFaderExitWorker()
    await worker.run()


_CANDLE_PIP = 0.01
_CANDLE_EXIT_MIN_CONF = 0.35
_CANDLE_EXIT_SCORE = -0.5
_CANDLE_WORKER_NAME = (__file__.replace("\\", "/").split("/")[-2] if "/" in __file__ else "").lower()


def _candle_tf_for_worker() -> str:
    name = _CANDLE_WORKER_NAME
    if "macro" in name or "trend_h1" in name or "manual" in name:
        return "H1"
    if "scalp" in name or "s5" in name or "fast" in name:
        return "M1"
    return "M5"


def _extract_candles(raw):
    candles = []
    for candle in raw or []:
        try:
            o = float(candle.get("open", candle.get("o")))
            h = float(candle.get("high", candle.get("h")))
            l = float(candle.get("low", candle.get("l")))
            c = float(candle.get("close", candle.get("c")))
        except Exception:
            continue
        if h <= 0 or l <= 0:
            continue
        candles.append((o, h, l, c))
    return candles


def _detect_candlestick_pattern(candles):
    if len(candles) < 2:
        return None
    o0, h0, l0, c0 = candles[-2]
    o1, h1, l1, c1 = candles[-1]
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    range1 = max(h1 - l1, _CANDLE_PIP * 0.1)
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


def _score_candle(*, candles, side, min_conf):
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
    score = max(-1.0, min(1.0, score))
    return score, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": bias}


def _exit_candle_reversal(side):
    tf = _candle_tf_for_worker()
    candles = (all_factors().get(tf) or {}).get("candles") or []
    if not candles:
        return None
    score, detail = _score_candle(candles=candles, side=side, min_conf=_CANDLE_EXIT_MIN_CONF)
    if score is None:
        return None
    if score <= _CANDLE_EXIT_SCORE:
        detail_type = detail.get("type") if isinstance(detail, dict) else None
        return f"candle_{detail_type}" if detail_type else "candle_reversal"
    return None
if __name__ == "__main__":
    asyncio.run(scalp_level_reject_exit_worker())

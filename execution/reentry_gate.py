"""Strategy-level re-entry guard using last closed trade metadata."""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None

_DB_PATH = Path("logs/stage_state.db")
_CONFIG_PATH = Path(os.getenv("REENTRY_CONFIG_PATH", "config/worker_reentry.yaml"))
_CONFIG_TTL_SEC = float(os.getenv("REENTRY_CONFIG_TTL_SEC", "10"))
_ENABLED = os.getenv("REENTRY_GATE_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DEFAULTS = {
    "cooldown_win_sec": 60,
    "cooldown_loss_sec": 180,
    "same_dir_reentry_pips": 1.8,
    "same_dir_mode": "return",
    "loss_dir_mode": "",
    "allow_jst_hours": [],
    "block_jst_hours": [],
    "return_wait_bias": "neutral",
    "max_open_trades": 0,
    "max_open_adverse_pips": 0.0,
    "max_open_avg_adverse_pips": 0.0,
    "max_open_trades_hard": 0,
    "stack_reentry_pips": 0.0,
}
_BIAS_COOLDOWN_SCALE = {
    "favor": float(os.getenv("REENTRY_BIAS_FAVOR_COOLDOWN_SCALE", "1.3")),
    "avoid": float(os.getenv("REENTRY_BIAS_AVOID_COOLDOWN_SCALE", "0.8")),
}
_BIAS_DISTANCE_SCALE = {
    "favor": float(os.getenv("REENTRY_BIAS_FAVOR_DISTANCE_SCALE", "1.25")),
    "avoid": float(os.getenv("REENTRY_BIAS_AVOID_DISTANCE_SCALE", "0.85")),
}
_BIAS_MIN_COOLDOWN_SEC = int(os.getenv("REENTRY_BIAS_MIN_COOLDOWN_SEC", "15"))
_ALIAS_BASE = {
    "mlr": "MicroLevelReactor",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "h1momentum": "H1Momentum",
    "m1scalper": "M1Scalper",
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
    "techfusion": "TechFusion",
    "macrotechfusion": "MacroTechFusion",
    "micropullbackfib": "MicroPullbackFib",
    "scalpreversalnwave": "ScalpReversalNWave",
    "rangecompressionbreak": "RangeCompressionBreak",
}
_CACHE: Dict[str, object] = {"ts": 0.0, "data": None}


@dataclass(slots=True)
class ReentryState:
    close_time: datetime
    close_price: Optional[float]
    result: str
    pl_pips: Optional[float]


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = value.strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        if "." in raw and "+" in raw and raw.rfind("+") > raw.find("."):
            head, frac_plus = raw.split(".", 1)
            frac, plus = frac_plus.split("+", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())[:6]
            raw = f"{head}.{frac_digits}+{plus}"
        elif "." in raw and "+" not in raw:
            head, frac = raw.split(".", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())[:6]
            raw = f"{head}.{frac_digits}+00:00"
        elif "+" not in raw:
            raw = f"{raw}+00:00"
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except Exception:
        try:
            trimmed = raw.split(".", 1)[0].rstrip("Z") + "+00:00"
            return datetime.fromisoformat(trimmed).astimezone(timezone.utc)
        except Exception:
            return None


def _base_strategy_tag(tag: Optional[str]) -> str:
    if not tag:
        return ""
    text = str(tag).strip()
    if not text:
        return ""
    base = text.split("-", 1)[0].strip()
    if not base:
        return text
    alias = _ALIAS_BASE.get(base.lower())
    return alias or base


def _load_config() -> Dict[str, Any]:
    if not _ENABLED:
        return {"defaults": dict(_DEFAULTS), "strategies": {}}
    now = time.monotonic()
    cached_ts = float(_CACHE.get("ts") or 0.0)
    if now - cached_ts < _CONFIG_TTL_SEC and isinstance(_CACHE.get("data"), dict):
        return _CACHE["data"]  # type: ignore[return-value]
    data: Dict[str, Any] = {"defaults": dict(_DEFAULTS), "strategies": {}}
    if yaml is None or not _CONFIG_PATH.exists():
        _CACHE["ts"] = now
        _CACHE["data"] = data
        return data
    try:
        payload = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        defaults = payload.get("defaults")
        if isinstance(defaults, dict):
            merged = dict(_DEFAULTS)
            merged.update(defaults)
            data["defaults"] = merged
        strategies = payload.get("strategies")
        if isinstance(strategies, dict):
            data["strategies"] = strategies
    _CACHE["ts"] = now
    _CACHE["data"] = data
    return data


def _merge_config(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, value in override.items():
        merged[key] = value
    return merged


def _coerce_hours(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    hours: list[int] = []
    for item in items:
        try:
            hour = int(item) % 24
        except Exception:
            continue
        hours.append(hour)
    return sorted(set(hours))


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _extract_open_stats(
    open_positions: dict,
    strategy: str,
    direction: str,
) -> Tuple[int, Optional[float], Optional[float], Optional[float]]:
    if not isinstance(open_positions, dict):
        return 0, None, None, None
    open_count = 0
    sum_pips = 0.0
    worst_pips: Optional[float] = None
    sum_units = 0.0
    sum_price = 0.0
    for pocket, info in open_positions.items():
        if pocket and str(pocket).startswith("__"):
            continue
        if not isinstance(info, dict):
            continue
        trades = info.get("open_trades") or []
        if not isinstance(trades, list):
            continue
        for tr in trades:
            if not isinstance(tr, dict):
                continue
            tag = tr.get("strategy_tag")
            if not tag:
                thesis = tr.get("entry_thesis")
                if isinstance(thesis, dict):
                    tag = thesis.get("strategy_tag") or thesis.get("strategy") or thesis.get("tag")
            base = _base_strategy_tag(tag)
            if not base or base != strategy:
                continue
            units = tr.get("units") or 0
            try:
                units = int(units)
            except Exception:
                units = 0
            if units == 0:
                continue
            if direction == "long" and units <= 0:
                continue
            if direction == "short" and units >= 0:
                continue
            pips_val = tr.get("unrealized_pl_pips")
            if pips_val is None:
                try:
                    pl_val = float(tr.get("unrealized_pl") or 0.0)
                    abs_units = abs(units)
                    pips_val = pl_val / (abs_units * 0.01) if abs_units else 0.0
                except Exception:
                    pips_val = 0.0
            try:
                pips = float(pips_val)
            except Exception:
                pips = 0.0
            entry_price = tr.get("price")
            try:
                entry_price = float(entry_price) if entry_price is not None else None
            except Exception:
                entry_price = None
            open_count += 1
            sum_pips += pips
            if worst_pips is None or pips < worst_pips:
                worst_pips = pips
            if entry_price is not None:
                weight = abs(units)
                if weight > 0:
                    sum_units += weight
                    sum_price += entry_price * weight
    if open_count <= 0:
        return 0, None, None, None
    avg_entry = None
    if sum_units > 0:
        avg_entry = sum_price / sum_units
    return open_count, worst_pips, sum_pips / open_count, avg_entry


def needs_open_positions(strategy_tag: Optional[str]) -> bool:
    if not _ENABLED:
        return False
    base = _base_strategy_tag(strategy_tag)
    if not base:
        return False
    cfg = _load_config()
    defaults = cfg.get("defaults") or {}
    overrides = (cfg.get("strategies") or {}).get(base) or {}
    if not isinstance(overrides, dict):
        overrides = {}
    merged = _merge_config(defaults, overrides)
    for key in (
        "max_open_trades",
        "max_open_trades_hard",
        "max_open_adverse_pips",
        "max_open_avg_adverse_pips",
        "stack_reentry_pips",
    ):
        try:
            if float(merged.get(key) or 0.0) > 0.0:
                return True
        except Exception:
            continue
    return False


def _get_state(strategy: str, direction: str) -> Optional[ReentryState]:
    if not _DB_PATH.exists():
        return None
    try:
        con = sqlite3.connect(_DB_PATH)
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT last_close_time, last_close_price, last_result, last_pl_pips
            FROM strategy_reentry_state
            WHERE strategy=? AND direction=?
            """,
            (strategy, direction),
        ).fetchone()
        con.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    close_time = _parse_iso(row["last_close_time"])
    if close_time is None:
        return None
    close_price = None
    try:
        if row["last_close_price"] is not None:
            close_price = float(row["last_close_price"])
    except Exception:
        close_price = None
    try:
        pl_pips = float(row["last_pl_pips"]) if row["last_pl_pips"] is not None else None
    except Exception:
        pl_pips = None
    result = str(row["last_result"] or "flat")
    return ReentryState(
        close_time=close_time,
        close_price=close_price,
        result=result,
        pl_pips=pl_pips,
    )


def allow_entry(
    *,
    strategy_tag: Optional[str],
    units: int,
    price: Optional[float],
    open_positions: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> Tuple[bool, str, Dict[str, object]]:
    if not _ENABLED:
        return True, "disabled", {}
    if not strategy_tag or units == 0:
        return True, "no_strategy", {}
    base = _base_strategy_tag(strategy_tag)
    if not base:
        return True, "no_strategy", {}
    cfg = _load_config()
    defaults = cfg.get("defaults") or {}
    overrides = (cfg.get("strategies") or {}).get(base) or {}
    if not isinstance(overrides, dict):
        overrides = {}
    merged = _merge_config(defaults, overrides)
    if merged.get("enabled") is False:
        return True, "disabled", {}
    bias = str(merged.get("return_wait_bias") or "neutral").strip().lower()
    cooldown_win = int(merged.get("cooldown_win_sec") or 0)
    cooldown_loss = int(merged.get("cooldown_loss_sec") or 0)
    min_pips = float(merged.get("same_dir_reentry_pips") or 0.0)
    cd_scale = _BIAS_COOLDOWN_SCALE.get(bias)
    if cd_scale:
        cooldown_win = max(_BIAS_MIN_COOLDOWN_SEC, int(round(cooldown_win * cd_scale)))
        cooldown_loss = max(_BIAS_MIN_COOLDOWN_SEC, int(round(cooldown_loss * cd_scale)))
    dist_scale = _BIAS_DISTANCE_SCALE.get(bias)
    if dist_scale:
        min_pips = max(0.0, round(min_pips * dist_scale, 3))
    direction = "long" if units > 0 else "short"
    now_dt = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    allow_hours = _coerce_hours(merged.get("allow_jst_hours"))
    block_hours = _coerce_hours(merged.get("block_jst_hours"))
    if allow_hours:
        jst_hour = (now_dt + timedelta(hours=9)).hour
        if jst_hour not in allow_hours:
            return False, "time_filter", {"jst_hour": jst_hour, "allow_jst_hours": allow_hours}
    elif block_hours:
        jst_hour = (now_dt + timedelta(hours=9)).hour
        if jst_hour in block_hours:
            return False, "time_block", {"jst_hour": jst_hour, "block_jst_hours": block_hours}

    max_open_trades = int(merged.get("max_open_trades") or 0)
    max_open_trades_hard = int(merged.get("max_open_trades_hard") or 0)
    max_open_adverse = float(merged.get("max_open_adverse_pips") or 0.0)
    max_open_avg = float(merged.get("max_open_avg_adverse_pips") or 0.0)
    stack_reentry_pips = float(merged.get("stack_reentry_pips") or 0.0)
    if open_positions and (
        max_open_trades > 0
        or max_open_trades_hard > 0
        or max_open_adverse > 0.0
        or max_open_avg > 0.0
        or stack_reentry_pips > 0.0
    ):
        open_count, worst_pips, avg_pips, avg_entry = _extract_open_stats(
            open_positions, base, direction
        )
        distance_pips = None
        distance_override = False
        if stack_reentry_pips > 0.0 and price is not None and avg_entry is not None:
            try:
                if direction == "long":
                    distance_pips = (avg_entry - float(price)) / 0.01
                else:
                    distance_pips = (float(price) - avg_entry) / 0.01
            except Exception:
                distance_pips = None
            if distance_pips is not None and distance_pips >= stack_reentry_pips:
                distance_override = True
        hard_cap = max_open_trades_hard if max_open_trades_hard > 0 else max_open_trades
        if hard_cap > 0 and open_count >= hard_cap:
            return False, "open_hard_cap", {
                "open_count": open_count,
                "max_open_trades_hard": hard_cap,
                "worst_unrealized_pips": worst_pips,
                "avg_unrealized_pips": avg_pips,
            }
        soft_reason = None
        soft_details: Dict[str, object] = {}
        if max_open_trades > 0 and open_count >= max_open_trades:
            soft_reason = "open_stack"
            soft_details = {
                "open_count": open_count,
                "max_open_trades": max_open_trades,
                "worst_unrealized_pips": worst_pips,
                "avg_unrealized_pips": avg_pips,
            }
        elif max_open_adverse > 0.0 and worst_pips is not None:
            threshold = -abs(max_open_adverse)
            if worst_pips <= threshold:
                soft_reason = "open_adverse"
                soft_details = {
                    "open_count": open_count,
                    "worst_unrealized_pips": worst_pips,
                    "max_open_adverse_pips": max_open_adverse,
                }
        elif max_open_avg > 0.0 and avg_pips is not None:
            threshold = -abs(max_open_avg)
            if avg_pips <= threshold:
                soft_reason = "open_avg_adverse"
                soft_details = {
                    "open_count": open_count,
                    "avg_unrealized_pips": avg_pips,
                    "max_open_avg_adverse_pips": max_open_avg,
                }
        if soft_reason:
            if not distance_override:
                return False, soft_reason, soft_details
            override_details = dict(soft_details)
            override_details.update(
                {
                    "stack_override": True,
                    "stack_block_reason": soft_reason,
                    "stack_distance_pips": round(distance_pips or 0.0, 2),
                    "stack_reentry_pips": stack_reentry_pips,
                }
            )
            return True, "stack_override", override_details

    state = _get_state(base, direction)
    if not state:
        return True, "no_state", {}

    distance_ok: Optional[bool] = None
    distance_details: Dict[str, object] = {}
    distance_mode = ""
    if min_pips > 0.0 and price is not None and state.close_price is not None:
        threshold = min_pips * 0.01
        mode = str(merged.get("same_dir_mode") or "return").strip().lower()
        if mode not in {"return", "follow", "both"}:
            mode = "return"
        if state.result == "loss":
            loss_mode = str(merged.get("loss_dir_mode") or "").strip().lower()
            if loss_mode in {"return", "follow", "both"}:
                mode = loss_mode
            elif mode == "return":
                # After a loss, allow either return or follow to avoid bottom-stop stalls.
                mode = "both"
        distance_mode = mode
        if direction == "long":
            allow_return = price <= state.close_price - threshold
            allow_follow = price >= state.close_price + threshold
        else:
            allow_return = price >= state.close_price + threshold
            allow_follow = price <= state.close_price - threshold
        if mode == "follow":
            allow = allow_follow
        elif mode == "both":
            allow = allow_return or allow_follow
        else:
            allow = allow_return
        distance_ok = allow
        if not allow:
            distance_details = {
                "price": float(price),
                "last_close_price": state.close_price,
                "same_dir_reentry_pips": min_pips,
                "same_dir_mode": mode,
                "return_wait_bias": bias,
            }

    cooldown = cooldown_win if state.result == "win" else cooldown_loss
    if cooldown > 0:
        elapsed = (now_dt - state.close_time).total_seconds()
        if elapsed < cooldown:
            remaining = int(max(1, cooldown - elapsed))
            if state.result == "loss" and distance_ok:
                allow_early = merged.get("loss_cooldown_allow_distance")
                if allow_early is None or _coerce_bool(allow_early):
                    details = {
                        "cooldown_remaining_sec": remaining,
                        "last_close_time": state.close_time.isoformat(),
                        "last_result": state.result,
                        "return_wait_bias": bias,
                        "same_dir_mode": distance_mode,
                    }
                    if distance_details:
                        details.update(distance_details)
                    return True, "cooldown_override_distance", details
            return False, "cooldown", {
                "cooldown_remaining_sec": remaining,
                "last_close_time": state.close_time.isoformat(),
                "last_result": state.result,
                "return_wait_bias": bias,
            }
    if distance_ok is False:
        return False, "price_distance", distance_details

    return True, "ok", {}

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
_ENABLED = os.getenv("REENTRY_GUARD_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DEFAULTS = {
    "cooldown_win_sec": 60,
    "cooldown_loss_sec": 180,
    "same_dir_reentry_pips": 1.8,
    "allow_jst_hours": [],
    "block_jst_hours": [],
    "return_wait_bias": "neutral",
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

    state = _get_state(base, direction)
    if not state:
        return True, "no_state", {}

    cooldown = cooldown_win if state.result == "win" else cooldown_loss
    if cooldown > 0:
        elapsed = (now_dt - state.close_time).total_seconds()
        if elapsed < cooldown:
            remaining = int(max(1, cooldown - elapsed))
            return False, "cooldown", {
                "cooldown_remaining_sec": remaining,
                "last_close_time": state.close_time.isoformat(),
                "last_result": state.result,
                "return_wait_bias": bias,
            }

    if min_pips > 0.0 and price is not None and state.close_price is not None:
        threshold = min_pips * 0.01
        if direction == "long":
            if price > state.close_price - threshold:
                return False, "price_distance", {
                    "price": float(price),
                    "last_close_price": state.close_price,
                    "same_dir_reentry_pips": min_pips,
                    "return_wait_bias": bias,
                }
        else:
            if price < state.close_price + threshold:
                return False, "price_distance", {
                    "price": float(price),
                    "last_close_price": state.close_price,
                    "same_dir_reentry_pips": min_pips,
                    "return_wait_bias": bias,
                }

    return True, "ok", {}

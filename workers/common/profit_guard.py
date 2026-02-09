"""
Profit giveback guard.
Blocks new entries when recent realized PnL gives back too much from the peak.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

from utils.env_utils import env_bool, env_float, env_get, env_int

LOG = logging.getLogger(__name__)

_DB = pathlib.Path("logs/trades.db")

@dataclass(frozen=True, slots=True)
class ProfitGuardCfg:
    env_prefix: Optional[str]

    enabled: bool
    mode: str
    lookback_min: int
    ttl_sec: float
    pockets_all: bool
    pockets: set[str]
    scope: str  # pocket|strategy

    min_peak_pips_default: float
    max_giveback_pips_default: float
    min_peak_jpy_default: float
    max_giveback_jpy_default: float


_CFG_CACHE: dict[str, ProfitGuardCfg] = {}
_cache: dict[str, tuple[float, "ProfitDecision"]] = {}


def _cfg_key(env_prefix: Optional[str]) -> str:
    return str(env_prefix or "").strip().upper()


def _strategy_env_bool(name: str, default: bool, *, prefix: Optional[str]) -> bool:
    return env_bool(
        name,
        default,
        prefix=prefix,
        allow_global_fallback=False,
    )


def _strategy_env_float(name: str, default: float, *, prefix: Optional[str]) -> float:
    return env_float(
        name,
        default,
        prefix=prefix,
        allow_global_fallback=False,
    )


def _strategy_env_get(name: str, default: Optional[str], *, prefix: Optional[str]) -> Optional[str]:
    return env_get(
        name,
        default,
        prefix=prefix,
        allow_global_fallback=False,
    )


def _get_cfg(env_prefix: Optional[str]) -> ProfitGuardCfg:
    key = _cfg_key(env_prefix)
    cached = _CFG_CACHE.get(key)
    if cached is not None:
        return cached

    prefix_norm = key or None

    enabled = _strategy_env_bool("PROFIT_GUARD_ENABLED", True, prefix=prefix_norm)
    mode = str(_strategy_env_get("PROFIT_GUARD_MODE", "block", prefix=prefix_norm) or "block").strip().lower()
    lookback_min = max(
        10,
        env_int(
            "PROFIT_GUARD_LOOKBACK_MIN",
            180,
            prefix=prefix_norm,
            allow_global_fallback=False,
        ),
    )
    ttl_sec = max(10.0, _strategy_env_float("PROFIT_GUARD_TTL_SEC", 30.0, prefix=prefix_norm))

    scope = str(_strategy_env_get("PROFIT_GUARD_SCOPE", "pocket", prefix=prefix_norm) or "pocket").strip().lower()
    if scope not in {"pocket", "strategy"}:
        scope = "pocket"

    raw_pockets = str(_strategy_env_get("PROFIT_GUARD_POCKETS", "scalp", prefix=prefix_norm) or "scalp").strip().lower()
    pockets_all = raw_pockets in {"*", "all"}
    pockets = set() if pockets_all else {item.strip() for item in raw_pockets.split(",") if item.strip()}

    cfg = ProfitGuardCfg(
        env_prefix=key or None,
        enabled=enabled,
        mode=mode,
        lookback_min=lookback_min,
        ttl_sec=ttl_sec,
        pockets_all=pockets_all,
        pockets=pockets,
        scope=scope,
        min_peak_pips_default=_strategy_env_float("PROFIT_GUARD_MIN_PEAK_PIPS", 6.0, prefix=prefix_norm),
        max_giveback_pips_default=_strategy_env_float("PROFIT_GUARD_MAX_GIVEBACK_PIPS", 3.0, prefix=prefix_norm),
        min_peak_jpy_default=_strategy_env_float("PROFIT_GUARD_MIN_PEAK_JPY", 0.0, prefix=prefix_norm),
        max_giveback_jpy_default=_strategy_env_float("PROFIT_GUARD_MAX_GIVEBACK_JPY", 0.0, prefix=prefix_norm),
    )
    _CFG_CACHE[key] = cfg
    return cfg


def _env_raw(key: str) -> Optional[str]:
    raw = os.getenv(key)
    if raw is None:
        return None
    if str(raw).strip() == "":
        return None
    return raw


def _lookup_float(keys: list[str], default: float) -> float:
    for key in keys:
        raw = _env_raw(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return default


def _threshold(name: str, pocket: str, default: float, *, cfg: ProfitGuardCfg) -> float:
    pocket_key = (pocket or "").strip().upper()
    keys: list[str] = []
    if cfg.env_prefix:
        if pocket_key:
            keys.append(f"{cfg.env_prefix}_UNIT_{name}_{pocket_key}")
            keys.append(f"{cfg.env_prefix}_{name}_{pocket_key}")
        keys.append(f"{cfg.env_prefix}_UNIT_{name}")
        keys.append(f"{cfg.env_prefix}_{name}")
    else:
        if pocket_key:
            keys.append(f"{name}_{pocket_key}")
        keys.append(name)
    return _lookup_float(keys, default)


def _should_apply(pocket: str, *, cfg: ProfitGuardCfg) -> bool:
    if not pocket:
        return False
    if cfg.pockets_all:
        return True
    if not cfg.pockets:
        return False
    return pocket.strip().lower() in cfg.pockets


@dataclass(frozen=True, slots=True)
class ProfitDecision:
    allowed: bool
    reason: str
    peak_pips: float = 0.0
    current_pips: float = 0.0
    peak_jpy: float = 0.0
    current_jpy: float = 0.0


def _query_guard(pocket: str, *, strategy_tag: Optional[str], cfg: ProfitGuardCfg) -> ProfitDecision:
    if not _DB.exists():
        return ProfitDecision(True, "no_db")
    if cfg.lookback_min <= 0:
        return ProfitDecision(True, "lookback_disabled")

    try:
        con = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True, timeout=2.0)
        con.row_factory = sqlite3.Row
    except Exception:
        return ProfitDecision(True, "db_open_failed")

    cutoff = f"-{cfg.lookback_min} minutes"
    try:
        params = [pocket, cutoff]
        sql = """
            SELECT id, pl_pips, realized_pl
            FROM trades
            WHERE pocket = ?
              AND close_time IS NOT NULL
              AND datetime(close_time) >= datetime('now', ?)
        """
        if cfg.scope == "strategy" and strategy_tag:
            sql += " AND coalesce(strategy_tag, strategy) = ?\n"
            params.append(str(strategy_tag))
        sql += " ORDER BY datetime(close_time) ASC, id ASC\n"
        rows = con.execute(sql, tuple(params)).fetchall()
    except Exception as exc:
        LOG.debug("[PROFIT_GUARD] query failed pocket=%s err=%s", pocket, exc)
        return ProfitDecision(True, "query_failed")
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not rows:
        return ProfitDecision(True, "no_recent_trades")

    current_pips = 0.0
    current_jpy = 0.0
    peak_pips = 0.0
    peak_jpy = 0.0
    for row in rows:
        try:
            current_pips += float(row["pl_pips"] or 0.0)
            current_jpy += float(row["realized_pl"] or 0.0)
        except Exception:
            continue
        peak_pips = max(peak_pips, current_pips)
        peak_jpy = max(peak_jpy, current_jpy)

    min_peak_pips = _threshold(
        "PROFIT_GUARD_MIN_PEAK_PIPS", pocket, cfg.min_peak_pips_default, cfg=cfg
    )
    min_peak_jpy = _threshold("PROFIT_GUARD_MIN_PEAK_JPY", pocket, cfg.min_peak_jpy_default, cfg=cfg)
    max_giveback_pips = _threshold(
        "PROFIT_GUARD_MAX_GIVEBACK_PIPS", pocket, cfg.max_giveback_pips_default, cfg=cfg
    )
    max_giveback_jpy = _threshold(
        "PROFIT_GUARD_MAX_GIVEBACK_JPY", pocket, cfg.max_giveback_jpy_default, cfg=cfg
    )

    if min_peak_pips <= 0 and min_peak_jpy <= 0:
        return ProfitDecision(True, "min_peak_disabled", peak_pips, current_pips, peak_jpy, current_jpy)
    if max_giveback_pips <= 0 and max_giveback_jpy <= 0:
        return ProfitDecision(True, "giveback_disabled", peak_pips, current_pips, peak_jpy, current_jpy)

    if peak_pips < max(0.0, min_peak_pips) and peak_jpy < max(0.0, min_peak_jpy):
        return ProfitDecision(True, "peak_below_min", peak_pips, current_pips, peak_jpy, current_jpy)

    giveback_pips = max(0.0, peak_pips - current_pips)
    giveback_jpy = max(0.0, peak_jpy - current_jpy)
    hit_pips = max_giveback_pips > 0 and giveback_pips >= max_giveback_pips
    hit_jpy = max_giveback_jpy > 0 and giveback_jpy >= max_giveback_jpy
    if hit_pips or hit_jpy:
        if hit_pips and hit_jpy:
            reason = (
                f"giveback={giveback_pips:.2f}p/{giveback_jpy:.0f}jpy "
                f"peak={peak_pips:.2f}p/{peak_jpy:.0f}jpy "
                f"curr={current_pips:.2f}p/{current_jpy:.0f}jpy "
                f"win={cfg.lookback_min}m"
            )
        elif hit_pips:
            reason = (
                f"giveback={giveback_pips:.2f}p peak={peak_pips:.2f}p "
                f"curr={current_pips:.2f}p win={cfg.lookback_min}m"
            )
        else:
            reason = (
                f"giveback={giveback_jpy:.0f}jpy peak={peak_jpy:.0f}jpy "
                f"curr={current_jpy:.0f}jpy win={cfg.lookback_min}m"
            )
        return ProfitDecision(False, reason, peak_pips, current_pips, peak_jpy, current_jpy)

    return ProfitDecision(True, "ok", peak_pips, current_pips, peak_jpy, current_jpy)


def is_allowed(
    pocket: str,
    *,
    strategy_tag: Optional[str] = None,
    env_prefix: Optional[str] = None,
) -> ProfitDecision:
    """
    Evaluate whether new entries are allowed based on recent profit giveback.
    """
    cfg = _get_cfg(env_prefix)
    if not cfg.enabled or cfg.mode in {"off", "disabled", "false", "no"}:
        return ProfitDecision(True, "disabled")
    if not _should_apply(pocket, cfg=cfg):
        return ProfitDecision(True, "skip_pocket")

    pocket_key = pocket.strip().lower()
    strat_key = str(strategy_tag or "").strip()
    cache_key = pocket_key
    if cfg.env_prefix:
        cache_key = f"{cfg.env_prefix}:{cache_key}"
    if cfg.scope == "strategy" and strat_key:
        cache_key = f"{cache_key}:{strat_key}"

    now = time.monotonic()
    cached = _cache.get(cache_key)
    if cached and now - cached[0] <= cfg.ttl_sec:
        return cached[1]

    decision = _query_guard(pocket, strategy_tag=strategy_tag, cfg=cfg)
    if cfg.mode != "block":
        decision = ProfitDecision(
            True,
            f"warn:{decision.reason}",
            decision.peak_pips,
            decision.current_pips,
            decision.peak_jpy,
            decision.current_jpy,
        )
    _cache[cache_key] = (now, decision)
    return decision

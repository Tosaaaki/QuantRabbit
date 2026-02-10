"""
Lightweight performance guard for workers.
Checks recent PF / win-rate per strategy_tag+pocket (and optional hour) and blocks entries when stats are poor.
Designed to be cheap: uses SQLite with short TTL caching.
"""

from __future__ import annotations

import logging
import os
import re
import pathlib
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from workers.common.quality_gate import current_regime
from utils.env_utils import env_bool, env_float, env_get, env_int

LOG = logging.getLogger(__name__)

_DB = pathlib.Path("logs/trades.db")

@dataclass(frozen=True, slots=True)
class PerfGuardCfg:
    env_prefix: Optional[str]

    enabled: bool
    mode: str
    lookback_days: int
    min_trades: int
    pf_min_default: float
    win_min_default: float
    ttl_sec: float

    hourly_enabled: bool
    hourly_min_trades: int

    relax_tags: set[str]
    split_directional: bool

    regime_filter_enabled: bool
    regime_min_trades: int

    failfast_min_trades: int
    failfast_pf: float
    failfast_win: float

    sl_loss_rate_min_trades: int
    sl_loss_rate_max_default: float

    pocket_enabled: bool
    pocket_lookback_days: int
    pocket_min_trades: int
    pocket_pf_min_default: float
    pocket_win_min_default: float
    pocket_ttl_sec: float

    scale_enabled: bool
    scale_lookback_days: int
    scale_min_trades: int
    scale_pf_min: float
    scale_win_min: float
    scale_avg_pips_min: float
    scale_step: float
    scale_max_mult: float
    scale_ttl_sec: float


_CFG_CACHE: dict[str, PerfGuardCfg] = {}


def _cfg_key(env_prefix: Optional[str]) -> str:
    return str(env_prefix or "").strip().upper()


def _strategy_env_get(name: str, default: Optional[str], *, env_prefix: Optional[str]) -> Optional[str]:
    return env_get(
        name,
        default,
        prefix=env_prefix,
        allow_global_fallback=False,
    )


def _strategy_env_bool(name: str, default: bool, *, env_prefix: Optional[str]) -> bool:
    return env_bool(
        name,
        default,
        prefix=env_prefix,
        allow_global_fallback=False,
    )


def _strategy_env_int(name: str, default: int, *, env_prefix: Optional[str]) -> int:
    return env_int(
        name,
        default,
        prefix=env_prefix,
        allow_global_fallback=False,
    )


def _strategy_env_float(name: str, default: float, *, env_prefix: Optional[str]) -> float:
    return env_float(
        name,
        default,
        prefix=env_prefix,
        allow_global_fallback=False,
    )


def _get_cfg(env_prefix: Optional[str]) -> PerfGuardCfg:
    key = _cfg_key(env_prefix)
    cached = _CFG_CACHE.get(key)
    if cached is not None:
        return cached

    raw_enabled = _strategy_env_get("PERF_GUARD_ENABLED", None, env_prefix=env_prefix)
    if raw_enabled is None:
        raw_enabled = _strategy_env_get("PERF_GUARD_GLOBAL_ENABLED", "1", env_prefix=env_prefix)
    enabled = str(raw_enabled).strip().lower() not in {"", "0", "false", "no", "off"}

    mode = str(_strategy_env_get("PERF_GUARD_MODE", "block", env_prefix=env_prefix) or "block").strip().lower()
    lookback_days = max(1, _strategy_env_int("PERF_GUARD_LOOKBACK_DAYS", 3, env_prefix=env_prefix))
    min_trades = max(5, _strategy_env_int("PERF_GUARD_MIN_TRADES", 12, env_prefix=env_prefix))
    pf_min_default = float(_strategy_env_float("PERF_GUARD_PF_MIN", 0.9, env_prefix=env_prefix) or 0.9)
    win_min_default = float(_strategy_env_float("PERF_GUARD_WIN_MIN", 0.48, env_prefix=env_prefix) or 0.48)
    ttl_sec = max(30.0, float(_strategy_env_float("PERF_GUARD_TTL_SEC", 120.0, env_prefix=env_prefix) or 120.0))

    hourly_enabled = _strategy_env_bool("PERF_GUARD_HOURLY", False, env_prefix=env_prefix)
    hourly_min_trades = max(5, _strategy_env_int("PERF_GUARD_HOURLY_MIN_TRADES", 12, env_prefix=env_prefix))

    relax_raw = _strategy_env_get("PERF_GUARD_RELAX_TAGS", None, env_prefix=env_prefix)
    if relax_raw is None:
        relax_raw = "M1Scalper,ImpulseRetrace"
    relax_tags = {tag.strip().lower() for tag in str(relax_raw).split(",") if tag.strip()}
    split_directional = _strategy_env_bool("PERF_GUARD_SPLIT_DIRECTIONAL", True, env_prefix=env_prefix)

    regime_filter_enabled = _strategy_env_bool("PERF_GUARD_REGIME_FILTER", True, env_prefix=env_prefix)
    regime_min_trades = max(5, _strategy_env_int("PERF_GUARD_REGIME_MIN_TRADES", 20, env_prefix=env_prefix))

    failfast_min_trades = max(0, _strategy_env_int("PERF_GUARD_FAILFAST_MIN_TRADES", 12, env_prefix=env_prefix))
    failfast_pf = max(0.0, _strategy_env_float("PERF_GUARD_FAILFAST_PF", 0.75, env_prefix=env_prefix))
    failfast_win = max(0.0, _strategy_env_float("PERF_GUARD_FAILFAST_WIN", 0.40, env_prefix=env_prefix))

    sl_loss_rate_min_trades = max(0, _strategy_env_int("PERF_GUARD_SL_LOSS_RATE_MIN_TRADES", 12, env_prefix=env_prefix))
    sl_loss_rate_max_default = _strategy_env_float("PERF_GUARD_SL_LOSS_RATE_MAX", 0.0, env_prefix=env_prefix)

    pocket_enabled = _strategy_env_bool("POCKET_PERF_GUARD_ENABLED", False, env_prefix=env_prefix)
    pocket_lookback_days = max(1, _strategy_env_int("POCKET_PERF_GUARD_LOOKBACK_DAYS", 7, env_prefix=env_prefix))
    pocket_min_trades = max(10, _strategy_env_int("POCKET_PERF_GUARD_MIN_TRADES", 60, env_prefix=env_prefix))
    pocket_pf_min_default = _strategy_env_float("POCKET_PERF_GUARD_PF_MIN", 0.95, env_prefix=env_prefix)
    pocket_win_min_default = _strategy_env_float("POCKET_PERF_GUARD_WIN_MIN", 0.50, env_prefix=env_prefix)
    pocket_ttl_sec = max(30.0, _strategy_env_float("POCKET_PERF_GUARD_TTL_SEC", 180.0, env_prefix=env_prefix))

    scale_enabled = _strategy_env_bool("PERF_SCALE_ENABLED", True, env_prefix=env_prefix)
    scale_lookback_days = max(1, _strategy_env_int("PERF_SCALE_LOOKBACK_DAYS", 7, env_prefix=env_prefix))
    scale_min_trades = max(5, _strategy_env_int("PERF_SCALE_MIN_TRADES", 20, env_prefix=env_prefix))
    scale_pf_min = _strategy_env_float("PERF_SCALE_PF_MIN", 1.15, env_prefix=env_prefix)
    scale_win_min = _strategy_env_float("PERF_SCALE_WIN_MIN", 0.55, env_prefix=env_prefix)
    scale_avg_pips_min = _strategy_env_float("PERF_SCALE_AVG_PIPS_MIN", 0.20, env_prefix=env_prefix)
    scale_step = max(0.0, _strategy_env_float("PERF_SCALE_STEP", 0.05, env_prefix=env_prefix))
    scale_max_mult = max(1.0, _strategy_env_float("PERF_SCALE_MAX_MULT", 1.25, env_prefix=env_prefix))
    scale_ttl_sec = max(30.0, _strategy_env_float("PERF_SCALE_TTL_SEC", 180.0, env_prefix=env_prefix))

    cfg = PerfGuardCfg(
        env_prefix=env_prefix,
        enabled=enabled,
        mode=mode,
        lookback_days=lookback_days,
        min_trades=min_trades,
        pf_min_default=pf_min_default,
        win_min_default=win_min_default,
        ttl_sec=ttl_sec,
        hourly_enabled=hourly_enabled,
        hourly_min_trades=hourly_min_trades,
        relax_tags=relax_tags,
        split_directional=split_directional,
        regime_filter_enabled=regime_filter_enabled,
        regime_min_trades=regime_min_trades,
        failfast_min_trades=failfast_min_trades,
        failfast_pf=failfast_pf,
        failfast_win=failfast_win,
        sl_loss_rate_min_trades=sl_loss_rate_min_trades,
        sl_loss_rate_max_default=sl_loss_rate_max_default,
        pocket_enabled=pocket_enabled,
        pocket_lookback_days=pocket_lookback_days,
        pocket_min_trades=pocket_min_trades,
        pocket_pf_min_default=pocket_pf_min_default,
        pocket_win_min_default=pocket_win_min_default,
        pocket_ttl_sec=pocket_ttl_sec,
        scale_enabled=scale_enabled,
        scale_lookback_days=scale_lookback_days,
        scale_min_trades=scale_min_trades,
        scale_pf_min=scale_pf_min,
        scale_win_min=scale_win_min,
        scale_avg_pips_min=scale_avg_pips_min,
        scale_step=scale_step,
        scale_max_mult=scale_max_mult,
        scale_ttl_sec=scale_ttl_sec,
    )
    _CFG_CACHE[key] = cfg
    return cfg

_DIRECTIONAL_TOKENS = {"bear", "bull", "long", "short"}

_cache: dict[tuple[str, str, str, Optional[int], Optional[str]], tuple[float, bool, str, int]] = {}
_pocket_cache: dict[tuple[str, str], tuple[float, bool, str, int]] = {}
_scale_cache: dict[
    tuple[str, str, str], tuple[float, float, str, int, float, float, float]
] = {}


def _threshold(name: str, pocket: str, default: float, cfg: PerfGuardCfg) -> float:
    pocket_key = (pocket or "").strip().upper()
    if pocket_key:
        raw = env_get(
            f"{name}_{pocket_key}",
            None,
            prefix=cfg.env_prefix,
            allow_global_fallback=False,
        )
        if raw is not None:
            try:
                return float(raw)
            except ValueError:
                pass
    raw = env_get(
        name,
        None,
        prefix=cfg.env_prefix,
        allow_global_fallback=False,
    )
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def _sl_loss_rate_max(pocket: str, cfg: PerfGuardCfg) -> float:
    # Default: enabled only for scalp pockets; other pockets default disabled.
    p = (pocket or "").strip().lower()
    if p.startswith("scalp"):
        return _threshold("PERF_GUARD_SL_LOSS_RATE_MAX", pocket, 0.65, cfg)
    return _threshold("PERF_GUARD_SL_LOSS_RATE_MAX", pocket, cfg.sl_loss_rate_max_default, cfg)


@dataclass(frozen=True, slots=True)
class PerfDecision:
    allowed: bool
    reason: str
    sample: int


@dataclass(frozen=True, slots=True)
class PerfScaleDecision:
    multiplier: float
    reason: str
    sample: int
    pf: float
    win_rate: float
    avg_pips: float


def _is_directional_tag(raw: str) -> bool:
    tokens = [t for t in re.split(r"[-_/]+", raw) if t]
    if not tokens:
        return False
    if "breakout" in tokens and ("up" in tokens or "down" in tokens):
        return True
    if any(t in _DIRECTIONAL_TOKENS for t in tokens):
        return True
    if "up" in tokens or "down" in tokens:
        return True
    return False


def _tag_variants(tag: str, *, split_directional: bool) -> Tuple[str, ...]:
    raw = str(tag).strip().lower()
    if not raw:
        return ("",)
    if split_directional and _is_directional_tag(raw):
        return (raw,)
    base = raw.split("-", 1)[0].strip()
    if base and base != raw:
        return (raw, base)
    return (raw,)


def _is_relaxed(tag: str, cfg: PerfGuardCfg) -> bool:
    if not cfg.relax_tags:
        return False
    for variant in _tag_variants(tag, split_directional=cfg.split_directional):
        if variant in cfg.relax_tags:
            return True
    return False


def _pocket_regime_label(pocket: str) -> Optional[str]:
    if not pocket:
        return None
    if pocket.lower() == "macro":
        reg = current_regime("H4", event_mode=False)
        if reg:
            return reg
        return current_regime("H1", event_mode=False)
    return current_regime("M1", event_mode=False)


def _query_perf_row(
    *,
    con: sqlite3.Connection,
    tag: str,
    tag_clause: str,
    params: list,
    pocket: str,
    hour: Optional[int],
    regime_label: Optional[str],
    cfg: PerfGuardCfg,
) -> Optional[sqlite3.Row]:
    regime_clause = ""
    if regime_label:
        col = "macro_regime" if pocket.lower() == "macro" else "micro_regime"
        regime_clause = f" AND LOWER(COALESCE({col}, '')) = ?"
        params.append(regime_label.lower())
    params.extend(
        [
            pocket,
            f"-{cfg.lookback_days} day",
            f"{hour:02d}" if hour is not None else None,
            f"{hour:02d}" if hour is not None else None,
        ]
    )
    try:
        return con.execute(
            f"""
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS profit,
              SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS loss,
              SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS win,
              SUM(pl_pips) AS sum_pips,
              -- Stop-loss exits that actually realized a loss (exclude BE/lock SL profits).
              SUM(CASE WHEN close_reason = 'STOP_LOSS_ORDER' AND pl_pips < 0 THEN 1 ELSE 0 END) AS sl_loss_n,
              -- Emergency: broker forced liquidation.
              SUM(CASE WHEN close_reason = 'MARKET_ORDER_MARGIN_CLOSEOUT' THEN 1 ELSE 0 END) AS margin_closeout_n
            FROM trades
            WHERE {tag_clause}
              AND pocket = ?
              AND close_time >= datetime('now', ?)
              AND (strftime('%H', close_time) = ? OR ? IS NULL)
            {regime_clause}
            """,
            params,
        ).fetchone()
    except Exception as exc:
        LOG.debug("[PERF_GUARD] query failed tag=%s pocket=%s err=%s", tag, pocket, exc)
        return None


def _query_perf_scale(tag: str, pocket: str, cfg: PerfGuardCfg) -> Tuple[float, float, float, int]:
    if not _DB.exists():
        return 1.0, 0.0, 0.0, 0
    try:
        con = sqlite3.connect(_DB)
        con.row_factory = sqlite3.Row
    except Exception:
        return 1.0, 0.0, 0.0, 0
    variants = _tag_variants(tag, split_directional=cfg.split_directional)
    if not variants or variants == ("",):
        return 1.0, 0.0, 0.0, 0
    placeholders = ",".join("?" for _ in variants)
    tag_clause = f"LOWER(COALESCE(NULLIF(strategy_tag, ''), strategy)) IN ({placeholders})"
    params = list(variants)
    params.extend([pocket, f"-{cfg.scale_lookback_days} day"])
    try:
        row = con.execute(
            f"""
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS profit,
              SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS loss,
              SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS win,
              SUM(pl_pips) AS sum_pips
            FROM trades
            WHERE {tag_clause}
              AND pocket = ?
              AND close_time >= datetime('now', ?)
            """,
            params,
        ).fetchone()
    except Exception as exc:
        LOG.debug("[PERF_SCALE] query failed tag=%s pocket=%s err=%s", tag, pocket, exc)
        return 1.0, 0.0, 0.0, 0
    n = int(row["n"] or 0) if row else 0
    if n <= 0:
        return 1.0, 0.0, 0.0, 0
    profit = float(row["profit"] or 0.0)
    loss = float(row["loss"] or 0.0)
    win = float(row["win"] or 0.0)
    sum_pips = float(row["sum_pips"] or 0.0)
    pf = profit / loss if loss > 0 else float("inf")
    win_rate = win / n if n else 0.0
    avg_pips = sum_pips / n if n else 0.0
    return pf, win_rate, avg_pips, n


def _query_perf(
    tag: str, pocket: str, hour: Optional[int], regime_label: Optional[str], cfg: PerfGuardCfg
) -> Tuple[bool, str, int]:
    if not _DB.exists():
        return True, "no_db", 0
    try:
        con = sqlite3.connect(_DB)
        con.row_factory = sqlite3.Row
    except Exception:
        return True, "db_open_failed", 0
    variants = _tag_variants(tag, split_directional=cfg.split_directional)
    if not variants or variants == ("",):
        return True, "empty_tag", 0
    placeholders = ",".join("?" for _ in variants)
    tag_clause = f"LOWER(COALESCE(NULLIF(strategy_tag, ''), strategy)) IN ({placeholders})"
    params = list(variants)
    try:
        row = None
        if regime_label:
            row = _query_perf_row(
                con=con,
                tag=tag,
                tag_clause=tag_clause,
                params=list(params),
                pocket=pocket,
                hour=hour,
                regime_label=regime_label,
                cfg=cfg,
            )
            if row is not None:
                n = int(row["n"] or 0)
                if n < cfg.regime_min_trades:
                    row = None
        if row is None:
            row = _query_perf_row(
                con=con,
                tag=tag,
                tag_clause=tag_clause,
                params=list(params),
                pocket=pocket,
                hour=hour,
                regime_label=None,
                cfg=cfg,
            )
    except Exception as exc:
        LOG.debug("[PERF_GUARD] query failed tag=%s pocket=%s err=%s", tag, pocket, exc)
        return True, "query_failed", 0
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not row:
        return True, "no_rows", 0
    n = int(row["n"] or 0)
    profit = float(row["profit"] or 0.0)
    loss = float(row["loss"] or 0.0)
    win = float(row["win"] or 0.0)
    sl_loss_n = int(row["sl_loss_n"] or 0)
    margin_closeout_n = int(row["margin_closeout_n"] or 0)
    pf = profit / loss if loss > 0 else float("inf")
    win_rate = win / n if n > 0 else 0.0

    # Emergency block: broker forced liquidation observed in this window.
    if margin_closeout_n > 0:
        return False, f"margin_closeout_n={margin_closeout_n} n={n}", n

    # Fail-fast (before warmup) when stats are clearly bad.
    if cfg.failfast_min_trades > 0 and n >= cfg.failfast_min_trades:
        if (cfg.failfast_pf > 0 and pf < cfg.failfast_pf) or (cfg.failfast_win > 0 and win_rate < cfg.failfast_win):
            return False, f"failfast:pf={pf:.2f} win={win_rate:.2f} n={n}", n

    # Stop-loss (loss) rate guard (before warmup). Only apply when PF < 1.0 (already losing).
    sl_rate_max = _sl_loss_rate_max(pocket, cfg)
    if (
        sl_rate_max > 0
        and cfg.sl_loss_rate_min_trades > 0
        and n >= cfg.sl_loss_rate_min_trades
        and pf < 1.0
    ):
        sl_rate = (float(sl_loss_n) / float(n)) if n > 0 else 0.0
        if sl_rate >= sl_rate_max:
            return False, f"sl_loss_rate={sl_rate:.2f} pf={pf:.2f} n={n}", n

    min_trades = cfg.hourly_min_trades if hour is not None else cfg.min_trades
    if n < min_trades:
        return True, f"warmup_n={n}", n

    pf_min = _threshold("PERF_GUARD_PF_MIN", pocket, cfg.pf_min_default, cfg)
    win_min = _threshold("PERF_GUARD_WIN_MIN", pocket, cfg.win_min_default, cfg)
    if pf < pf_min or win_rate < win_min:
        return False, f"pf={pf:.2f} win={win_rate:.2f} n={n}", n
    return True, f"pf={pf:.2f} win={win_rate:.2f} n={n}", n


def _query_pocket_perf(pocket: str, cfg: PerfGuardCfg) -> Tuple[bool, str, int]:
    if not _DB.exists():
        return True, "no_db", 0
    try:
        con = sqlite3.connect(_DB)
        con.row_factory = sqlite3.Row
    except Exception:
        return True, "db_open_failed", 0
    try:
        row = con.execute(
            """
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS profit,
              SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS loss,
              SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS win
            FROM trades
            WHERE pocket = ?
              AND close_time >= datetime('now', ?)
            """,
            (pocket, f"-{cfg.pocket_lookback_days} day"),
        ).fetchone()
    except Exception as exc:
        LOG.debug("[POCKET_GUARD] query failed pocket=%s err=%s", pocket, exc)
        return True, "query_failed", 0
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not row:
        return True, "no_rows", 0
    n = int(row["n"] or 0)
    if n < cfg.pocket_min_trades:
        return True, f"warmup_n={n}", n
    profit = float(row["profit"] or 0.0)
    loss = float(row["loss"] or 0.0)
    win = float(row["win"] or 0.0)
    pf = profit / loss if loss > 0 else float("inf")
    win_rate = win / n if n > 0 else 0.0

    pf_min = _threshold("POCKET_PERF_GUARD_PF_MIN", pocket, cfg.pocket_pf_min_default, cfg)
    win_min = _threshold("POCKET_PERF_GUARD_WIN_MIN", pocket, cfg.pocket_win_min_default, cfg)
    if pf < pf_min or win_rate < win_min:
        return False, f"pf={pf:.2f} win={win_rate:.2f} n={n}", n
    return True, f"pf={pf:.2f} win={win_rate:.2f} n={n}", n


def is_allowed(
    tag: str,
    pocket: str,
    *,
    hour: Optional[int] = None,
    env_prefix: Optional[str] = None,
) -> PerfDecision:
    """
    hour: optional UTC hour string (0-23) to apply hourly PF filter. When None, aggregate filter only.
    """
    cfg = _get_cfg(env_prefix)
    if not cfg.enabled or cfg.mode in {"off", "disabled", "false", "no"}:
        return PerfDecision(True, "disabled", 0)
    relaxed = _is_relaxed(tag, cfg)
    regime_label = _pocket_regime_label(pocket) if cfg.regime_filter_enabled else None
    key = (_cfg_key(env_prefix), tag, pocket, hour if cfg.hourly_enabled else None, regime_label)
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and now - cached[0] <= cfg.ttl_sec:
        _, ok, reason, sample = cached
        return PerfDecision(ok, reason, sample)

    # hourly check first (stricter). If enabled and blocked, return immediately.
    if cfg.hourly_enabled and hour is not None:
        ok_hour, reason_hour, sample_hour = _query_perf(tag, pocket, hour, regime_label, cfg)
        if not ok_hour:
            reason_txt = f"hour{hour}:{reason_hour}"
            allowed = ok_hour if cfg.mode == "block" else True
            if cfg.mode != "block":
                reason_txt = f"warn:{reason_txt}"
            elif relaxed:
                allowed = True
                reason_txt = f"relaxed:{reason_txt}"
            _cache[key] = (now, allowed, reason_txt, sample_hour)
            return PerfDecision(allowed, reason_txt, sample_hour)

    ok, reason, sample = _query_perf(tag, pocket, None, regime_label, cfg)
    allowed = ok if cfg.mode == "block" else True
    if not ok and cfg.mode == "block" and relaxed:
        allowed = True
        reason_txt = f"relaxed:{reason}"
    else:
        reason_txt = reason if cfg.mode == "block" else f"warn:{reason}"
    _cache[key] = (now, allowed, reason_txt, sample)
    return PerfDecision(allowed, reason_txt, sample)


def is_pocket_allowed(pocket: str, *, env_prefix: Optional[str] = None) -> PerfDecision:
    """
    Pocket-level PF/win guard. Useful when an entire pocket degrades.
    """
    cfg = _get_cfg(env_prefix)
    if not cfg.pocket_enabled:
        return PerfDecision(True, "disabled", 0)
    pocket_key = (pocket or "").strip().lower()
    if not pocket_key or pocket_key == "manual":
        return PerfDecision(True, "skip", 0)
    key = (_cfg_key(env_prefix), pocket_key)
    now = time.monotonic()
    cached = _pocket_cache.get(key)
    if cached and now - cached[0] <= cfg.pocket_ttl_sec:
        return PerfDecision(cached[1], cached[2], cached[3])
    allowed, reason, n = _query_pocket_perf(pocket_key, cfg)
    _pocket_cache[key] = (now, allowed, reason, n)
    return PerfDecision(allowed, reason, n)


def perf_scale(tag: str, pocket: str, *, env_prefix: Optional[str] = None) -> PerfScaleDecision:
    cfg = _get_cfg(env_prefix)
    if not cfg.scale_enabled:
        return PerfScaleDecision(1.0, "disabled", 0, 1.0, 0.0, 0.0)
    key = (_cfg_key(env_prefix), str(tag).strip().lower(), str(pocket).strip().lower())
    now = time.time()
    cached = _scale_cache.get(key)
    if cached and (now - cached[0]) < cfg.scale_ttl_sec:
        return PerfScaleDecision(cached[1], cached[2], cached[3], cached[4], cached[5], cached[6])

    pf, win_rate, avg_pips, n = _query_perf_scale(tag, pocket, cfg)
    if n < cfg.scale_min_trades:
        dec = PerfScaleDecision(1.0, "insufficient", n, pf, win_rate, avg_pips)
        _scale_cache[key] = (now, dec.multiplier, dec.reason, dec.sample, dec.pf, dec.win_rate, dec.avg_pips)
        return dec

    score = 0
    if pf >= cfg.scale_pf_min:
        score += 1
    if win_rate >= cfg.scale_win_min:
        score += 1
    if avg_pips >= cfg.scale_avg_pips_min:
        score += 1
    mult = min(cfg.scale_max_mult, 1.0 + score * cfg.scale_step)
    if mult < 1.0:
        mult = 1.0
    reason = "boost" if mult > 1.0 else "flat"
    dec = PerfScaleDecision(mult, reason, n, pf, win_rate, avg_pips)
    _scale_cache[key] = (now, dec.multiplier, dec.reason, dec.sample, dec.pf, dec.win_rate, dec.avg_pips)
    return dec

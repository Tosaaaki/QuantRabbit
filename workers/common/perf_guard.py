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
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, Tuple

from workers.common.quality_gate import current_regime

LOG = logging.getLogger(__name__)

_DB = pathlib.Path("logs/trades.db")

_RAW_ENABLED = os.getenv("PERF_GUARD_ENABLED")
if _RAW_ENABLED is None:
    _RAW_ENABLED = os.getenv("PERF_GUARD_GLOBAL_ENABLED", "1")
_ENABLED = str(_RAW_ENABLED).strip().lower() not in {"", "0", "false", "no"}
_MODE = os.getenv("PERF_GUARD_MODE", "block").strip().lower()
_LOOKBACK_DAYS = max(1, int(float(os.getenv("PERF_GUARD_LOOKBACK_DAYS", "3"))))
_MIN_TRADES = max(5, int(float(os.getenv("PERF_GUARD_MIN_TRADES", "12"))))
_PF_MIN = float(os.getenv("PERF_GUARD_PF_MIN", "0.9") or 0.9)
_WIN_MIN = float(os.getenv("PERF_GUARD_WIN_MIN", "0.48") or 0.48)
_TTL_SEC = max(30.0, float(os.getenv("PERF_GUARD_TTL_SEC", "120")) or 120.0)
_HOURLY_ENABLED = os.getenv("PERF_GUARD_HOURLY", "0").strip().lower() not in {"", "0", "false", "no"}
_HOURLY_MIN_TRADES = max(5, int(float(os.getenv("PERF_GUARD_HOURLY_MIN_TRADES", "12"))))
_RAW_RELAX_TAGS = os.getenv("PERF_GUARD_RELAX_TAGS")
if _RAW_RELAX_TAGS is None:
    _RAW_RELAX_TAGS = "M1Scalper,ImpulseRetrace"
_RELAX_TAGS = {tag.strip().lower() for tag in _RAW_RELAX_TAGS.split(",") if tag.strip()}

_RAW_RESET_AT = (os.getenv("PERF_GUARD_RESET_AT") or os.getenv("PERF_GUARD_RESET_TS") or "").strip()
_RAW_RESET_TAGS = (os.getenv("PERF_GUARD_RESET_TAGS") or "").strip()
_RESET_TAGS = {tag.strip().lower() for tag in _RAW_RESET_TAGS.split(",") if tag.strip()}

_PERF_METRIC = (os.getenv("PERF_GUARD_METRIC", "realized_pl") or "realized_pl").strip().lower()
_POCKET_METRIC = (os.getenv("POCKET_PERF_GUARD_METRIC", _PERF_METRIC) or _PERF_METRIC).strip().lower()


def _metric_expr(metric: str) -> str:
    if metric in {"pips", "pip", "pl_pips"}:
        return "COALESCE(pl_pips, 0)"
    if metric in {"realized_pl", "realized", "pl"}:
        return "COALESCE(realized_pl, 0)"
    if metric in {"net_pl", "net", "net_realized"}:
        return "(COALESCE(realized_pl, 0) + COALESCE(commission, 0) + COALESCE(financing, 0))"
    return "COALESCE(pl_pips, 0)"


_PERF_EXPR = _metric_expr(_PERF_METRIC)
_POCKET_EXPR = _metric_expr(_POCKET_METRIC)
_SPLIT_DIRECTIONAL = os.getenv("PERF_GUARD_SPLIT_DIRECTIONAL", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_DIRECTIONAL_TOKENS = {"bear", "bull", "long", "short"}

_REGIME_FILTER_ENABLED = os.getenv("PERF_GUARD_REGIME_FILTER", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_REGIME_MIN_TRADES = max(5, int(float(os.getenv("PERF_GUARD_REGIME_MIN_TRADES", "20"))))

# Pocket-level guard (optional; defaults disabled)
_POCKET_ENABLED = os.getenv("POCKET_PERF_GUARD_ENABLED", "0").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_POCKET_LOOKBACK_DAYS = max(1, int(float(os.getenv("POCKET_PERF_GUARD_LOOKBACK_DAYS", "7"))))
_POCKET_MIN_TRADES = max(10, int(float(os.getenv("POCKET_PERF_GUARD_MIN_TRADES", "60"))))
_POCKET_PF_MIN = float(os.getenv("POCKET_PERF_GUARD_PF_MIN", "0.95") or 0.95)
_POCKET_WIN_MIN = float(os.getenv("POCKET_PERF_GUARD_WIN_MIN", "0.50") or 0.5)
_POCKET_TTL_SEC = max(30.0, float(os.getenv("POCKET_PERF_GUARD_TTL_SEC", "180")) or 180.0)

# Performance-based sizing (boost only)
_SCALE_ENABLED = os.getenv("PERF_SCALE_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
_SCALE_LOOKBACK_DAYS = max(1, int(float(os.getenv("PERF_SCALE_LOOKBACK_DAYS", "7"))))
_SCALE_MIN_TRADES = max(5, int(float(os.getenv("PERF_SCALE_MIN_TRADES", "20"))))
_SCALE_PF_MIN = float(os.getenv("PERF_SCALE_PF_MIN", "1.15") or 1.15)
_SCALE_WIN_MIN = float(os.getenv("PERF_SCALE_WIN_MIN", "0.55") or 0.55)
_SCALE_AVG_PIPS_MIN = float(os.getenv("PERF_SCALE_AVG_PIPS_MIN", "0.20") or 0.20)
_SCALE_STEP = max(0.0, float(os.getenv("PERF_SCALE_STEP", "0.05") or 0.05))
_SCALE_MAX_MULT = max(1.0, float(os.getenv("PERF_SCALE_MAX_MULT", "1.25") or 1.25))
_SCALE_TTL_SEC = max(30.0, float(os.getenv("PERF_SCALE_TTL_SEC", "180")) or 180.0)

_cache: dict[tuple[str, str, Optional[int], Optional[str]], tuple[float, bool, str, int]] = {}
_pocket_cache: dict[str, tuple[float, bool, str, int]] = {}
_scale_cache: dict[tuple[str, str], tuple[float, float, str, int, float, float, float]] = {}


def _threshold(name: str, pocket: str, default: float) -> float:
    pocket_key = (pocket or "").strip().upper()
    if pocket_key:
        raw = os.getenv(f"{name}_{pocket_key}")
        if raw is not None:
            try:
                return float(raw)
            except ValueError:
                pass
    raw = os.getenv(name)
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


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


def _tag_variants(tag: str) -> Tuple[str, ...]:
    raw = str(tag).strip().lower()
    if not raw:
        return ("",)
    if _SPLIT_DIRECTIONAL and _is_directional_tag(raw):
        return (raw,)
    base = raw.split("-", 1)[0].strip()
    if base and base != raw:
        return (raw, base)
    return (raw,)


def _is_relaxed(tag: str) -> bool:
    if not _RELAX_TAGS:
        return False
    for variant in _tag_variants(tag):
        if variant in _RELAX_TAGS:
            return True
    return False


def _parse_reset_at(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        if raw.isdigit():
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except ValueError:
        pass
    try:
        cleaned = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


_RESET_AT = _parse_reset_at(_RAW_RESET_AT)


def _reset_at_for_tag(tag: str) -> Optional[datetime]:
    if _RESET_AT is None:
        return None
    if not _RESET_TAGS:
        return _RESET_AT
    for variant in _tag_variants(tag):
        if variant in _RESET_TAGS:
            return _RESET_AT
    return None


def _effective_lookback(tag: str) -> str:
    base = f"-{_LOOKBACK_DAYS} day"
    reset_at = _reset_at_for_tag(tag)
    if reset_at is None:
        return base
    now = datetime.now(timezone.utc)
    delta_sec = (now - reset_at).total_seconds()
    if delta_sec <= 0:
        return "-0 second"
    max_sec = _LOOKBACK_DAYS * 86400
    if delta_sec < max_sec:
        return f"-{int(delta_sec)} seconds"
    return base


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
    lookback: str,
) -> Optional[sqlite3.Row]:
    regime_clause = ""
    if regime_label:
        col = "macro_regime" if pocket.lower() == "macro" else "micro_regime"
        regime_clause = f" AND LOWER(COALESCE({col}, '')) = ?"
        params.append(regime_label.lower())
    params.extend(
        [
            pocket,
            lookback,
            f"{hour:02d}" if hour is not None else None,
            f"{hour:02d}" if hour is not None else None,
        ]
    )
    try:
        return con.execute(
            f"""
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN {_PERF_EXPR} > 0 THEN {_PERF_EXPR} ELSE 0 END) AS profit,
              SUM(CASE WHEN {_PERF_EXPR} < 0 THEN ABS({_PERF_EXPR}) ELSE 0 END) AS loss,
              SUM(CASE WHEN {_PERF_EXPR} > 0 THEN 1 ELSE 0 END) AS win,
              SUM({_PERF_EXPR}) AS sum_metric
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


def _query_perf_scale(tag: str, pocket: str) -> Tuple[float, float, float, int]:
    if not _DB.exists():
        return 1.0, 0.0, 0.0, 0
    try:
        con = sqlite3.connect(_DB)
        con.row_factory = sqlite3.Row
    except Exception:
        return 1.0, 0.0, 0.0, 0
    variants = _tag_variants(tag)
    if not variants or variants == ("",):
        return 1.0, 0.0, 0.0, 0
    placeholders = ",".join("?" for _ in variants)
    tag_clause = f"LOWER(COALESCE(NULLIF(strategy_tag, ''), strategy)) IN ({placeholders})"
    params = list(variants)
    params.extend([pocket, f"-{_SCALE_LOOKBACK_DAYS} day"])
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
    tag: str, pocket: str, hour: Optional[int], regime_label: Optional[str]
) -> Tuple[bool, str, int]:
    if not _DB.exists():
        return True, "no_db", 0
    try:
        con = sqlite3.connect(_DB)
        con.row_factory = sqlite3.Row
    except Exception:
        return True, "db_open_failed", 0
    variants = _tag_variants(tag)
    if not variants or variants == ("",):
        return True, "empty_tag", 0
    placeholders = ",".join("?" for _ in variants)
    tag_clause = f"LOWER(COALESCE(NULLIF(strategy_tag, ''), strategy)) IN ({placeholders})"
    params = list(variants)
    try:
        row = None
        lookback = _effective_lookback(tag)
        if regime_label:
            row = _query_perf_row(
                con=con,
                tag=tag,
                tag_clause=tag_clause,
                params=list(params),
                pocket=pocket,
                hour=hour,
                regime_label=regime_label,
                lookback=lookback,
            )
            if row is not None:
                n = int(row["n"] or 0)
                if n < _REGIME_MIN_TRADES:
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
                lookback=lookback,
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
    min_trades = _HOURLY_MIN_TRADES if hour is not None else _MIN_TRADES
    if n < min_trades:
        return True, f"warmup_n={n}", n
    profit = float(row["profit"] or 0.0)
    loss = float(row["loss"] or 0.0)
    win = float(row["win"] or 0.0)
    pf = profit / loss if loss > 0 else float("inf")
    win_rate = win / n if n > 0 else 0.0

    pf_min = _threshold("PERF_GUARD_PF_MIN", pocket, _PF_MIN)
    win_min = _threshold("PERF_GUARD_WIN_MIN", pocket, _WIN_MIN)
    if pf < pf_min or win_rate < win_min:
        return False, f"pf={pf:.2f} win={win_rate:.2f} n={n}", n
    return True, f"pf={pf:.2f} win={win_rate:.2f} n={n}", n


def _query_pocket_perf(pocket: str) -> Tuple[bool, str, int]:
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
              SUM(CASE WHEN {expr} > 0 THEN {expr} ELSE 0 END) AS profit,
              SUM(CASE WHEN {expr} < 0 THEN ABS({expr}) ELSE 0 END) AS loss,
              SUM(CASE WHEN {expr} > 0 THEN 1 ELSE 0 END) AS win
            FROM trades
            WHERE pocket = ?
              AND close_time >= datetime('now', ?)
            """.format(expr=_POCKET_EXPR),
            (pocket, f"-{_POCKET_LOOKBACK_DAYS} day"),
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
    if n < _POCKET_MIN_TRADES:
        return True, f"warmup_n={n}", n
    profit = float(row["profit"] or 0.0)
    loss = float(row["loss"] or 0.0)
    win = float(row["win"] or 0.0)
    pf = profit / loss if loss > 0 else float("inf")
    win_rate = win / n if n > 0 else 0.0

    pf_min = _threshold("POCKET_PERF_GUARD_PF_MIN", pocket, _POCKET_PF_MIN)
    win_min = _threshold("POCKET_PERF_GUARD_WIN_MIN", pocket, _POCKET_WIN_MIN)
    if pf < pf_min or win_rate < win_min:
        return False, f"pf={pf:.2f} win={win_rate:.2f} n={n}", n
    return True, f"pf={pf:.2f} win={win_rate:.2f} n={n}", n


def is_allowed(tag: str, pocket: str, *, hour: Optional[int] = None) -> PerfDecision:
    """
    hour: optional UTC hour string (0-23) to apply hourly PF filter. When None, aggregate filter only.
    """
    if not _ENABLED or _MODE in {"off", "disabled", "false", "no"}:
        return PerfDecision(True, "disabled", 0)
    relaxed = _is_relaxed(tag)
    regime_label = _pocket_regime_label(pocket) if _REGIME_FILTER_ENABLED else None
    key = (tag, pocket, hour if _HOURLY_ENABLED else None, regime_label)
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and now - cached[0] <= _TTL_SEC:
        _, ok, reason, sample = cached
        return PerfDecision(ok, reason, sample)

    # hourly check first (stricter). If enabled and blocked, return immediately.
    if _HOURLY_ENABLED and hour is not None:
        ok_hour, reason_hour, sample_hour = _query_perf(tag, pocket, hour, regime_label)
        if not ok_hour:
            reason_txt = f"hour{hour}:{reason_hour}"
            allowed = ok_hour if _MODE == "block" else True
            if _MODE != "block":
                reason_txt = f"warn:{reason_txt}"
            elif relaxed:
                allowed = True
                reason_txt = f"relaxed:{reason_txt}"
            _cache[key] = (now, allowed, reason_txt, sample_hour)
            return PerfDecision(allowed, reason_txt, sample_hour)

    ok, reason, sample = _query_perf(tag, pocket, None, regime_label)
    allowed = ok if _MODE == "block" else True
    if not ok and _MODE == "block" and relaxed:
        allowed = True
        reason_txt = f"relaxed:{reason}"
    else:
        reason_txt = reason if _MODE == "block" else f"warn:{reason}"
    _cache[key] = (now, allowed, reason_txt, sample)
    return PerfDecision(allowed, reason_txt, sample)


def is_pocket_allowed(pocket: str) -> PerfDecision:
    """
    Pocket-level PF/win guard. Useful when an entire pocket degrades.
    """
    if not _POCKET_ENABLED:
        return PerfDecision(True, "disabled", 0)
    pocket_key = (pocket or "").strip().lower()
    if not pocket_key or pocket_key == "manual":
        return PerfDecision(True, "skip", 0)
    now = time.monotonic()
    cached = _pocket_cache.get(pocket_key)
    if cached and now - cached[0] <= _POCKET_TTL_SEC:
        return PerfDecision(cached[1], cached[2], cached[3])
    allowed, reason, n = _query_pocket_perf(pocket_key)
    _pocket_cache[pocket_key] = (now, allowed, reason, n)
    return PerfDecision(allowed, reason, n)


def perf_scale(tag: str, pocket: str) -> PerfScaleDecision:
    if not _SCALE_ENABLED:
        return PerfScaleDecision(1.0, "disabled", 0, 1.0, 0.0, 0.0)
    key = (str(tag).strip().lower(), str(pocket).strip().lower())
    now = time.time()
    cached = _scale_cache.get(key)
    if cached and (now - cached[0]) < _SCALE_TTL_SEC:
        return PerfScaleDecision(cached[1], cached[2], cached[3], cached[4], cached[5], cached[6])

    pf, win_rate, avg_pips, n = _query_perf_scale(tag, pocket)
    if n < _SCALE_MIN_TRADES:
        dec = PerfScaleDecision(1.0, "insufficient", n, pf, win_rate, avg_pips)
        _scale_cache[key] = (now, dec.multiplier, dec.reason, dec.sample, dec.pf, dec.win_rate, dec.avg_pips)
        return dec

    score = 0
    if pf >= _SCALE_PF_MIN:
        score += 1
    if win_rate >= _SCALE_WIN_MIN:
        score += 1
    if avg_pips >= _SCALE_AVG_PIPS_MIN:
        score += 1
    mult = min(_SCALE_MAX_MULT, 1.0 + score * _SCALE_STEP)
    if mult < 1.0:
        mult = 1.0
    reason = "boost" if mult > 1.0 else "flat"
    dec = PerfScaleDecision(mult, reason, n, pf, win_rate, avg_pips)
    _scale_cache[key] = (now, dec.multiplier, dec.reason, dec.sample, dec.pf, dec.win_rate, dec.avg_pips)
    return dec

"""
Runtime SLO guard for order preflight.

Evaluates recent strategy-control metrics (`data_lag_ms` / `decision_latency_ms`)
and blocks new entries when execution quality is degraded.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import pathlib
import sqlite3
import time
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_csv_set(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {token.strip().lower() for token in str(raw).split(",") if token.strip()}


def _strategy_variants(strategy_tag: Optional[str]) -> set[str]:
    raw = str(strategy_tag or "").strip().lower()
    if not raw:
        return set()
    out = {raw}
    if "-" in raw:
        out.add(raw.split("-", 1)[0].strip())
    return {item for item in out if item}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = max(0.0, min(100.0, float(pct)))
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    if pct >= 100.0:
        return float(sorted_vals[-1])
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


@dataclass(frozen=True, slots=True)
class SLODecision:
    allowed: bool
    reason: str
    sample: int
    data_lag_latest_ms: float | None = None
    data_lag_p95_ms: float | None = None
    decision_latency_latest_ms: float | None = None
    decision_latency_p95_ms: float | None = None


_CACHE: tuple[float, SLODecision] | None = None


def _enabled() -> bool:
    return _env_bool("ORDER_SLO_GUARD_ENABLED", False)


def _metrics_db_path() -> pathlib.Path:
    return pathlib.Path(os.getenv("ORDER_SLO_GUARD_DB_PATH", "logs/metrics.db"))


def _mode_filter() -> str:
    return str(os.getenv("ORDER_SLO_GUARD_MODE_FILTER", "strategy_control") or "").strip().lower()


def _applies_to(pocket: str, strategy_tag: Optional[str]) -> bool:
    apply_pockets = _env_csv_set(
        "ORDER_SLO_GUARD_APPLY_POCKETS",
        "scalp_fast,scalp,micro",
    )
    pocket_key = str(pocket or "").strip().lower()
    if apply_pockets and "*" not in apply_pockets and pocket_key not in apply_pockets:
        return False

    variants = _strategy_variants(strategy_tag)
    allow = _env_csv_set("ORDER_SLO_GUARD_STRATEGY_ALLOWLIST", "")
    if allow:
        return any(v in allow for v in variants)

    deny = _env_csv_set("ORDER_SLO_GUARD_STRATEGY_BLOCKLIST", "")
    if deny and any(v in deny for v in variants):
        return False
    return True


def _query_decision() -> SLODecision:
    db_path = _metrics_db_path()
    if not db_path.exists():
        return SLODecision(allowed=True, reason="no_metrics_db", sample=0)

    lookback_sec = max(10, _env_int("ORDER_SLO_GUARD_LOOKBACK_SEC", 180))
    sample_min = max(1, _env_int("ORDER_SLO_GUARD_SAMPLE_MIN", 8))
    max_rows = max(sample_min * 4, _env_int("ORDER_SLO_GUARD_MAX_ROWS", 500))
    mode = _mode_filter()

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.5)
    except Exception:
        return SLODecision(allowed=True, reason="metrics_db_open_failed", sample=0)

    try:
        rows = con.execute(
            """
            SELECT metric, value
            FROM metrics
            WHERE metric IN ('data_lag_ms', 'decision_latency_ms')
              AND datetime(ts) >= datetime('now', ?)
              AND (
                    ? = ''
                    OR (
                        json_valid(tags)
                        AND LOWER(COALESCE(json_extract(tags, '$.mode'), '')) = ?
                    )
              )
            ORDER BY ts DESC
            LIMIT ?
            """,
            (f"-{lookback_sec} seconds", mode, mode, int(max_rows)),
        ).fetchall()
    except Exception:
        return SLODecision(allowed=True, reason="metrics_query_failed", sample=0)
    finally:
        try:
            con.close()
        except Exception:
            pass

    lag_values: list[float] = []
    latency_values: list[float] = []
    for metric, value in rows:
        try:
            val = float(value)
        except Exception:
            continue
        if metric == "data_lag_ms":
            lag_values.append(val)
        elif metric == "decision_latency_ms":
            latency_values.append(val)

    sample = min(len(lag_values), len(latency_values))
    if sample < sample_min:
        return SLODecision(
            allowed=True,
            reason="slo_warmup",
            sample=sample,
            data_lag_latest_ms=(lag_values[0] if lag_values else None),
            decision_latency_latest_ms=(latency_values[0] if latency_values else None),
        )

    lag_latest = lag_values[0] if lag_values else None
    latency_latest = latency_values[0] if latency_values else None
    lag_p95 = _percentile(lag_values, 95.0) if lag_values else None
    latency_p95 = _percentile(latency_values, 95.0) if latency_values else None

    lag_latest_max = max(0.0, _env_float("ORDER_SLO_GUARD_DATA_LAG_MAX_MS", 4000.0))
    latency_latest_max = max(
        0.0, _env_float("ORDER_SLO_GUARD_DECISION_LATENCY_MAX_MS", 4000.0)
    )
    lag_p95_max = max(0.0, _env_float("ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS", 6000.0))
    latency_p95_max = max(
        0.0, _env_float("ORDER_SLO_GUARD_DECISION_LATENCY_P95_MAX_MS", 6000.0)
    )

    if lag_latest is not None and lag_latest_max > 0 and lag_latest > lag_latest_max:
        return SLODecision(
            allowed=False,
            reason="data_lag_latest_exceeded",
            sample=sample,
            data_lag_latest_ms=lag_latest,
            data_lag_p95_ms=lag_p95,
            decision_latency_latest_ms=latency_latest,
            decision_latency_p95_ms=latency_p95,
        )
    if (
        latency_latest is not None
        and latency_latest_max > 0
        and latency_latest > latency_latest_max
    ):
        return SLODecision(
            allowed=False,
            reason="decision_latency_latest_exceeded",
            sample=sample,
            data_lag_latest_ms=lag_latest,
            data_lag_p95_ms=lag_p95,
            decision_latency_latest_ms=latency_latest,
            decision_latency_p95_ms=latency_p95,
        )
    if lag_p95 is not None and lag_p95_max > 0 and lag_p95 > lag_p95_max:
        return SLODecision(
            allowed=False,
            reason="data_lag_p95_exceeded",
            sample=sample,
            data_lag_latest_ms=lag_latest,
            data_lag_p95_ms=lag_p95,
            decision_latency_latest_ms=latency_latest,
            decision_latency_p95_ms=latency_p95,
        )
    if latency_p95 is not None and latency_p95_max > 0 and latency_p95 > latency_p95_max:
        return SLODecision(
            allowed=False,
            reason="decision_latency_p95_exceeded",
            sample=sample,
            data_lag_latest_ms=lag_latest,
            data_lag_p95_ms=lag_p95,
            decision_latency_latest_ms=latency_latest,
            decision_latency_p95_ms=latency_p95,
        )

    return SLODecision(
        allowed=True,
        reason="healthy",
        sample=sample,
        data_lag_latest_ms=lag_latest,
        data_lag_p95_ms=lag_p95,
        decision_latency_latest_ms=latency_latest,
        decision_latency_p95_ms=latency_p95,
    )


def decide(*, pocket: str, strategy_tag: Optional[str]) -> SLODecision:
    if not _enabled():
        return SLODecision(allowed=True, reason="disabled", sample=0)
    if not _applies_to(pocket, strategy_tag):
        return SLODecision(allowed=True, reason="scope_skip", sample=0)

    global _CACHE
    now_mono = time.monotonic()
    ttl_sec = max(0.2, _env_float("ORDER_SLO_GUARD_TTL_SEC", 2.0))
    if _CACHE is not None:
        ts_mono, cached = _CACHE
        if now_mono - ts_mono <= ttl_sec:
            return cached

    decision = _query_decision()
    _CACHE = (now_mono, decision)
    return decision


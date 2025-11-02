from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from google.cloud import bigquery

_PROJECT = (
    os.getenv("BQ_PROJECT")
    or os.getenv("RISK_MODEL_PROJECT")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
)
_DATASET = os.getenv("BQ_DATASET", "quantrabbit")
_TABLE = os.getenv("BQ_TRADES_TABLE", "trades_raw")

_LOOKBACK_SHORT = int(os.getenv("BQ_STATS_LOOKBACK_SHORT", "7"))
_LOOKBACK_LONG = int(os.getenv("BQ_STATS_LOOKBACK_LONG", "30"))
_MIN_TRADES_SHORT = int(os.getenv("BQ_STATS_MIN_TRADES_SHORT", "8"))
_MIN_TRADES_LONG = int(os.getenv("BQ_STATS_MIN_TRADES_LONG", "20"))
_REFRESH_SEC = int(os.getenv("BQ_STATS_REFRESH_SEC", "900"))
_STATE_PATH = Path(os.getenv("BQ_STATS_STATE", "logs/bq_strategy_stats.json"))
_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

_STATE_LOCK = threading.Lock()
_CACHE: Dict[str, object] = {"ts": 0.0, "stats": {}}


@dataclass
class WindowStats:
    lookback: int
    trades: int
    wins: int
    losses: int
    avg_pips: float
    pf: float | None
    realized_pl: float

    @property
    def win_rate(self) -> float:
        if self.trades <= 0:
            return 0.0
        return float(self.wins) / float(self.trades)


def _client() -> bigquery.Client:
    if _PROJECT:
        return bigquery.Client(project=_PROJECT)
    return bigquery.Client()


def _load_state() -> None:
    if not _STATE_PATH.exists():
        return
    try:
        data = json.loads(_STATE_PATH.read_text())
        stats = data.get("stats") or data.get("entries") or {}
        with _STATE_LOCK:
            _CACHE["stats"] = stats
            ts_val = data.get("ts")
            if isinstance(ts_val, (int, float)):
                _CACHE["ts"] = float(ts_val)
            else:
                _CACHE["ts"] = time.time()
    except Exception as exc:  # pragma: no cover - best effort
        logging.warning("[BQ_STATS] failed to load state: %s", exc)


def _write_state(payload: dict) -> None:
    try:
        _STATE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as exc:  # pragma: no cover
        logging.warning("[BQ_STATS] failed to persist state: %s", exc)


def _query_strategy_windows() -> Dict[Tuple[str, str], Dict[int, WindowStats]]:
    client = _client()
    table_ref = f"`{client.project}.{_DATASET}.{_TABLE}`"
    lookback_short = max(_LOOKBACK_SHORT, 1)
    lookback_long = max(_LOOKBACK_LONG, lookback_short)
    query = f"""
    WITH base AS (
      SELECT pocket, strategy, close_time, pl_pips, realized_pl
      FROM {table_ref}
      WHERE state = 'CLOSED'
        AND close_time IS NOT NULL
    ),
    short_window AS (
      SELECT {_LOOKBACK_SHORT} AS lookback, pocket, strategy, close_time, pl_pips, realized_pl
      FROM base
      WHERE close_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_short} DAY)
    ),
    long_window AS (
      SELECT {_LOOKBACK_LONG} AS lookback, pocket, strategy, close_time, pl_pips, realized_pl
      FROM base
      WHERE close_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_long} DAY)
    ),
    unioned AS (
      SELECT * FROM short_window
      UNION ALL
      SELECT * FROM long_window
    )
    SELECT
      lookback,
      pocket,
      strategy,
      COUNT(*) AS trades,
      SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
      SUM(CASE WHEN pl_pips < 0 THEN 1 ELSE 0 END) AS losses,
      AVG(pl_pips) AS avg_pips,
      SAFE_DIVIDE(
        SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END),
        NULLIF(ABS(SUM(CASE WHEN pl_pips < 0 THEN pl_pips ELSE 0 END)), 0)
      ) AS pf,
      SUM(realized_pl) AS realized_pl
    FROM unioned
    GROUP BY lookback, pocket, strategy
    """
    rows = client.query(query).result()
    stats: Dict[Tuple[str, str], Dict[int, WindowStats]] = {}
    for row in rows:
        pocket = (row.get("pocket") or "").lower()
        strategy = row.get("strategy") or ""
        if not pocket or not strategy:
            continue
        lookback = int(row.get("lookback") or 0)
        trades = int(row.get("trades") or 0)
        wins = int(row.get("wins") or 0)
        losses = int(row.get("losses") or 0)
        avg_pips = float(row.get("avg_pips") or 0.0)
        pf = row.get("pf")
        pf_val = float(pf) if pf is not None else None
        realized_pl = float(row.get("realized_pl") or 0.0)
        key = (pocket, strategy)
        windows = stats.setdefault(key, {})
        windows[lookback] = WindowStats(
            lookback=lookback,
            trades=trades,
            wins=wins,
            losses=losses,
            avg_pips=avg_pips,
            pf=pf_val,
            realized_pl=realized_pl,
        )
    return stats


def refresh_stats() -> Dict[Tuple[str, str], Dict[int, WindowStats]]:
    stats = _query_strategy_windows()
    payload: Dict[str, object] = {
        "ts": time.time(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookbacks": {
            "short": _LOOKBACK_SHORT,
            "long": _LOOKBACK_LONG,
        },
        "stats": {},
    }
    serializable: Dict[str, object] = {}
    for (pocket, strategy), windows in stats.items():
        key = f"{pocket}:{strategy}"
        serializable[key] = {
            str(lookback): {
                "trades": win.trades,
                "wins": win.wins,
                "losses": win.losses,
                "avg_pips": win.avg_pips,
                "pf": win.pf,
                "realized_pl": win.realized_pl,
            }
            for lookback, win in windows.items()
        }
    payload["stats"] = serializable
    _write_state(payload)
    with _STATE_LOCK:
        _CACHE["stats"] = serializable
        _CACHE["ts"] = payload["ts"]
    return stats


def _window_priority(windows: Dict[int, WindowStats]) -> Optional[WindowStats]:
    short = windows.get(_LOOKBACK_SHORT)
    long = windows.get(_LOOKBACK_LONG)
    if short and short.trades >= _MIN_TRADES_SHORT:
        return short
    if long and long.trades >= _MIN_TRADES_LONG:
        return long
    return short or long


def _compute_multiplier(window: Optional[WindowStats]) -> Optional[float]:
    if window is None or window.trades <= 0:
        return None
    pf = window.pf if window.pf is not None else 1.0
    win_rate = window.win_rate
    trades = window.trades
    # Conservative caps
    if trades >= 20 and pf >= 2.0 and win_rate >= 0.6:
        return 1.4
    if pf >= 1.6 and win_rate >= 0.55:
        return 1.25
    if pf >= 1.3 and win_rate >= 0.52:
        return 1.15
    if pf <= 0.6 or win_rate <= 0.35:
        return 0.5
    if pf <= 0.8 or win_rate <= 0.42:
        return 0.7
    if pf <= 0.95:
        return 0.85
    return 1.0


def multiplier_hint(pocket: str, strategy: str) -> Optional[float]:
    key = f"{(pocket or '').lower()}:{strategy or ''}"
    with _STATE_LOCK:
        stats = _CACHE.get("stats") or {}
        windows = stats.get(key)
    if not isinstance(windows, dict):
        return None
    window_objs: Dict[int, WindowStats] = {}
    for lookback_str, payload in windows.items():
        try:
            lookback = int(lookback_str)
        except (TypeError, ValueError):
            continue
        try:
            win_obj = WindowStats(
                lookback=lookback,
                trades=int(payload.get("trades") or 0),
                wins=int(payload.get("wins") or 0),
                losses=int(payload.get("losses") or 0),
                avg_pips=float(payload.get("avg_pips") or 0.0),
                pf=float(payload["pf"]) if payload.get("pf") is not None else None,
                realized_pl=float(payload.get("realized_pl") or 0.0),
            )
        except Exception:
            continue
        window_objs[lookback] = win_obj
    selected = _window_priority(window_objs)
    return _compute_multiplier(selected)


def get_stats(pocket: str, strategy: str) -> Optional[dict]:
    key = f"{(pocket or '').lower()}:{strategy or ''}"
    with _STATE_LOCK:
        stats = _CACHE.get("stats") or {}
        payload = stats.get(key)
    if isinstance(payload, dict):
        return dict(payload)
    return None


async def bq_stats_loop() -> None:
    if not _PROJECT:
        logging.info("[BQ_STATS] disabled (BQ project not configured)")
        while True:
            await asyncio.sleep(_REFRESH_SEC)
    loop = asyncio.get_running_loop()
    _load_state()
    while True:
        started = time.time()
        try:
            stats = await loop.run_in_executor(None, refresh_stats)
            logging.info("[BQ_STATS] refreshed stats for %d strategies", len(stats))
        except Exception as exc:
            logging.warning("[BQ_STATS] refresh failed: %s", exc)
        elapsed = time.time() - started
        sleep_for = max(_REFRESH_SEC - elapsed, 60.0)
        await asyncio.sleep(sleep_for)


# load state on import so synchronous callers see cached values
try:
    _load_state()
except Exception:  # pragma: no cover
    pass

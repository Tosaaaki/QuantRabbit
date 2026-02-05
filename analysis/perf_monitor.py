"""
analysis.perf_monitor
~~~~~~~~~~~~~~~~~~~~~
Pocket (micro / macro) ごとに
PF, Sharpe, 勝率、平均 pips を 5 分おきに更新し
SQLite `logs/trades.db` 内に保存。

Hot path note:
- Workers may call snapshot() every loop. Avoid pandas full-table reads; use SQL aggregates + TTL cache.
"""

from __future__ import annotations

import math
import os
import pathlib
import sqlite3
import time
from typing import Any, Dict, Iterable

_DB = pathlib.Path("logs/trades.db")
_DB.parent.mkdir(exist_ok=True)

_DB_TIMEOUT = float(os.getenv("PERF_DB_TIMEOUT_SEC", "8.0"))
_BUSY_TIMEOUT_MS = int(os.getenv("PERF_DB_BUSY_TIMEOUT_MS", str(int(_DB_TIMEOUT * 1000))))
_JOURNAL_MODE = os.getenv("PERF_DB_JOURNAL_MODE", "WAL")
_SYNCHRONOUS = os.getenv("PERF_DB_SYNCHRONOUS", "NORMAL")
_TEMP_STORE = os.getenv("PERF_DB_TEMP_STORE", "MEMORY")
_READONLY_URI = f"file:{_DB}?mode=ro"

_TTL_SEC = max(0.0, float(os.getenv("PERF_MONITOR_TTL_SEC", "300") or 300.0))
_STRATEGY_TTL_SEC = max(
    0.0, float(os.getenv("PERF_MONITOR_STRATEGY_TTL_SEC", str(_TTL_SEC)) or _TTL_SEC)
)

_CACHE_SNAPSHOT: dict[str, Any] = {"ts": 0.0, "data": None}
_CACHE_SNAPSHOT_YEN: dict[str, Any] = {"ts": 0.0, "data": None}
_CACHE_STRATEGY: dict[int, dict[str, Any]] = {}
_CACHE_HOURLY: dict[str, Any] = {"ts": 0.0, "data": None}

_HOURLY_TTL_SEC = max(0.0, float(os.getenv("PERF_HOURLY_TTL_SEC", "600") or 600.0))
_HOURLY_LOOKBACK_DAYS = max(1, int(float(os.getenv("PERF_HOURLY_LOOKBACK_DAYS", "14") or 14)))


def _connect(readonly: bool = False) -> sqlite3.Connection:
    """
    Get SQLite connection.
    - readonly=True は mode=ro で接続し、書き込みロックを避ける。
    - readonly=False はスキーマ初期化用に使用。
    """
    if readonly:
        con = sqlite3.connect(
            _READONLY_URI,
            uri=True,
            timeout=_DB_TIMEOUT,
            isolation_level=None,
        )
    else:
        con = sqlite3.connect(_DB, timeout=_DB_TIMEOUT)
    try:
        con.execute(f"PRAGMA journal_mode={_JOURNAL_MODE}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA synchronous={_SYNCHRONOUS}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA temp_store={_TEMP_STORE}")
    except sqlite3.Error:
        pass
    return con


def _ensure_schema() -> None:
    """trades テーブルが存在しない / 欠損カラムがある場合に補正する。"""
    with _connect(readonly=False) as con:
        con.execute(
            """
    CREATE TABLE IF NOT EXISTS trades (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      transaction_id INTEGER,
      ticket_id TEXT,
      pocket TEXT,
      instrument TEXT,
      units INTEGER,
      entry_price REAL,
      closed_units INTEGER,
      close_price REAL,
      pl_pips REAL,
      realized_pl REAL,
      commission REAL,
      financing REAL,
      entry_time TEXT,
      open_time TEXT,
      close_time TEXT,
      close_reason TEXT,
      state TEXT,
      updated_at TEXT,
      version TEXT DEFAULT 'v1',
      unrealized_pl REAL,
      strategy TEXT,
      macro_regime TEXT,
      micro_regime TEXT,
      client_order_id TEXT,
      strategy_tag TEXT,
      entry_thesis TEXT
    )
    """
        )
        existing = {row[1] for row in con.execute("PRAGMA table_info(trades)")}  # pragma: no cover
        columns: dict[str, str] = {
            "transaction_id": "INTEGER",
            "ticket_id": "TEXT",
            "pocket": "TEXT",
            "instrument": "TEXT",
            "units": "INTEGER",
            "closed_units": "INTEGER",
            "entry_price": "REAL",
            "close_price": "REAL",
            "pl_pips": "REAL",
            "realized_pl": "REAL",
            "commission": "REAL",
            "financing": "REAL",
            "entry_time": "TEXT",
            "open_time": "TEXT",
            "close_time": "TEXT",
            "close_reason": "TEXT",
            "state": "TEXT",
            "updated_at": "TEXT",
            "version": "TEXT DEFAULT 'v1'",
            "unrealized_pl": "REAL",
            "strategy": "TEXT",
            "macro_regime": "TEXT",
            "micro_regime": "TEXT",
            "client_order_id": "TEXT",
            "strategy_tag": "TEXT",
            "entry_thesis": "TEXT",
        }
        for name, ddl in columns.items():
            if name not in existing:
                con.execute(f"ALTER TABLE trades ADD COLUMN {name} {ddl}")
        con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uniq_trades_tx_trade
            ON trades(transaction_id, ticket_id)
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticket ON trades(ticket_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time)")


_ensure_schema()


def _clone_snapshot(src: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {k: dict(v) for k, v in src.items()}


def _clone_rows(src: list[dict]) -> list[dict]:
    return [dict(row) for row in src]


def _clone_hourly(
    src: dict[str, dict[int, dict[str, float]]],
) -> dict[str, dict[int, dict[str, float]]]:
    return {
        pocket: {int(hour): dict(metrics) for hour, metrics in hours.items()}
        for pocket, hours in src.items()
    }


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _query_rows(sql: str, params: Iterable[object] = ()) -> list[dict[str, Any]]:
    if not _DB.exists():
        return []
    with _connect(readonly=True) as con:
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA read_uncommitted=1")
        except sqlite3.Error:
            pass
        rows = con.execute(sql, list(params)).fetchall()
        return [dict(row) for row in rows]


def _calc_sharpe(*, avg: float, avg_sq: float, n: int) -> float:
    if n <= 1:
        return 0.0
    var = avg_sq - (avg * avg)
    if var <= 0:
        return 0.0
    # Sample std (ddof=1) to match pandas default.
    var *= n / max(1.0, float(n - 1))
    sd = math.sqrt(max(0.0, var))
    if sd <= 0:
        return 0.0
    return avg / sd


def _sum_jpy_from_row(row: dict[str, Any]) -> float:
    realized_n = _as_int(row.get("realized_n"), 0)
    realized_sum = _as_float(row.get("realized_sum"), 0.0)
    if realized_n > 0:
        return realized_sum
    sum_abs_closed = _as_float(row.get("sum_abs_closed_units"), 0.0)
    yen_closed = _as_float(row.get("yen_from_closed"), 0.0)
    yen_units = _as_float(row.get("yen_from_units"), 0.0)
    return yen_closed if sum_abs_closed > 0 else yen_units


def _sum_jpy_units_only(row: dict[str, Any]) -> float:
    # USD/JPY: 1 pip = 0.01 JPY per 1 unit. (pips * abs(units) * 0.01)
    return _as_float(row.get("yen_from_units"), 0.0)


def _aggregate_by_pocket() -> list[dict[str, Any]]:
    return _query_rows(
        """
        SELECT
          pocket,
          COUNT(*) AS n_rows,
          COUNT(pl_pips) AS n_pips,
          SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS profit,
          SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS loss,
          SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
          AVG(pl_pips) AS avg_pips,
          AVG(pl_pips * pl_pips) AS avg_sq,
          SUM(pl_pips) AS sum_pips,
          SUM(ABS(COALESCE(closed_units, 0))) AS sum_abs_closed_units,
          SUM(ABS(COALESCE(units, 0))) AS sum_abs_units,
          SUM(pl_pips * ABS(COALESCE(closed_units, 0)) * 0.01) AS yen_from_closed,
          SUM(pl_pips * ABS(COALESCE(units, 0)) * 0.01) AS yen_from_units,
          SUM(CASE WHEN realized_pl IS NOT NULL THEN 1 ELSE 0 END) AS realized_n,
          SUM(COALESCE(realized_pl, 0) + COALESCE(commission, 0) + COALESCE(financing, 0)) AS realized_sum
        FROM trades
        WHERE pocket IS NOT NULL AND pocket != ''
        GROUP BY pocket
        """
    )


def _aggregate_by_strategy() -> list[dict[str, Any]]:
    return _query_rows(
        """
        SELECT
          CASE
            WHEN strategy_tag IS NOT NULL AND strategy_tag != '' THEN strategy_tag
            WHEN strategy IS NOT NULL AND strategy != '' THEN strategy
            WHEN pocket IS NOT NULL AND pocket != '' THEN pocket
            ELSE 'unknown'
          END AS strategy_key,
          COUNT(*) AS n_rows,
          SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
          SUM(CASE WHEN pl_pips < 0 THEN 1 ELSE 0 END) AS losses,
          SUM(pl_pips) AS sum_pips,
          AVG(pl_pips) AS avg_pips,
          SUM(ABS(COALESCE(closed_units, 0))) AS sum_abs_closed_units,
          SUM(ABS(COALESCE(units, 0))) AS sum_abs_units,
          SUM(pl_pips * ABS(COALESCE(closed_units, 0)) * 0.01) AS yen_from_closed,
          SUM(pl_pips * ABS(COALESCE(units, 0)) * 0.01) AS yen_from_units,
          SUM(CASE WHEN realized_pl IS NOT NULL THEN 1 ELSE 0 END) AS realized_n,
          SUM(COALESCE(realized_pl, 0) + COALESCE(commission, 0) + COALESCE(financing, 0)) AS realized_sum
        FROM trades
        GROUP BY 1
        """
    )


def _aggregate_by_hour(lookback_days: int) -> list[dict[str, Any]]:
    return _query_rows(
        """
        SELECT
          pocket,
          strftime('%H', close_time) AS hour,
          COUNT(*) AS n_rows,
          SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS profit,
          SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS loss,
          SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS wins,
          AVG(pl_pips) AS avg_pips
        FROM trades
        WHERE close_time IS NOT NULL
          AND close_time >= datetime('now', ?)
        GROUP BY pocket, hour
        """,
        (f"-{max(1, int(lookback_days))} day",),
    )


def snapshot() -> Dict[str, Dict[str, float]]:
    """
    戻り値:
    {
      'micro': {'pf':1.3,'sharpe':0.9,'win_rate':0.6,'avg_pips':8.2},
      'macro': {...},
      'scalp': {...}
    }
    """
    now = time.monotonic()
    cached_ts = _as_float(_CACHE_SNAPSHOT.get("ts"), 0.0)
    cached = _CACHE_SNAPSHOT.get("data")
    if _TTL_SEC > 0 and isinstance(cached, dict) and (now - cached_ts) < _TTL_SEC:
        return _clone_snapshot(cached)

    rows = _aggregate_by_pocket()
    if not rows:
        payload: dict[str, dict[str, float]] = {"micro": {}, "macro": {}, "scalp": {}}
        _CACHE_SNAPSHOT["ts"] = now
        _CACHE_SNAPSHOT["data"] = payload
        return _clone_snapshot(payload)

    result: dict[str, dict[str, float]] = {}
    for row in rows:
        pocket = str(row.get("pocket") or "").strip()
        if not pocket:
            continue
        n_rows = _as_int(row.get("n_rows"), 0)
        profit = _as_float(row.get("profit"), 0.0)
        loss = _as_float(row.get("loss"), 0.0)
        pf = (profit / loss) if loss else float("inf")
        wins = _as_float(row.get("wins"), 0.0)
        win_rate = (wins / n_rows) if n_rows else 0.0
        avg = _as_float(row.get("avg_pips"), 0.0)
        avg_sq = _as_float(row.get("avg_sq"), 0.0)
        n_pips = _as_int(row.get("n_pips"), 0)
        sharpe = _calc_sharpe(avg=avg, avg_sq=avg_sq, n=n_pips)
        sum_pips = _as_float(row.get("sum_pips"), 0.0)
        sum_jpy = _sum_jpy_from_row(row)
        result[pocket] = {
            "pf": round(pf, 2),
            "sharpe": round(sharpe, 2),
            "win_rate": round(win_rate, 2),
            "avg_pips": round(avg, 2),
            "sum_pips": round(sum_pips, 2),
            "sum_jpy": round(sum_jpy),
            "sample": int(n_rows),
        }
    for key in ("micro", "macro", "scalp"):
        result.setdefault(key, {})

    _CACHE_SNAPSHOT["ts"] = now
    _CACHE_SNAPSHOT["data"] = result
    return _clone_snapshot(result)


def snapshot_strategy(limit: int = 30) -> list[dict]:
    """
    strategy_tag 単位の集計を返す（数量は JPY換算も付与）。
    limit 件を sum_jpy 降順で返却。
    """
    lim = max(1, int(limit))
    now = time.monotonic()
    cached = _CACHE_STRATEGY.get(lim)
    if cached and _STRATEGY_TTL_SEC > 0 and (now - _as_float(cached.get("ts"), 0.0)) < _STRATEGY_TTL_SEC:
        data = cached.get("data")
        if isinstance(data, list):
            return _clone_rows(data)

    rows = _aggregate_by_strategy()
    if not rows:
        _CACHE_STRATEGY[lim] = {"ts": now, "data": []}
        return []

    out: list[dict] = []
    for row in rows:
        strategy = str(row.get("strategy_key") or "unknown")
        trades = _as_int(row.get("n_rows"), 0)
        wins = _as_int(row.get("wins"), 0)
        losses = _as_int(row.get("losses"), 0)
        sum_pips = _as_float(row.get("sum_pips"), 0.0)
        avg_pips = _as_float(row.get("avg_pips"), 0.0)
        sum_jpy = _sum_jpy_from_row(row)
        out.append(
            {
                "strategy": strategy,
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / trades, 3) if trades else 0.0,
                "sum_pips": round(sum_pips, 2),
                "avg_pips": round(avg_pips, 2),
                "sum_jpy": round(sum_jpy),
            }
        )
    out.sort(key=lambda r: float(r.get("sum_jpy") or 0.0), reverse=True)
    out = out[:lim]
    _CACHE_STRATEGY[lim] = {"ts": now, "data": out}
    return _clone_rows(out)


def snapshot_hourly() -> dict[str, dict[int, dict[str, float]]]:
    """
    Hourly pocket snapshot for the recent N days.

    Returns:
    {
      'micro': {0: {'pf':1.1,'win_rate':0.54,'avg_pips':0.8,'sample':24}, ...},
      'macro': {...},
      'scalp': {...}
    }
    """
    now = time.monotonic()
    cached_ts = _as_float(_CACHE_HOURLY.get("ts"), 0.0)
    cached = _CACHE_HOURLY.get("data")
    if _HOURLY_TTL_SEC > 0 and isinstance(cached, dict) and (now - cached_ts) < _HOURLY_TTL_SEC:
        return _clone_hourly(cached)

    rows = _aggregate_by_hour(_HOURLY_LOOKBACK_DAYS)
    if not rows:
        payload: dict[str, dict[int, dict[str, float]]] = {}
        _CACHE_HOURLY["ts"] = now
        _CACHE_HOURLY["data"] = payload
        return _clone_hourly(payload)

    result: dict[str, dict[int, dict[str, float]]] = {}
    for row in rows:
        pocket = str(row.get("pocket") or "").strip()
        if not pocket:
            continue
        hour_raw = row.get("hour")
        try:
            hour = int(hour_raw)
        except Exception:
            continue
        n_rows = _as_int(row.get("n_rows"), 0)
        profit = _as_float(row.get("profit"), 0.0)
        loss = _as_float(row.get("loss"), 0.0)
        pf = (profit / loss) if loss else float("inf")
        wins = _as_float(row.get("wins"), 0.0)
        win_rate = (wins / n_rows) if n_rows else 0.0
        avg = _as_float(row.get("avg_pips"), 0.0)
        result.setdefault(pocket, {})[hour] = {
            "pf": round(pf, 2),
            "win_rate": round(win_rate, 2),
            "avg_pips": round(avg, 2),
            "sample": int(n_rows),
        }

    _CACHE_HOURLY["ts"] = now
    _CACHE_HOURLY["data"] = result
    return _clone_hourly(result)


def snapshot_with_yen() -> dict[str, dict[str, float]]:
    """
    pocket単位のスナップショットに JPY換算の総損益を付与して返す。
    戻り値例:
      {'micro': {'pf':..., 'sum_jpy': 12000.0, ...}, ...}
    """
    now = time.monotonic()
    cached_ts = _as_float(_CACHE_SNAPSHOT_YEN.get("ts"), 0.0)
    cached = _CACHE_SNAPSHOT_YEN.get("data")
    if _TTL_SEC > 0 and isinstance(cached, dict) and (now - cached_ts) < _TTL_SEC:
        return _clone_snapshot(cached)

    rows = _aggregate_by_pocket()
    if not rows:
        payload: dict[str, dict[str, float]] = {"micro": {}, "macro": {}, "scalp": {}}
        _CACHE_SNAPSHOT_YEN["ts"] = now
        _CACHE_SNAPSHOT_YEN["data"] = payload
        return _clone_snapshot(payload)

    result: dict[str, dict[str, float]] = {}
    for row in rows:
        pocket = str(row.get("pocket") or "").strip()
        if not pocket:
            continue
        n_rows = _as_int(row.get("n_rows"), 0)
        profit = _as_float(row.get("profit"), 0.0)
        loss = _as_float(row.get("loss"), 0.0)
        pf = (profit / loss) if loss else float("inf")
        wins = _as_float(row.get("wins"), 0.0)
        win_rate = (wins / n_rows) if n_rows else 0.0
        avg = _as_float(row.get("avg_pips"), 0.0)
        sum_pips = _as_float(row.get("sum_pips"), 0.0)
        sum_jpy = _sum_jpy_units_only(row)
        result[pocket] = {
            "pf": round(pf, 2),
            "win_rate": round(win_rate, 2),
            "avg_pips": round(avg, 2),
            "sum_pips": round(sum_pips, 2),
            "sum_jpy": round(sum_jpy),
        }
    for key in ("micro", "macro", "scalp"):
        result.setdefault(key, {})

    _CACHE_SNAPSHOT_YEN["ts"] = now
    _CACHE_SNAPSHOT_YEN["data"] = result
    return _clone_snapshot(result)


if __name__ == "__main__":
    # 既存 trades.db を破壊せずに集計結果だけを表示する。
    import pprint

    print("Pocket snapshot with JPY:")
    pprint.pp(snapshot_with_yen())
    print("\nStrategy snapshot (top 30 by JPY P/L):")
    pprint.pp(snapshot_strategy())

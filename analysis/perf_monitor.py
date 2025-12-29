"""
analysis.perf_monitor
~~~~~~~~~~~~~~~~~~~~~
Pocket (micro / macro) ごとに
PF, Sharpe, 勝率、平均 pips を 5 分おきに更新し
SQLite `logs/trades.db` 内に保存。
"""

from __future__ import annotations
import os
import sqlite3
import pathlib
import pandas as pd
from typing import Dict

_DB = pathlib.Path("logs/trades.db")
_DB.parent.mkdir(exist_ok=True)

_DB_TIMEOUT = float(os.getenv("PERF_DB_TIMEOUT_SEC", "8.0"))
_BUSY_TIMEOUT_MS = int(os.getenv("PERF_DB_BUSY_TIMEOUT_MS", str(int(_DB_TIMEOUT * 1000))))
_JOURNAL_MODE = os.getenv("PERF_DB_JOURNAL_MODE", "WAL")
_SYNCHRONOUS = os.getenv("PERF_DB_SYNCHRONOUS", "NORMAL")
_TEMP_STORE = os.getenv("PERF_DB_TEMP_STORE", "MEMORY")
_READONLY_URI = f"file:{_DB}?mode=ro"


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
        existing = {
            row[1]
            for row in con.execute("PRAGMA table_info(trades)")  # pragma: no cover
        }
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
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_ticket ON trades(ticket_id)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time)"
        )


_ensure_schema()


def _load_df() -> pd.DataFrame:
    # 読み取り専用接続でロックを避ける
    with _connect(readonly=True) as con:
        try:
            con.execute("PRAGMA read_uncommitted=1")
        except sqlite3.Error:
            pass
        return pd.read_sql_query("SELECT * FROM trades", con, parse_dates=["close_time"])


def snapshot() -> Dict[str, Dict[str, float]]:
    """
    戻り値:
    {
      'micro': {'pf':1.3,'sharpe':0.9,'win_rate':0.6,'avg_pips':8.2},
      'macro': {...},
      'scalp': {...}
    }
    """
    df = _load_df()
    if df.empty:
        return {"micro": {}, "macro": {}, "scalp": {}}

    result: Dict[str, Dict[str, float]] = {}
    for pocket, sub in df.groupby("pocket"):
        profit = sub.loc[sub.pl_pips > 0, "pl_pips"].sum()
        loss = abs(sub.loc[sub.pl_pips < 0, "pl_pips"].sum())
        pf = profit / loss if loss else float("inf")
        win_rate = (sub.pl_pips > 0).mean()
        avg = sub.pl_pips.mean()
        sum_pips = float(sub.pl_pips.sum())
        units = _pick_units(sub)
        realized_jpy = _realized_sum_jpy(sub)
        sum_jpy = realized_jpy if realized_jpy is not None else _yen_from_pips(sub.pl_pips, units)

        # シャープ比：日次換算を簡易に pips/sd
        sd = sub.pl_pips.std()
        sharpe = avg / sd if sd else 0.0

        result[pocket] = {
            "pf": round(pf, 2),
            "sharpe": round(sharpe, 2),
            "win_rate": round(win_rate, 2),
            "avg_pips": round(avg, 2),
            "sum_pips": round(sum_pips, 2),
            "sum_jpy": round(sum_jpy),
        }
    for key in ("micro", "macro", "scalp"):
        result.setdefault(key, {})
    return result


def _pick_units(sub: pd.DataFrame) -> pd.Series:
    """
    単位選択: closed_units があれば優先し、無ければ units。
    """
    if "closed_units" in sub.columns:
        ser = sub["closed_units"].fillna(0)
        if ser.abs().sum() > 0:
            return ser
    if "units" in sub.columns:
        return sub["units"].fillna(0)
    return pd.Series(0, index=sub.index)


def _realized_sum_jpy(sub: pd.DataFrame) -> float | None:
    """realized_pl + commission + financing を合算。列が無ければ None を返す。"""
    if "realized_pl" not in sub.columns:
        return None
    if not sub["realized_pl"].notna().any():
        return None
    realized = sub["realized_pl"].fillna(0)
    commission = sub["commission"].fillna(0) if "commission" in sub.columns else 0
    financing = sub["financing"].fillna(0) if "financing" in sub.columns else 0
    try:
        return float((realized + commission + financing).sum())
    except Exception:
        return float(realized.sum())


def _yen_from_pips(pl_pips: pd.Series, units: pd.Series) -> float:
    """
    JPY換算の損益を計算する。
    USD/JPY は 1pip=0.01 JPY、pip価値=units*0.01。
    """
    try:
        return float((pl_pips.astype(float) * units.abs().astype(float) * 0.01).sum())
    except Exception:
        return 0.0


def snapshot_strategy(limit: int = 30) -> list[dict]:
    """
    strategy_tag 単位の集計を返す（数量は JPY換算も付与）。
    limit 件を sum_jpy 降順で返却。
    """
    df = _load_df()
    if df.empty:
        return []
    if "units" not in df.columns:
        df["units"] = 0
    tag = None
    for key in ("strategy_tag", "strategy", "pocket"):
        if key in df.columns:
            tag = key
            break
    if tag is None:
        return []

    rows = []
    for strat, sub in df.groupby(tag):
        trades = len(sub)
        wins = int((sub.pl_pips > 0).sum())
        losses = int((sub.pl_pips < 0).sum())
        sum_pips = float(sub.pl_pips.sum())
        avg_pips = float(sub.pl_pips.mean())
        units = _pick_units(sub)
        realized_jpy = _realized_sum_jpy(sub)
        sum_jpy = realized_jpy if realized_jpy is not None else _yen_from_pips(sub.pl_pips, units)
        rows.append(
            {
                "strategy": str(strat),
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / trades, 3) if trades else 0.0,
                "sum_pips": round(sum_pips, 2),
                "avg_pips": round(avg_pips, 2),
                "sum_jpy": round(sum_jpy),
            }
        )
    rows.sort(key=lambda r: r["sum_jpy"], reverse=True)
    return rows[:limit]


def snapshot_with_yen() -> dict[str, dict[str, float]]:
    """
    pocket単位のスナップショットに JPY換算の総損益を付与して返す。
    戻り値例:
      {'micro': {'pf':..., 'sum_jpy': 12000.0, ...}, ...}
    """
    df = _load_df()
    if df.empty:
        return {"micro": {}, "macro": {}, "scalp": {}}
    if "units" not in df.columns:
        df["units"] = 0

    result: Dict[str, Dict[str, float]] = {}
    for pocket, sub in df.groupby("pocket"):
        profit = sub.loc[sub.pl_pips > 0, "pl_pips"].sum()
        loss = abs(sub.loc[sub.pl_pips < 0, "pl_pips"].sum())
        pf = profit / loss if loss else float("inf")
        win_rate = (sub.pl_pips > 0).mean()
        avg = sub.pl_pips.mean()
        sum_jpy = _yen_from_pips(sub.pl_pips, sub.units)

        result[pocket] = {
            "pf": round(pf, 2),
            "win_rate": round(win_rate, 2),
            "avg_pips": round(avg, 2),
            "sum_pips": round(float(sub.pl_pips.sum()), 2),
            "sum_jpy": round(sum_jpy),
        }
    for key in ("micro", "macro", "scalp"):
        result.setdefault(key, {})
    return result


if __name__ == "__main__":
    # 既存 trades.db を破壊せずに集計結果だけを表示する。
    import pprint

    print("Pocket snapshot with JPY:")
    pprint.pp(snapshot_with_yen())
    print("\nStrategy snapshot (top 30 by JPY P/L):")
    pprint.pp(snapshot_strategy())

"""
analysis.perf_monitor
~~~~~~~~~~~~~~~~~~~~~
Pocket (micro / macro) ごとに
PF, Sharpe, 勝率、平均 pips を 5 分おきに更新し
SQLite `logs/trades.db` 内に保存。
"""

from __future__ import annotations
import math
import sqlite3
import pathlib
import pandas as pd
from typing import Dict

_DB = pathlib.Path("logs/trades.db")
_DB.parent.mkdir(exist_ok=True)
con = sqlite3.connect(_DB)

# 統一スキーマ: position_manager に合わせる（id は OANDA の transaction ID）
con.execute(
    """
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY,
  ticket_id TEXT UNIQUE,
  pocket TEXT,
  instrument TEXT,
  units INTEGER,
  entry_price REAL,
  close_price REAL,
  pl_pips REAL,
  entry_time TEXT,
  close_time TEXT,
  strategy TEXT,
  macro_regime TEXT,
  micro_regime TEXT
)
"""
)
con.commit()

# 既存テーブルの互換マイグレーション（不足列の追加）
try:
    cols = {row[1] for row in con.execute("PRAGMA table_info(trades)").fetchall()}
    for missing in ("strategy", "macro_regime", "micro_regime"):
        if missing not in cols:
            con.execute(f"ALTER TABLE trades ADD COLUMN {missing} TEXT")
    con.commit()
except Exception:
    pass


def _load_df() -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM trades", con, parse_dates=["close_time"])


def snapshot() -> Dict[str, Dict[str, float]]:
    """
    戻り値:
    {
      'micro': {'pf':1.3,'sharpe':0.9,'win_rate':0.6,'avg_pips':8.2},
      'macro': {...}
    }
    """
    df = _load_df()
    if df.empty:
        return {"micro": {}, "macro": {}}

    result: Dict[str, Dict[str, float]] = {}
    for pocket, sub in df.groupby("pocket"):
        profit = sub.loc[sub.pl_pips > 0, "pl_pips"].sum()
        loss = abs(sub.loc[sub.pl_pips < 0, "pl_pips"].sum())
        pf = profit / loss if loss else float("inf")
        win_rate = (sub.pl_pips > 0).mean()
        avg = sub.pl_pips.mean()

        # シャープ比：日次換算を簡易に pips/sd
        sd = sub.pl_pips.std(ddof=0)
        if sd and not math.isnan(sd):
            sharpe = avg / sd
        else:
            sharpe = 0.0

        result[pocket] = {
            "pf": round(pf, 2),
            "sharpe": round(sharpe, 2),
            "win_rate": round(win_rate, 2),
            "avg_pips": round(avg, 2),
        }
    return result


if __name__ == "__main__":
    # ダミー行でテスト
    con.execute("DELETE FROM trades")
    con.executemany(
        "INSERT INTO trades(pocket,open_time,close_time,pl_pips) VALUES (?,?,?,?)",
        [
            ("micro", "2025-06-23T12:00", "2025-06-23T12:05", 8.5),
            ("micro", "2025-06-23T12:10", "2025-06-23T12:15", -6.0),
            ("macro", "2025-06-23T12:00", "2025-06-23T13:30", 42.0),
        ],
    )
    con.commit()
    import pprint

    pprint.pp(snapshot())

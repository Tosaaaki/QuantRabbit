#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sqlite3
from typing import Optional

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required for gpt_exit_report.py") from exc

DB_PATH = pathlib.Path("logs/trades.db")


def load_data(days: int, version: Optional[str]) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError("logs/trades.db not found")
    con = sqlite3.connect(DB_PATH)
    try:
        tables = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "exit_advice_events" not in tables or "trades" not in tables:
            return pd.DataFrame()
        df = pd.read_sql_query(
            """
            SELECT e.*, t.pl_pips AS realized_pips, t.entry_time, t.close_time,
                   t.strategy AS trade_strategy, t.pocket AS trade_pocket
            FROM exit_advice_events e
            LEFT JOIN trades t ON t.ticket_id = e.trade_id
            """,
            con,
            parse_dates=["recorded_at", "entry_time", "close_time"],
        )
    finally:
        con.close()
    for col in ("recorded_at", "entry_time", "close_time"):
        if col in df.columns and col in df:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(None)
    if days > 0:
        cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).tz_localize(None)
        df = df[df["recorded_at"] >= cutoff]
    if version:
        df = df[df["version"].str.upper() == version.upper()]
    return df


def summarize(df: pd.DataFrame) -> str:
    if df.empty:
        return "No exit advisor events for the specified window."
    lines = []
    lines.append(f"Events: {len(df)} (from {df['recorded_at'].min()} to {df['recorded_at'].max()})")
    grouped = df.groupby("event_type").size().sort_values(ascending=False)
    lines.append("By event type:")
    for ev, count in grouped.items():
        lines.append(f"- {ev}: {count}")

    # close effectiveness
    close_df = df[df["event_type"] == "close_request"].dropna(subset=["realized_pips"])
    if not close_df.empty:
        win = (close_df["realized_pips"] > 0).mean()
        avg = close_df["realized_pips"].mean()
        lines.append(
            f"Close outcomes: Win={win:.2%}, AvgPnL={avg:.2f}p over {len(close_df)} closes with realized PnL"
        )
    adjust_df = df[df["event_type"] == "adjust"]
    if not adjust_df.empty:
        adj_stats = adjust_df.groupby("action").agg(
            n=("id", "count"),
            avg_conf=("confidence", "mean"),
        )
        lines.append("Adjust events:")
        for idx, row in adj_stats.iterrows():
            lines.append(f"- {idx}: {row['n']} events, avg conf={row['avg_conf']:.2f}")

    conf_stats = df.groupby("event_type")["confidence"].mean()
    lines.append("Average confidence by event:")
    for ev, conf in conf_stats.items():
        lines.append(f"- {ev}: {conf:.2f}")

    strat = df.groupby("strategy").agg(count=("id", "count"), avg_conf=("confidence", "mean"))
    lines.append("Top strategies by GPT advice volume:")
    for idx, row in strat.sort_values("count", ascending=False).head(5).iterrows():
        lines.append(f"- {idx or '(unknown)'}: {row['count']} events, conf={row['avg_conf']:.2f}")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize GPT exit advisor impact")
    ap.add_argument("--days", type=int, default=3, help="Lookback window in days")
    ap.add_argument("--version", type=str, default=None, help="Filter by trade version (e.g., V1/V2)")
    args = ap.parse_args()

    df = load_data(args.days, args.version)
    print(summarize(df))


if __name__ == "__main__":
    main()

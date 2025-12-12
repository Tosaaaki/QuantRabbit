#!/usr/bin/env python3
"""
Export recent trade outcomes to a tuner-friendly CSV.

This is intended for online tuner shadow runs. It pulls from logs/trades.db and
emits a flat file with the columns expected by autotune/online_tuner.py:
timestamp, reason, pips, strategy, regime, hazard_ticks, events, grace_used_ms,
scratch_hits.
"""

import argparse
import csv
import os
import sqlite3
from typing import Iterable, Optional, Tuple


def fetch_trades(db_path: str, limit: Optional[int] = None) -> Iterable[Tuple]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        sql = """
        SELECT close_time, close_reason, pl_pips, strategy, pocket
        FROM trades
        WHERE close_time IS NOT NULL
        ORDER BY close_time DESC
        """
        if limit:
            sql += " LIMIT ?"
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)
        for row in cur.fetchall():
            yield row
    finally:
        conn.close()


def export_csv(rows: Iterable[Tuple], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "reason",
                "pips",
                "strategy",
                "regime",
                "hazard_ticks",
                "events",
                "grace_used_ms",
                "scratch_hits",
            ]
        )
        for ts, reason, pips, strategy, regime in rows:
            writer.writerow(
                [
                    ts,
                    reason or "",
                    pips if pips is not None else 0.0,
                    strategy or "",
                    regime or "",
                    0,
                    0,
                    0,
                    0,
                ]
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        default="logs/trades.db",
        help="Path to trades.db (default: logs/trades.db)",
    )
    ap.add_argument(
        "--out",
        default="tmp/exit_eval_live.csv",
        help="Output CSV path (default: tmp/exit_eval_live.csv)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit (most recent).",
    )
    args = ap.parse_args()

    rows = list(fetch_trades(args.db, limit=args.limit))
    if not rows:
        print(f"[export_exit_eval] no rows found in {args.db}")
        return
    export_csv(rows, args.out)
    print(f"[export_exit_eval] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="logs/trades.db")
    ap.add_argument(
        "--hours",
        type=float,
        default=15.0,
        help="lookback hours (default: 15 for JST last 18:00)",
    )
    ap.add_argument(
        "--bucket",
        choices=("strategy", "hold_time"),
        default="strategy",
        help="aggregation mode (default: strategy)",
    )
    ns = ap.parse_args()

    con = sqlite3.connect(ns.trades)
    con.row_factory = sqlite3.Row
    now_utc = dt.datetime.now(dt.timezone.utc)
    threshold = now_utc - dt.timedelta(hours=ns.hours)

    if ns.bucket == "hold_time":
        _report_hold_time(con, threshold)
    else:
        _report_by_strategy(con, threshold)
    con.close()


def _report_by_strategy(con: sqlite3.Connection, threshold: dt.datetime) -> None:
    rows = con.execute(
        """
        SELECT client_order_id, strategy_tag, pl_pips, realized_pl, close_time, updated_at
        FROM trades WHERE state='CLOSED'
        """
    ).fetchall()
    UTC = dt.timezone.utc
    agg = defaultdict(lambda: {"count": 0, "pips": 0.0, "jpy": 0.0})
    for r in rows:
        ts_raw = (r["close_time"] or r["updated_at"] or "").replace("Z", "+00:00")
        try:
            t = dt.datetime.fromisoformat(ts_raw)
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
        except Exception:
            continue
        if t < threshold:
            continue
        tag = (r["strategy_tag"] or "").strip().lower() or "unknown"
        agg[tag]["count"] += 1
        agg[tag]["pips"] += float(r["pl_pips"] or 0.0)
        agg[tag]["jpy"] += float(r["realized_pl"] or 0.0)

    for tag, stats in sorted(agg.items(), key=lambda kv: kv[1]["pips"]):
        print(
            f"{tag}\tcount={stats['count']}\tpips={stats['pips']:.2f}\tjpy={stats['jpy']:.2f}"
        )


def _report_hold_time(con: sqlite3.Connection, threshold: dt.datetime) -> None:
    threshold_iso = threshold.isoformat().replace("+00:00", "Z")
    rows = con.execute(
        """
        WITH stats AS (
          SELECT
            (strftime('%s', close_time) - strftime('%s', entry_time)) AS hold_sec,
            pl_pips
          FROM trades
          WHERE state='CLOSED'
            AND entry_time IS NOT NULL
            AND close_time IS NOT NULL
            AND close_time >= ?
        )
        SELECT
          CASE
            WHEN hold_sec < 60 THEN '<60s'
            WHEN hold_sec < 180 THEN '1-3m'
            WHEN hold_sec < 600 THEN '3-10m'
            WHEN hold_sec < 1800 THEN '10-30m'
            ELSE '30m+'
          END AS bucket,
          COUNT(*) AS cnt,
          ROUND(SUM(pl_pips), 2) AS sum_pips,
          ROUND(AVG(pl_pips), 3) AS avg_pips
        FROM stats
        GROUP BY bucket
        ORDER BY
          CASE bucket
            WHEN '<60s' THEN 0
            WHEN '1-3m' THEN 1
            WHEN '3-10m' THEN 2
            WHEN '10-30m' THEN 3
            ELSE 4
          END
        """
    , (threshold_iso,)).fetchall()
    total = sum(r["cnt"] for r in rows)
    lt60 = next((r["cnt"] for r in rows if r["bucket"] == "<60s"), 0)
    ratio = (lt60 / total) if total else 0.0
    print(f"Hold <60s ratio: {lt60}/{total} ({ratio:.1%})")
    for r in rows:
        print(
            f"{r['bucket']}\tcount={r['cnt']}\tpips={r['sum_pips']}\tavg={r['avg_pips']}"
        )


if __name__ == "__main__":
    main()

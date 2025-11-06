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
    ap.add_argument("--hours", type=float, default=15.0, help="lookback hours (default: 15 for JST last 18:00)")
    ns = ap.parse_args()

    con = sqlite3.connect(ns.trades)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT client_order_id, strategy_tag, pl_pips, realized_pl, close_time, updated_at "
        "FROM trades WHERE state='CLOSED'"
    ).fetchall()
    con.close()

    now_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    threshold = now_utc - dt.timedelta(hours=ns.hours)
    UTC = dt.timezone.utc

    agg = defaultdict(lambda: {"count": 0, "pips": 0.0, "jpy": 0.0})
    for r in rows:
        ts = (r["close_time"] or r["updated_at"] or "").replace("Z", "+00:00")
        try:
            t = dt.datetime.fromisoformat(ts)
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

    for tag, s in sorted(agg.items(), key=lambda kv: kv[1]["pips"]):
        print(f"{tag}\tcount={s['count']}\tpips={s['pips']:.2f}\tjpy={s['jpy']:.2f}")


if __name__ == "__main__":
    main()


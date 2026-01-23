#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill trades.strategy_tag and trades.client_order_id from logs/orders.db.

It scans trades without a strategy_tag (or missing client_order_id), looks up the
corresponding order in orders.db via ticket_id, finds the earliest submit_attempt
payload for that client_order_id, and extracts entry_thesis.strategy_tag.

Usage:
  python scripts/backfill_strategy_tags.py

Idempotent: safe to run multiple times.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _classify_from_client_id(client_id: str | None) -> str | None:
    if not client_id:
        return None
    cid = client_id.strip()
    mapping = (
        ("qr-fast-", "fast_scalp"),
        ("qr-mirror-tight-", "mirror_spike_tight"),
        ("qr-mirror-s5-", "mirror_spike_s5"),
        ("qr-mirror-", "mirror_spike"),
        ("qr-impulse-s5-", "impulse_break_s5"),
        ("qr-imp-retest-", "impulse_retest_s5"),
        ("qr-imp-momo-", "impulse_momentum_s5"),
        ("qr-sqz-s5-", "squeeze_break_s5"),
        ("qr-pullrun-s5-", "pullback_runner_s5"),
        ("qr-pullback-s5-", "pullback_s5"),
        ("qr-pullback-", "pullback_scalp"),
        ("qr-vwap-s5-", "vwap_magnet_s5"),
    )
    for pref, tag in mapping:
        if cid.startswith(pref):
            return tag
    if "TrendMA" in cid:
        return "main.TrendMA"
    if "Donchian" in cid:
        return "main.Donchian55"
    if "BBRSI" in cid or "BB_RSI" in cid:
        return "main.BB_RSI"
    return None

TRADES_DB = Path("logs/trades.db")
ORDERS_DB = Path("logs/orders.db")


def ensure_columns(con: sqlite3.Connection) -> None:
    cur = con.execute("PRAGMA table_info(trades)")
    cols = {row[1] for row in cur.fetchall()}
    needed = {
        "client_order_id": "TEXT",
        "strategy_tag": "TEXT",
        "entry_thesis": "TEXT",
    }
    for name, ddl in needed.items():
        if name not in cols:
            con.execute(f"ALTER TABLE trades ADD COLUMN {name} {ddl}")
    con.commit()


def backfill() -> int:
    if not TRADES_DB.exists() or not ORDERS_DB.exists():
        print("trades.db or orders.db not found under logs/")
        return 0

    tcon = sqlite3.connect(TRADES_DB)
    tcon.row_factory = sqlite3.Row
    ensure_columns(tcon)

    # Pull candidates without strategy_tag
    rows = tcon.execute(
        """
        SELECT DISTINCT ticket_id
        FROM trades
        WHERE strategy_tag IS NULL OR strategy_tag = ''
        """
    ).fetchall()
    if not rows:
        print("No trades require backfill (strategy_tag already present).")
        return 0
    ticket_ids = [str(r["ticket_id"]) for r in rows if r["ticket_id"]]

    ocon = sqlite3.connect(ORDERS_DB)
    ocon.row_factory = sqlite3.Row

    updated = 0
    for tid in ticket_ids:
        filled = ocon.execute(
            """
            SELECT client_order_id
            FROM orders
            WHERE ticket_id = ? AND status = 'filled'
            ORDER BY id ASC
            LIMIT 1
            """,
            (tid,),
        ).fetchone()
        if not filled or not filled["client_order_id"]:
            continue
        client_id = filled["client_order_id"]
        attempt = ocon.execute(
            """
            SELECT request_json
            FROM orders
            WHERE client_order_id = ? AND status = 'submit_attempt'
            ORDER BY id ASC
            LIMIT 1
            """,
            (client_id,),
        ).fetchone()
        strategy_tag = None
        thesis_json = None
        if attempt and attempt["request_json"]:
            try:
                payload = json.loads(attempt["request_json"]) or {}
                thesis = payload.get("entry_thesis") or {}
                if isinstance(thesis, dict):
                    strategy_tag = thesis.get("strategy_tag")
                    thesis_json = json.dumps(thesis, ensure_ascii=False)
            except Exception:
                pass
        # Heuristic fallback from client_id if thesis is missing
        if not strategy_tag:
            strategy_tag = _classify_from_client_id(client_id)

        tcon.execute(
            """
            UPDATE trades
            SET client_order_id = COALESCE(?, client_order_id),
                strategy_tag = COALESCE(?, strategy_tag),
                entry_thesis = COALESCE(?, entry_thesis)
            WHERE ticket_id = ?
            """,
            (client_id, strategy_tag, thesis_json, tid),
        )
        updated += tcon.total_changes

    tcon.commit()
    tcon.close()
    ocon.close()
    return updated


if __name__ == "__main__":
    changed = backfill()
    print(f"Backfill updated rows: {changed}")

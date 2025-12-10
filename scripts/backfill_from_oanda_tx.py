#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill trades.client_order_id and trades.strategy_tag from OANDA transactions_*.jsonl.

This scans logs/oanda/transactions_*.jsonl for ORDER_FILL entries, extracts
clientOrderID and maps it to OANDA tradeID (both newly opened and closed).
Then updates logs/trades.db with the client_order_id and a best-effort
strategy_tag derived from the client_id prefix (e.g., qr-fast- => fast_scalp,
qr-pullback-s5- => pullback_s5, etc.) or main loop tags embedded in the id.

Usage:
  python scripts/backfill_from_oanda_tx.py

Safe to run multiple times; only fills missing fields.
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, Optional


TRADES_DB = Path("logs/trades.db")
TX_GLOB = "logs/oanda/transactions_*.jsonl"


def classify_from_client_id(client_id: Optional[str]) -> Optional[str]:
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
        ("qr-onepip-s1-", "onepip_maker_s1"),
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


def build_tid_to_client_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in sorted(glob.glob(TX_GLOB)):
        try:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    if not line or line[0] != "{":
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("type") != "ORDER_FILL":
                        continue
                    cid = obj.get("clientOrderID") or obj.get("clientExtensions", {}).get("id")
                    if not cid:
                        continue
                    # open
                    opened = obj.get("tradeOpened") or {}
                    tid = opened.get("tradeID")
                    if tid and tid not in mapping:
                        mapping[str(tid)] = cid
                    # closed (useful when opening log is missing)
                    for closed in obj.get("tradesClosed") or []:
                        ct = closed.get("tradeID")
                        if ct and ct not in mapping:
                            mapping[str(ct)] = cid
        except FileNotFoundError:
            continue
    return mapping


def backfill_from_oanda() -> int:
    if not TRADES_DB.exists():
        print("trades.db not found under logs/")
        return 0
    mapping = build_tid_to_client_map()
    if not mapping:
        print("No clientOrderID mapping found in logs/oanda/transactions_*.jsonl")
        return 0
    con = sqlite3.connect(TRADES_DB)
    con.row_factory = sqlite3.Row
    updated = 0
    for tid, cid in mapping.items():
        tag = classify_from_client_id(cid)
        cur = con.execute(
            """
            UPDATE trades
            SET client_order_id = COALESCE(?, client_order_id),
                strategy_tag = COALESCE(?, strategy_tag)
            WHERE ticket_id = ?
              AND (client_order_id IS NULL OR TRIM(client_order_id) = '')
            """,
            (cid, tag, tid),
        )
        updated += cur.rowcount or 0
    con.commit()
    con.close()
    return updated


if __name__ == "__main__":
    count = backfill_from_oanda()
    print(f"OANDA backfill updated rows: {count}")

#!/usr/bin/env python3
"""Fetch OANDA transactions and rebuild local trades.db records.

Usage:
  python scripts/resync_trades_from_oanda.py --from 2025-10-15T00:00:00Z --to 2025-10-16T00:00:00Z
  python scripts/resync_trades_from_oanda.py --days 1  # fetch from now-1day to now
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

from utils.secrets import get_secret

DB_PATH = Path("logs/trades.db")
PIP_JPY = 0.01
PIP_DEFAULT = 0.0001


def parse_time(ts: str | None) -> datetime | None:
    if not ts:
        return None
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    if "+" in ts[19:] and "." in ts:
        main, rest = ts.split(".", 1)
        frac, zone = rest.split("+", 1)
        ts = f"{main}.{frac[:6]}+{zone}"
    elif "-" in ts[19:] and "." in ts:
        main, rest = ts.split(".", 1)
        frac, zone = rest.split("-", 1)
        ts = f"{main}.{frac[:6]}-{zone}"
    return datetime.fromisoformat(ts)


def format_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_meta(ext: Dict[str, Any] | None) -> Dict[str, Any]:
    if not ext:
        return {}
    pocket = None
    tag = ext.get("tag") or ""
    if "=" in tag:
        _, value = tag.split("=", 1)
        pocket = value.strip() or None
    strategy = macro = micro = None
    comment = ext.get("comment") or ""
    for part in comment.split("|"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        key = k.strip().lower()
        value = v.strip()
        if key == "strategy":
            strategy = value
        elif key == "macro":
            macro = value
        elif key == "micro":
            micro = value
    meta = {}
    if pocket:
        meta["pocket"] = pocket
    if strategy:
        meta["strategy"] = strategy
    if macro:
        meta["macro_regime"] = macro
    if micro:
        meta["micro_regime"] = micro
    return meta


def fetch_transactions(start: datetime, end: datetime) -> List[Dict[str, Any]]:
    token = get_secret("oanda_token")
    account = get_secret("oanda_account_id")
    try:
        practice = get_secret("oanda_practice").lower() == "true"
    except Exception:
        practice = True
    host = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"

    params = {
        "from": format_iso(start),
        "to": format_iso(end),
        "pageSize": 1000,
    }
    headers = {"Authorization": f"Bearer {token}"}

    initial = requests.get(
        f"{host}/v3/accounts/{account}/transactions",
        params=params,
        headers=headers,
        timeout=10,
    )
    initial.raise_for_status()
    data = initial.json()
    transactions = data.get("transactions", []) or []
    pages = data.get("pages", []) or []

    for url in pages:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        transactions.extend(resp.json().get("transactions", []) or [])

    transactions.sort(key=lambda x: int(x.get("id", 0)))
    return transactions


def compute_pips(entry_price: float, close_price: float, units: int, instrument: str) -> float:
    pip = PIP_JPY if instrument.endswith("_JPY") else PIP_DEFAULT
    direction = 1 if units > 0 else -1
    return (close_price - entry_price) / pip * direction


def ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    return con


def upsert_trades(con: sqlite3.Connection, rows: Iterable[Dict[str, Any]]) -> None:
    sql = (
        "INSERT INTO trades (pocket, open_time, close_time, pl_pips, ticket_id, instrument, units, "
        "entry_price, close_price, entry_time, strategy, macro_regime, micro_regime, realized_pl, state, "
        "close_reason, unrealized_pl, updated_at, version) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(ticket_id) DO UPDATE SET "
        "pocket=excluded.pocket, open_time=excluded.open_time, close_time=excluded.close_time, "
        "pl_pips=excluded.pl_pips, instrument=excluded.instrument, units=excluded.units, "
        "entry_price=excluded.entry_price, close_price=excluded.close_price, entry_time=excluded.entry_time, "
        "strategy=excluded.strategy, macro_regime=excluded.macro_regime, micro_regime=excluded.micro_regime, "
        "realized_pl=excluded.realized_pl, state=excluded.state, close_reason=excluded.close_reason, "
        "unrealized_pl=excluded.unrealized_pl, updated_at=excluded.updated_at, version=excluded.version"
    )
    now_iso = format_iso(datetime.utcnow())
    batch = []
    for row in rows:
        batch.append(
            (
                row.get("pocket"),
                row.get("open_time"),
                row.get("close_time"),
                row.get("pl_pips"),
                row.get("ticket_id"),
                row.get("instrument"),
                row.get("units"),
                row.get("entry_price"),
                row.get("close_price"),
                row.get("entry_time"),
                row.get("strategy"),
                row.get("macro_regime"),
                row.get("micro_regime"),
                row.get("realized_pl"),
                row.get("state"),
                row.get("close_reason"),
                row.get("unrealized_pl", 0.0),
                now_iso,
                row.get("version", "V2"),
            )
        )
    if batch:
        con.executemany(sql, batch)
        con.commit()


def build_trade_rows(transactions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    meta_by_order: Dict[str, Dict[str, Any]] = {}
    trade_entries: Dict[str, Dict[str, Any]] = {}
    closed_rows: List[Dict[str, Any]] = []

    for tx in transactions:
        tx_type = tx.get("type")
        time_iso = tx.get("time")
        instrument = tx.get("instrument") or "USD_JPY"
        if tx_type == "MARKET_ORDER":
            meta = parse_meta(tx.get("clientExtensions"))
            stop_price = (tx.get("stopLossOnFill") or {}).get("price")
            tp_price = (tx.get("takeProfitOnFill") or {}).get("price")
            if stop_price:
                meta["order_stop"] = float(stop_price)
            if tp_price:
                meta["order_tp"] = float(tp_price)
            for key in filter(None, [tx.get("id"), tx.get("batchID")]):
                meta_by_order[str(key)] = meta
            continue

        if tx_type != "ORDER_FILL":
            continue

        order_id = tx.get("orderID") or tx.get("id")
        meta = meta_by_order.get(str(order_id)) or meta_by_order.get(str(tx.get("batchID")), {})

        def register_open(item: Dict[str, Any]) -> None:
            trade_id = str(item.get("tradeID") or item.get("id") or "")
            if not trade_id:
                return
            entry_price = float(item.get("price", tx.get("price", 0.0)))
            units = int(float(item.get("units", 0)))
            entry_time = parse_time(time_iso)
            trade_entries[trade_id] = {
                "trade_id": trade_id,
                "entry_price": entry_price,
                "units": units,
                "entry_time": entry_time,
                "instrument": instrument,
                "meta": dict(meta),
            }

        opened_single = tx.get("tradeOpened")
        if opened_single:
            register_open(opened_single)
        for item in tx.get("tradesOpened") or []:
            register_open(item)

        closed_items = list(tx.get("tradesClosed") or []) + list(tx.get("tradesReduced") or [])
        for item in closed_items:
            trade_id = str(item.get("tradeID") or "")
            if not trade_id:
                continue
            entry = trade_entries.get(trade_id)
            if entry is None:
                continue
            entry_price = entry["entry_price"]
            entry_units = entry["units"]
            close_price = float(item.get("price", tx.get("price", entry_price)))
            realized_pl = float(item.get("realizedPL", tx.get("pl", 0.0)))
            close_time = parse_time(time_iso)
            pl_pips = compute_pips(entry_price, close_price, entry_units, instrument)
            row_meta = dict(entry.get("meta") or {})
            pocket = row_meta.get("pocket") or ("macro" if abs(entry_units) >= 100000 else "micro")
            closed_rows.append(
                {
                    "ticket_id": trade_id,
                    "pocket": pocket,
                    "instrument": instrument,
                    "units": entry_units,
                    "entry_price": entry_price,
                    "close_price": close_price,
                    "entry_time": format_iso(entry["entry_time"]),
                    "open_time": format_iso(entry["entry_time"]),
                    "close_time": format_iso(close_time),
                    "pl_pips": pl_pips,
                    "realized_pl": realized_pl,
                    "state": "CLOSED",
                    "close_reason": tx.get("reason"),
                    "strategy": row_meta.get("strategy"),
                    "macro_regime": row_meta.get("macro_regime"),
                    "micro_regime": row_meta.get("micro_regime"),
                    "version": row_meta.get("version", "V2"),
                    "unrealized_pl": 0.0,
                }
            )
            units_closed = int(float(item.get("units", entry_units)))
            if entry_units < 0 and units_closed > 0:
                units_closed = -units_closed
            if entry_units > 0 and units_closed < 0:
                units_closed = -units_closed
            remaining = entry_units - units_closed
            if entry_units > 0 and remaining < 0:
                remaining = 0
            if entry_units < 0 and remaining > 0:
                remaining = 0
            if remaining == 0:
                trade_entries.pop(trade_id, None)
            else:
                entry["units"] = remaining
                trade_entries[trade_id] = entry

    open_rows: List[Dict[str, Any]] = []
    for entry in trade_entries.values():
        row_meta = dict(entry.get("meta") or {})
        entry_units = entry["units"]
        instrument = entry["instrument"]
        pocket = row_meta.get("pocket") or ("macro" if abs(entry_units) >= 100000 else "micro")
        open_rows.append(
            {
                "ticket_id": entry["trade_id"],
                "pocket": pocket,
                "instrument": instrument,
                "units": entry_units,
                "entry_price": entry["entry_price"],
                "close_price": None,
                "entry_time": format_iso(entry["entry_time"]),
                "open_time": format_iso(entry["entry_time"]),
                "close_time": None,
                "pl_pips": None,
                "realized_pl": None,
                "state": "OPEN",
                "close_reason": None,
                "strategy": row_meta.get("strategy"),
                "macro_regime": row_meta.get("macro_regime"),
                "micro_regime": row_meta.get("micro_regime"),
                "version": row_meta.get("version", "V2"),
                "unrealized_pl": None,
            }
        )

    return closed_rows, open_rows


def apply_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_ticket: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        ticket = row.get("ticket_id")
        if not ticket:
            continue
        by_ticket[ticket] = row
    return list(by_ticket.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill OANDA transactions into trades.db")
    parser.add_argument("--from", dest="from_ts", help="ISO timestamp (UTC) inclusive")
    parser.add_argument("--to", dest="to_ts", help="ISO timestamp (UTC) exclusive")
    parser.add_argument("--days", type=int, default=2, help="Lookback window in days if --from not set")
    args = parser.parse_args()

    if args.from_ts:
        start = parse_time(args.from_ts)
    else:
        start = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=args.days)
    if start is None:
        raise SystemExit("Invalid --from timestamp")

    if args.to_ts:
        end = parse_time(args.to_ts)
    else:
        end = datetime.utcnow().replace(tzinfo=timezone.utc)
    if end is None:
        raise SystemExit("Invalid --to timestamp")

    transactions = fetch_transactions(start, end)
    closed_rows, open_rows = build_trade_rows(transactions)

    closed_rows = apply_rows(closed_rows)
    open_rows = apply_rows(open_rows)

    con = ensure_db()
    upsert_trades(con, closed_rows + open_rows)
    print(f"synced closed={len(closed_rows)} open={len(open_rows)} into {DB_PATH}")


if __name__ == "__main__":
    main()

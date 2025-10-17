#!/usr/bin/env python3
"""
Summarize recent trades from Firestore and optionally sync into local SQLite (logs/trades.db).

Usage:
  python scripts/trades_report.py               # print last 20 trades + last 3 days summary
  python scripts/trades_report.py --sync-sqlite  # also upsert into logs/trades.db
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dateutil import parser
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter  # type: ignore
import httpx

DB_PATH = Path("logs/trades.db")


def _pip_size(instrument: str) -> float:
    # USD_JPY etc: 1 pip = 0.01
    if instrument.endswith("_JPY"):
        return 0.01
    return 0.0001


def _to_dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str) and v:
        try:
            return parser.isoparse(v)
        except Exception:
            return None
    return None


def _row_from_doc(x: Dict[str, Any]) -> Dict[str, Any]:
    instrument = x.get("instrument") or "USD_JPY"
    units = int(x.get("units") or 0)
    entry_price = x.get("fill_price") or x.get("price")
    close_price = x.get("close_price")
    # Compute pips if prices available
    pl_pips = None
    try:
        if entry_price is not None and close_price is not None and units:
            direction = 1 if units > 0 else -1
            pipsize = _pip_size(instrument)
            pl_pips = (float(close_price) - float(entry_price)) * direction / pipsize
    except Exception:
        pl_pips = None

    row = {
        "ticket_id": str(x.get("trade_id") or ""),
        "pocket": x.get("pocket"),
        "instrument": instrument,
        "units": units,
        "entry_price": float(entry_price) if entry_price is not None else None,
        "close_price": float(close_price) if close_price is not None else None,
        "pl_pips": float(pl_pips) if pl_pips is not None else None,
        "entry_time": (x.get("fill_time") or x.get("ts")) or None,
        "close_time": x.get("close_time") or None,
        "strategy": x.get("strategy"),
        "macro_regime": x.get("macro_regime"),
        "micro_regime": x.get("micro_regime"),
        "realized_pl": x.get("realized_pl"),
        "state": x.get("state"),
    }
    return row


def _row_from_open_trade(tr: Dict[str, Any]) -> Dict[str, Any]:
    instrument = tr.get("instrument") or "USD_JPY"
    units = int(tr.get("initialUnits") or tr.get("units") or 0)
    entry_price = tr.get("price")
    ticket = tr.get("id") or tr.get("tradeID")

    row = {
        "ticket_id": str(ticket or ""),
        "pocket": tr.get("clientExtensions", {}).get("tag", "").split("=")[-1] if tr.get("clientExtensions") else None,
        "instrument": instrument,
        "units": units,
        "entry_price": float(entry_price) if entry_price is not None else None,
        "close_price": None,
        "pl_pips": None,
        "entry_time": tr.get("openTime"),
        "close_time": None,
        "strategy": None,
        "macro_regime": None,
        "micro_regime": None,
        "realized_pl": None,
        "state": "OPEN",
    }

    ext = tr.get("clientExtensions") or {}
    comment = ext.get("comment") or ""
    for part in comment.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k == "strategy":
                row["strategy"] = v
            elif k == "macro":
                row["macro_regime"] = v
            elif k == "micro":
                row["micro_regime"] = v

    if not row.get("pocket"):
        row["pocket"] = "macro" if abs(units) >= 100000 else "micro"
    return row


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
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
  micro_regime TEXT,
  realized_pl REAL,
  state TEXT
)
"""
    )
    # Backfill missing columns for older DBs
    try:
        cols = {row[1] for row in con.execute("PRAGMA table_info(trades)").fetchall()}
        for coldef in (
            ("strategy", "TEXT"),
            ("macro_regime", "TEXT"),
            ("micro_regime", "TEXT"),
            ("realized_pl", "REAL"),
            ("state", "TEXT"),
        ):
            if coldef[0] not in cols:
                con.execute(f"ALTER TABLE trades ADD COLUMN {coldef[0]} {coldef[1]}")
        con.commit()
    except Exception:
        pass
    # Ensure UNIQUE index on ticket_id for upsert
    try:
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_ticket ON trades(ticket_id)")
        con.commit()
    except Exception:
        pass
    return con


def _upsert_sqlite(con: sqlite3.Connection, rows: Iterable[Dict[str, Any]]):
    sql = (
        "INSERT INTO trades(ticket_id,pocket,instrument,units,entry_price,close_price,pl_pips,entry_time,close_time,"
        "strategy,macro_regime,micro_regime,realized_pl,state)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT(ticket_id) DO UPDATE SET "
        " pocket=excluded.pocket, instrument=excluded.instrument, units=excluded.units,"
        " entry_price=excluded.entry_price, close_price=excluded.close_price, pl_pips=excluded.pl_pips,"
        " entry_time=excluded.entry_time, close_time=excluded.close_time, strategy=excluded.strategy,"
        " macro_regime=excluded.macro_regime, micro_regime=excluded.micro_regime,"
        " realized_pl=excluded.realized_pl, state=excluded.state"
    )
    data = [
        (
            r.get("ticket_id"),
            r.get("pocket"),
            r.get("instrument"),
            r.get("units"),
            r.get("entry_price"),
            r.get("close_price"),
            r.get("pl_pips"),
            r.get("entry_time"),
            r.get("close_time"),
            r.get("strategy"),
            r.get("macro_regime"),
            r.get("micro_regime"),
            r.get("realized_pl"),
            r.get("state"),
        )
        for r in rows
    ]
    if data:
        con.executemany(sql, data)
        con.commit()


def _sync_open_trades(con: sqlite3.Connection) -> int:
    try:
        token = get_secret("oanda_token")
        account = get_secret("oanda_account_id")
        try:
            practice = get_secret("oanda_practice").lower() == "true"
        except Exception:
            practice = True
        host = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        url = f"{host}/v3/accounts/{account}/openTrades"
        headers = {"Authorization": f"Bearer {token}"}
    except Exception:
        return 0

    try:
        with httpx.Client(timeout=7.0) as client:
            r = client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json().get("trades", []) or []
    except Exception:
        return 0

    rows = [_row_from_open_trade(tr) for tr in data]
    _upsert_sqlite(con, rows)
    return len(rows)


def fetch_firestore(max_docs: int = 5000, page_size: int = 500, since_iso: str | None = None, until_iso: str | None = None) -> List[Dict[str, Any]]:
    fs = firestore.Client(project="quantrabbit")
    q = fs.collection("trades").order_by("ts", direction=firestore.Query.DESCENDING)
    # 上限（日付）だけは単一フィールドに限定してクエリ、下限はループで打ち切り
    if until_iso:
        try:
            q = q.where(filter=FieldFilter("ts", "<=", until_iso))  # type: ignore
        except Exception:
            q = q.where("ts", "<=", until_iso)  # fallback

    rows: List[Dict[str, Any]] = []
    last_doc = None
    fetched = 0
    while fetched < max_docs:
        q_page = q.limit(page_size)
        if last_doc is not None:
            q_page = q_page.start_after(last_doc)
        page_docs = list(q_page.stream())
        if not page_docs:
            break
        for d in page_docs:
            x = d.to_dict() or {}
            # since 指定があり、古すぎれば打ち切り
            if since_iso:
                ts = x.get("ts") or x.get("close_time") or x.get("fill_time")
                if isinstance(ts, str) and ts < since_iso:
                    fetched = max_docs  # break all
                    break
            rows.append(_row_from_doc(x))
            fetched += 1
            if fetched >= max_docs:
                break
        last_doc = page_docs[-1]
    return rows


def summarize(rows: List[Dict[str, Any]]):
    # Latest 20 CLOSED printable
    closed = [r for r in rows if r.get("state") == "CLOSED"]
    latest = closed[:20]
    # By day (UTC) over ALL closed rows in window
    agg: Dict[str, Dict[str, Any]] = {}
    for r in closed:
        dt = _to_dt(r.get("close_time") or r.get("entry_time"))
        if not dt:
            continue
        dkey = dt.astimezone(timezone.utc).date().isoformat()
        a = agg.setdefault(dkey, {"n": 0, "sum_pl": 0.0})
        a["n"] += 1
        try:
            a["sum_pl"] += float(r.get("realized_pl") or 0)
        except Exception:
            pass
    return latest, agg


def main():
    ap = argparse.ArgumentParser(description="Print Firestore trades summary and optional SQLite sync.")
    ap.add_argument("--sync-sqlite", action="store_true", help="Upsert into logs/trades.db as well")
    ap.add_argument("--days", type=int, default=14, help="How many days back to pull (via ts) [default: 14]")
    ap.add_argument("--max", type=int, default=5000, help="Max documents to pull [default: 5000]")
    args = ap.parse_args()

    # Time window
    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(days=args.days)
    since_iso = since_dt.replace(tzinfo=None).isoformat(timespec="seconds")
    rows = fetch_firestore(max_docs=args.max, page_size=500, since_iso=since_iso)
    latest, agg = summarize(rows)

    print("Latest 5 CLOSED trades:")
    for r in latest[:5]:
        print(
            f"{r.get('close_time') or r.get('entry_time')} "
            f"{r.get('instrument')} {r.get('action','')} {r.get('units')} "
            f"pl={r.get('realized_pl')} strategy={r.get('strategy')} pocket={r.get('pocket')}"
        )
    print("\nDaily summary (UTC, CLOSED):")
    for d in sorted(agg.keys()):
        print(f"{d} n={agg[d]['n']} sum_pl={round(agg[d]['sum_pl'],2)}")

    if args.sync_sqlite:
        con = _ensure_db()
        open_count = _sync_open_trades(con)
        firestore_rows = [r for r in rows if r.get("ticket_id")]
        _upsert_sqlite(con, firestore_rows)
        print(
            f"\nSynced {len(firestore_rows)} Firestore rows and {open_count} open trades into {DB_PATH}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick PnL reconciliation against OANDA transactions."""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
from collections import defaultdict
from typing import Dict, Tuple

import requests

from utils.secrets import get_secret


def _auth() -> Tuple[str, Dict[str, str]]:
    token = get_secret("oanda_token")
    account = get_secret("oanda_account_id")  # ensure secret exists
    try:
        practice = get_secret("oanda_practice").lower() == "true"
    except Exception:
        practice = True
    host = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
    return host, {"Authorization": f"Bearer {token}"}


def _fetch_transactions(hours: float) -> Dict[str, float]:
    host, headers = _auth()
    account = get_secret("oanda_account_id")
    url = f"{host}/v3/accounts/{account}/transactions"
    start = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc) - dt.timedelta(hours=hours)
    params: Dict[str, str] = {"pageSize": "500"}
    page = 1
    totals = defaultdict(float)
    while True:
        params["pageNumber"] = str(page)
        params["from"] = start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        transactions = payload.get("transactions", []) or []
        if not transactions:
            break
        for tx in transactions:
            t_type = tx.get("type")
            if t_type not in {"ORDER_FILL", "TRADE_CLOSE"}:
                continue
            pl = tx.get("pl")
            financing = tx.get("financing")
            if pl is not None:
                totals["pl"] += float(pl)
            if financing is not None:
                totals["financing"] += float(financing)
        page_info = payload.get("pages", {})
        page_count = int(page_info.get("pageCount", page))
        if page >= page_count:
            break
        page += 1
    return totals


def _sum_trades(db_path: str, hours: float) -> Dict[str, float]:
    since = dt.datetime.utcnow() - dt.timedelta(hours=hours)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT pocket, COALESCE(SUM(realized_pl),0) AS pnl
        FROM trades
        WHERE close_time IS NOT NULL AND close_time >= ?
        GROUP BY pocket
        """,
        (since.isoformat(timespec="seconds"),),
    ).fetchall()
    totals = {row["pocket"] or "(unknown)": float(row["pnl"]) for row in rows}
    con.close()
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24.0, help="lookback window in hours (default: 24)")
    parser.add_argument(
        "--db",
        default="logs/trades.db",
        help="path to local trades.db (default: logs/trades.db)",
    )
    args = parser.parse_args()

    db_totals = _sum_trades(args.db, args.hours)
    oanda_totals = _fetch_transactions(args.hours)

    print("=== Local trades.db ===")
    if not db_totals:
        print("  (no rows)")
    for pocket, value in sorted(db_totals.items()):
        print(f"  {pocket:>6}: {value:>10.2f} JPY")

    print("\n=== OANDA transactions ===")
    print(f"  realized pl : {oanda_totals['pl']:.2f} JPY")
    print(f"  financing   : {oanda_totals['financing']:.2f} JPY")
    print(f"  combined    : {oanda_totals['pl'] + oanda_totals['financing']:.2f} JPY")

    recorded = sum(db_totals.values())
    realized = oanda_totals["pl"] + oanda_totals["financing"]
    delta = realized - recorded
    print("\n=== Difference ===")
    print(f"  OANDA - trades.db : {delta:.2f} JPY")


if __name__ == "__main__":
    main()

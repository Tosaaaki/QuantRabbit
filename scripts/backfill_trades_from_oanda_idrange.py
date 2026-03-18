#!/usr/bin/env python3
"""
Backfill/repair trades.db from OANDA transactions (idrange) on local V2.

Why:
  When trade-close ingestion misses some ORDER_FILL closures (network/rate-limit/etc),
  trades.db can lose realized P/L attribution. This script replays a transaction ID
  range and re-parses it through PositionManager logic (idempotent via
  uniq(transaction_id, ticket_id)).

Usage examples:
  python scripts/backfill_trades_from_oanda_idrange.py --last-n 5000
  python scripts/backfill_trades_from_oanda_idrange.py --from-id 430000 --to-id 434043
  python scripts/backfill_trades_from_oanda_idrange.py --last-n 8000 --dry-run

Notes:
  - Local-only (no VM/GCP). Uses OANDA REST and logs/*.db.
  - By default disables PositionManager init-time backfills to keep the run focused.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.secrets import get_secret


def _mask(s: str) -> str:
    s = str(s or "")
    if len(s) <= 4:
        return "****"
    return f"{s[:2]}****{s[-2:]}"


def _oanda_host() -> str:
    practice = False
    try:
        practice = get_secret("oanda_practice").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        practice = False
    return (
        "https://api-fxpractice.oanda.com"
        if practice
        else "https://api-fxtrade.oanda.com"
    )


@dataclass(frozen=True)
class _OandaCreds:
    host: str
    account: str
    headers: dict[str, str]


def _load_creds() -> _OandaCreds:
    token = get_secret("oanda_token")
    account = get_secret("oanda_account_id")
    host = _oanda_host()
    return _OandaCreds(
        host=host, account=account, headers={"Authorization": f"Bearer {token}"}
    )


def _get_last_transaction_id(creds: _OandaCreds) -> int:
    url = f"{creds.host}/v3/accounts/{creds.account}/transactions"
    resp = requests.get(url, headers=creds.headers, params={"sinceID": 0}, timeout=10)
    resp.raise_for_status()
    payload = resp.json() or {}
    try:
        return int(payload.get("lastTransactionID") or 0)
    except (TypeError, ValueError):
        return 0


def _fetch_transactions_idrange(
    creds: _OandaCreds, from_id: int, to_id: int, *, chunk_size: int = 500
) -> list[dict[str, Any]]:
    if from_id <= 0 or to_id <= 0 or from_id > to_id:
        return []
    out: list[dict[str, Any]] = []
    url = f"{creds.host}/v3/accounts/{creds.account}/transactions/idrange"
    cur = int(from_id)
    while cur <= to_id:
        chunk_to = min(to_id, cur + max(1, int(chunk_size)) - 1)
        resp = requests.get(
            url,
            headers=creds.headers,
            params={"from": cur, "to": chunk_to},
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json() or {}
        txs = payload.get("transactions") or []
        if isinstance(txs, list):
            for tx in txs:
                if isinstance(tx, dict):
                    out.append(tx)
        cur = chunk_to + 1
    return out


def _iter_closure_trade_ids(transactions: Iterable[dict[str, Any]]) -> set[str]:
    trade_ids: set[str] = set()
    for tx in transactions:
        if tx.get("type") != "ORDER_FILL":
            continue
        for closed in tx.get("tradesClosed") or []:
            if isinstance(closed, dict) and closed.get("tradeID"):
                trade_ids.add(str(closed.get("tradeID")))
        reduced = tx.get("tradeReduced")
        if isinstance(reduced, dict) and reduced.get("tradeID"):
            trade_ids.add(str(reduced.get("tradeID")))
        for reduced in tx.get("tradesReduced") or []:
            if isinstance(reduced, dict) and reduced.get("tradeID"):
                trade_ids.add(str(reduced.get("tradeID")))
    return trade_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill trades.db from OANDA transactions idrange."
    )
    parser.add_argument("--from-id", type=int, help="Start transaction id (inclusive).")
    parser.add_argument(
        "--to-id",
        type=int,
        help="End transaction id (inclusive). Default: lastTransactionID.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=5000,
        help="Backfill last N transactions when from/to not set.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="idrange chunk size."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and summarize but do not write trades.db.",
    )
    args = parser.parse_args()

    # Keep this run focused; avoid PositionManager init-time backfills.
    os.environ.setdefault("POSITION_MANAGER_BACKFILL_ATTR", "0")
    os.environ.setdefault("POSITION_MANAGER_CASHFLOW_BACKFILL", "0")
    os.environ.setdefault("POSITION_MANAGER_CASHFLOW_BACKFILL_OANDA", "0")

    creds = _load_creds()
    last_id = _get_last_transaction_id(creds)
    if last_id <= 0:
        print(
            json.dumps(
                {"ok": False, "reason": "failed_to_get_lastTransactionID"},
                ensure_ascii=False,
            )
        )
        return 2

    to_id = int(args.to_id or last_id)
    if args.from_id is not None:
        from_id = int(args.from_id)
    else:
        last_n = max(1, int(args.last_n))
        from_id = max(1, to_id - last_n + 1)

    t0 = time.time()
    transactions = _fetch_transactions_idrange(
        creds, from_id, to_id, chunk_size=args.chunk_size
    )
    dt_ms = int((time.time() - t0) * 1000)
    closure_trade_ids = _iter_closure_trade_ids(transactions)
    summary = {
        "ok": True,
        "account_masked": _mask(creds.account),
        "from_id": from_id,
        "to_id": to_id,
        "lastTransactionID": last_id,
        "transactions": len(transactions),
        "closure_trade_ids": len(closure_trade_ids),
        "fetch_ms": dt_ms,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    from execution.position_manager import PositionManager  # noqa: E402

    pm = PositionManager()
    saved = pm._parse_and_save_trades(transactions)  # noqa: SLF001 - internal backfill
    summary["saved_records"] = len(saved or [])
    summary["parse_meta"] = pm._last_sync_parse_meta
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

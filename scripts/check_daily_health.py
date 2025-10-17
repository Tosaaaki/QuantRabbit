#!/usr/bin/env python3
"""Run daily health checks comparing local trades.db with OANDA transactions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

from scripts import reconcile_pnl


def _serialize_totals(label: str, data: Dict[str, float]) -> Dict[str, float]:
    return {f"{label}_{k}": round(v, 2) for k, v in sorted(data.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare trades.db PnL against OANDA over the given window.")
    parser.add_argument("--hours", type=float, default=24.0, help="Lookback window in hours (default: 24)")
    parser.add_argument(
        "--db",
        default="logs/trades.db",
        help="Path to local trades.db (default: logs/trades.db)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("DAILY_HEALTH_MAX_DELTA", "5.0")),
        help="Allowed absolute delta in JPY before marking the check as failed (default: 5.0)",
    )
    args = parser.parse_args()

    db_totals = reconcile_pnl._sum_trades(args.db, args.hours)
    oanda_totals = reconcile_pnl._fetch_transactions(args.hours)

    recorded = sum(db_totals.values())
    realized = oanda_totals.get("pl", 0.0) + oanda_totals.get("financing", 0.0)
    delta = realized - recorded

    payload = {
        "hours": args.hours,
        "threshold": round(args.threshold, 2),
        "delta": round(delta, 2),
    }
    payload.update(_serialize_totals("db", db_totals))
    payload.update({
        "oanda_realized_pl": round(oanda_totals.get("pl", 0.0), 2),
        "oanda_financing": round(oanda_totals.get("financing", 0.0), 2),
        "oanda_combined": round(realized, 2),
    })

    print(json.dumps(payload, ensure_ascii=False))

    if abs(delta) > args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

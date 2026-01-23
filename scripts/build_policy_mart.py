#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analytics.policy_mart import PolicyMartClient


def main() -> None:
    ap = argparse.ArgumentParser(description="Create/refresh the BigQuery policy mart view.")
    ap.add_argument("--project", default=None, help="GCP project id (defaults to env)")
    ap.add_argument("--dataset", default=None, help="BigQuery dataset id")
    ap.add_argument("--table", default=None, help="Trades table id (default trades_raw)")
    ap.add_argument("--view", default="policy_mart_view", help="View name to create")
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--min-trades", type=int, default=0)
    ap.add_argument("--timezone", default=None)
    ap.add_argument("--vol-buckets", default=None, help="Comma-separated ATR bucket thresholds")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    thresholds = None
    if args.vol_buckets:
        thresholds = [float(x.strip()) for x in args.vol_buckets.split(",") if x.strip()]

    client = PolicyMartClient(
        project_id=args.project,
        dataset_id=args.dataset or os.getenv("BQ_DATASET", "quantrabbit"),
        trades_table=args.table or os.getenv("BQ_TRADES_TABLE", "trades_raw"),
        timezone=args.timezone or None,
        vol_thresholds=thresholds,
    )
    client.create_view(
        view_name=args.view,
        lookback_days=args.lookback_days,
        min_trades=args.min_trades,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build deterministic market context for ops playbook."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts import gpt_ops_report


def main() -> int:
    ap = argparse.ArgumentParser(description="Build market context JSON for ops playbook")
    ap.add_argument("--output", default=os.getenv("OPS_PLAYBOOK_MARKET_CONTEXT_PATH", "logs/market_context_latest.json"))
    ap.add_argument("--events-path", default=os.getenv("OPS_PLAYBOOK_EVENTS_PATH", "logs/market_events.json"))
    ap.add_argument(
        "--market-external-path",
        default=os.getenv("OPS_PLAYBOOK_EXTERNAL_SNAPSHOT_PATH", "logs/market_external_snapshot.json"),
    )
    ap.add_argument(
        "--macro-snapshot-path",
        default=os.getenv("OPS_PLAYBOOK_MACRO_SNAPSHOT_PATH", "fixtures/macro_snapshots/latest.json"),
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    now_utc = datetime.now(timezone.utc)
    factors = gpt_ops_report._load_factors()
    events = gpt_ops_report._load_events(Path(args.events_path), now_utc=now_utc)
    external_snapshot = gpt_ops_report._read_json(Path(args.market_external_path)) or {}
    macro_snapshot = gpt_ops_report._load_macro_snapshot(Path(args.macro_snapshot_path))

    payload = gpt_ops_report._build_market_context(
        factors=factors,
        events=events,
        now_utc=now_utc,
        external_snapshot=external_snapshot,
        macro_snapshot=macro_snapshot,
    )

    out_path = Path(args.output)
    gpt_ops_report._write_json(out_path, payload)
    logging.info("[MARKET_CONTEXT] wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

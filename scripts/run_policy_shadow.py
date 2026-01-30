#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analytics.policy_generator import summarize_policy_rows
from analytics.policy_ledger import PolicyLedger
from analytics.policy_mart import PolicyMartClient
from analytics.policy_diff import normalize_policy_diff


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate policy shadow diff (LLM disabled).")
    ap.add_argument("--project", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--min-trades", type=int, default=0)
    ap.add_argument("--output", default="logs/policy_diff_shadow.json")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if os.getenv("POLICY_SHADOW_ENABLED", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        logging.info("[POLICY_SHADOW] disabled (POLICY_SHADOW_ENABLED=0)")
        return

    client = PolicyMartClient(
        project_id=args.project,
        dataset_id=args.dataset or os.getenv("BQ_DATASET", "quantrabbit"),
        trades_table=args.table or os.getenv("BQ_TRADES_TABLE", "trades_raw"),
    )
    rows = client.fetch_rows(lookback_days=args.lookback_days, min_trades=args.min_trades)
    summary = summarize_policy_rows(rows)

    payload = normalize_policy_diff(
        {"no_change": True, "reason": "llm_disabled", "source": "shadow_stub"},
        source="shadow_stub",
    )

    out_path = Path(args.output)
    _write_json(out_path, payload)
    logging.info("[POLICY_SHADOW] diff written: %s", out_path)

    if not args.no_ledger:
        ledger = PolicyLedger(project_id=args.project)
        ledger.record(payload, status="shadow", summary=summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from pathlib import Path

from analytics.policy_apply import apply_policy_diff_to_paths
from analytics.policy_generator import (
    DEFAULT_VERTEX_LOCATION,
    DEFAULT_VERTEX_MODEL,
    generate_policy_diff,
)
from analytics.policy_ledger import PolicyLedger
from analytics.policy_mart import PolicyMartClient


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Vertex AI policy cycle.")
    ap.add_argument("--project", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--min-trades", type=int, default=0)
    ap.add_argument("--output", default="logs/policy_diff_latest.json")
    ap.add_argument("--apply", action="store_true", help="Apply policy diff to overlay")
    ap.add_argument("--overlay-path", default="logs/policy_overlay.json")
    ap.add_argument("--history-dir", default="logs/policy_history")
    ap.add_argument("--latest-path", default="logs/policy_latest.json")
    ap.add_argument("--no-vertex", action="store_true", help="Disable Vertex AI generation")
    ap.add_argument("--vertex-model", default=None)
    ap.add_argument("--vertex-location", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-ledger", action="store_true", help="Disable BQ/GCS/Firestore ledger")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    client = PolicyMartClient(
        project_id=args.project,
        dataset_id=args.dataset or os.getenv("BQ_DATASET", "quantrabbit"),
        trades_table=args.table or os.getenv("BQ_TRADES_TABLE", "trades_raw"),
    )
    rows = client.fetch_rows(lookback_days=args.lookback_days, min_trades=args.min_trades)
    diff, summary = generate_policy_diff(
        rows,
        use_vertex=not args.no_vertex,
        project_id=args.project,
        location=args.vertex_location or DEFAULT_VERTEX_LOCATION,
        model=args.vertex_model or DEFAULT_VERTEX_MODEL,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        fallback_min_trades=args.min_trades or 12,
    )

    output_path = Path(args.output)
    _write_json(output_path, diff)
    logging.info("[POLICY] diff written: %s", output_path)

    ledger = None if args.no_ledger else PolicyLedger(project_id=args.project)
    if ledger:
        ledger.record(diff, status="generated", summary=summary)

    if args.apply:
        updated, changed, flags = apply_policy_diff_to_paths(
            diff,
            overlay_path=Path(args.overlay_path),
            history_dir=Path(args.history_dir),
            latest_path=Path(args.latest_path),
        )
        logging.info(
            "[POLICY] applied=%s reentry=%s tuning=%s overlay=%s",
            changed,
            flags.get("reentry"),
            flags.get("tuning"),
            args.overlay_path,
        )
        if ledger and changed:
            ledger.record(
                updated,
                status="applied",
                summary={"applied": changed, "flags": flags},
            )


if __name__ == "__main__":
    main()

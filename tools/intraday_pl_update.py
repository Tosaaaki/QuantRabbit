#!/usr/bin/env python3
"""Update the read-only UTC intraday P/L snapshot.

Trader runtime accounting uses UTC 00:00 as the day boundary. Slack or operator
messages may display JST, but P/L selection and target progress stay UTC-based.
This tool never places, cancels, or closes orders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from daily_target import (
    DEFAULT_DAY_START_DIR,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_SNAPSHOT_PATH,
    compute_daily_target,
    format_daily_target_block,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTRADAY_DIR = REPO_ROOT / "logs" / "intraday_pl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write/read UTC intraday P/L progress.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    parser.add_argument("--day-start-dir", type=Path, default=DEFAULT_DAY_START_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INTRADAY_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print only; do not write logs/intraday_pl.")
    parser.add_argument(
        "--extension-gate",
        choices=("yes", "no"),
        default="no",
        help="Explicit favorable-market extension gate for +10% mode.",
    )
    args = parser.parse_args()

    metrics = compute_daily_target(
        snapshot_path=args.snapshot,
        execution_ledger_db=args.execution_ledger_db,
        day_start_dir=args.day_start_dir,
        dry_run=args.dry_run,
        extension_gate=args.extension_gate == "yes",
    )
    payload = metrics.to_dict()
    if args.dry_run:
        print(format_daily_target_block(metrics))
        print("[dry-run] UTC intraday P/L output not written")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{metrics.trading_day_utc}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"Wrote UTC intraday P/L snapshot: {output}")
    print(format_daily_target_block(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

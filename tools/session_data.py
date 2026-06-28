#!/usr/bin/env python3
"""Print the operator session-start data block.

This is read-only except for the day-start NAV record created by
tools/daily_target.py when the UTC trading day has no record yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from daily_target import (
    DEFAULT_DAY_START_DIR,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_SNAPSHOT_PATH,
    compute_daily_target,
    format_daily_target_block,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print session-start QuantRabbit data.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--execution-ledger-db", type=Path, default=DEFAULT_EXECUTION_LEDGER_DB)
    parser.add_argument("--day-start-dir", type=Path, default=DEFAULT_DAY_START_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Do not persist a missing day-start NAV record.")
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
    print(format_daily_target_block(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

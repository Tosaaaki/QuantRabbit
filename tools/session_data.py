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
    DailyTargetMetrics,
    compute_daily_target,
    _format_jpy,
    format_daily_target_block,
)


FIVE_PCT_PATH_BOARD_TEMPLATE = """## 5% PATH BOARD
Remaining to +5%: {remaining_to_5pct}

Path A / HERO:
Pair / side / vehicle:
Expected pips:
Suggested units:
Expected contribution:
Entry:
TP:
SL:
Status: live / armable / blocked
Exact blocker if blocked:

Path B / SECOND SHOT:
Pair / side / vehicle:
Expected pips:
Suggested units:
Expected contribution:
Entry:
TP:
SL:
Status:
Exact blocker if blocked:

Path C / NO HONEST PATH:
Exact blocker:
Next trigger:
Shelf-life:"""


ATTACK_STACK_TEMPLATE = """## ATTACK STACK
Hero thesis:
Why this thesis can still reach +5% today:

NOW:
Pair / side / vehicle:
Entry:
TP:
SL:
Units:
Why now:

RELOAD:
Pair / side / vehicle:
Entry:
TP:
SL:
Units:
Why this is better price, not hesitation:

SECOND SHOT:
Pair / side / vehicle:
Entry:
TP:
SL:
Units:
Why this is same theme, different expression:

If any slot is empty:
Exact blocker:
Next trigger:
Shelf-life:"""


FIVE_PCT_PATH_RULES = """## 5% PATH RULES
- Under +5%, trader must name an A/S path or exact blocker.
- B/C trades cannot be the +5% target path.
- One distant pending order is not enough.
- "Trigger not printed yet" is an arm condition for LIMIT/STOP, not a dead thesis.
- The path must map to ATTACK STACK."""


def format_path_board(metrics: DailyTargetMetrics) -> str:
    """Return the required FULL_TRADER +5% path board template."""
    return FIVE_PCT_PATH_BOARD_TEMPLATE.format(
        remaining_to_5pct=_format_jpy(metrics.remaining_to_5pct_yen),
    )


def format_attack_stack() -> str:
    """Return the required FULL_TRADER attack stack template."""
    return ATTACK_STACK_TEMPLATE


def format_full_trader_board(metrics: DailyTargetMetrics) -> str:
    """Return the complete session-start path contract block."""
    return "\n\n".join((format_path_board(metrics), format_attack_stack(), FIVE_PCT_PATH_RULES))


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
    print()
    print(format_full_trader_board(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

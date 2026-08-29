#!/usr/bin/env python3
"""Update prospective shock-follow ledgers and their independent scorecard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_shock_follow import run_incremental  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-charts", type=Path, required=True)
    parser.add_argument("--broker-snapshot", type=Path, required=True)
    parser.add_argument("--signal-ledger", type=Path, required=True)
    parser.add_argument("--outcome-ledger", type=Path, required=True)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--corrective-ledger", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-due", type=int, default=12)
    args = parser.parse_args()
    result = run_incremental(
        pair_charts_path=args.pair_charts,
        broker_snapshot_path=args.broker_snapshot,
        signal_ledger_path=args.signal_ledger,
        outcome_ledger_path=args.outcome_ledger,
        scorecard_path=args.scorecard,
        corrective_ledger_path=args.corrective_ledger,
        config_path=args.config,
        max_due=args.max_due,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

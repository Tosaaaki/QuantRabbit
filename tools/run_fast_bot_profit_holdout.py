#!/usr/bin/env python3
"""Run the pre-registered, zero-authority fast-bot profit holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_profit_holdout import (  # noqa: E402
    run_evaluation,
    run_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--shadow", type=Path, required=True)
    select.add_argument("--raw-signal-ledger", type=Path, required=True)
    select.add_argument("--policy", type=Path, required=True)
    select.add_argument("--selected-ledger", type=Path, required=True)
    select.add_argument("--decision-ledger", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--report", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--raw-signal-ledger", type=Path, required=True)
    evaluate.add_argument("--selected-ledger", type=Path, required=True)
    evaluate.add_argument("--decision-ledger", type=Path, required=True)
    evaluate.add_argument("--outcome-ledger", type=Path, required=True)
    evaluate.add_argument("--truth-scorecard", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--report", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "select":
        result = run_selection(
            raw_shadow_path=args.shadow,
            raw_signal_ledger_path=args.raw_signal_ledger,
            policy_path=args.policy,
            selected_ledger_path=args.selected_ledger,
            decision_ledger_path=args.decision_ledger,
            output_path=args.output,
            report_path=args.report,
        )
        invalid = result["status"] in {
            "BLOCKED_SOURCE_INTEGRITY",
            "BLOCKED_HISTORY_INTEGRITY",
        }
    else:
        result = run_evaluation(
            policy_path=args.policy,
            raw_signal_ledger_path=args.raw_signal_ledger,
            selected_ledger_path=args.selected_ledger,
            decision_ledger_path=args.decision_ledger,
            outcome_ledger_path=args.outcome_ledger,
            truth_scorecard_path=args.truth_scorecard,
            output_path=args.output,
            report_path=args.report,
        )
        invalid = result["status"] == "REJECT_INVALID_HOLDOUT_COHORT"
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())

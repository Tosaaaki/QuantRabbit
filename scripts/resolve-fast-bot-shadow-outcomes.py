#!/usr/bin/env python3
"""Resolve due fast-bot shadows from exact read-only OANDA S5 bid/ask truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_truth import resolve_due_fast_bot_outcomes_from_oanda  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-ledger", type=Path, default=ROOT / "data" / "fast_bot_shadow_ledger.jsonl")
    parser.add_argument("--outcome-ledger", type=Path, default=ROOT / "data" / "fast_bot_outcome_ledger.jsonl")
    parser.add_argument("--scorecard", type=Path, default=ROOT / "data" / "fast_bot_scorecard.json")
    args = parser.parse_args()
    result = resolve_due_fast_bot_outcomes_from_oanda(
        shadow_ledger_path=args.shadow_ledger,
        outcome_ledger_path=args.outcome_ledger,
        scorecard_path=args.scorecard,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] in {"NO_DUE_SIGNALS", "RESOLVED", "RESOLVED_WITH_ERRORS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

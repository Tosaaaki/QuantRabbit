#!/usr/bin/env python3
"""Resolve due technical forward shadows from read-only OANDA S5 truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.technical_forecast_forward_truth import (  # noqa: E402
    resolve_due_forward_outcomes_from_oanda,
)


def main() -> int:
    args = _parse_args()
    result = resolve_due_forward_outcomes_from_oanda(
        candidate_path=args.candidate,
        shadow_ledger_path=args.shadow_ledger,
        outcome_ledger_path=args.outcome_ledger,
        scorecard_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] not in {
        "LEDGER_OR_CANDIDATE_INVALID",
        "OUTCOME_PERSISTENCE_FAILED",
    } else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "config" / "technical_forecast_forward_candidate_v1.json",
    )
    parser.add_argument(
        "--shadow-ledger",
        type=Path,
        default=ROOT / "data" / "technical_forecast_forward_shadow_ledger.jsonl",
    )
    parser.add_argument(
        "--outcome-ledger",
        type=Path,
        default=ROOT / "data" / "technical_forecast_forward_outcome_ledger.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "technical_forecast_forward_scorecard.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

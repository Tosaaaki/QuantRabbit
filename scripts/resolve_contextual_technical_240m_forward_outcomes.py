#!/usr/bin/env python3
"""Resolve due contextual 240-minute shadows from exact OANDA S5 bid/ask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.contextual_technical_forward import (  # noqa: E402
    resolve_due_outcomes_from_oanda,
)


def main() -> int:
    args = _parse_args()
    result = resolve_due_outcomes_from_oanda(
        candidate_path=args.candidate,
        shadow_ledger_path=args.shadow_ledger,
        outcome_ledger_path=args.outcome_ledger,
        scorecard_path=args.scorecard,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "config" / "contextual_technical_240m_forward_candidate_v1.json",
    )
    parser.add_argument(
        "--shadow-ledger",
        type=Path,
        default=ROOT / "data" / "contextual_technical_240m_forward_shadow_ledger.jsonl",
    )
    parser.add_argument(
        "--outcome-ledger",
        type=Path,
        default=ROOT / "data" / "contextual_technical_240m_forward_outcome_ledger.jsonl",
    )
    parser.add_argument(
        "--scorecard",
        type=Path,
        default=ROOT / "data" / "contextual_technical_240m_forward_scorecard.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

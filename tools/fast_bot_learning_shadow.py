#!/usr/bin/env python3
"""Emit the separate diagnostic fast-bot learning shadow cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_learning import run_fast_bot_learning_shadow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime-contract",
        type=Path,
        default=ROOT / "data" / "hierarchical_bot_regime.json",
    )
    parser.add_argument(
        "--broker-snapshot",
        type=Path,
        default=ROOT / "data" / "position_guardian_broker_snapshot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "fast_bot_learning_shadow.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "data" / "fast_bot_learning_seat_ledger.jsonl",
    )
    args = parser.parse_args()
    result = run_fast_bot_learning_shadow(
        regime_contract_path=args.regime_contract,
        broker_snapshot_path=args.broker_snapshot,
        output_path=args.output,
        ledger_path=args.ledger,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] != "LEARNING_LEDGER_INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())

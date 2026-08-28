#!/usr/bin/env python3
"""Update the separated fast-bot corrective challenger ledger and scorecard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_corrective_challenger import run_incremental  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-ledger", type=Path, required=True)
    parser.add_argument("--outcome-ledger", type=Path, required=True)
    parser.add_argument("--challenger-ledger", type=Path, required=True)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-due", type=int, default=12)
    args = parser.parse_args()
    result = run_incremental(
        shadow_ledger_path=args.shadow_ledger,
        outcome_ledger_path=args.outcome_ledger,
        challenger_ledger_path=args.challenger_ledger,
        scorecard_path=args.scorecard,
        config_path=args.config,
        max_due=args.max_due,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

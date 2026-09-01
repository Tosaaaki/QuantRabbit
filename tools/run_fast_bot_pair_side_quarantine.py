#!/usr/bin/env python3
"""Select the prospective pair-side-quarantined fast-bot cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_pair_side_quarantine import run_selection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-signal-ledger", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--selected-ledger", type=Path, required=True)
    parser.add_argument("--decision-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_selection(
        raw_signal_ledger_path=args.raw_signal_ledger,
        policy_path=args.policy,
        selected_ledger_path=args.selected_ledger,
        decision_ledger_path=args.decision_ledger,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

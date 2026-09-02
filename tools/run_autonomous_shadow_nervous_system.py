#!/usr/bin/env python3
"""Advance the autonomous nervous system from resident shadow evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.autonomous_shadow_integration import (  # noqa: E402
    run_autonomous_shadow_integration,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-ledger", type=Path, required=True)
    parser.add_argument("--shock-guard-decision-ledger", type=Path, required=True)
    parser.add_argument("--outcome-ledger", type=Path, required=True)
    parser.add_argument("--learning-episode-ledger", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-signals", type=int, default=128)
    args = parser.parse_args()
    result = run_autonomous_shadow_integration(
        shadow_ledger_path=args.shadow_ledger,
        shock_guard_decision_ledger_path=args.shock_guard_decision_ledger,
        outcome_ledger_path=args.outcome_ledger,
        learning_episode_ledger_path=args.learning_episode_ledger,
        state_root=args.state_root,
        output_path=args.output,
        report_path=args.report,
        max_signals=args.max_signals,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

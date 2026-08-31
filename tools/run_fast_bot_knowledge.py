#!/usr/bin/env python3
"""Build versioned fast-bot shadow episodes, scorecard, and knowledge ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_knowledge import run_fast_bot_knowledge  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-ledger", type=Path, required=True)
    parser.add_argument("--outcome-ledger", type=Path, required=True)
    parser.add_argument("--challenger-ledger", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "fast_bot_corrective_challenger_v1.json",
    )
    parser.add_argument("--episode-ledger", type=Path, required=True)
    parser.add_argument("--knowledge-ledger", type=Path, required=True)
    parser.add_argument("--scorecard", type=Path, required=True)
    args = parser.parse_args()
    result = run_fast_bot_knowledge(
        shadow_ledger_path=args.shadow_ledger,
        outcome_ledger_path=args.outcome_ledger,
        challenger_ledger_path=args.challenger_ledger,
        config_path=args.config,
        episode_ledger_path=args.episode_ledger,
        knowledge_ledger_path=args.knowledge_ledger,
        scorecard_path=args.scorecard,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate and seal one AI regime/tuning supervision candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.ai_regime_supervision import write_ai_regime_supervision  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--regime",
        type=Path,
        default=ROOT / "data" / "hierarchical_bot_regime.json",
    )
    parser.add_argument(
        "--scorecard",
        type=Path,
        default=ROOT / "data" / "fast_bot_scorecard.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "ai_regime_supervision.json",
    )
    args = parser.parse_args()
    result = write_ai_regime_supervision(
        args.candidate,
        args.regime,
        args.scorecard,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": "SEALED_SUPERVISION",
                "output": str(args.output),
                "pair_count": len(result["pairs"]),
                "ai_order_authority": result["ai_order_authority"],
                "broker_mutation_allowed": result["broker_mutation_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

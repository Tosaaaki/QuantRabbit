#!/usr/bin/env python3
"""Run the zero-authority durable shock/protective-SL shadow gate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_shock_guard import load_config, run_guard_cycle  # noqa: E402


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-charts", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "fast_bot_shock_guard_v1.json")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--decision-ledger", type=Path, required=True)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair", default="EUR_USD")
    args = parser.parse_args()
    config, config_sha = load_config(args.config)
    result = run_guard_cycle(
        pair_charts=_object(args.pair_charts),
        shadow=_object(args.shadow),
        config=config,
        config_sha256=config_sha,
        state_path=args.state,
        decision_ledger_path=args.decision_ledger,
        scorecard_path=args.scorecard,
        output_path=args.output,
        now_utc=datetime.now(timezone.utc),
        pair=args.pair,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

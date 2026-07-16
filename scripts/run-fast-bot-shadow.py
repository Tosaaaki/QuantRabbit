#!/usr/bin/env python3
"""Run the deterministic 30-second bot in broker-read-only shadow mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot import run_fast_bot_shadow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-pair-charts", type=Path, default=ROOT / "data" / "position_guardian_pair_charts.json")
    parser.add_argument("--slow-pair-charts", type=Path, default=ROOT / "data" / "pair_charts.json")
    parser.add_argument("--broker-snapshot", type=Path, default=ROOT / "data" / "position_guardian_broker_snapshot.json")
    parser.add_argument("--guardian-events", type=Path, default=ROOT / "data" / "guardian_events.json")
    parser.add_argument("--ai-supervision", type=Path, default=ROOT / "data" / "ai_regime_supervision.json")
    parser.add_argument("--regime-output", type=Path, default=ROOT / "data" / "hierarchical_bot_regime.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "fast_bot_shadow.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / "data" / "fast_bot_shadow_ledger.jsonl")
    parser.add_argument("--episode-handoff", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "fast_bot_shadow_report.md")
    args = parser.parse_args()
    result = run_fast_bot_shadow(
        fast_pair_charts_path=args.fast_pair_charts,
        slow_pair_charts_path=args.slow_pair_charts,
        broker_snapshot_path=args.broker_snapshot,
        guardian_events_path=args.guardian_events,
        ai_supervision_path=args.ai_supervision,
        regime_output_path=args.regime_output,
        shadow_output_path=args.output,
        shadow_ledger_path=args.ledger,
        report_path=args.report,
        episode_handoff_path=args.episode_handoff,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

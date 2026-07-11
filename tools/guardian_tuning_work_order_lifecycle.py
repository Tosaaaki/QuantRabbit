#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.guardian_wake_dispatcher import transition_tuning_work_order  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically consume or supersede one reviewed guardian tuning work order "
            "using the same lock and compare-and-swap contract as the dispatcher."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=ROOT / "data" / "guardian_tuning_work_order.json",
    )
    parser.add_argument("--work-order-id", required=True)
    parser.add_argument("--expected-observation-id", required=True)
    parser.add_argument("--status", choices=("CONSUMED", "SUPERSEDED"), required=True)
    parser.add_argument("--consumed-by", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--experiment-result", required=True)
    parser.add_argument(
        "--experiment-evidence-ref",
        required=True,
        help="Existing evidence artifact as path#sha256=<64 hex characters>",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = transition_tuning_work_order(
        path=args.path,
        work_order_id=args.work_order_id,
        expected_observation_id=args.expected_observation_id,
        status=args.status,
        consumed_by=args.consumed_by,
        experiment_id=args.experiment_id,
        experiment_result=args.experiment_result,
        experiment_evidence_ref=args.experiment_evidence_ref,
        now=datetime.now(timezone.utc),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {
        "WORK_ORDER_TERMINAL_WRITTEN",
        "WORK_ORDER_ALREADY_TERMINAL",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())

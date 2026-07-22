#!/usr/bin/env python3
"""Prepare, run, and inspect the reviewed G2 historical DOJO TRAIN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_historical_train_control import (  # noqa: E402
    DojoHistoricalTrainControlError,
    generation_status,
    prepare_generation,
    run_next_job,
)


DEFAULT_RUN_CONTROL = REPO_ROOT / "config" / "dojo_g2_historical_run_control_v1.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run-next", "status"):
        command = commands.add_parser(name)
        command.add_argument(
            "--run-control", type=Path, default=DEFAULT_RUN_CONTROL
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_generation(
                repo_root=REPO_ROOT, run_control_path=args.run_control
            )
        elif args.command == "run-next":
            result = run_next_job(
                repo_root=REPO_ROOT, run_control_path=args.run_control
            )
        else:
            result = generation_status(
                repo_root=REPO_ROOT, run_control_path=args.run_control
            )
    except (DojoHistoricalTrainControlError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "REJECTED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "broker_mutation_allowed": False,
                    "live_permission": False,
                    "order_authority": "NONE",
                    "promotion_eligible": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

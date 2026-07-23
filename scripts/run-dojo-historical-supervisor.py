#!/usr/bin/env python3
"""Launch or inspect the session-independent historical DOJO supervisor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_historical_crash_supervisor import (  # noqa: E402
    DojoHistoricalCrashRecoveryError,
)
from quant_rabbit.dojo_historical_supervisor import (  # noqa: E402
    DojoHistoricalSupervisorError,
    launch_supervised_transition,
    run_supervisor_child,
    supervisor_status,
)
from quant_rabbit.dojo_historical_train_control import (  # noqa: E402
    DojoHistoricalTrainControlError,
)


DEFAULT_RUN_CONTROL = (
    REPO_ROOT / "config" / "dojo_g2_parallel_rooms_run_control_v6.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "status"):
        command = commands.add_parser(name)
        command.add_argument("--run-control", type=Path, default=DEFAULT_RUN_CONTROL)
    child = commands.add_parser("_child", help=argparse.SUPPRESS)
    child.add_argument("--run-control", type=Path, required=True)
    child.add_argument("--request", type=Path, required=True)
    child.add_argument("--lease-path", type=Path, required=True)
    child.add_argument("--lease-fd", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "launch":
            result = launch_supervised_transition(
                repo_root=REPO_ROOT,
                run_control_path=args.run_control,
                child_script=Path(__file__),
            )
        elif args.command == "status":
            result = supervisor_status(
                repo_root=REPO_ROOT, run_control_path=args.run_control
            )
        else:
            return run_supervisor_child(
                repo_root=REPO_ROOT,
                run_control_path=args.run_control,
                request_path=args.request,
                lease_path=args.lease_path,
                lease_descriptor=args.lease_fd,
            )
    except (
        DojoHistoricalCrashRecoveryError,
        DojoHistoricalSupervisorError,
        DojoHistoricalTrainControlError,
        OSError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "REJECTED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "partial_economics_reported": False,
                    "trainer_action_allowed": False,
                    "automatic_deployment_allowed": False,
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
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

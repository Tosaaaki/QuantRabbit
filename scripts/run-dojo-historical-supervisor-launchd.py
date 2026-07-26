#!/usr/bin/env python3
"""Bound one historical DOJO supervisor probe for launchd.

The supervisor performs deep custody verification before it can claim work. A
FileProvider-backed archive may block an ``open(2)`` indefinitely while the
remote object is not materialized. This wrapper keeps that condition from
pinning a launchd worker forever; it never changes the paper-only authority
contract or bypasses the signed remote-attestation gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = REPO_ROOT / "scripts" / "run-dojo-historical-supervisor.py"
DEFAULT_RUN_CONTROL = (
    REPO_ROOT / "config" / "dojo_g2_parallel_rooms_run_control_v6.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-control", type=Path, default=DEFAULT_RUN_CONTROL)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    command = [
        sys.executable,
        str(SUPERVISOR),
        "launch",
        "--run-control",
        str(args.run_control),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            timeout=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(
            json.dumps(
                {
                    "status": "DEFERRED",
                    "reason": "SUPERVISOR_PROBE_TIMEOUT",
                    "timeout_seconds": args.timeout_seconds,
                    "gate_bypassed": False,
                    "paper_only": True,
                    "broker_mutation_allowed": False,
                    "live_permission": False,
                    "order_authority": "NONE",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 124
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

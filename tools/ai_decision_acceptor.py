#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_rabbit.ai_decision_acceptor import monitor_candidate
from quant_rabbit.ai_trading_runtime import AIRuntimeError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ai_trading_runtime.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Accept the first completed AI decision for one prepared run.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--initial-candidate-sha256", required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.1)
    args = parser.parse_args()
    try:
        outcome = monitor_candidate(
            config_path=args.config,
            manifest_path=args.manifest,
            candidate_path=args.candidate,
            repo_root=args.repo_root,
            state_root=args.state_root,
            initial_candidate_sha256=args.initial_candidate_sha256,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    except (AIRuntimeError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "FAILED_TO_START",
            "code": getattr(exc, "code", "ACCEPTOR_START_FAILED"),
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))
    return 0 if outcome["status"] == "ACCEPTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

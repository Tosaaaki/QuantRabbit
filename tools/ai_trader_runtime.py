#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_rabbit.ai_trading_runtime import AIRuntimeError, accept_run, prepare_run


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "ai_trading_runtime.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and accept model-neutral AI trading runs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--state-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--profile", required=True)
    accept = subparsers.add_parser("accept")
    accept.add_argument("--manifest", type=Path, required=True)
    accept.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_run(
                config_path=args.config,
                profile=args.profile,
                repo_root=args.repo_root,
                state_root=args.state_root,
            )
            payload = {
                "status": "READY" if result.ready else "BLOCKED",
                "run_id": result.run_id,
                "profile": result.profile,
                "kind": result.kind,
                "manifest_path": str(result.manifest_path),
                "candidate_path": str(result.candidate_path),
                "blockers": list(result.blockers),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if result.ready else 2
        result = accept_run(
            config_path=args.config,
            manifest_path=args.manifest,
            candidate_path=args.candidate,
            repo_root=args.repo_root,
            state_root=args.state_root,
        )
        print(json.dumps({
            "status": result.status,
            "run_id": result.run_id,
            "profile": result.profile,
            "kind": result.kind,
            "receipt_path": str(result.receipt_path),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except AIRuntimeError as exc:
        print(json.dumps({"status": "REJECTED", "code": exc.code, "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

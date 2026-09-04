#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_rabbit.ai_evidence_adapter import (
    EvidenceAdapterError,
    EvidencePaths,
    write_ai_evidence_packet,
)
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
    evidence = subparsers.add_parser("build-evidence")
    evidence.add_argument("--output", type=Path, required=True)
    evidence.add_argument("--broker-snapshot", type=Path, default=Path("data/broker_snapshot.json"))
    evidence.add_argument("--pair-charts", type=Path, default=Path("data/pair_charts.json"))
    evidence.add_argument("--market-context-matrix", type=Path, default=Path("data/market_context_matrix.json"))
    evidence.add_argument("--news-health", type=Path, default=Path("data/news_health.json"))
    evidence.add_argument("--news-snapshot", type=Path, default=Path("data/news_items.json"))
    evidence.add_argument("--daily-target-state", type=Path, default=Path("data/daily_target_state.json"))
    evidence.add_argument("--capture-economics", type=Path, default=Path("data/capture_economics.json"))
    evidence.add_argument("--execution-timing", type=Path, default=Path("data/execution_timing_audit.json"))
    accept = subparsers.add_parser("accept")
    accept.add_argument("--manifest", type=Path, required=True)
    accept.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build-evidence":
            def resolved(path: Path) -> Path:
                return path if path.is_absolute() else args.repo_root / path

            result = write_ai_evidence_packet(
                EvidencePaths(
                    broker_snapshot=resolved(args.broker_snapshot),
                    pair_charts=resolved(args.pair_charts),
                    market_context_matrix=resolved(args.market_context_matrix),
                    news_health=resolved(args.news_health),
                    news_snapshot=resolved(args.news_snapshot),
                    daily_target_state=resolved(args.daily_target_state),
                    capture_economics=resolved(args.capture_economics),
                    execution_timing=resolved(args.execution_timing),
                ),
                args.output,
            )
            packet = json.loads(result.output_path.read_text(encoding="utf-8"))
            print(json.dumps({
                "status": packet.get("status", "BLOCKED"),
                "blocking_sources": packet.get("blocking_sources", []),
                "output_path": str(result.output_path),
                "packet_sha256": result.packet_sha256,
                "size_bytes": result.size_bytes,
                "written": result.written,
            }, ensure_ascii=False, sort_keys=True))
            return 0
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
    except (AIRuntimeError, EvidenceAdapterError) as exc:
        code = exc.code if isinstance(exc, AIRuntimeError) else "EVIDENCE_PACKET_REJECTED"
        print(json.dumps({"status": "REJECTED", "code": code, "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the January-development r13 AI inventory v2 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256
from quant_rabbit.dojo_r13_ai_inventory_oos import (
    B_INVENTORY_ONLY,
    C_FORECAST_INVENTORY,
    _atomic_json,
    _file_sha256,
    load_prepared_study,
)
from quant_rabbit.dojo_r13_ai_inventory_oos_v2 import (
    MAX_PHASE_B_CALLS,
    V2_CONTRACT,
    aggregate_oos_v2,
    build_factory_contract,
    build_regime_cache,
    build_strategy_regime_matrix,
    calibrate_v2,
    deterministic_oos_session_v2,
    seal_walk_forward_contract,
    worker_session_v2,
)


def initialize(*, source_root: Path, output_root: Path) -> dict[str, Any]:
    source = source_root.resolve(strict=True)
    output = output_root.resolve()
    if source == output or source in output.parents:
        raise ValueError("v2 output must be separate from immutable v1 input")
    study, _ = load_prepared_study(source)
    source_study_path = source / "study.json"
    study_bytes, study_file_sha = _file_sha256(source_study_path)
    body = {
        "contract": V2_CONTRACT,
        "schema_version": 2,
        "study_id": "r13-2025-01-ohlc-ai-inventory-oos-v2",
        "v1_prepared_input_root": str(source),
        "v1_prepared_study_sha256": study["study_sha256"],
        "v1_study_file_sha256": study_file_sha,
        "v1_study_file_bytes": study_bytes,
        "immutable_r13_baseline_was_reexecuted": False,
        "immutable_v1_was_edited_or_deleted": False,
        "january_role": (
            "MECHANISM_DISCOVERY_INTEGRATION_CALIBRATION_MONTH_"
            "NOT_FINAL_MODEL_VALIDATION"
        ),
        "source_quote_coverage_proved": False,
        "classification": (
            "EXPERIMENTAL_SAME_INCOMPLETE_SOURCE_PAIRED_DIFFERENCE"
        ),
        "arms": [
            "A_BOT_ONLY",
            "B_INVENTORY_ONLY",
            "C_FORECAST_INVENTORY",
        ],
        "authority": {
            "research_only": True,
            "paper_replay_only": True,
            "live_permission": False,
            "broker_mutation_allowed": False,
            "order_authority": "NONE",
            "automatic_deployment_allowed": False,
            "promotion_eligible": False,
        },
    }
    result = {**body, "v2_study_sha256": canonical_portfolio_sha256(body)}
    _atomic_json(output / "study-v2.json", result)
    return result


def _interactive_provider(
    *, output_root: Path, worker_id: str, worker_model: str
):
    def provider(envelope: dict[str, Any]) -> dict[str, Any]:
        envelope_path = (
            output_root
            / "worker-envelopes"
            / f"{envelope['envelope_sha256']}.json"
        )
        if envelope_path.exists():
            existing = json.loads(envelope_path.read_text(encoding="utf-8"))
            if existing != envelope:
                raise ValueError("existing Worker envelope hash collision")
        else:
            _atomic_json(envelope_path, envelope)
        print(
            json.dumps(
                {
                    "kind": "V2_WORKER_ENVELOPE",
                    "worker_id": worker_id,
                    "worker_model": worker_model,
                    "fresh_context_required": True,
                    "filesystem_network_replay_result_access_allowed": False,
                    "envelope_file": str(envelope_path.resolve()),
                    "envelope": envelope,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        payload = json.loads(input())
        if isinstance(payload, dict) and set(payload) == {"response"}:
            payload = payload["response"]
        if not isinstance(payload, dict):
            raise ValueError("Worker response must be a JSON object")
        return payload

    return provider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "initialize",
            "build-regime-cache",
            "build-regime-matrix",
            "calibrate",
            "deterministic-session",
            "worker-session",
            "aggregate",
        ),
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--partition", choices=("CALIBRATION", "OOS"))
    parser.add_argument("--coordinate-id")
    parser.add_argument("--arm")
    parser.add_argument("--session-output", type=Path)
    parser.add_argument("--sessions-root", type=Path)
    parser.add_argument("--worker-id", default="codex-fresh-worker-v2")
    parser.add_argument("--worker-model", default="gpt-5")
    parser.add_argument("--max-ai-calls", type=int, default=MAX_PHASE_B_CALLS)
    args = parser.parse_args()

    if args.command == "initialize":
        result = initialize(
            source_root=args.source_root,
            output_root=args.output_root,
        )
    elif args.command == "build-regime-cache":
        result = build_regime_cache(
            source_root=args.source_root,
            output_root=args.output_root,
        )
    elif args.command == "build-regime-matrix":
        if args.partition is None:
            parser.error("build-regime-matrix requires --partition")
        result = build_strategy_regime_matrix(
            source_root=args.source_root,
            output_root=args.output_root,
            partition=args.partition,
        )
    elif args.command == "calibrate":
        result = calibrate_v2(
            source_root=args.source_root,
            output_root=args.output_root,
        )
    elif args.command in {"deterministic-session", "worker-session"}:
        if (
            not args.coordinate_id
            or args.arm not in {B_INVENTORY_ONLY, C_FORECAST_INVENTORY}
            or args.session_output is None
        ):
            parser.error(
                "session requires --coordinate-id, --arm, and --session-output"
            )
        if args.command == "deterministic-session":
            result = deterministic_oos_session_v2(
                source_root=args.source_root,
                output_root=args.output_root,
                coordinate_id=args.coordinate_id,
                arm=args.arm,
                session_output=args.session_output,
                max_ai_calls=args.max_ai_calls,
            )
        else:
            result = worker_session_v2(
                source_root=args.source_root,
                output_root=args.output_root,
                coordinate_id=args.coordinate_id,
                arm=args.arm,
                session_output=args.session_output,
                response_provider=_interactive_provider(
                    output_root=args.output_root,
                    worker_id=args.worker_id,
                    worker_model=args.worker_model,
                ),
                worker_id=args.worker_id,
                worker_model=args.worker_model,
                max_ai_calls=args.max_ai_calls,
            )
    else:
        if args.sessions_root is None:
            parser.error("aggregate requires --sessions-root")
        result = aggregate_oos_v2(
            source_root=args.source_root,
            output_root=args.output_root,
            sessions_root=args.sessions_root,
        )
        seal_walk_forward_contract(
            output_root=args.output_root,
            oos_result=result,
        )
        build_factory_contract(
            output_root=args.output_root,
            oos_result=result,
        )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "contract": result["contract"],
                "sha256": next(
                    (
                        value
                        for key, value in result.items()
                        if key.endswith("_sha256")
                    ),
                    None,
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

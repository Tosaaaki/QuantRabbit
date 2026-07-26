#!/usr/bin/env python3
"""Seal and run the r13 paired inventory-supervision research generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_paired_inventory_counterfactual import (
    build_paired_inventory_plan,
    replay_paired_inventory_transcript,
)
from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256


def _load(path: Path) -> dict[str, Any]:
    with path.resolve(strict=True).open("rb") as handle:
        return json.load(handle)


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load(path)["coordinate_runtimes"]
    return {coordinate_id: row for coordinate_id, row in payload.items()}


def seal_plan(args: argparse.Namespace) -> int:
    job_result = _load(args.job_result)
    if (
        job_result.get("job_status") != "COMPLETE"
        or job_result.get("complete_coordinate_count") != 12
        or job_result.get("failed_coordinate_count") != 0
        or job_result.get("authority", {}).get("live_permission") is not False
        or job_result.get("authority", {}).get("broker_mutation_allowed") is not False
        or job_result.get("authority", {}).get("order_authority") != "NONE"
    ):
        raise ValueError("r13 baseline is incomplete or violates research authority")
    transcript_sha = {
        row["coordinate_id"]: row["transcript_file_sha256"]
        for row in job_result["economic_transcript_artifacts"]
    }
    plan = build_paired_inventory_plan(
        study_id="g2-r13-2025-01-ohlc-paired-inventory-v1",
        source_job_sha256=job_result["job_sha256"],
        source_job_result_sha256=job_result["economic_job_result_sha256"],
        transcript_sha256_by_coordinate=transcript_sha,
        calibration_start_epoch=1_735_768_800,
        calibration_end_epoch=1_736_546_400,
        oos_blocks=[
            {"block_id": "", "start_epoch": start, "end_epoch": end}
            for start, end in zip(
                (
                    1_736_546_400,
                    1_736_805_600,
                    1_737_064_800,
                    1_737_324_000,
                    1_737_583_200,
                    1_737_842_400,
                    1_738_015_200,
                    1_738_188_000,
                ),
                (
                    1_736_805_600,
                    1_737_064_800,
                    1_737_324_000,
                    1_737_583_200,
                    1_737_842_400,
                    1_738_015_200,
                    1_738_188_000,
                    1_738_447_200,
                ),
            )
        ],
        source_quote_coverage_proved=bool(
            job_result["source_quote_coverage_proved"]
        ),
        researcher_prior_aggregate_outcome_exposure=True,
    )
    plan["implementation_file_sha256"] = _file_sha(args.implementation)
    plan["plan_sha256"] = canonical_portfolio_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    _write_exclusive(args.output, plan)
    print(plan["plan_sha256"])
    return 0


def run_coordinate(args: argparse.Namespace) -> int:
    plan = _load(args.plan)
    job_result = _load(args.job_result)
    runtimes = _runtime_map(args.coordinate_runtimes)
    artifacts = {
        row["coordinate_id"]: row
        for row in job_result["economic_transcript_artifacts"]
    }
    portfolio_results = job_result["portfolio_results_by_coordinate"]
    coordinate_ids = (
        [args.coordinate_id]
        if args.coordinate_id
        else sorted(portfolio_results)
    )
    for coordinate_id in coordinate_ids:
        artifact = artifacts[coordinate_id]
        transcript = args.evidence_dir / artifact["transcript_filename"]
        result = replay_paired_inventory_transcript(
            transcript_path=transcript,
            plan=plan,
            baseline_result=portfolio_results[coordinate_id],
            cost_scenario=runtimes[coordinate_id]["cost_scenario"],
        )
        output = args.output_dir / f"{coordinate_id}.paired-inventory.json"
        _write_exclusive(output, result)
        print(f"{coordinate_id} {result['result_sha256']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-plan")
    seal.add_argument("--job-result", type=Path, required=True)
    seal.add_argument("--implementation", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.set_defaults(handler=seal_plan)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--job-result", type=Path, required=True)
    run.add_argument("--coordinate-runtimes", type=Path, required=True)
    run.add_argument("--evidence-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--coordinate-id")
    run.set_defaults(handler=run_coordinate)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

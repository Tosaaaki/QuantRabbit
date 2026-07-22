#!/usr/bin/env python3
"""Seal a V3 market calendar or generate/verify coverage diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_sparse_coverage_proof_v3 import (  # noqa: E402
    DojoSparseCoverageProofV3Error,
    build_coverage_proof_receipt,
    build_sealed_market_calendar_artifact,
    read_bounded_json_artifact,
    verify_coverage_proof_receipt,
    verify_sealed_market_calendar_artifact,
    write_json_exclusive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    seal = commands.add_parser(
        "seal-calendar",
        help="validate an explicit all-grid-slot calendar spec and publish it once",
    )
    seal.add_argument("--spec", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)

    generate = commands.add_parser(
        "generate",
        help="write candidate-only diagnostics for complete pair-slot references",
    )
    generate.add_argument("--calendar", type=Path, required=True)
    generate.add_argument("--coverage-input", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser(
        "verify",
        help="rebuild and compare a published candidate-only diagnostic",
    )
    verify.add_argument("--calendar", type=Path, required=True)
    verify.add_argument("--coverage-input", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "seal-calendar":
        spec = read_bounded_json_artifact(args.spec, field="calendar spec")
        calendar = build_sealed_market_calendar_artifact(spec)
        output = write_json_exclusive(args.output, calendar)
        return {
            "status": "SEALED_MARKET_CALENDAR_WRITTEN",
            "output": str(output),
            "calendar_artifact_sha256": calendar["calendar_artifact_sha256"],
            "expected_open_slot_count": calendar["expected_open_slot_count"],
            "verified_closed_slot_count": calendar["verified_closed_slot_count"],
            "authority": calendar["authority"],
        }

    calendar_value = read_bounded_json_artifact(
        args.calendar, field="sealed market calendar"
    )
    calendar = verify_sealed_market_calendar_artifact(calendar_value)
    coverage = read_bounded_json_artifact(args.coverage_input, field="coverage input")
    if args.command == "generate":
        receipt = build_coverage_proof_receipt(
            calendar_artifact=calendar,
            coverage_input=coverage,
        )
        output = write_json_exclusive(args.output, receipt)
        return {
            "status": "SPARSE_COVERAGE_DIAGNOSTIC_WRITTEN",
            "output": str(output),
            "receipt_sha256": receipt["receipt_sha256"],
            "coverage_input_sha256": receipt["coverage_input_sha256"],
            "expected_pair_slot_count": receipt["expected_pair_slot_count"],
            "observed_pair_slot_count": receipt["observed_pair_slot_count"],
            "legitimate_no_candle_pair_slot_count": receipt[
                "legitimate_no_candle_pair_slot_count"
            ],
            "proof_eligible": receipt["proof_eligible"],
            "proof_classification": receipt["proof_classification"],
            "authority": receipt["authority"],
        }

    receipt_value = read_bounded_json_artifact(
        args.receipt, field="coverage proof receipt"
    )
    receipt = verify_coverage_proof_receipt(
        receipt_value,
        calendar_artifact=calendar,
        coverage_input=coverage,
    )
    return {
        "status": "SPARSE_COVERAGE_DIAGNOSTIC_VERIFIED",
        "receipt_sha256": receipt["receipt_sha256"],
        "coverage_input_sha256": receipt["coverage_input_sha256"],
        "proof_eligible": receipt["proof_eligible"],
        "proof_classification": receipt["proof_classification"],
        "authority": receipt["authority"],
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _run(args)
    except (DojoSparseCoverageProofV3Error, OSError) as exc:
        result = {"status": "BLOCKED", "error": str(exc)}
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Versioned audit entrypoint for the trusted guardian threshold evaluator.

The content-addressed copy of this file is provenance only.  Evidence builders
and fresh queue validation execute the imported declarative implementation in
``src`` and never import or execute Python copied under ``data``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.guardian_tuning_evaluator import (  # noqa: E402
    EVALUATOR_NAME,
    FIXED_ACCEPTANCE_THRESHOLD,
    OBJECTIVE,
    PRIMARY_METRIC,
    SUPPORTED_THRESHOLD_PARAMETERS,
    evaluate_precommitted_threshold_cohort,
)


def evaluate(
    payload: dict[str, Any],
    *,
    parameter: str,
    current_value: float,
    candidate_value: float,
    primary_metric: str,
    objective: str,
    acceptance_threshold: float,
) -> dict[str, Any]:
    return evaluate_precommitted_threshold_cohort(
        payload,
        parameter=parameter,
        current_value=current_value,
        candidate_value=candidate_value,
        primary_metric=primary_metric,
        objective=objective,
        acceptance_threshold=acceptance_threshold,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one canonical forward-only confidence-floor cohort."
    )
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument(
        "--parameter",
        choices=tuple(sorted(SUPPORTED_THRESHOLD_PARAMETERS)),
        required=True,
    )
    parser.add_argument("--current-value", type=float, required=True)
    parser.add_argument("--candidate-value", type=float, required=True)
    parser.add_argument("--primary-metric", choices=(PRIMARY_METRIC,), default=PRIMARY_METRIC)
    parser.add_argument("--objective", choices=(OBJECTIVE,), default=OBJECTIVE)
    parser.add_argument(
        "--acceptance-threshold",
        type=float,
        choices=(FIXED_ACCEPTANCE_THRESHOLD,),
        default=FIXED_ACCEPTANCE_THRESHOLD,
    )
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.source_data.read_text())
        if not isinstance(payload, dict):
            raise ValueError("source data must be a JSON object")
        result = evaluate(
            payload,
            parameter=args.parameter,
            current_value=args.current_value,
            candidate_value=args.candidate_value,
            primary_metric=args.primary_metric,
            objective=args.objective,
            acceptance_threshold=args.acceptance_threshold,
        )
        code = 0
    except (OSError, OverflowError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "status": "EVALUATION_FAILED",
            "evaluator": EVALUATOR_NAME,
            "error": str(exc),
        }
        code = 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

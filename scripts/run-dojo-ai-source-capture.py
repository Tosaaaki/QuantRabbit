#!/usr/bin/env python3
"""Separate-process signer for paper-only AI source acquisition receipts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TextIO

from quant_rabbit.dojo_ai_source_capture import (
    AiSourceCaptureError,
    capture_registered_ai_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a code-owned registered read-only adapter and seal its result "
            "with a lifecycle-bound paper-only Ed25519 acquisition receipt."
        )
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-role", required=True)
    parser.add_argument("--cutoff-utc", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output = stdout or sys.stdout

    try:
        receipt = capture_registered_ai_source(
            experiment_id=args.experiment_id,
            room_id=args.room_id,
            candidate_id=args.candidate_id,
            source_role=args.source_role,
            cutoff_utc=args.cutoff_utc,
        )
    except AiSourceCaptureError as exc:
        parser.error(str(exc))
    output.write(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

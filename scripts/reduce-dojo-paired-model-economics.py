#!/usr/bin/env python3
"""Reduce complete applied paired-model replay results into economic terms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_paired_economic_reducer import (
    reduce_paired_model_economics,
)


def _read(path: Path) -> dict[str, Any]:
    with path.resolve(strict=True).open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ai-execution-cost-jpy", type=float)
    args = parser.parse_args()
    results = [_read(path) for path in sorted(args.results_dir.glob("*.json"))]
    reduced = reduce_paired_model_economics(
        results,
        ai_execution_cost_jpy=args.ai_execution_cost_jpy,
    )
    _write_exclusive(args.output, reduced)
    print(reduced["reducer_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

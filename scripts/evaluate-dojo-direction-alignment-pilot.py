#!/usr/bin/env python3
"""Evaluate a sealed PAPER direction-alignment paired pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_rabbit.dojo_direction_alignment_pilot import (
    evaluate_direction_alignment_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_direction_alignment_plan(args.plan)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

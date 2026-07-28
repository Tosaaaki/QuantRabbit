#!/usr/bin/env python3
"""Produce one idempotent Paper strategy-lab decision on stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_rabbit.paper_champion_challenger import generate_strategy_candidate


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--observed-at-utc", required=True)
    args = parser.parse_args()
    registry = _load(args.registry) if args.registry else {}
    result = generate_strategy_candidate(
        policy=_load(args.policy),
        evidence=_load(args.evidence),
        registry=registry,
        observed_at_utc=args.observed_at_utc,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

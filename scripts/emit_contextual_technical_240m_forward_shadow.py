#!/usr/bin/env python3
"""Emit the locked contextual 240-minute technical forward shadow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.contextual_technical_forward import (  # noqa: E402
    emit_forward_shadow_from_oanda,
)


def main() -> int:
    args = _parse_args()
    result = emit_forward_shadow_from_oanda(
        candidate_path=args.candidate,
        shadow_path=args.output,
        shadow_ledger_path=args.ledger,
    )
    print(json.dumps({
        "status": result.get("status"),
        "decision_id": result.get("decision_id"),
        "signal_count": len(result.get("signals") or ()),
        "shadow_only": True,
        "live_order_enabled": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "config" / "contextual_technical_240m_forward_candidate_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "contextual_technical_240m_forward_shadow.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "data" / "contextual_technical_240m_forward_shadow_ledger.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

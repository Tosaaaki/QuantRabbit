#!/usr/bin/env python3
"""Prepare high-information windows or run the archived-worker A/B replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_rabbit.dojo_legacy_worker_comparison import (
    build_loss_window_packet,
    load_archived_candles,
    load_result_ledger,
    run_comparison,
)


ARCHIVE = Path(
    "/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z"
)
RESULT = ARCHIVE / "logs/archive_legacy/backtest_20251001_20251022_full.json"
CANDLE_DIR = ARCHIVE / "logs/archive_legacy"
CODE = {
    "M1Scalper": ARCHIVE / "archive/strategies/scalping/m1_scalper.py",
    "PulseBreak": ARCHIVE / "archive/strategies/scalping/pulse_break.py",
    "RangeFader": ARCHIVE / "archive/strategies/scalping/range_fader.py",
    "TrendMA": ARCHIVE / "archive/strategies/trend/ma_cross.py",
}


def _candles() -> list[Path]:
    return sorted(CANDLE_DIR.glob("candles_M1_202510*.json"))[:19]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    if args.command == "prepare":
        result = load_result_ledger(RESULT)
        result = build_loss_window_packet(
            result_ledger=result,
            candles=load_archived_candles(_candles()),
            result_path=RESULT,
            code_paths=CODE,
        )
    else:
        if args.policy is None:
            parser.error("evaluate requires --policy")
        result = run_comparison(
            result_path=RESULT,
            candle_paths=_candles(),
            policy_path=args.policy,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "output": str(args.output), "sha256": result.get("packet_sha256") or result.get("result_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Advance diagnostic fast-bot episodes from one sealed primary-cycle handoff."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot import load_fast_bot_episode_handoff  # noqa: E402
from quant_rabbit.fast_bot_episode import run_fast_bot_episode_shadow  # noqa: E402


SUCCESS_STATUSES = {"UPDATED", "NO_NEW_EVENT"}


def _cycle_time(value: object) -> datetime:
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("episode handoff cycle clock must be UTC-aware")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "fast_bot_episode_state.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / "data" / "fast_bot_episode_ledger.jsonl")
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=ROOT / "data" / "fast_bot_episode_sources",
    )
    args = parser.parse_args()

    if os.environ.get("QR_LIVE_ENABLED", "0") != "0":
        print("fast-bot episode runner requires QR_LIVE_ENABLED=0", file=sys.stderr)
        return 2
    if os.environ.get("QR_AUTOTRADE_LOCK_HELD", "0") != "0":
        print("fast-bot episode runner refuses the shared live lock", file=sys.stderr)
        return 2
    if os.environ.get("QR_AUTOTRADE_LOCK_OWNER_TOKEN"):
        print("fast-bot episode runner refuses a live-lock owner token", file=sys.stderr)
        return 2

    try:
        handoff = load_fast_bot_episode_handoff(args.handoff)
        cycle_time = _cycle_time(handoff["cycle_generated_at_utc"])
        processed_at = datetime.now(timezone.utc)
        result: object = {}
        for recovery_attempt in range(2):
            result = run_fast_bot_episode_shadow(
                regime_contract=handoff["regime_contract"],
                fast_pair_charts=handoff["fast_pair_charts"],
                slow_pair_charts=handoff["slow_pair_charts"],
                output_path=args.output,
                ledger_path=args.ledger,
                source_archive_dir=args.source_archive,
                now_utc=cycle_time,
                processed_at_utc=processed_at,
            )
            status = (
                str(result.get("status") or "")
                if isinstance(result, dict)
                else ""
            )
            if status == "RECOVERED_PENDING_BATCH" and recovery_attempt == 0:
                continue
            break
    except (OSError, TypeError, ValueError) as error:
        print(
            f"fast-bot episode handoff failed: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if status in SUCCESS_STATUSES:
        return 0
    if status == "LOCK_BUSY":
        return 75
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

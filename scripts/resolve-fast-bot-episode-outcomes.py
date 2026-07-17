#!/usr/bin/env python3
"""Project confirmed episodes and resolve mature exact-S5 shadow vehicles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SUCCESS_STATUSES = {
    "PROJECTED_NO_DUE",
    "NO_DUE_VEHICLES",
    "RESOLVED",
    "RESOLVED_WITH_ERRORS",
}


def derive_episode_truth_paths(episode_ledger_path: Path) -> dict[str, Path]:
    """Derive every truth artifact from one episode-ledger sibling prefix."""

    ledger = episode_ledger_path.resolve(strict=False)
    stem = ledger.stem
    prefix = stem[: -len("_ledger")] if stem.endswith("_ledger") else stem
    if not prefix:
        raise ValueError("episode ledger name has no truth-artifact prefix")
    parent = ledger.parent
    return {
        "vehicle_ledger_path": parent / f"{prefix}_vehicle_ledger.jsonl",
        "outcome_ledger_path": parent / f"{prefix}_outcome_ledger.jsonl",
        "scorecard_path": parent / f"{prefix}_scorecard.json",
        "lock_path": parent / f"{prefix}_truth.lock",
    }


def _runtime_safety_error() -> str | None:
    if os.environ.get("QR_LIVE_ENABLED", "0") != "0":
        return "episode outcome resolver requires QR_LIVE_ENABLED=0"
    if os.environ.get("QR_AUTOTRADE_LOCK_HELD", "0") != "0":
        return "episode outcome resolver refuses the shared live lock"
    if os.environ.get("QR_AUTOTRADE_LOCK_OWNER_TOKEN"):
        return "episode outcome resolver refuses a live-lock owner token"
    return None


def _load_truth_dependencies():
    # Keep imports behind the runtime-safety check.  The detached worker is the
    # only production caller, and this boundary makes tests able to prove that
    # an unsafe invocation cannot construct a broker client or touch a ledger.
    from quant_rabbit.broker.oanda import OandaReadOnlyClient
    from quant_rabbit.fast_bot_episode_truth import (
        run_fast_bot_episode_truth_cycle,
    )

    return run_fast_bot_episode_truth_cycle, OandaReadOnlyClient


def run_episode_outcome_resolution(
    *,
    handoffs: Sequence[Mapping[str, Any]] = (),
    episode_ledger_path: Path,
    source_archive_dir: Path,
) -> dict[str, Any]:
    """Run one lock-bounded projection/resolution cycle with GET-only OANDA."""

    safety_error = _runtime_safety_error()
    if safety_error is not None:
        return {
            "status": "RUNTIME_SAFETY_REJECTED",
            "vehicle_projection_status": "FAILED",
            "broker_read": False,
            "ledger_appended": 0,
            "error": safety_error,
            "order_authority": "NONE",
            "shadow_only": True,
            "live_permission": False,
            "broker_mutation_allowed": False,
        }
    truth_cycle, read_only_client = _load_truth_dependencies()
    paths = derive_episode_truth_paths(episode_ledger_path)
    return truth_cycle(
        handoffs=tuple(handoffs),
        episode_ledger_path=episode_ledger_path.resolve(strict=False),
        source_archive_dir=source_archive_dir.resolve(strict=False),
        vehicle_ledger_path=paths["vehicle_ledger_path"],
        outcome_ledger_path=paths["outcome_ledger_path"],
        scorecard_path=paths["scorecard_path"],
        lock_path=paths["lock_path"],
        client_factory=read_only_client,
    )


def exit_code_for_result(result: Mapping[str, Any]) -> int:
    status = str(result.get("status") or "")
    if status in SUCCESS_STATUSES:
        return 0
    if status == "LOCK_BUSY":
        return 75
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-ledger",
        type=Path,
        default=ROOT / "data" / "fast_bot_episode_ledger.jsonl",
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=ROOT / "data" / "fast_bot_episode_sources",
    )
    parser.add_argument("--handoff", type=Path, action="append", default=[])
    args = parser.parse_args()

    safety_error = _runtime_safety_error()
    if safety_error is not None:
        print(safety_error, file=sys.stderr)
        return 2

    from quant_rabbit.fast_bot import load_fast_bot_episode_handoff

    try:
        handoffs = tuple(load_fast_bot_episode_handoff(path) for path in args.handoff)
        result = run_episode_outcome_resolution(
            handoffs=handoffs,
            episode_ledger_path=args.episode_ledger,
            source_archive_dir=args.source_archive,
        )
    except (OSError, TypeError, ValueError) as error:
        print(
            f"fast-bot episode outcome resolution failed: {type(error).__name__}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return exit_code_for_result(result)


if __name__ == "__main__":
    raise SystemExit(main())

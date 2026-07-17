#!/usr/bin/env python3
"""Build a sealed exact-28 historical S5 source manifest for research only.

The explicit ``__main__`` guard is required because the strict cache scanner may
use multiprocessing with the ``spawn`` start method.  This command never loads
or persists raw candles; it freezes acquisition metadata and content hashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_historical_s5 import (  # noqa: E402
    EXPLICIT_RUN_SCOPE_POLICY,
    HistoricalS5CacheError,
    build_historical_s5_manifest,
    write_historical_s5_manifest,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allowed-run-id",
        action="append",
        required=True,
        dest="allowed_run_ids",
        help="Repeat for every acquisition run admitted to the exact source set.",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        required=True,
        help="Explicit bounded process count; use 1 for a serial verification.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Build, validate, and persist the metadata-only exact-28 manifest."""

    manifest = build_historical_s5_manifest(
        args.history_root,
        pairs=DEFAULT_TRADER_PAIRS,
        allowed_run_ids=tuple(args.allowed_run_ids),
        scan_workers=args.scan_workers,
    )
    blockers: list[str] = []
    if manifest.get("expected_pair_count") != len(DEFAULT_TRADER_PAIRS):
        blockers.append("EXPECTED_PAIR_COUNT_MISMATCH")
    if manifest.get("selected_pair_count") != len(DEFAULT_TRADER_PAIRS):
        blockers.append("SELECTED_PAIR_COUNT_MISMATCH")
    if manifest.get("complete_pair_coverage") is not True:
        blockers.append("INCOMPLETE_PAIR_COVERAGE")
    if manifest.get("missing_pairs") != []:
        blockers.append("MISSING_PAIRS_PRESENT")
    if manifest.get("all_selected_sources_acquisition_receipted") is not True:
        blockers.append("ACQUISITION_RECEIPT_GAP")
    if manifest.get("run_scope_policy") != EXPLICIT_RUN_SCOPE_POLICY:
        blockers.append("RUN_SCOPE_POLICY_MISMATCH")
    if manifest.get("summary_run_ids_exact_set_proved") is not True:
        blockers.append("SUMMARY_RUN_SCOPE_NOT_EXACT")
    if manifest.get("historical_only") is not True:
        blockers.append("HISTORICAL_ONLY_FLAG_MISSING")
    if manifest.get("forward_proof_eligible") is not False:
        blockers.append("FORWARD_PROOF_FLAG_UNSAFE")
    if manifest.get("order_authority") != "NONE":
        blockers.append("ORDER_AUTHORITY_UNSAFE")
    if blockers:
        raise HistoricalS5CacheError(
            "exact-28 historical manifest failed closed: " + ",".join(blockers)
        )

    write_historical_s5_manifest(args.output, manifest)
    return {
        "status": "VERIFIED",
        "output": str(args.output.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "expected_pair_count": manifest["expected_pair_count"],
        "selected_pair_count": manifest["selected_pair_count"],
        "allowed_run_ids": manifest["allowed_run_ids"],
        "common_declared_from_utc": manifest["common_declared_from_utc"],
        "common_declared_to_utc": manifest["common_declared_to_utc"],
        "all_selected_sources_acquisition_receipted": manifest[
            "all_selected_sources_acquisition_receipted"
        ],
        "raw_candles_persisted": False,
        "historical_only": True,
        "forward_proof_eligible": False,
        "order_authority": "NONE",
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary = run(_parse_args(argv))
    except (HistoricalS5CacheError, OSError, ValueError) as error:
        cause = error.__cause__
        cause_text = f"; cause={cause!r}" if cause is not None else ""
        print(f"BLOCKED: {error}{cause_text}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

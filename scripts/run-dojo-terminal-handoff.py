#!/usr/bin/env python3
"""Validate and lineage-bind one complete DOJO trainer terminal bundle.

This local research command requires run.json, evaluation.json, and cells.json
from one terminal directory.  It has no model, replay-runner, Drive, broker,
order, promotion, or live-trading capability.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_ai_trainer_packet import (  # noqa: E402
    DojoAITrainerPacketError,
)
from quant_rabbit.dojo_candidate_lineage_registry import (  # noqa: E402
    CandidateLineageError,
)
from quant_rabbit.dojo_terminal_handoff import (  # noqa: E402
    LINEAGE_PRESENT_AT_HANDOFF,
    RETROSPECTIVE_ADMIN_BINDING,
    DojoTerminalHandoffError,
    coordinate_terminal_handoff,
    receipt_store_status,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    bind = commands.add_parser("bind", help="validate and bind one terminal bundle")
    bind.add_argument("--terminal-dir", type=Path, required=True)
    bind.add_argument("--sealed-study", type=Path, required=True)
    bind.add_argument("--lineage-events", type=Path, required=True)
    bind.add_argument("--artifact-root", type=Path, required=True)
    bind.add_argument("--receipt-events", type=Path, required=True)
    bind.add_argument("--expected-lineage-tip-sha256", required=True)
    bind.add_argument("--expected-receipt-tip-sha256", required=True)
    bind.add_argument(
        "--binding-timing-classification",
        choices=(LINEAGE_PRESENT_AT_HANDOFF, RETROSPECTIVE_ADMIN_BINDING),
        required=True,
    )
    bind.add_argument("--registry-id")
    bind.add_argument("--lineage-prefix")
    bind.add_argument("--created-by")
    bind.add_argument(
        "--event-at-utc",
        help="Strict aware instant; defaults to the current UTC clock.",
    )

    status = commands.add_parser("status", help="verify the receipt SHA chain")
    status.add_argument("--receipt-events", type=Path, required=True)
    return parser


def _canonical_line(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _event_at(value: str | None) -> datetime | str:
    return value if value is not None else datetime.now(timezone.utc)


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "status":
        return receipt_store_status(args.receipt_events)
    if args.command != "bind":
        raise DojoTerminalHandoffError("unsupported terminal hand-off command")
    return coordinate_terminal_handoff(
        terminal_dir=args.terminal_dir,
        sealed_study_path=args.sealed_study,
        lineage_events_dir=args.lineage_events,
        artifact_root=args.artifact_root,
        receipt_events_dir=args.receipt_events,
        expected_lineage_tip_sha256=args.expected_lineage_tip_sha256,
        expected_receipt_tip_sha256=args.expected_receipt_tip_sha256,
        binding_timing_classification=args.binding_timing_classification,
        event_at_utc=_event_at(args.event_at_utc),
        registry_id=args.registry_id,
        lineage_prefix=args.lineage_prefix,
        created_by=args.created_by,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except (
        CandidateLineageError,
        DojoAITrainerPacketError,
        DojoTerminalHandoffError,
        OSError,
    ) as exc:
        print(
            _canonical_line(
                {
                    "status": "REJECTED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "proof_eligible": False,
                    "promotion_eligible": False,
                    "live_permission": False,
                    "order_authority": "NONE",
                    "broker_mutation_allowed": False,
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(_canonical_line(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

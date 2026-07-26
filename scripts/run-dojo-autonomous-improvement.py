#!/usr/bin/env python3
"""CLI for the paper-only autonomous DOJO evidence lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.analysis.market_status import compute_market_status  # noqa: E402
from quant_rabbit.dojo_autonomous_improvement import (  # noqa: E402
    append_candidate_event,
    append_shadow_assessment,
    append_shadow_outcome,
    initialize_research_root,
    validate_research_root,
)


DEFAULT_ROOT = Path("research/data/dojo_autonomous_improvement_v1")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("input JSON must be an object")
    return value


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include timezone information")
    return parsed.astimezone(timezone.utc)


def _assert_assessment_market_open(
    payload: dict,
    *,
    recorded_at_utc: datetime,
) -> None:
    """Fail closed instead of creating a new AI assessment while FX is closed."""

    as_of_utc = _parse_utc(payload.get("as_of_utc"), field="as_of_utc")
    recorded_status = compute_market_status(recorded_at_utc)
    as_of_status = compute_market_status(as_of_utc)
    if not recorded_status.is_fx_open or not as_of_status.is_fx_open:
        reason = recorded_status.closed_reason or as_of_status.closed_reason
        raise ValueError(
            "AI_ASSESSMENT_MARKET_CLOSED: "
            f"new assessment disabled while FX is closed ({reason})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("init")
    assessment = commands.add_parser("append-assessment")
    assessment.add_argument("input", type=Path)
    outcome = commands.add_parser("append-outcome")
    outcome.add_argument("input", type=Path)
    event = commands.add_parser("append-candidate-event")
    event.add_argument("event_type")
    event.add_argument("input", type=Path)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)

    if args.command == "validate":
        result = validate_research_root(args.root)
    elif args.command == "init":
        implementation = (
            Path(__file__).read_bytes()
            + (
                REPO_ROOT / "src/quant_rabbit/dojo_autonomous_improvement.py"
            ).read_bytes()
        )
        row, appended = initialize_research_root(
            args.root,
            recorded_at_utc=now,
            implementation_sha256=hashlib.sha256(implementation).hexdigest(),
        )
        result = {"appended": appended, "event_sha256": row["event_sha256"]}
    elif args.command == "append-assessment":
        payload = _json(args.input)
        _assert_assessment_market_open(payload, recorded_at_utc=now)
        row, appended = append_shadow_assessment(
            args.root / "ai_shadow_ledger.jsonl",
            payload,
            recorded_at_utc=now,
        )
        result = {"appended": appended, "event_sha256": row["event_sha256"]}
    elif args.command == "append-outcome":
        row, appended = append_shadow_outcome(
            args.root / "ai_shadow_ledger.jsonl",
            _json(args.input),
            recorded_at_utc=now,
        )
        result = {"appended": appended, "event_sha256": row["event_sha256"]}
    else:
        row, appended = append_candidate_event(
            args.root,
            event_type=args.event_type,
            payload=_json(args.input),
            recorded_at_utc=now,
        )
        result = {"appended": appended, "event_sha256": row["event_sha256"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

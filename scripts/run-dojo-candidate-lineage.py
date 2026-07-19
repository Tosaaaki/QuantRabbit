#!/usr/bin/env python3
"""Operate the local first-write DOJO candidate lineage registry."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_rabbit.dojo_candidate_lineage_registry import (  # noqa: E402
    CandidateLineageError,
    bind_result,
    initialize_registry,
    seal_study_attempt,
    status_artifact,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _event_at(value: str | None) -> datetime | str:
    return value if value is not None else _now_utc()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO,
        help="Real directory under which every bound artifact must remain.",
    )


def _add_event_time(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--event-at-utc",
        help="Canonical UTC Z timestamp; defaults to the current UTC clock.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Create the lineage genesis event.")
    _add_common(init)
    init.add_argument("--registry-id", required=True)
    init.add_argument("--lineage-prefix", required=True)
    init.add_argument("--created-by", required=True)
    _add_event_time(init)

    study = commands.add_parser(
        "seal-study", help="Bind the next pre-sealed trainer study."
    )
    _add_common(study)
    study.add_argument("--sealed-study", type=Path, required=True)
    study.add_argument("--expected-tip-sha256", required=True)
    study.add_argument("--previous-evaluation-sha256")
    study.add_argument("--previous-evaluation-artifact-sha256")
    study.add_argument("--previous-evaluation-artifact-size-bytes", type=int)
    _add_event_time(study)

    result = commands.add_parser(
        "bind-result", help="Bind the pending verified trainer evaluation."
    )
    _add_common(result)
    result.add_argument("--evaluation", type=Path, required=True)
    result.add_argument("--expected-tip-sha256", required=True)
    _add_event_time(result)

    status = commands.add_parser("status", help="Verify and print derived state.")
    _add_common(status)
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    events_dir = args.events_dir.absolute()
    artifact_root = args.artifact_root.absolute()
    if args.command == "init":
        initialize_registry(
            events_dir,
            artifact_root=artifact_root,
            registry_id=args.registry_id,
            lineage_prefix=args.lineage_prefix,
            created_by=args.created_by,
            event_at_utc=_event_at(args.event_at_utc),
        )
    elif args.command == "seal-study":
        seal_study_attempt(
            events_dir,
            artifact_root=artifact_root,
            sealed_study_path=args.sealed_study,
            expected_tip_sha256=args.expected_tip_sha256,
            event_at_utc=_event_at(args.event_at_utc),
            previous_evaluation_sha256=args.previous_evaluation_sha256,
            previous_evaluation_artifact_sha256=(
                args.previous_evaluation_artifact_sha256
            ),
            previous_evaluation_artifact_size_bytes=(
                args.previous_evaluation_artifact_size_bytes
            ),
        )
    elif args.command == "bind-result":
        bind_result(
            events_dir,
            artifact_root=artifact_root,
            evaluation_path=args.evaluation,
            expected_tip_sha256=args.expected_tip_sha256,
            event_at_utc=_event_at(args.event_at_utc),
        )
    elif args.command != "status":
        raise CandidateLineageError("unsupported candidate lineage command")
    return status_artifact(events_dir, artifact_root=artifact_root)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except (CandidateLineageError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

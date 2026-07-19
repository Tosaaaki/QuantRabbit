#!/usr/bin/env python3
"""Operate the local first-write DOJO historical holdout burn registry."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_rabbit.dojo_holdout_burn import (  # noqa: E402
    HoldoutBurnError,
    append_burn_intent,
    append_legacy_burn,
    bind_result,
    initialize_registry,
    reserve_holdout,
    status_artifact,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _event_at(value: str | None) -> datetime | str:
    return value if value is not None else _now_utc()


def _add_event_time(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--event-at-utc",
        help="Canonical UTC Z timestamp; defaults to the current UTC clock.",
    )


def _add_window(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instrument", action="append", required=True)
    parser.add_argument("--target-outcome-domain", required=True)
    parser.add_argument("--window-from-utc", required=True)
    parser.add_argument("--window-to-utc", required=True)
    parser.add_argument("--granularity", required=True)
    parser.add_argument("--input-modality", action="append", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Create the genesis event.")
    init.add_argument("--events-dir", type=Path, required=True)
    init.add_argument("--registry-id", required=True)
    init.add_argument("--created-by", required=True)
    _add_event_time(init)

    legacy = commands.add_parser(
        "legacy-burn", help="Record a historically exposed outcome window."
    )
    legacy.add_argument("--events-dir", type=Path, required=True)
    legacy.add_argument("--burn-id", required=True)
    legacy.add_argument("--task-kind", required=True)
    legacy.add_argument("--selection-lineage-id", required=True)
    _add_window(legacy)
    legacy.add_argument("--legacy-source", required=True)
    legacy.add_argument("--legacy-evidence-sha256", required=True)
    legacy.add_argument("--reason", required=True)
    _add_event_time(legacy)

    reserve = commands.add_parser(
        "reserve", help="Reserve a locally unopened historical outcome window."
    )
    reserve.add_argument("--events-dir", type=Path, required=True)
    reserve.add_argument("--reservation-id", required=True)
    reserve.add_argument("--candidate-id", required=True)
    reserve.add_argument("--task-kind", required=True)
    reserve.add_argument("--family-id", required=True)
    reserve.add_argument("--selection-lineage-id", required=True)
    reserve.add_argument("--unused-relative-to", required=True)
    _add_window(reserve)
    reserve.add_argument("--prompt-set-sha256", required=True)
    reserve.add_argument("--model-policy-sha256", required=True)
    reserve.add_argument("--scorer-sha256", required=True)
    reserve.add_argument("--code-sha256", required=True)
    reserve.add_argument("--corpus-manifest-sha256", required=True)
    reserve.add_argument("--custody-policy-sha256", required=True)
    _add_event_time(reserve)

    burn = commands.add_parser(
        "burn-intent", help="Permanently burn immediately before reveal."
    )
    burn.add_argument("--events-dir", type=Path, required=True)
    burn.add_argument("--reservation-id", required=True)
    burn.add_argument("--reveal-material-sha256", required=True)
    burn.add_argument("--revealed-by", required=True)
    burn.add_argument(
        "--acknowledge-permanent-consumption",
        action="store_true",
        required=True,
    )
    _add_event_time(burn)

    result = commands.add_parser("result-bound", help="Bind result bytes.")
    result.add_argument("--events-dir", type=Path, required=True)
    result.add_argument("--reservation-id", required=True)
    result.add_argument("--result-sha256", required=True)
    result.add_argument("--result-contract", required=True)
    result.add_argument("--bound-by", required=True)
    _add_event_time(result)

    status = commands.add_parser("status", help="Verify and print derived state.")
    status.add_argument("--events-dir", type=Path, required=True)
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    events_dir = args.events_dir.absolute()
    if args.command == "init":
        initialize_registry(
            events_dir,
            registry_id=args.registry_id,
            created_by=args.created_by,
            event_at_utc=_event_at(args.event_at_utc),
        )
    elif args.command == "legacy-burn":
        append_legacy_burn(
            events_dir,
            burn={
                "burn_id": args.burn_id,
                "task_kind": args.task_kind,
                "selection_lineage_id": args.selection_lineage_id,
                "instruments": args.instrument,
                "target_outcome_domain": args.target_outcome_domain,
                "window_from_utc": args.window_from_utc,
                "window_to_utc": args.window_to_utc,
                "granularity": args.granularity,
                "input_modalities": args.input_modality,
                "legacy_source": args.legacy_source,
                "legacy_evidence_sha256": args.legacy_evidence_sha256,
                "reason": args.reason,
            },
            event_at_utc=_event_at(args.event_at_utc),
        )
    elif args.command == "reserve":
        reserve_holdout(
            events_dir,
            reservation={
                "reservation_id": args.reservation_id,
                "candidate_id": args.candidate_id,
                "task_kind": args.task_kind,
                "family_id": args.family_id,
                "selection_lineage_id": args.selection_lineage_id,
                "unused_relative_to": args.unused_relative_to,
                "instruments": args.instrument,
                "target_outcome_domain": args.target_outcome_domain,
                "window_from_utc": args.window_from_utc,
                "window_to_utc": args.window_to_utc,
                "granularity": args.granularity,
                "input_modalities": args.input_modality,
                "prompt_set_sha256": args.prompt_set_sha256,
                "model_policy_sha256": args.model_policy_sha256,
                "scorer_sha256": args.scorer_sha256,
                "code_sha256": args.code_sha256,
                "corpus_manifest_sha256": args.corpus_manifest_sha256,
                "custody_policy_sha256": args.custody_policy_sha256,
            },
            event_at_utc=_event_at(args.event_at_utc),
        )
    elif args.command == "burn-intent":
        append_burn_intent(
            events_dir,
            intent={
                "reservation_id": args.reservation_id,
                "reveal_material_sha256": args.reveal_material_sha256,
                "revealed_by": args.revealed_by,
                "permanence_acknowledged": args.acknowledge_permanent_consumption,
            },
            event_at_utc=_event_at(args.event_at_utc),
        )
    elif args.command == "result-bound":
        bind_result(
            events_dir,
            result={
                "reservation_id": args.reservation_id,
                "result_sha256": args.result_sha256,
                "result_contract": args.result_contract,
                "bound_by": args.bound_by,
            },
            event_at_utc=_event_at(args.event_at_utc),
        )
    elif args.command != "status":
        raise HoldoutBurnError("unsupported holdout registry command")
    return status_artifact(events_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except (HoldoutBurnError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

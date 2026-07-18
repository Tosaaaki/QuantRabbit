#!/usr/bin/env python3
"""Build or score the exact locked 90-cell DOJO prompt experiment phase."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.dojo_prompt_phase import (
    PHASE_RANKS,
    build_phase_manifest,
    score_prompt_phase,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO / "research/registries/dojo_prompt_experiment_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    manifest_parser.add_argument(
        "--phase-id", choices=sorted(PHASE_RANKS), required=True
    )
    manifest_parser.add_argument(
        "--assignment", type=Path, action="append", required=True
    )
    manifest_parser.add_argument("--locked-at-utc")
    manifest_parser.add_argument("--output", type=Path, required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    score_parser.add_argument("--manifest", type=Path, required=True)
    score_parser.add_argument(
        "--score-cell",
        action="append",
        default=[],
        metavar="CELL_ID=PATH",
    )
    score_parser.add_argument(
        "--failure-cell",
        action="append",
        default=[],
        metavar="CELL_ID=REASON",
    )
    score_parser.add_argument("--sealed-at-utc")
    score_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    registry = _read_object(args.registry)
    if args.command == "build-manifest":
        manifest = build_phase_manifest(
            registry,
            phase_id=args.phase_id,
            assignments=[_read_object(path) for path in args.assignment],
            locked_at_utc=_parse_or_now(args.locked_at_utc),
        )
        _exclusive_write_json(args.output, manifest)
        result = {
            "status": "DOJO_AI_PHASE_MANIFEST_LOCKED",
            "phase_id": manifest["phase_id"],
            "allocated_cell_count": manifest["allocated_cell_count"],
            "phase_manifest_sha256": manifest["phase_manifest_sha256"],
            "output": str(args.output),
            "live_permission": False,
        }
    else:
        scored = _parse_score_cells(args.score_cell)
        failures = _parse_failure_cells(args.failure_cell)
        score = score_prompt_phase(
            registry,
            _read_object(args.manifest),
            scored_cells=scored,
            terminal_failures=failures,
            sealed_at_utc=_parse_or_now(args.sealed_at_utc),
        )
        _exclusive_write_json(args.output, score)
        result = {
            "status": "DOJO_AI_PHASE_SCORED_DIAGNOSTIC",
            "phase_id": score["phase_id"],
            "allocated_cell_count": score["allocated_cell_count"],
            "valid_response_cell_count": score["valid_response_cell_count"],
            "response_failure_cell_count": score["response_failure_cell_count"],
            "phase_score_sha256": score["phase_score_sha256"],
            "prompt_selection_allowed": False,
            "output": str(args.output),
            "live_permission": False,
        }
    print(json.dumps(result, sort_keys=True))
    return 0


def _parse_score_cells(values: list[str]) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for value in values:
        cell_id, separator, path_text = value.partition("=")
        if not separator or not cell_id or cell_id in parsed:
            raise ValueError("--score-cell must use unique CELL_ID=PATH values")
        parsed[cell_id] = _read_object(Path(path_text))
    return parsed


def _parse_failure_cells(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        cell_id, separator, reason = value.partition("=")
        if not separator or not cell_id or not reason or cell_id in parsed:
            raise ValueError("--failure-cell must use unique CELL_ID=REASON values")
        parsed[cell_id] = reason
    return parsed


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parse_or_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())

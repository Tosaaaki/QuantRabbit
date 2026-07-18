#!/usr/bin/env python3
"""Operate the append-only DOJO worker prospective-smoke lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_rabbit.dojo_worker_forward import (  # noqa: E402
    audit_lifecycle,
    build_day_seal,
    build_final_receipt,
    build_precommit,
    build_start_receipt,
    validate_precommit,
    validate_result_manifest,
    validate_start_receipt,
    write_new_json,
)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _load_parents(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    precommit = validate_precommit(_strict_json(run_dir / "precommit.json"))
    start = validate_start_receipt(_strict_json(run_dir / "start.json"), precommit)
    return precommit, start


def _precommit(args: argparse.Namespace) -> dict[str, Any]:
    artifact = build_precommit(_strict_json(args.spec), now_utc=_now())
    write_new_json(args.run_dir / "precommit.json", artifact)
    return artifact


def _start(args: argparse.Namespace) -> dict[str, Any]:
    precommit = validate_precommit(_strict_json(args.run_dir / "precommit.json"))
    receipt = build_start_receipt(precommit, now_utc=_now())
    write_new_json(args.run_dir / "start.json", receipt)
    return receipt


def _seal_day(args: argparse.Namespace) -> dict[str, Any]:
    precommit, start = _load_parents(args.run_dir)
    previous = None
    if args.ordinal > 1:
        previous = _strict_json(
            args.run_dir / "days" / f"day-{args.ordinal - 1:03d}.json"
        )
    receipt = build_day_seal(
        precommit,
        start,
        previous,
        _strict_json(args.source_manifest),
        ordinal=args.ordinal,
        now_utc=_now(),
    )
    write_new_json(args.run_dir / "days" / f"day-{args.ordinal:03d}.json", receipt)
    return receipt


def _finalize(args: argparse.Namespace) -> dict[str, Any]:
    precommit, start = _load_parents(args.run_dir)
    days = [
        _strict_json(path)
        for path in sorted((args.run_dir / "days").glob("day-*.json"))
    ]
    normalized_results = validate_result_manifest(
        _strict_json(args.result_manifest), precommit
    )
    receipt = build_final_receipt(
        precommit,
        start,
        days,
        normalized_results,
        now_utc=_now(),
    )
    write_new_json(args.run_dir / "result-manifest.json", normalized_results)
    write_new_json(args.run_dir / "final.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    precommit = subparsers.add_parser(
        "precommit", help="freeze candidates and mechanics before the window"
    )
    precommit.add_argument("--spec", type=Path, required=True)
    precommit.add_argument("--run-dir", type=Path, required=True)
    precommit.set_defaults(handler=_precommit)

    start = subparsers.add_parser("start", help="seal start before the first day")
    start.add_argument("--run-dir", type=Path, required=True)
    start.set_defaults(handler=_start)

    seal_day = subparsers.add_parser(
        "seal-day", help="seal one complete UTC market-source day in order"
    )
    seal_day.add_argument("--run-dir", type=Path, required=True)
    seal_day.add_argument("--ordinal", type=int, required=True)
    seal_day.add_argument("--source-manifest", type=Path, required=True)
    seal_day.set_defaults(handler=_seal_day)

    finalize = subparsers.add_parser(
        "finalize", help="score all fixed candidates after the window matures"
    )
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--result-manifest", type=Path, required=True)
    finalize.set_defaults(handler=_finalize)

    status = subparsers.add_parser("status", help="audit without mutating evidence")
    status.add_argument("--run-dir", type=Path, required=True)
    status.set_defaults(
        handler=lambda args: audit_lifecycle(args.run_dir, now_utc=_now())
    )

    args = parser.parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

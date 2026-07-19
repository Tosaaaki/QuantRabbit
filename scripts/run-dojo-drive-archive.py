#!/usr/bin/env python3
"""Plan, finalize, or verify a local DOJO tar.zst archive chunk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_drive_archive import (  # noqa: E402
    DojoDriveArchiveError,
    finalize_archive,
    plan_archive,
    verify_finalized_archive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="seal a cell or month archive plan")
    plan.add_argument("--source-run", type=Path, required=True)
    plan.add_argument("--destination", type=Path, required=True)
    plan.add_argument("--chunk-kind", choices=("cell", "month"), required=True)
    plan.add_argument("--chunk-id", required=True)

    finalize = commands.add_parser("finalize", help="build and publish tar.zst")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--zstd-bin", default="zstd")

    verify = commands.add_parser("verify", help="verify a finalized local chunk")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--zstd-bin", default="zstd")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "plan":
            result = plan_archive(
                source_run=args.source_run,
                destination=args.destination,
                chunk_kind=args.chunk_kind,
                chunk_id=args.chunk_id,
            )
        elif args.command == "finalize":
            result = finalize_archive(plan_path=args.plan, zstd_bin=args.zstd_bin)
        else:
            result = verify_finalized_archive(
                plan_path=args.plan, zstd_bin=args.zstd_bin
            )
    except DojoDriveArchiveError as exc:
        print(json.dumps({"status": "REJECTED", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

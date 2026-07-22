#!/usr/bin/env python3
"""Verify or execute one externally attested historical DOJO raw reclaim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_historical_raw_reclaim import (
    reclaim_historical_job_raw,
    verify_historical_job_raw_reclaim,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "reclaim"):
        command = commands.add_parser(name)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--archive-receipt", type=Path, required=True)
        command.add_argument("--remote-receipt", type=Path, required=True)
        command.add_argument("--expected-drive-parent-id", required=True)
        if name == "reclaim":
            command.add_argument(
                "--confirm-reclaim",
                action="store_true",
                help="required acknowledgement for allowlisted raw unlink",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common: dict[str, Any] = {
        "run_root": args.run_root,
        "archive_receipt_path": args.archive_receipt,
        "remote_receipt_path": args.remote_receipt,
        "expected_drive_parent_id": args.expected_drive_parent_id,
    }
    if args.command == "verify":
        result = verify_historical_job_raw_reclaim(**common)
    else:
        if not args.confirm_reclaim:
            raise SystemExit("reclaim requires --confirm-reclaim")
        result = reclaim_historical_job_raw(**common)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

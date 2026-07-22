#!/usr/bin/env python3
"""Verify or reclaim raw files from remotely verified legacy cell archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_legacy_cell_raw_reclaim import (  # noqa: E402
    DojoLegacyCellRawReclaimError,
    reclaim_legacy_cell_raw,
    verify_legacy_cell_raw_reclaim,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("verify", "verify eligibility without changing the source run"),
        ("reclaim", "unlink only the sealed remotely verified raw targets"),
    ):
        child = commands.add_parser(command, help=help_text)
        child.add_argument("--source-run", type=Path, required=True)
        child.add_argument("--archive-root", type=Path, required=True)
        child.add_argument("--remote-receipts-dir", type=Path, required=True)
        child.add_argument("--expected-drive-parent-id", required=True)
        child.add_argument("--zstd-bin", default="zstd")
        if command == "reclaim":
            child.add_argument(
                "--confirm-reclaim",
                action="store_true",
                help="required acknowledgement that verified raw will be unlinked",
            )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "reclaim" and not args.confirm_reclaim:
        print(
            json.dumps(
                {
                    "status": "REJECTED",
                    "error": "reclaim requires --confirm-reclaim",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    options = {
        "source_run": args.source_run,
        "archive_root": args.archive_root,
        "remote_receipts_dir": args.remote_receipts_dir,
        "expected_drive_parent_id": args.expected_drive_parent_id,
        "zstd_bin": args.zstd_bin,
    }
    try:
        if args.command == "verify":
            result = verify_legacy_cell_raw_reclaim(**options)
        else:
            result = reclaim_legacy_cell_raw(**options)
    except (DojoLegacyCellRawReclaimError, OSError) as exc:
        print(
            json.dumps({"status": "REJECTED", "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

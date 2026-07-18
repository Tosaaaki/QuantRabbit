#!/usr/bin/env python3
"""Create or verify a deterministic DOJO materialization V2 receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant_rabbit.dojo_materialization import (  # noqa: E402
    MaterializationError,
    build_materialization_receipt,
    publish_receipt_exclusive,
    verify_materialized_archive,
    verify_materialization_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Original carrier. Required when creating; optional during later "
            "regular-only archive verification."
        ),
    )
    parser.add_argument("--materialized-root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--output", type=Path, help="Create one new receipt with O_EXCL."
    )
    action.add_argument("--receipt", type=Path, help="Verify an existing receipt.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output is not None:
            if args.source_root is None:
                raise MaterializationError("--source-root is required with --output")
            receipt = build_materialization_receipt(
                source_root=args.source_root,
                materialized_root=args.materialized_root,
            )
            publish_receipt_exclusive(args.output, receipt)
            status = "CREATED"
            path = args.output
        else:
            receipt = (
                verify_materialization_receipt(
                    receipt_path=args.receipt,
                    source_root=args.source_root,
                    materialized_root=args.materialized_root,
                )
                if args.source_root is not None
                else verify_materialized_archive(
                    receipt_path=args.receipt,
                    materialized_root=args.materialized_root,
                )
            )
            status = "VERIFIED"
            path = args.receipt
    except (MaterializationError, OSError) as exc:
        print(f"dojo-materialization: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": status,
                "receipt": str(path),
                "receipt_sha256": receipt["receipt_sha256"],
                "file_count": receipt["file_count"],
                "source_regular_count": receipt["source_regular_count"],
                "source_symlink_count": receipt["source_symlink_count"],
                "live_permission": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

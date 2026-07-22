#!/usr/bin/env python3
"""Create or verify an append-only historical-generation supersede receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_historical_supersede_receipt import (
    DojoHistoricalSupersedeReceiptError,
    create_historical_supersede_receipt,
    verify_historical_supersede_receipt_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--old-root", type=Path, required=True)
        command.add_argument("--new-root", type=Path, required=True)
        if name == "verify":
            command.add_argument("--receipt", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "old_root": args.old_root,
        "new_root": args.new_root,
    }
    if args.command == "create":
        receipt = create_historical_supersede_receipt(**common)
        receipt_path = (
            args.new_root
            / "transition-receipts"
            / (
                "supersede-"
                f"{receipt['transition_identity_sha256']}-"
                f"{receipt['receipt_sha256']}.json"
            )
        )
        return {
            "status": "SUPERSEDE_RECEIPT_CREATED_OR_REOPENED",
            "receipt_path": str(receipt_path.resolve(strict=True)),
            "receipt": receipt,
        }
    receipt = verify_historical_supersede_receipt_file(args.receipt, **common)
    return {
        "status": "SUPERSEDE_RECEIPT_VERIFIED",
        "receipt_path": str(args.receipt.resolve(strict=True)),
        "receipt": receipt,
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _run(args)
    except (DojoHistoricalSupersedeReceiptError, OSError) as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

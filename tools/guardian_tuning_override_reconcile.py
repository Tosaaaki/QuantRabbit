#!/usr/bin/env python3
"""Reconcile evidence override stages after a lifecycle crash window."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from quant_rabbit.guardian_tuning_overrides import (  # noqa: E402
    DEFAULT_OVERRIDE_PATH,
    DEFAULT_WORK_ORDER_PATH,
    reconcile_pending_overrides,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Confirm only staged overrides with matching CONSUMED terminal evidence."
    )
    parser.add_argument("--override-path", type=Path, default=DEFAULT_OVERRIDE_PATH)
    parser.add_argument("--queue-path", type=Path, default=DEFAULT_WORK_ORDER_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = reconcile_pending_overrides(
            path=args.override_path,
            queue_path=args.queue_path,
            now=datetime.now(timezone.utc),
        )
    except Exception as exc:  # CLI boundary: fail closed and report no secrets.
        result = {
            "status": "OVERRIDE_RECONCILIATION_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "live_permission_allowed": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return (
        0
        if result["status"]
        in {"NO_PENDING_OVERRIDE_CONFIRMATIONS", "PENDING_OVERRIDES_RECONCILED"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect, validate, persist, and seal one DOJO worker source day."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_rabbit.broker.oanda import OandaReadOnlyClient  # noqa: E402
from quant_rabbit.dojo_worker_source import (  # noqa: E402
    collect_and_seal_day,
    verify_collected_day,
)


class _RetryingReadOnlyClient:
    def __init__(self, *, attempts: int) -> None:
        self._client = OandaReadOnlyClient()
        self._attempts = attempts

    @property
    def base_url(self) -> str:
        return self._client.base_url

    def get_json(self, path: str, query: dict[str, str]) -> dict[str, Any]:
        error: Exception | None = None
        for attempt in range(1, self._attempts + 1):
            try:
                return self._client.get_json(path, query)
            except Exception as exc:  # noqa: BLE001 - bounded acquisition retry.
                error = exc
                if attempt < self._attempts:
                    time.sleep(float(attempt))
        assert error is not None
        raise error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect", help="perform the exact read-only OANDA acquisition and day seal"
    )
    collect.add_argument("--run-dir", type=Path, required=True)
    collect.add_argument("--ordinal", type=int, required=True)
    collect.add_argument("--attempts", type=int, default=3, choices=range(1, 6))

    verify = subparsers.add_parser(
        "verify", help="re-open and re-hash an already collected day without network"
    )
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--ordinal", type=int, required=True)

    args = parser.parse_args()
    if args.command == "collect":
        result = collect_and_seal_day(
            args.run_dir,
            ordinal=args.ordinal,
            client=_RetryingReadOnlyClient(attempts=args.attempts),
            repo_root=REPO,
            collector_paths=[Path(__file__)],
        )
    else:
        result = verify_collected_day(args.run_dir, ordinal=args.ordinal)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

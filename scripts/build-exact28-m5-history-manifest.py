#!/usr/bin/env python3
"""Build a compact, fail-closed exact-28 M5/BA history manifest.

This command performs local validation only. It never constructs an OANDA
client, fetches candles, reads an account, or creates order authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_rabbit.exact28_m5_history_manifest import (  # noqa: E402
    DEFAULT_PERIOD_FROM_UTC,
    DEFAULT_PERIOD_TO_UTC,
    build_exact28_m5_history_manifest,
    write_exact28_m5_history_manifest,
)


DEFAULT_OUTPUT = REPO_ROOT / "research/data/exact28_m5_history_manifest_2020_2026.json"


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    utc = parsed.astimezone(timezone.utc)
    if utc.utcoffset() != timezone.utc.utcoffset(utc):
        raise argparse.ArgumentTypeError("timestamp must normalize to UTC")
    return utc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="clean staged oanda_history root containing annual M5 runs",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--from-utc",
        type=_parse_utc,
        default=DEFAULT_PERIOD_FROM_UTC,
    )
    parser.add_argument(
        "--to-utc",
        type=_parse_utc,
        default=DEFAULT_PERIOD_TO_UTC,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_exact28_m5_history_manifest(
        args.root,
        period_from_utc=args.from_utc,
        period_to_utc=args.to_utc,
    )
    write_exact28_m5_history_manifest(args.output, manifest)
    result = {
        "status": "VERIFIED",
        "manifest_path": str(args.output.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "pair_count": manifest["expected_pair_count"],
        "shard_count": manifest["expected_shard_count"],
        "pair_shard_count": manifest["selected_pair_shard_count"],
        "raw_candles_embedded": manifest["raw_candles_embedded"],
        "order_authority": manifest["order_authority"],
        "live_permission": manifest["live_permission"],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


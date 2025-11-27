#!/usr/bin/env python3
"""
Quick helper to fetch the latest OANDA candles for ad-hoc inspections.

Example:
    python scripts/fetch_latest_candles.py --granularity M1 --count 120
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.secrets import get_secret  # noqa: E402


def _oanda_base_url() -> str:
    try:
        practice_flag = get_secret("oanda_practice").strip().lower() in {"1", "true", "yes"}
    except KeyError:
        practice_flag = False
    return "https://api-fxpractice.oanda.com" if practice_flag else "https://api-fxtrade.oanda.com"


def _default_output(instrument: str, granularity: str) -> pathlib.Path:
    safe_instrument = instrument.replace("/", "_")
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "logs" / f"candles_{safe_instrument}_{granularity}_{ts}.json"


def fetch_candles(
    instrument: str,
    granularity: str,
    count: int,
    price_type: str,
    include_weekends: bool,
) -> Dict[str, Any]:
    token = get_secret("oanda_token")
    params = {
        "granularity": granularity,
        "count": str(max(1, min(count, 500))),
        "price": price_type,
    }
    if include_weekends:
        params["includeFirst"] = "true"
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{_oanda_base_url()}/v3/instruments/{instrument}/candles?{query}")
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the latest OANDA candles.")
    parser.add_argument("--instrument", default="USD_JPY", help="Instrument symbol (default: USD_JPY)")
    parser.add_argument("--granularity", default="M1", help="OANDA granularity (default: M1)")
    parser.add_argument("--count", type=int, default=60, help="Number of candles to fetch (<=500)")
    parser.add_argument(
        "--price",
        default="M",
        help="Price component (O/H/L/C). Default M (mid prices).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Optional output file. Defaults to logs/candles_<instrument>_<granularity>_<UTC>.json",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print to stdout without writing a file.",
    )
    parser.add_argument(
        "--include-weekends",
        action="store_true",
        help="Set includeFirst=true to include partial weekend candles.",
    )
    args = parser.parse_args()

    data = fetch_candles(
        instrument=args.instrument,
        granularity=args.granularity,
        count=args.count,
        price_type=args.price,
        include_weekends=args.include_weekends,
    )

    json_output = json.dumps(data, indent=2, ensure_ascii=False)
    print(json_output)

    if args.no_save:
        return

    output_path = args.output or _default_output(args.instrument, args.granularity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_output, encoding="utf-8")
    print(f"[candles] saved -> {output_path}")


if __name__ == "__main__":
    main()

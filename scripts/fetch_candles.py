#!/usr/bin/env python3
"""
Fetch historical OANDA candles across a date range and persist as JSON.

Example:
  PYTHONPATH=. python3 scripts/fetch_candles.py \
      --instrument USD_JPY \
      --granularity S5 \
      --start 2025-10-01T00:00:00Z \
      --end   2025-11-01T00:00:00Z \
      --out tmp/candles_USDJPY_202510_S5.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import requests

from utils.secrets import get_secret


ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class CandleRequest:
    instrument: str
    granularity: str
    start: dt.datetime
    end: dt.datetime
    price: str = "M"
    include_first: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OANDA candles into JSON")
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--granularity", required=True)
    parser.add_argument("--start", required=True, help="ISO timestamp e.g. 2025-10-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="ISO timestamp (exclusive)")
    parser.add_argument("--price", default="M")
    parser.add_argument("--chunk-hours", type=int, default=6, help="Chunk window in hours (default 6)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    return parser.parse_args()


def _iso(dt_obj: dt.datetime) -> str:
    return dt_obj.strftime(ISO_FORMAT)


def _load_credentials() -> Dict[str, str]:
    token = get_secret("oanda_token")
    account = get_secret("oanda_account_id")
    host = "https://api-fxtrade.oanda.com"
    try:
        if get_secret("oanda_practice").lower() == "true":
            host = "https://api-fxpractice.oanda.com"
    except Exception:
        pass
    return {"token": token, "account": account, "host": host}


def fetch_chunk(creds: Dict[str, str], req: CandleRequest) -> List[Dict]:
    params = {
        "from": _iso(req.start),
        "to": _iso(req.end),
        "granularity": req.granularity,
        "price": req.price,
        "includeFirst": "true" if req.include_first else "false",
    }
    headers = {"Authorization": f"Bearer {creds['token']}"}
    url = f"{creds['host']}/v3/instruments/{req.instrument}/candles"

    for attempt in range(5):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            candles = payload.get("candles") or []
            if not isinstance(candles, list):
                raise ValueError("Unexpected payload structure")
            return candles
        except Exception as exc:  # noqa: BLE001
            wait = min(5 * (attempt + 1), 30)
            body = ""
            try:
                body = f" | body={resp.text[:300]!r}"
            except Exception:
                pass
            print(
                f"[fetch] chunk {params['from']} -> {params['to']} failed: {exc}{body}; retry in {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch candles for window {params['from']} -> {params['to']}")


def main() -> int:
    args = parse_args()
    try:
        start = dt.datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        end = dt.datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except ValueError as exc:
        print(f"invalid timestamp: {exc}", file=sys.stderr)
        return 1
    if start >= end:
        print("start must be before end", file=sys.stderr)
        return 1

    creds = _load_credentials()
    chunk = dt.timedelta(hours=max(1, args.chunk_hours))
    cursor = start
    all_candles: Dict[str, Dict] = {}
    include_first = True

    while cursor < end:
        window_end = min(cursor + chunk, end)
        req = CandleRequest(
            instrument=args.instrument,
            granularity=args.granularity,
            start=cursor,
            end=window_end,
            price=args.price,
            include_first=include_first,
        )
        candles = fetch_chunk(creds, req)
        for candle in candles:
            ts = candle.get("time")
            if not isinstance(ts, str):
                continue
            all_candles[ts] = candle
        include_first = False  # only include first candle once
        cursor = window_end

    ordered = [all_candles[k] for k in sorted(all_candles)]
    payload = {
        "instrument": args.instrument,
        "granularity": args.granularity,
        "price": args.price,
        "start": _iso(start),
        "end": _iso(end),
        "candles": ordered,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"[fetch] wrote {len(ordered)} candles to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

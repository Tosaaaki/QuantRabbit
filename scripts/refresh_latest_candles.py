#!/usr/bin/env python3
"""
Fetch latest OANDA candles and persist a normalized JSON for BQ upload.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List

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


def _default_output(granularity: str) -> pathlib.Path:
    return ROOT / "logs" / "oanda" / f"candles_{granularity}_latest.json"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_candle(raw: Dict[str, Any], instrument: str, timeframe: str) -> Dict[str, Any]:
    mid = raw.get("mid") or {}
    return {
        "time": raw.get("time") or raw.get("timestamp") or raw.get("ts"),
        "open": _safe_float(mid.get("o", raw.get("open"))),
        "high": _safe_float(mid.get("h", raw.get("high"))),
        "low": _safe_float(mid.get("l", raw.get("low"))),
        "close": _safe_float(mid.get("c", raw.get("close"))),
        "volume": int(raw.get("volume") or 0),
        "instrument": raw.get("instrument") or instrument,
        "timeframe": raw.get("timeframe") or timeframe,
        "mid": mid,
        "complete": raw.get("complete"),
    }


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
    parser = argparse.ArgumentParser(description="Refresh latest OANDA candles into a normalized JSON file.")
    parser.add_argument("--instrument", default="USD_JPY", help="Instrument symbol (default: USD_JPY)")
    parser.add_argument("--granularity", default="M1", help="OANDA granularity (default: M1)")
    parser.add_argument("--count", type=int, default=500, help="Number of candles to fetch (<=500)")
    parser.add_argument(
        "--price",
        default="M",
        help="Price component (O/H/L/C). Default M (mid prices).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Output file. Defaults to logs/oanda/candles_<TF>_latest.json",
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
    candles = data.get("candles", [])
    normalized: List[Dict[str, Any]] = [
        _normalize_candle(c, args.instrument, args.granularity) for c in candles
    ]

    output_path = args.output or _default_output(args.granularity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(),
        "instrument": args.instrument,
        "timeframe": args.granularity,
        "candles": normalized,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[candles] wrote {len(normalized)} -> {output_path}")


if __name__ == "__main__":
    main()

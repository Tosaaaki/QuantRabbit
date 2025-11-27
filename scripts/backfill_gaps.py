"""
Gap detection and candle backfill utility.

- Detect tick gaps in logs/replay/<instrument>/<instrument>_ticks_*.jsonl
- For each gap, fetch OANDA candles (M1/H1/H4) for the missing window
- Backfill factor_cache via indicators.factor_cache.on_candle
- Persist fetched candles to replay logs for auditability

Usage (on VM):
  . .venv/bin/activate
  python scripts/backfill_gaps.py --instrument USD_JPY --lookback-hours 48 --gap-sec 3

Notes:
- This script is best-effort; it skips gaps shorter than --gap-sec.
- OANDA API is used directly with from/to to minimize load; ensure secrets are configured.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indicators.factor_cache import on_candle
from market_data.replay_logger import log_candle
from utils.secrets import get_secret


def _parse_iso(ts: str) -> dt.datetime:
    """Parse ISO 8601 strings with timezone info."""
    iso = ts.replace("Z", "+00:00")
    if "." in iso:
        head, rest = iso.split(".", 1)
        frac = rest
        tz = ""
        if "+" in rest:
            frac, tz_tail = rest.split("+", 1)
            tz = "+" + tz_tail
        elif "-" in rest:
            frac, tz_tail = rest.split("-", 1)
            tz = "-" + tz_tail
        frac = "".join(ch for ch in frac if ch.isdigit())
        frac = (frac[:6]).ljust(6, "0")
        iso = f"{head}.{frac}{tz}"
    return dt.datetime.fromisoformat(iso).astimezone(dt.timezone.utc)


def _iter_tick_times(
    base_dir: Path, instrument: str, *, lookback_hours: int
) -> Iterable[dt.datetime]:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=lookback_hours)
    for path in sorted((base_dir / instrument).glob(f"{instrument}_ticks_*.jsonl")):
        try:
            day_str = path.name.split("_")[-1].split(".")[0]
            day_dt = dt.datetime.strptime(day_str, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
        except Exception:
            day_dt = None
        if day_dt and day_dt.date() < cutoff.date():
            continue
        with path.open() as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    ts_raw = obj.get("ts")
                    if not ts_raw:
                        continue
                    ts = _parse_iso(ts_raw)
                    if ts >= cutoff:
                        yield ts
                except Exception:
                    continue


def detect_gaps(
    times: Sequence[dt.datetime], *, gap_sec: float
) -> List[Tuple[dt.datetime, dt.datetime]]:
    gaps: List[Tuple[dt.datetime, dt.datetime]] = []
    if len(times) < 2:
        return gaps
    prev = times[0]
    for cur in times[1:]:
        delta = (cur - prev).total_seconds()
        if delta > gap_sec:
            gaps.append((prev, cur))
        prev = cur
    return gaps


async def fetch_candles_range(
    instrument: str,
    granularity: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    client: httpx.AsyncClient,
) -> List[dict]:
    token = get_secret("oanda_token")
    try:
        pract = str(get_secret("oanda_practice")).lower() == "true"
    except Exception:
        pract = False
    host = "https://api-fxpractice.oanda.com" if pract else "https://api-fxtrade.oanda.com"
    params = {
        "price": "M",
        "granularity": granularity,
        "from": start.isoformat(),
        "to": end.isoformat(),
    }
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{host}/v3/instruments/{instrument}/candles"
    candles: List[dict] = []
    try:
        r = await client.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        for c in data.get("candles", []):
            ts = _parse_iso(c["time"])
            candles.append(
                {
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "time": ts,
                }
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[BACKFILL] fetch failed {instrument} {granularity} {start}->{end}: {exc}")
    candles.sort(key=lambda x: x["time"])
    return candles


async def backfill_gaps(
    instrument: str,
    gaps: Sequence[Tuple[dt.datetime, dt.datetime]],
    *,
    dry_run: bool,
) -> None:
    if not gaps:
        print("[BACKFILL] No gaps detected.")
        return
    async with httpx.AsyncClient() as client:
        for start, end in gaps:
            print(f"[BACKFILL] gap {start.isoformat()} -> {end.isoformat()}")
            for tf in ("M1", "H1", "H4"):
                candles = await fetch_candles_range(instrument, tf, start, end, client=client)
                if not candles:
                    print(f"[BACKFILL]  {tf}: no candles fetched")
                    continue
                print(f"[BACKFILL]  {tf}: fetched {len(candles)} candles")
                if dry_run:
                    continue
                for c in candles:
                    try:
                        await on_candle(tf, c)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[BACKFILL]  {tf} on_candle failed: {exc}")
                        continue
                    try:
                        log_candle(instrument, tf, c)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[BACKFILL]  {tf} log_candle failed: {exc}")


async def main():
    parser = argparse.ArgumentParser(description="Detect tick gaps and backfill candles from OANDA.")
    parser.add_argument("--instrument", default="USD_JPY", help="Instrument name (default: USD_JPY)")
    parser.add_argument("--lookback-hours", type=int, default=48, help="Lookback window for gap detection")
    parser.add_argument("--gap-sec", type=float, default=3.0, help="Threshold seconds to treat as a gap")
    parser.add_argument("--dry-run", action="store_true", help="Detect and fetch counts only, no writes")
    args = parser.parse_args()

    base_dir = Path("logs/replay")
    ticks = list(_iter_tick_times(base_dir, args.instrument, lookback_hours=args.lookback_hours))
    ticks.sort()
    gaps = detect_gaps(ticks, gap_sec=args.gap_sec)
    print(f"[GAP] inspected {len(ticks)} ticks, detected {len(gaps)} gaps (> {args.gap_sec}s)")
    await backfill_gaps(args.instrument, gaps, dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Fetch OANDA candles and update factor_cache for Claude discretionary trading.
Replaces the bot-era WebSocket candle feed.

Usage:
    python3 scripts/trader_tools/refresh_factor_cache.py [--pair USD_JPY] [--quiet]

Fetches M1(200), M5(200), H1(100), H4(50) candles from OANDA REST API,
feeds them into factor_cache.on_candle(), and persists to disk.
"""
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tomli
from indicators.factor_cache import on_candle, all_factors

OANDA_BASE = "https://api-fxtrade.oanda.com"

def _load_config():
    with open(ROOT / "config" / "env.toml", "rb") as f:
        cfg = tomli.load(f)
    return cfg["oanda_token"], cfg["oanda_account_id"]

def _fetch_candles(token: str, instrument: str, granularity: str, count: int) -> list:
    url = (
        f"{OANDA_BASE}/v3/instruments/{instrument}/candles"
        f"?granularity={granularity}&count={count}&price=M"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        candles = []
        for c in data.get("candles", []):
            if not c.get("complete", False) and granularity != "M1":
                continue
            mid = c["mid"]
            candles.append({
                "time": c["time"],
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
            })
        return candles
    except Exception as e:
        print(f"ERROR fetching {instrument} {granularity}: {e}", file=sys.stderr)
        return []

TF_MAP = {
    "M1": 200,
    "M5": 200,
    "H1": 100,
    "H4": 50,
}

async def refresh(pair: str = "USD_JPY", quiet: bool = False):
    token, _ = _load_config()
    total = 0
    for tf, count in TF_MAP.items():
        candles = _fetch_candles(token, pair, tf, count)
        if not candles:
            if not quiet:
                print(f"  {tf}: no candles fetched")
            continue
        for c in candles:
            await on_candle(tf, c)
        total += len(candles)
        if not quiet:
            last = candles[-1]
            print(f"  {tf}: {len(candles)} candles, last close={last['close']}")

    if not quiet:
        factors = all_factors()
        m1 = factors.get("M1", {})
        print(f"\nCache updated: M1 close={m1.get('close')} rsi={m1.get('rsi', 'N/A'):.1f}")
    return total

def main():
    pair = "USD_JPY"
    quiet = False
    for arg in sys.argv[1:]:
        if arg == "--quiet":
            quiet = True
        elif arg == "--pair":
            pass
        elif not arg.startswith("-"):
            pair = arg
    if not quiet:
        print(f"Refreshing factor_cache for {pair}...")
    count = asyncio.run(refresh(pair, quiet))
    if not quiet:
        print(f"Done. {count} candles processed.")

if __name__ == "__main__":
    main()

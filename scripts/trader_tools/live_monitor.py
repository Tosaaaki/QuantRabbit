#!/usr/bin/env python3
"""
Live Trading Monitor — continuously computes and writes a single JSON file
that Claude trading tasks read for instant decision-making.

Runs every 30s via launchd. No LLM cost. Pure Python.

Output: logs/live_monitor.json
  - Current bid/ask/spread for all pairs
  - S5 micro-momentum (last 2 min of 5-second candles)
  - M1 indicators (RSI, ADX, StochRSI, EMA slopes, MACD)
  - M5 indicators (same set)
  - H1 bias summary (from existing technicals files)
  - Open positions with UPL, age, trail status
  - Account summary

Usage:
    python3 scripts/trader_tools/live_monitor.py          # one-shot
    python3 scripts/trader_tools/live_monitor.py --loop 30 # loop every 30s (for testing)
"""

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tomli
import pandas as pd
from indicators.calc_core import IndicatorEngine

OANDA_BASE = "https://api-fxtrade.oanda.com"
ALL_PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
OUTPUT_PATH = ROOT / "logs" / "live_monitor.json"

# Minimal keys for fast scalp decisions
SCALP_KEYS = [
    "rsi", "adx", "plus_di", "minus_di", "stoch_rsi", "macd_hist",
    "ema_slope_5", "ema_slope_10", "cci", "bb_upper", "bb_lower", "bb_mid",
    "close", "atr_pips", "regime", "vwap_gap", "bbw",
]


def _load_config():
    with open(ROOT / "config" / "env.toml", "rb") as f:
        cfg = tomli.load(f)
    return cfg["oanda_token"], cfg["oanda_account_id"]


def _api_get(token: str, path: str, timeout: int = 8):
    req = urllib.request.Request(
        f"{OANDA_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


# --- Data Fetchers ---

def fetch_pricing(token: str, acc: str) -> dict:
    """Current bid/ask for all pairs."""
    instruments = ",".join(ALL_PAIRS)
    data = _api_get(token, f"/v3/accounts/{acc}/pricing?instruments={instruments}")
    result = {}
    for p in data.get("prices", []):
        pair = p["instrument"]
        bids = p.get("bids", [])
        asks = p.get("asks", [])
        bid = float(bids[0]["price"]) if bids else 0
        ask = float(asks[0]["price"]) if asks else 0
        pip = 0.01 if bid > 50 else 0.0001
        spread = round((ask - bid) / pip, 1)
        result[pair] = {"bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 5), "spread_pips": spread}
    return result


def fetch_s5_candles(token: str, pair: str, count: int = 24) -> list:
    """Last ~2min of 5-second candles for micro-momentum."""
    url = f"/v3/instruments/{pair}/candles?granularity=S5&count={count}&price=M"
    try:
        data = _api_get(token, url, timeout=5)
        candles = []
        for c in data.get("candles", []):
            mid = c["mid"]
            candles.append({
                "time": c["time"],
                "close": float(mid["c"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
            })
        return candles
    except Exception:
        return []


def compute_micro_momentum(s5_candles: list, pip_size: float) -> dict:
    """Compute micro-momentum from S5 candles."""
    if len(s5_candles) < 6:
        return {"direction": "FLAT", "velocity": 0, "range_pips": 0}

    closes = [c["close"] for c in s5_candles]
    highs = [c["high"] for c in s5_candles]
    lows = [c["low"] for c in s5_candles]

    # Direction: compare last 6 vs first 6
    half = len(closes) // 2
    first_avg = sum(closes[:half]) / half
    last_avg = sum(closes[half:]) / (len(closes) - half)
    diff_pips = round((last_avg - first_avg) / pip_size, 1)

    if diff_pips > 0.5:
        direction = "UP"
    elif diff_pips < -0.5:
        direction = "DOWN"
    else:
        direction = "FLAT"

    # Velocity: pips per minute over the window
    window_seconds = len(s5_candles) * 5
    velocity = round(diff_pips / (window_seconds / 60), 2) if window_seconds > 0 else 0

    # Range
    range_pips = round((max(highs) - min(lows)) / pip_size, 1)

    return {"direction": direction, "velocity": velocity, "range_pips": range_pips, "diff_pips": diff_pips}


def fetch_candles_and_compute(token: str, pair: str, granularity: str, count: int) -> dict:
    """Fetch candles and compute indicators."""
    url = f"/v3/instruments/{pair}/candles?granularity={granularity}&count={count}&price=M"
    try:
        data = _api_get(token, url, timeout=8)
        candles = []
        for c in data.get("candles", []):
            # Include incomplete candle for M1
            if not c.get("complete", False) and granularity not in ("M1", "S5"):
                continue
            mid = c["mid"]
            candles.append({
                "time": c["time"],
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
            })
        if len(candles) < 20:
            return {}
        df = pd.DataFrame(candles)
        df.rename(columns={"time": "timestamp"}, inplace=True)
        factors = IndicatorEngine.compute(df)
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in factors.items() if k in SCALP_KEYS and v is not None}
    except Exception as e:
        return {"error": str(e)}


def fetch_positions(token: str, acc: str) -> tuple:
    """Fetch open trades and account summary."""
    trades_data = _api_get(token, f"/v3/accounts/{acc}/openTrades")
    summary_data = _api_get(token, f"/v3/accounts/{acc}/summary")

    positions = []
    for t in trades_data.get("trades", []):
        units = int(t["currentUnits"])
        pair = t["instrument"]
        pip_size = 0.01 if float(t["price"]) > 50 else 0.0001
        entry = float(t["price"])
        upl = float(t.get("unrealizedPL", "0"))

        sl_order = t.get("stopLossOrder", {})
        tp_order = t.get("takeProfitOrder", {})
        trail_order = t.get("trailingStopLossOrder", {})

        sl = float(sl_order["price"]) if sl_order.get("price") else None
        tp = float(tp_order["price"]) if tp_order.get("price") else None
        trail_dist = float(trail_order["distance"]) if trail_order.get("distance") else None

        # Calculate UPL in pips
        # For shorts, profit = entry - current; for longs, profit = current - entry
        # We approximate from UPL and units
        open_time = t["openTime"][:19]
        try:
            opened = datetime(
                int(open_time[:4]), int(open_time[5:7]), int(open_time[8:10]),
                int(open_time[11:13]), int(open_time[14:16]), int(open_time[17:19]),
                tzinfo=timezone.utc,
            )
            age_min = round((datetime.now(timezone.utc) - opened).total_seconds() / 60, 1)
        except Exception:
            age_min = 0

        is_be = sl is not None and abs(sl - entry) < pip_size * 0.5

        positions.append({
            "id": t["id"],
            "pair": pair,
            "units": units,
            "entry": entry,
            "upl": upl,
            "sl": sl,
            "tp": tp,
            "trail": trail_dist,
            "is_be": is_be,
            "has_trail": trail_dist is not None,
            "age_min": age_min,
            "opened": open_time,
        })

    s = summary_data["account"]
    account = {
        "nav": round(float(s["NAV"])),
        "upl": round(float(s["unrealizedPL"])),
        "margin_used": round(float(s["marginUsed"])),
        "margin_avail": round(float(s["marginAvailable"])),
        "open_trades": int(s["openTradeCount"]),
    }

    return positions, account


def load_h1_bias() -> dict:
    """Load H1 bias from existing technicals files (computed by refresh_factor_cache)."""
    bias = {}
    for pair in ALL_PAIRS:
        path = ROOT / "logs" / f"technicals_{pair}.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            h1 = data.get("timeframes", {}).get("H1", {})
            h4 = data.get("timeframes", {}).get("H4", {})
            bias[pair] = {
                "h1_regime": h1.get("regime", "unknown"),
                "h1_adx": h1.get("adx"),
                "h1_rsi": h1.get("rsi"),
                "h1_plus_di": h1.get("plus_di"),
                "h1_minus_di": h1.get("minus_di"),
                "h4_regime": h4.get("regime", "unknown"),
                "h4_adx": h4.get("adx"),
                "h4_ema_slope_5": h4.get("ema_slope_5"),
            }
        except Exception:
            continue
    return bias


# --- Main ---

def build_monitor() -> dict:
    """Build the complete monitor snapshot."""
    token, acc = _load_config()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fetch pricing (1 API call for all pairs)
    pricing = fetch_pricing(token, acc)

    # Fetch positions + account (2 API calls)
    positions, account = fetch_positions(token, acc)

    # H1/H4 bias from existing files (no API call)
    h1_bias = load_h1_bias()

    # Per-pair data: S5 + M1 + M5 indicators
    pairs = {}
    for pair in ALL_PAIRS:
        pip_size = 0.01 if pair.endswith("JPY") else 0.0001
        price_data = pricing.get(pair, {})

        # S5 micro-momentum
        s5 = fetch_s5_candles(token, pair, count=24)
        micro = compute_micro_momentum(s5, pip_size)

        # M1 indicators
        m1 = fetch_candles_and_compute(token, pair, "M1", 100)

        # M5 indicators
        m5 = fetch_candles_and_compute(token, pair, "M5", 100)

        # Combine
        pairs[pair] = {
            "price": price_data,
            "micro": micro,
            "m1": m1,
            "m5": m5,
            "bias": h1_bias.get(pair, {}),
        }

    return {
        "timestamp": now,
        "pairs": pairs,
        "positions": positions,
        "account": account,
    }


def main():
    loop_sec = None
    for arg in sys.argv[1:]:
        if arg == "--loop":
            idx = sys.argv.index("--loop")
            if idx + 1 < len(sys.argv):
                loop_sec = int(sys.argv[idx + 1])

    while True:
        try:
            monitor = build_monitor()
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(monitor, f, indent=2)
            ts = monitor["timestamp"]
            n_pos = len(monitor["positions"])
            nav = monitor["account"]["nav"]
            print(f"[{ts}] Monitor updated. {n_pos} positions, NAV={nav}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)

        if loop_sec is None:
            break
        time.sleep(loop_sec)


if __name__ == "__main__":
    main()

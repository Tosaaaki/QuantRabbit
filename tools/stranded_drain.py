#!/usr/bin/env python3
"""
Stranded Drain — passive exit for accumulated counter-side bot inventory.

Reads logs/bot_brake_state.json. For every pair flagged drain_mode:
  1. Calculate weighted-avg entry price of stranded side
  2. Place a LIMIT close order at avg_entry +/- 0.5 ATR (close to BE)
     This way, when price retraces, OANDA closes the inventory automatically.
  3. Skip pairs that already have a recent drain order alive (avoid duplicates)

Why this exists:
  The bot's "no SL, TP only" model wins fast on the trending side but leaves
  a growing counter-side bag. The previous death cycle was: bag grows -> margin
  fills -> closeout at the worst price -> all profit returned.
  This script gives the bag a dignified exit: BE-ish takeprofit on retracement.

Usage:
    python3 tools/stranded_drain.py
    python3 tools/stranded_drain.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_MAIN_ROOT = ROOT
if not (_MAIN_ROOT / "config" / "env.toml").exists():
    _git_common = Path(subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(ROOT), text=True
    ).strip())
    _MAIN_ROOT = _git_common.resolve().parent

sys.path.insert(0, str(ROOT / "tools"))

from technicals_json import load_technicals_timeframes
from range_bot import (
    BOT_TAG, BOT_MARKET_TAG, fetch_open_trades, fetch_pending_orders,
    fetch_prices, get_tag, load_config, oanda_api, signed_units, format_price,
)

BOT_TAGS = {BOT_TAG, BOT_MARKET_TAG, "trend_bot_market"}
DRAIN_TAG = "drain"
BRAKE_STATE_PATH = _MAIN_ROOT / "logs" / "bot_brake_state.json"
LOG_FILE = _MAIN_ROOT / "logs" / "live_trade_log.txt"

# Drain target: avg_entry +/- 0.5 ATR
DRAIN_ATR_FRACTION = 0.5
# Min ATR pips to trust the drain target (lowered for tight pairs like EUR/USD)
MIN_ATR_PIPS = 1.0
# GTD hours
DRAIN_GTD_HOURS = 6


def append_log(line: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as fh:
        fh.write(f"[{now}] {line}\n")


def load_brake_state() -> dict:
    if not BRAKE_STATE_PATH.exists():
        return {}
    try:
        return json.loads(BRAKE_STATE_PATH.read_text())
    except Exception:
        return {}


def load_pair_atr_pips(pair: str) -> float:
    f = _MAIN_ROOT / f"logs/technicals_{pair}.json"
    try:
        tfs = load_technicals_timeframes(f)
        m5 = tfs.get("M5", {}) or {}
        v = m5.get("atr_pips")
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def weighted_avg_entry(trades: list[dict]) -> tuple[float, int]:
    """Returns (avg_price, total_units) for given trades."""
    total_u = 0
    weighted = 0.0
    for t in trades:
        u = abs(signed_units(t))
        try:
            p = float(t.get("price", 0))
        except (TypeError, ValueError):
            continue
        if u <= 0 or p <= 0:
            continue
        total_u += u
        weighted += p * u
    if total_u == 0:
        return 0.0, 0
    return weighted / total_u, total_u


def existing_drain_order_for(pair: str, side: str, pending_orders: list[dict]) -> dict | None:
    """Find an existing drain LIMIT close order for pair/side.

    A drain order is a TAKE_PROFIT_ORDER on a specific tradeID, OR a LIMIT
    order with tag == DRAIN_TAG that closes the side.
    """
    for o in pending_orders:
        ext = o.get("clientExtensions", {}) or {}
        if ext.get("tag") == DRAIN_TAG and o.get("instrument") == pair:
            # check direction matches "close <side>"
            try:
                u = int(o.get("units", 0))
            except (TypeError, ValueError):
                u = 0
            # A close-LONG is SHORT order (negative units), close-SHORT is LONG (positive)
            if side == "LONG" and u < 0:
                return o
            if side == "SHORT" and u > 0:
                return o
    return None


def place_drain_limit(token: str, acct: str, pair: str, side: str,
                      target_price: float, units_total: int, dry_run: bool) -> tuple[bool, str]:
    """Place a single LIMIT order that, when filled, will reduce the stranded side.

    For LONG inventory drain: SHORT LIMIT @ target_price (above current). Reduces longs on fill.
    For SHORT inventory drain: LONG LIMIT @ target_price (below current). Reduces shorts on fill.
    OANDA in hedge mode would open opposite — so instead we use TP on individual trades.
    """
    # Use per-trade TP setting via PUT /trades/{id}/orders for safety in hedge mode.
    # This function is replaced by set_take_profit_on_trades below.
    return False, "place_drain_limit not used; see set_take_profit_on_trades"


def set_take_profit_on_trades(token: str, acct: str, pair: str,
                               trades: list[dict], target_price: float,
                               dry_run: bool) -> list[str]:
    """Set TAKE_PROFIT order on each stranded trade pointing at the drain target.

    OANDA endpoint: PUT /v3/accounts/{acct}/trades/{tradeID}/orders
    Body: {"takeProfit": {"price": "...", "timeInForce": "GTC"}}
    """
    actions = []
    target_str = format_price(target_price, pair)
    for t in trades:
        trade_id = t.get("id")
        # Skip if an existing TP is already at this level (within 1 pip)
        existing_tp = t.get("takeProfitOrder", {}) or {}
        if existing_tp:
            try:
                cur_tp = float(existing_tp.get("price", 0))
                if abs(cur_tp - target_price) < pip_size(pair):
                    continue
            except (TypeError, ValueError):
                pass
        body = {
            "takeProfit": {
                "price": target_str,
                "timeInForce": "GTC",
                "clientExtensions": {"tag": DRAIN_TAG, "comment": "stranded_drain"},
            }
        }
        if dry_run:
            actions.append(f"[DRY] {pair} trade {trade_id} TP -> {target_str}")
            continue
        try:
            oanda_api(f"/v3/accounts/{acct}/trades/{trade_id}/orders",
                      token, acct, method="PUT", data=body)
            actions.append(f"{pair} trade {trade_id} TP -> {target_str} OK")
            append_log(f"STRANDED_DRAIN {pair} trade {trade_id} TP -> {target_str}")
        except Exception as e:
            actions.append(f"{pair} trade {trade_id} TP FAILED: {e}")
    return actions


def drain_pair(token: str, acct: str, pair: str, side: str,
               bot_trades: list[dict], prices: dict,
               dry_run: bool) -> list[str]:
    """Drain stranded SIDE on PAIR by setting TP at avg_entry +/- 0.5 ATR."""
    # Pick trades on the stranded side
    if side == "LONG":
        side_trades = [t for t in bot_trades
                       if t.get("instrument") == pair and signed_units(t) > 0]
    else:
        side_trades = [t for t in bot_trades
                       if t.get("instrument") == pair and signed_units(t) < 0]

    if not side_trades:
        return [f"{pair} {side}: no bot trades"]

    avg_entry, total_units = weighted_avg_entry(side_trades)
    if avg_entry <= 0 or total_units == 0:
        return [f"{pair} {side}: avg_entry calc failed"]

    atr_pips = load_pair_atr_pips(pair)
    if atr_pips < MIN_ATR_PIPS:
        return [f"{pair} {side}: atr_pips={atr_pips:.1f} below min, skip"]

    pip = pip_size(pair)
    drain_offset = atr_pips * DRAIN_ATR_FRACTION * pip

    if side == "LONG":
        # need price to RISE to close LONG: target above avg_entry
        target = avg_entry + drain_offset
        cur = prices.get(pair, {}).get("bid", 0)
        if cur >= target:
            # Already above target — set TP a bit higher to capture
            target = cur + (atr_pips * 0.2 * pip)
    else:
        # need price to FALL to close SHORT: target below avg_entry
        target = avg_entry - drain_offset
        cur = prices.get(pair, {}).get("ask", 0)
        if cur <= target:
            target = cur - (atr_pips * 0.2 * pip)

    actions = [f"{pair} {side}: avg_entry={avg_entry:.5f} drain_target={target:.5f} "
               f"({len(side_trades)} trades, {total_units}u, atr_pips={atr_pips:.1f})"]
    actions.extend(set_take_profit_on_trades(token, acct, pair, side_trades, target, dry_run))
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Stranded Drain")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    print(f"=== STRANDED DRAIN === {now_utc.strftime('%Y-%m-%d %H:%M:%SZ')}")

    brake = load_brake_state()
    if not brake:
        print("No brake_state.json — skip")
        return 0

    drain_pairs = {p: b for p, b in brake.get("pairs", {}).items()
                   if b.get("drain_mode") and b.get("stranded_side")}
    if not drain_pairs:
        print("No pairs in drain_mode")
        return 0

    token, acct = load_config()
    prices = fetch_prices(token, acct)
    open_trades = fetch_open_trades(token, acct)
    bot_trades = [t for t in open_trades if get_tag(t) in BOT_TAGS]

    print(f"Drain candidates: {len(drain_pairs)}")
    for pair, b in drain_pairs.items():
        side = b["stranded_side"]
        print(f"--- {pair} {side} (ratio={b.get('imbalance_ratio')}, "
              f"upl L/S={b.get('long_upl')}/{b.get('short_upl')}) ---")
        actions = drain_pair(token, acct, pair, side, bot_trades, prices, args.dry_run)
        for a in actions:
            print(f"  {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

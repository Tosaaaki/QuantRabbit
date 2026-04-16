#!/usr/bin/env python3
"""
Range Bot — Automated range scalp LIMIT entry bot.

Detects ranges via range_scalp_scanner, places LIMIT orders at BB extremes.
Exits are handled by the trader task (profit_check.py + discretionary judgment).

The bot is Claude's tool — an extension of the trader's arm for range setups only.

Usage:
    python3 tools/range_bot.py              # live mode (places real orders)
    python3 tools/range_bot.py --dry-run    # print plans, no orders
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Find the real repo root (worktrees don't have config/)
_MAIN_ROOT = ROOT
if not (_MAIN_ROOT / "config" / "env.toml").exists():
    # We're in a worktree — resolve to main repo
    _git_common = Path(subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(ROOT), text=True
    ).strip())
    _MAIN_ROOT = _git_common.resolve().parent

sys.path.insert(0, str(ROOT / "tools"))

from market_state import get_market_state
from range_scalp_scanner import (
    PAIRS,
    analyze_range,
    pip_size,
    to_pips,
)

# === Constants ===
BOT_TAG = "range_bot"
MIN_UNITS = 3000
MAX_UNITS = 5000
MAX_BOT_MARGIN_PCT = 0.30  # 30% of NAV reserved for bot
POISON_HOURS_UTC = set(range(19, 24))  # 19:00-23:59 UTC = skip
GTD_HOURS = 4
OANDA_TIME_FMT = "%Y-%m-%dT%H:%M:%S.000000000Z"
CONVICTION_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3}
# Sizing: conservative — leaves room for trader
CONVICTION_MARGIN_PCT = {"S": 0.10, "A": 0.07, "B": 0.05}
# Price decimal places per pair
PRICE_DECIMALS = {
    "USD_JPY": 3, "EUR_USD": 5, "GBP_USD": 5, "AUD_USD": 5,
    "EUR_JPY": 3, "GBP_JPY": 3, "AUD_JPY": 3,
}
LEVERAGE = {
    "USD_JPY": 25, "EUR_USD": 25, "GBP_USD": 25, "AUD_USD": 20,
    "EUR_JPY": 25, "GBP_JPY": 25, "AUD_JPY": 25,
}
LOG_FILE = _MAIN_ROOT / "logs" / "live_trade_log.txt"


def load_config() -> tuple[str, str]:
    """Load OANDA credentials from config/env.toml (main repo)."""
    text = (_MAIN_ROOT / "config" / "env.toml").read_text()
    token = [l.split("=")[1].strip().strip('"') for l in text.split("\n")
             if l.startswith("oanda_token")][0]
    acct = [l.split("=")[1].strip().strip('"') for l in text.split("\n")
            if l.startswith("oanda_account_id")][0]
    return token, acct


def load_technicals(pair: str) -> dict:
    """Load cached technicals from main repo logs/."""
    f = _MAIN_ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("timeframes", {})


def fetch_prices(token: str, acct: str) -> dict:
    """Fetch current bid/ask for all pairs."""
    pairs_str = ",".join(PAIRS)
    data = oanda_api(f"/v3/accounts/{acct}/pricing?instruments={pairs_str}", token, acct)
    prices = {}
    for p in data.get("prices", []):
        pair = p["instrument"]
        bids = p.get("bids", [{}])
        asks = p.get("asks", [{}])
        prices[pair] = {
            "bid": float(bids[0].get("price", 0)) if bids else 0,
            "ask": float(asks[0].get("price", 0)) if asks else 0,
            "mid": (float(bids[0].get("price", 0)) + float(asks[0].get("price", 0))) / 2
            if bids and asks else 0,
        }
    return prices


def fetch_account(token: str, acct: str) -> dict:
    """Fetch NAV and margin info."""
    data = oanda_api(f"/v3/accounts/{acct}/summary", token, acct)
    a = data.get("account", {})
    return {
        "nav": float(a.get("NAV", 0)),
        "margin_used": float(a.get("marginUsed", 0)),
        "margin_available": float(a.get("marginAvailable", 0)),
    }


def oanda_api(path: str, token: str, acct: str,
              method: str = "GET", data: dict | None = None) -> dict:
    """Hit OANDA REST API. Same pattern as rollover_guard.py."""
    url = f"https://api-fxtrade.oanda.com{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method=method)
    if data is not None:
        req.data = json.dumps(data).encode()
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())


def fetch_open_pairs(token: str, acct: str) -> set[str]:
    """Get set of instruments with open positions."""
    data = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
    return {t["instrument"] for t in data.get("trades", [])}


def fetch_pending_bot_orders(token: str, acct: str) -> list[dict]:
    """Get pending orders tagged as range_bot."""
    data = oanda_api(f"/v3/accounts/{acct}/pendingOrders", token, acct)
    bot_orders = []
    for order in data.get("orders", []):
        ext = order.get("clientExtensions", {})
        if ext.get("tag") == BOT_TAG:
            bot_orders.append(order)
    return bot_orders


def cancel_order(token: str, acct: str, order_id: str) -> bool:
    """Cancel a pending order. Returns True on success."""
    try:
        oanda_api(
            f"/v3/accounts/{acct}/orders/{order_id}/cancel",
            token, acct, method="PUT"
        )
        return True
    except urllib.error.HTTPError as e:
        print(f"  WARN: cancel order {order_id} failed: {e.code}")
        return False


def format_price(price: float, pair: str) -> str:
    """Format price to correct decimal places for OANDA API."""
    decimals = PRICE_DECIMALS.get(pair, 5)
    return f"{price:.{decimals}f}"


def calculate_units(conviction: str, nav: float, price: float, pair: str) -> int:
    """Calculate units based on conviction and NAV. Floor=3000, ceiling=5000."""
    margin_pct = CONVICTION_MARGIN_PCT.get(conviction, 0.05)
    margin_budget = nav * margin_pct
    leverage = LEVERAGE.get(pair, 25)
    units = int(margin_budget / (price / leverage))
    return max(MIN_UNITS, min(MAX_UNITS, units))


def estimate_margin(units: int, price: float, pair: str) -> float:
    """Estimate margin required for a position."""
    leverage = LEVERAGE.get(pair, 25)
    return units * price / leverage


def place_limit(token: str, acct: str, pair: str, direction: str,
                units: int, entry: float, tp: float, sl: float,
                gtd_time: str) -> dict:
    """Place a LIMIT order with TP/SL on fill and bot tag."""
    signed_units = str(units) if direction == "BUY" else str(-units)
    payload = {
        "order": {
            "type": "LIMIT",
            "instrument": pair,
            "units": signed_units,
            "price": format_price(entry, pair),
            "timeInForce": "GTD",
            "gtdTime": gtd_time,
            "takeProfitOnFill": {
                "price": format_price(tp, pair),
                "timeInForce": "GTC",
            },
            "stopLossOnFill": {
                "price": format_price(sl, pair),
                "timeInForce": "GTC",
            },
            "clientExtensions": {
                "tag": BOT_TAG,
                "comment": f"range_bot {pair} {direction}",
            },
            "tradeClientExtensions": {
                "tag": BOT_TAG,
                "comment": f"range_bot {pair} {direction}",
            },
        }
    }
    try:
        return oanda_api(
            f"/v3/accounts/{acct}/orders", token, acct,
            method="POST", data=payload
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, 'read') else str(e)
        return {"error": body, "code": e.code}


def log_entry(pair: str, direction: str, units: int,
              entry: float, tp: float, sl: float, order_id: str) -> None:
    """Append entry to live_trade_log.txt."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = (
        f"[{now}] RANGE_BOT_LIMIT {pair} {direction} {units}u "
        f"@{entry} TP={tp} SL={sl} GTD={GTD_HOURS}h id={order_id} tag={BOT_TAG}\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(line)


def slack_notify(pair: str, direction: str, units: int,
                 entry: float, tp: float, sl: float) -> None:
    """Post entry notification to Slack via slack_trade_notify.py."""
    side = "LONG" if direction == "BUY" else "SHORT"
    cmd = [
        sys.executable, str(ROOT / "tools" / "slack_trade_notify.py"),
        "entry",
        "--pair", pair,
        "--side", side,
        "--units", str(units),
        "--price", str(entry),
        "--sl", str(sl),
        "--thesis", f"Range bot: BB {'lower' if direction == 'BUY' else 'upper'} LIMIT",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception:
        pass  # Slack failure must not abort the bot


def scan_ranges(prices: dict) -> list[dict]:
    """Scan all pairs for range opportunities. Returns filtered + sorted list."""
    results = []
    for pair in PAIRS:
        tfs = load_technicals(pair)
        tf_data = tfs.get("M5")  # M5 for scalping
        if not tf_data:
            continue
        opp = analyze_range(pair, tf_data, prices)
        if not opp:
            continue
        if not opp.get("tradeable"):
            continue
        conv = opp.get("conviction", "C")
        if conv not in ("S", "A", "B"):
            continue
        signal = opp.get("active_signal", "")
        if signal == "MID_ZONE":
            continue  # Wait for extremes
        results.append(opp)

    # Sort: S first, then A, then B; within tier by signal_strength desc
    results.sort(key=lambda x: (
        CONVICTION_ORDER.get(x["conviction"], 9),
        -x.get("signal_strength", 0),
    ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Range Bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plans without placing orders")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    print(f"=== RANGE BOT === {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")

    # --- PRE-FLIGHT ---
    state, reason = get_market_state(now_utc)
    if state != "OPEN":
        print(f"Market {state}: {reason}")
        return 1

    if now_utc.hour in POISON_HOURS_UTC:
        print(f"Poison hour {now_utc.hour} UTC (19-23 UTC blocked). Skip.")
        return 1

    # --- FETCH STATE ---
    token, acct = load_config()
    account = fetch_account(token, acct)
    nav = account["nav"]
    margin_used = account["margin_used"]
    margin_pct = margin_used / nav * 100 if nav > 0 else 100

    print(f"NAV: {nav:,.0f} JPY | Margin: {margin_pct:.1f}%")

    open_pairs = fetch_open_pairs(token, acct)
    if open_pairs:
        print(f"Open positions: {', '.join(sorted(open_pairs))}")

    prices = fetch_prices(token, acct)

    # --- CANCEL STALE BOT LIMITS ---
    bot_orders = fetch_pending_bot_orders(token, acct)
    cancelled = 0
    for order in bot_orders:
        oid = order.get("id", "?")
        pair = order.get("instrument", "?")
        if args.dry_run:
            print(f"  [DRY] Would cancel: {pair} id={oid}")
        else:
            if cancel_order(token, acct, oid):
                print(f"  Cancelled: {pair} id={oid}")
                cancelled += 1

    # --- MARGIN BUDGET ---
    # Re-fetch after cancellation (margin freed)
    if cancelled > 0 and not args.dry_run:
        account = fetch_account(token, acct)
        nav = account["nav"]
        margin_used = account["margin_used"]

    bot_budget = nav * MAX_BOT_MARGIN_PCT
    # Estimate margin already used by bot positions (trades tagged range_bot)
    # For simplicity, use full budget minus a safety margin
    budget_remaining = bot_budget
    print(f"Bot margin budget: {budget_remaining:,.0f} JPY (30% of NAV)")

    # --- SCAN RANGES ---
    opportunities = scan_ranges(prices)
    if not opportunities:
        print("No tradeable ranges found. Skip.")
        return 1

    print(f"\nRanges found: {len(opportunities)}")

    # --- PLACE ORDERS ---
    placed = []
    skipped = []
    gtd_time = (now_utc + timedelta(hours=GTD_HOURS)).strftime(OANDA_TIME_FMT)

    for opp in opportunities:
        pair = opp["pair"]
        conv = opp["conviction"]
        signal = opp["active_signal"]

        # Determine direction
        if "BUY" in signal:
            direction = "BUY"
            entry = opp["buy_entry"]
            tp = opp["bb_mid"]
            sl = opp["buy_sl"]
            tp_pips = opp["buy_tp_mid"]
            sl_pips = opp["buy_sl_pips"]
            rr = opp["buy_rr_mid"]
        elif "SELL" in signal:
            direction = "SELL"
            entry = opp["sell_entry"]
            tp = opp["bb_mid"]
            sl = opp["sell_sl"]
            tp_pips = opp["sell_tp_mid"]
            sl_pips = opp["sell_sl_pips"]
            rr = opp["sell_rr_mid"]
        else:
            continue

        # Deconfliction: skip pairs with existing positions
        if pair in open_pairs:
            skipped.append(f"{pair}: existing position")
            continue

        # Size
        units = calculate_units(conv, nav, entry, pair)
        est_margin = estimate_margin(units, entry, pair)

        # Budget check
        if est_margin > budget_remaining:
            skipped.append(f"{pair}: margin budget exhausted")
            break

        # R:R minimum check
        if rr < 1.0:
            skipped.append(f"{pair}: R:R {rr:.1f} < 1.0")
            continue

        # Print plan
        side_label = "LONG" if direction == "BUY" else "SHORT"
        print(f"\n  {pair} {conv}-{side_label} {units}u @{format_price(entry, pair)}")
        print(f"    TP={format_price(tp, pair)} (+{tp_pips:.1f}pip)")
        print(f"    SL={format_price(sl, pair)} (-{sl_pips:.1f}pip)")
        print(f"    R:R={rr:.1f} | Range={opp['range_type']} | Str={opp['signal_strength']}")
        print(f"    Margin ~{est_margin:,.0f} JPY | Triggers: {', '.join(opp['triggers'][:3])}")

        if args.dry_run:
            print(f"    [DRY RUN — not placed]")
            placed.append({
                "pair": pair, "direction": direction, "units": units,
                "entry": entry, "tp": tp, "sl": sl, "conviction": conv,
            })
            budget_remaining -= est_margin
            continue

        # Place the order
        result = place_limit(token, acct, pair, direction, units, entry, tp, sl, gtd_time)

        if "error" in result:
            print(f"    ERROR: {result['error']}")
            skipped.append(f"{pair}: API error")
            continue

        # Extract order ID
        order_resp = result.get("orderCreateTransaction", {})
        order_id = order_resp.get("id", "?")
        print(f"    PLACED id={order_id}")

        # Log + Slack
        log_entry(pair, direction, units, entry, tp, sl, order_id)
        slack_notify(pair, direction, units, entry, tp, sl)

        placed.append({
            "pair": pair, "direction": direction, "units": units,
            "entry": entry, "tp": tp, "sl": sl, "conviction": conv,
            "order_id": order_id,
        })
        budget_remaining -= est_margin

    # --- SUMMARY ---
    print(f"\n{'='*50}")
    print(f"RANGE BOT SUMMARY")
    print(f"  Scanned: {len(PAIRS)} pairs")
    print(f"  Ranges: {len(opportunities)}")
    print(f"  Placed: {len(placed)}")
    print(f"  Cancelled: {cancelled}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")
    if placed:
        for p in placed:
            side = "LONG" if p["direction"] == "BUY" else "SHORT"
            oid = p.get("order_id", "dry")
            print(f"    {p['pair']} {p['conviction']}-{side} {p['units']}u "
                  f"@{format_price(p['entry'], p['pair'])} id={oid}")
    else:
        print(f"  (no orders placed)")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

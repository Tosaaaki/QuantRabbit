#!/usr/bin/env python3
"""Snapshot current bot-tagged inventory for the LLM inventory director."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from bot_policy import get_pair_policy, load_policy
from bot_trade_manager import (
    BOT_TAGS,
    deadlock_profile,
    evaluate_trade,
    fetch_account_summary,
    fetch_open_trades,
    fetch_pending_orders,
    get_tag,
    hold_minutes,
    load_config,
    parse_oanda_time,
    pending_margin,
    trade_margin,
)
from range_bot import fetch_prices, format_price
from range_bot import is_entry_pending_order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    token, acct = load_config()
    policy, policy_notes = load_policy()
    account = fetch_account_summary(token, acct)
    prices = fetch_prices(token, acct)
    open_trades = fetch_open_trades(token, acct)
    pending_orders = fetch_pending_orders(token, acct)

    bot_trades = [trade for trade in open_trades if get_tag(trade) in BOT_TAGS]
    bot_pending = [
        order for order in pending_orders
        if get_tag(order) in BOT_TAGS and is_entry_pending_order(order)
    ]
    projected_margin = account["margin_used"] + sum(
        pending_margin(order, prices) for order in pending_orders if is_entry_pending_order(order)
    )
    projected_pct = (projected_margin / account["nav"]) if account["nav"] > 0 else 1.0
    deadlock = deadlock_profile(bot_trades, prices, account["nav"])

    payload = {
        "generated_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": {
            "global_status": policy["global_status"],
            "projected_margin_cap": policy["projected_margin_cap"],
            "panic_margin_cap": policy["panic_margin_cap"],
            "release_margin_cap": policy["release_margin_cap"],
            "notes": policy["notes"],
            "warnings": policy_notes,
        },
        "account": account | {"projected_margin_pct": projected_pct},
        "deadlock": deadlock,
        "pending": [],
        "trades": [],
    }

    for order in bot_pending:
        pair = order.get("instrument", "?")
        policy_row = get_pair_policy(policy, pair)
        created = parse_oanda_time(order.get("createTime"))
        payload["pending"].append({
            "id": order.get("id"),
            "pair": pair,
            "units": int(order.get("units", 0)),
            "price": float(order.get("price", 0)),
            "tag": get_tag(order),
            "age_min": max(0.0, (now_utc - created).total_seconds() / 60) if created else None,
            "margin": pending_margin(order, prices),
            "policy": policy_row,
        })

    for trade in bot_trades:
        pair = trade.get("instrument", "?")
        policy_row = get_pair_policy(policy, pair)
        review = evaluate_trade(trade, prices, now_utc, policy_row)
        payload["trades"].append({
            "id": trade.get("id"),
            "pair": pair,
            "tag": review["tag"],
            "side": review["side"],
            "units": abs(int(trade.get("currentUnits", 0))),
            "entry": review["entry"],
            "current_price": review["current_price"],
            "upl": review["upl"],
            "hold_min": hold_minutes(trade, now_utc),
            "progress": review["progress"],
            "tempo": review["tempo"],
            "scalp_state": review["scalp_state"],
            "stale_min": review["scalp_profile"]["stale_min"],
            "full_close_min": review["scalp_profile"]["full_close_min"],
            "trapped": review["trapped"],
            "trap_reason": review["trap_reason"],
            "margin": trade_margin(trade, prices),
            "policy": policy_row,
        })

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"=== BOT INVENTORY SNAPSHOT === {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print(
        "Tags: "
        "range_bot=passive LIMIT worker inventory | "
        "range_bot_market=fast range MARKET inventory | "
        "trend_bot_market=fast trend MARKET inventory"
    )
    print(
        f"Policy: {policy['global_status']} | projected_cap={policy['projected_margin_cap']*100:.1f}% "
        f"| panic={policy['panic_margin_cap']*100:.1f}% | release={policy['release_margin_cap']*100:.1f}%"
    )
    if policy_notes:
        print(f"Policy warnings: {'; '.join(policy_notes)}")
    print(
        f"Account: NAV={account['nav']:,.0f} JPY | margin={account['margin_pct']*100:.1f}% "
        f"| projected={projected_pct*100:.1f}% | closeout={account['margin_closeout_pct']:.3f}"
    )
    print(
        f"Deadlock: gross={deadlock['gross_margin']:,.0f} JPY ({deadlock['gross_pct']*100:.1f}%) "
        f"| gap={deadlock['balance_gap']:.2f} | deadlocked={'YES' if deadlock['deadlocked'] else 'NO'}"
    )
    print(f"\nPending bot orders: {len(payload['pending'])}")
    for item in payload["pending"]:
        side = "LONG" if item["units"] > 0 else "SHORT"
        age = "?" if item["age_min"] is None else f"{item['age_min']:.0f}m"
        print(
            f"  {item['pair']} {side} id={item['id']} @{format_price(item['price'], item['pair'])} "
            f"age={age} policy={item['policy']['mode']}/{item['policy']['pending']} "
            f"ownership={item['policy']['ownership']} tempo={item['policy']['tempo']} cap={item['policy']['max_pending']}"
        )
    print(f"\nOpen bot trades: {len(payload['trades'])}")
    for item in payload["trades"]:
        print(
            f"  {item['pair']} {item['side']} id={item['id']} "
            f"tag={item['tag']} tempo={item['tempo']} "
            f"entry={format_price(item['entry'], item['pair'])} "
            f"now={format_price(item['current_price'], item['pair'])} "
            f"UPL={item['upl']:+.0f} hold={item['hold_min']:.0f}m "
            f"progress={item['progress']:.0%} state={item['scalp_state']} "
            f"timeout={item['stale_min']}/{item['full_close_min']}m "
            f"trapped={'YES' if item['trapped'] else 'NO'} reason={item['trap_reason'] or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

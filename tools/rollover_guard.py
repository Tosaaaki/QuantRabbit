#!/usr/bin/env python3
"""
Rollover Guard — Remove SL/Trailing before OANDA daily rollover, restore after.

OANDA daily maintenance at 5 PM ET causes spread spikes that hunt SLs.
This script removes SL/Trailing orders before the spike and restores them after.

Usage:
    python3 tools/rollover_guard.py remove   # Remove all SL/Trailing, save state
    python3 tools/rollover_guard.py restore  # Restore saved SL/Trailing
    python3 tools/rollover_guard.py status   # Show current guard state
"""
from __future__ import annotations
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "logs" / "rollover_guard_state.json"


def load_config():
    cfg = {}
    for line in open(ROOT / "config" / "env.toml"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def oanda_api(path, token, acct, method="GET", data=None):
    url = f"https://api-fxtrade.oanda.com{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method=method)
    if data is not None:
        req.data = json.dumps(data).encode()
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def remove_sls(token: str, acct: str):
    """Remove all SL/Trailing from open trades. Save state for restoration."""
    trades_resp = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
    trades = trades_resp.get("trades", [])

    if not trades:
        print("No open trades. Nothing to remove.")
        return

    saved = []
    removed_count = 0

    for trade in trades:
        trade_id = trade["id"]
        pair = trade["instrument"]
        units = trade.get("currentUnits", trade.get("initialUnits", "0"))
        sl_order = trade.get("stopLossOrder")
        trailing_order = trade.get("trailingStopLossOrder")

        if not sl_order and not trailing_order:
            continue

        entry = {
            "trade_id": trade_id,
            "pair": pair,
            "units": units,
        }

        # Build the cancellation payload
        cancel_payload = {}

        if sl_order:
            entry["sl_price"] = sl_order["price"]
            entry["sl_time_in_force"] = sl_order.get("timeInForce", "GTC")
            cancel_payload["stopLoss"] = {"timeInForce": "GTC", "price": "0"}

        if trailing_order:
            entry["trailing_distance"] = trailing_order["distance"]
            entry["trailing_time_in_force"] = trailing_order.get("timeInForce", "GTC")
            cancel_payload["trailingStopLoss"] = {"timeInForce": "GTC", "distance": "0"}

        # Remove SL/Trailing via OANDA API
        # To cancel: set price to a value that effectively disables it
        # Actually, OANDA allows removing by sending an empty object or using cancel endpoint
        # The correct way: PUT /trades/{id}/orders with stopLoss/trailingStopLoss not included
        # OR: use the order cancel endpoint for the specific SL order ID

        orders_to_cancel = []
        if sl_order:
            orders_to_cancel.append(("SL", sl_order["id"], sl_order["price"]))
        if trailing_order:
            orders_to_cancel.append(("Trail", trailing_order["id"], trailing_order["distance"]))

        success = True
        for order_type, order_id, value in orders_to_cancel:
            try:
                oanda_api(
                    f"/v3/accounts/{acct}/orders/{order_id}/cancel",
                    token, acct, method="PUT"
                )
                print(f"  Removed {order_type} from {pair} trade#{trade_id} (was: {value})")
                removed_count += 1
            except Exception as e:
                print(f"  ERROR removing {order_type} from {pair} trade#{trade_id}: {e}")
                success = False

        if success:
            saved.append(entry)

    if saved:
        STATE_FILE.write_text(json.dumps({
            "removed_at": datetime.now(timezone.utc).isoformat(),
            "trades": saved,
        }, indent=2))
        print(f"\nRemoved {removed_count} SL/Trailing orders from {len(saved)} trades.")
        print(f"State saved to {STATE_FILE}")
        print(f"Run 'python3 tools/rollover_guard.py restore' after rollover.")
    else:
        print("No SL/Trailing orders found on any trade.")


def restore_sls(token: str, acct: str):
    """Restore SL/Trailing from saved state."""
    if not STATE_FILE.exists():
        print("No saved rollover guard state. Nothing to restore.")
        return

    state = json.loads(STATE_FILE.read_text())
    saved_trades = state.get("trades", [])
    removed_at = state.get("removed_at", "unknown")

    print(f"Restoring SLs removed at {removed_at}")
    print()

    # Get current open trades to verify they still exist
    trades_resp = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
    open_trade_ids = {t["id"] for t in trades_resp.get("trades", [])}

    restored = 0
    skipped = 0

    for entry in saved_trades:
        trade_id = entry["trade_id"]
        pair = entry["pair"]

        if trade_id not in open_trade_ids:
            print(f"  SKIP {pair} trade#{trade_id} — no longer open (closed/TP/SL hit)")
            skipped += 1
            continue

        payload = {}

        if "sl_price" in entry:
            payload["stopLoss"] = {
                "price": entry["sl_price"],
                "timeInForce": entry.get("sl_time_in_force", "GTC"),
            }

        if "trailing_distance" in entry:
            payload["trailingStopLoss"] = {
                "distance": entry["trailing_distance"],
                "timeInForce": entry.get("trailing_time_in_force", "GTC"),
            }

        if not payload:
            continue

        try:
            oanda_api(
                f"/v3/accounts/{acct}/trades/{trade_id}/orders",
                token, acct, method="PUT", data=payload
            )
            parts = []
            if "sl_price" in entry:
                parts.append(f"SL={entry['sl_price']}")
            if "trailing_distance" in entry:
                parts.append(f"Trail={entry['trailing_distance']}")
            print(f"  Restored {pair} trade#{trade_id}: {', '.join(parts)}")
            restored += 1
        except Exception as e:
            print(f"  ERROR restoring {pair} trade#{trade_id}: {e}")

    # Clean up state file
    STATE_FILE.unlink()
    print(f"\nRestored {restored} trades. Skipped {skipped} (no longer open).")
    print(f"State file removed.")


def show_status():
    """Show current rollover guard state."""
    if not STATE_FILE.exists():
        print("Rollover guard: No saved state. All SLs are live.")
        return

    state = json.loads(STATE_FILE.read_text())
    removed_at = state.get("removed_at", "unknown")
    trades = state.get("trades", [])

    print(f"Rollover guard: {len(trades)} SL/Trailing orders REMOVED (saved at {removed_at})")
    print()
    for entry in trades:
        parts = []
        if "sl_price" in entry:
            parts.append(f"SL={entry['sl_price']}")
        if "trailing_distance" in entry:
            parts.append(f"Trail={entry['trailing_distance']}")
        print(f"  {entry['pair']} trade#{entry['trade_id']}: {', '.join(parts)}")
    print()
    print("Run 'python3 tools/rollover_guard.py restore' to re-apply.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("remove", "restore", "status"):
        print("Usage: python3 tools/rollover_guard.py [remove|restore|status]")
        sys.exit(1)

    action = sys.argv[1]

    if action == "status":
        show_status()
        return

    cfg = load_config()
    token = cfg["oanda_token"]
    acct = cfg["oanda_account_id"]

    if action == "remove":
        if STATE_FILE.exists():
            print("WARNING: Previous rollover guard state exists. SLs were already removed.")
            print("Run 'restore' first, or delete logs/rollover_guard_state.json manually.")
            show_status()
            return
        remove_sls(token, acct)
    elif action == "restore":
        restore_sls(token, acct)


if __name__ == "__main__":
    main()

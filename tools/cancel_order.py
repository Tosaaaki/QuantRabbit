#!/usr/bin/env python3
"""Cancel a pending order and optionally auto-log the action."""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    cfg = {}
    for line in open(ROOT / "config" / "env.toml"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip().strip('"')
    return cfg


def oanda_request(path: str, method: str = "GET") -> dict:
    cfg = load_config()
    url = f"https://api-fxtrade.oanda.com{path}"
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Authorization": f"Bearer {cfg['oanda_token']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def append_log(order: dict, order_id: str, reason: str) -> None:
    pair = order.get("instrument", "?")
    units = abs(int(order.get("units", 0)))
    side = "LONG" if int(order.get("units", 0)) > 0 else "SHORT"
    price = order.get("price", "?")
    tag = (order.get("clientExtensions") or {}).get("tag", "?")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = (
        f"[{now}] CANCEL_ORDER {pair} {side} {units}u @{price} id={order_id} "
        f"tag={tag} reason={reason}\n"
    )
    with open(ROOT / "logs" / "live_trade_log.txt", "a") as fh:
        fh.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order_id")
    parser.add_argument("--reason", default="manual_cancel")
    parser.add_argument("--auto-log", action="store_true")
    args = parser.parse_args()

    order = oanda_request(f"/v3/accounts/{load_config()['oanda_account_id']}/orders/{args.order_id}")["order"]
    result = oanda_request(
        f"/v3/accounts/{load_config()['oanda_account_id']}/orders/{args.order_id}/cancel",
        method="PUT",
    )
    if args.auto_log:
        append_log(order, args.order_id, args.reason)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Trade closer — PUT /trades/{id}/close でポジションを安全に決済する。

Usage:
    python3 tools/close_trade.py {tradeID} [units]

Examples:
    python3 tools/close_trade.py 465743          # 全決済
    python3 tools/close_trade.py 465743 1000     # 部分決済 (1000u)
"""

import sys
import json
import urllib.request

def load_config():
    lines = open("config/env.toml").read().split("\n")
    token = [l.split("=")[1].strip().strip('"') for l in lines if l.startswith("oanda_token")][0]
    acct = [l.split("=")[1].strip().strip('"') for l in lines if l.startswith("oanda_account_id")][0]
    return token, acct

def close_trade(trade_id: str, units: str | None = None):
    token, acct = load_config()
    url = f"https://api-fxtrade.oanda.com/v3/accounts/{acct}/trades/{trade_id}/close"

    body = {}
    if units:
        body["units"] = units

    data = json.dumps(body).encode() if body else b"{}"
    req = urllib.request.Request(url, data=data, method="PUT", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })

    try:
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())

        if "orderFillTransaction" in result:
            fill = result["orderFillTransaction"]
            inst = fill.get("instrument", "?")
            pl = fill.get("pl", "0")
            price = fill.get("price", "?")
            closed_units = fill.get("units", "?")
            print(f"CLOSED: {inst} {closed_units}u @{price} P/L={pl}")
            return result
        else:
            print(json.dumps(result, indent=2))
            return result

    except urllib.error.HTTPError as e:
        err = json.loads(e.read())
        print(f"ERROR {e.code}: {err.get('errorMessage', err)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    trade_id = sys.argv[1]
    units = sys.argv[2] if len(sys.argv) > 2 else None
    close_trade(trade_id, units)

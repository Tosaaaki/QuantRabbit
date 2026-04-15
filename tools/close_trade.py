#!/usr/bin/env python3
"""Trade closer — PUT /trades/{id}/close でポジションを安全に決済する。

Usage:
    python3 tools/close_trade.py {tradeID} [units] [--reason REASON] [--auto-log] [--auto-slack]

Examples:
    python3 tools/close_trade.py 465743                              # 全決済 (manual log/slack)
    python3 tools/close_trade.py 465743 1000                         # 部分決済
    python3 tools/close_trade.py 465743 --reason zombie_hold --auto-log --auto-slack  # 全自動
    python3 tools/close_trade.py 465743 1000 --reason half_tp --auto-log --auto-slack  # 部分+全自動
"""

import sys
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    cfg = {}
    for line in open(ROOT / "config" / "env.toml"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def close_trade(trade_id: str, units: str | None = None, reason: str = "", auto_log: bool = False, auto_slack: bool = False):
    cfg = load_config()
    token = cfg["oanda_token"]
    acct = cfg["oanda_account_id"]
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
            # Determine side from units sign
            u = int(closed_units) if closed_units != "?" else 0
            side = "LONG" if u < 0 else "SHORT"  # close fills have opposite sign
            abs_units = abs(u)
            is_partial = units is not None

            # Get spread from fill
            half_spread = float(fill.get("halfSpreadCost", "0"))
            # Estimate spread in pips
            pip_mult = 100 if "JPY" in inst else 10000
            spread_pips = abs(half_spread * 2 / abs_units * pip_mult) if abs_units > 0 else 0

            print(f"CLOSED: {inst} {closed_units}u @{price} P/L={pl} JPY")

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            reason_str = f" reason={reason}" if reason else ""
            action = "PARTIAL_CLOSE" if is_partial else "CLOSE"

            # Auto-log to live_trade_log.txt
            if auto_log:
                log_line = f"[{now_str}] {action} {inst} {side} {abs_units}u @{price} P/L={pl}JPY Sp={spread_pips:.1f}pip{reason_str} id={trade_id}\n"
                log_path = ROOT / "logs" / "live_trade_log.txt"
                with open(log_path, "a") as f:
                    f.write(log_line)
                print(f"  → logged to live_trade_log.txt")

            # Auto-slack notification
            if auto_slack:
                try:
                    import subprocess
                    slack_cmd = [
                        sys.executable, str(ROOT / "tools" / "slack_trade_notify.py"),
                        "close",
                        "--pair", inst,
                        "--side", side,
                        "--units", str(abs_units),
                        "--price", str(price),
                        "--pl", f"{pl}JPY",
                    ]
                    subprocess.run(slack_cmd, capture_output=True, timeout=10)
                    print(f"  → Slack notified")
                except Exception as e:
                    print(f"  → Slack failed: {e}")

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

    # Parse args: trade_id [units] [--reason X] [--auto-log] [--auto-slack]
    trade_id = sys.argv[1]
    units = None
    reason = ""
    auto_log = False
    auto_slack = False

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--reason" and i + 1 < len(sys.argv):
            reason = sys.argv[i + 1]
            i += 2
        elif arg == "--auto-log":
            auto_log = True
            i += 1
        elif arg == "--auto-slack":
            auto_slack = True
            i += 1
        elif not arg.startswith("--") and units is None:
            units = arg
            i += 1
        else:
            i += 1

    close_trade(trade_id, units, reason, auto_log, auto_slack)

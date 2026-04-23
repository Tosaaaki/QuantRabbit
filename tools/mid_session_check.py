#!/usr/bin/env python3
"""
Mid-session lightweight check — replaces full session_data.py on 2nd+ cycle.

Only fetches what changes within a recurring session (seconds, not hours):
1. Slack user messages (new since session start)
2. OANDA prices + spreads (live)
3. Open trades with P&L (live)

Everything else (technicals, macro, news, memory, fib, S-scan) is stable
within the same lock-held session window — no need to re-fetch.

Usage: python3 tools/mid_session_check.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from pricing_probe import probe_market

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(ROOT / ".venv" / "bin" / "python")
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]


def load_config():
    cfg = {}
    for line in open(ROOT / "config" / "env.toml"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def oanda_api(path, token, acct):
    url = f"https://api-fxtrade.oanda.com{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def main():
    t0 = time.time()
    cfg = load_config()
    token = cfg["oanda_token"]
    acct = cfg["oanda_account_id"]

    # 1. Slack (user messages since session start)
    print("=== SLACK (user messages) ===")
    ts_file = ROOT / "logs" / ".slack_last_read_ts"
    slack_args = [
        VENV_PYTHON, "tools/slack_read.py",
        "--channel", "C0APAELAQDN",
        "--user-only", "--no-update-ts",
        "--limit", "5",
    ]
    if ts_file.exists():
        last_ts = ts_file.read_text().strip()
        if last_ts:
            slack_args += ["--after", last_ts]
    try:
        r = subprocess.run(slack_args, capture_output=True, text=True, timeout=10, cwd=str(ROOT))
        print(r.stdout.strip() if r.stdout.strip() else "(no new user messages)")
    except Exception as e:
        print(f"(skip: {e})")

    # 2. OANDA prices + spreads
    print("\n=== PRICES ===")
    try:
        prices = oanda_api(
            f"/v3/accounts/{acct}/pricing?instruments={','.join(PAIRS)}", token, acct
        )
        for p in prices.get("prices", []):
            pair = p["instrument"]
            bid = float(p["bids"][0]["price"])
            ask = float(p["asks"][0]["price"])
            pip_factor = 100 if "JPY" in pair else 10000
            spread_pip = (ask - bid) * pip_factor
            warn = " ⚠️ wide" if spread_pip > 2.0 else ""
            print(f"{pair} bid={p['bids'][0]['price']} ask={p['asks'][0]['price']} Sp={spread_pip:.1f}pip{warn}")
    except Exception as e:
        print(f"ERROR: {e}")

    # 2b. Short pricing probe
    print("\n=== LIVE TAPE ===")
    try:
        cfg_obj = {
            "oanda_token": token,
            "oanda_account_id": acct,
            "oanda_base_url": cfg.get("oanda_base_url", "https://api-fxtrade.oanda.com"),
            "oanda_stream_url": cfg.get("oanda_stream_url", "").strip(),
        }
        probe = probe_market(cfg_obj, pairs=PAIRS, samples=6, interval_sec=0.40, write_cache=True)
        print(
            f"mode={probe.get('mode_used')} "
            f"duration={float(probe.get('duration_sec', 0.0)):.2f}s"
        )
        for pair in PAIRS:
            summary = probe["pairs"].get(pair) or {}
            if summary.get("tape") == "unavailable":
                print(f"{pair}: unavailable")
                continue
            print(
                f"{pair}: {summary.get('bias')} | tape={summary.get('tape')} | "
                f"move={float(summary.get('delta_pips', 0.0)):+.1f}pip | "
                f"spread avg/max={float(summary.get('avg_spread_pips', 0.0)):.1f}/"
                f"{float(summary.get('max_spread_pips', 0.0)):.1f}pip"
            )
    except Exception as e:
        print(f"(skip: {e})")

    # 3. Open trades with unrealized P&L
    print("\n=== TRADES ===")
    try:
        trades = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
        for t in trades.get("trades", []):
            print(f"{t['instrument']} {t['currentUnits']}u @{t['price']} PL={t.get('unrealizedPL', 0)} id={t['id']}")
        if not trades.get("trades"):
            print("(no open trades)")
    except Exception as e:
        print(f"ERROR: {e}")

    # 4. Account margin
    print("\n=== ACCOUNT ===")
    try:
        summary = oanda_api(f"/v3/accounts/{acct}/summary", token, acct).get("account", {})
        nav = float(summary.get("NAV", 0))
        margin_used = float(summary.get("marginUsed", 0))
        margin_pct = (margin_used / nav * 100) if nav > 0 else 0
        margin_warn = ""
        if margin_pct >= 95:
            margin_warn = " 🚨 CRITICAL"
        elif margin_pct >= 90:
            margin_warn = " 🚨 DANGER"
        print(f"NAV:{summary.get('NAV')} Margin:{margin_pct:.1f}%{margin_warn}")
    except Exception as e:
        print(f"ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\n[mid_session_check: {elapsed:.1f}s]")


if __name__ == "__main__":
    main()

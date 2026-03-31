#!/usr/bin/env python3
"""
Session Data — fetch all data needed at trader session start in a single command

Consolidates Bash steps ②③④ into one. This single script covers:
1. Technical refresh (refresh_factor_cache)
2. OANDA: prices, positions, account
3. Macro view (macro_view)
4. Adaptive Technicals
5. Slack: user messages
6. Memory recall: lessons for held pairs
7. Today's performance

Usage: python3 tools/session_data.py [--state-ts LAST_SLACK_TS]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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


def run_script(args, timeout=30):
    """Run a script in a subprocess. Does not abort on failure."""
    try:
        r = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT)
        )
        return r.stdout.strip()
    except Exception as e:
        return f"(skip: {e})"


def section(title):
    print(f"\n=== {title} ===")


def main():
    t0 = time.time()

    # Parse args
    last_slack_ts = ""
    if "--state-ts" in sys.argv:
        idx = sys.argv.index("--state-ts")
        if idx + 1 < len(sys.argv):
            last_slack_ts = sys.argv[idx + 1]

    cfg = load_config()
    token = cfg["oanda_token"]
    acct = cfg["oanda_account_id"]

    # 1. Technical refresh
    section("TECH REFRESH")
    out = run_script(
        [sys.executable, "tools/refresh_factor_cache.py", "--all", "--quiet"],
        timeout=45,
    )
    print(out[:200] if out else "done")

    # 2. OANDA: prices, positions, account
    section("PRICES")
    spread_data = {}  # pair -> spread_pips (referenced in other sections)
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
            spread_data[pair] = spread_pip
            warn = " ⚠️ spread wide" if spread_pip > 2.0 else ""
            print(
                f"{pair} bid={p['bids'][0]['price']} ask={p['asks'][0]['price']} Sp={spread_pip:.1f}pip{warn}"
            )
    except Exception as e:
        print(f"ERROR: {e}")

    section("TRADES")
    held_pairs = set()
    try:
        trades = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
        for t in trades.get("trades", []):
            pair = t["instrument"]
            held_pairs.add(pair)
            print(
                f"{pair} {t['currentUnits']}u @{t['price']} PL={t.get('unrealizedPL', 0)} id={t['id']}"
            )
        if not trades.get("trades"):
            print("(no open trades)")
    except Exception as e:
        print(f"ERROR: {e}")

    section("ACCOUNT")
    try:
        summary = oanda_api(f"/v3/accounts/{acct}/summary", token, acct).get(
            "account", {}
        )
        nav = float(summary.get("NAV", 0))
        margin_used = float(summary.get("marginUsed", 0))
        margin_pct = (margin_used / nav * 100) if nav > 0 else 0
        print(
            f"NAV:{summary.get('NAV')} Bal:{summary.get('balance')} "
            f"Margin:{summary.get('marginUsed')}/{summary.get('marginAvailable')} "
            f"({margin_pct:.1f}%)"
        )
    except Exception as e:
        print(f"ERROR: {e}")

    # 2b. Pending Orders (limit orders, TP/SL check)
    section("PENDING ORDERS")
    try:
        pending = oanda_api(f"/v3/accounts/{acct}/pendingOrders", token, acct)
        orders = pending.get("orders", [])
        if orders:
            for o in orders:
                otype = o.get("type", "?")
                pair = o.get("instrument", "?")
                units = o.get("units", "?")
                price = o.get("price", "?")
                gtd = o.get("gtdTime", "GTC")[:16] if o.get("gtdTime") else "GTC"
                print(f"{otype} {pair} {units}u @{price} exp={gtd} id={o.get('id', '?')}")
        else:
            print("(no pending orders)")
    except Exception as e:
        print(f"ERROR: {e}")

    # 2c. Trade attached orders (TP/SL/Trailing)
    section("TRADE PROTECTIONS")
    try:
        for t in trades.get("trades", []):
            protections = []
            if t.get("takeProfitOrder"):
                protections.append(f"TP={t['takeProfitOrder'].get('price', '?')}")
            if t.get("stopLossOrder"):
                protections.append(f"SL={t['stopLossOrder'].get('price', '?')}")
            if t.get("trailingStopLossOrder"):
                protections.append(f"Trail={t['trailingStopLossOrder'].get('distance', '?')}")
            if protections:
                print(f"{t['instrument']} id={t['id']}: {' | '.join(protections)}")
            else:
                print(f"{t['instrument']} id={t['id']}: ⚠️ NO PROTECTION")
    except Exception:
        pass

    # 2d. News digest (created by Cowork on 1-hour interval)
    section("NEWS DIGEST")
    news_digest = ROOT / "logs" / "news_digest.md"
    if news_digest.exists():
        digest_text = news_digest.read_text().strip()
        # Freshness check: file modification time
        age_min = (time.time() - news_digest.stat().st_mtime) / 60
        if age_min > 90:
            print(f"⚠️ news stale ({age_min:.0f}min ago)")
        print(digest_text[:2000])  # max 2000 chars
    else:
        print("(news_digest.md not found — Cowork qr-news-digest task not run)")

    # 2e. API parser structured data (re-fetch if cache is stale)
    out = run_script(
        [sys.executable, "tools/news_fetcher.py", "--if-stale", "60"],
        timeout=20,
    )
    if out and "skip" not in out:
        print(out[:200])
    # Show summary if cache exists
    news_cache = ROOT / "logs" / "news_cache.json"
    if news_cache.exists():
        out = run_script([sys.executable, "tools/news_fetcher.py", "--summary"])
        if out and "no cache" not in out:
            section("NEWS DATA (structured)")
            print(out)

    # 3. Macro view
    section("MACRO VIEW")
    out = run_script([sys.executable, "tools/macro_view.py"])
    print(out)

    # 4. Adaptive technicals
    section("ADAPTIVE TECHNICALS")
    out = run_script([sys.executable, "tools/adaptive_technicals.py"])
    print(out)

    # 4b. Fib Wave Analysis
    section("FIB WAVE ANALYSIS")
    out = run_script([sys.executable, "tools/fib_wave.py", "--all"])
    if out:
        for line in out.strip().split("\n")[:50]:
            print(line)

    # 5. Slack: user messages
    section("SLACK (user messages)")
    slack_args = [
        sys.executable,
        "tools/slack_read.py",
        "--channel",
        "C0APAELAQDN",
        "--user-only",
    ]
    if last_slack_ts:
        slack_args += ["--after", last_slack_ts, "--limit", "20"]
    else:
        slack_args += ["--limit", "5"]
    out = run_script(slack_args)
    print(out if out else "(no user messages)")

    # 6. Memory recall (lessons for held pairs)
    if held_pairs:
        section("MEMORY RECALL")
        memory_dir = ROOT / "collab_trade" / "memory"
        for pair in sorted(held_pairs):
            try:
                r = subprocess.run(
                    [sys.executable, "recall.py", "search", f"{pair} lessons failures", "--top", "2"],
                    capture_output=True, text=True, timeout=15, cwd=str(memory_dir),
                )
                out = r.stdout.strip()
            except Exception as e:
                out = f"(skip: {e})"
            if out and "skip" not in out:
                print(f"--- {pair} ---")
                # limit to 10 lines
                lines = out.split("\n")[:10]
                print("\n".join(lines))

    # 7. Today's performance
    section("PERFORMANCE (today)")
    out = run_script([sys.executable, "tools/trade_performance.py", "--days", "1"])
    if out:
        lines = out.split("\n")[:20]
        print("\n".join(lines))

    elapsed = time.time() - t0
    print(f"\n[session_data: {elapsed:.1f}s]")


if __name__ == "__main__":
    main()

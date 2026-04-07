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

    # Parse args — last_slack_ts from CLI or auto-read from file
    last_slack_ts = ""
    if "--state-ts" in sys.argv:
        idx = sys.argv.index("--state-ts")
        if idx + 1 < len(sys.argv):
            last_slack_ts = sys.argv[idx + 1]
    if not last_slack_ts:
        ts_file = ROOT / "logs" / ".slack_last_read_ts"
        if ts_file.exists():
            last_slack_ts = ts_file.read_text().strip()

    cfg = load_config()
    token = cfg["oanda_token"]
    acct = cfg["oanda_account_id"]

    # 1. Technical refresh
    section("TECH REFRESH")
    out = run_script(
        [VENV_PYTHON, "tools/refresh_factor_cache.py", "--all", "--quiet"],
        timeout=45,
    )
    print(out[:200] if out else "done")

    # 1b. M5 PRICE ACTION — output FIRST so the model reads chart shape before indicators
    section("M5 PRICE ACTION (read this FIRST — before indicators)")
    try:
        # Fetch M5 candles for all pairs (last 20 bars)
        for pair in PAIRS:
            candles_resp = oanda_api(
                f"/v3/instruments/{pair}/candles?granularity=M5&count=20&price=M",
                token, acct,
            )
            candles = candles_resp.get("candles", [])
            if not candles:
                continue
            # Analyze candle shape
            bodies = []
            upper_wicks = []
            lower_wicks = []
            pip_factor = 100 if "JPY" in pair else 10000
            directions = []  # 1=bull, -1=bear, 0=doji
            for c in candles:
                mid = c.get("mid", {})
                o, h, l, cl = float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"])
                body = abs(cl - o) * pip_factor
                bodies.append(body)
                if cl >= o:  # bull
                    upper_wicks.append((h - cl) * pip_factor)
                    lower_wicks.append((o - l) * pip_factor)
                    directions.append(1)
                else:  # bear
                    upper_wicks.append((h - o) * pip_factor)
                    lower_wicks.append((cl - l) * pip_factor)
                    directions.append(-1)
                if body < 0.3:
                    directions[-1] = 0

            # First half vs second half — momentum change detection
            half = len(bodies) // 2
            first_avg = sum(bodies[:half]) / max(half, 1)
            second_avg = sum(bodies[half:]) / max(len(bodies) - half, 1)
            if second_avg > first_avg * 1.3:
                momentum = "accelerating (bodies growing)"
            elif second_avg < first_avg * 0.7:
                momentum = "exhausting (bodies shrinking)"
            else:
                momentum = "steady"

            # Recent direction (last 5 candles)
            recent = directions[-5:]
            bulls = sum(1 for d in recent if d > 0)
            bears = sum(1 for d in recent if d < 0)
            if bulls >= 4:
                bias = "buyers dominant"
            elif bears >= 4:
                bias = "sellers dominant"
            elif bulls >= 3:
                bias = "buyers leaning"
            elif bears >= 3:
                bias = "sellers leaning"
            else:
                bias = "contested"

            # Wick analysis (reversal pressure)
            avg_upper = sum(upper_wicks[-5:]) / 5
            avg_lower = sum(lower_wicks[-5:]) / 5
            wick_note = ""
            if avg_upper > second_avg * 0.5 and avg_upper > avg_lower * 1.5:
                wick_note = " | upper wicks expanding (selling pressure)"
            elif avg_lower > second_avg * 0.5 and avg_lower > avg_upper * 1.5:
                wick_note = " | lower wicks expanding (buying pressure)"

            # High/low update pattern
            last_c = candles[-1]["mid"]
            prev_c = candles[-2]["mid"]
            hh = float(last_c["h"]) > float(prev_c["h"])
            ll = float(last_c["l"]) < float(prev_c["l"])
            hl_note = ""
            if hh and not ll:
                hl_note = " | making higher highs"
            elif ll and not hh:
                hl_note = " | making lower lows"
            elif hh and ll:
                hl_note = " | range expanding"

            last_price = float(candles[-1]["mid"]["c"])
            print(f"{pair} @{last_price:.5g}: {bias}, {momentum}{wick_note}{hl_note}")
    except Exception as e:
        print(f"(skip: {e})")

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
        margin_warn = ""
        if margin_pct >= 95:
            margin_warn = " 🚨 CRITICAL — force half-close now (rule: 95%+)"
        elif margin_pct >= 90:
            margin_warn = " 🚨 DANGER — no new entries (rule: 90%+)"
        print(
            f"NAV:{summary.get('NAV')} Bal:{summary.get('balance')} "
            f"Margin:{summary.get('marginUsed')}/{summary.get('marginAvailable')} "
            f"({margin_pct:.1f}%){margin_warn}"
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
        [VENV_PYTHON, "tools/news_fetcher.py", "--if-stale", "60"],
        timeout=20,
    )
    if out and "skip" not in out:
        print(out[:200])
    # Show summary if cache exists
    news_cache = ROOT / "logs" / "news_cache.json"
    if news_cache.exists():
        out = run_script([VENV_PYTHON, "tools/news_fetcher.py", "--summary"])
        if out and "no cache" not in out:
            section("NEWS DATA (structured)")
            print(out)

    # 3. Macro view
    section("MACRO VIEW")
    out = run_script([VENV_PYTHON, "tools/macro_view.py"])
    print(out)

    # 4. Adaptive technicals
    section("ADAPTIVE TECHNICALS")
    out = run_script([VENV_PYTHON, "tools/adaptive_technicals.py"])
    print(out)

    # 4a. S-Conviction Scan (TF × indicator pattern detection)
    section("S-CONVICTION CANDIDATES")
    out = run_script([VENV_PYTHON, "tools/s_conviction_scan.py"])
    print(out)

    # 4b. Fib Wave Analysis
    section("FIB WAVE ANALYSIS")
    out = run_script([VENV_PYTHON, "tools/fib_wave.py", "--all"])
    if out:
        for line in out.strip().split("\n")[:50]:
            print(line)

    # 5. Slack: user messages
    section("SLACK (user messages)")
    slack_args = [
        VENV_PYTHON,
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
                    [VENV_PYTHON, "recall.py", "search", f"{pair} lessons failures", "--top", "2"],
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

    # 6b. Quality Audit (if issues exist from last audit run)
    audit_path = ROOT / "logs" / "quality_audit.md"
    if audit_path.exists():
        age_min = (time.time() - audit_path.stat().st_mtime) / 60
        if age_min < 60:  # only show if recent (< 1 hour)
            section("QUALITY AUDIT ISSUES (read and fix)")
            audit_text = audit_path.read_text().strip()
            # Print up to 1500 chars — enough for the issues list
            print(audit_text[:1500])

    # 7. Today's performance
    section("PERFORMANCE (today)")
    out = run_script([VENV_PYTHON, "tools/trade_performance.py", "--days", "1"])
    if out:
        lines = out.split("\n")[:20]
        print("\n".join(lines))

    elapsed = time.time() - t0
    print(f"\n[session_data: {elapsed:.1f}s]")


if __name__ == "__main__":
    main()

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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


PAIR_CURRENCIES = {
    "USD_JPY": ("USD", "JPY"),
    "EUR_USD": ("EUR", "USD"),
    "GBP_USD": ("GBP", "USD"),
    "AUD_USD": ("AUD", "USD"),
    "EUR_JPY": ("EUR", "JPY"),
    "GBP_JPY": ("GBP", "JPY"),
    "AUD_JPY": ("AUD", "JPY"),
}

CORRELATED_PAIRS = [
    ("EUR_USD", "GBP_USD", "anti-USD"),
    ("EUR_JPY", "GBP_JPY", "anti-JPY"),
    ("AUD_USD", "AUD_JPY", "pro-AUD"),
]


def _load_technicals(root, pair):
    f = root / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("timeframes", {})


def _print_currency_pulse(root):
    """Cross-currency triangulation at H4, M15, M1."""
    # Gather DI signals per currency per TF
    ccy_signals = {c: {tf: [] for tf in ["H4", "M15", "M1"]} for c in ["USD", "JPY", "EUR", "GBP", "AUD"]}

    all_tech = {}
    for pair in PAIRS:
        all_tech[pair] = _load_technicals(root, pair)

    for pair, (base, quote) in PAIR_CURRENCIES.items():
        tech = all_tech.get(pair, {})
        for tf in ["H4", "M15", "M1"]:
            d = tech.get(tf, {})
            di_p = d.get("plus_di", 0)
            di_m = d.get("minus_di", 0)
            adx = d.get("adx", 0)
            if adx < 5:
                continue
            gap = (di_p - di_m) * min(adx / 25.0, 1.5)
            ccy_signals[base][tf].append((pair, gap))
            ccy_signals[quote][tf].append((pair, -gap))

    # Print per currency
    strongest_h4 = ("", -999)
    weakest_h4 = ("", 999)

    for ccy in ["USD", "JPY", "EUR", "GBP", "AUD"]:
        parts = []
        for tf in ["H4", "M15", "M1"]:
            sigs = ccy_signals[ccy][tf]
            if not sigs:
                parts.append(f"{tf}=?")
                continue
            avg = sum(g for _, g in sigs) / len(sigs)
            if avg > 3:
                label = f"BID(+{avg:.0f})"
            elif avg < -3:
                label = f"offered({avg:.0f})"
            else:
                label = f"neutral({avg:+.0f})"
            parts.append(f"{tf}={label}")

            if tf == "H4":
                if avg > strongest_h4[1]:
                    strongest_h4 = (ccy, avg)
                if avg < weakest_h4[1]:
                    weakest_h4 = (ccy, avg)

        # Detect H4↔M15 conflict
        h4_sigs = ccy_signals[ccy]["H4"]
        m15_sigs = ccy_signals[ccy]["M15"]
        h4_avg = sum(g for _, g in h4_sigs) / len(h4_sigs) if h4_sigs else 0
        m15_avg = sum(g for _, g in m15_sigs) / len(m15_sigs) if m15_sigs else 0

        conflict = ""
        if h4_avg > 3 and m15_avg < -3:
            conflict = " → H4↔M15 CONFLICT (H4 bid but M15 selling)"
        elif h4_avg < -3 and m15_avg > 3:
            conflict = " → H4↔M15 CONFLICT (H4 offered but M15 buying)"
        elif abs(h4_avg) <= 3 and abs(m15_avg) <= 3:
            conflict = " → No direction"
        else:
            conflict = ""

        print(f"{ccy}: {' | '.join(parts)}{conflict}")

    # M1 synchrony detection
    for ccy in ["USD", "JPY", "EUR", "GBP", "AUD"]:
        m1_sigs = ccy_signals[ccy]["M1"]
        if len(m1_sigs) >= 3:
            all_pos = all(g > 0 for _, g in m1_sigs)
            all_neg = all(g < 0 for _, g in m1_sigs)
            if all_pos:
                print(f"⚠ M1 synchrony: {ccy} BID across {len(m1_sigs)}/{len(m1_sigs)} crosses")
            elif all_neg:
                print(f"⚠ M1 synchrony: {ccy} OFFERED across {len(m1_sigs)}/{len(m1_sigs)} crosses")

    # Correlation break detection
    for pair_a, pair_b, label in CORRELATED_PAIRS:
        tech_a = all_tech.get(pair_a, {}).get("M15", {})
        tech_b = all_tech.get(pair_b, {}).get("M15", {})
        di_a = tech_a.get("plus_di", 0) - tech_a.get("minus_di", 0)
        di_b = tech_b.get("plus_di", 0) - tech_b.get("minus_di", 0)
        adx_a = tech_a.get("adx", 0)
        adx_b = tech_b.get("adx", 0)
        # Break = one clearly directional (ADX>15, |DI gap|>5) and other flat or opposite
        if adx_a > 15 and abs(di_a) > 5 and (adx_b < 12 or di_a * di_b < 0):
            dir_a = "bull" if di_a > 0 else "bear"
            dir_b = "flat" if adx_b < 12 else ("bull" if di_b > 0 else "bear")
            print(f"⚠ Correlation break ({label}): {pair_a} M15={dir_a}(ADX={adx_a:.0f}) vs {pair_b} M15={dir_b}(ADX={adx_b:.0f})")
        elif adx_b > 15 and abs(di_b) > 5 and (adx_a < 12 or di_a * di_b < 0):
            dir_a = "flat" if adx_a < 12 else ("bull" if di_a > 0 else "bear")
            dir_b = "bull" if di_b > 0 else "bear"
            print(f"⚠ Correlation break ({label}): {pair_a} M15={dir_a}(ADX={adx_a:.0f}) vs {pair_b} M15={dir_b}(ADX={adx_b:.0f})")

    # Best vehicle
    if strongest_h4[0] and weakest_h4[0] and strongest_h4[0] != weakest_h4[0]:
        # Find the pair that matches strongest vs weakest
        best_pair = None
        best_dir = None
        for pair, (base, quote) in PAIR_CURRENCIES.items():
            if base == strongest_h4[0] and quote == weakest_h4[0]:
                best_pair = pair
                best_dir = "LONG"
            elif base == weakest_h4[0] and quote == strongest_h4[0]:
                best_pair = pair
                best_dir = "SHORT"
        if best_pair:
            print(f"Best vehicle: {strongest_h4[0]}(+{strongest_h4[1]:.0f}) vs {weakest_h4[0]}({weakest_h4[1]:.0f}) → {best_pair} {best_dir}")
        else:
            print(f"Best vehicle: {strongest_h4[0]}(+{strongest_h4[1]:.0f}) vs {weakest_h4[0]}({weakest_h4[1]:.0f}) → no direct pair")


def _print_h4_position(root):
    """Where in the H4 wave — lifecycle label for each pair."""
    for pair in PAIRS:
        tech = _load_technicals(root, pair)
        h4 = tech.get("H4", {})
        if not h4:
            print(f"{pair}: NO H4 DATA")
            continue

        stoch_rsi = h4.get("stoch_rsi", 0.5)
        s5 = h4.get("ema_slope_5", 0)
        s10 = h4.get("ema_slope_10", 0)
        adx = h4.get("adx", 0)
        di_p = h4.get("plus_di", 0)
        di_m = h4.get("minus_di", 0)
        vwap_gap = h4.get("vwap_gap", 0)
        cci = h4.get("cci", 0)

        # Direction
        bull = di_p > di_m
        direction = "BULL" if bull else "BEAR"
        if adx < 20:
            direction = "NEUTRAL"

        # StRSI zone
        if stoch_rsi >= 0.9:
            zone = "ceiling!"
        elif stoch_rsi >= 0.7:
            zone = "upper"
        elif stoch_rsi >= 0.4:
            zone = "mid"
        elif stoch_rsi >= 0.15:
            zone = "lower"
        else:
            zone = "floor"

        # Momentum quality (slope comparison)
        abs_s5 = abs(s5)
        abs_s10 = abs(s10)
        if abs_s10 > 1e-8:
            ratio = abs_s5 / abs_s10
        else:
            ratio = 1.0

        if ratio > 1.15:
            accel = "accelerating"
        elif ratio < 0.75:
            accel = "decelerating"
        else:
            accel = "steady"

        # Lifecycle label
        if direction == "NEUTRAL":
            label = "NO TREND"
        elif stoch_rsi >= 0.9 and bull:
            label = "EXHAUSTING BULL" if accel != "accelerating" else "LATE BULL (accel)"
        elif stoch_rsi <= 0.1 and not bull:
            label = "EXHAUSTING BEAR" if accel != "accelerating" else "LATE BEAR (accel)"
        elif stoch_rsi >= 0.9 and not bull:
            label = "EARLY BEAR" if adx > 25 else "BEAR SETUP"
        elif stoch_rsi <= 0.1 and bull:
            label = "EARLY BULL" if adx > 25 else "BULL SETUP"
        elif stoch_rsi >= 0.6:
            label = f"LATE {direction}" if accel == "decelerating" else f"MID {direction}"
        elif stoch_rsi <= 0.3:
            label = f"EARLY {direction}" if accel != "decelerating" else f"BOTTOM {direction}"
        else:
            label = f"MID {direction}"

        room = "room" if 0.15 < stoch_rsi < 0.85 else "limited" if 0.85 <= stoch_rsi or stoch_rsi <= 0.15 else "mid"

        print(
            f"{pair}: StRSI={stoch_rsi:.2f}({zone}) "
            f"slope={accel} "
            f"VWAP={vwap_gap:+.0f}pip "
            f"→ {label} [{room}]"
        )


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

    # === Session time marker (first thing in output) ===
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    if 0 <= hour < 3:
        session_label = "Tokyo"
    elif 3 <= hour < 6:
        session_label = "Tokyo (pre-London positioning)"
    elif 6 <= hour < 8:
        session_label = "Tokyo-London overlap"
    elif 8 <= hour < 12:
        session_label = "London"
    elif 12 <= hour < 16:
        session_label = "London-NY overlap"
    elif 16 <= hour < 21:
        session_label = "NY"
    else:
        session_label = "Late NY (rollover zone)"
    utc_stamp = now_utc.strftime('%Y-%m-%d %H:%M UTC')
    print(f"=== SESSION: {utc_stamp} | {session_label} ===")
    print(f"state.md timestamp: **Last Updated**: {utc_stamp}")

    # === PARALLEL BLOCK: Start heavy I/O tasks concurrently ===
    executor = ThreadPoolExecutor(max_workers=4)

    def _run_tech_refresh():
        return run_script(
            [VENV_PYTHON, "tools/refresh_factor_cache.py", "--all", "--quiet"],
            timeout=45,
        )

    def _run_m5_candles():
        results = {}
        for pair in PAIRS:
            try:
                candles_resp = oanda_api(
                    f"/v3/instruments/{pair}/candles?granularity=M5&count=20&price=M",
                    token, acct,
                )
                results[pair] = candles_resp.get("candles", [])
            except Exception as e:
                results[pair] = []
        return results

    def _run_memory_recall(pairs_list):
        recall_results = {}
        memory_dir = ROOT / "collab_trade" / "memory"
        for pair in sorted(pairs_list):
            try:
                r = subprocess.run(
                    [VENV_PYTHON, "recall.py", "search", f"{pair} lessons failures", "--top", "2"],
                    capture_output=True, text=True, timeout=15, cwd=str(memory_dir),
                )
                recall_results[pair] = r.stdout.strip()
            except Exception as e:
                recall_results[pair] = f"(skip: {e})"
        return recall_results

    # Submit heavy tasks in parallel
    future_tech = executor.submit(_run_tech_refresh)
    future_m5 = executor.submit(_run_m5_candles)
    # Memory recall needs held_pairs — we'll get that from OANDA first (fast)

    # Quick OANDA calls (< 1s total) to get held_pairs for memory recall
    held_pairs = set()
    trades_data = {}
    try:
        trades_data = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
        for t in trades_data.get("trades", []):
            held_pairs.add(t["instrument"])
    except Exception:
        pass

    # Now submit memory recall with held_pairs
    future_recall = executor.submit(_run_memory_recall, held_pairs) if held_pairs else None

    # Wait for tech refresh (needed before adaptive_technicals)
    tech_out = future_tech.result()
    section("TECH REFRESH")
    print(tech_out[:200] if tech_out else "done")

    # Wait for M5 candles and display
    m5_candles = future_m5.result()
    section("M5 PRICE ACTION (read this FIRST — before indicators)")
    try:
        for pair in PAIRS:
            candles = m5_candles.get(pair, [])
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

    # Load pair edge stats from strategy_feedback.json (written by trade_performance.py)
    pair_edge = {}
    feedback_path = ROOT / "logs" / "strategy_feedback.json"
    if feedback_path.exists():
        try:
            fb = json.loads(feedback_path.read_text())
            bp = fb.get("by_pair", {})
            if isinstance(bp, dict):
                pair_edge = bp  # keys are pair names, values are stat dicts
            elif isinstance(bp, list):
                for entry in bp:
                    pair_edge[entry.get("pair", "")] = entry
        except Exception:
            pass

    section("TRADES")
    try:
        trades = trades_data  # already fetched above for held_pairs
        for t in trades.get("trades", []):
            pair = t['instrument']
            units = int(t.get('currentUnits', 0))
            side = "LONG" if units > 0 else "SHORT"
            # Inline pair edge
            edge_str = ""
            pe = pair_edge.get(pair)
            if pe:
                wr = pe.get("win_rate", 0)
                total = pe.get("total_pl_jpy", pe.get("total_pl", 0))
                edge_str = f" | edge: {wr:.0%} WR, {total:+.0f}JPY total"
            print(
                f"{pair} {t['currentUnits']}u @{t['price']} PL={t.get('unrealizedPL', 0)} id={t['id']}{edge_str}"
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

    # Churn detection: today's entry count per pair
    log_path = ROOT / "logs" / "live_trade_log.txt"
    if log_path.exists():
        today_str = now_utc.strftime("%Y-%m-%d")
        from collections import Counter
        entry_counts = Counter()
        for line in log_path.read_text().strip().split("\n"):
            if "ENTRY" in line and today_str in line:
                for p in PAIRS:
                    if p in line:
                        entry_counts[p] += 1
                        break
        if entry_counts:
            parts = [f"{p}×{c}" for p, c in entry_counts.most_common()]
            total = sum(entry_counts.values())
            churn_warn = ""
            max_pair_count = entry_counts.most_common(1)[0][1] if entry_counts else 0
            if max_pair_count >= 4:
                churn_warn = " ⚠️ churn risk — same pair 4+ times"
            print(f"Today's entries: {' '.join(parts)} | total {total}{churn_warn}")

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

    # 2d2. Economic calendar (upcoming events from news_cache)
    section("UPCOMING EVENTS (next 4h)")
    try:
        nc = ROOT / "logs" / "news_cache.json"
        if nc.exists():
            ncdata = json.loads(nc.read_text())
            cal = ncdata.get("calendar", [])
            if cal:
                shown = 0
                for ev in cal[:10]:
                    title = ev.get("event", "")
                    impact = ev.get("impact", "")
                    country = ev.get("country", "")
                    ev_time = ev.get("time", "")
                    if title:
                        impact_str = f" ({impact} impact)" if impact else ""
                        ccy_str = f" — {country}" if country else ""
                        print(f"{ev_time} {title}{impact_str}{ccy_str}")
                        shown += 1
                if shown == 0:
                    print("(no upcoming events)")
            else:
                print("(no calendar data)")
        else:
            print("(news_cache.json not found)")
    except Exception as e:
        print(f"(skip: {e})")

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

    # 3b. Currency Pulse (cross-currency × 3 TFs)
    section("CURRENCY PULSE (cross-currency × H4/M15/M1)")
    try:
        _print_currency_pulse(ROOT)
    except Exception as e:
        print(f"(currency pulse error: {e})")

    # 3c. H4 Position (where in the wave)
    section("H4 POSITION (where in the wave)")
    try:
        _print_h4_position(ROOT)
    except Exception as e:
        print(f"(h4 position error: {e})")

    # 4. Adaptive technicals
    section("ADAPTIVE TECHNICALS")
    out = run_script([VENV_PYTHON, "tools/adaptive_technicals.py"])
    print(out)

    # 4a. S-Conviction Scan (TF × indicator pattern detection)
    section("S-CONVICTION CANDIDATES")
    out = run_script([VENV_PYTHON, "tools/s_conviction_scan.py"])
    print(out)

    # 4b. Fib Wave Analysis (M5)
    section("FIB WAVE ANALYSIS (M5)")
    out = run_script([VENV_PYTHON, "tools/fib_wave.py", "--all"])
    if out:
        for line in out.strip().split("\n")[:50]:
            print(line)

    # 4c. Fib Wave Analysis (H1) — multi-TF confluence
    section("FIB WAVE ANALYSIS (H1)")
    out = run_script([VENV_PYTHON, "tools/fib_wave.py", "--all", "H1", "100"])
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
        "--no-update-ts",
    ]
    if last_slack_ts:
        slack_args += ["--after", last_slack_ts, "--limit", "20"]
    else:
        slack_args += ["--limit", "5"]
    out = run_script(slack_args)
    print(out if out else "(no user messages)")

    # 6. Memory recall (lessons for held pairs) — already fetched in parallel
    if held_pairs and future_recall is not None:
        recall_results = future_recall.result()
        section("MEMORY RECALL")
        for pair in sorted(held_pairs):
            out = recall_results.get(pair, "")
            if out and "skip" not in out:
                print(f"--- {pair} ---")
                lines = out.split("\n")[:10]
                print("\n".join(lines))

    # 6b. Quality Audit (always show if recent — includes Regime Map + Range Opportunities)
    audit_path = ROOT / "logs" / "quality_audit.md"
    if audit_path.exists():
        age_min = (time.time() - audit_path.stat().st_mtime) / 60
        if age_min < 90:  # show if within last 90 min (covers up to 3× 30-min audit cycles)
            audit_text = audit_path.read_text().strip()
            if "CLEAN" not in audit_text:
                section(f"QUALITY AUDIT ({age_min:.0f}min ago — regime + analysis)")
                print(audit_text[:3000])

    # 7. Today's performance
    section("PERFORMANCE (today)")
    out = run_script([VENV_PYTHON, "tools/trade_performance.py", "--days", "1"])
    if out:
        lines = out.split("\n")[:20]
        print("\n".join(lines))

    # 8. Pre-filled templates (v8.4 — model fills blanks, can't skip fields)
    section("WRITE TEMPLATES (copy to state.md, fill ALL blanks)")

    # Self-check template with pre-filled entry count
    entry_count_str = ""
    try:
        today_str = time.strftime("%Y-%m-%d", time.gmtime())
        log_path = ROOT / "logs" / "live_trade_log.txt"
        if log_path.exists():
            from collections import Counter as _C
            _ec = _C()
            for _line in log_path.read_text().strip().split("\n"):
                if "ENTRY" in _line and today_str in _line:
                    for _p in PAIRS:
                        if _p in _line:
                            _ec[_p] += 1
                            break
            if _ec:
                _top = _ec.most_common(1)[0]
                entry_count_str = f"{sum(_ec.values())} total. Most: {_top[0]} ×{_top[1]}"
            else:
                entry_count_str = "0 today"
    except Exception:
        entry_count_str = "? (check log)"

    print(f"""
## Self-check
Entries today: {entry_count_str}. Fixated? [YES: why / NO]
Last 3 closed: [W/L/W]. Streak: [hot/cold/neutral]. If 2+L → B max, don't chase
Bias: Am I holding ___ because thesis alive, or because cutting = admitting wrong?""")

    # Position management template per held trade
    held_trades = trades.get("trades", []) if trades else []
    if held_trades:
        print("\n## Position Management (fill ALL 4 for each C)")
        for t in held_trades:
            pair = t.get("instrument", "?").replace("/", "_")
            tid = t.get("id", "?")
            units = t.get("currentUnits", "?")
            upl = t.get("unrealizedPL", "0")
            # Load M15 data for this pair
            m15_hint = ""
            m1_hint = ""
            h4_hint = ""
            try:
                tech = _load_technicals(ROOT, pair)
                m15 = tech.get("M15", {})
                m1_data = tech.get("M1", {})
                h4 = tech.get("H4", {})
                if m15:
                    di_gap = m15.get("plus_di", 0) - m15.get("minus_di", 0)
                    hist = m15.get("macd_hist", 0)
                    hist_dir = "expanding" if abs(hist) > 0.0001 else "shrinking"
                    m15_hint = f"DI gap={di_gap:+.0f} MACD hist={hist_dir}"
                if h4:
                    sr = h4.get("stoch_rsi", 0.5)
                    zone = "ceiling!" if sr >= 0.9 else "upper" if sr >= 0.7 else "mid" if sr >= 0.4 else "lower" if sr >= 0.15 else "floor"
                    h4_hint = f"StRSI={sr:.2f}({zone})"
                # M1 currency pulse for base currency
                base_ccy = pair[:3]
                m1_sigs = []
                for p2, (b2, q2) in PAIR_CURRENCIES.items():
                    m1d = _load_technicals(ROOT, p2).get("M1", {})
                    if m1d:
                        gap = m1d.get("plus_di", 0) - m1d.get("minus_di", 0)
                        if b2 == base_ccy:
                            m1_sigs.append(gap)
                        elif q2 == base_ccy:
                            m1_sigs.append(-gap)
                if m1_sigs:
                    avg = sum(m1_sigs) / len(m1_sigs)
                    m1_label = "BID" if avg > 3 else "offered" if avg < -3 else "neutral"
                    m1_hint = f"{base_ccy} M1={m1_label}({avg:+.0f}) across {len(m1_sigs)} crosses"
            except Exception:
                pass

            print(f"""
### {pair} {units}u id={tid} UPL={upl}
  A — Close now: {upl} JPY
  B — Half TP: close ___u, trail ___pip
  C — Hold REQUIRES all 4:
    (1) Changed since last session: ___
    (2) Entry TF: ___
        M15: {m15_hint} → [with/against] ___
        M1: {m1_hint} → [supports/threatens] ___
    (3) H4: {h4_hint} → Room? [YES/NO]
    (4) Enter NOW @current? [YES/NO]
  → Chosen: [A/B/C]""")

    # Conviction template (for new entries)
    print(f"""
## New Entry Template (v8.4 — fill ALL fields)
  Thesis: [CHART story, not indicators]
  Last 5 M5: bodies [___] × [___], wicks [___] → Buyers defending? [YES @___ / NO → PASS]
  Regime: [___] | Type: [___] | Expected: [___] → Zombie: [___Z]
  Theme confidence: [proving/confirmed/late] → size: [B/A/S]
  AGAINST: ___
  If wrong: ___
  H4 position: {h4_hint if held_trades else "StRSI=___ →"} [early/mid/late/exhausting]
  Cross-currency: [currency] M15 [bid/offered] across [N] pairs → [currency-wide/pair-specific] → conviction [UP/DOWN]
  Event asymmetry: [event] at [time]. Positioned for [___]. [favorable/unfavorable]
  Margin: ___% → worst case ___% | → Conviction: [S/A/B/C] Size: ___u""")

    elapsed = time.time() - t0
    print(f"\n[session_data: {elapsed:.1f}s]")


if __name__ == "__main__":
    main()

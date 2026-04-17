#!/usr/bin/env python3
"""
Range Scalp Scanner — Find and rate range trading opportunities across all 7 pairs.

The range scalping edge:
  In a confirmed range, risk is BOUNDED (SL outside range).
  Bounded risk = size up. Range + size = explosive profits.
  BB lower buy → BB mid TP = ~50% of range = highest probability trade in FX.
  Rotation: BUY lower → TP mid → SELL upper → TP mid → repeat.

Output: Ready-to-trade range scalp plans with exact levels, sizing, R:R.

Usage:
    python3 tools/range_scalp_scanner.py              # all 7 pairs, M5 + H1
    python3 tools/range_scalp_scanner.py EUR_USD       # single pair detail
    python3 tools/range_scalp_scanner.py --json        # JSON output for programmatic use
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [
    # Original 7 majors + JPY crosses
    "USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    # Added: NZD/CAD/CHF universe (spread < 3.0 pip verified)
    "NZD_USD", "USD_CAD", "USD_CHF", "EUR_GBP",
    "NZD_JPY", "CAD_JPY",
    "EUR_CHF", "AUD_NZD", "AUD_CAD",
]

# Typical spreads in pips (conservative estimates — actual may vary by session)
TYPICAL_SPREADS = {
    "USD_JPY": 0.4, "EUR_USD": 0.6, "GBP_USD": 0.9, "AUD_USD": 0.7,
    "EUR_JPY": 1.2, "GBP_JPY": 2.0, "AUD_JPY": 1.0,
    # Added pairs (live spreads measured 2026-04-17)
    "NZD_USD": 1.5, "USD_CAD": 2.0, "USD_CHF": 1.6, "EUR_GBP": 1.5,
    "NZD_JPY": 3.0, "CAD_JPY": 2.4,
    "EUR_CHF": 1.9, "AUD_NZD": 2.5, "AUD_CAD": 2.7,
}

# JPY per pip for sizing calculations (approximate at current rates)
JPY_PER_PIP_10K = {
    "USD_JPY": 100, "EUR_USD": 159, "GBP_USD": 159, "AUD_USD": 159,
    "EUR_JPY": 100, "GBP_JPY": 100, "AUD_JPY": 100,
    # Added pairs (approximate at current rates)
    "NZD_USD": 159, "USD_CAD": 116, "USD_CHF": 203, "EUR_GBP": 187,
    "NZD_JPY": 100, "CAD_JPY": 100,
    "EUR_CHF": 203, "AUD_NZD": 130, "AUD_CAD": 116,
}


def load_config():
    text = (ROOT / "config" / "env.toml").read_text()
    token = [l.split("=")[1].strip().strip('"') for l in text.split("\n") if l.startswith("oanda_token")][0]
    acct = [l.split("=")[1].strip().strip('"') for l in text.split("\n") if l.startswith("oanda_account_id")][0]
    return token, acct


def oanda_api(path, token):
    url = f"https://api-fxtrade.oanda.com{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


def load_technicals(pair: str) -> dict:
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("timeframes", {})


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def to_pips(diff: float, pair: str) -> float:
    return round(diff / pip_size(pair), 1)


def get_spread(pair: str, prices: dict) -> float:
    """Get current spread from OANDA prices, fall back to typical."""
    p = prices.get(pair, {})
    if p.get("ask") and p.get("bid"):
        return to_pips(p["ask"] - p["bid"], pair)
    return TYPICAL_SPREADS.get(pair, 1.0)


def fetch_prices(token: str, acct: str) -> dict:
    """Fetch current bid/ask for all pairs."""
    pairs_str = ",".join(PAIRS)
    data = oanda_api(f"/v3/accounts/{acct}/pricing?instruments={pairs_str}", token)
    prices = {}
    for p in data.get("prices", []):
        pair = p["instrument"]
        bids = p.get("bids", [{}])
        asks = p.get("asks", [{}])
        prices[pair] = {
            "bid": float(bids[0].get("price", 0)) if bids else 0,
            "ask": float(asks[0].get("price", 0)) if asks else 0,
            "mid": (float(bids[0].get("price", 0)) + float(asks[0].get("price", 0))) / 2 if bids and asks else 0,
        }
    return prices


def fetch_account(token: str, acct: str) -> dict:
    """Fetch NAV and margin info."""
    data = oanda_api(f"/v3/accounts/{acct}/summary", token)
    a = data.get("account", {})
    return {
        "nav": float(a.get("NAV", 0)),
        "margin_used": float(a.get("marginUsed", 0)),
        "margin_available": float(a.get("marginAvailable", 0)),
    }


def analyze_range(pair: str, tf_data: dict, prices: dict) -> dict | None:
    """Analyze a single pair+TF for range scalping opportunity.

    Returns None if not a range, or a dict with full scalp plan.
    """
    if not tf_data:
        return None

    adx = tf_data.get("adx", 50)
    bb_upper = tf_data.get("bb_upper", 0)
    bb_mid = tf_data.get("bb_mid", 0)
    bb_lower = tf_data.get("bb_lower", 0)
    bb_span = tf_data.get("bb_span_pips", 0)
    bbw = tf_data.get("bbw", 999)
    stoch_rsi = tf_data.get("stoch_rsi", 0.5)
    cci = tf_data.get("cci", 0)
    rsi = tf_data.get("rsi", 50)
    atr_pips = tf_data.get("atr_pips", 0)
    close = tf_data.get("close", 0)
    kc_width = tf_data.get("kc_width", 0)
    high_hits = tf_data.get("high_hits", 0)
    low_hits = tf_data.get("low_hits", 0)
    lower_wick_avg = tf_data.get("lower_wick_avg_pips", 0)
    upper_wick_avg = tf_data.get("upper_wick_avg_pips", 0)
    macd_hist = tf_data.get("macd_hist", 0)
    plus_di = tf_data.get("plus_di", 0)
    minus_di = tf_data.get("minus_di", 0)

    ps = pip_size(pair)
    spread = get_spread(pair, prices)

    if bb_upper == 0 or bb_lower == 0 or bb_upper <= bb_lower:
        return None

    # BB position (0 = at lower band, 1 = at upper band)
    bb_pos = (close - bb_lower) / (bb_upper - bb_lower)
    bb_pos = max(0, min(1, bb_pos))

    # Range classification
    # Strict range: ADX < 25 (relaxed from 20 for more opportunities)
    # Semi-range: ADX 20-30 with oscillation evidence
    is_strict_range = adx < 25
    is_oscillating = high_hits >= 2 and low_hits >= 2  # touched both bands
    is_tradeable_size = bb_span > spread * 5  # range must be 5× spread minimum

    # Squeeze detection (BB inside KC = breakout imminent, DON'T range trade)
    is_squeeze = bbw < kc_width * 0.9 if kc_width > 0 else bbw < 0.0015

    # Range health: symmetric touches = healthy range
    touch_ratio = min(high_hits, low_hits) / max(high_hits, low_hits, 1)
    is_symmetric = touch_ratio > 0.4  # at least 40% ratio

    # BB width trend (narrowing = squeeze forming = danger)
    # Use BBW vs ATR ratio — if BBW < ATR*1.2, range is too compressed
    bbw_atr_ratio = bb_span / atr_pips if atr_pips > 0 else 0

    # Spread coverage ratio
    spread_coverage = bb_span / spread if spread > 0 else 0

    # Determine if this is a range opportunity
    range_type = None
    if is_squeeze:
        range_type = "SQUEEZE"  # not tradeable as range
    elif is_strict_range and is_oscillating and is_tradeable_size:
        range_type = "CLEAN_RANGE"
    elif is_strict_range and is_tradeable_size and (high_hits >= 1 or low_hits >= 1):
        range_type = "FORMING_RANGE"
    elif adx < 30 and is_oscillating and is_tradeable_size:
        range_type = "SEMI_RANGE"
    else:
        return None  # not a range

    if range_type == "SQUEEZE":
        return {
            "pair": pair,
            "range_type": "SQUEEZE",
            "tradeable": False,
            "reason": "BB inside KC — breakout imminent, not range tradeable",
            "bb_span": bb_span,
            "spread_coverage": spread_coverage,
        }

    # === SIGNAL DETECTION ===

    # Buy signal (price at or near BB lower)
    buy_signal_strength = 0
    buy_triggers = []

    if bb_pos < 0.20:
        buy_signal_strength += 3
        buy_triggers.append(f"BB pos={bb_pos:.2f} (lower zone)")
    elif bb_pos < 0.35:
        buy_signal_strength += 1
        buy_triggers.append(f"BB pos={bb_pos:.2f} (approaching lower)")

    if stoch_rsi < 0.10:
        buy_signal_strength += 3
        buy_triggers.append(f"StochRSI={stoch_rsi:.2f} (extreme oversold)")
    elif stoch_rsi < 0.25:
        buy_signal_strength += 1
        buy_triggers.append(f"StochRSI={stoch_rsi:.2f} (oversold)")

    if cci < -150:
        buy_signal_strength += 2
        buy_triggers.append(f"CCI={cci:.0f} (extreme)")
    elif cci < -100:
        buy_signal_strength += 1
        buy_triggers.append(f"CCI={cci:.0f} (oversold)")

    if rsi < 35:
        buy_signal_strength += 1
        buy_triggers.append(f"RSI={rsi:.0f} (oversold)")

    if lower_wick_avg > 1.0:
        buy_signal_strength += 1
        buy_triggers.append(f"Lower wick avg={lower_wick_avg:.1f}pip (buyers absorbing)")

    # Sell signal (price at or near BB upper)
    sell_signal_strength = 0
    sell_triggers = []

    if bb_pos > 0.80:
        sell_signal_strength += 3
        sell_triggers.append(f"BB pos={bb_pos:.2f} (upper zone)")
    elif bb_pos > 0.65:
        sell_signal_strength += 1
        sell_triggers.append(f"BB pos={bb_pos:.2f} (approaching upper)")

    if stoch_rsi > 0.90:
        sell_signal_strength += 3
        sell_triggers.append(f"StochRSI={stoch_rsi:.2f} (extreme overbought)")
    elif stoch_rsi > 0.75:
        sell_signal_strength += 1
        sell_triggers.append(f"StochRSI={stoch_rsi:.2f} (overbought)")

    if cci > 150:
        sell_signal_strength += 2
        sell_triggers.append(f"CCI={cci:.0f} (extreme)")
    elif cci > 100:
        sell_signal_strength += 1
        sell_triggers.append(f"CCI={cci:.0f} (overbought)")

    if rsi > 65:
        sell_signal_strength += 1
        sell_triggers.append(f"RSI={rsi:.0f} (overbought)")

    if upper_wick_avg > 1.0:
        sell_signal_strength += 1
        sell_triggers.append(f"Upper wick avg={upper_wick_avg:.1f}pip (sellers rejecting)")

    # Determine active signal
    active_signal = None
    signal_strength = 0
    triggers = []

    if buy_signal_strength >= 4:
        active_signal = "BUY"
        signal_strength = buy_signal_strength
        triggers = buy_triggers
    elif sell_signal_strength >= 4:
        active_signal = "SELL"
        signal_strength = sell_signal_strength
        triggers = sell_triggers
    elif buy_signal_strength >= 2:
        active_signal = "BUY_WATCH"
        signal_strength = buy_signal_strength
        triggers = buy_triggers
    elif sell_signal_strength >= 2:
        active_signal = "SELL_WATCH"
        signal_strength = sell_signal_strength
        triggers = sell_triggers
    else:
        active_signal = "MID_ZONE"
        signal_strength = 0
        triggers = [f"BB pos={bb_pos:.2f} (mid — wait for extremes)"]

    # === TRADE PLAN ===

    # Entry levels
    buy_entry = bb_lower + spread * ps  # above lower band by spread
    sell_entry = bb_upper - spread * ps  # below upper band by spread

    # TP levels
    tp_mid_from_buy = to_pips(bb_mid - buy_entry, pair)
    tp_upper_from_buy = to_pips(bb_upper - buy_entry, pair)
    tp_mid_from_sell = to_pips(sell_entry - bb_mid, pair)
    tp_lower_from_sell = to_pips(sell_entry - bb_lower, pair)

    # SL levels (outside range + buffer)
    sl_buffer_pips = max(2.0, atr_pips * 0.3)
    buy_sl = bb_lower - sl_buffer_pips * ps
    sell_sl = bb_upper + sl_buffer_pips * ps
    buy_sl_pips = to_pips(buy_entry - buy_sl, pair)
    sell_sl_pips = to_pips(sell_sl - sell_entry, pair)

    # R:R calculation
    buy_rr_mid = tp_mid_from_buy / buy_sl_pips if buy_sl_pips > 0 else 0
    buy_rr_full = tp_upper_from_buy / buy_sl_pips if buy_sl_pips > 0 else 0
    sell_rr_mid = tp_mid_from_sell / sell_sl_pips if sell_sl_pips > 0 else 0
    sell_rr_full = tp_lower_from_sell / sell_sl_pips if sell_sl_pips > 0 else 0

    # Conviction rating for range
    conviction = "C"
    if range_type == "CLEAN_RANGE" and is_symmetric and spread_coverage > 8:
        if signal_strength >= 6:
            conviction = "S"
        elif signal_strength >= 4:
            conviction = "A"
        else:
            conviction = "B"
    elif range_type in ("SEMI_RANGE", "FORMING_RANGE") and spread_coverage > 5:
        if signal_strength >= 6:
            conviction = "A"
        elif signal_strength >= 4:
            conviction = "B"
        else:
            conviction = "C"

    # DI balance (in range, DI+/DI- should be relatively balanced)
    di_balance = abs(plus_di - minus_di)
    di_warning = di_balance > 15  # one side strongly dominant = not a clean range

    return {
        "pair": pair,
        "range_type": range_type,
        "tradeable": True,
        "conviction": conviction,
        "active_signal": active_signal,
        "signal_strength": signal_strength,
        "triggers": triggers,
        # Levels
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "bb_pos": round(bb_pos, 3),
        "bb_span": bb_span,
        "close": close,
        "spread": spread,
        "spread_coverage": round(spread_coverage, 1),
        "atr_pips": atr_pips,
        # Buy plan
        "buy_entry": buy_entry,
        "buy_tp_mid": round(tp_mid_from_buy, 1),
        "buy_tp_full": round(tp_upper_from_buy, 1),
        "buy_sl": buy_sl,
        "buy_sl_pips": round(buy_sl_pips, 1),
        "buy_rr_mid": round(buy_rr_mid, 2),
        "buy_rr_full": round(buy_rr_full, 2),
        # Sell plan
        "sell_entry": sell_entry,
        "sell_tp_mid": round(tp_mid_from_sell, 1),
        "sell_tp_full": round(tp_lower_from_sell, 1),
        "sell_sl": sell_sl,
        "sell_sl_pips": round(sell_sl_pips, 1),
        "sell_rr_mid": round(sell_rr_mid, 2),
        "sell_rr_full": round(sell_rr_full, 2),
        # Range health
        "touch_ratio": round(touch_ratio, 2),
        "high_hits": high_hits,
        "low_hits": low_hits,
        "bbw_atr_ratio": round(bbw_atr_ratio, 2),
        "is_symmetric": is_symmetric,
        "di_warning": di_warning,
        "di_balance": round(di_balance, 1),
        # Oscillator state
        "stoch_rsi": stoch_rsi,
        "cci": cci,
        "rsi": rsi,
        "adx": adx,
    }


def format_opportunity(opp: dict, tf: str, account: dict) -> str:
    """Format a single range opportunity for human reading."""
    if not opp or not opp.get("tradeable"):
        return ""

    pair = opp["pair"]
    ps = pip_size(pair)
    nav = account.get("nav", 128000)

    lines = []

    # Header
    signal_icon = {
        "BUY": "BUY NOW",
        "SELL": "SELL NOW",
        "BUY_WATCH": "BUY watch",
        "SELL_WATCH": "SELL watch",
        "MID_ZONE": "wait",
    }.get(opp["active_signal"], "?")

    conv = opp["conviction"]
    lines.append(f"{'='*60}")
    lines.append(f"  {pair} ({tf}) — {opp['range_type']} | {conv}-{signal_icon}")
    lines.append(f"{'='*60}")

    # Range stats
    lines.append(f"  Range: {opp['bb_span']:.1f}pip | Spread: {opp['spread']:.1f}pip | "
                 f"Coverage: {opp['spread_coverage']:.1f}x | ATR: {opp['atr_pips']:.1f}pip")
    lines.append(f"  ADX={opp['adx']:.0f} | Touches: upper={opp['high_hits']:.0f} lower={opp['low_hits']:.0f} "
                 f"| Symmetry: {opp['touch_ratio']:.0%}")

    # BB position bar
    pos = opp["bb_pos"]
    bar_len = 30
    pos_idx = int(pos * bar_len)
    bar = "▁" * pos_idx + "●" + "▁" * (bar_len - pos_idx)
    zone = "LOWER" if pos < 0.25 else "UPPER" if pos > 0.75 else "MID"
    lines.append(f"  BB: [{bar}] {pos:.0%} ({zone})")
    lines.append(f"       {opp['bb_lower']:.5g}        {opp['bb_mid']:.5g}        {opp['bb_upper']:.5g}")

    # Warnings
    if opp.get("di_warning"):
        lines.append(f"  ⚠ DI imbalance: {opp['di_balance']:.0f} — range may break")

    # Active signal and triggers
    lines.append(f"")
    lines.append(f"  Signal: {signal_icon} (strength={opp['signal_strength']})")
    for t in opp["triggers"]:
        lines.append(f"    • {t}")

    # Trade plan
    lines.append(f"")
    if opp["active_signal"] in ("BUY", "BUY_WATCH"):
        lines.append(f"  ▶ BUY @{opp['buy_entry']:.5g} (BB lower + spread)")
        lines.append(f"    TP1: {opp['bb_mid']:.5g} (+{opp['buy_tp_mid']:.1f}pip to BB mid) R:R={opp['buy_rr_mid']:.1f}")
        lines.append(f"    TP2: {opp['bb_upper']:.5g} (+{opp['buy_tp_full']:.1f}pip to BB upper) R:R={opp['buy_rr_full']:.1f}")
        lines.append(f"    SL:  {opp['buy_sl']:.5g} (-{opp['buy_sl_pips']:.1f}pip below range)")

        # Sizing suggestion
        jpy_pip = JPY_PER_PIP_10K.get(pair, 100)
        if conv == "S":
            size_pct, target_margin = 30, 0.30
        elif conv == "A":
            size_pct, target_margin = 15, 0.15
        elif conv == "B":
            size_pct, target_margin = 5, 0.05
        else:
            size_pct, target_margin = 2, 0.02

        margin_budget = nav * target_margin
        price = opp["close"]
        leverage = 25 if "JPY" in pair else 20
        suggested_units = int(margin_budget / (price / leverage))
        expected_profit_mid = opp["buy_tp_mid"] * (jpy_pip / 10000 * suggested_units) if "JPY" not in pair else opp["buy_tp_mid"] * (suggested_units / 100)

        lines.append(f"    Size: {suggested_units:,}u ({conv}, {size_pct}% NAV) → "
                     f"TP1 = ~{expected_profit_mid:.0f} JPY")

    elif opp["active_signal"] in ("SELL", "SELL_WATCH"):
        lines.append(f"  ▶ SELL @{opp['sell_entry']:.5g} (BB upper - spread)")
        lines.append(f"    TP1: {opp['bb_mid']:.5g} (+{opp['sell_tp_mid']:.1f}pip to BB mid) R:R={opp['sell_rr_mid']:.1f}")
        lines.append(f"    TP2: {opp['bb_lower']:.5g} (+{opp['sell_tp_full']:.1f}pip to BB lower) R:R={opp['sell_rr_full']:.1f}")
        lines.append(f"    SL:  {opp['sell_sl']:.5g} (-{opp['sell_sl_pips']:.1f}pip above range)")

        jpy_pip = JPY_PER_PIP_10K.get(pair, 100)
        if conv == "S":
            size_pct, target_margin = 30, 0.30
        elif conv == "A":
            size_pct, target_margin = 15, 0.15
        elif conv == "B":
            size_pct, target_margin = 5, 0.05
        else:
            size_pct, target_margin = 2, 0.02

        margin_budget = nav * target_margin
        price = opp["close"]
        leverage = 25 if "JPY" in pair else 20
        suggested_units = int(margin_budget / (price / leverage))
        expected_profit_mid = opp["sell_tp_mid"] * (jpy_pip / 10000 * suggested_units) if "JPY" not in pair else opp["sell_tp_mid"] * (suggested_units / 100)

        lines.append(f"    Size: {suggested_units:,}u ({conv}, {size_pct}% NAV) → "
                     f"TP1 = ~{expected_profit_mid:.0f} JPY")

    else:
        # Mid-zone — show both plans
        lines.append(f"  Waiting for extremes. Plans ready:")
        lines.append(f"    BUY  @{opp['buy_entry']:.5g} → TP {opp['bb_mid']:.5g} (+{opp['buy_tp_mid']:.1f}pip) SL {opp['buy_sl']:.5g}")
        lines.append(f"    SELL @{opp['sell_entry']:.5g} → TP {opp['bb_mid']:.5g} (+{opp['sell_tp_mid']:.1f}pip) SL {opp['sell_sl']:.5g}")

    # Rotation plan
    lines.append(f"")
    lines.append(f"  Rotation: BUY @{opp['bb_lower']:.5g} → TP @{opp['bb_mid']:.5g} → "
                 f"SELL @{opp['bb_upper']:.5g} → TP @{opp['bb_mid']:.5g} → repeat")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Range Scalp Scanner")
    parser.add_argument("pairs", nargs="*", help="Specific pairs (default: all)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    target_pairs = args.pairs if args.pairs else PAIRS

    token, acct = load_config()
    prices = fetch_prices(token, acct)
    account = fetch_account(token, acct)

    results = {}

    for pair in target_pairs:
        tfs = load_technicals(pair)
        pair_results = {}

        for tf in ["M5", "H1"]:
            tf_data = tfs.get(tf)
            if not tf_data:
                continue
            opp = analyze_range(pair, tf_data, prices)
            if opp:
                pair_results[tf] = opp

        if pair_results:
            results[pair] = pair_results

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # === HUMAN-READABLE OUTPUT ===

    nav = account.get("nav", 0)
    margin_pct = account.get("margin_used", 0) / nav * 100 if nav > 0 else 0

    print(f"\n{'#'*60}")
    print(f"  RANGE SCALP SCANNER")
    print(f"  NAV: {nav:,.0f} JPY | Margin: {margin_pct:.1f}%")
    print(f"{'#'*60}")

    # Classify results
    hot = []    # BUY/SELL signal active
    warm = []   # BUY_WATCH/SELL_WATCH
    cold = []   # MID_ZONE or FORMING
    squeeze = [] # SQUEEZE (breakout)
    no_range = []

    for pair in target_pairs:
        pair_results = results.get(pair, {})
        if not pair_results:
            no_range.append(pair)
            continue

        # Prefer M5 for scalping
        opp = pair_results.get("M5") or pair_results.get("H1")
        if not opp:
            no_range.append(pair)
            continue

        if not opp.get("tradeable"):
            squeeze.append((pair, opp))
        elif opp["active_signal"] in ("BUY", "SELL"):
            hot.append((pair, opp, pair_results))
        elif opp["active_signal"] in ("BUY_WATCH", "SELL_WATCH"):
            warm.append((pair, opp, pair_results))
        else:
            cold.append((pair, opp, pair_results))

    # HOT — Trade NOW
    if hot:
        print(f"\n{'!'*60}")
        print(f"  🔥 RANGE SCALP — TRADE NOW ({len(hot)} pairs)")
        print(f"{'!'*60}")
        for pair, opp, pair_results in sorted(hot, key=lambda x: -x[1]["signal_strength"]):
            for tf, tf_opp in pair_results.items():
                if tf_opp.get("tradeable"):
                    print(format_opportunity(tf_opp, tf, account))

    # WARM — Approaching entry
    if warm:
        print(f"\n{'~'*60}")
        print(f"  APPROACHING ({len(warm)} pairs) — LIMITs ready")
        print(f"{'~'*60}")
        for pair, opp, pair_results in sorted(warm, key=lambda x: -x[1]["signal_strength"]):
            m5 = pair_results.get("M5")
            if m5 and m5.get("tradeable"):
                print(format_opportunity(m5, "M5", account))

    # COLD — Range exists but mid-zone
    if cold:
        print(f"\n{'-'*60}")
        print(f"  RANGE DETECTED ({len(cold)} pairs) — waiting for extremes")
        print(f"{'-'*60}")
        for pair, opp, pair_results in cold:
            m5 = pair_results.get("M5")
            if m5 and m5.get("tradeable"):
                lines = []
                lines.append(f"  {pair}: {opp['range_type']} | BB pos={opp['bb_pos']:.0%} | "
                             f"Range={opp['bb_span']:.1f}pip | Coverage={opp['spread_coverage']:.1f}x")
                lines.append(f"    BUY zone: <{opp['bb_lower']:.5g} | SELL zone: >{opp['bb_upper']:.5g}")
                print("\n".join(lines))

    # SQUEEZE
    if squeeze:
        print(f"\n  SQUEEZE (breakout pending): {', '.join(p for p,_ in squeeze)}")

    # No range
    if no_range:
        print(f"\n  TRENDING (no range): {', '.join(no_range)}")

    # MTF confirmation
    mtf_confirmed = []
    for pair in target_pairs:
        pr = results.get(pair, {})
        if pr.get("M5", {}).get("tradeable") and pr.get("H1", {}).get("tradeable"):
            mtf_confirmed.append(pair)

    if mtf_confirmed:
        print(f"\n  MTF CONFIRMED (M5+H1 both range): {', '.join(mtf_confirmed)}")

    # Summary
    total_hot = len(hot)
    total_opportunities = len(hot) + len(warm) + len(cold)
    print(f"\n  Summary: {total_hot} signals NOW | {total_opportunities} range opportunities | "
          f"{len(squeeze)} squeeze | {len(no_range)} trending")
    print()


if __name__ == "__main__":
    main()

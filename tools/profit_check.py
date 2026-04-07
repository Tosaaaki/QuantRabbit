#!/usr/bin/env python3
"""
Profit Check — Ask the data whether to take profit on open positions right now.

The TP counterpart to pretrade_check. The trader makes the decision; the tool just presents facts.

Usage:
    python3 tools/profit_check.py              # Check all positions with unrealized profit
    python3 tools/profit_check.py --all        # Check all positions including unrealized losses
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
STATE_PATH = ROOT / "collab_trade" / "state.md"

# pip multipliers
PIP_MULT = {
    "USD_JPY": 100, "EUR_JPY": 100, "GBP_JPY": 100, "AUD_JPY": 100,
    "EUR_USD": 10000, "GBP_USD": 10000, "AUD_USD": 10000,
}


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


def load_technicals(pair: str) -> dict:
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("timeframes", {})


def get_peak_from_state(pair: str) -> dict | None:
    """Extract peak record from state.md"""
    if not STATE_PATH.exists():
        return None
    content = STATE_PATH.read_text()
    # Look for patterns like: peak: +3,200 JPY @1.33620
    lines = content.split("\n")
    in_section = False
    peak_info = {}
    for line in lines:
        if pair in line and ("##" in line):
            in_section = True
        elif in_section:
            if line.startswith("##") or line.startswith("---"):
                break
            if "ピーク" in line or "peak" in line.lower():
                m = re.search(r'[+]?([\d,]+)円', line)
                if m:
                    peak_info["peak_yen"] = float(m.group(1).replace(",", ""))
                m2 = re.search(r'@([\d]+\.[\d]+)', line)
                if m2:
                    peak_info["peak_price"] = float(m2.group(1))
    return peak_info if peak_info else None


def classify_momentum(d: dict) -> str:
    """Determine M5 momentum direction and strength"""
    macd_hist = d.get("macd_hist", 0)
    stoch_rsi = d.get("stoch_rsi", 0.5)
    ema_slope = d.get("ema_slope_5", 0)

    signals = []
    # MACD histogram direction
    if macd_hist > 0:
        signals.append("MACD_H+")
    else:
        signals.append("MACD_H-")

    # StochRSI position
    if stoch_rsi > 0.8:
        signals.append(f"StRSI={stoch_rsi:.2f}(overbought)")
    elif stoch_rsi < 0.2:
        signals.append(f"StRSI={stoch_rsi:.2f}(oversold)")
    else:
        signals.append(f"StRSI={stoch_rsi:.2f}")

    # EMA slope
    if abs(ema_slope) < 0.0001:
        signals.append("slope=flat")
    elif ema_slope > 0:
        signals.append("slope↑")
    else:
        signals.append("slope↓")

    return " ".join(signals)


def check_cross_pair_correlation(pair: str, side: str, all_technicals: dict) -> list[str]:
    """7-pair correlation check"""
    notes = []
    # Extract currency from pair
    base, quote = pair.split("_")

    # Find pairs sharing the same base or quote currency
    for other_pair in PAIRS:
        if other_pair == pair:
            continue
        other_base, other_quote = other_pair.split("_")
        tfs = all_technicals.get(other_pair, {})
        m5 = tfs.get("M5", {})
        if not m5:
            continue

        other_slope = m5.get("ema_slope_5", 0)
        other_stoch = m5.get("stoch_rsi", 0.5)

        # Same base currency moving same direction = currency-level move
        if other_base == base:
            if side == "LONG" and other_slope > 0.0002:
                notes.append(f"{other_pair} also {base} bid direction (slope={other_slope:.4f})")
            elif side == "SHORT" and other_slope < -0.0002:
                notes.append(f"{other_pair} also {base} offer direction (slope={other_slope:.4f})")
            elif side == "LONG" and other_slope < -0.0002:
                notes.append(f"⚠ {other_pair} is {base} offer direction (slope={other_slope:.4f}) = adverse")

        # Same quote currency
        if other_quote == quote:
            if side == "LONG" and other_slope > 0.0002:
                notes.append(f"{other_pair} also {quote} offer direction = correlation aligned")
            elif side == "SHORT" and other_slope < -0.0002:
                notes.append(f"{other_pair} also {quote} bid direction = correlation aligned")

    return notes[:4]  # Max 4 notes


def check_sr_distance(d: dict) -> str:
    """S/R distance"""
    parts = []
    cluster_high = d.get("cluster_high_gap", None)
    cluster_low = d.get("cluster_low_gap", None)
    swing_high = d.get("swing_dist_high", None)
    swing_low = d.get("swing_dist_low", None)

    if cluster_high is not None and 0 < cluster_high < 10:
        parts.append(f"res {cluster_high:.1f}pip close")
    if cluster_low is not None and 0 < cluster_low < 10:
        parts.append(f"sup {cluster_low:.1f}pip close")
    if swing_high is not None:
        parts.append(f"swing high {swing_high:.1f}pip")
    if swing_low is not None:
        parts.append(f"swing low {swing_low:.1f}pip")

    return " | ".join(parts) if parts else "no S/R data"


def assess_position(trade: dict, all_technicals: dict) -> dict:
    """Take profit assessment for a single position"""
    pair = trade["instrument"]
    units = int(trade["currentUnits"])
    side = "LONG" if units > 0 else "SHORT"
    upl = float(trade.get("unrealizedPL", 0))
    entry_price = float(trade["price"])
    trade_id = trade["id"]

    tfs = all_technicals.get(pair, {})
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})
    h4 = tfs.get("H4", {})

    # ATR from H1
    atr_pips = h1.get("atr_pips", 0)

    # Calculate pip distance
    pip_mult = PIP_MULT.get(pair, 10000)
    current_price = m5.get("close", 0) or h1.get("close", 0)
    if current_price and entry_price:
        if side == "LONG":
            pip_distance = (current_price - entry_price) * pip_mult
        else:
            pip_distance = (entry_price - current_price) * pip_mult
    else:
        pip_distance = 0

    # ATR ratio
    atr_ratio = pip_distance / atr_pips if atr_pips > 0 else 0

    result = {
        "pair": pair,
        "side": side,
        "units": abs(units),
        "upl": upl,
        "pip_distance": pip_distance,
        "atr_pips": atr_pips,
        "atr_ratio": atr_ratio,
        "trade_id": trade_id,
        "signals": [],
        "recommendation": "HOLD",
    }

    # --- Signal collection ---
    take_signals = 0  # positive = take profit signals
    hold_signals = 0  # positive = hold signals

    # 1. ATR ratio
    if atr_ratio >= 1.5:
        result["signals"].append(f"ATR ratio {atr_ratio:.1f}x ← sufficient profit margin")
        take_signals += 2
    elif atr_ratio >= 1.0:
        result["signals"].append(f"ATR ratio {atr_ratio:.1f}x ← trigger reached")
        take_signals += 1
    elif atr_ratio >= 0.5:
        result["signals"].append(f"ATR ratio {atr_ratio:.1f}x ← room to run")
    else:
        result["signals"].append(f"ATR ratio {atr_ratio:.1f}x ← below ATR")
        hold_signals += 1

    # 2. M5 Momentum
    if m5:
        momentum = classify_momentum(m5)
        result["signals"].append(f"M5 momentum: {momentum}")

        macd_hist = m5.get("macd_hist", 0)
        stoch_rsi = m5.get("stoch_rsi", 0.5)

        # Momentum fading for longs
        if side == "LONG":
            if macd_hist < 0 and stoch_rsi > 0.8:
                result["signals"].append("⚠ M5 MACD reversal+overbought = momentum fading")
                take_signals += 2
            elif stoch_rsi > 0.9:
                result["signals"].append("⚠ M5 StochRSI>0.9 = overbought zone")
                take_signals += 1
        else:  # SHORT
            if macd_hist > 0 and stoch_rsi < 0.2:
                result["signals"].append("⚠ M5 MACD reversal+oversold = momentum fading")
                take_signals += 2
            elif stoch_rsi < 0.1:
                result["signals"].append("⚠ M5 StochRSI<0.1 = oversold")
                take_signals += 1

    # 3. H1 structure
    if h1:
        adx = h1.get("adx", 0)
        plus_di = h1.get("plus_di", 0)
        minus_di = h1.get("minus_di", 0)
        h1_trend = "DI+ dominant" if plus_di > minus_di else "DI- dominant"

        # Is H1 aligned with position?
        h1_aligned = (side == "LONG" and plus_di > minus_di) or \
                     (side == "SHORT" and minus_di > plus_di)

        if h1_aligned and adx > 25:
            result["signals"].append(f"H1 ADX={adx:.0f} {h1_trend} thesis intact")
            hold_signals += 2
        elif h1_aligned:
            result["signals"].append(f"H1 ADX={adx:.0f} {h1_trend} thesis direction aligned (weak)")
            hold_signals += 1
        else:
            result["signals"].append(f"⚠ H1 ADX={adx:.0f} {h1_trend} thesis adverse")
            take_signals += 2

        # H1 divergence against position
        div_score = h1.get("div_rsi_score", 0) + h1.get("div_macd_score", 0)
        if div_score > 0:
            result["signals"].append(f"⚠ H1 divergence detected (score={div_score:.1f})")
            take_signals += 1

        # ADX < 20 = range → BB mid is natural TP
        if adx < 20:
            bb_mid = h1.get("bb_mid", 0)
            if bb_mid:
                bb_dist = (bb_mid - current_price) * pip_mult if side == "LONG" else (current_price - bb_mid) * pip_mult
                if 0 < bb_dist < 5:
                    result["signals"].append(f"⚠ H1 range (ADX<20) + BB mid {bb_dist:.1f}pip away = natural TP")
                    take_signals += 2

    # 4. Cross-pair correlation
    corr_notes = check_cross_pair_correlation(pair, side, all_technicals)
    if corr_notes:
        for note in corr_notes:
            result["signals"].append(note)
            if "adverse" in note:
                take_signals += 1
            elif "aligned" in note or "direction" in note:
                hold_signals += 1

    # 5. S/R distance
    if h1:
        sr = check_sr_distance(h1)
        if "close" in sr:
            result["signals"].append(f"S/R: {sr}")
            take_signals += 1
        else:
            result["signals"].append(f"S/R: {sr}")

    # 6. Peak comparison from state.md
    peak = get_peak_from_state(pair)
    if peak and "peak_yen" in peak:
        peak_yen = peak["peak_yen"]
        if upl > 0 and peak_yen > 0:
            drawdown = peak_yen - upl
            if drawdown > peak_yen * 0.3:
                result["signals"].append(f"⚠ Peak {peak_yen:+,.0f} JPY, pulled back {drawdown:,.0f} JPY ({drawdown/peak_yen*100:.0f}%)")
                take_signals += 2

    # --- Recommendation ---
    if upl <= 0:
        result["recommendation"] = "LOSS_POSITION"
    elif take_signals >= 4:
        result["recommendation"] = "TAKE_PROFIT"
    elif take_signals >= 2 and hold_signals < 3:
        result["recommendation"] = "HALF_TP"
    elif hold_signals >= 3:
        result["recommendation"] = "HOLD"
    elif atr_ratio < 0.5:
        result["recommendation"] = "HOLD(below ATR)"
    else:
        result["recommendation"] = "REVIEW"

    result["take_signals"] = take_signals
    result["hold_signals"] = hold_signals

    return result


def format_result(r: dict) -> str:
    """Format the result for a single position"""
    icon = {
        "TAKE_PROFIT": "🔴",
        "HALF_TP": "🟡",
        "HOLD": "🟢",
        "HOLD(below ATR)": "🟢",
        "REVIEW": "🟡",
        "LOSS_POSITION": "⚪",
    }.get(r["recommendation"], "⚪")

    lines = []
    lines.append(
        f"{icon} {r['pair']} {r['side']} {r['units']}u | "
        f"UPL: {r['upl']:+,.0f} JPY ({r['pip_distance']:+.1f}pip) | "
        f"ATR ratio: {r['atr_ratio']:.1f}x (ATR={r['atr_pips']:.1f}pip) | "
        f"→ {r['recommendation']}"
    )
    for sig in r["signals"]:
        lines.append(f"  {sig}")

    return "\n".join(lines)


def main():
    show_all = "--all" in sys.argv

    cfg = load_config()
    token = cfg["oanda_token"]
    acct = cfg["oanda_account_id"]

    # Get open trades
    try:
        trades_resp = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
        trades = trades_resp.get("trades", [])
    except Exception as e:
        print(f"ERROR getting trades: {e}")
        sys.exit(1)

    if not trades:
        print("=== PROFIT CHECK: no open positions ===")
        return

    # Load all technicals
    all_technicals = {}
    for pair in PAIRS:
        all_technicals[pair] = load_technicals(pair)

    # Assess each position
    results = []
    for trade in trades:
        r = assess_position(trade, all_technicals)
        if show_all or r["upl"] > 0:
            results.append(r)

    # Output
    print(f"=== PROFIT CHECK ({len(results)}/{len(trades)} positions) ===")
    print()

    # Sort: TAKE_PROFIT first, then HALF_TP, then REVIEW, then HOLD
    priority = {"TAKE_PROFIT": 0, "HALF_TP": 1, "REVIEW": 2, "HOLD": 3, "HOLD(below ATR)": 4, "LOSS_POSITION": 5}
    results.sort(key=lambda r: priority.get(r["recommendation"], 3))

    for r in results:
        print(format_result(r))
        print()

    # Summary
    tp_count = sum(1 for r in results if r["recommendation"] in ("TAKE_PROFIT", "HALF_TP"))
    if tp_count > 0:
        print(f"--- {tp_count} take profit candidate(s). Default is to take profit. Justify holding. ---")
    else:
        print(f"--- No take profit candidates. All positions HOLD. ---")


if __name__ == "__main__":
    main()

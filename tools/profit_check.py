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

from config_loader import get_oanda_config
from technicals_json import load_technicals_timeframes

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
STATE_PATH = ROOT / "collab_trade" / "state.md"

# pip multipliers
PIP_MULT = {
    "USD_JPY": 100, "EUR_JPY": 100, "GBP_JPY": 100, "AUD_JPY": 100,
    "EUR_USD": 10000, "GBP_USD": 10000, "AUD_USD": 10000,
}


def load_config():
    return get_oanda_config()


def oanda_api(path, cfg):
    url = f"{cfg['oanda_base_url']}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {cfg['oanda_token']}"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def load_technicals(pair: str) -> dict:
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return load_technicals_timeframes(f)


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


def assess_pullback_quality(pair: str, side: str, all_technicals: dict) -> dict:
    """Pullback Quality Data Panel — raw indicator data for trader to read and interpret.

    Shows 12 indicators organized by category. No verdict, no score, no recommendation.
    The trader reads this data panel and writes their own pullback interpretation.
    """
    tfs = all_technicals.get(pair, {})
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})

    if not h1 or not m5:
        return {"has_data": False, "panels": []}

    panels = []

    # --- Panel 1: H1 Trend Health ---
    h1_ema_slope_20 = h1.get("ema_slope_20", 0)
    h1_macd_hist = h1.get("macd_hist", 0)
    h1_div_rsi = h1.get("div_rsi_score", 0)
    h1_div_macd = h1.get("div_macd_score", 0)
    h1_div_total = h1_div_rsi + h1_div_macd
    h1_chaikin = h1.get("chaikin_vol", 0)

    panel_h1 = f"H1: ema_slope_20={h1_ema_slope_20:+.4f}  macd_hist={h1_macd_hist:+.5f}  div={h1_div_total:.1f}  chaikin={h1_chaikin:+.3f}"
    panels.append(panel_h1)

    # --- Panel 2: M5 Volume & Volatility ---
    m5_chaikin = m5.get("chaikin_vol", 0)
    m5_bbw = m5.get("bbw", 0)
    m5_kc = m5.get("kc_width", 0)
    m5_donchian = m5.get("donchian_width", 0)
    bb_vs_kc = "BB<KC" if (m5_kc > 0 and m5_bbw < m5_kc) else "BB>=KC" if m5_kc > 0 else "n/a"

    panel_vol = f"M5: chaikin={m5_chaikin:+.3f}  bbw={m5_bbw:.5f}  kc={m5_kc:.5f}  ({bb_vs_kc})  donch={m5_donchian:.5f}"
    panels.append(panel_vol)

    # --- Panel 3: Candle Character ---
    m5_lower_wick = m5.get("lower_wick_avg_pips", 0)
    m5_upper_wick = m5.get("upper_wick_avg_pips", 0)

    panel_candle = f"Candle: lower_wick={m5_lower_wick:.2f}pip  upper_wick={m5_upper_wick:.2f}pip"
    panels.append(panel_candle)

    # --- Panel 4: Structure (room to run) ---
    if side == "LONG":
        cluster_gap = m5.get("cluster_high_gap", h1.get("cluster_high_gap", 0))
        swing_dist = m5.get("swing_dist_high", h1.get("swing_dist_high", 0))
    else:
        cluster_gap = m5.get("cluster_low_gap", h1.get("cluster_low_gap", 0))
        swing_dist = m5.get("swing_dist_low", h1.get("swing_dist_low", 0))

    cluster_str = f"{cluster_gap:.1f}pip" if cluster_gap is not None else "n/a"
    swing_str = f"{swing_dist:.1f}pip" if swing_dist is not None else "n/a"
    panel_struct = f"Structure: cluster_gap={cluster_str}  swing_dist={swing_str}"
    panels.append(panel_struct)

    # --- Panel 5: Momentum (ROC) ---
    m5_roc5 = m5.get("roc5", 0)
    m5_roc10 = m5.get("roc10", 0)
    panel_roc = f"ROC: roc5={m5_roc5:+.4f}  roc10={m5_roc10:+.4f}"
    panels.append(panel_roc)

    # --- Panel 6: Cross-pair alignment ---
    base, quote = pair.split("_")
    aligned_pairs = []
    adverse_pairs = []
    for other_pair in PAIRS:
        if other_pair == pair:
            continue
        other_base, other_quote = other_pair.split("_")
        if other_base != base and other_quote != quote:
            continue
        other_m5 = all_technicals.get(other_pair, {}).get("M5", {})
        other_slope = other_m5.get("ema_slope_5", 0)
        if other_base == base:
            if (side == "LONG" and other_slope > 0.0001) or (side == "SHORT" and other_slope < -0.0001):
                aligned_pairs.append(other_pair)
            elif (side == "LONG" and other_slope < -0.0001) or (side == "SHORT" and other_slope > 0.0001):
                adverse_pairs.append(other_pair)
        elif other_quote == quote:
            if (side == "LONG" and other_slope > 0.0001) or (side == "SHORT" and other_slope < -0.0001):
                aligned_pairs.append(other_pair)
            elif (side == "LONG" and other_slope < -0.0001) or (side == "SHORT" and other_slope > 0.0001):
                adverse_pairs.append(other_pair)

    total_related = len(aligned_pairs) + len(adverse_pairs)
    aligned_str = ",".join(aligned_pairs) if aligned_pairs else "none"
    adverse_str = ",".join(adverse_pairs) if adverse_pairs else "none"
    panel_xpair = f"Cross-pair: {len(aligned_pairs)}/{total_related} aligned ({aligned_str})  adverse ({adverse_str})"
    panels.append(panel_xpair)

    return {"has_data": True, "panels": panels}


def assess_position(trade: dict, all_technicals: dict) -> dict:
    """Take profit assessment for a single position"""
    pair = trade["instrument"]
    units = int(trade["currentUnits"])
    side = "LONG" if units > 0 else "SHORT"
    upl = float(trade.get("unrealizedPL", 0))
    entry_price = float(trade["price"])
    trade_id = trade["id"]

    # Time held calculation
    from datetime import datetime, timezone
    open_time_str = trade.get("openTime", "")
    time_held_str = ""
    if open_time_str:
        try:
            open_time = datetime.fromisoformat(open_time_str.replace("Z", "+00:00").split(".")[0] + "+00:00")
            now = datetime.now(timezone.utc)
            elapsed = now - open_time
            hours = int(elapsed.total_seconds() // 3600)
            mins = int((elapsed.total_seconds() % 3600) // 60)
            if hours > 0:
                time_held_str = f"{hours}h{mins:02d}m"
            else:
                time_held_str = f"{mins}m"
        except Exception:
            time_held_str = "?"

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
        "time_held": time_held_str,
        "signals": [],
        "recommendation": "HOLD",
    }

    # --- Signal collection ---
    # Design: take_signals = reasons to exit. hold_signals = reasons the position is ACTIVELY working.
    # H1 alignment and cross-pair correlation are CONTEXT (displayed, not counted).
    # Only M5 active momentum counts as a hold signal — "the move is still happening NOW."
    take_signals = 0
    hold_signals = 0

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
        hold_signals += 1  # Too early to take profit

    # 2. M5 Momentum
    if m5:
        momentum = classify_momentum(m5)
        result["signals"].append(f"M5 momentum: {momentum}")

        macd_hist = m5.get("macd_hist", 0)
        stoch_rsi = m5.get("stoch_rsi", 0.5)
        ema_slope = m5.get("ema_slope_5", 0)

        # Momentum fading → take signal
        if side == "LONG":
            if macd_hist < 0 and stoch_rsi > 0.8:
                result["signals"].append("⚠ M5 MACD reversal+overbought = momentum fading")
                take_signals += 2
            elif stoch_rsi > 0.9:
                result["signals"].append("⚠ M5 StochRSI>0.9 = overbought zone")
                take_signals += 1
            # M5 ACTIVELY strong → hold signal (the only real hold signal)
            if macd_hist > 0 and 0.2 < stoch_rsi < 0.8 and ema_slope > 0.0001:
                result["signals"].append("M5 momentum active (MACD+, StRSI mid, slope↑)")
                hold_signals += 1
        else:  # SHORT
            if macd_hist > 0 and stoch_rsi < 0.2:
                result["signals"].append("⚠ M5 MACD reversal+oversold = momentum fading")
                take_signals += 2
            elif stoch_rsi < 0.1:
                result["signals"].append("⚠ M5 StochRSI<0.1 = oversold")
                take_signals += 1
            # M5 ACTIVELY strong SHORT
            if macd_hist < 0 and 0.2 < stoch_rsi < 0.8 and ema_slope < -0.0001:
                result["signals"].append("M5 momentum active (MACD-, StRSI mid, slope↓)")
                hold_signals += 1

    # 3. H1 structure — CONTEXT ONLY (displayed, NOT counted in hold_signals)
    #    H1 trend doesn't change for hours. It's not a reason to HOLD — it's background.
    #    Only H1 ADVERSE counts (as a take signal) because that means thesis is dead.
    if h1:
        adx = h1.get("adx", 0)
        plus_di = h1.get("plus_di", 0)
        minus_di = h1.get("minus_di", 0)
        h1_trend = "DI+ dominant" if plus_di > minus_di else "DI- dominant"

        h1_aligned = (side == "LONG" and plus_di > minus_di) or \
                     (side == "SHORT" and minus_di > plus_di)

        if h1_aligned and adx > 25:
            result["signals"].append(f"H1 ADX={adx:.0f} {h1_trend} (context — not a hold signal)")
        elif h1_aligned:
            result["signals"].append(f"H1 ADX={adx:.0f} {h1_trend} weak (context)")
        else:
            result["signals"].append(f"⚠ H1 ADX={adx:.0f} {h1_trend} THESIS ADVERSE")
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

    # 4. Cross-pair correlation — CONTEXT ONLY (displayed, not counted as hold)
    #    Adverse correlation is still a take signal.
    corr_notes = check_cross_pair_correlation(pair, side, all_technicals)
    if corr_notes:
        for note in corr_notes:
            if "adverse" in note:
                result["signals"].append(note)
                take_signals += 1
            else:
                result["signals"].append(f"{note} (context)")

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
        if peak_yen > 0:
            drawdown = peak_yen - upl
            if upl <= 0:
                result["signals"].append(
                    f"🚨 Peak {peak_yen:+,.0f} JPY has fully round-tripped to {upl:+,.0f} JPY ({drawdown:,.0f} JPY give-back)"
                )
                take_signals += 4
            elif drawdown > peak_yen * 0.5:
                result["signals"].append(
                    f"🚨 Peak {peak_yen:+,.0f} JPY, gave back {drawdown:,.0f} JPY ({drawdown/peak_yen*100:.0f}%)"
                )
                take_signals += 3
            elif drawdown > peak_yen * 0.3:
                result["signals"].append(f"⚠ Peak {peak_yen:+,.0f} JPY, pulled back {drawdown:,.0f} JPY ({drawdown/peak_yen*100:.0f}%)")
                take_signals += 2

    # 7. Time-held penalty — based on April data:
    #    Losers cut <30m: avg -354/trade. Losers held >2h: avg -818/trade (2.3× worse).
    #    39 slow-cut losers (>2h) cost -31,890 JPY = 75% of ALL losses.
    if time_held_str and time_held_str != "?":
        try:
            open_time = datetime.fromisoformat(open_time_str.replace("Z", "+00:00").split(".")[0] + "+00:00")
            now = datetime.now(timezone.utc)
            elapsed_min = (now - open_time).total_seconds() / 60
            if elapsed_min >= 480:  # 8h+
                result["signals"].append(f"⚠ ZOMBIE: held {time_held_str} (8h+). Close unless this is a deliberate swing.")
                take_signals += 3
            elif elapsed_min >= 240:  # 4h+
                result["signals"].append(f"⚠ Extended hold: {time_held_str} (4h+). Losers held >2h avg -818/trade.")
                take_signals += 2
            elif elapsed_min >= 120:  # 2h+
                result["signals"].append(f"⚠ Held {time_held_str} (2h+). If losing, cut now — slow cuts cost 2.3× more.")
                if upl <= 0:
                    take_signals += 1  # Only penalize if losing
        except Exception:
            pass

    # --- Recommendation ---
    # Design change (v8.2): HALF_TP no longer gated by hold_signals < 3.
    # Old: HALF_TP required take >= 2 AND hold < 3 (impossible with H1 trend + correlation).
    # New: HALF_TP triggers on take >= 2 regardless. H1/correlation are context, not hold votes.
    if upl <= 0:
        result["recommendation"] = "LOSS_POSITION"
    elif take_signals >= 4:
        result["recommendation"] = "TAKE_PROFIT"
    elif take_signals >= 2:
        result["recommendation"] = "HALF_TP"
    elif atr_ratio < 0.5 or hold_signals >= 2:
        result["recommendation"] = "HOLD"
    else:
        result["recommendation"] = "REVIEW"

    result["take_signals"] = take_signals
    result["hold_signals"] = hold_signals

    # --- Pullback Quality Check (for profitable positions at ATR×0.8+) ---
    if upl > 0 and atr_ratio >= 0.8:
        pq = assess_pullback_quality(pair, side, all_technicals)
        result["pullback_quality"] = pq
    else:
        result["pullback_quality"] = None

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
    time_str = f" | held: {r['time_held']}" if r.get("time_held") else ""
    lines.append(
        f"{icon} {r['pair']} {r['side']} {r['units']}u | "
        f"UPL: {r['upl']:+,.0f} JPY ({r['pip_distance']:+.1f}pip) | "
        f"ATR ratio: {r['atr_ratio']:.1f}x (ATR={r['atr_pips']:.1f}pip){time_str} | "
        f"→ {r['recommendation']}"
    )
    for sig in r["signals"]:
        lines.append(f"  {sig}")

    # Pullback Quality data panel (if available)
    pq = r.get("pullback_quality")
    if pq and pq.get("has_data"):
        lines.append("  ── Pullback Data (read and interpret) ──")
        for panel in pq["panels"]:
            lines.append(f"  {panel}")

    return "\n".join(lines)


def main():
    show_all = "--all" in sys.argv

    # --- Market state check (time-based only) ---
    from market_state import get_market_state
    mkt_state, mkt_reason = get_market_state()

    cfg = load_config()
    token = cfg["oanda_token"]
    acct = cfg["oanda_account_id"]

    # Get open trades
    try:
        trades_resp = oanda_api(f"/v3/accounts/{acct}/openTrades", cfg)
        trades = trades_resp.get("trades", [])
    except Exception as e:
        print(f"ERROR getting trades: {e}")
        sys.exit(1)

    if not trades:
        print("=== PROFIT CHECK: no open positions ===")
        return

    # If market is CLOSED or ROLLOVER, suppress TP recommendations
    if mkt_state in ("CLOSED", "ROLLOVER"):
        print(f"=== PROFIT CHECK — MARKET {mkt_state} ===")
        print(f"  {mkt_reason}")
        print(f"  Spread is unreliable. TP recommendations SUSPENDED.")
        print(f"  DO NOT execute market orders. Wait for market to reopen.")
        print()
        # Still list positions for reference (info only)
        for trade in trades:
            pair = trade["instrument"]
            upl = float(trade.get("unrealizedPL", 0))
            units = int(trade["currentUnits"])
            side = "LONG" if units > 0 else "SHORT"
            print(f"  {pair} {side} {abs(units)}u | UPL: {upl:+,.0f} JPY | HOLD(MARKET {mkt_state})")
        print()
        print(f"--- All recommendations suspended until market reopens. ---")
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

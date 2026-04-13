#!/usr/bin/env python3
"""
Protection Check — Evaluate TP/SL/Trailing Stop/BE protection status for all open positions

For each position:
- Display current protection status (TP/SL/Trailing presence)
- TP recommendation based on structural levels (swing/cluster/BB/Ichimoku)
- SL evaluation: structural level menu (NOT ATR formula). ATR shown as size reference only
- BE move and trailing stop recommendations
- Output is DATA, not orders. The trader decides

Usage:
    python3 tools/protection_check.py          # Check all trades
"""
from __future__ import annotations
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]

PIP_MULT = {
    "USD_JPY": 100, "EUR_JPY": 100, "GBP_JPY": 100, "AUD_JPY": 100,
    "EUR_USD": 10000, "GBP_USD": 10000, "AUD_USD": 10000,
}

PRICE_DECIMALS = {
    "USD_JPY": 3, "EUR_JPY": 3, "GBP_JPY": 3, "AUD_JPY": 3,
    "EUR_USD": 5, "GBP_USD": 5, "AUD_USD": 5,
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


def fmt_price(price: float, pair: str) -> str:
    decimals = PRICE_DECIMALS.get(pair, 5)
    return f"{price:.{decimals}f}"


def pips_to_price(pips: float, pair: str) -> float:
    return pips / PIP_MULT.get(pair, 10000)


def find_structural_levels(pair: str, side: str, current_price: float, tfs: dict) -> list[tuple[float, float, str]]:
    """
    Collect structural levels (S/R) in the TP direction and sort by distance.
    Returns: [(price, distance_pips, label), ...]

    Returns only levels in the TP direction (LONG → levels above, SHORT → levels below)
    """
    pip_mult = PIP_MULT.get(pair, 10000)
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})
    candidates = []

    def add_candidate(price: float, label: str):
        if price <= 0:
            return
        dist_pips = (price - current_price) * pip_mult
        if side == "SHORT":
            dist_pips = -dist_pips  # SHORT: below current = positive distance
        if dist_pips > 1.0:  # Only levels more than 1 pip away
            candidates.append((price, dist_pips, label))

    # --- H1 structural levels ---
    # Swing high/low
    swing_low = h1.get("swing_dist_low")
    swing_high = h1.get("swing_dist_high")
    if side == "SHORT" and swing_low and swing_low > 3:
        add_candidate(current_price - pips_to_price(swing_low, pair), f"H1 swing low ({swing_low:.0f}pip)")
    elif side == "LONG" and swing_high and swing_high > 3:
        add_candidate(current_price + pips_to_price(swing_high, pair), f"H1 swing high ({swing_high:.0f}pip)")

    # Cluster
    cluster_low = h1.get("cluster_low_gap")
    cluster_high = h1.get("cluster_high_gap")
    if side == "SHORT" and cluster_low and cluster_low > 3:
        add_candidate(current_price - pips_to_price(cluster_low, pair), f"H1 cluster ({cluster_low:.0f}pip)")
    elif side == "LONG" and cluster_high and cluster_high > 3:
        add_candidate(current_price + pips_to_price(cluster_high, pair), f"H1 cluster ({cluster_high:.0f}pip)")

    # BB (H1)
    bb_lower = h1.get("bb_lower")
    bb_mid = h1.get("bb_mid")
    bb_upper = h1.get("bb_upper")
    if side == "SHORT":
        if bb_mid and bb_mid < current_price:
            add_candidate(bb_mid, "H1 BB mid")
        if bb_lower and bb_lower < current_price:
            add_candidate(bb_lower, "H1 BB lower")
    else:
        if bb_mid and bb_mid > current_price:
            add_candidate(bb_mid, "H1 BB mid")
        if bb_upper and bb_upper > current_price:
            add_candidate(bb_upper, "H1 BB upper")

    # Ichimoku cloud edges (pip values → price)
    ichi_a = h1.get("ichimoku_span_a_gap")
    ichi_b = h1.get("ichimoku_span_b_gap")
    if ichi_a is not None and abs(ichi_a) > 3:
        ichi_a_price = current_price + pips_to_price(ichi_a, pair)
        if (side == "SHORT" and ichi_a_price < current_price) or (side == "LONG" and ichi_a_price > current_price):
            add_candidate(ichi_a_price, f"H1 Cloud SpanA ({abs(ichi_a):.0f}pip)")
    if ichi_b is not None and abs(ichi_b) > 3:
        ichi_b_price = current_price + pips_to_price(ichi_b, pair)
        if (side == "SHORT" and ichi_b_price < current_price) or (side == "LONG" and ichi_b_price > current_price):
            add_candidate(ichi_b_price, f"H1 Cloud SpanB ({abs(ichi_b):.0f}pip)")

    # --- M5 structural levels ---
    m5_swing_low = m5.get("swing_dist_low")
    m5_swing_high = m5.get("swing_dist_high")
    if side == "SHORT" and m5_swing_low and m5_swing_low > 3:
        add_candidate(current_price - pips_to_price(m5_swing_low, pair), f"M5 swing low ({m5_swing_low:.0f}pip)")
    elif side == "LONG" and m5_swing_high and m5_swing_high > 3:
        add_candidate(current_price + pips_to_price(m5_swing_high, pair), f"M5 swing high ({m5_swing_high:.0f}pip)")

    m5_cluster_low = m5.get("cluster_low_gap")
    m5_cluster_high = m5.get("cluster_high_gap")
    if side == "SHORT" and m5_cluster_low and m5_cluster_low > 3:
        add_candidate(current_price - pips_to_price(m5_cluster_low, pair), f"M5 cluster ({m5_cluster_low:.0f}pip)")
    elif side == "LONG" and m5_cluster_high and m5_cluster_high > 3:
        add_candidate(current_price + pips_to_price(m5_cluster_high, pair), f"M5 cluster ({m5_cluster_high:.0f}pip)")

    # M5 BB
    m5_bb_lower = m5.get("bb_lower")
    m5_bb_mid = m5.get("bb_mid")
    m5_bb_upper = m5.get("bb_upper")
    if side == "SHORT":
        if m5_bb_lower and m5_bb_lower < current_price:
            add_candidate(m5_bb_lower, "M5 BB lower")
        if m5_bb_mid and m5_bb_mid < current_price:
            add_candidate(m5_bb_mid, "M5 BB mid")
    else:
        if m5_bb_upper and m5_bb_upper > current_price:
            add_candidate(m5_bb_upper, "M5 BB upper")
        if m5_bb_mid and m5_bb_mid > current_price:
            add_candidate(m5_bb_mid, "M5 BB mid")

    # Sort by distance (nearest first)
    candidates.sort(key=lambda x: x[1])
    return candidates


def find_structural_sl_levels(pair: str, side: str, entry_price: float, current_price: float, tfs: dict) -> list[tuple[float, float, str]]:
    """
    Collect structural levels in the SL direction (invalidation side) and sort by distance from entry.
    Returns: [(price, distance_pips, label), ...]

    SL direction is opposite of TP: LONG → levels below entry, SHORT → levels above entry.
    Levels must be beyond entry price in the loss direction.
    """
    pip_mult = PIP_MULT.get(pair, 10000)
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})
    h4 = tfs.get("H4", {})
    candidates = []

    def add_candidate(price: float, label: str):
        if price <= 0:
            return
        # Distance from entry in the SL direction (always positive)
        if side == "LONG":
            dist_pips = (entry_price - price) * pip_mult  # Below entry = positive
        else:
            dist_pips = (price - entry_price) * pip_mult  # Above entry = positive
        if dist_pips > 2.0:  # Must be at least 2 pip from entry (not noise)
            candidates.append((price, dist_pips, label))

    # --- H1 structural levels (strongest) ---
    swing_low = h1.get("swing_dist_low")
    swing_high = h1.get("swing_dist_high")
    if side == "LONG" and swing_low and swing_low > 3:
        # SL below current price at H1 swing low
        add_candidate(current_price - pips_to_price(swing_low, pair), f"H1 swing low ({swing_low:.0f}pip from current)")
    elif side == "SHORT" and swing_high and swing_high > 3:
        add_candidate(current_price + pips_to_price(swing_high, pair), f"H1 swing high ({swing_high:.0f}pip from current)")

    # H1 Cluster (price concentration = strong S/R)
    cluster_low = h1.get("cluster_low_gap")
    cluster_high = h1.get("cluster_high_gap")
    if side == "LONG" and cluster_low and cluster_low > 3:
        add_candidate(current_price - pips_to_price(cluster_low, pair), f"H1 cluster support ({cluster_low:.0f}pip)")
    elif side == "SHORT" and cluster_high and cluster_high > 3:
        add_candidate(current_price + pips_to_price(cluster_high, pair), f"H1 cluster resistance ({cluster_high:.0f}pip)")

    # H1 BB (invalidation side)
    bb_lower = h1.get("bb_lower")
    bb_mid = h1.get("bb_mid")
    bb_upper = h1.get("bb_upper")
    if side == "LONG":
        if bb_lower and bb_lower < entry_price:
            add_candidate(bb_lower, "H1 BB lower")
        if bb_mid and bb_mid < entry_price:
            add_candidate(bb_mid, "H1 BB mid")
    else:
        if bb_upper and bb_upper > entry_price:
            add_candidate(bb_upper, "H1 BB upper")
        if bb_mid and bb_mid > entry_price:
            add_candidate(bb_mid, "H1 BB mid")

    # H1 Ichimoku cloud edges (invalidation side)
    ichi_a = h1.get("ichimoku_span_a_gap")
    ichi_b = h1.get("ichimoku_span_b_gap")
    if ichi_a is not None and abs(ichi_a) > 3:
        ichi_a_price = current_price + pips_to_price(ichi_a, pair)
        if (side == "LONG" and ichi_a_price < entry_price) or (side == "SHORT" and ichi_a_price > entry_price):
            add_candidate(ichi_a_price, f"H1 Cloud SpanA ({abs(ichi_a):.0f}pip)")
    if ichi_b is not None and abs(ichi_b) > 3:
        ichi_b_price = current_price + pips_to_price(ichi_b, pair)
        if (side == "LONG" and ichi_b_price < entry_price) or (side == "SHORT" and ichi_b_price > entry_price):
            add_candidate(ichi_b_price, f"H1 Cloud SpanB ({abs(ichi_b):.0f}pip)")

    # --- M5 structural levels (closer, for scalp SL) ---
    m5_swing_low = m5.get("swing_dist_low")
    m5_swing_high = m5.get("swing_dist_high")
    if side == "LONG" and m5_swing_low and m5_swing_low > 3:
        add_candidate(current_price - pips_to_price(m5_swing_low, pair), f"M5 swing low ({m5_swing_low:.0f}pip)")
    elif side == "SHORT" and m5_swing_high and m5_swing_high > 3:
        add_candidate(current_price + pips_to_price(m5_swing_high, pair), f"M5 swing high ({m5_swing_high:.0f}pip)")

    m5_cluster_low = m5.get("cluster_low_gap")
    m5_cluster_high = m5.get("cluster_high_gap")
    if side == "LONG" and m5_cluster_low and m5_cluster_low > 3:
        add_candidate(current_price - pips_to_price(m5_cluster_low, pair), f"M5 cluster ({m5_cluster_low:.0f}pip)")
    elif side == "SHORT" and m5_cluster_high and m5_cluster_high > 3:
        add_candidate(current_price + pips_to_price(m5_cluster_high, pair), f"M5 cluster ({m5_cluster_high:.0f}pip)")

    # M5 BB (invalidation side)
    m5_bb_lower = m5.get("bb_lower")
    m5_bb_mid = m5.get("bb_mid")
    m5_bb_upper = m5.get("bb_upper")
    if side == "LONG":
        if m5_bb_lower and m5_bb_lower < entry_price:
            add_candidate(m5_bb_lower, "M5 BB lower")
    else:
        if m5_bb_upper and m5_bb_upper > entry_price:
            add_candidate(m5_bb_upper, "M5 BB upper")

    # --- H4 structural levels (for swing SL) ---
    h4_swing_low = h4.get("swing_dist_low")
    h4_swing_high = h4.get("swing_dist_high")
    if side == "LONG" and h4_swing_low and h4_swing_low > 5:
        add_candidate(current_price - pips_to_price(h4_swing_low, pair), f"H4 swing low ({h4_swing_low:.0f}pip)")
    elif side == "SHORT" and h4_swing_high and h4_swing_high > 5:
        add_candidate(current_price + pips_to_price(h4_swing_high, pair), f"H4 swing high ({h4_swing_high:.0f}pip)")

    # Sort by distance from entry (nearest first)
    candidates.sort(key=lambda x: x[1])
    return candidates


def assess_protection(trade: dict, all_technicals: dict, cfg: dict) -> dict:
    """Evaluate protection status of a single position"""
    pair = trade["instrument"]
    units = int(trade["currentUnits"])
    side = "LONG" if units > 0 else "SHORT"
    upl = float(trade.get("unrealizedPL", 0))
    entry_price = float(trade["price"])
    trade_id = trade["id"]

    tfs = all_technicals.get(pair, {})
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})
    pip_mult = PIP_MULT.get(pair, 10000)

    atr_pips = h1.get("atr_pips", 0)
    current_price = m5.get("close", 0) or h1.get("close", 0) or entry_price

    if side == "LONG":
        pip_profit = (current_price - entry_price) * pip_mult
    else:
        pip_profit = (entry_price - current_price) * pip_mult

    # --- Current protection status ---
    tp_order = trade.get("takeProfitOrder")
    sl_order = trade.get("stopLossOrder")
    trailing_order = trade.get("trailingStopLossOrder")

    has_tp = tp_order is not None
    has_sl = sl_order is not None
    has_trailing = trailing_order is not None
    has_any_protection = has_tp or has_sl or has_trailing

    tp_price = float(tp_order["price"]) if has_tp else None
    sl_price = float(sl_order["price"]) if has_sl else None
    trailing_distance = float(trailing_order["distance"]) if has_trailing else None

    sl_dist_pips = None
    sl_atr_ratio = None
    if sl_price is not None:
        if side == "LONG":
            sl_dist_pips = (entry_price - sl_price) * pip_mult
        else:
            sl_dist_pips = (sl_price - entry_price) * pip_mult
        if atr_pips > 0:
            sl_atr_ratio = sl_dist_pips / atr_pips

    tp_remaining_pips = None
    if tp_price is not None:
        if side == "LONG":
            tp_remaining_pips = (tp_price - current_price) * pip_mult
        else:
            tp_remaining_pips = (current_price - tp_price) * pip_mult

    # --- Structural levels ---
    structural_levels = find_structural_levels(pair, side, current_price, tfs)
    structural_sl_levels = find_structural_sl_levels(pair, side, entry_price, current_price, tfs)

    # --- Build output ---
    current_status = []
    recommendations = []
    warnings = []
    put_commands = []  # Immediately executable commands

    acct = cfg["oanda_account_id"]

    # Current status
    if has_tp:
        current_status.append(f"TP: {fmt_price(tp_price, pair)} (remaining {tp_remaining_pips:+.1f}pip)")
    else:
        current_status.append("TP: none")
    if has_sl:
        sl_info = f"SL: {fmt_price(sl_price, pair)} ({sl_dist_pips:.1f}pip"
        if sl_atr_ratio is not None:
            sl_info += f", ATR x{sl_atr_ratio:.1f}"
        sl_info += ")"
        current_status.append(sl_info)
    else:
        current_status.append("SL: none")
    if has_trailing:
        trailing_pips = trailing_distance * pip_mult
        current_status.append(f"Trailing: {trailing_pips:.1f}pip")
    else:
        current_status.append("Trailing: none")

    if not has_any_protection:
        warnings.append("NO PROTECTION")

    # --- SL Evaluation (structural level based — NOT ATR formula) ---
    if atr_pips > 0:
        if has_sl:
            # Check if existing SL is at a meaningful structural level
            if sl_atr_ratio is not None and sl_atr_ratio < 0.7:
                warnings.append(f"SL too tight (ATR x{sl_atr_ratio:.1f}) -- noise stop-out risk")
                if structural_sl_levels:
                    recommendations.append("📍 Structural SL candidates (widen to):")
                    for i, (price, dist, label) in enumerate(structural_sl_levels[:4]):
                        atr_x = dist / atr_pips
                        marker = " <- nearest structural" if i == 0 else ""
                        recommendations.append(f"  {i+1}. {fmt_price(price, pair)} = {label} (ATR x{atr_x:.1f}){marker}")
                else:
                    recommendations.append(f"  No structural levels found. Consider removing SL (discretionary) or widening to ATR x1.0+ ({atr_pips:.0f}pip)")
            elif sl_atr_ratio is not None and sl_atr_ratio > 2.5:
                warnings.append(f"SL too wide: {sl_dist_pips:.1f}pip (ATR x{sl_atr_ratio:.1f})")
                if structural_sl_levels:
                    recommendations.append("📍 Structural SL candidates (tighten to):")
                    for i, (price, dist, label) in enumerate(structural_sl_levels[:4]):
                        atr_x = dist / atr_pips
                        recommendations.append(f"  {i+1}. {fmt_price(price, pair)} = {label} (ATR x{atr_x:.1f})")
                else:
                    recommendations.append(f"  SL is wide but no structural levels found. Your call: keep, tighten, or remove")
        else:
            # No SL — show structural candidates as menu (no auto-recommendation)
            if structural_sl_levels:
                recommendations.append("📍 Structural SL candidates (if you want SL):")
                for i, (price, dist, label) in enumerate(structural_sl_levels[:4]):
                    atr_x = dist / atr_pips
                    recommendations.append(f"  {i+1}. {fmt_price(price, pair)} = {label} (ATR x{atr_x:.1f})")
                recommendations.append(f"  ATR={atr_pips:.0f}pip (size reference only, not placement)")
            else:
                recommendations.append(f"  SL: no structural levels found. ATR={atr_pips:.0f}pip (size reference). Discretionary management or skip SL")

    # --- TP Recommendation (structural level based) ---
    if atr_pips > 0 and structural_levels:
        if not has_tp:
            # No TP → recommend nearest structural level
            nearest = structural_levels[0]
            recommendations.append(f"TP recommendation: {fmt_price(nearest[0], pair)} ({nearest[2]}, {nearest[1]:.0f}pip away)")
            if len(structural_levels) > 1:
                recommendations.append(f"  Half TP @{fmt_price(nearest[0], pair)} → trailing recommended for remainder")
        else:
            tp_atr_ratio = tp_remaining_pips / atr_pips if tp_remaining_pips and atr_pips > 0 else 0
            if tp_remaining_pips and tp_remaining_pips < 0:
                recommendations.append(f"TP already breached (remaining {tp_remaining_pips:.1f}pip)")
            elif tp_atr_ratio > 2.0:
                warnings.append(f"TP too wide: {tp_remaining_pips:.1f}pip remaining (ATR x{tp_atr_ratio:.1f})")
                # Show structural level menu
                recommendations.append("📍 Structural TP candidates:")
                for i, (price, dist, label) in enumerate(structural_levels[:5]):
                    atr_x = dist / atr_pips
                    marker = " <- recommended" if i == 0 else ""
                    recommendations.append(f"  {i+1}. {fmt_price(price, pair)} = {label} (ATR x{atr_x:.1f}){marker}")
                # Output fix command for nearest level
                best = structural_levels[0]
                put_commands.append(
                    f'# TP fix {pair} id={trade_id} → {fmt_price(best[0], pair)} ({best[2]})\n'
                    f'python3 -c "import urllib.request,json; '
                    f"req=urllib.request.Request('https://api-fxtrade.oanda.com/v3/accounts/{acct}/trades/{trade_id}/orders',"
                    f'data=json.dumps({{"takeProfit":{{"price":"{fmt_price(best[0], pair)}","timeInForce":"GTC"}}}}).encode(),'
                    f"headers={{'Authorization':'Bearer '+open('config/env.toml').read().split('oanda_token')[1].split('\"')[1],'Content-Type':'application/json'}},"
                    f"method='PUT'); urllib.request.urlopen(req)\""
                )
    elif atr_pips > 0 and not structural_levels:
        # No structural levels found → ATR-based fallback
        atr_tp_pips = atr_pips * 1.0
        if side == "LONG":
            atr_tp_price = current_price + pips_to_price(atr_tp_pips, pair)
        else:
            atr_tp_price = current_price - pips_to_price(atr_tp_pips, pair)
        recommendations.append(f"TP recommendation (ATR fallback): {fmt_price(atr_tp_price, pair)} (ATR x1.0 = {atr_tp_pips:.1f}pip)")

    # --- BE Recommendation ---
    if atr_pips > 0:
        if pip_profit > atr_pips * 1.5:
            rec_trail = atr_pips * 0.5
            recommendations.append(f"Trailing strongly recommended: unrealized profit {pip_profit:.1f}pip (ATR x{pip_profit/atr_pips:.1f}) → trail {rec_trail:.0f}pip")
            trail_price_dist = pips_to_price(rec_trail, pair)
            put_commands.append(
                f'# Trailing set {pair} id={trade_id} ({rec_trail:.0f}pip)\n'
                f'python3 -c "import urllib.request,json; '
                f"req=urllib.request.Request('https://api-fxtrade.oanda.com/v3/accounts/{acct}/trades/{trade_id}/orders',"
                f'data=json.dumps({{"trailingStopLoss":{{"distance":"{trail_price_dist:.5f}","timeInForce":"GTC"}}}}).encode(),'
                f"headers={{'Authorization':'Bearer '+open('config/env.toml').read().split('oanda_token')[1].split('\"')[1],'Content-Type':'application/json'}},"
                f"method='PUT'); urllib.request.urlopen(req)\""
            )
        elif pip_profit > atr_pips * 0.8:
            if side == "LONG":
                be_price = entry_price + pips_to_price(1, pair)
            else:
                be_price = entry_price - pips_to_price(1, pair)
            recommendations.append(f"Consider BE: unrealized profit {pip_profit:.1f}pip (ATR x{pip_profit/atr_pips:.1f}) → SL→{fmt_price(be_price, pair)}")
        elif pip_profit <= 0:
            recommendations.append("BE: N/A (unrealized loss)")
        else:
            recommendations.append(f"BE: too early (unrealized profit {pip_profit:.1f}pip, ATR x{pip_profit/atr_pips:.1f})")

    # --- Trailing Stop Recommendation ---
    if atr_pips > 0 and pip_profit > atr_pips * 1.0 and not has_trailing:
        rec_trail = atr_pips * 0.6
        trail_price_dist = pips_to_price(rec_trail, pair)
        recommendations.append(f"Trailing recommended: {rec_trail:.0f}pip (ATR x0.6)")
        if not any("Trailing" in cmd for cmd in put_commands):
            put_commands.append(
                f'# Trailing set {pair} id={trade_id} ({rec_trail:.0f}pip)\n'
                f'python3 -c "import urllib.request,json; '
                f"req=urllib.request.Request('https://api-fxtrade.oanda.com/v3/accounts/{acct}/trades/{trade_id}/orders',"
                f'data=json.dumps({{"trailingStopLoss":{{"distance":"{trail_price_dist:.5f}","timeInForce":"GTC"}}}}).encode(),'
                f"headers={{'Authorization':'Bearer '+open('config/env.toml').read().split('oanda_token')[1].split('\"')[1],'Content-Type':'application/json'}},"
                f"method='PUT'); urllib.request.urlopen(req)\""
            )

    return {
        "pair": pair,
        "side": side,
        "units": abs(units),
        "trade_id": trade_id,
        "entry_price": entry_price,
        "current_price": current_price,
        "upl": upl,
        "pip_profit": pip_profit,
        "atr_pips": atr_pips,
        "has_any_protection": has_any_protection,
        "current_status": current_status,
        "warnings": warnings,
        "recommendations": recommendations,
        "put_commands": put_commands,
        "structural_levels": structural_levels,
    }


def format_result(r: dict) -> str:
    pair = r["pair"]
    lines = []
    atr_text = f"ATR={r['atr_pips']:.1f}pip" if r["atr_pips"] > 0 else "ATR=N/A"
    lines.append(
        f"{r['pair']} {r['side']} {'-' if r['side'] == 'SHORT' else '+'}{r['units']}u "
        f"id={r['trade_id']} | entry={fmt_price(r['entry_price'], pair)} | "
        f"UPL={r['upl']:+,.0f}JPY ({r['pip_profit']:+.1f}pip) | {atr_text}"
    )

    status_line = "  Current: " + " | ".join(r["current_status"])
    if not r["has_any_protection"]:
        status_line = "  Current: *** NO PROTECTION ***"
    lines.append(status_line)

    for w in r["warnings"]:
        if w != "NO PROTECTION":
            lines.append(f"  ⚠️ {w}")

    for rec in r["recommendations"]:
        lines.append(f"  {rec}")

    return "\n".join(lines)


def _is_us_dst(dt) -> bool:
    """Check if a UTC datetime falls within US Daylight Saving Time.
    DST: 2nd Sunday of March 02:00 ET — 1st Sunday of November 02:00 ET.
    """
    year = dt.year
    # 2nd Sunday of March
    mar1 = dt.replace(month=3, day=1, hour=0, minute=0, second=0, microsecond=0)
    # weekday: 0=Mon, 6=Sun
    days_to_sun = (6 - mar1.weekday()) % 7
    dst_start = mar1.replace(day=1 + days_to_sun + 7)  # 2nd Sunday
    dst_start = dst_start.replace(hour=7)  # 02:00 ET = 07:00 UTC

    # 1st Sunday of November
    nov1 = dt.replace(month=11, day=1, hour=0, minute=0, second=0, microsecond=0)
    days_to_sun = (6 - nov1.weekday()) % 7
    dst_end = nov1.replace(day=1 + days_to_sun)  # 1st Sunday
    dst_end = dst_end.replace(hour=6)  # 02:00 ET(EDT) = 06:00 UTC

    return dst_start <= dt < dst_end


def _check_spreads_wide() -> tuple[bool, float]:
    """Check if current spreads are abnormally wide (rollover/thin market indicator).

    Returns:
        (is_wide, max_spread_ratio) — max_spread_ratio is the highest
        current_spread / normal_spread across all pairs.
    """
    # Normal spread baselines (pip) — conservative estimates for live account
    NORMAL_SPREAD = {
        "USD_JPY": 0.4, "EUR_USD": 0.5, "GBP_USD": 0.9,
        "AUD_USD": 0.5, "EUR_JPY": 0.8, "GBP_JPY": 1.5, "AUD_JPY": 0.8,
    }
    pip_factor = {
        "USD_JPY": 100, "EUR_JPY": 100, "GBP_JPY": 100, "AUD_JPY": 100,
        "EUR_USD": 10000, "GBP_USD": 10000, "AUD_USD": 10000,
    }
    try:
        cfg = load_config()
        token = cfg["oanda_token"]
        acct = cfg["oanda_account_id"]
        pairs_str = ",".join(NORMAL_SPREAD.keys())
        resp = oanda_api(f"/v3/accounts/{acct}/pricing?instruments={pairs_str}", token, acct)
        max_ratio = 0.0
        for p in resp.get("prices", []):
            pair = p["instrument"]
            if pair not in NORMAL_SPREAD:
                continue
            bid = float(p["bids"][0]["price"])
            ask = float(p["asks"][0]["price"])
            spread_pip = (ask - bid) * pip_factor.get(pair, 10000)
            ratio = spread_pip / NORMAL_SPREAD[pair]
            if ratio > max_ratio:
                max_ratio = ratio
        # Spread > 2x normal = still wide
        return max_ratio >= 2.0, max_ratio
    except Exception:
        # If pricing fails, assume wide (safer to keep SLs removed)
        return True, 999.0


def detect_thin_market() -> tuple[bool, str, bool, int]:
    """Detect thin liquidity conditions (holidays, weekend proximity, off-hours, rollover).

    Returns:
        (is_thin, reason_str, is_rollover, minutes_to_rollover)
        - is_rollover: True if within the rollover danger window
        - minutes_to_rollover: minutes until rollover (negative = past rollover). 999 if not near.
    """
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    hour = now.hour
    reasons = []
    is_rollover = False
    minutes_to_rollover = 999

    # Friday after 18:00 UTC or Saturday/Sunday
    if weekday == 4 and hour >= 18:
        reasons.append("Friday late session (thin liquidity)")
    if weekday in (5, 6):
        reasons.append("Weekend")

    # Check for known holidays (Good Friday, Christmas, New Year, etc.)
    month_day = (now.month, now.day)
    holiday_dates = {
        (1, 1): "New Year",
        (12, 25): "Christmas",
        (12, 26): "Boxing Day",
    }
    if month_day in holiday_dates:
        reasons.append(holiday_dates[month_day])

    # Easter/Good Friday: approximate check (varies by year)
    # For 2026: Good Friday = April 3
    if now.year == 2026 and now.month == 4 and now.day == 3:
        reasons.append("Good Friday")
    if now.year == 2027 and now.month == 3 and now.day == 26:
        reasons.append("Good Friday")

    # Tokyo dead zone: 03:00-06:00 UTC (after NY close, before London)
    if 3 <= hour < 6:
        reasons.append(f"Low-liquidity window ({hour:02d}:00 UTC)")

    # --- OANDA daily rollover detection ---
    # Rollover at 5 PM ET: 21:00 UTC (summer/EDT) or 22:00 UTC (winter/EST)
    # Spread spike window: ~20 min before to ~30 min after (spreads can stay wide 30+ min)
    rollover_hour = 21 if _is_us_dst(now) else 22
    rollover_time = now.replace(hour=rollover_hour, minute=0, second=0, microsecond=0)
    delta = (rollover_time - now).total_seconds() / 60  # minutes until rollover

    # If rollover already passed today, check distance from past rollover
    if delta < -60:
        # More than 1 hour past rollover — not relevant
        pass
    elif -30 <= delta <= 20:
        # Within time-based danger window: 20 min before to 30 min after
        if delta >= 0:
            # Before rollover — always dangerous
            is_rollover = True
            minutes_to_rollover = int(delta)
            reasons.append(f"ROLLOVER in {int(delta)} min ({rollover_hour}:00 UTC)")
        else:
            # After rollover — check actual spreads before declaring safe
            spreads_wide, spread_ratio = _check_spreads_wide()
            if spreads_wide:
                is_rollover = True
                minutes_to_rollover = int(delta)
                reasons.append(
                    f"ROLLOVER window (spreads still {spread_ratio:.1f}x normal, "
                    f"{rollover_hour}:00 UTC was {int(-delta)} min ago)"
                )
            else:
                # Spreads normalized — rollover is over
                minutes_to_rollover = int(delta)
    elif -60 <= delta < -30:
        # 30-60 min after rollover — only flag if spreads are still abnormally wide
        spreads_wide, spread_ratio = _check_spreads_wide()
        if spreads_wide:
            is_rollover = True
            minutes_to_rollover = int(delta)
            reasons.append(
                f"ROLLOVER extended (spreads still {spread_ratio:.1f}x normal, "
                f"{rollover_hour}:00 UTC was {int(-delta)} min ago)"
            )
    elif 0 < delta <= 20:
        minutes_to_rollover = int(delta)

    return bool(reasons), ", ".join(reasons), is_rollover, minutes_to_rollover


def main():
    cfg = load_config()
    token = cfg["oanda_token"]
    acct = cfg["oanda_account_id"]

    try:
        trades_resp = oanda_api(f"/v3/accounts/{acct}/openTrades", token, acct)
        trades = trades_resp.get("trades", [])
    except Exception as e:
        print(f"ERROR getting trades: {e}")
        sys.exit(1)

    if not trades:
        print("=== PROTECTION CHECK: No open positions ===")
        return

    # Thin market / rollover detection
    is_thin, thin_reason, is_rollover, min_to_rollover = detect_thin_market()
    if is_rollover:
        print(f"🔴 ROLLOVER WINDOW: {thin_reason}")
        print(f"   → REMOVE all SL/Trailing NOW. Spreads will spike.")
        print(f"   → Run: python3 tools/rollover_guard.py remove")
        print()
    elif is_thin:
        print(f"⚠️  THIN MARKET DETECTED: {thin_reason}")
        print(f"   → SL/Trail will get noise-clipped. Discretionary management recommended.")
        print(f"   → Do NOT add tight SL. Do NOT add trailing stops.")
        print()

    # Check if rollover guard has saved SLs waiting to be restored
    guard_state_file = ROOT / "logs" / "rollover_guard_state.json"
    if guard_state_file.exists() and not is_rollover:
        print(f"✅ Rollover passed. Saved SLs waiting to be restored.")
        print(f"   → Run: python3 tools/rollover_guard.py restore")
        print()

    all_technicals = {}
    for pair in PAIRS:
        all_technicals[pair] = load_technicals(pair)

    results = []
    all_put_commands = []
    for trade in trades:
        r = assess_protection(trade, all_technicals, cfg)
        results.append(r)
        if not is_thin and not is_rollover:  # Don't suggest fix commands during thin market / rollover
            all_put_commands.extend(r.get("put_commands", []))

    # Sort: unprotected first, then by UPL descending
    results.sort(key=lambda r: (r["has_any_protection"], -abs(r["upl"])))

    print(f"=== PROTECTION CHECK ({len(results)} trades) ===")
    print()

    unprotected_count = 0
    for r in results:
        print(format_result(r))
        print()
        if not r["has_any_protection"]:
            unprotected_count += 1

    if unprotected_count > 0:
        if is_thin:
            print(f"--- {unprotected_count} trades without SL (THIN MARKET — this is correct. Discretionary management.) ---")
        else:
            print(f"--- {unprotected_count} trades with NO PROTECTION (consider adding if you won't be watching) ---")
    else:
        print(f"--- All {len(results)} trades have some form of protection ---")

    # 3-option prompt for each position (forces structured thinking)
    print()
    print("=== POSITION MANAGEMENT — fill in 3 options for each position ===")
    for r in results:
        pair = r["pair"]
        upl = r["upl"]
        pip_profit = r["pip_profit"]
        print(f"\n{pair} {r['side']} {r['units']}u | UPL={upl:+,.0f}JPY ({pip_profit:+.1f}pip)")
        print(f"  A. Hold + adjust (new SL/TP to match current conditions): ___")
        print(f"  B. Cut and re-enter (close now, re-enter at better setup): ___")
        print(f"  C. Hold as-is (current protection is optimal because): ___")
        print(f"  → Decision: ___")

    # Output immediately executable commands (suppressed during thin market)
    if all_put_commands:
        print(f"\n=== Fix Commands ({len(all_put_commands)} items) — copy-paste to execute immediately ===")
        for cmd in all_put_commands:
            print()
            print(cmd)
    elif is_thin and unprotected_count > 0:
        print(f"\n(Fix commands suppressed — thin market. SL/Trail would get noise-clipped.)")


if __name__ == "__main__":
    main()

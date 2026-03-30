#!/usr/bin/env python3
"""
Protection Check — 全保有ポジションのTP/SL/Trailing Stop/BE保護状況を評価・推奨

各ポジションについて:
- 現在の保護状況（TP/SL/Trailing有無）を表示
- 構造的レベル(swing/cluster/BB/Ichimoku)ベースのTP推奨
- ATRベースのSL推奨値
- BE移動・トレイリングストップの推奨
- 即実行可能なPUTコマンドを出力

Usage:
    python3 tools/protection_check.py          # 全トレードチェック
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
    構造的レベル(S/R)を収集して距離順にソート。
    Returns: [(price, distance_pips, label), ...]

    TP方向のレベルのみ返す（LONG→上のレベル、SHORT→下のレベル）
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
        if dist_pips > 1.0:  # 1pip以上先のもののみ
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
            add_candidate(ichi_a_price, f"H1 雲SpanA ({abs(ichi_a):.0f}pip)")
    if ichi_b is not None and abs(ichi_b) > 3:
        ichi_b_price = current_price + pips_to_price(ichi_b, pair)
        if (side == "SHORT" and ichi_b_price < current_price) or (side == "LONG" and ichi_b_price > current_price):
            add_candidate(ichi_b_price, f"H1 雲SpanB ({abs(ichi_b):.0f}pip)")

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


def assess_protection(trade: dict, all_technicals: dict, cfg: dict) -> dict:
    """1ポジションの保護状況を評価"""
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

    # --- Build output ---
    current_status = []
    recommendations = []
    warnings = []
    put_commands = []  # 即実行可能なコマンド

    acct = cfg["oanda_account_id"]

    # Current status
    if has_tp:
        current_status.append(f"TP: {fmt_price(tp_price, pair)} (残{tp_remaining_pips:+.1f}pip)")
    else:
        current_status.append("TP: なし")
    if has_sl:
        sl_info = f"SL: {fmt_price(sl_price, pair)} ({sl_dist_pips:.1f}pip"
        if sl_atr_ratio is not None:
            sl_info += f", ATR x{sl_atr_ratio:.1f}"
        sl_info += ")"
        current_status.append(sl_info)
    else:
        current_status.append("SL: なし")
    if has_trailing:
        trailing_pips = trailing_distance * pip_mult
        current_status.append(f"Trailing: {trailing_pips:.1f}pip")
    else:
        current_status.append("Trailing: なし")

    if not has_any_protection:
        warnings.append("NO PROTECTION")

    # --- SL Recommendation ---
    if atr_pips > 0:
        recommended_sl_pips = atr_pips * 1.2
        if has_sl:
            if sl_atr_ratio is not None and sl_atr_ratio < 0.7:
                warnings.append(f"SL tight (ATR x{sl_atr_ratio:.1f}) -- ノイズで刈られるリスク")
                if side == "LONG":
                    better_sl = entry_price - pips_to_price(recommended_sl_pips, pair)
                else:
                    better_sl = entry_price + pips_to_price(recommended_sl_pips, pair)
                recommendations.append(f"SL拡大推奨: {fmt_price(better_sl, pair)} (ATR x1.2)")
            elif sl_atr_ratio is not None and sl_atr_ratio > 2.5:
                warnings.append(f"SL広すぎ: {sl_dist_pips:.1f}pip (ATR x{sl_atr_ratio:.1f})")
                if side == "LONG":
                    better_sl = entry_price - pips_to_price(recommended_sl_pips, pair)
                else:
                    better_sl = entry_price + pips_to_price(recommended_sl_pips, pair)
                recommendations.append(f"SL縮小推奨: {fmt_price(better_sl, pair)} (ATR x1.2 = {recommended_sl_pips:.1f}pip)")
                put_commands.append(
                    f'# SL修正 {pair} id={trade_id}\n'
                    f'python3 -c "import urllib.request,json; '
                    f"req=urllib.request.Request('https://api-fxtrade.oanda.com/v3/accounts/{acct}/trades/{trade_id}/orders',"
                    f'data=json.dumps({{"stopLoss":{{"price":"{fmt_price(better_sl, pair)}","timeInForce":"GTC"}}}}).encode(),'
                    f"headers={{'Authorization':'Bearer '+open('config/env.toml').read().split('oanda_token')[1].split('\"')[1],'Content-Type':'application/json'}},"
                    f"method='PUT'); urllib.request.urlopen(req)\""
                )
        else:
            if side == "LONG":
                rec_sl = entry_price - pips_to_price(recommended_sl_pips, pair)
            else:
                rec_sl = entry_price + pips_to_price(recommended_sl_pips, pair)
            recommendations.append(f"SL推奨: {fmt_price(rec_sl, pair)} (ATR x1.2 = {recommended_sl_pips:.1f}pip)")

    # --- TP Recommendation (構造的レベルベース) ---
    if atr_pips > 0 and structural_levels:
        if not has_tp:
            # TPなし → 最寄り構造的レベルを推奨
            nearest = structural_levels[0]
            recommendations.append(f"TP推奨: {fmt_price(nearest[0], pair)} ({nearest[2]}, {nearest[1]:.0f}pip先)")
            if len(structural_levels) > 1:
                recommendations.append(f"  半利確@{fmt_price(nearest[0], pair)} → 残りtrailing推奨")
        else:
            tp_atr_ratio = tp_remaining_pips / atr_pips if tp_remaining_pips and atr_pips > 0 else 0
            if tp_remaining_pips and tp_remaining_pips < 0:
                recommendations.append(f"TP既にブレイク済み (残{tp_remaining_pips:.1f}pip)")
            elif tp_atr_ratio > 2.0:
                warnings.append(f"TP広すぎ: 残{tp_remaining_pips:.1f}pip (ATR x{tp_atr_ratio:.1f})")
                # 構造的レベルのメニューを表示
                recommendations.append("📍 構造的TP候補:")
                for i, (price, dist, label) in enumerate(structural_levels[:5]):
                    atr_x = dist / atr_pips
                    marker = " ← 推奨" if i == 0 else ""
                    recommendations.append(f"  {i+1}. {fmt_price(price, pair)} = {label} (ATR x{atr_x:.1f}){marker}")
                # 最寄りレベルでの修正コマンドを出力
                best = structural_levels[0]
                put_commands.append(
                    f'# TP修正 {pair} id={trade_id} → {fmt_price(best[0], pair)} ({best[2]})\n'
                    f'python3 -c "import urllib.request,json; '
                    f"req=urllib.request.Request('https://api-fxtrade.oanda.com/v3/accounts/{acct}/trades/{trade_id}/orders',"
                    f'data=json.dumps({{"takeProfit":{{"price":"{fmt_price(best[0], pair)}","timeInForce":"GTC"}}}}).encode(),'
                    f"headers={{'Authorization':'Bearer '+open('config/env.toml').read().split('oanda_token')[1].split('\"')[1],'Content-Type':'application/json'}},"
                    f"method='PUT'); urllib.request.urlopen(req)\""
                )
    elif atr_pips > 0 and not structural_levels:
        # 構造的レベルが見つからない → ATRベースのフォールバック
        atr_tp_pips = atr_pips * 1.0
        if side == "LONG":
            atr_tp_price = current_price + pips_to_price(atr_tp_pips, pair)
        else:
            atr_tp_price = current_price - pips_to_price(atr_tp_pips, pair)
        recommendations.append(f"TP推奨(ATRフォールバック): {fmt_price(atr_tp_price, pair)} (ATR x1.0 = {atr_tp_pips:.1f}pip)")

    # --- BE Recommendation ---
    if atr_pips > 0:
        if pip_profit > atr_pips * 1.5:
            rec_trail = atr_pips * 0.5
            recommendations.append(f"Trailing強く推奨: 含み益{pip_profit:.1f}pip (ATR x{pip_profit/atr_pips:.1f}) → trail {rec_trail:.0f}pip")
            trail_price_dist = pips_to_price(rec_trail, pair)
            put_commands.append(
                f'# Trailing設定 {pair} id={trade_id} ({rec_trail:.0f}pip)\n'
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
            recommendations.append(f"BE検討: 含み益{pip_profit:.1f}pip (ATR x{pip_profit/atr_pips:.1f}) → SL→{fmt_price(be_price, pair)}")
        elif pip_profit <= 0:
            recommendations.append("BE: N/A (含み損)")
        else:
            recommendations.append(f"BE: まだ早い (含み益{pip_profit:.1f}pip, ATR x{pip_profit/atr_pips:.1f})")

    # --- Trailing Stop Recommendation ---
    if atr_pips > 0 and pip_profit > atr_pips * 1.0 and not has_trailing:
        rec_trail = atr_pips * 0.6
        trail_price_dist = pips_to_price(rec_trail, pair)
        recommendations.append(f"Trailing推奨: {rec_trail:.0f}pip (ATR x0.6)")
        if not any("Trailing" in cmd for cmd in put_commands):
            put_commands.append(
                f'# Trailing設定 {pair} id={trade_id} ({rec_trail:.0f}pip)\n'
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
        f"UPL={r['upl']:+,.0f}円 ({r['pip_profit']:+.1f}pip) | {atr_text}"
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
        print("=== PROTECTION CHECK: ポジションなし ===")
        return

    all_technicals = {}
    for pair in PAIRS:
        all_technicals[pair] = load_technicals(pair)

    results = []
    all_put_commands = []
    for trade in trades:
        r = assess_protection(trade, all_technicals, cfg)
        results.append(r)
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
        print(f"--- ⚠️ {unprotected_count} trades with NO PROTECTION ---")
    else:
        print(f"--- All {len(results)} trades have some form of protection ---")

    # 即実行コマンドを出力
    if all_put_commands:
        print(f"\n=== 修正コマンド ({len(all_put_commands)}件) — コピペで即実行 ===")
        for cmd in all_put_commands:
            print()
            print(cmd)


if __name__ == "__main__":
    main()

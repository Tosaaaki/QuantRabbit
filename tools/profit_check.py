#!/usr/bin/env python3
"""
Profit Check — 保有ポジションの「今、利確すべきか」をデータで問う

pretrade_checkのTP版。判断はトレーダーがする。道具は事実を出すだけ。

Usage:
    python3 tools/profit_check.py              # 含み益ポジ全件チェック
    python3 tools/profit_check.py --all        # 含み損も含めて全件チェック
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
    """state.mdからピーク記録を抽出"""
    if not STATE_PATH.exists():
        return None
    content = STATE_PATH.read_text()
    # ピーク: +3,200円 @1.33620 のパターンを探す
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
                m2 = re.search(r'@([\d.]+)', line)
                if m2:
                    peak_info["peak_price"] = float(m2.group(1))
    return peak_info if peak_info else None


def classify_momentum(d: dict) -> str:
    """M5モメンタムの方向と強さを判定"""
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
        signals.append(f"StRSI={stoch_rsi:.2f}(過熱)")
    elif stoch_rsi < 0.2:
        signals.append(f"StRSI={stoch_rsi:.2f}(売られ)")
    else:
        signals.append(f"StRSI={stoch_rsi:.2f}")

    # EMA slope
    if abs(ema_slope) < 0.0001:
        signals.append("slope横ばい")
    elif ema_slope > 0:
        signals.append("slope↑")
    else:
        signals.append("slope↓")

    return " ".join(signals)


def check_cross_pair_correlation(pair: str, side: str, all_technicals: dict) -> list[str]:
    """7ペア相関チェック"""
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
                notes.append(f"{other_pair}も{base}買い方向(slope={other_slope:.4f})")
            elif side == "SHORT" and other_slope < -0.0002:
                notes.append(f"{other_pair}も{base}売り方向(slope={other_slope:.4f})")
            elif side == "LONG" and other_slope < -0.0002:
                notes.append(f"⚠ {other_pair}は{base}売り方向(slope={other_slope:.4f}) = 逆行")

        # Same quote currency
        if other_quote == quote:
            if side == "LONG" and other_slope > 0.0002:
                notes.append(f"{other_pair}も{quote}売り方向 = 相関一致")
            elif side == "SHORT" and other_slope < -0.0002:
                notes.append(f"{other_pair}も{quote}買い方向 = 相関一致")

    return notes[:4]  # Max 4 notes


def check_sr_distance(d: dict) -> str:
    """S/R距離"""
    parts = []
    cluster_high = d.get("cluster_high_gap", None)
    cluster_low = d.get("cluster_low_gap", None)
    swing_high = d.get("swing_dist_high", None)
    swing_low = d.get("swing_dist_low", None)

    if cluster_high is not None and 0 < cluster_high < 10:
        parts.append(f"レジ{cluster_high:.1f}pip近")
    if cluster_low is not None and 0 < cluster_low < 10:
        parts.append(f"サポ{cluster_low:.1f}pip近")
    if swing_high is not None:
        parts.append(f"swing高{swing_high:.1f}pip")
    if swing_low is not None:
        parts.append(f"swing安{swing_low:.1f}pip")

    return " | ".join(parts) if parts else "S/Rデータなし"


def assess_position(trade: dict, all_technicals: dict) -> dict:
    """1ポジションの利確判定"""
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
        result["signals"].append(f"ATR比 {atr_ratio:.1f}x ← 十分な利幅")
        take_signals += 2
    elif atr_ratio >= 1.0:
        result["signals"].append(f"ATR比 {atr_ratio:.1f}x ← トリガー到達")
        take_signals += 1
    elif atr_ratio >= 0.5:
        result["signals"].append(f"ATR比 {atr_ratio:.1f}x ← まだ伸びる余地")
    else:
        result["signals"].append(f"ATR比 {atr_ratio:.1f}x ← ATR未達")
        hold_signals += 1

    # 2. M5 Momentum
    if m5:
        momentum = classify_momentum(m5)
        result["signals"].append(f"M5モメンタム: {momentum}")

        macd_hist = m5.get("macd_hist", 0)
        stoch_rsi = m5.get("stoch_rsi", 0.5)

        # Momentum fading for longs
        if side == "LONG":
            if macd_hist < 0 and stoch_rsi > 0.8:
                result["signals"].append("⚠ M5 MACD反転+過熱 = モメンタム減衰")
                take_signals += 2
            elif stoch_rsi > 0.9:
                result["signals"].append("⚠ M5 StochRSI>0.9 = 過熱圏")
                take_signals += 1
        else:  # SHORT
            if macd_hist > 0 and stoch_rsi < 0.2:
                result["signals"].append("⚠ M5 MACD反転+売られすぎ = モメンタム減衰")
                take_signals += 2
            elif stoch_rsi < 0.1:
                result["signals"].append("⚠ M5 StochRSI<0.1 = 売られすぎ")
                take_signals += 1

    # 3. H1 structure
    if h1:
        adx = h1.get("adx", 0)
        plus_di = h1.get("plus_di", 0)
        minus_di = h1.get("minus_di", 0)
        h1_trend = "DI+優位" if plus_di > minus_di else "DI-優位"

        # Is H1 aligned with position?
        h1_aligned = (side == "LONG" and plus_di > minus_di) or \
                     (side == "SHORT" and minus_di > plus_di)

        if h1_aligned and adx > 25:
            result["signals"].append(f"H1 ADX={adx:.0f} {h1_trend} テーゼ健在")
            hold_signals += 2
        elif h1_aligned:
            result["signals"].append(f"H1 ADX={adx:.0f} {h1_trend} テーゼ方向一致(弱)")
            hold_signals += 1
        else:
            result["signals"].append(f"⚠ H1 ADX={adx:.0f} {h1_trend} テーゼ逆行")
            take_signals += 2

        # H1 divergence against position
        div_score = h1.get("div_rsi_score", 0) + h1.get("div_macd_score", 0)
        if div_score > 0:
            result["signals"].append(f"⚠ H1 ダイバージェンス検出(score={div_score:.1f})")
            take_signals += 1

        # ADX < 20 = range → BB mid is natural TP
        if adx < 20:
            bb_mid = h1.get("bb_mid", 0)
            if bb_mid:
                bb_dist = (bb_mid - current_price) * pip_mult if side == "LONG" else (current_price - bb_mid) * pip_mult
                if 0 < bb_dist < 5:
                    result["signals"].append(f"⚠ H1レンジ(ADX<20) + BB mid残{bb_dist:.1f}pip = 自然なTP")
                    take_signals += 2

    # 4. Cross-pair correlation
    corr_notes = check_cross_pair_correlation(pair, side, all_technicals)
    if corr_notes:
        for note in corr_notes:
            result["signals"].append(note)
            if "逆行" in note:
                take_signals += 1
            elif "一致" in note or "方向" in note:
                hold_signals += 1

    # 5. S/R distance
    if h1:
        sr = check_sr_distance(h1)
        if "近" in sr:
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
                result["signals"].append(f"⚠ ピーク{peak_yen:+,.0f}円から{drawdown:,.0f}円戻し({drawdown/peak_yen*100:.0f}%)")
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
        result["recommendation"] = "HOLD(ATR未達)"
    else:
        result["recommendation"] = "REVIEW"

    result["take_signals"] = take_signals
    result["hold_signals"] = hold_signals

    return result


def format_result(r: dict) -> str:
    """1ポジションの結果をフォーマット"""
    icon = {
        "TAKE_PROFIT": "🔴",
        "HALF_TP": "🟡",
        "HOLD": "🟢",
        "HOLD(ATR未達)": "🟢",
        "REVIEW": "🟡",
        "LOSS_POSITION": "⚪",
    }.get(r["recommendation"], "⚪")

    lines = []
    lines.append(
        f"{icon} {r['pair']} {r['side']} {r['units']}u | "
        f"UPL: {r['upl']:+,.0f}円 ({r['pip_distance']:+.1f}pip) | "
        f"ATR比: {r['atr_ratio']:.1f}x (ATR={r['atr_pips']:.1f}pip) | "
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
        print("=== PROFIT CHECK: ポジションなし ===")
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
    priority = {"TAKE_PROFIT": 0, "HALF_TP": 1, "REVIEW": 2, "HOLD": 3, "HOLD(ATR未達)": 4, "LOSS_POSITION": 5}
    results.sort(key=lambda r: priority.get(r["recommendation"], 3))

    for r in results:
        print(format_result(r))
        print()

    # Summary
    tp_count = sum(1 for r in results if r["recommendation"] in ("TAKE_PROFIT", "HALF_TP"))
    if tp_count > 0:
        print(f"--- {tp_count}件の利確候補あり。デフォルトは利確。持つなら根拠を言え。 ---")
    else:
        print(f"--- 利確候補なし。全ポジHOLD。 ---")


if __name__ == "__main__":
    main()

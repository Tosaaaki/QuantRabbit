#!/usr/bin/env python3
"""
Macro View — Synthesizes currency strength and macro view from technicals across 7 pairs

Reads technicals_*.json and produces:
1. Per-currency strength score (calculated from H1 ADX × DI direction)
2. Overall theme (USD strong/weak, risk-on/off, etc.)
3. MTF alignment (pairs where H1 and M5 agree = high-confidence opportunity)

Output is 3-5 lines so the trader can grasp the macro picture at a glance.
"""
from __future__ import annotations

import json
from pathlib import Path
from technicals_json import load_technicals_timeframes

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]

# Define direction for each currency pair
# base/quote: base strong = price rises, quote strong = price falls
PAIR_CURRENCIES = {
    "USD_JPY": ("USD", "JPY"),
    "EUR_USD": ("EUR", "USD"),
    "GBP_USD": ("GBP", "USD"),
    "AUD_USD": ("AUD", "USD"),
    "EUR_JPY": ("EUR", "JPY"),
    "GBP_JPY": ("GBP", "JPY"),
    "AUD_JPY": ("AUD", "JPY"),
}


def load_technicals(pair: str) -> dict:
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return load_technicals_timeframes(f)


def calc_currency_strength() -> dict[str, float]:
    """Calculate per-currency score from H1 ADX × DI direction"""
    scores: dict[str, list[float]] = {c: [] for c in ["USD", "EUR", "GBP", "AUD", "JPY"]}

    for pair in PAIRS:
        tfs = load_technicals(pair)
        h1 = tfs.get("H1", {})
        if not h1:
            continue

        adx = h1.get("adx", 0)
        di_plus = h1.get("plus_di", 0)
        di_minus = h1.get("minus_di", 0)
        base, quote = PAIR_CURRENCIES[pair]

        # Weight DI difference by ADX: stronger trend = more influence on score
        direction = (di_plus - di_minus) / max(di_plus + di_minus, 1)  # -1 to 1
        weight = min(adx / 30, 1.5)  # ADX 30 = weight 1.0, ADX 45 = weight 1.5
        signal = direction * weight

        scores[base].append(signal)
        scores[quote].append(-signal)  # quote currency is inverted

    # Average
    result = {}
    for ccy, vals in scores.items():
        result[ccy] = sum(vals) / len(vals) if vals else 0
    return result


def find_mtf_aligned(pair: str, tfs: dict) -> str | None:
    """Determine whether H1 and M5 direction are aligned"""
    h1 = tfs.get("H1", {})
    m5 = tfs.get("M5", {})
    if not h1 or not m5:
        return None

    h1_bull = h1.get("plus_di", 0) > h1.get("minus_di", 0)
    m5_bull = m5.get("plus_di", 0) > m5.get("minus_di", 0)
    h1_adx = h1.get("adx", 0)
    m5_adx = m5.get("adx", 0)

    if h1_bull == m5_bull and h1_adx > 20 and m5_adx > 20:
        direction = "BULL" if h1_bull else "BEAR"
        return f"{direction}(H1 ADX={h1_adx:.0f} M5 ADX={m5_adx:.0f})"
    return None


def find_divergences(pair: str, tfs: dict) -> list[str]:
    """Detect divergences on H1"""
    alerts = []
    h1 = tfs.get("H1", {})
    if not h1:
        return alerts

    div_rsi = h1.get("div_rsi_score", 0)
    div_macd = h1.get("div_macd_score", 0)
    div_rsi_kind = h1.get("div_rsi_kind", 0)
    div_macd_kind = h1.get("div_macd_kind", 0)

    kind_names = {1: "RegBull", -1: "RegBear", 2: "HidBull", -2: "HidBear"}

    if abs(div_rsi) >= 0.5:
        alerts.append(f"RSI {kind_names.get(div_rsi_kind, '?')}={div_rsi:.1f}")
    if abs(div_macd) >= 0.5:
        alerts.append(f"MACD {kind_names.get(div_macd_kind, '?')}={div_macd:.1f}")

    return alerts


def main():
    # 1. Currency strength
    strength = calc_currency_strength()
    sorted_ccy = sorted(strength.items(), key=lambda x: x[1], reverse=True)

    strongest = sorted_ccy[0]
    weakest = sorted_ccy[-1]

    strength_line = " > ".join(
        f"{ccy}({score:+.2f})" for ccy, score in sorted_ccy
    )
    print(f"💪 Currency strength: {strength_line}")

    # 2. Theme determination
    usd_score = strength.get("USD", 0)
    jpy_score = strength.get("JPY", 0)

    themes = []
    if usd_score > 0.3:
        themes.append("USD strong")
    elif usd_score < -0.3:
        themes.append("USD weak")
    if jpy_score > 0.3:
        themes.append("JPY strong (risk-off)")
    elif jpy_score < -0.3:
        themes.append("JPY weak (risk-on)")

    if not themes:
        themes.append("no clear direction")

    print(f"🎯 Theme: {' + '.join(themes)} | Strongest={strongest[0]} Weakest={weakest[0]}")

    # 3. MTF-aligned pairs (high-confidence opportunities)
    aligned = []
    for pair in PAIRS:
        tfs = load_technicals(pair)
        result = find_mtf_aligned(pair, tfs)
        if result:
            aligned.append(f"{pair} {result}")

    if aligned:
        print(f"✅ MTF aligned: {' | '.join(aligned)}")
    else:
        print("⚠️ No MTF-aligned pairs — wait for a high-confidence opportunity")

    # 4. H1 divergences (reversal signals)
    all_divs = []
    for pair in PAIRS:
        tfs = load_technicals(pair)
        divs = find_divergences(pair, tfs)
        if divs:
            all_divs.append(f"{pair}: {', '.join(divs)}")

    if all_divs:
        print(f"🔄 H1 Divergence: {' | '.join(all_divs)}")


if __name__ == "__main__":
    main()

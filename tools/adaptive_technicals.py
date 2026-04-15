#!/usr/bin/env python3
"""
Adaptive Technicals — Automatically selects and displays indicators suited to the current situation

Not a fixed set of 9. Draws from an arsenal of 84, picking what works for the situation now.

Usage:
    python3 tools/adaptive_technicals.py          # all pairs
    python3 tools/adaptive_technicals.py USD_JPY  # single pair detail
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]


def load_technicals(pair: str) -> dict:
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("timeframes", {})


def classify_situation(tf_data: dict) -> list[str]:
    """Classify the situation for this timeframe from ADX/BBW/RSI etc. Multiple situations can overlap."""
    situations = []
    adx = tf_data.get("adx", 0)
    bbw = tf_data.get("bbw", 999)
    rsi = tf_data.get("rsi", 50)
    stoch_rsi = tf_data.get("stoch_rsi", 0.5)
    cci = tf_data.get("cci", 0)
    div_rsi = tf_data.get("div_rsi_score", 0)
    div_macd = tf_data.get("div_macd_score", 0)

    # Strong trend
    if adx > 30:
        situations.append("strong_trend")
    # Trend fatigue
    if adx > 20 and (div_rsi > 0 or div_macd > 0):
        situations.append("trend_fatigue")
    # Range / consolidation
    if adx < 20:
        situations.append("range")
    # Squeeze (low BBW)
    if bbw < 0.002:
        situations.append("squeeze")
    # Reversal candidate
    if stoch_rsi <= 0.05 or stoch_rsi >= 0.95 or abs(cci) > 200:
        situations.append("reversal")
    # Overbought/oversold
    if rsi > 70 or rsi < 30:
        situations.append("extreme_rsi")

    return situations if situations else ["neutral"]


def format_base(d: dict) -> str:
    """Base indicators (always shown)."""
    return (
        f"RSI={d.get('rsi', 0):.0f} "
        f"ADX={d.get('adx', 0):.0f} "
        f"DI+={d.get('plus_di', 0):.0f}/DI-={d.get('minus_di', 0):.0f} "
        f"StRSI={d.get('stoch_rsi', 0):.2f}"
    )


def momentum_quality_tag(d: dict) -> str:
    """Classify trend momentum as FRESH/MATURE/EXHAUSTING/REVERSING."""
    s5 = d.get("ema_slope_5", 0)
    s10 = d.get("ema_slope_10", 0)
    hist = d.get("macd_hist", 0)
    macd = d.get("macd", 0)
    roc5 = d.get("roc5", 0)
    roc10 = d.get("roc10", 0)
    di_p = d.get("plus_di", 0)
    di_m = d.get("minus_di", 0)
    bull = di_p > di_m

    # Reversing: MACD hist sign opposite to trend direction
    if bull and hist < 0:
        return "REVERSING"
    if not bull and hist > 0:
        return "REVERSING"

    # Compare absolute slopes (direction-independent)
    abs_s5 = abs(s5)
    abs_s10 = abs(s10)

    if abs_s10 < 1e-8:
        return "MATURE"

    ratio = abs_s5 / abs_s10 if abs_s10 > 0 else 1.0

    if ratio > 1.15:
        return "FRESH"
    elif ratio < 0.75:
        return "EXHAUSTING"
    else:
        return "MATURE"


def format_adaptive(d: dict, situations: list[str]) -> str:
    """Additional indicators selected for the current situation."""
    parts = []

    if "strong_trend" in situations:
        mtag = momentum_quality_tag(d)
        parts.append(
            f"[TREND {mtag}] EMAslope5={d.get('ema_slope_5', 0):.4f} "
            f"slope10={d.get('ema_slope_10', 0):.4f} "
            f"MACD_H={d.get('macd_hist', 0):.5f} "
            f"ROC5={d.get('roc5', 0):.3f} ROC10={d.get('roc10', 0):.3f}"
        )

    if "trend_fatigue" in situations:
        div_parts = []
        if d.get("div_rsi_score", 0) > 0:
            kind_map = {1: "bull", 2: "bear", 3: "hid_bull", 4: "hid_bear"}
            div_parts.append(f"RSI_div={d['div_rsi_score']:.1f}({kind_map.get(d.get('div_rsi_kind', 0), '?')})")
        if d.get("div_macd_score", 0) > 0:
            kind_map = {1: "bull", 2: "bear", 3: "hid_bull", 4: "hid_bear"}
            div_parts.append(f"MACD_div={d['div_macd_score']:.1f}({kind_map.get(d.get('div_macd_kind', 0), '?')})")
        parts.append(f"[FATIGUE] {' '.join(div_parts)}")

    if "range" in situations or "squeeze" in situations:
        parts.append(
            f"[RANGE] BB={d.get('bb_lower', 0):.3f}/{d.get('bb_mid', 0):.3f}/{d.get('bb_upper', 0):.3f} "
            f"BBW={d.get('bbw', 0):.5f} "
            f"CCI={d.get('cci', 0):.0f} "
            f"VWAP_gap={d.get('vwap_gap', 0):.1f}pip"
        )

    if "squeeze" in situations:
        parts.append(f"[SQUEEZE] BBW={d.get('bbw', 0):.5f} <- breakout alert")

    if "reversal" in situations:
        parts.append(
            f"[REVERSAL] CCI={d.get('cci', 0):.0f} "
            f"StRSI={d.get('stoch_rsi', 0):.2f} "
            f"ATR={d.get('atr_pips', 0):.1f}pip"
        )

    if "extreme_rsi" in situations:
        parts.append(
            f"[EXTREME] RSI={d.get('rsi', 0):.0f} "
            f"VWAP_gap={d.get('vwap_gap', 0):.1f}pip "
            f"BB_pos={'above BB' if d.get('close', 0) > d.get('bb_upper', 999) else 'below BB' if d.get('close', 0) < d.get('bb_lower', 0) else 'inside BB'}"
        )

    # Ichimoku (position relative to cloud is always useful)
    ichi = d.get("ichimoku_cloud_pos", None)
    if ichi is not None and ichi != 0:
        parts.append(f"Ichi_cloud={ichi:.0f}pip {'above' if ichi > 0 else 'below'}")

    return " | ".join(parts) if parts else ""


def format_alerts(pair: str, all_tfs: dict) -> list[str]:
    """Cross-TF alerts (for detecting inter-pair relationships)."""
    alerts = []
    for tf_name, d in all_tfs.items():
        sr = d.get("stoch_rsi", 0.5)
        if sr <= 0.01:
            alerts.append(f"⚡ {tf_name} StochRSI=0.0 (extreme oversold)")
        elif sr >= 0.99:
            alerts.append(f"⚡ {tf_name} StochRSI=1.0 (extreme overbought)")
        if abs(d.get("cci", 0)) > 200:
            alerts.append(f"⚡ {tf_name} CCI={d.get('cci', 0):.0f} (extreme)")
        if d.get("div_rsi_score", 0) > 0 or d.get("div_macd_score", 0) > 0:
            alerts.append(f"⚡ {tf_name} divergence detected")
        if d.get("bbw", 999) < 0.001:
            alerts.append(f"⚡ {tf_name} extreme BB squeeze BBW={d.get('bbw', 0):.5f}")
    return alerts


def main():
    target_pairs = sys.argv[1:] if len(sys.argv) > 1 else PAIRS
    detail_mode = len(target_pairs) == 1

    for pair in target_pairs:
        tfs = load_technicals(pair)
        if not tfs:
            print(f"{pair}: NO DATA")
            continue

        print(f"\n{'='*60}")
        print(f"  {pair}")
        print(f"{'='*60}")

        # Alerts (highest priority — shown first)
        alerts = format_alerts(pair, tfs)
        if alerts:
            for a in alerts:
                print(f"  {a}")
            print()

        # Per-TF display
        for tf_name in ["H4", "H1", "M15", "M5", "M1"]:
            d = tfs.get(tf_name)
            if not d:
                continue

            situations = classify_situation(d)
            sit_str = ",".join(situations)

            print(f"  {tf_name} [{sit_str}]")
            print(f"    Base: {format_base(d)}")

            adaptive = format_adaptive(d, situations)
            if adaptive:
                print(f"    +Add: {adaptive}")

            if detail_mode:
                # Detail mode: show all indicators
                print(f"    Full: close={d.get('close', 0)} ATR={d.get('atr_pips', 0):.1f}pip "
                      f"MACD={d.get('macd', 0):.5f} MACD_H={d.get('macd_hist', 0):.5f} "
                      f"CCI={d.get('cci', 0):.0f} ROC5={d.get('roc5', 0):.3f} "
                      f"EMAslope5/10/20={d.get('ema_slope_5', 0):.4f}/{d.get('ema_slope_10', 0):.4f}/{d.get('ema_slope_20', 0):.4f} "
                      f"VWAP_gap={d.get('vwap_gap', 0):.1f}pip "
                      f"BB={d.get('bb_lower', 0):.3f}-{d.get('bb_upper', 0):.3f} BBW={d.get('bbw', 0):.5f}")


if __name__ == "__main__":
    main()

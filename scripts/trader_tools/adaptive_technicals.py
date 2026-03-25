#!/usr/bin/env python3
"""
Adaptive Technicals — 状況に応じた指標を自動選択して表示

固定9個じゃない。84個の武器庫から、今の状況に効くものを出す。

Usage:
    python3 scripts/trader_tools/adaptive_technicals.py          # 全ペア
    python3 scripts/trader_tools/adaptive_technicals.py USD_JPY  # 1ペア詳細
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]


def load_technicals(pair: str) -> dict:
    f = ROOT / f"logs/technicals_{pair}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text()).get("timeframes", {})


def classify_situation(tf_data: dict) -> list[str]:
    """ADX/BBW/RSI等からその時間足の状況を分類。複数の状況が重複しうる。"""
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
    """ベース指標（必ず表示）"""
    return (
        f"RSI={d.get('rsi', 0):.0f} "
        f"ADX={d.get('adx', 0):.0f} "
        f"DI+={d.get('plus_di', 0):.0f}/DI-={d.get('minus_di', 0):.0f} "
        f"StRSI={d.get('stoch_rsi', 0):.2f}"
    )


def format_adaptive(d: dict, situations: list[str]) -> str:
    """状況に応じた追加指標"""
    parts = []

    if "strong_trend" in situations:
        parts.append(
            f"[TREND] EMAslope5={d.get('ema_slope_5', 0):.4f} "
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
        parts.append(f"[SQUEEZE] BBW={d.get('bbw', 0):.5f} ← ブレイク警戒")

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
            f"BB_pos={'上抜' if d.get('close', 0) > d.get('bb_upper', 999) else '下抜' if d.get('close', 0) < d.get('bb_lower', 0) else 'BB内'}"
        )

    # Ichimoku（雲との位置関係は常に有用）
    ichi = d.get("ichimoku_cloud_pos", None)
    if ichi is not None and ichi != 0:
        parts.append(f"Ichi雲={ichi:.0f}pip{'上' if ichi > 0 else '下'}")

    return " | ".join(parts) if parts else ""


def format_alerts(pair: str, all_tfs: dict) -> list[str]:
    """TF横断のアラート（ペア間関係性を見るため）"""
    alerts = []
    for tf_name, d in all_tfs.items():
        sr = d.get("stoch_rsi", 0.5)
        if sr <= 0.01:
            alerts.append(f"⚡ {tf_name} StochRSI=0.0 (売られすぎ極限)")
        elif sr >= 0.99:
            alerts.append(f"⚡ {tf_name} StochRSI=1.0 (買われすぎ極限)")
        if abs(d.get("cci", 0)) > 200:
            alerts.append(f"⚡ {tf_name} CCI={d.get('cci', 0):.0f} (極端)")
        if d.get("div_rsi_score", 0) > 0 or d.get("div_macd_score", 0) > 0:
            alerts.append(f"⚡ {tf_name} ダイバージェンス検出")
        if d.get("bbw", 999) < 0.001:
            alerts.append(f"⚡ {tf_name} BB超squeeze BBW={d.get('bbw', 0):.5f}")
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

        # アラート（最優先で表示）
        alerts = format_alerts(pair, tfs)
        if alerts:
            for a in alerts:
                print(f"  {a}")
            print()

        # TF別表示
        for tf_name in ["H4", "H1", "M5", "M1"]:
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
                # 詳細モード: 全指標表示
                print(f"    Full: close={d.get('close', 0)} ATR={d.get('atr_pips', 0):.1f}pip "
                      f"MACD={d.get('macd', 0):.5f} MACD_H={d.get('macd_hist', 0):.5f} "
                      f"CCI={d.get('cci', 0):.0f} ROC5={d.get('roc5', 0):.3f} "
                      f"EMAslope5/10/20={d.get('ema_slope_5', 0):.4f}/{d.get('ema_slope_10', 0):.4f}/{d.get('ema_slope_20', 0):.4f} "
                      f"VWAP_gap={d.get('vwap_gap', 0):.1f}pip "
                      f"BB={d.get('bb_lower', 0):.3f}-{d.get('bb_upper', 0):.3f} BBW={d.get('bbw', 0):.5f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Macro View — 7ペアのテクニカルから通貨強弱とマクロビューを合成

technicals_*.json を読んで:
1. 通貨別の強弱スコア（H1 ADX×DI方向で計算）
2. 全体のテーマ（USD強/弱、リスクオン/オフ等）
3. MTFの整合性（H1とM5が一致してるペア = 高確度チャンス）

出力は3-5行。traderが一目でマクロを掴めるように。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ["USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "EUR_JPY", "GBP_JPY", "AUD_JPY"]

# ペアから通貨ペアの方向を定義
# base/quote: base強 = 価格上昇, quote強 = 価格下降
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
    return json.loads(f.read_text()).get("timeframes", {})


def calc_currency_strength() -> dict[str, float]:
    """H1のADX×DI方向から通貨別スコアを算出"""
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

        # DI差をADXで重み付け: トレンドが強いほどスコアに影響
        direction = (di_plus - di_minus) / max(di_plus + di_minus, 1)  # -1 to 1
        weight = min(adx / 30, 1.5)  # ADX30で重み1.0、45で1.5
        signal = direction * weight

        scores[base].append(signal)
        scores[quote].append(-signal)  # quote通貨は逆

    # 平均化
    result = {}
    for ccy, vals in scores.items():
        result[ccy] = sum(vals) / len(vals) if vals else 0
    return result


def find_mtf_aligned(pair: str, tfs: dict) -> str | None:
    """H1とM5の方向が一致してるか判定"""
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
    """H1のダイバージェンスを検出"""
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
    # 1. 通貨強弱
    strength = calc_currency_strength()
    sorted_ccy = sorted(strength.items(), key=lambda x: x[1], reverse=True)

    strongest = sorted_ccy[0]
    weakest = sorted_ccy[-1]

    strength_line = " > ".join(
        f"{ccy}({score:+.2f})" for ccy, score in sorted_ccy
    )
    print(f"💪 通貨強弱: {strength_line}")

    # 2. テーマ判定
    usd_score = strength.get("USD", 0)
    jpy_score = strength.get("JPY", 0)

    themes = []
    if usd_score > 0.3:
        themes.append("USD強")
    elif usd_score < -0.3:
        themes.append("USD弱")
    if jpy_score > 0.3:
        themes.append("JPY強(リスクオフ)")
    elif jpy_score < -0.3:
        themes.append("JPY弱(リスクオン)")

    if not themes:
        themes.append("方向感薄い")

    print(f"🎯 テーマ: {' + '.join(themes)} | 最強={strongest[0]} 最弱={weakest[0]}")

    # 3. MTF一致ペア（高確度チャンス）
    aligned = []
    for pair in PAIRS:
        tfs = load_technicals(pair)
        result = find_mtf_aligned(pair, tfs)
        if result:
            aligned.append(f"{pair} {result}")

    if aligned:
        print(f"✅ MTF一致: {' | '.join(aligned)}")
    else:
        print("⚠️ MTF一致ペアなし — 高確度チャンスを待て")

    # 4. H1ダイバージェンス（転換予兆）
    all_divs = []
    for pair in PAIRS:
        tfs = load_technicals(pair)
        divs = find_divergences(pair, tfs)
        if divs:
            all_divs.append(f"{pair}: {', '.join(divs)}")

    if all_divs:
        print(f"🔄 H1 Div: {' | '.join(all_divs)}")


if __name__ == "__main__":
    main()

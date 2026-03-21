"""
共同トレード用テクニカル計算ヘルパー

使い方:
  python3 collab_trade/indicators/quick_calc.py USD_JPY M5 50
  python3 collab_trade/indicators/quick_calc.py EUR_USD H1 20
  python3 collab_trade/indicators/quick_calc.py AUD_USD M1 100

パラメータをいじりたい時はこのファイルを直接編集。
本体(indicators/)には影響しない。
"""

import sys, os, json, urllib.request
import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# collab_trade版のインジケーターを使う
from indicators.calc_core import IndicatorEngine

# ============================================================
# パラメータ（自由にいじれ）
# ============================================================
# IndicatorEngine内のデフォルトを上書きしたい場合は
# calc_core.py を直接編集する。ここでは取得パラメータのみ。

DEFAULT_GRANULARITY = "M5"
DEFAULT_COUNT = 50

# ============================================================
# OANDA API
# ============================================================
def load_config():
    env_path = os.path.join(ROOT, "config", "env.toml")
    lines = open(env_path).read().split("\n")
    cfg = {}
    for line in lines:
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"')
    return cfg

def fetch_candles(pair, granularity, count):
    cfg = load_config()
    base = "https://api-fxtrade.oanda.com"
    url = f"{base}/v3/instruments/{pair}/candles?granularity={granularity}&count={count}&price=M"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {cfg['oanda_token']}"})
    data = json.loads(urllib.request.urlopen(req).read())

    rows = []
    for c in data["candles"]:
        if c["complete"] or True:  # include current candle
            m = c["mid"]
            rows.append({
                "time": c["time"],
                "open": float(m["o"]),
                "high": float(m["h"]),
                "low": float(m["l"]),
                "close": float(m["c"]),
                "volume": int(c["volume"]),
            })
    return pd.DataFrame(rows)

def fetch_price(pair):
    cfg = load_config()
    base = "https://api-fxtrade.oanda.com"
    url = f"{base}/v3/accounts/{cfg['oanda_account_id']}/pricing?instruments={pair}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {cfg['oanda_token']}"})
    data = json.loads(urllib.request.urlopen(req).read())
    p = data["prices"][0]
    return float(p["bids"][0]["price"]), float(p["asks"][0]["price"])

# ============================================================
# メイン
# ============================================================
def analyze(pair, granularity=None, count=None):
    granularity = granularity or DEFAULT_GRANULARITY
    count = int(count or DEFAULT_COUNT)

    df = fetch_candles(pair, granularity, count)
    indicators = IndicatorEngine.compute(df)

    # 現在価格
    bid, ask = fetch_price(pair)
    spread = (ask - bid) * (100 if "JPY" in pair else 10000)

    print(f"\n{'='*60}")
    print(f"  {pair} | {granularity} x {count}本 | Bid:{bid:.5g} Ask:{ask:.5g} Sp:{spread:.1f}pip")
    print(f"{'='*60}")

    # トレンド・モメンタム
    print(f"\n--- トレンド ---")
    print(f"  ADX: {indicators.get('adx', 0):.1f}  DI+: {indicators.get('plus_di', 0):.1f}  DI-: {indicators.get('minus_di', 0):.1f}")
    print(f"  EMA12: {indicators.get('ema12', 0):.5g}  EMA20: {indicators.get('ema20', 0):.5g}")
    print(f"  EMAスロープ: 5={indicators.get('ema_slope_5', 0):.6f}  10={indicators.get('ema_slope_10', 0):.6f}  20={indicators.get('ema_slope_20', 0):.6f}")
    print(f"  MACD: {indicators.get('macd', 0):.6f}  Signal: {indicators.get('macd_signal', 0):.6f}  Hist: {indicators.get('macd_hist', 0):.6f}")

    # オシレーター
    print(f"\n--- オシレーター ---")
    print(f"  RSI: {indicators.get('rsi', 0):.1f}  StochRSI: {indicators.get('stoch_rsi', 0):.1f}  CCI: {indicators.get('cci', 0):.1f}")

    # ボラティリティ
    print(f"\n--- ボラティリティ ---")
    print(f"  ATR: {indicators.get('atr_pips', 0):.1f}pip")
    print(f"  BB: Upper={indicators.get('bb_upper', 0):.5g}  Mid={indicators.get('bb_mid', 0):.5g}  Lower={indicators.get('bb_lower', 0):.5g}")
    print(f"  BB幅: {indicators.get('bbw', 0):.4f}  BB span: {indicators.get('bb_span_pips', 0):.1f}pip")

    # 価格構造
    print(f"\n--- 価格構造 ---")
    print(f"  VWAP乖離: {indicators.get('vwap_gap', 0):.1f}pip")
    print(f"  Ichimoku雲: SpanA={indicators.get('ichimoku_span_a_gap', 0):.1f}pip  SpanB={indicators.get('ichimoku_span_b_gap', 0):.1f}pip  位置={indicators.get('ichimoku_cloud_pos', 0):.1f}pip")
    print(f"  Swing距離: High={indicators.get('swing_dist_high', 0):.1f}pip  Low={indicators.get('swing_dist_low', 0):.1f}pip")

    # ダイバージェンス
    print(f"\n--- ダイバージェンス ---")
    div_rsi_kind = indicators.get('div_rsi_kind', 0)
    div_macd_kind = indicators.get('div_macd_kind', 0)
    kind_label = {1: "Regular Bullish", 2: "Hidden Bullish", -1: "Regular Bearish", -2: "Hidden Bearish", 0: "なし"}
    print(f"  RSI: {kind_label.get(int(div_rsi_kind), 'なし')}  Score: {indicators.get('div_rsi_score', 0):.2f}  Age: {indicators.get('div_rsi_age', 0):.0f}本")
    print(f"  MACD: {kind_label.get(int(div_macd_kind), 'なし')}  Score: {indicators.get('div_macd_score', 0):.2f}  Age: {indicators.get('div_macd_age', 0):.0f}本")
    print(f"  統合: {indicators.get('div_score', 0):.2f}")

    # 判断サマリー
    print(f"\n--- 判断材料 ---")
    rsi = indicators.get('rsi', 50)
    adx = indicators.get('adx', 0)
    di_plus = indicators.get('plus_di', 0)
    di_minus = indicators.get('minus_di', 0)

    if adx > 25:
        if di_plus > di_minus:
            print(f"  📈 トレンド: 上昇 (ADX={adx:.0f}, DI+={di_plus:.0f} > DI-={di_minus:.0f})")
        else:
            print(f"  📉 トレンド: 下降 (ADX={adx:.0f}, DI-={di_minus:.0f} > DI+={di_plus:.0f})")
    else:
        print(f"  ↔️ レンジ (ADX={adx:.0f})")

    if rsi > 70:
        print(f"  ⚠️ RSI過熱（買われすぎ）: {rsi:.0f}")
    elif rsi < 30:
        print(f"  ⚠️ RSI過冷（売られすぎ）: {rsi:.0f}")

    print()
    return indicators

if __name__ == "__main__":
    pair = sys.argv[1] if len(sys.argv) > 1 else "USD_JPY"
    gran = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_GRANULARITY
    cnt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_COUNT
    analyze(pair, gran, cnt)

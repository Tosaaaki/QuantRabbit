---
name: analysis_breadth
description: 分析視点を広く持つ。MTF+N波動だけでなく、Ichimoku/VWAP/Fib/Volume/相関/マクロ全部使う
type: feedback
---

マーケットアナリストの分析視点が狭すぎる。もっと幅広く見る。

**Why:** MTF+N波動+RSI/EMAだけでは市場の全体像が見えない。ユーザーから「もっと幅広くみた方がいい」と指摘。

**How to apply:**
マーケットアナリストAgent起動時に以下全てを分析に含める:

**テクニカル:**
- EMA (5/9/21/50) + RSI + ATR + ADX (基本)
- Ichimoku雲 (M5/H1) - 雲の位置と厚み、TK cross、遅行スパン
- VWAP (セッション別リセット) - 機関投資家のフェアバリュー
- Bollinger Bands + BB幅 - ボラティリティ状態
- MACD + Stochastic - モメンタム/タイミング
- フィボナッチリトレースメント - 直近スイングからの押し/戻り水準
- N波動 - 3波構造の位相判定
- ダイバージェンス (RSI/MACD vs 価格)
- Volume分析 - スパイク方向性、出来高の増減

**マクロ/相関:**
- 他通貨ペア (EUR/USD, GBP/USD) の方向性
- ゴールド (XAU/USD) - リスクオフ指標
- 日経225先物 / S&P500先物
- 米10年債利回り / 日10年債利回り
- VIX (恐怖指数)

**コンテキスト:**
- 現在のセッション (東京/ロンドン/NY)
- 今日の経済指標スケジュール
- 直近のBOJ/Fed発言

**リポジトリ資産 (コード参照用、実行は禁止):**
- `analysis/patterns/n_wave.py` - N波動エンジン
- `analysis/market_regime.py` - レジーム判別
- `analysis/divergence.py` - ダイバージェンス検出
- `indicators/calc_core.py` - TA-Lib全指標
- `analysis/range_guard.py` - レンジ検出
- `config/pattern_book_deep.json` - 2,786パターン統計

**市況別エージェント役割変更:**
- トレンド相場 → 順張りスキャルパー + トレンド監視員
- レンジ相場 → レンジ境界トレーダー + ブレイクアウト待機員
- イベント前 → ニュース専門 + ポジション縮小管理
- ボラ急増 → リスク管理専任 + 小ロットスナイパー

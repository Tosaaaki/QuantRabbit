---
name: mtf-dashboard
description: "全ペアのH4/H1/M15/M5テクニカル指標を一覧表示。MTFダッシュボード。"
trigger: "Use when the user says 'ダッシュボード', 'dashboard', 'MTF', 'テクニカル', 'インジケーター', or asks for a multi-timeframe overview."
---

# MTFダッシュボードスキル

全ペアのマルチタイムフレーム指標を一覧表示する。

## 実行手順

### Step 1: データ取得

1. OANDA API: 全ペア最新価格
2. shared_state.json: technicals_summary, regime
3. factor_cache（利用可能なら）: 詳細指標

### Step 2: 出力

```
## 📊 MTFダッシュボード

### 価格
| ペア | Bid | Ask | 日中変動 |
|------|-----|-----|---------|
| USD/JPY | 159.124 | 159.132 | -18.4pip |
...

### H1 指標
| ペア | Trend | RSI | ADX | StochRSI | MACD | VWAP乖離 |
|------|-------|-----|-----|----------|------|----------|
| USD/JPY | 🔴BEAR | 37.8 | 24.7 | 0.15 | -neg | -5pip |
| EUR/USD | ⚪RANGE | 48.3 | 22.5 | 0.65 | flat | +2pip |
...

### M5 指標
| ペア | Trend | RSI | ADX | StochRSI | 直近3本方向 |
|------|-------|-----|-----|----------|------------|
| USD/JPY | 🔴BEAR | 42.7 | 27.9 | 0.20 | ↓↓→ |
...

### レジーム判定
| ペア | H1 | M5 | 方向一致 |
|------|-----|-----|---------|
| USD/JPY | trending_bear | trending_bear | ✅一致 |
| EUR/USD | range | range_mild_bull | ⚠️不一致 |
...

### Direction Matrix
| ペア | Swing Bias | 注記 |
|------|-----------|------|
(shared_stateのdirection_matrixがあれば表示)
```

### アイコン凡例
- 🔴 = Bear/Short有利
- 🟢 = Bull/Long有利
- ⚪ = レンジ/方向不明
- ⚠️ = MTF不一致（エントリー注意）
- ✅ = MTF一致（セットアップ有利）

---
name: regime-check
description: "全ペアの市場レジーム（トレンド/レンジ/高ボラ）を判定・表示。"
trigger: "Use when the user says 'レジーム', 'regime', '市況', '環境認識', or asks about market conditions."
---

# レジームチェックスキル

## 実行手順

### Step 1: データ取得

1. shared_state.json: regime, technicals_summary
2. OANDA API: 最新価格

### Step 2: レジーム判定基準

| レジーム | ADX条件 | RSI条件 | 特徴 |
|----------|---------|---------|------|
| trending_bull | >25, +DI>>-DI | >50 | 明確な上昇トレンド |
| trending_bear | >25, -DI>>+DI | <50 | 明確な下降トレンド |
| range | <20 | 40-60 | 方向性なし |
| volatile | >30 | 極端値 | 高ボラ、方向不安定 |
| trending_oversold | >20 | <30 | トレンド中だがRSI極端 |
| breakout_pending | <15→急上昇 | - | スクイーズからの放れ |

### Step 3: 出力

```
## 🌐 市場レジーム

### マクロテーマ
[USD弱含み / リスクオン / JPY売り] etc.

### ペア別レジーム
| ペア | H4 | H1 | M5 | 推奨アクション |
|------|-----|-----|-----|--------------|
| USD/JPY | weak_bear | trending_bear | trending_bear | SHORT継続 |
| EUR/USD | bear | range | range_bull | 様子見(MTF不一致) |
| GBP/USD | bear | range_bear | range_bull | SKIP(BOEイベント) |
...

### レジーム変化検知
⚠️ EUR/USD: H1 trending_bear → range に変化（2時間前）
⚠️ AUD_JPY: M5 range → range_bull に変化（30分前）

### 推奨
- トレンドフォロー有効: USD/JPY
- レンジ戦略有効: EUR/USD
- 待機推奨: GBP/USD (イベント), EUR/JPY (oversold extreme)
```

---
name: volatility-scan
description: "ATR/BB幅で全ペアのボラティリティを比較。今一番動いてるペアを特定。"
trigger: "Use when the user says 'ボラ', 'volatility', 'ATR', 'BB', '動いてる', or asks which pair is most active."
---

# ボラティリティスキャンスキル

## 実行手順

### Step 1: データ取得

OANDA APIから各ペアのH1キャンドル直近20本を取得。

### Step 2: 計算

各ペアについて:
- **ATR(14)**: Average True Range
- **BB幅**: ボリンジャーバンド上限-下限 (20SMA ±2σ)
- **直近1H変動**: 直近1本の(High-Low)
- **日中変動**: 当日Open→現在の変動幅
- **ATRランク**: ATR/平均ATR比率（平時比でどれだけ動いてるか）

### Step 3: 出力

```
## 📊 ボラティリティスキャン

### H1 ATRランキング
| # | ペア | ATR(14) | ATR比率 | BB幅 | 直近1H | 日中変動 | 状態 |
|---|------|---------|---------|------|--------|---------|------|
| 1 | GBP/JPY | 35.2pip | 1.4x | 42pip | 15.1pip | -28pip | 🔥高ボラ |
| 2 | USD/JPY | 22.1pip | 1.1x | 28pip | 8.2pip | -18pip | 普通 |
| 3 | EUR/USD | 12.3pip | 0.8x | 15pip | 4.1pip | +5pip | 😴低ボラ |
...

### ボラ状態
🔥 高ボラ (ATR比率>1.3): GBP/JPY, AUD/JPY
⚡ やや高 (1.1-1.3): USD/JPY
😴 低ボラ (<0.8): EUR/USD

### スキャルプ推奨
- 高ボラ → ワイドSL推奨 (ATR×1.5)
- 低ボラ → タイトSLでも可、ただし値幅小さい
- **最適ペア**: GBP/JPY (十分な値幅 + トレンドあり)
```

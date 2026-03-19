---
name: correlation-matrix
description: "主要ペア間の相関マトリクスを計算・表示。ヘッジ判断・相関リスク検知用。"
trigger: "Use when the user says '相関', 'correlation', 'ヘッジ', or asks about pair relationships."
---

# 相関マトリクススキル

## 実行手順

### Step 1: データ取得

OANDA APIから全ペアのH1キャンドル直近50本を取得:
```
GET /v3/instruments/{pair}/candles?granularity=H1&count=50
```

### Step 2: 相関計算

各ペアのclose変化率(%)を計算し、ピアソン相関係数を算出。

```python
import numpy as np
# returns = [(close[i] - close[i-1]) / close[i-1] for each candle]
# correlation = np.corrcoef(returns_pair1, returns_pair2)[0,1]
```

### Step 3: 出力

```
## 📈 相関マトリクス (H1, 直近50本)

|       | UJ    | EU    | GU    | AU    | EJ    | GJ    | AJ    |
|-------|-------|-------|-------|-------|-------|-------|-------|
| UJ    | 1.00  | -0.85 | -0.72 | -0.68 | 0.45  | 0.52  | 0.38  |
| EU    |       | 1.00  | 0.92  | 0.78  | 0.65  | 0.58  | 0.55  |
...

### ⚠️ 相関リスク警告

現在のポジション:
- AUD_USD SHORT 1000u

相関分析:
- AUD/USD ↔ EUR/USD: r=0.78 (高相関)
  → もしEU SHORTも入ると実質2倍のUSD LONGポジション

### ヘッジ候補
- AUD/USD SHORTのヘッジ: AUD/JPY LONG (r=-0.55, 逆相関)
```

### 相関の解釈
| 範囲 | 意味 | アクション |
|------|------|-----------|
| 0.8~1.0 | 強い正相関 | 同方向ポジは実質2倍リスク |
| 0.5~0.8 | 中程度正相関 | 注意。ロット調整検討 |
| -0.3~0.3 | 無相関 | 独立。分散効果あり |
| -0.8~-0.5 | 中程度逆相関 | ヘッジ候補 |
| -1.0~-0.8 | 強い逆相関 | 同方向=ヘッジ効果 |

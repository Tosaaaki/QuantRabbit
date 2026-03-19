---
name: position-sizer
description: "口座リスク計算機。SL幅とリスク%から最適ユニット数を即計算。通貨ペア・口座残高・レバレッジから自動算出。"
trigger: "Use when the user says 'ロット', 'サイズ', 'リスク計算', 'position size', 'lot', or asks how many units to trade."
---

# 口座リスク計算機スキル

「UJ 30pipでリスク1%」→ 即座にユニット数を回答。

## 使い方

- 「EU 25pip リスク1%」
- 「UJ SL 159.600 リスク500円」
- 「ロット計算 GJ 40pip」

## 実行手順

### Step 1: OANDA口座情報取得

```bash
# config/env.tomlから認証情報を読み、口座サマリーを取得
```

必要データ:
- Balance (口座残高)
- 対象ペアの現在価格

### Step 2: 計算

#### 基本公式
```
リスク額 = Balance × リスク% (デフォルト1%)
pip値 = (0.0001 / 現在価格) × ユニット数 × JPY換算レート
  ※JPYペアの場合: pip値 = 0.01 × ユニット数 / 1（JPY口座なのでそのまま）
  ※USDペアの場合: pip値 = 0.0001 × ユニット数 × USD/JPY
ユニット数 = リスク額 / (SL_pips × 1pip当たりの価値)
```

#### JPYペア（USD/JPY, EUR/JPY等）
```
1pip = 0.01
pip値(JPY) = 0.01 × units = units / 100
units = リスク額 / (SL_pips × 0.01)
       = リスク額 × 100 / SL_pips
```

#### USDペア（EUR/USD, GBP/USD等）
```
1pip = 0.0001
pip値(USD) = 0.0001 × units
pip値(JPY) = 0.0001 × units × USD/JPY
units = リスク額 / (SL_pips × 0.0001 × USD/JPY)
```

### Step 3: 出力

```
## ポジションサイズ計算

| 項目 | 値 |
|------|-----|
| ペア | EUR/USD |
| 方向 | SHORT |
| Entry | 1.14755 |
| SL | 1.15100 (34.5pip) |
| リスク% | 1.0% |
| 口座残高 | 28,610 JPY |
| リスク額 | 286 JPY |
| **推奨ユニット数** | **178u** |
| 実際リスク | 284 JPY (0.99%) |
| 最大レバ25倍上限 | 6,250u |

### TP別リターン
| TP距離 | R:R | 期待利益 |
|--------|-----|---------|
| 20pip | 0.58:1 | 166 JPY |
| 34.5pip | 1:1 | 286 JPY |
| 50pip | 1.45:1 | 414 JPY |
```

## デフォルト値

- リスク%未指定: **1%**
- ペア未指定: shared_stateの保有ポジのペア
- SL未指定: 「SL何pip？」と聞く（計算不能）

---
name: move-sl
description: "ストップロスを変更。「EU SLをBEに」「UJ SL 159.200」で即実行。"
trigger: "Use when the user says 'SL', 'ストップ', 'BE', 'ブレークイーブン', 'breakeven', or asks to move/change a stop loss."
---

# SL移動スキル

## 使い方

- 「EU SLをBEに」→ ストップロスをエントリー価格に設定
- 「UJ SL 159.200」→ 指定価格にSL移動
- 「AU SL +10pip」→ 現在SLから10pip利益方向に移動

## 実行手順

### Step 1: 対象トレード取得

OANDA APIでopenTradesから対象トレードを取得。Entry価格・現在SL・方向を確認。

### Step 2: SL価格計算

- **BE (ブレークイーブン)**: SL = Entry価格 + スプレッド分（微調整）
- **絶対値**: ユーザー指定価格をそのまま使用
- **相対値 (+Npip)**:
  - SHORT: 現在SLからN pip下げる (SL - N×pip単位)
  - LONG: 現在SLからN pip上げる (SL + N×pip単位)

### Step 3: 安全チェック

- SLが現在価格より不利な方向にあることを確認
  - SHORT: SL > 現在bid
  - LONG: SL < 現在ask
- SLを利益方向に動かす場合は問題なし
- SLを損失方向に広げる場合は警告を出す

### Step 4: 実行

```
PUT /v3/accounts/{acct}/trades/{trade_id}/orders
{
  "stopLoss": {
    "price": "159.200",
    "timeInForce": "GTC"
  }
}
```

### Step 5: 確認

```
✅ SL変更完了
USD/JPY SHORT 250u: SL 159.600 → 159.200 (40pip縮小)
現在リスク: 7.6pip (以前: 47.6pip)
```

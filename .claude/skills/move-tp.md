---
name: move-tp
description: "テイクプロフィットを変更。「UJ TP 158.5」で即実行。"
trigger: "Use when the user says 'TP', 'テイクプロフィット', 'take profit', '利確', or asks to move/change take profit."
---

# TP移動スキル

## 使い方

- 「UJ TP 158.500」→ 指定価格にTP移動
- 「EU TP 1.14200」→ 指定価格にTP
- 「AU TP削除」→ TP解除

## 実行手順

### Step 1: 対象トレード取得

OANDA APIでopenTradesから対象トレードを取得。

### Step 2: 実行

```
PUT /v3/accounts/{acct}/trades/{trade_id}/orders
{
  "takeProfit": {
    "price": "158.500",
    "timeInForce": "GTC"
  }
}
```

TP削除の場合: takeProfitの設定を空にする。

### Step 3: 確認

```
✅ TP変更完了
USD/JPY SHORT 250u: TP 158.900 → 158.500 (40pip拡大)
現在Entry→TP距離: 80.8pip | R:R = 2.1:1 (SL=159.200基準)
```

R:R計算も表示して判断材料にする。

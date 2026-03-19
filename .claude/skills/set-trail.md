---
name: set-trail
description: "トレーリングストップを設定。「UJ トレール 20pip」で即実行。"
trigger: "Use when the user says 'トレール', 'trail', 'トレーリング', or asks to set a trailing stop."
---

# トレーリングストップ設定スキル

## 使い方

- 「UJ トレール 20pip」
- 「EUにトレーリング30pip」

## 実行手順

### Step 1: 対象トレード特定

OANDA APIでopenTradesから対象ペアを検索。

### Step 2: トレーリングストップ設定

```
PUT /v3/accounts/{acct}/trades/{trade_id}/orders
{
  "trailingStopLoss": {
    "distance": "0.200",  // 20pip = 0.200 (JPYペア) or 0.00200 (USDペア)
    "timeInForce": "GTC"
  }
}
```

pip→価格距離の変換:
- JPYペア: distance = pips × 0.01
- USDペア: distance = pips × 0.0001

### Step 3: 確認

```
✅ トレーリングストップ設定完了
USD/JPY SHORT 250u: Trail=20pip (0.200)
現在価格: 159.124 → 現在のSL水準: 159.324
```

既存のSLがある場合は上書きされる旨を通知。

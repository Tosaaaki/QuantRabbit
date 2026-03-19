---
name: limit-order
description: "指値/逆指値注文。「UJ LIMIT BUY 158.800 300u」で指値エントリー。"
trigger: "Use when the user says '指値', 'limit', '逆指値', 'stop order', 'pending', or asks to place a limit/stop order."
---

# 指値注文スキル

## 使い方

- 「UJ LIMIT BUY 158.800 300u SL 158.500 TP 159.500」
- 「EU STOP SELL 1.14500 500u」→ 逆指値売り
- 「指値一覧」→ 現在のペンディング注文一覧
- 「指値キャンセル [orderID]」→ キャンセル

## 注文タイプ

- **LIMIT**: 現在より有利な価格で約定（押し目買い/戻り売り）
- **STOP**: 現在より不利な価格で約定（ブレイクアウト追従）

## 実行手順

### Step 1: 確認

```
📋 指値注文確認:
| 項目 | 値 |
|------|-----|
| タイプ | LIMIT |
| ペア | USD/JPY |
| 方向 | BUY |
| 価格 | 158.800 |
| 数量 | 300u |
| 現在価格 | 159.124 |
| 価格差 | 32.4pip |
| SL | 158.500 (30pip) |
| TP | 159.500 (70pip) |
| R:R | 2.33:1 |
| 有効期限 | GTC (キャンセルまで有効) |

実行する？
```

### Step 2: 注文

```
POST /v3/accounts/{acct}/orders
{
  "order": {
    "type": "LIMIT",
    "instrument": "USD_JPY",
    "units": "300",
    "price": "158.800",
    "timeInForce": "GTC",
    "stopLossOnFill": {"price": "158.500"},
    "takeProfitOnFill": {"price": "159.500"}
  }
}
```

### 指値一覧

```
GET /v3/accounts/{acct}/pendingOrders
```

### 指値キャンセル

```
PUT /v3/accounts/{acct}/orders/{order_id}/cancel
```

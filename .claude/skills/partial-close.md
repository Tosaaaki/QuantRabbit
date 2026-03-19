---
name: partial-close
description: "部分利確の計算と実行。「EU 半分利確」でR:R維持しながら部分決済。"
trigger: "Use when the user says '部分利確', '半分', 'partial', '一部決済', or asks to close part of a position."
---

# 部分利確計算機スキル

## 使い方

- 「EU 半分利確」→ 50%決済
- 「AU 300u利確」→ 指定ユニット決済
- 「UJ R:R 1:1分だけ利確」→ SL分の利益を確保して残りをフリーライド

## 実行手順

### Step 1: 現在ポジション取得

OANDA APIでopenTradesから対象を取得。

### Step 2: 計算

```
例: EUR/USD SHORT 500u @1.14755, 現在1.14550 (含み益+20.5pip)

半分利確 (250u):
- 決済分: 250u × 20.5pip × 0.0001 × USD/JPY = +xxx JPY
- 残り: 250u, SLをBEに移動 → リスクフリー

R:R 1:1利確:
- SL距離: 34.5pip → SL分の利益 = 34.5pip
- 現在含み益: 20.5pip → まだR:R 1:1未達
- R:R 1:1到達価格: 1.14410
```

### Step 3: 実行

```
PUT /v3/accounts/{acct}/trades/{trade_id}/close
{"units": "250"}  # 部分クローズ
```

その後、残りポジションのSLをBEに移動することを提案。

### Step 4: 出力

```
## 💰 部分利確

✅ EUR/USD SHORT 250u決済 @1.14550 → +xxx JPY

残りポジション:
| 数量 | Entry | 現SL | 推奨SL |
|------|-------|------|--------|
| 250u | 1.14755 | 1.15100 | 1.14755(BE) |

SLをBEに移動する？ → リスクフリーのフリーライドポジションに
```

---
name: close-pair
description: "特定ペアのポジションだけをクローズ。「UJ決済」「EU閉じて」で即実行。"
trigger: "Use when the user says '[pair]決済', '[pair]閉じて', '[pair] close', or asks to close a specific pair's position."
---

# ペア決済スキル

特定ペアのポジションだけをクローズする。

## 使い方

- 「UJ決済」「EU閉じて」「AU close」
- 「EURのポジション閉じて」

## ペア名マッピング

UJ=USD_JPY, EU=EUR_USD, GU=GBP_USD, AU=AUD_USD, EJ=EUR_JPY, GJ=GBP_JPY, AJ=AUD_JPY

## 実行手順

### Step 1: 対象ポジション確認

OANDA APIでopenTradesを取得し、対象ペアのトレードを表示:

```
⚠️ [ペア]決済確認:
| ID | 方向 | 数量 | Entry | UPL |
（該当トレード一覧）
全部閉じる？ or 特定IDだけ？
```

### Step 2: 実行

```
PUT /v3/accounts/{acct}/trades/{trade_id}/close
```

同一ペアに複数ポジがある場合は全部クローズ（ユーザーが特定IDを指定しない限り）。

### Step 3: 結果報告

クローズ結果と実現P&Lを表示。

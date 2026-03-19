---
name: market-order
description: "成行注文で即エントリー。「EU SHORT 500」でSL/TP自動設定付き成行注文。"
trigger: "Use when the user says '成行', 'market', 'エントリー', or gives a direct order like '[pair] SHORT/LONG [units]'."
---

# 成行エントリースキル

## 使い方

- 「EU SHORT 500」→ EUR/USD 500u成行売り + SL/TP自動
- 「UJ LONG 300 SL 158.800 TP 160.000」→ 指定SL/TP付き
- 「AU SHORT 1000 SL 20pip TP 30pip」→ pip指定

## 実行手順

### Step 1: パラメータ確認

```
📋 注文確認:
| 項目 | 値 |
|------|-----|
| ペア | EUR/USD |
| 方向 | SHORT |
| 数量 | 500u |
| 現在価格 | 1.14720 (bid) |
| SL | 1.15020 (30pip) |
| TP | 1.14320 (40pip) |
| R:R | 1.33:1 |
| リスク額 | xxx JPY (口座のx.x%) |

実行する？
```

**ユーザー確認必須。確認なしで注文しない。**

### Step 2: SL/TP未指定時のデフォルト

- SL未指定: 「SL何pip？」と聞く（デフォルトなし。SLなし注文は禁止）
- TP未指定: SL × 1.5 をデフォルト提案

### Step 3: 注文実行

```python
POST /v3/accounts/{acct}/orders
{
  "order": {
    "type": "MARKET",
    "instrument": "EUR_USD",
    "units": "-500",  # マイナス=SELL
    "timeInForce": "FOK",
    "stopLossOnFill": {"price": "1.15020", "timeInForce": "GTC"},
    "takeProfitOnFill": {"price": "1.14320", "timeInForce": "GTC"}
  }
}
```

### Step 4: ログ記録 + 結果報告

`logs/live_trade_log.txt` にエントリー記録を追記:
```
[YYYY-MM-DDTHH:MM:SSZ] MANUAL: EUR_USD S 500u @1.14720 SL=1.15020(30pip) TP=1.14320(40pip) R:R=1.33:1
```

```
✅ 注文約定
EUR_USD SHORT 500u @1.14720
SL=1.15020 TP=1.14320 R:R=1.33:1
Trade ID: 463xxx
```

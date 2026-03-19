---
name: close-all
description: "全オープンポジションをOANDA APIで一括クローズ。「全決済」で即実行。"
trigger: "Use when the user says '全決済', 'close all', '全クローズ', '全部閉じて', or asks to close all positions."
---

# 全決済スキル

全オープンポジションを一括クローズする。

## 実行手順

### Step 1: 確認

OANDA APIでopenTradesを取得し、クローズ対象を表示:

```
⚠️ 全決済確認:
| # | ペア | 方向 | 数量 | UPL |
|---|------|------|------|-----|
| 1 | USD/JPY | SHORT | 250 | +45 |
| 2 | EUR/USD | SHORT | 500 | +1 |

合計UPL: +46 JPY
本当に全決済する？ (y/n)
```

**ユーザーの確認を必ず待つ。確認なしでクローズしない。**

### Step 2: 実行

確認後、各トレードに対して:
```
PUT /v3/accounts/{acct}/trades/{trade_id}/close
```

### Step 3: 結果報告

```
✅ 全決済完了
| # | ペア | 実現P&L |
|---|------|---------|
| 1 | USD/JPY | +45 JPY |
| 2 | EUR/USD | +1 JPY |
合計: +46 JPY
```

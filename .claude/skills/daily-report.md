---
name: daily-report
description: "今日のトレード損益・勝率・平均R:R・最大DD・反省点をまとめてログ保存。"
trigger: "Use when the user says '日次', 'daily', '今日のまとめ', '今日の成績', 'EOD', or asks for daily trading summary."
---

# 日次レポートスキル

## 実行手順

### Step 1: データ取得

1. OANDA API: 今日のクローズ済みトレード
   ```
   GET /v3/accounts/{acct}/trades?state=CLOSED&count=50
   ```
   当日分をフィルタ
2. OANDA API: 口座サマリー (NAV, Balance)
3. logs/live_trade_log.txt: 当日のエントリー・判断根拠

### Step 2: 統計計算

- トレード数、勝率
- 合計P&L、平均P&L
- 最大勝ち/最大負け
- 平均R:R
- Profit Factor (総利益 ÷ 総損失)
- 最大ドローダウン (口座内日中)
- ペア別P&L
- セッション別P&L (東京/ロンドン/NY)

### Step 3: 出力

```
## 📊 日次レポート: 2026-03-19

### サマリー
| 指標 | 値 |
|------|-----|
| トレード数 | N |
| 勝率 | xx% (xW / xL) |
| 合計P&L | +/-xxx JPY |
| 平均R:R | x.x:1 |
| Profit Factor | x.xx |
| 最大勝ち | +xxx JPY (ペア) |
| 最大負け | -xxx JPY (ペア) |
| NAV開始 | xxx JPY |
| NAV終了 | xxx JPY |
| 日次リターン | +/-x.x% |

### トレード一覧
| # | ペア | 方向 | Entry | Exit | P&L | R:R | 保有時間 |
...

### ペア別
| ペア | トレード数 | P&L | 勝率 |
...

### 良かった点
- (分析結果から自動抽出)

### 改善点
- (分析結果から自動抽出)

### 明日の注意点
- (マクロイベント、持越しポジション等)
```

### Step 4: 保存

`docs/trade_journal/daily/YYYY-MM-DD.md` に保存。

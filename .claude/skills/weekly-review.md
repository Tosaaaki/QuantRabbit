---
name: weekly-review
description: "週間トレードログから統計・パターン分析・改善点を抽出。"
trigger: "Use when the user says '週次', 'weekly', '今週のまとめ', '週間レビュー', or asks for weekly performance review."
---

# 週次レビュースキル

## 実行手順

### Step 1: データ取得

1. OANDA API: 今週のクローズ済みトレード (月曜〜現在)
2. 日次レポート: `docs/trade_journal/daily/` から今週分
3. memory: `project_trading_strategy.md` の教訓

### Step 2: 週間統計

日次レポートの上位互換 + 以下を追加:
- **曜日別勝率**: 月〜金で差があるか
- **時間帯別勝率**: 東京/ロンドン/NY
- **連勝/連敗**: 最大連勝数と連敗数
- **Play別成績**: どのセットアップが最も稼いだか
- **感情パターン**: (日記があれば) 焦り時の勝率 vs 冷静時

### Step 3: 出力

```
## 📈 週次レビュー: 2026-03-17 〜 03-19

### 週間サマリー
| 指標 | 値 | 前週比 |
|------|-----|--------|
| トレード数 | N | +/-N |
| 勝率 | xx% | +/-x% |
| 週間P&L | +/-xxx JPY | - |
| Profit Factor | x.xx | - |
| 最大DD | -xxx JPY | - |
| NAV変動 | xxx → xxx | +x.x% |

### パターン分析
🏆 最高パフォーマンスPlay: "H1 Bear + M5 Bounce Peak" (5W/1L, +450 JPY)
💀 最悪パフォーマンスPlay: "RSI Extreme Counter" (0W/3L, -180 JPY)

### 曜日・時間帯
| 曜日 | 勝率 | P&L |
| 時間帯 | 勝率 | P&L |

### 🔑 今週の教訓
1. xxx
2. xxx
3. xxx

### 📋 来週の改善アクション
1. xxx → (具体的アクション)
2. xxx → (具体的アクション)
```

### Step 4: 保存

`docs/trade_journal/weekly/YYYY-WXX.md` に保存。
教訓を `project_trading_strategy.md` に追記。

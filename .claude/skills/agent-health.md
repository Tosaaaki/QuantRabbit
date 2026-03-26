---
name: agent-health
description: "全Scheduled Taskの稼働状況・最終実行時間・エラー有無を一覧。エージェント健康診断。"
trigger: "Use when the user says '健康診断', 'health', 'エージェント状態', 'タスク状態', 'agent status', or asks about scheduled task health."
---

# エージェント健康診断スキル

## 実行手順

### Step 1: データ取得

1. ロック状態: `python3 tools/task_lock.py status`
2. Scheduled Tasks一覧: `~/.claude/scheduled-tasks/` ディレクトリ確認
3. 各タスクのSKILL.md読み取り
4. shared_state.json: 各エージェントの最終更新時間
5. live_trade_log.txt: 最終ログ時刻

### Step 2: 各エージェント診断

| チェック項目 | 正常基準 |
|-------------|---------|
| 最終実行 | 想定間隔×2以内 |
| ロック | スタックしてない |
| ログ出力 | 最終ログが想定間隔内 |
| エラー | stderr出力なし |
| shared_state更新 | 各担当セクションが新鮮 |

### Step 3: 出力

```
## 🏥 エージェント健康診断

| エージェント | 間隔 | 最終実行 | 経過 | 状態 |
|-------------|------|---------|------|------|
| scalp-trader | 5min | 10:06Z | 19min | ⚠️遅延 |
| market-radar | 7min | 10:21Z | 4min | ✅正常 |
| macro-intel | 19min | 09:45Z | 40min | ⚠️遅延 |
| secretary | 11min | 10:15Z | 10min | ✅正常 |

### ロック状態
| ロック | 状態 | 所有者 | 経過時間 |
|--------|------|--------|---------|
| global_agent | RUNNING | scalp_trader | 242s |

### 問題検知
⚠️ scalp-trader: 最終実行から19分経過 (通常5分間隔)
  → 原因候補: ロック待ち / タスクスケジューラ遅延 / エラー停止

### 推奨アクション
1. scalp-traderの次回実行を待つ (あと1分で次のスケジュール)
2. 問題が続く場合: ロック強制解放を検討
```

---
name: force-unlock
description: "スタックしたグローバルロックを安全に強制解放。"
trigger: "Use when the user says 'ロック解放', 'unlock', 'force release', 'ロック解除', or asks to release a stuck lock."
---

# ロック強制解放スキル

## 実行手順

### Step 1: 現在のロック状態確認

```bash
python3 scripts/trader_tools/task_lock.py status
```

### Step 2: 安全チェック

ロックが本当にスタックしているか確認:
- PIDが生きているか
- 経過時間がタイムアウトを超えているか
- 所有者タスクの最終ログ時刻

### Step 3: ユーザー確認

```
⚠️ ロック強制解放確認:
| ロック | 所有者 | PID | 経過 | PID状態 |
|--------|--------|-----|------|---------|
| global_agent | scalp_trader | 37607 | 842s | alive |

PIDがまだ生きています。強制解放すると実行中のタスクに影響する可能性があります。
本当に解放する？
```

### Step 4: 実行

```bash
python3 scripts/trader_tools/task_lock.py release global_agent
```

### Step 5: 確認

```
✅ ロック解放完了: global_agent
次のタスクが正常にロックを取得できます。
```

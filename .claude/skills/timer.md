---
name: timer
description: "リマインダー/タイマー設定。「15分後にリマインド」でScheduled Task一発セット。"
trigger: "Use when the user says 'タイマー', 'timer', 'リマインド', 'remind', 'アラーム', '〇分後に', or asks to be reminded of something."
---

# タイマー/リマインダースキル

## 使い方

- 「15分後にリマインド: EU TP確認」
- 「12:00にリマインド: BOE発表」
- 「30分タイマー」

## 実行手順

### Step 1: 時間計算

ユーザーの指定を解析:
- 「15分後」→ 現在時刻 + 15分 → ISO 8601 fireAt
- 「12:00」→ 今日の12:00 JST → ISO 8601 fireAt
- 「明日9時」→ 明日の09:00 JST → ISO 8601 fireAt

### Step 2: Scheduled Task作成

`mcp__scheduled-tasks__create_scheduled_task` を使用:
```json
{
  "taskId": "reminder-YYYYMMDD-HHMM",
  "prompt": "ユーザーへのリマインダー: [メッセージ内容]。このメッセージを表示して、関連するOANDA APIデータ（ポジション状況等）も一緒に提示してください。",
  "description": "リマインダー: [メッセージ]",
  "fireAt": "2026-03-19T12:00:00+09:00"
}
```

### Step 3: 確認

```
⏰ リマインダー設定完了
内容: 「BOE発表 — GBPポジション確認」
時刻: 12:00 JST (あと47分)
Task ID: reminder-20260319-1200
```

### キャンセル

「タイマーキャンセル」→ `update_scheduled_task` で enabled=false

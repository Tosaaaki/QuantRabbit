---
name: memo
description: "構造化メモ。「メモ: 〇〇」でMEMORY.mdやshared_stateに記録。"
trigger: "Use when the user says 'メモ', 'memo', 'note', '記録', 'remember', or asks to save a note."
---

# メモスキル

## 使い方

- 「メモ: AUDは今週強い。RBA影響。」
- 「トレードメモ: EUのBE停止はやめたほうがいい」
- 「永続メモ: 159.5以上は介入リスク」

## メモの種類

### 1. トレードメモ (一時的)
→ shared_state.json の `user_notes` に保存。当日有効。
```json
"user_notes": [
  {"text": "AUDは今週強い", "at": "2026-03-19T10:30:00Z"}
]
```

### 2. 永続メモ
→ memory の適切なファイルに追記。セッション跨ぎで有効。
`project_trading_strategy.md` や新規memoryファイル

### 3. エージェント向けメモ
→ shared_state.json の `user_directives` + プロンプトMDに追記
(relay-messageスキルと同じ経路)

## 実行手順

### Step 1: メモ分類

内容から自動判定:
- トレード関連 → トレードメモ or 永続メモ
- 「永続」「ずっと」「忘れないで」→ 永続メモ
- 「〇〇に伝えて」→ エージェント向けメモ

### Step 2: 保存

対象ファイルに追記。

### Step 3: 確認

```
📝 メモ保存完了
内容: 「AUDは今週強い。RBA影響。」
保存先: shared_state.json (トレードメモ)
有効期限: 今日中
```

---
name: relay-message
description: "エージェントへの指示伝達。「scalp-traderに〇〇伝えて」でプロンプトMDやshared_stateに反映。"
trigger: "Use when the user says '伝えて', 'relay', '指示', 'トレーダーに', or asks to send instructions to an agent."
---

# プロンプト更新（指示伝達）スキル

## 使い方

- 「scalp-traderに『EUはLONG禁止』伝えて」
- 「全エージェントに『AUD SHORT禁止』」
- 「市場メモ: BOE12時で荒れる」

## 実行手順

### Step 1: 伝達先特定

| キーワード | 伝達先 |
|-----------|--------|
| scalp-trader, トレーダー | docs/SCALP_TRADER_PROMPT.md + shared_state |
| market-radar, レーダー | docs/MARKET_RADAR_PROMPT.md |
| macro-intel, マクロ | docs/MACRO_INTEL_PROMPT.md |
| 全エージェント, 全員 | shared_state.json の alerts |
| メモ, ノート | shared_state.json の user_notes |

### Step 2: 反映方法

#### shared_state.json への追記 (即時反映)
```json
{
  "user_directives": [
    {
      "message": "EUR/USD LONG禁止。H4がまだBEAR。",
      "from": "user",
      "at": "2026-03-19T10:30:00Z",
      "expires": "2026-03-19T18:00:00Z"
    }
  ]
}
```

#### プロンプトMDへの追記 (次回実行から反映)
対象プロンプトファイルの「ユーザー指示」セクションに追記。

### Step 3: 確認

```
✅ 指示伝達完了

伝達先: scalp-trader
内容: 「EUR/USD LONG禁止。H4がまだBEAR。」
方法:
- shared_state.json の user_directives に追記（即時反映）
- docs/SCALP_TRADER_PROMPT.md にも追記（次回実行から反映）
有効期限: 今日18:00Zまで (期限後自動削除)
```

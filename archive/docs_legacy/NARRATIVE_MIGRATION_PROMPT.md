# Claude Code用プロンプト: ナラティブレイヤー移行

以下をClaude Codeに貼り付けて実行してください。

---

## やること

ナラティブレイヤーを導入した。セッション間のストーリー連続性を確保して、コンテキスト切れでも「別人」にならないようにする。

`docs/ANALYST_PROMPT.md` と `docs/TRADER_PROMPT.md` は既に更新済み。
あとは **analyst定期タスクのSKILL.md** を更新して、新しいSTEP 4（相場ストーリー記述）とSTEP 6（market_narrative更新）を反映する必要がある。

### 1. analyst SKILL.md更新

`~/.claude/scheduled-tasks/analyst/SKILL.md` を開いて、以下を反映:

- **STEP 4を新設**: 「相場ストーリーを書け（毎サイクル必須）」
  - shared_stateの `market_narrative` を毎サイクル更新
  - ストーリーに含めるもの: (1)経緯 (2)現在地 (3)転換条件 (4)traderへの示唆
  - 詳細は `docs/ANALYST_PROMPT.md` のSTEP 4セクションを参照して、そのまま反映

- **STEP 5→6にずらす**: ONE ACTION選択とshared_state更新のステップ番号を繰り下げ

- **shared_state更新コード**: `market_narrative` フィールドの定義を追加
  ```python
  state['market_narrative'] = {
      'updated_at': now_utc,
      'story': '経緯→現在地→転換条件→示唆を含む5-10行のナラティブ',
      'key_thesis': [...],  # ペアごとのテーゼ（narrative, basis, invalidation, history）
      'session_learnings': [...]  # 今日学んだこと
  }
  ```
  詳細フォーマットは `docs/ANALYST_PROMPT.md` のSTEP 6を参照

### 2. trader SKILL.md確認

`~/.claude/scheduled-tasks/trader/SKILL.md` に以下が含まれているか確認:
- Step 0「ストーリーを掴め」が入っているか
- `docs/TRADER_PROMPT.md` を毎サイクル冒頭で参照する指示があるか

もしSKILL.mdが `docs/TRADER_PROMPT.md` を直接参照する形式なら、TRADER_PROMPT.md側は既にStep 0を追加済みなので変更不要。
SKILL.mdに独自のフロー定義がコピーされている場合は、Step 0を追加すること。

### 3. 変更後の確認

- analyst定期タスクを1回手動実行して、shared_stateに `market_narrative` が書き込まれることを確認
- trader定期タスクを1回手動実行して、Step 0でmarket_narrativeを読んでいることを確認
- docs/CHANGELOG.md に追記済み（2026-03-22T10:00Z）

### 背景

共同トレードで+11.9%/日（+1,760円）を達成したが、自動トレードではセッション間の記憶断絶がボトルネック。
analystが相場の「物語」を書き、traderがそれを読むことで、セッションが切れてもストーリーが繋がる。

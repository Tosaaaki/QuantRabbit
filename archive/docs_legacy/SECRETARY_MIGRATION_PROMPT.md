# Secretary Recording Migration — Claude Code実行用プロンプト

**このプロンプトをClaude Codeに貼り付けて実行してください。**

## やること

secretary の SKILL.md を更新して、ポジション変化検知・自動記録機能を追加する。

## 手順

1. `~/.claude/scheduled-tasks/secretary/SKILL.md` を開く

2. プロンプト本文を以下の方針で更新:
   - **最新のプロンプトは `docs/SECRETARY_PROMPT.md` にある** — これを読んでSKILL.mdに反映
   - 核心: secretaryは「記録係」に拡張された
   - 毎サイクル `python3 scripts/trader_tools/position_diff.py` を実行
   - `shared_state.json` の `collab_mode` を見て `--collab` フラグを付ける
   - 共同トレード中は `state.md` と `daily/trades.md` が自動更新される
   - 旧エージェント名(scalp-fast, swing-trader, market-radar, macro-intel)はv4体制に更新(trader, analyst)

3. SKILL.md のフロントマター(schedule等)はそのまま維持

4. `docs/CHANGELOG.md` に以下を追記:
   ```
   - 2026-03-23: secretary v4.1 — 記録係に拡張。position_diff.pyでポジション変化検知・自動記録。collab_modeフラグ対応
   ```

## 確認

変更後、以下を確認:
- `position_diff.py` が `scripts/trader_tools/` に存在する（既に作成済み）
- SKILL.md のプロンプトに `position_diff.py` の呼び出しが含まれている
- `collab_mode` の検知ロジックが含まれている

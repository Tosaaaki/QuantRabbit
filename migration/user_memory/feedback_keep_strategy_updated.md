---
name: 戦略メモリを常時更新
description: project_trading_strategy.mdを変更のたびに最新状態に保つ。戦略は常に変化する
type: feedback
---

project_trading_strategy.md は常に最新を維持しろ。

**Why:** 戦略は毎セッション・毎会話で進化し続ける。古い記述が残ると次のセッションのClaudeが旧思想で動く。

**How to apply:** TRADER_PROMPT/monitor/戦略方針に変更を加えたら、その場でproject_trading_strategy.mdも更新する。CHANGELOG追記と同じタイミングで。「後でまとめて」は禁止 — 変更した瞬間にやれ。

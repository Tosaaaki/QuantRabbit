# Archive — v3 (5エージェント体制)

2026-03-20にv4（3エージェント体制）へ移行。ここにv3のファイルを保存。

## 復元方法

### プロンプト復元
```bash
cp docs/archive/v3_prompts/SCALP_FAST_PROMPT.md docs/
cp docs/archive/v3_prompts/SWING_TRADER_PROMPT.md docs/
cp docs/archive/v3_prompts/MARKET_RADAR_PROMPT.md docs/
cp docs/archive/v3_prompts/MACRO_INTEL_PROMPT.md docs/
cp docs/archive/v3_prompts/SECRETARY_PROMPT.md docs/
```

### スケジュールタスク復元
```bash
# 旧タスクのSKILL.mdを ~/.claude/scheduled-tasks/{task}/ にコピー
for task in scalp-fast swing-trader market-radar macro-intel secretary; do
  mkdir -p ~/.claude/scheduled-tasks/$task
  cp docs/archive/v3_scheduled_tasks/${task}_SKILL.md ~/.claude/scheduled-tasks/$task/SKILL.md
done
# 新タスク(trader, analyst)を削除
rm -rf ~/.claude/scheduled-tasks/trader ~/.claude/scheduled-tasks/analyst
```

### CLAUDE.md復元
git logからv3時点のCLAUDE.mdをcheckoutする。

## ファイル一覧

### v3_prompts/
- SCALP_FAST_PROMPT.md — 高速スキャルプ(2分間隔)
- SWING_TRADER_PROMPT.md — スウィング(10分間隔)
- MARKET_RADAR_PROMPT.md — 監視・急変検知(7分間隔)
- MACRO_INTEL_PROMPT.md — マクロ分析・戦略改善(19分間隔)
- SECRETARY_PROMPT.md — アカウンタビリティ監査(11分間隔)

### v3_scheduled_tasks/
- scalp-fast_SKILL.md
- swing-trader_SKILL.md
- market-radar_SKILL.md
- macro-intel_SKILL.md
- secretary_SKILL.md

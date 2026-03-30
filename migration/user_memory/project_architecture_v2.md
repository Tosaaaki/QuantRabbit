---
name: Architecture v8 — trader一本の裁量トレードシステム
description: 2026-03-26。trader(Sonnet, 1分cron, 2分セッション)が唯一の定期タスク。旧エージェント・ボット群は全てアーカイブ済
type: project
---

v8 (2026-03-26): traderを正のシステムとして昇格。旧遺産を全てアーカイブ。

**アーキテクチャ:**
- **trader** — Sonnet / 1分cron / 最大2分セッション
- 分析・ニュース・ポジション管理・トレード執行を全て一人で実行
- 記憶: state.md（短期）、strategy_memory.md（長期）、memory.db（ベクトル検索）

**v8で整理したもの:**
- 無効scheduled tasks 7個削除 (scalp-trader, market-radar, macro-intel, auto-commit-push, secretary, scalp-fast, swing-trader)
- scripts/直下の旧スクリプト162個 → scripts/archive/
- workers/ → workers_archive/
- addons/, analysis/, tests/ → *_archive/
- logs/の旧ファイル → logs/archive_legacy/

**残っている正のシステム:**
- `~/.claude/scheduled-tasks/trader/` — 唯一の定期トレードタスク
- `scripts/trader_tools/` — traderの道具
- `indicators/` — テクニカル計算エンジン
- `collab_trade/` — 状態・記憶・日次記録
- `.claude/rules/` — ルール
- `.claude/skills/` — スラッシュコマンド

**Why:** v1-v6の遺産(ボットワーカー、マルチエージェント、バックテスト群)がリポジトリの9割を占めていた。traderだけが実際に稼働しているシステム。
**How to apply:** 新しいコードはscripts/trader_tools/に追加。旧コードが必要な場合は*_archive/から探す。

# QuantRabbit — Claude裁量FXスキャルプシステム

このリポジトリはClaudeによる裁量FXスキャルピングの運用基盤。
ボット/自動売買ではない。Claudeが自分の頭で判断してOANDA APIで直接注文する。

## アーキテクチャ

4つの定期タスク(Claude Code Scheduled Tasks)が連携して動く:

| タスク | モデル | 間隔 | 役割 |
|--------|--------|------|------|
| scalp-trader | Opus | 5分 | 市況分析→裁量判断→OANDA直接注文 |
| market-radar | Sonnet | 7分 | ポジション監視・急変検知・レジーム変化検知 |
| macro-intel | Sonnet | 19分 | マクロ分析・戦略改善・ツール開発 |
| secretary | Sonnet | 11分 | エージェント監視・状況レポート・異常検知 |

- 排他制御: `scripts/trader_tools/task_lock.py` でグローバルロック（`global_agent`）。互いに素な間隔で衝突最小化
- エージェント間連携: `logs/shared_state.json`
- 各タスクのプロンプト: `docs/*_PROMPT.md`
- タスク定義: `~/.claude/scheduled-tasks/*/SKILL.md`
- セットアップ: `bash scripts/trader_tools/setup_scheduled_tasks.sh`

## 絶対ルール

- **ボット禁止**: `while True` + `sleep` の常駐スクリプトを書かない。`workers/` のコードは参照のみ
- **OANDA直接注文**: urllib で REST API を叩く。ボットプロセス経由禁止
- **裁量トレーダーであれ**: ルール実行マシンではない。市況を読んで自分で判断する
- テクニカル指標は `indicators/factor_cache.py` から取得(手計算しない)
- 注文は必ず `logs/live_trade_log.txt` にファイル記録

## 主要ディレクトリ

- `docs/` — 各タスクのプロンプト、トレードログ
- `scripts/trader_tools/` — 分析ツール、ロック機構、セットアップ
- `indicators/` — テクニカル指標キャッシュ
- `logs/` — 共有状態、トレードログ、ロックファイル
- `config/env.toml` — OANDA APIキー等(gitignore対象)

## ユーザーコマンド

- 「トレード開始」→ マルチエージェント裁量トレードセッション起動
- 「秘書」→ トレーディング秘書モード(状況確認・指示伝達)

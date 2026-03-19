# QuantRabbit — Claude裁量FXスキャルプシステム

このリポジトリはClaudeによる裁量FXスキャルピングの運用基盤。
ボット/自動売買ではない。Claudeが自分の頭で判断してOANDA APIで直接注文する。
テクニカル計算・機械的ポジション管理はPythonスクリプトが担う。

## アーキテクチャ (v3)

### Python層（LLMコストゼロ、30秒間隔）
- `scripts/trader_tools/live_monitor.py` — launchdで30秒ごとに実行
  - データ: pricing, S5/M1/M5指標(divergence, Ichimoku, VWAP含む), H1/H4バイアス
  - シグナルスコアリングv4: 7ペア×2方向、5カテゴリ(Direction+Timing+Confluence+Macro+Session)、最大+10点
  - ペアプロファイル: pair別のspread gate, SL/TP範囲, ADX閾値, セッション適性, ペア性格
  - 機械的ポジ管理: `logs/trade_registry.json` のルールに従いtrail/partial/close実行
  - リスク: margin使用率, ドローダウン, サーキットブレーカー
  - 出力: `logs/live_monitor.json`

### Claude層（裁量判断のみ）

| タスク | モデル | 間隔 | ロック | 役割 |
|--------|--------|------|--------|------|
| scalp-fast | Sonnet | 2分 | なし | モニター読む→ペア特性+コンフルエンス裁量→3-8pipスキャルプ |
| swing-trader | Opus | 10分 | global | H1/H4+divergence/Ichimoku深い分析→10-50pipスウィング |
| market-radar | Sonnet | 7分 | global | ポジション監視・急変検知・レジーム変化検知 |
| macro-intel | Sonnet | 19分 | global | マクロ分析・戦略改善・ツール開発 |
| secretary | Sonnet | 11分 | global | エージェント監視・状況レポート・異常検知 |

- 排他制御: `scripts/trader_tools/task_lock.py` でグローバルロック（`global_agent`）
- エージェント間連携: `logs/shared_state.json`
- ポジション所有権: `logs/trade_registry.json`
- 各タスクのプロンプト: `docs/*_PROMPT.md`
- タスク定義: `~/.claude/scheduled-tasks/*/SKILL.md`

## 絶対ルール

- **ボット禁止**: `while True` + `sleep` の常駐スクリプトを書かない。`workers/` のコードは参照のみ
- **Pythonツールは可**: テクニカル計算・機械的ポジ管理のスクリプトは可。トレード判断はClaude裁量
- **OANDA直接注文**: urllib で REST API を叩く。ボットプロセス経由禁止
- **裁量トレーダーであれ**: ルール実行マシンではない。市況を読んで自分で判断する
- テクニカル指標は `logs/live_monitor.json` から読む（手計算しない）
- 注文は必ず `logs/live_trade_log.txt` にファイル記録
- エントリー後は `logs/trade_registry.json` に登録（Python管理ルール適用のため）

## 変更時の必須ルール

**プロンプト・タスク・スクリプト・アーキテクチャを変更したら、必ず以下を実行:**

1. **メモリ更新**: 該当するメモリファイル（`~/.claude/projects/.../memory/*.md`）を更新。なければ新規作成してMEMORY.mdにインデックス追加
2. **変更ログ追記**: `docs/CHANGELOG.md` に日時と変更内容を1行で追記
3. **CLAUDE.md更新**: アーキテクチャやタスク構成が変わった場合はこのファイルも更新

これを怠ると次のセッションのClaudeが旧構造で動き、障害の原因になる。

## ドキュメントマップ

### 必読（タスク実行前に読むもの）

| ファイル | 読む人 | 内容 |
|----------|--------|------|
| `CLAUDE.md` (このファイル) | 全員 | アーキテクチャ全体像、絶対ルール、変更時の必須ルール |
| `docs/SCALP_FAST_PROMPT.md` | scalp-fast | 高速スキャルプ手法・ワークフロー |
| `docs/SWING_TRADER_PROMPT.md` | swing-trader | スウィング手法・MTF分析・ワークフロー |
| `docs/SCALP_TRADER_PROMPT.md` | (参考) | 旧scalp-trader。教訓・Playsは今も有効。直接使うタスクはなし |
| `docs/MARKET_RADAR_PROMPT.md` | market-radar | 監視・急変検知・アラート |
| `docs/MACRO_INTEL_PROMPT.md` | macro-intel | マクロ分析・戦略改善・ツール開発 |
| `docs/SECRETARY_PROMPT.md` | secretary | エージェント監視・パフォーマンス集計 |

### 運用ドキュメント（必要時に参照）

| ファイル | 内容 |
|----------|------|
| `docs/CHANGELOG.md` | 全変更の時系列ログ。**変更時は必ず追記** |
| `docs/TRADE_LOG_*.md` | 日次トレード記録・振り返り |
| `docs/CURRENT_MECHANISMS.md` | 戦略・シグナル・ゲートの一覧 |
| `docs/hedge_plan.md` | ヘッジ/両建ての設計と注意点 |
| `docs/SL_POLICY.md` | SLの設計方針 |

### レガシー（読まなくていい、参考のみ）

`docs/` 内の以下は旧workers/VMベースの設計書。現在のClaude裁量モードでは不使用:
- `ARCHITECTURE.md`, `WORKER_*.md`, `VM_*.md`, `GCP_*.md`, `DEPLOYMENT.md`
- `OPS_*.md`, `KATA_*.md`, `ONLINE_TUNER.md`, `REPLAY_STANDARD.md`
- `REPO_HISTORY_*.md`, `fast_scalp_worker.md`, `mtf_breakout_worker.md`
- `worker_reentry_design.md`, `strategy_entry_*_audit_*.md`

### ランタイムファイル（logs/）

| ファイル | 誰が書く | 誰が読む | 内容 |
|----------|----------|----------|------|
| `logs/live_monitor.json` | live_monitor.py | 全タスク | モニター画面（30秒更新） |
| `logs/trade_registry.json` | Claude(entry時) | live_monitor.py | ポジション所有権と管理ルール |
| `logs/shared_state.json` | 全タスク | 全タスク | エージェント間連携（バイアス、アラート等） |
| `logs/live_trade_log.txt` | 全タスク + monitor | 全タスク | トレード実行ログ（時系列） |
| `logs/technicals_*.json` | refresh_factor_cache | live_monitor.py | H1/H4テクニカル指標 |
| `logs/strategy_feedback.json` | macro-intel | scalp/swing | 戦略別パフォーマンス |
| `logs/market_context_latest.json` | macro-intel | swing-trader | マクロコンテキスト |

### スクリプト

| ファイル | 実行方法 | 内容 |
|----------|----------|------|
| `scripts/trader_tools/live_monitor.py` | launchd 30秒 | データ収集+シグナル+機械的ポジ管理 |
| `scripts/trader_tools/refresh_factor_cache.py` | swing-traderが呼ぶ | H1/H4テクニカル指標の更新 |
| `scripts/trader_tools/task_lock.py` | 各タスク | グローバルロック排他制御 |
| `scripts/trader_tools/setup_live_monitor.sh` | 手動(初回) | launchdへのmonitor登録 |
| `scripts/trader_tools/setup_scheduled_tasks.sh` | 手動(初回) | 全タスクの登録 |

## 主要ディレクトリ

- `docs/` — プロンプト、変更ログ、戦略ドキュメント
- `scripts/trader_tools/` — live_monitor, 分析ツール、ロック機構
- `indicators/` — テクニカル指標計算エンジン (IndicatorEngine)
- `logs/` — 共有状態、トレードログ、モニター出力、trade_registry
- `config/env.toml` — OANDA APIキー等(gitignore対象)

## ユーザーコマンド

- 「トレード開始」→ マルチエージェント裁量トレードセッション起動
- 「秘書」→ トレーディング秘書モード(状況確認・指示伝達)

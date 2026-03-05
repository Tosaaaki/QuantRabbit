# Docs Index

## ローカル運用方針（現行）
- 本番運用はローカルV2導線（`scripts/local_v2_stack.sh`）のみ。
- VM/GCP/Cloud Run は現行運用に存在しない前提で、関連コマンドは実行対象外。
- 旧VM/GCP資料は履歴アーカイブとしてのみ保持する。


本リポジトリの主要ドキュメントの位置を整理した索引です。迷ったらここから辿ってください。

## 運用 / Runbook
- `AGENTS.md`: エージェント運用の最重要ルール（要約）
- `docs/AGENT_COLLAB_HUB.md`: タスク開始前の運用チェック手順（本体）
- `docs/OPS_LOCAL_RUNBOOK.md`: ローカル運用手順（local_v2_stack / local LLM lane / ログ配置）
- `docs/OPS_GCP_RUNBOOK.md`: 廃止済みクラウド運用の履歴アーカイブ（実行対象外）
- `docs/VM_OPERATIONS.md`: 廃止済みVM操作手順の履歴アーカイブ（実行対象外）
- `docs/VM_BOOTSTRAP.md`: 廃止済みVM構築手順の履歴アーカイブ（実行対象外）
- `docs/GCP_PLATFORM.md`: 廃止済みGCP基盤情報の履歴アーカイブ（実行対象外）
- `docs/DEPLOYMENT.md`: 廃止済みVM/GCPデプロイ手順の履歴アーカイブ（実行対象外）
- `docs/GCP_DEPLOY_SETUP.md`: 廃止済みgcloudセットアップ手順の履歴アーカイブ（実行対象外）
- `docs/OPS_CURRENT.md`: 時限の運用設定（攻め設定・mask 済み unit など）
- `docs/OPS_SKILLS.md`: 日次運用スキルの使い方
- `docs/LOCAL_LANE_SPLIT.md`: VM本番repoとローカル実売買repoを分離する運用手順
- `scripts/agent_whiteboard.py`: 共有ホワイトボードCLI（local-only / `logs/agent_whiteboard.db`、`auto-session` で開始〜完了を自動記録）
- `workers/common/agent_whiteboard.py`: 共有ホワイトボード永続化ロジック（SQLite）

## アーキテクチャ / 仕様
- `docs/ARCHITECTURE.md`: システム全体フロー、コンポーネント、データスキーマ
- `docs/RISK_AND_EXECUTION.md`: エントリー/EXIT/リスク制御、状態遷移、OANDA マッピング
- `docs/WORKER_REFACTOR_LOG.md`: ワーカー再編（データ供給・制御・ENTRY/EXIT分離）の確定仕様
- `docs/WORKER_ROLE_MATRIX_V2.md`: V2 役割分離方針（データ/制御/戦略/注文/ポジション）
- `docs/SL_POLICY.md`: SL（損切り）決定フロー、環境変数、反映後確認手順（ローカル優先）
- `docs/OBSERVABILITY.md`: データ鮮度、ログ、SLO/アラート、検証パイプライン
- `docs/RANGE_MODE.md`: レンジモード強化とオンラインチューニングの運用要点

## 検証 / チューニング / リプレイ
- `docs/REPLAY_STANDARD.md`: 実運用寄せの標準リプレイ手順
- `docs/ONLINE_TUNER.md`: オンラインチューナの詳細
- `docs/autotune_taskboard.md`: チューニング関連タスク
- `docs/FORECAST.md`: オフライン予測（scikit-learn）と足履歴 backfill/学習手順
- `docs/WFO_OVERFIT_REPORT.md`: WFO/PBO-lite/DSR 近似レポートの運用

## タスク / 変更管理
- `docs/TASKS.md`: リポジトリ全体タスク台帳
- `docs/TRADE_FINDINGS.md`: 改善/敗因の単一台帳（全担当者がここへ追記）

## その他
- `docs/README_GCP_MULTI.md`: 別プロジェクトで動かすための手順書
- `docs/KNOWN_ISSUES.md`: 既知の問題（履歴）

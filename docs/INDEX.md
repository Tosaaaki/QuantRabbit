# Docs Index

本リポジトリの主要ドキュメントの位置を整理した索引です。迷ったらここから辿ってください。

## 運用 / Runbook
- `AGENTS.md`: エージェント運用の最重要ルール（要約）
- `docs/OPS_GCP_RUNBOOK.md`: GCP/VM 運用の統合ランブック（IAP/SSH 復旧や metadata deploy 含む）
- `docs/GCP_PLATFORM.md`: GCP 基盤/IAM/Storage/BQ/Secret の詳細
- `docs/DEPLOYMENT.md`: デプロイ手順（gcloud デフォルト不要）
- `docs/VM_OPERATIONS.md`: VM 操作コマンドまとめ
- `docs/VM_BOOTSTRAP.md`: 新規 VM ブートストラップ
- `docs/GCP_DEPLOY_SETUP.md`: OS Login/IAP/gcloud セットアップ
- `docs/OPS_CURRENT.md`: 時限の運用設定（攻め設定・mask 済み unit など）
- `docs/OPS_SKILLS.md`: 日次運用スキルの使い方

## アーキテクチャ / 仕様
- `docs/ARCHITECTURE.md`: システム全体フロー、コンポーネント、データスキーマ
- `docs/RISK_AND_EXECUTION.md`: エントリー/EXIT/リスク制御、状態遷移、OANDA マッピング
- `docs/WORKER_REFACTOR_LOG.md`: ワーカー再編（データ供給・制御・ENTRY/EXIT分離）の確定仕様
- `docs/SL_POLICY.md`: SL（損切り）決定フロー、環境変数、VMでの確認手順
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

## その他
- `docs/README_GCP_MULTI.md`: 別プロジェクトで動かすための手順書
- `docs/KNOWN_ISSUES.md`: 既知の問題（履歴）

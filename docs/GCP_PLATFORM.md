# GCP Platform Details

このドキュメントは GCP 側の基盤・サービス構成（API/IAM/Storage/BQ/Secret など）をまとめた補助資料です。

## 1. GCP 基盤/VM（API・IAM・ネットワーク）
- 有効化 API: `compute.googleapis.com` / `storage.googleapis.com` / `secretmanager.googleapis.com` / `logging.googleapis.com`
  - 任意: `bigquery.googleapis.com`、Firestore 使用時は Firestore API
- IAM ロール（運用者/SA）
  - OS Login/IAP: `roles/compute.osAdminLogin`, `roles/compute.instanceAdmin.v1`, `roles/iap.tunnelResourceAccessor`（IAP 経由時）
  - VM 実行権限: `roles/storage.objectAdmin`, `roles/logging.logWriter`, `roles/secretmanager.secretAccessor`
  - BQ 利用時: `roles/bigquery.dataEditor` / `roles/bigquery.jobUser`
  - Firestore 利用時: `roles/datastore.user`（読み取りのみなら `roles/datastore.viewer`）
- VM/Compute
  - 例: e2-small / Ubuntu 22.04
  - OS Login 有効 + metadata `enable-oslogin=TRUE`
  - IAP 経由の場合は tcp:22 を 35.235.240.0/20 から許可
  - VM の Service Account を付与（Secret Manager/Storage/Logging/BQ の権限を持たせる）
- 認証 (ADC): VM SA が基本。SA キー運用時は `GOOGLE_APPLICATION_CREDENTIALS` を設定（または `gcloud auth application-default login`）
- VM 上で `gsutil`/`bq` を使う運用（バックアップ/Looker/UI 初期化）では gcloud SDK を導入する
- OS Login 鍵: `scripts/gcloud_doctor.sh -S -G` で鍵生成/登録（TTL 30d）
- ブートストラップ手段
  - Terraform: `infra/terraform/main.tf`（metadata startup script が repo clone + venv 構築。`analysis/gpt_decider.py` / `indicators/calc_core.py` を上書きするため事前確認）
  - 単体VM: `startup_script.sh`（Secret Manager から `/home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env` を生成、OANDA 欠損時は `MOCK_TICK_STREAM=1` を付与して `quantrabbit.service` を起動）
  - 手動: `docs/VM_BOOTSTRAP.md` を参照
- Cloud Logging 連携: `startup_script.sh` が Ops Agent 用の `config.yaml` を書き出す（エージェント導入済みの場合に有効）

## 2. VM サービス/環境ファイル
- main systemd: `ops/systemd/quantrabbit.service`（`EnvironmentFile=/home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env`）
- ワーカー/タイマー: `systemd/*.service` / `systemd/*.timer`（cleanup/autotune/level-map/各戦略）
- まとめて導入: `scripts/install_trading_services.sh --all`
- 個別導入: `scripts/install_trading_services.sh --units <unit>`
- `systemd/quant-level-map.service` は `PROJECT`/`BQ_PROJECT`/`GCS_BUCKET`/`GOOGLE_APPLICATION_CREDENTIALS` が埋め込みのため環境ごとに上書き
- `systemd/quant-autotune.service` は `AUTOTUNE_BQ_TABLE`/`AUTOTUNE_BQ_SETTINGS_TABLE` が固定値のため環境ごとに上書き
- `systemd/quant-autotune-ui.service` は 8088/TCP で待受け。外部公開する場合は FW/IAP 前提で設計
- `systemd/*.service` / `ops/systemd/quantrabbit.service` は `User=tossaki` と `/home/tossaki/QuantRabbit` 前提のため、VM ユーザが異なる場合は編集する
- `ops/env/quant-v2-runtime.env` と `config/env.toml` は内容を一致させる（systemd とアプリの参照元が異なる）
- `scripts/vm.sh` を使う場合は `scripts/vm.env` に PROJECT/ZONE/INSTANCE を固定
- `startup_script.sh` は OANDA の最小キーと `TUNER_*` を `ops/env/quant-v2-runtime.env` に固定で書き込むため、GCS/BQ/リスク系は手動追記が必要

### VM 内ブートストラップ（最小）
```bash
sudo -u <user> -H bash -lc '
  cd /home/<user>
  git clone https://github.com/Tosaaaki/QuantRabbit.git
  cd QuantRabbit
  python3 -m venv .venv && . .venv/bin/activate
  pip install -r requirements.txt
  cp config/env.example.toml config/env.toml
'
```

### cloud-init 注意
- `user-data` の `bootcmd` で長時間タスク（git/pip/サービス起動など）を走らせると、起動シーケンスが止まり SSH が立ち上がらないことがある。
- 重い処理は systemd の oneshot/timer へ逃がし、詰まった場合は metadata の `user-data` を削除して `/var/lib/cloud/instances/*` と `/var/lib/cloud/sem` をクリア後に再起動する。

### `config/env.toml` 最低限
- `gcp_project_id`, `gcp_location`, `GCS_BACKUP_BUCKET`, `ui_bucket_name`, `BQ_PROJECT/BQ_DATASET/BQ_TRADES_TABLE`, `oanda_account_id`, `oanda_token`, `oanda_practice`

## 3. 運用負荷ゼロ化（systemd mask）
- 目的: 不要戦略ユニットの誤起動を完全に遮断する（`ops/env/quant-v2-runtime.env` の誤設定でも起動しない）。
- 方針: 対象ユニットを `/dev/null` へ symlink して mask。バックアップは `/etc/systemd/system/qr_mask_backup_<UTC timestamp>/` に保存。
- 解除: `sudo systemctl unmask <unit>` → `sudo systemctl daemon-reload`。

### 参考コマンド
```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)
backup_dir=/etc/systemd/system/qr_mask_backup_$ts
sudo mkdir -p "$backup_dir"
sudo cp -a /etc/systemd/system/<unit>.service "$backup_dir/" 2>/dev/null || true
sudo ln -sf /dev/null /etc/systemd/system/<unit>.service
sudo systemctl daemon-reload
```

## 4. Storage / GCS
- バケット
  - `GCS_BACKUP_BUCKET`（ログ退避）, `ui_bucket_name`（UI）, 任意で `analytics_bucket_name`
  - env マップ: `GCS_UI_BUCKET`（= `ui_bucket_name`）, `GCS_ANALYTICS_BUCKET`（= `analytics_bucket_name`）
- バックアップ
  - 日次: `utils/backup_to_gcs.sh`
  - 定時: `quant-core-backup.timer`（`/usr/local/bin/qr-gcs-backup-core`）
    - `ops/env/quant-core-backup.env` の load/D-state/mem/swap ガードにより
      高負荷時は自動 skip（トレード導線優先）
    - legacy `/etc/cron.hourly/qr-gcs-backup-core` は無効化して運用する
  - 軽量整理: `scripts/cleanup_logs.sh`（`systemd/cleanup-qr-logs.timer`）
- UI スナップショット: `analytics/gcs_publisher.py` → `ui_state_object_path`（既定 `realtime/ui_state.json`）
- ロットインサイト: `analytics/lot_pattern_analyzer.py` → `analytics/lot_insights.json`（analytics bucket か UI bucket）
- エクスカーションレポート: `scripts/run_excursion_report.sh` → `gs://<ui_bucket>/excursion/`（`systemd/quant-excursion-report.timer`）
- 既定バケット: `utils/backup_to_gcs.sh` は `GCS_BACKUP_BUCKET` 未設定時 `fx-backups` に退避するため必ず指定
- UI の例外パス: `excursion_reports_dir`（ローカル参照先）、`excursion_latest_object_path`（GCS 参照先, 既定 `excursion/latest.txt`）
- `gsutil` がない場合は GCS アップロードがスキップされる（バックアップ/エクスカーション）

## 5. BigQuery / Firestore / UI
- BigQuery 環境変数（`ops/env/quant-v2-runtime.env` 側で指定）
  - 必須: `BQ_PROJECT`（未設定時は `GOOGLE_CLOUD_PROJECT`）, `BQ_DATASET`, `BQ_TRADES_TABLE`, `BQ_LOCATION`
  - 任意: `BQ_MAX_EXPORT`, `BQ_EXPORT_BATCH_SIZE`, `BQ_INSERT_TIMEOUT_SEC`, `BQ_RETRY_TIMEOUT_SEC`, `BQ_RETRY_INITIAL_SEC`, `BQ_RETRY_MAX_SEC`, `BQ_RETRY_MULTIPLIER`, `PIPELINE_DB_READ_TIMEOUT_SEC`, `BQ_CANDLES_TABLE`, `BQ_REALTIME_METRICS_TABLE`, `BQ_RECOMMENDATION_TABLE`, `BQ_STRATEGY_MODEL`
- `quant-bq-sync.service` は `scripts/run_sync_pipeline.py --limit 1200 --bq-interval 300` を既定とし、
  BigQuery 送信は `analytics/bq_exporter.py` 側で `BQ_EXPORT_BATCH_SIZE` 単位に分割する。
  VMで `insertAll` の `RetryError`（SSL EOF 等）が出る場合は、まず `BQ_EXPORT_BATCH_SIZE` と
  `BQ_RETRY_TIMEOUT_SEC` を下げてハング時間を短縮する。
- BQ ML を使う場合は `CREATE MODEL` 権限が必要（`analytics/strategy_optimizer_job.py`）
- 同期パイプライン: `scripts/run_sync_pipeline.py`（`analytics/bq_exporter.py` が dataset/table を自動作成）
- Looker/UI 初期化: `scripts/setup_looker_sources.sh`（UI 用 SA, bucket, dataset, view 作成）
- オートチューン: `AUTOTUNE_BQ_TABLE`, `AUTOTUNE_BQ_SETTINGS_TABLE`（`systemd/quant-autotune.service`）
- Realtime KPI: `analytics/realtime_metrics_job.py`（`BQ_REALTIME_METRICS_TABLE`, `REALTIME_METRICS_LOOKBACK_MIN`, `REALTIME_METRICS_RETENTION_H`）
- 戦略レコメンド: `analytics/strategy_optimizer_job.py`（`BQ_RECOMMENDATION_TABLE`, `BQ_STRATEGY_MODEL`）
- ローソク export: `analytics/candle_export_job.py`（`BQ_CANDLES_TABLE` と OANDA 認証）
- Firestore（任意）: `FIRESTORE_PROJECT`, `FIRESTORE_COLLECTION`, `FIRESTORE_DOCUMENT`, `FIRESTORE_STRATEGY_ENABLE`, `FIRESTORE_STRATEGY_TTL_SEC`, `FIRESTORE_STRATEGY_APPLY_SLTP`
- 戦略スコア書き込み: `scripts/generate_strategy_scores.py`（Firestore へ `strategy_scores/current` を上書き）
- Level map（任意）: `LEVEL_MAP_ENABLE`, `LEVEL_MAP_OBJECT_PATH`, `LEVEL_MAP_TTL_SEC`, `LEVEL_MAP_BUCKET`（未設定なら analytics/UI bucket）
- Cloud Run UI（任意）: `cloudrun/autotune_ui/`（Cloud Build/Run API 有効化、BQ 読み取り権限 + UI bucket 参照権限）
- 参考値: `gcp_pubsub_topic`, `ui_service_account_email` は現行コード未参照（設定しても動作には影響なし）
- Realtime/レコメンド/ローソク/スコアは systemd では未配備。Cloud Scheduler/Cloud Run か cron/systemd で起動する
- `RealtimeMetricsClient` の適用閾値は `REALTIME_METRICS_TTL` / `CONF_POLICY_*` で調整可能

## 6. Secret Manager / 認証情報
- Secret 名は `config/env.toml` のキー名と同一（例: `oanda_token`, `oanda_account_id`）
- 推奨シークレット（最低限）
  - `oanda_token`, `oanda_account_id`
- Secret Manager の参照プロジェクトは `GCP_PROJECT` / `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_PROJECT_NUMBER` で決定（未設定だと `quantrabbit`）
- `scripts/refresh_env_from_gcp.py` で Secret/環境変数から `config/env.toml` を生成できる
- Secret Manager を使わない場合は `DISABLE_GCP_SECRET_MANAGER=1`

## 7. LLM
- LLM は現行運用で使用しない。再導入時に設計を作り直す。

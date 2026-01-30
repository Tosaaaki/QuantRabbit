# GCP マルチ環境 手順書

「別プロジェクトの GCP で QuantRabbit を動かす」ための最短手順です。新規 VM 作成の流れは `docs/VM_BOOTSTRAP.md` に集約しています。詳細は `AGENTS.md`, `docs/GCP_DEPLOY_SETUP.md`, `README.md` を参照してください。

## 0. 前提ロール
- デプロイ/運用: `roles/compute.osAdminLogin`, `roles/compute.instanceAdmin.v1`, `roles/iap.tunnelResourceAccessor`（IAP 経由なら）
- ログ/バックアップ: `roles/storage.objectAdmin`, `roles/logging.logWriter`
- BigQuery 同期: `roles/bigquery.dataEditor`, `roles/bigquery.jobUser`
- UI/Looker 用 SA: `roles/storage.objectViewer`, `roles/bigquery.dataViewer`

## 1. プロジェクト初期セットアップ（1 回）
1) API 有効化: Compute / Storage / BigQuery。  
2) バケット作成:  
   - バックアップ用: 例 `gs://fx-backups-<proj>` → env `GCS_BACKUP_BUCKET`  
   - UI 用: 例 `gs://fx-ui-realtime-<proj>` → env `ui_bucket_name`  
   - Terraform state 用（使う場合）: 例 `gs://qr-tf-state-<proj>`  
3) BigQuery: データセット `quantrabbit`（任意名で可）、テーブル `trades_raw`。`scripts/setup_looker_sources.sh` を使うとビューもまとめて作成。  
4) SA 作成（例）:  
   - `qr-deployer`（Compute/IAP/Storage/Logging/BQ 権限）  
   - `qr-ui`（objectViewer + bq dataViewer）  
   キーを使う場合は VM で `GOOGLE_APPLICATION_CREDENTIALS` を設定。

## 2. VM 準備
- Terraform 利用: `infra/terraform/main.tf` の backend バケットと `project_id/region` を新プロジェクト用に変更 → `terraform init -backend-config=bucket=<tf-state-bucket>` → `terraform apply`。  
- 手動なら e2-small/Ubuntu22.04 で OS Login 有効、IAP ルール開放。インスタンス metadata に `enable-oslogin=TRUE`。

## 3. リポ/設定ブートストラップ（VM 内）
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
- `config/env.toml` を埋める（最低限）: `gcp_project_id`, `gcp_location`, `GCS_BACKUP_BUCKET`, `ui_bucket_name`, `BQ_PROJECT/BQ_DATASET/BQ_TRADES_TABLE`, `OPENAI_API_KEY`, `OPENAI_MODEL_DECIDER`, `OANDA_ACCOUNT`, `OANDA_TOKEN`, `oanda_practice`。  
- SA キー運用なら `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`。

## 4. デプロイ・起動
```bash
scripts/gcloud_doctor.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -t -k ~/.ssh/gcp_oslogin_qr -c
scripts/deploy_to_vm.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -i -t -k ~/.ssh/gcp_oslogin_qr \
  [-K <SA_KEY> -A <SA_EMAIL>]
```
- `quantrabbit.service` が稼働していれば再起動されます。初回は `config/env.toml` が必要。
- ログ確認: `scripts/tail_vm_logs.sh -c "sudo journalctl -u quantrabbit.service -f"`。

## 5. 定常ジョブ
- バックアップ（GCS）: `utils/backup_to_gcs.sh` を cron/systemd で毎日。`GCS_BACKUP_BUCKET` が必要。  
- BigQuery 同期: `scripts/run_sync_pipeline.py --interval 180 --limit 2000` を常駐。状態ファイルは `logs/bq_sync_state.json`。  
- オートチューナ（任意）: `scripts/run_online_tuner.py` を 5–15 分間隔で。既定はシャドウモード。

## 6. ワーカー/構成の要点
- メイン 60 秒ループ: `main.py`（regime/focus/GPT → strategy → risk_guard → order_manager）。  
- ワーカー: core は廃止済み。各戦略ごとの専用ワーカー（`workers/<strategy>/`）を `*_ENABLED` と systemd で制御。  
- グループスイッチ: `SCALP_WORKERS_ENABLED`, `MICRO_WORKERS_ENABLED`, `MACRO_WORKERS_ENABLED`, `MICRO_DELEGATE_TO_WORKER`（専用 micro_* へ委譲）。

## 7. 動作確認クイックチェック
- systemd: `sudo systemctl status quantrabbit.service`  
- ログ tail: `journalctl -u quantrabbit.service -n 200 -f`  
- 今日の損益: `sqlite3 logs/trades.db "select date(close_time), count(*), round(sum(pl_pips),2) from trades where date(close_time)=date('now') group by 1;"`  
- BigQuery 反映確認: `bq head -n 5 <PROJECT>:<DATASET>.trades_raw`

## 8. 参考
- OS Login/IAP/Doctor 詳細: `docs/GCP_DEPLOY_SETUP.md`
- 安全装置・リスク基準: `AGENTS.md` 6.x
- ダッシュボード/Looker: `scripts/setup_looker_sources.sh`, `cloudrun/autotune_ui/`

# GCP マルチ環境 手順書（廃止済みアーカイブ）

## 現行運用
- 本番運用はローカルV2導線（`scripts/local_v2_stack.sh`）のみ。
- GCP マルチ環境運用は現行の対象外。
- 本書は過去の環境構築履歴としてのみ保持する。

> 以降の章は履歴記録であり、現行運用で実行しないこと。

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
- `config/env.toml` を埋める（最低限）: `gcp_project_id`, `gcp_location`, `GCS_BACKUP_BUCKET`, `ui_bucket_name`, `BQ_PROJECT/BQ_DATASET/BQ_TRADES_TABLE`, `OANDA_ACCOUNT`, `OANDA_TOKEN`, `oanda_practice`。  
- SA キー運用なら `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`。

## 4. デプロイ・起動
```bash
scripts/gcloud_doctor.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -t -k ~/.ssh/gcp_oslogin_qr -c
scripts/deploy_to_vm.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -i -t -k ~/.ssh/gcp_oslogin_qr \
  [-K <SA_KEY> -A <SA_EMAIL>]
```
- 旧 `quantrabbit.service` 運用が残っている VM では本稿の再起動対象になり得ます。初回は `config/env.toml` が必要。  
- ログ確認: `scripts/tail_vm_logs.sh -c "sudo journalctl -u quantrabbit.service -f"`（補助用途）。

## 5. 定常ジョブ
- バックアップ（GCS）: `utils/backup_to_gcs.sh` を cron/systemd で毎日。`GCS_BACKUP_BUCKET` が必要。  
- BigQuery 同期: `scripts/run_sync_pipeline.py --interval 180 --limit 2000` を常駐。状態ファイルは `logs/bq_sync_state.json`。  
- オートチューナ（任意）: `scripts/run_online_tuner.py` を 5–15 分間隔で。既定はシャドウモード。

## 6. ワーカー/構成の要点
- メイン 60 秒ループ: `main.py` ベースの補助構成（主に旧/移行時）。  
- ワーカー: core は廃止済み。各戦略ごとの専用ワーカー（`workers/<strategy>/`）を `*_ENABLED` と systemd で制御。  
- グループスイッチ: `SCALP_WORKERS_ENABLED`, `MICRO_WORKERS_ENABLED`, `MACRO_WORKERS_ENABLED`, `MICRO_DELEGATE_TO_WORKER`（専用 micro_* へ委譲）。

## 7. 動作確認クイックチェック
- systemd: `sudo systemctl status quant-order-manager.service` または `quant-strategy-control.service`  
- ログ tail: `journalctl -u quant-order-manager.service -n 200 -f`  
- 今日の損益: `sqlite3 logs/trades.db "select date(close_time), count(*), round(sum(pl_pips),2) from trades where date(close_time)=date('now') group by 1;"`  
- BigQuery 反映確認: `bq head -n 5 <PROJECT>:<DATASET>.trades_raw`

## 8. 参考
- OS Login/IAP/Doctor 詳細: `docs/GCP_DEPLOY_SETUP.md`
- 安全装置・リスク基準: `docs/RISK_AND_EXECUTION.md`
- ダッシュボード/Looker: `scripts/setup_looker_sources.sh`, `cloudrun/autotune_ui/`

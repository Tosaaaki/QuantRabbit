# VM Bootstrap (New Instance)

新しい VM を作っても「すぐ動く」状態にするための最短手順です。既存 VM の運用は `docs/DEPLOYMENT.md` / `docs/VM_OPERATIONS.md` を参照してください。

## 0) 先に決めること
- **ユーザー**: 既定は `tossaki`（`startup_script.sh` もこれを前提に準備）
  - 変更したい場合は `QR_USER=<user>` を付けて実行すると systemd も自動で上書きされます。
- **Secret Manager を使うか**: 使う場合は `oanda_token` などを事前に作成。
- **IAP/OS Login**: 既定は OS Login + IAP 前提（外部 IP なしでも運用可能）。

## 1) 必須 IAM / API（新プロジェクト時）
- API: `compute.googleapis.com`, `storage.googleapis.com`, `secretmanager.googleapis.com`, `logging.googleapis.com`
- IAM（運用者）:
  - `roles/compute.osAdminLogin`
  - `roles/compute.instanceAdmin.v1`
  - `roles/iap.tunnelResourceAccessor`（IAP 利用時）

## 2) シークレット / 環境変数
### A. Secret Manager を使う場合（推奨）
最低限の Secret を作成:
- `oanda_token`
- `oanda_account_id`
- `oanda_practice`

運用で必要なら追加:
- `gcp_project_id`, `ui_bucket_name`, `GCS_BACKUP_BUCKET`, `BQ_PROJECT`, `BQ_DATASET`, `BQ_TRADES_TABLE`

`startup_script.sh` は `/etc/quantrabbit.env` を作成し、`scripts/refresh_env_from_gcp.py` で `config/env.toml` も同期します。

### B. Secret Manager を使わない場合
`/etc/quantrabbit.env` に最低限を手で書く（または `deploy_via_metadata.sh -e` で注入）:

```bash
OANDA_TOKEN=...
OANDA_ACCOUNT=...
OANDA_PRACTICE=true
GCS_BACKUP_BUCKET=fx-backups-<proj>
ui_bucket_name=fx-ui-realtime-<proj>
BQ_PROJECT=<proj>
BQ_DATASET=quantrabbit
BQ_TRADES_TABLE=trades_raw
```

書き換え後は `scripts/refresh_env_from_gcp.py` を一度実行して `config/env.toml` を同期してください。

## 3) VM 作成方法（2 ルート）
### 3-A) Terraform（推奨）
1. `infra/terraform/terraform.tfvars` を新プロジェクトに合わせて更新
2. backend bucket を用意して `terraform init`
3. `terraform apply`

`infra/terraform/main.tf` は `startup_script.sh` をそのまま実行する構成です。

### 3-B) 手動作成 + startup_script.sh
1. Ubuntu 22.04 / e2-small を作成
2. metadata: `enable-oslogin=TRUE`
3. `startup_script.sh` を startup script として登録

```bash
# 例: 既存 VM に metadata で反映
scripts/gcloud_doctor.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -t -c

gcloud compute instances add-metadata <INSTANCE> \
  --project <PROJECT> --zone <ZONE> \
  --metadata enable-oslogin=TRUE \
  --metadata-from-file startup-script=./startup_script.sh
```

`QR_USER` を変える場合は `startup_script.sh` 実行時に `QR_USER=<user>` を付与してください。

## 4) 起動後の確認
```bash
# systemd
sudo systemctl status quantrabbit.service

# ログ
journalctl -u quantrabbit.service -n 200 --no-pager
```

`/etc/quantrabbit.env` と `config/env.toml` が一致していることを確認。

## 5) デプロイ / 運用開始
```bash
scripts/gcloud_doctor.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -t -c
scripts/vm.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> deploy -b main -i --restart quantrabbit.service -t
```

ワーカー運用の場合は必要な unit を入れて起動:
```bash
scripts/install_trading_services.sh --all
```

## 6) うまく起動しない時
- OS Login/IAP: `docs/GCP_DEPLOY_SETUP.md` を確認
- IAP/SSH が不安定: `scripts/deploy_via_metadata.sh` を使用
- ヘルス確認: `scripts/tail_vm_logs.sh` または `docs/VM_OPERATIONS.md`

# GCP Deploy Setup – gcloud 前提のゼロトラブル手順

gcloud 未導入・設定不備・IAP/OS Login/Compute API の典型エラーを潰すための実践手順です。初回セットアップとトラブル時の再診断に使用します。新規 VM 作成の流れは `docs/VM_BOOTSTRAP.md` を参照してください。

## 1. SDK インストール

- macOS: `brew install --cask google-cloud-sdk`（または `brew install google-cloud-sdk`）
- Debian/Ubuntu: `scripts/install_gcloud.sh`（apt 経由）
- 代替: 公式手順 https://cloud.google.com/sdk/docs/install

コマンド例（共通）

```bash
scripts/install_gcloud.sh
gcloud init
gcloud auth login
```

## 2. 基本設定

```bash
gcloud config set project <PROJECT_ID>
gcloud config set compute/zone asia-northeast1-a
gcloud services enable compute.googleapis.com --project <PROJECT_ID>
```

## 3. OS Login / IAP 前提

- IAM 付与（自分のユーザに）
  - `roles/compute.osLogin`
  - IAP 経由なら `roles/iap.tunnelResourceAccessor`
- インスタンス/プロジェクト メタデータ: `enable-oslogin=TRUE`
- OS Login 鍵登録（30 日 TTL）

```bash
ssh-keygen -t ed25519 -f ~/.ssh/gcp_oslogin_qr -N '' -C 'oslogin-quantrabbit'
gcloud compute os-login ssh-keys add --key-file ~/.ssh/gcp_oslogin_qr.pub --ttl 30d
gcloud compute os-login describe-profile
```

## 4. 事前健診（Doctor）

```bash
# インスタンス名/ゾーンは必要に応じて変更
scripts/gcloud_doctor.sh -p <PROJECT_ID> -z asia-northeast1-a -m fx-trader-vm -c
# IAP 経由の場合
scripts/gcloud_doctor.sh -p <PROJECT_ID> -z asia-northeast1-a -m fx-trader-vm -t -c -k ~/.ssh/gcp_oslogin_qr
```

主なチェック内容
- gcloud インストール/ログイン/プロジェクト設定
- Compute API 有効化
- インスタンス存在確認
- OS Login プロファイル検査
- （任意）SSH 接続テスト（IAP/鍵を含む）

## 5. デプロイ（推奨フロー）

```bash
# 依存アップデート込み、IAP 経由の例
scripts/deploy_to_vm.sh -i -t -k ~/.ssh/gcp_oslogin_qr -p <PROJECT_ID>

# ログ確認
gcloud compute ssh fx-trader-vm \
  --project <PROJECT_ID> --zone asia-northeast1-a \
  --tunnel-through-iap \
  --ssh-key-file ~/.ssh/gcp_oslogin_qr \
  --command 'journalctl -u quantrabbit.service -f'
```

## 6. 典型エラーと対処

- gcloud が見つからない: `scripts/install_gcloud.sh` 実行 → 新しいシェルで `gcloud init`
- `Permission denied (publickey)`: OS Login 有効化/IAM 権限/鍵 TTL を確認、`--ssh-key-file` を明示
- `Compute API not enabled`: `gcloud services enable compute.googleapis.com --project <PROJECT_ID>`
- IAP 経由で失敗: `roles/iap.tunnelResourceAccessor` 付与、`--tunnel-through-iap` とファイアウォールの許可
- インスタンスが見つからない: 名前/ゾーン/プロジェクトを確認（`gcloud compute instances list`）

### 6.1 IAP 鍵認証の再発防止手順（推奨）

以下を 1 コマンドで実行し、1回で収束しない場合は metadata デプロイへ切り替えます。

```bash
scripts/recover_iap_ssh_auth.sh \
  -p <PROJECT_ID> \
  -z asia-northeast1-a \
  -m fx-trader-vm \
  -k ~/.ssh/gcp_oslogin_qr \
  -r 6 \
  -s 3
```

成功したら通常のデプロイに戻ります。失敗時は `docs/OPS_GCP_RUNBOOK.md` の
「IAP 鍵認証が繰り返し失敗する時の標準手順（最優先）」を実行してから再投入してください。

## 7. 補足

- `scripts/deploy_to_vm.sh` は内部で `scripts/gcloud_doctor.sh` を呼び出し、前提が崩れていれば早期に失敗・案内します。
- OS Login を使わない暫定運用は非推奨（セキュリティ/運用の一貫性が下がります）。

---

## 8. ヘッドレス運用（アクティブなユーザーアカウントなしでも操作）

サービスアカウント（SA）を用いることで、ユーザーのアクティブアカウントがない環境でも GCP/VM 操作が可能です。

1) SA の作成と権限（例）

```bash
gcloud iam service-accounts create qr-deployer --display-name "QuantRabbit Deployer"
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member serviceAccount:qr-deployer@<PROJECT_ID>.iam.gserviceaccount.com \
  --role roles/compute.osAdminLogin
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member serviceAccount:qr-deployer@<PROJECT_ID>.iam.gserviceaccount.com \
  --role roles/compute.instanceAdmin.v1
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member serviceAccount:qr-deployer@<PROJECT_ID>.iam.gserviceaccount.com \
  --role roles/iap.tunnelResourceAccessor   # IAP を使う場合
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member serviceAccount:qr-deployer@<PROJECT_ID>.iam.gserviceaccount.com \
  --role roles/viewer
```

2) SA キーの作成（最小限の扱いに注意）

```bash
gcloud iam service-accounts keys create ~/.gcp/qr-deployer.json \
  --iam-account=qr-deployer@<PROJECT_ID>.iam.gserviceaccount.com
```

3) Doctor/Deploy での利用

```bash
# Doctor がアクティブアカウント不在でも SA キーで自動有効化
scripts/gcloud_doctor.sh -p <PROJECT_ID> -z asia-northeast1-a -m fx-trader-vm -K ~/.gcp/qr-deployer.json -t -c

# デプロイ時に SA キーを渡す（必要ならインパーソネート指定も可）
scripts/deploy_to_vm.sh -p <PROJECT_ID> -t -k ~/.ssh/gcp_oslogin_qr \
  -K ~/.gcp/qr-deployer.json \
  -A qr-deployer@<PROJECT_ID>.iam.gserviceaccount.com

4) OS Login 鍵（SA 用）を登録（必要時）

```bash
# SA をインパーソネートしつつ、OS Login に公開鍵を登録（鍵が無ければ -G で作成）
scripts/gcloud_doctor.sh -p <PROJECT_ID> -A qr-deployer@<PROJECT_ID>.iam.gserviceaccount.com \
  -S -G -k ~/.ssh/gcp_oslogin_qr.pub
```
```

注記
- SA で OS Login を使うには `roles/compute.osLogin`（または osAdminLogin）が必要です。インスタンス/プロジェクトに `enable-oslogin=TRUE` を設定してください。
- キーファイルは厳重に管理し、不要になったら無効化・削除し、可能ならユーザーからのインパーソネートに切り替えてキーレス運用を検討してください。

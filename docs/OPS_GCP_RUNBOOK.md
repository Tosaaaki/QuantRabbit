# GCP / VM Operations Runbook

このドキュメントは GCP/VM 運用に関する統合ランブックです。詳細なデプロイ手順は `docs/DEPLOYMENT.md`、OS Login/IAP のセットアップは `docs/GCP_DEPLOY_SETUP.md` を優先参照してください。

## 1. デプロイ / GCP アクセス
- 原則 OS Login + IAP。
- `scripts/gcloud_doctor.sh` で前提検診（Compute API 有効化 / OS Login 鍵登録 / IAP 確認）→ `scripts/deploy_to_vm.sh` でデプロイ。

### クイックコマンド（proj/zone/inst は適宜置換）
```bash
# Doctor（一括検診 + 鍵登録 + OS Login メタデータ競合是正）
scripts/gcloud_doctor.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -E -S -G -O -t -k ~/.ssh/gcp_oslogin_qr -c

# デプロイ（venv 依存更新付き/IAP）
scripts/deploy_to_vm.sh -i -t -k ~/.ssh/gcp_oslogin_qr -p quantrabbit

# ログ追尾
gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a \
  --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_qr \
  --command 'journalctl -u quantrabbit.service -f'
```

- `gcloud_doctor.sh` の `-O` は instance metadata を `enable-oslogin=TRUE, block-project-ssh-keys=TRUE` に固定して、project metadata との競合を除去する。
- `gcloud_doctor.sh` の `-T` は `-S` 実行時の OS Login 鍵 TTL を制御する（既定 `none`=無期限。例: `-T 30d`）。

### IAP 鍵認証が繰り返し失敗する時の標準手順（最優先）
```bash
scripts/recover_iap_ssh_auth.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -k ~/.ssh/gcp_oslogin_quantrabbit -r 6 -s 3
```

- 成功時はそのまま通常デプロイへ戻る:
  - `scripts/deploy_to_vm.sh -i -t -k ~/.ssh/gcp_oslogin_quantrabbit -p quantrabbit`
- 失敗時は fallback で metadata デプロイ:
  - `scripts/deploy_via_metadata.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -b main -i`

再試行ロジック:
- `scripts/gcloud_doctor.sh` を IAP + OS Login 強制（`-t -S -G -O`）で 6回繰り返し実行
- `Permission denied (publickey)` を受けた場合は鍵再登録を行い再試行

### フォールバック（vm.sh が失敗する場合の直書き）
1) 
```bash
gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a \
  --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit \
  --command "sudo -u tossaki -H bash -lc 'cd /home/tossaki/QuantRabbit && git fetch --all -q || true && git checkout -q main || git checkout -b main origin/main || true && git pull --ff-only && if [ -d .venv ]; then source .venv/bin/activate && pip install -r requirements.txt; fi'""
```
2)
```bash
gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a \
  --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit \
  --command "sudo systemctl daemon-reload && sudo systemctl restart quantrabbit.service && sudo systemctl status --no-pager -l quantrabbit.service || true"
```

## 2. IAP/SSH 不安定時の無 SSH 反映（metadata 経由）
- 目的: `failed to connect to backend` 等で SSH/IAP が落ちても反映を止めない。

### 反映
```bash
scripts/deploy_via_metadata.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -b main -i
```

### 健康レポート取得
- `-r` を付ける（serial に status/trades/signal を出力）。

### 仕組み
- `startup-script` に `deploy_id` を埋め込み、`/var/lib/quantrabbit/deploy_id` で重複実行を抑止。

### 後片付け（任意）
```bash
gcloud compute instances remove-metadata fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --keys startup-script
```

## 3. IAP/SSH 落ちが続く場合の予防
- 目的: SSH サービス/guest agent 停止で port 22 が閉じる事象を自動復旧。
- 監視: `systemd/quant-ssh-watchdog.timer`（1分間隔）→ `scripts/ssh_watchdog.sh`
- 導入:
```bash
sudo bash scripts/install_trading_services.sh --units "quant-ssh-watchdog.service quant-ssh-watchdog.timer"
```
- 備考: `/home/tossaki/QuantRabbit` 前提。ユーザが異なる場合は unit を編集。

## 4. 情報不足対策（無 SSH でも状態確認）
- 目的: trades/signal の最終時刻を GCS に書き出し、外部から即確認。
- 監視: `systemd/quant-health-snapshot.timer`（1分間隔）→ `scripts/publish_health_snapshot.py`
- 出力先: `ui_bucket_name` 優先、未設定なら `GCS_BACKUP_BUCKET`（`HEALTH_OBJECT_PATH` で上書き）
- アップロード: `google-cloud-storage` → `gcloud/gsutil` → metadata REST の順にフォールバック。
- ローカル集約（IAP 不要）:
```bash
scripts/collect_gcs_status.py --ui-bucket fx-ui-realtime --backup-bucket quantrabbit-logs --host fx-trader-vm --project quantrabbit
```
- 取得例: `gcloud storage cat gs://<bucket>/<object>`
- ローカル: `/home/tossaki/QuantRabbit/logs/health_snapshot.json` にも保存（バックアップ対象）。

- 監査の自動化（V2運用監視）:
```bash
sudo bash scripts/install_trading_services.sh --repo /home/tossaki/QuantRabbit \
  --units quant-v2-audit.service quant-v2-audit.timer
sudo systemctl enable --now quant-v2-audit.timer
```
- 監査結果:
  - `journalctl -u quant-v2-audit.service -n 200 --no-pager`
  - `cat /home/tossaki/QuantRabbit/logs/ops_v2_audit_latest.json`

- `EnvironmentFile` の同一パス重複を即時是正（drop-in のみ編集）:
```bash
sudo -u tossaki -H python3 /home/tossaki/QuantRabbit/scripts/dedupe_systemd_envfiles.py --apply --services "quant-*.service"
sudo systemctl daemon-reload
```

- ops系 unit から legacy `/etc/quantrabbit.env` 注入を除去:
```bash
sudo python3 /home/tossaki/QuantRabbit/scripts/dedupe_systemd_envfiles.py --apply \
  --services "quant-online-tuner.service" "quant-autotune-ui.service" "quant-bq-sync.service" \
             "quant-health-snapshot.service" "quant-level-map.service" \
             "quant-strategy-optimizer.service" "quant-ui-snapshot.service" \
  --remove-envfile "/etc/quantrabbit.env"
sudo systemctl daemon-reload
```

## 5. OS Login 権限不足・VM 運用ルール
- OS Login 権限不足時は `roles/compute.osAdminLogin` を付与（検証: `sudo -n true && echo SUDO_OK`）。
- 本番 VM `fx-trader-vm` は原則 `main` ブランチ稼働。
- スタッシュ/未コミットはブランチ切替前に解消。
- VM 削除禁止。再起動やブランチ切替で代替し、`gcloud compute instances delete` 等には触れない。

## 6. IAP/SSH 不調時の反映/確認（代替フロー）
- 反映:
```bash
scripts/deploy_via_metadata.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -b main -i -e local/vm_env_overrides.env
```
- GitHub 到達不安定時:
```bash
scripts/deploy_bundle_to_gcs.sh -b quantrabbit-logs
scripts/deploy_via_metadata.sh ... -g gs://quantrabbit-logs/deploy/qr_bundle_*.tar.gz
```
- 確認:
```bash
gcloud compute instances get-serial-port-output fx-trader-vm --zone=asia-northeast1-a --project=quantrabbit --port=1 | rg 'startup-script|deploy_id'
```
- 反映マーカー: `realtime/startup_<hostname>_<deploy_id>.json` が UI bucket に生成される（無 SSH で deploy 実行確認）。
- SSH 自己回復: `quant-ssh-watchdog.timer` が `ssh/sshd` と `google-guest-agent` を 1 分ごとに再起動監視。
- ヘルス可視化: `quant-health-snapshot.timer` が `/home/tossaki/QuantRabbit/logs/health_snapshot.json` を 1 分ごとに更新し、`ui_bucket_name`（未設定なら `GCS_BACKUP_BUCKET`）の `realtime/health_<hostname>.json` へ送信。
- UI 可視化: `quant-ui-snapshot.timer` が `realtime/ui_state.json` を 1 分ごとに更新（orders/signals/healthbeat を metrics に同梱）。
- まだ復帰しない場合:
```bash
gcloud compute instances add-metadata fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --metadata-from-file user-data=local/cloudinit_ssh_repair.yaml
```

## 7. 参照先（基盤/サービス詳細）
- GCP 基盤・IAM・Storage・BigQuery・Secret などの詳細は `docs/GCP_PLATFORM.md` を参照。
- セットアップ手順は `docs/GCP_DEPLOY_SETUP.md` と `docs/README_GCP_MULTI.md` を参照。
- VM 操作コマンドは `docs/VM_OPERATIONS.md` を参照。
- 新規 VM ブートストラップは `docs/VM_BOOTSTRAP.md` を参照。

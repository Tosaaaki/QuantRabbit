デプロイ手順（更新版 / gcloud デフォルト不要）

この手順は gcloud の「アクティブなアカウント/プロジェクト」を設定せずに、その都度 `--project`/`--zone`/`--account` を明示して実行します。新規 VM 作成は `docs/VM_BOOTSTRAP.md` を参照してください。

前提
- gcloud が利用可能で、必要なアカウントで認証済み（必要なら `gcloud auth login <ACCOUNT>`）
- OS Login 権限あり（IAP 経由の場合は `roles/iap.tunnelResourceAccessor`）
- VM 側のリポジトリは `~/QuantRabbit`（変更したい場合は `-d` で指定）

1) 便利ファイルの準備（初回のみおすすめ）
- `scripts/vm.env.example` を `scripts/vm.env` にコピーし、値を編集
  - 例: `PROJECT=quantrabbit`, `ZONE=asia-northeast1-a`, `INSTANCE=fx-trader-vm`
  - 例: `KEYFILE=$HOME/.ssh/gcp_oslogin_quantrabbit`, `USE_IAP=1`, `REMOTE_DIR=/home/tossaki/QuantRabbit`
- Makefile のショートカットが使えるようになります

2) デプロイ方法（make 経由・推奨）
- 最新をデプロイして再起動
```
make vm-deploy BRANCH=main
```
- ログ追尾
```
make vm-tail VM_SERVICE=quantrabbit.service
```

3) デプロイ方法（直接コマンド）
- ブランチ指定で pull、`.venv` があれば依存をインストール、systemd 再起動（IAP 経由）
```
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  deploy -b main -i --restart quantrabbit.service -t
```

4) VM ログ/DB の取得
- まとめて取得（DB と `logs/replay`）
```
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  pull-logs -r /home/tossaki/QuantRabbit/logs -o ./remote_logs -t
```
- その場で SQL 集計
```
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  sql -f /home/tossaki/QuantRabbit/logs/trades.db \
  -q "SELECT DATE(close_time), COUNT(*), ROUND(SUM(pl_pips),2) FROM trades WHERE DATE(close_time)=DATE('now') GROUP BY 1;" -t
```

5) Cloud Run（autotune-ui）反映手順
- サービス: `autotune-ui` / リージョン: `asia-northeast1` / プロジェクト: `quantrabbit`
- イメージ: `gcr.io/quantrabbit/autotune-ui`

1. イメージを再ビルド（Cloud Build）
```bash
cd /Users/tossaki/Documents/App/QuantRabbit
gcloud builds submit --project=quantrabbit --config=cloudrun/autotune_ui/cloudbuild.yaml .
```

2. デプロイ
```bash
gcloud run deploy autotune-ui \
  --project=quantrabbit \
  --region=asia-northeast1 \
  --image=gcr.io/quantrabbit/autotune-ui \
  --set-env-vars "GCS_UI_BUCKET=fx-ui-realtime,UI_STATE_OBJECT_PATH=realtime/ui_state.json" \
  --platform managed \
  --allow-unauthenticated
```

3. ops 制御を有効化する場合（推奨は Secret Manager）
```bash
gcloud run deploy autotune-ui \
  --project=quantrabbit \
  --region=asia-northeast1 \
  --image=gcr.io/quantrabbit/autotune-ui \
  --set-env-vars "GCS_UI_BUCKET=fx-ui-realtime,UI_STATE_OBJECT_PATH=realtime/ui_state.json" \
  --set-secrets "ui_ops_token=ui-ops-token:latest" \
  --platform managed \
  --allow-unauthenticated
```

4. 反映確認（必須）
```bash
gcloud run services describe autotune-ui \
  --project=quantrabbit \
  --region=asia-northeast1 \
  --format='value(status.url,status.latestReadyRevisionName)'

curl -s https://autotune-ui-duqk34guwq-an.a.run.app/dashboard?tab=summary | rg -n "システム構成図|QuantRabbit UI"
```

5-α) dashboard 変更分（今回: `templates/autotune/dashboard.html` / `apps/autotune_ui.py`）を確実に反映する手順
- Cloud Build + デプロイを同一コミット単位で実行:
```bash
cd /Users/tossaki/Documents/App/QuantRabbit
git add templates/autotune/dashboard.html apps/autotune_ui.py
git commit -m "fix: dashboard metrics diagnostics and architecture tab"

gcloud builds submit --project=quantrabbit --config=cloudrun/autotune_ui/cloudbuild.yaml .
gcloud run deploy autotune-ui \
  --project=quantrabbit \
  --region=asia-northeast1 \
  --image=gcr.io/quantrabbit/autotune-ui \
  --platform managed \
  --allow-unauthenticated
```
- 反映確認（重要）
```bash
URL="https://autotune-ui-duqk34guwq-an.a.run.app"
curl -s "${URL}/dashboard?tab=summary&t=$(date +%s%3N)" | rg -n "システム構成図|tab=architecture|snapshot.metrics|データ取得ログ|更新"
curl -s "${URL}/dashboard?tab=ops&t=$(date +%s%3N)" | rg -n "システム構成図|tab=architecture|データソース|metrics"
```
- `gcloud run services describe autotune-ui --project=quantrabbit --region=asia-northeast1` で `latestReadyRevisionName` と `traffic` が新リビジョンを指しているかを確認し、ログで起動エラーがないことを確認。

- `ui_ops_token` 未設定時は、`/dashboard?tab=ops` では
  `ui_ops_token が未設定のため戦略制御は行えません。` が表示される動作は正常です。

### 補足: 今回の反映で採用した確定ルート
- `dashboard` 改修後、必ず `--config=cloudrun/autotune_ui/cloudbuild.yaml` で `gcr.io/quantrabbit/autotune-ui` を再ビルドする。
- デプロイは 5) の `gcloud run deploy ...` で新リビジョンを作る。
- 直後に `latestReadyRevisionName` と `traffic` が同一の新リビジョンを指していることを確認する。
  - 旧リビジョン（例: `...-00069`）が残っているだけなら反映されていない状態。
- 起動失敗が出る場合は以下を確認する。
  ```bash
  gcloud run services logs read autotune-ui --region=asia-northeast1 --limit=50
  ```
  - 典型的原因: 起動時 import 失敗（`ModuleNotFoundError` / `HealthCheckContainerError`）
- 動作確認 URL:
  - サービス公開 URL: https://autotune-ui-duqk34guwq-an.a.run.app
  - 画面確認: `/dashboard?tab=ops&t=<unix_millis>`

### dashboard 反映専用チェック（今回の症状: 数字が `-` のまま）
- 反映後、まず `ソース gcs` を目視確認し、`metrics` が空でないことを確認する。
```bash
URL="https://autotune-ui-duqk34guwq-an.a.run.app"
curl -s "${URL}/dashboard?tab=summary&t=$(date +%s%3N)" | rg -n "ソース gcs|metrics missing|status-pill warn|データ取得ログ"
curl -s "${URL}/dashboard?tab=history&t=$(date +%s%3N)" | rg -n "更新|日次損益|前日比|総取引|stat-value -|ソース gcs"
```
- システム構成図は新タブで表示されることを確認する。
```bash
curl -s "${URL}/dashboard?tab=architecture&t=$(date +%s%3N)" | rg -n "システム構成図|データ供給|戦略実行"
```
- API 側も同時確認する（`/api/snapshot` が `gcs` 由来になるか）。
```bash
curl -s "${URL}/api/snapshot" | rg -n '"snapshot_source"|"metrics"'
```

6) 旧スクリプトとの互換
- `scripts/deploy_to_vm.sh` はラッパとして `scripts/vm.sh` を呼び出すよう置換済みです
- 旧呼び出しでも動作しますが、今後は `scripts/vm.sh` または Makefile を推奨します

トラブルシュート
- `Permission denied (publickey)`
  - OS Login 有効化、IAM 権限、OS Login 公開鍵の TTL、有効な `-k` 指定を確認
- IAP で失敗
  - `-t` 指定と `roles/iap.tunnelResourceAccessor` の付与を確認
- gcloud アカウント関連
  - `-A <ACCOUNT>` を付ける場合は `gcloud auth login <ACCOUNT>` で認証済みであること

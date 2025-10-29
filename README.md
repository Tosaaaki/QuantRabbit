QuantRabbit – 運用/デプロイ ガイド（更新版）

概要
- 本リポジトリは USD/JPY 自動売買エージェント QuantRabbit のコードベースです。
- デプロイ／運用手順を gcloud のデフォルト設定に依存しない方式へ刷新しました。

必読ドキュメント
- docs/DEPLOYMENT.md（新デプロイ手順の詳細）
- docs/VM_OPERATIONS.md（VM 操作ヘルパ、ログ取得、SQL 等）

クイックスタート（VM へのデプロイ）
1) 便利設定（初回のみ）
   - `scripts/vm.env.example` を `scripts/vm.env` にコピーし、あなたの環境に合わせて編集
2) デプロイ（Make 経由）
   - `make vm-deploy BRANCH=main`
3) ログ追尾
   - `make vm-tail VM_SERVICE=quantrabbit.service`

直接コマンドで実行したい場合
- `scripts/vm.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> deploy -b main -i --restart quantrabbit.service -t`
  - `-t`: IAP トンネル（外部 IP 無しでも接続）
  - `-k`: OS Login の SSH 鍵を明示（例: `~/.ssh/gcp_oslogin_quantrabbit`）
 - `-A`: gcloud アカウントを明示（必要な場合）

トラブルシュート（権限 / systemd / vm.sh）
- OS Login で SSH は通るが `sudo` 不可・`/home/tossaki/QuantRabbit` に入れない・`systemctl` が使えない
  - 原因: IAM に `roles/compute.osAdminLogin` が付与されていない
  - 付与（例）: `gcloud projects add-iam-policy-binding quantrabbit --member="user:www.tosakiweb.net@gmail.com" --role="roles/compute.osAdminLogin"`
  - 検証: `gcloud compute ssh fx-trader-vm --project quantrabbit --zone asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command "sudo -n true && echo SUDO_OK || echo NO_SUDO"`

- `scripts/vm.sh deploy` がリモート実行の quoting 問題で失敗する
  - 回避: 直接コマンドで小さく分けて実行
    - `gcloud compute ssh ... --command "sudo -u tossaki -H bash -lc 'cd /home/tossaki/QuantRabbit && git fetch --all -q || true && git checkout -q main || git checkout -b main origin/main || true && git pull --ff-only && if [ -d .venv ]; then source .venv/bin/activate && pip install -r requirements.txt; fi'"`
    - `gcloud compute ssh ... --command "sudo systemctl daemon-reload && sudo systemctl restart quantrabbit.service && sudo systemctl status --no-pager -l quantrabbit.service || true"`

- 稼働確認/ログ
  - `scripts/vm.sh ... tail -s quantrabbit.service -t`
  - `gcloud compute ssh ... --command "journalctl -u quantrabbit.service -n 200 -f --output=short-iso"`

ログ/DB の取得
- まとめて取得: `scripts/vm.sh ... pull-logs -r /home/tossaki/QuantRabbit/logs -o ./remote_logs -t`
- リモート SQL: `scripts/vm.sh ... sql -f /home/tossaki/QuantRabbit/logs/trades.db -q "SELECT COUNT(*) FROM trades;" -t`

旧スクリプトとの互換
- `scripts/deploy_to_vm.sh` はラッパとして `scripts/vm.sh` を呼ぶように置換済みです。
- 既存の呼び出しはそのまま動きますが、新方式への移行を推奨します。

補足
- IAP を使う場合は `roles/iap.tunnelResourceAccessor` が必要です。
- OS Login の有効化・鍵の登録・IAM 権限が必要です。

レガシー README
- 以前の README の全文は `README.legacy` に退避しています。

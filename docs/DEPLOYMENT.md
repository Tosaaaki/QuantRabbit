デプロイ手順（更新版 / gcloud デフォルト不要）

この手順は gcloud の「アクティブなアカウント/プロジェクト」を設定せずに、その都度 `--project`/`--zone`/`--account` を明示して実行します。

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

5) 旧スクリプトとの互換
- `scripts/deploy_to_vm.sh` はラッパとして `scripts/vm.sh` を呼び出すよう置換済みです
- 旧呼び出しでも動作しますが、今後は `scripts/vm.sh` または Makefile を推奨します

トラブルシュート
- `Permission denied (publickey)`
  - OS Login 有効化、IAM 権限、OS Login 公開鍵の TTL、有効な `-k` 指定を確認
- IAP で失敗
  - `-t` 指定と `roles/iap.tunnelResourceAccessor` の付与を確認
- gcloud アカウント関連
  - `-A <ACCOUNT>` を付ける場合は `gcloud auth login <ACCOUNT>` で認証済みであること

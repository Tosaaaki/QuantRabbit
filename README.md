# QuantRabbit – USD/JPY Autonomous Trading Agent

QuantRabbit は USD/JPY で 24/7 自律運用する無裁量トレーディング・エージェントです。実装と運用の責務は下記の 2 つのドキュメントを中心に整理されています。

- `AGENTS.md` – コンポーネント契約、データスキーマ、制御フロー、安全装置までを網羅したエージェント仕様
- `docs/DEPLOYMENT.md` – デプロイフローと IaC、Cloud Build/Run のハンドブック
- `docs/VM_OPERATIONS.md` – VM 操作ヘルパ、ログ採取、SQLite リモート実行

## Quick Start (VM Deployment)
1. 初回準備: `scripts/vm.env.example` を `scripts/vm.env` にコピーし、プロジェクト/ゾーン/インスタンス情報を設定する
2. デプロイ: `make vm-deploy BRANCH=main`
3. ログ追尾: `make vm-tail VM_SERVICE=quantrabbit.service`

Make を使わない場合は `scripts/vm.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> deploy -b main -i --restart quantrabbit.service -t` を直接呼び出す。`-t` は IAP トンネル、`-k` は OS Login 用 SSH 鍵、必要に応じて `-A` で gcloud アカウントを指定する。

## Always‑on VM Access (IAP + OS Login)
- 1回の初期設定で、以後「アカウント有効化」不要の常時アクセスを実現。
  - 付与ロール（プロジェクト）: `roles/compute.osAdminLogin`, `roles/iap.tunnelResourceAccessor`, `roles/compute.viewer`
  - インスタンス/プロジェクト メタデータ: `enable-oslogin=TRUE`
  - SSH 鍵登録（30 日 TTL）: `gcloud compute os-login ssh-keys add --key-file ~/.ssh/gcp_oslogin_quantrabbit.pub --ttl 30d`
- 接続テスト（IAP 経由）:
  - `gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command "sudo -n true && echo SUDO_OK"`
- 運用は `scripts/vm.sh` 経由（deploy/tail/sql/pull-logs）を推奨。詳細は AGENTS.md の「10. GCE SSH / OS Login ガイド」を参照。

## Architecture Snapshot
- Tick/Candle 取得は `market_data/*` が担当し、`indicators/*` でテクニカル要因を集計する
- レジーム判定とフォーカス決定 (`analysis/regime_classifier.py` / `focus_decider.py`) を経由し、`analysis/gpt_decider.py` が GPT 系モデルで戦略配分を指示
- `strategies/*` が pocket 別のエントリー候補を返し、`execution/*` がステージ管理、リスク審査、発注、クローズまでを連結
- 運用要件、リスクガード、トークン制御、デプロイ戦略などの詳細は `AGENTS.md` に記載

## scripts/vm.sh の主なオプション
- `deploy`: ブランチの pull・依存再インストール (`-i`)・systemd 再起動 (`--restart`) までを一括実行
- `tail`: `systemd` ログの追尾 (`-s quantrabbit.service`)
- `sql`: リモート SQLite を IAP 経由で実行し結果を出力
- `pull-logs`: `/home/tossaki/QuantRabbit/logs` をローカルへ同期

## Troubleshooting
- OS Login は `roles/compute.osAdminLogin` と `roles/iap.tunnelResourceAccessor` を付与してから `sudo` 動作を確認する  
  `gcloud compute ssh fx-trader-vm --project quantrabbit --zone asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command "sudo -n true && echo SUDO_OK || echo NO_SUDO"`
- `scripts/vm.sh deploy` のリモート quoting が崩れた場合は `gcloud compute ssh ... --command` を小さなステップで実行し、`pip install` と `systemctl restart` を分けて呼ぶ

## Logs / DB Access
- ログ同期: `scripts/vm.sh ... pull-logs -r /home/tossaki/QuantRabbit/logs -o ./remote_logs -t`
- SQLite 実行: `scripts/vm.sh ... sql -f /home/tossaki/QuantRabbit/logs/trades.db -q "SELECT COUNT(*) FROM trades;" -t`
- `logs/*.db` のスキーマや運用メモは `AGENTS.md` セクション 2.5 を参照

## Legacy README
過去の README は `README.legacy` に保存しています。

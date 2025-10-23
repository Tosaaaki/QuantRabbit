# QuantRabbit – GPT × OANDA USD/JPY Auto‑Trading Bot

QuantRabbit は **テクニカル × ニュース × GPT 裁量判断** を組み合わせ、  
USD/JPY で 1 日 +100 pips を狙う完全自動売買システムです。  

* **テクニカル** : ta‑lib で計算した MA / BB / RSI / ADX …  
* **ニュース** : Forex Factory / DailyFX RSS → VM fetcher → Cloud Run summarizer (GPT‑4o‑mini)  
* **GPT** : レジーム補足 + 戦略順位 + lot 配分を 60 秒ごとに判断  
* **Pocket 方式** : 同じ口座内で _micro_（スキャル）／_macro_（順張り）を tag 管理  
* **インフラ** : GCE VM + Cloud Storage + Pub/Sub + Cloud Run (news summarizer)  
* **月額コスト** : VM ≈ $19 + Cloud Run ≈ $0.6 + GPT ≈ $4.5 ＝ **$ <25**  

---

## Directory Map

.
├── main.py                  # 60 秒取引ループ（入口）
├── Dockerfile               # VM / Cloud Run 共通
├── requirements.txt         # ライブラリ pin
│
├── config/
│   ├── env.toml             # OPENAI / OANDA / GCP 設定
│   └── pool.yaml            # 手法メタ定義
│
├── market_data/             # ⇢ データ取得
│   ├── tick_fetcher.py      # OANDA WebSocket
│   ├── candle_fetcher.py    # Tick → Candle 生成
│   └── news_fetcher.py      # RSS → GCS raw/ アップ
│
├── indicators/              # ⇢ テクニカル
│   ├── calc_core.py         # ta‑lib ラッパ
│   └── factor_cache.py      # 最新指標共有
│
├── analysis/                # ⇢ 判断ロジック
│   ├── regime_classifier.py # Trend / Range / Breakout
│   ├── focus_decider.py     # micro/macro/event 判定
│   ├── gpt_prompter.py      # GPT 入力生成
│   ├── gpt_decider.py       # OpenAI 呼び出し
│   ├── perf_monitor.py      # PF / Sharpe 更新
│   └── summary_ingestor.py  # GCS summary/ → news.db
│
├── strategies/              # ⇢ 手法プラグイン
│   ├── trend/ma_cross.py
│   ├── breakout/donchian55.py
│   └── mean_reversion/bb_rsi.py
│
├── signals/
│   └── pocket_allocator.py  # lot を micro/macro に分配
│
├── execution/               # ⇢ 発注 & リスク
│   ├── risk_guard.py        # lot/SL/TP & DD監視
│   ├── order_manager.py     # OANDA REST 発注
│   └── position_manager.py  # (今後追加)
│
├── utils/
│   ├── cost_guard.py        # GPT トークン累計管理
│   └── backup_to_gcs.sh     # SQLite/logs nightly backup
│
├── logs/                    # SQLite DB 等
│   ├── trades.db
│   └── news.db
│
└── infra/terraform/         # ⇢ IaC
├── main.tf   (VM)
├── run.tf    (Cloud Run)
└── storage.tf(GCS / PubSub)

---

## Quick Start (Local Demo)

```bash
# 1. clone & install
git clone https://github.com/<your>/QuantRabbit.git
cd QuantRabbit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ta-lib   # macOS
# Ubuntu Linux: build from source
# sudo apt-get update && apt-get install -y build-essential wget
# wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
# tar xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
# ./configure --prefix=/usr && make && sudo make install

# 2. config
cp config/env.toml  config/env.local.toml   # 編集してキーを投入
#   openai_api_key / openai_model_decider / openai_model_summarizer を必要に応じて設定
cp config/pool.yaml config/pool.local.yaml # 有効戦略を調整

# 3. run (practice account, small lot)
python main.py

# pool.yaml example
```yaml
strategies:
  - name: TrendMA
    sl: 30
    tp: 60
    enabled: true
  - name: Donchian55
    sl: 55
    tp: 110
    enabled: true
  - name: BB_RSI
    sl: 10
    tp: 15
    enabled: true
  - name: NewsSpikeReversal
    sl: 10
    tp: 20
    enabled: true
```

Trade Loop Overview
	1.	Tick → Candle(M1) 生成 → factor_cache 更新
	2.	regime_classifier で Macro/Micro レジーム判定
	3.	GPT デシジョン（既定は gpt-5-mini、env で切り替え）に指標 + ニュース + 成績を渡し
	→ focus_tag + weight_macro + 戦略順位 を受け取る
	4.	pocket_allocator で lot を micro/macro に分配
	5.	Strategy プラグイン (MA クロス / Donchian55 / BB+RSI) がシグナルを返す
	6.	risk_guard が lot/SL/TP をクランプし OANDA REST 発注
	7.	成績は logs/trades.db に保存 → perf_monitor が PF/Sharpe 更新
	8.	夜間 cron で DB & ログを GCS へバックアップ

---

## データ同期パイプライン（VM → BigQuery）

トレード履歴 (`logs/trades.db`) を BigQuery へ常時同期するための常駐スクリプトを追加しています。

1. **環境変数**  
   - `BQ_PROJECT`（省略時は `GOOGLE_CLOUD_PROJECT` を利用）  
   - `BQ_DATASET`（既定: `quantrabbit`）  
   - `BQ_TRADES_TABLE`（既定: `trades_raw`）  
   - 任意: `BQ_MAX_EXPORT`（1 バッチの最大行数）

2. **起動コマンド**

```bash
source .venv/bin/activate
python scripts/run_sync_pipeline.py \
  --interval 120 \
  --limit 2000
```

`--once` を渡すと単発実行、`--verbose` で DEBUG ログを出力します。標準出力と `logs/pipeline.log` に実行ログが残ります。

3. **BigQuery テーブル**  
   スクリプトが起動すると、存在しない場合でも dataset/table を自動作成し、`ticket_id`（OANDA tradeID）をキーにアップサートします。`logs/bq_sync_state.json` には最終同期時刻が保存されるため、監視に利用できます。

4. **ロット調整インサイト**  
   `analytics/lot_pattern_analyzer.py` が BigQuery のトレード履歴 (lookback 既定 14 日) を集計し、Pocket × Side 別の勝率 / PF / 標準偏差からロット倍率を提案します。結果は BigQuery `lot_insights` テーブルに追記され、`analytics/lot_insights.json`（GCS UI バケットまたは `GCS_ANALYTICS_BUCKET`）に JSON スナップショットを保存します。手動実行やパラメータ変更は `scripts/generate_lot_insights.py` を利用してください。

5. **systemd への登録例**

```ini
[Unit]
Description=QuantRabbit trade sync pipeline
After=network-online.target

[Service]
WorkingDirectory=/opt/quantrabbit
Environment="BQ_PROJECT=quantrabbit"
ExecStart=/opt/quantrabbit/.venv/bin/python scripts/run_sync_pipeline.py --interval 180
Restart=always

[Install]
WantedBy=multi-user.target
```

タイマーや cron と組み合わせる場合でも、`run_sync_pipeline.py` が `PositionManager.sync_trades()` → BigQuery export を順に実行するため、VM 上で同一スクリプトを呼び出すだけで同期が完了します。

---

## オートチューニング ダッシュボード（ウェブ UI）

FastAPI 製の承認 UI を Cloud Run 上で公開し、チューニング結果の確認・承認をブラウザから行えます。バックエンドは BigQuery `autotune_runs` テーブルを参照します。

1. **BigQuery テーブル作成**  
   例: `cloudrun/autotune_ui/bq_autotune_runs.sql` を実行
   ```bash
   bq query --use_legacy_sql=false < cloudrun/autotune_ui/bq_autotune_runs.sql
   ```

2. **チューニング結果を BigQuery へ記録**  
   `scripts/tune_scalp.py` に `--bq-table` を渡すか、環境変数 `AUTOTUNE_BQ_TABLE` を設定します。
   ```bash
   AUTOTUNE_BQ_TABLE=quantrabbit.autotune_runs \
   python scripts/tune_scalp.py --trials-per-strategy 20 --bq-table quantrabbit.autotune_runs
   ```

3. **Cloud Run へデプロイ**  
   ```bash
   gcloud builds submit --tag gcr.io/$PROJECT/autotune-ui .
   gcloud run deploy autotune-ui \
     --image gcr.io/$PROJECT/autotune-ui \
     --region asia-northeast1 \
     --platform managed \
     --allow-unauthenticated \
     --set-env-vars AUTOTUNE_BQ_TABLE=quantrabbit.autotune_runs
   ```

4. **アクセス**  
   デプロイ後に表示される `https://autotune-ui-xxxx.run.app` が承認ダッシュボードの URL です。テーブルの `status` を更新すると、VM が参照する `configs/scalp_active_params.json` を人手で戻す前にレビュー履歴を残せます。

BigQuery では `status` 列が `pending/approved/rejected` を保持し、UI からの承認・却下操作で更新されます。VM 上の FastAPI UI も同じテーブルを参照するため、ブラウザからの操作でどちらも同期します。

---

## Dashboards: Looker Studio 接続

Looker Studio から GCS（リアルタイム UI）と BigQuery（履歴集計）へ接続するためのブートストラップを同梱しました。

- 1 回セットアップ: `scripts/setup_looker_sources.sh`
- 以降は Looker Studio 側でデータソースを追加するだけです。

1) GCS リアルタイム JSON（UI 用）
- コネクタ: Google Cloud Storage
- サービスアカウント: `ui-dashboard-sa@<project>.iam.gserviceaccount.com` のキーを使用
- オブジェクト: `gs://fx-ui-realtime/realtime/ui_state.json`
- フィールド例:
  - `generated_at`（DateTime）
  - `new_trades`（Record → JSON 展開用にカスタム関数）
  - `recent_trades`（Record → 同上、UNNEST 相当の展開で可視化）
  - `open_positions`（Record → カスタムフィールドで net units 抽出）

2) BigQuery 集計（履歴向け）
- コネクタ: BigQuery → プロジェクト `<project>` → データセット `quantrabbit`
- テーブル/ビュー: `trades_raw` / `trades_recent_view` / `trades_daily_features`
- 推奨更新間隔: 15 分〜1 時間
- 可視化例:
  - 指標カード: 当日 `SUM(pl_pips)`
  - 円グラフ: ポケット別 `win_rate`
  - 折れ線: `close_time` の時系列 P/L
  - ツリーマップ: `strategy × pl_pips`

セットアップ（自動化）

```bash
# 環境に合わせて上書き可（未指定は config/env(.example).toml から読み取り）
GCP_PROJECT=quantrabbit \
UI_BUCKET=fx-ui-realtime \
BQ_DATASET=quantrabbit \
BQ_LOCATION=asia-northeast1 \
UI_SA_EMAIL=ui-dashboard-sa@quantrabbit.iam.gserviceaccount.com \
KEY_OUT=./ui-dashboard-sa.json \
./scripts/setup_looker_sources.sh
```

スクリプトが行うこと:
- SA 作成（存在すれば skip）とキー発行（`.gitignore` 対応済）
- GCS バケット `gs://fx-ui-realtime` の作成と `objectViewer` 付与
- プレースホルダ `realtime/ui_state.json` を配置
- BigQuery dataset `quantrabbit` の作成
- `trades_raw` が存在する場合、`trades_recent_view` と `trades_daily_features` を作成

接続前チェックと検証
- `scripts/run_sync_pipeline.py` が一度以上走り `trades_raw` にデータがある
- Looker Studio のデータソースプレビューでレコードが表示される
- 権限エラー時は SA の IAM を再確認（`roles/storage.objectViewer`, `roles/bigquery.dataViewer`, `roles/bigquery.jobUser`）

注意事項
- サービスアカウントキー（`ui-dashboard-sa.json`）は厳重に保管し、Git にコミットしないでください。
- GCS 側の UI JSON は `analytics/gcs_publisher.py` が出力します（`ui_bucket_name`, `ui_state_object_path`）。
- BigQuery への同期は `scripts/run_sync_pipeline.py`（`BQ_*` 環境変数）で制御します。


## Ops: GCE SSH / OS Login

推奨は OS Login（IAM + 一時鍵）。外部 IP がない場合は IAP を使用。

- 前提ロール: `roles/compute.osLogin` か `roles/compute.osAdminLogin`
- IAP 経由時: `roles/iap.tunnelResourceAccessor`

Setup (一度だけ)

```bash
# OS Login を（必要なら）インスタンスで有効化
gcloud compute instances add-metadata fx-trader-vm \
  --zone asia-northeast1-b --metadata enable-oslogin=TRUE

# 鍵を作成して OS Login に登録（30 日 TTL）
ssh-keygen -t ed25519 -f ~/.ssh/gcp_oslogin_quantrabbit -N '' -C 'oslogin-quantrabbit'
gcloud compute os-login ssh-keys add \
  --key-file ~/.ssh/gcp_oslogin_quantrabbit.pub --ttl 30d
```

Connect

```bash
# 外部 IP あり
gcloud compute ssh fx-trader-vm \
  --project quantrabbit --zone asia-northeast1-b \
  --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit

# 外部 IP なし / IAP 経由
gcloud compute ssh fx-trader-vm \
  --project quantrabbit --zone asia-northeast1-b \
  --tunnel-through-iap \
  --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit

# 直接 SSH（OS Login ユーザ名は describe-profile で確認）
ssh -i ~/.ssh/gcp_oslogin_quantrabbit <oslogin_username>@<EXTERNAL_IP>
```

Troubleshooting

- `Permission denied (publickey)` のとき:
  - OS Login 有効化状態を確認（enable-oslogin=TRUE）
  - IAM osLogin 権限があるか確認
  - OS Login 側の公開鍵 TTL 切れに注意（再登録）
  - `--ssh-key-file` で鍵を明示、`--troubleshoot` で診断
- OS Login 有効時はメタデータ `ssh-keys` は無視されます。
- 組織ポリシー `compute.requireOsLogin` が有効な場合は OS Login のみ許可。

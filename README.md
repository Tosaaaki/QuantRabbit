# QuantRabbit – GPT × OANDA USD/JPY Auto‑Trading Bot

QuantRabbit は **テクニカル × ニュース × GPT 裁量判断** を組み合わせ、  
USD/JPY で 1 日 +100 pips を狙う完全自動売買システムです。  

* **テクニカル** : ta‑lib で計算した MA / BB / RSI / ADX …  
* **ニュース** : Forex Factory / DailyFX RSS → Cloud Run → GPT‑4o‑mini 要約  
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

## Quick Start (Local Demo)

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
cp config/pool.yaml config/pool.local.yaml # 有効戦略を調整

# 3. run (practice account, small lot)
python main.py

## Deploy to Cloud Run

アクティブな gcloud アカウント/プロジェクトが `www.tosakiweb.net@gmail.com / quantrabbit` のままであれば、
ローカルのコンテキストをそのまま利用してデプロイします。切り替えていない場合でも
`--project=quantrabbit` 付きの Cloud Build + Cloud Run を自動実行するようになりました。

```bash
./scripts/deploy_fx_trader.sh          # 取引サービス (cloudrun.trader_service)
./scripts/deploy_news_pipeline.sh      # ニュース summarizer 群
./scripts/deploy_exit_manager.sh       # exit_manager_service (GPT exit判定)

# インパーソネーションで実行したいとき
DEPLOY_IMPERSONATE_SA=event-logger-sa@quantrabbit.iam.gserviceaccount.com ./scripts/deploy_fx_trader.sh
```

`fx-exit-manager` (Cloud Run) は exit_manager_service を HTTP 起動する常駐サービスです。
Cloud Scheduler で 1〜2 分ごとに `https://fx-exit-manager-<PROJECT>.asia-northeast1.run.app` を叩くと
GPT exit ロジックが本番でも継続実行されます。手動テストは `./scripts/trigger_exit_manager.sh`
で行えます。

ニュース系リソースは `scripts/deploy_news_pipeline.sh` で同じポリシー／手順で更新できます。

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
	3.	GPT (4o‑mini) に指標 + ニュース + 成績を渡し
→ focus_tag + weight_macro + 戦略順位 を受け取る
	4.	pocket_allocator で lot を micro/macro に分配
	5.	Strategy プラグイン (MA クロス / Donchian55 / BB+RSI) がシグナルを返す
	6.	risk_guard が lot/SL/TP をクランプし OANDA REST 発注
	7.	ExitManager が BE（建値）移動・ATR トレーリング・時間/RSI 退出を適用
	8.	成績は logs/trades.db に保存 → perf_monitor が PF/Sharpe 更新
	9.	夜間 cron で DB & ログを GCS へバックアップ

---

## GCP マルチアカウント運用メモ

- `gcloud config configurations create <name>` でプロジェクトごとの設定を作成し、作業前に `gcloud config configurations activate <name>` で切り替える。
- 各設定内で `gcloud config set project <PROJECT_ID>` と `gcloud config set account <ACCOUNT>` を実行しておけば、アクティブアカウントを書き換えても他の設定には影響しない。
- CLI を混在させたい場合は `gcloud --configuration=<name> <command>` でコマンド単位に指定する。
- アプリやスクリプトはサービスアカウント鍵を利用し、`export GOOGLE_APPLICATION_CREDENTIALS=/actual/path/service-account.json` を設定したターミナル内だけで権限を切り替える。
- 異なるプロジェクトを同時に操作する際は、ターミナルを分けてそれぞれの環境変数を保持するのが安全。

### QuantRabbit プロジェクト (www.tosakiweb.net@gmail.com) をアクティブ化する手順

```
cd /Users/tossaki/Documents/App/QuantRabbit
gcloud config configurations activate www-tosakiweb
gcloud config list --format='value(core.account,core.project)'
```

- 1 行目のディレクトリに移動してから上記 2 コマンドを実行すると、アカウント `www.tosakiweb.net@gmail.com`・プロジェクト `quantrabbit` がアクティブになる。
- `gcloud auth list` で `* www.tosakiweb.net@gmail.com` が付いているか確認し、必要に応じて `gcloud auth application-default login` または `gcloud auth application-default set-quota-project quantrabbit` で ADC も合わせる。
- 他プロジェクトへ戻る際は、対応する構成 (`default` など) を再度 `gcloud config configurations activate ...` で切り替える。

---

## Retrospectives / Kaizen

- Daily/Weekly Retro: `python scripts/retro_report.py --days 1` (or `--days 7`).
  - Prints a concise PF/Sharpe/WinRate summary, pocket and strategy breakdowns.
  - Also writes a markdown file under `reports/daily/` or `reports/weekly/`.
- Incident Postmortems: use the GitHub template `.github/ISSUE_TEMPLATE/postmortem.md`.
  - Capture summary, impact, timeline, root cause, and action items.
- Continuous Tuning:
  - `ExitManager` reads per‑strategy/pocket policy from SQLite (`exit_policy`).
  - `Kaizen` job audits closed trades (MFE/MAE from OANDA M1) and auto‑adjusts:
    - Earlier BE if many "gave_back" losses, and tighter trailing.
    - Slight relaxation if exits are too tight with no give‑back.
  - Policies persist in `logs/trades.db` so they improve over time.

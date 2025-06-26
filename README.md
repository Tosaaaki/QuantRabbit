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
cp config/env.toml config/env.local.toml   # 編集してキーを投入

# 3. run (practice account, small lot)
python main.py

Trade Loop Overview
	1.	Tick → Candle(M1) 生成 → factor_cache 更新
	2.	regime_classifier で Macro/Micro レジーム判定
	3.	GPT (4o‑mini) に指標 + ニュース + 成績を渡し
→ focus_tag + weight_macro + 戦略順位 を受け取る
	4.	pocket_allocator で lot を micro/macro に分配
	5.	Strategy プラグイン (MA クロス / Donchian55 / BB+RSI) がシグナルを返す
	6.	risk_guard が lot/SL/TP をクランプし OANDA REST 発注
	7.	成績は logs/trades.db に保存 → perf_monitor が PF/Sharpe 更新
	8.	夜間 cron で DB & ログを GCS へバックアップ


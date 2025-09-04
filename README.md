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
cp config/env.toml  config/env.local.toml   # 編集してキーを投入
cp config/pool.yaml config/pool.local.yaml # 有効戦略を調整

# 主な設定例（env.local.toml）
# oanda_practice = "true"  # 安全のためデフォルトは practice を推奨

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
	3.	GPT (4o‑mini) に指標 + ニュース + 成績を渡し
→ focus_tag + weight_macro + 戦略順位 を受け取る
	4.	pocket_allocator で lot を micro/macro に分配
	5.	Strategy プラグイン (MA クロス / Donchian55 / BB+RSI) がシグナルを返す
	6.	risk_guard が lot/SL/TP をクランプし OANDA REST 発注
	7.	成績は logs/trades.db に保存 → perf_monitor が PF/Sharpe 更新
	8.	夜間 cron で DB & ログを GCS へバックアップ

---

## Ops: VM への反映手順（SSH 不可時の代替）

通常は GitHub に PR → Squash Merge → VM 側で `git pull`（`startup-script` が起動時に実行）で反映します。
ただし、SSH が通らない・一時的に手早く反映したい場合は、GCE インスタンスの `startup-script` メタデータを一時的に差し替え、再起動時に差分ファイルを上書きする「オーバーレイ方式」が使えます。

- 前提: `gcloud` 認証済み、プロジェクトは `quantrabbit`、ゾーンは `asia-northeast1-a`、対象 VM は `fx-trader-vm`。
- 反映対象: `main.py` と `execution/risk_guard.py`（必要に応じて増減可）。

手順（ローカルで実行）

```bash
# 1) 現行 startup-script を取得
gcloud compute instances describe fx-trader-vm \
  --zone=asia-northeast1-a \
  --format='get(metadata.items.[key:startup-script].value)' > /tmp/original_startup.sh

# 2) 取得したスクリプト末尾に、ローカルの変更ファイルを書き込む処理を追記
python3 - <<'PY'
from pathlib import Path
orig = Path('/tmp/original_startup.sh').read_text()
# SecretManager を venv Python で読みつつ、失敗しても続行
orig = orig.replace("python3 - <<'PY'","PYBIN='/home/quantrabbit/QuantRabbit/.venv/bin/python'\n$PYBIN - <<'PY' || true")
main = Path('main.py').read_text()
rg = Path('execution/risk_guard.py').read_text()
overlay = "\n# ---- Overlay patched files ----\n"
overlay += "TARGET_DIR=\"/home/quantrabbit/QuantRabbit\"\n"
overlay += "install -o quantrabbit -g quantrabbit -m 0644 /dev/stdin \"$TARGET_DIR/main.py\" <<'PYMAIN'\n" + main + "\nPYMAIN\n"
overlay += "install -d -o quantrabbit -g quantrabbit -m 0755 \"$TARGET_DIR/execution\"\n"
overlay += "install -o quantrabbit -g quantrabbit -m 0644 /dev/stdin \"$TARGET_DIR/execution/risk_guard.py\" <<'PYRG'\n" + rg + "\nPYRG\n"
overlay += "systemctl restart quantrabbit || true\nsleep 2\nsystemctl status --no-pager --lines=50 quantrabbit || true\n"
Path('/tmp/patched_startup.sh').write_text(orig + overlay)
print('Wrote /tmp/patched_startup.sh')
PY

# 3) startup-script を置換して VM を再起動
gcloud compute instances add-metadata fx-trader-vm \
  --zone=asia-northeast1-a \
  --metadata-from-file startup-script=/tmp/patched_startup.sh
gcloud compute instances reset fx-trader-vm --zone=asia-northeast1-a

# 4) シリアルポートログで適用を確認
gcloud compute instances get-serial-port-output fx-trader-vm \
  --zone=asia-northeast1-a --port=1 | tail -n 200
```

注意
- この方式は「即時の暫定反映」を目的とした運用カバーです。最終的には必ず GitHub に変更を push し、通常の GitOps（PR → Merge → 起動時 `git pull`）に戻してください。
- `startup-script` を元に戻す場合は、上記で生成した追記部分（Overlay セクション）を取り除いて再適用してください。

ポリシー
- 「VM へ反映したら必ず GitHub にプッシュ」— 直接オーバーレイした変更は、直後にブランチ作成→コミット→PR→マージを実施して、リポジトリと VM の乖離を解消します。

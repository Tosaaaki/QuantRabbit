# AGENT.me  –  QuantRabbit Agent Specification

## 1. ミッション
> **狙い**	: USD/JPY で 1 日 +100 pips を実現する、 24/7 無裁量トレーディング・エージェント。  
> **境界**	: 発注・リスクは機械的、曖昧判断とニュース解釈は GPT‑5 系 (既定 gpt‑5‑mini) に委譲。

補助資料: 運用/デプロイの手順面は `README.md` と `docs/` 配下を参照する。

## 目次
- [1. ミッション](#1-ミッション)
- [2. コンポーネント間の契約](#2-コンポーネント間の契約)
- [3. ライフサイクル](#3-ライフサイクル)
- [4. 環境変数 / Secret 一覧](#4-環境変数--secret-一覧)
- [5. トークン & コストガード](#5-トークン--コストガード)
- [6. 安全装置](#6-安全装置)
- [7. デプロイ手順（更新版 / 要約）](#7-デプロイ手順更新版--要約)
- [8. チーム運用ルール](#8-チーム運用ルール)
- [9. 参考ドキュメント](#9-参考ドキュメント)
- [10. GCE SSH / OS Login ガイド](#10-gce-ssh--os-login-ガイド)

---

## 2. コンポーネント間の契約

| レイヤ | 担当 | 期待する入出力 |
|--------|------|----------------|
| **DataFetcher** | `market_data/*` | + Tick JSON<br>+ Candle dict<br>+ raw News JSON (→ GCS) |
| **IndicatorEngine** | `indicators/*` | ← Candle deque<br>→ factors dict {ma10, rsi, …} |
| **Regime & Focus** | `analysis/regime_classifier.py` / `focus_decider.py` | ← factors<br>→ macro/micro レジーム・weight_macro |
| **GPT Decider** | `analysis/gpt_decider.py` | ← focus + perf + news<br>→ JSON {focus_tag, weight_macro, weight_scalp, ranked_strategies} |
| **Strategy Plugin** | `strategies/*` | ← factors<br>→ dict {action, sl_pips, tp_pips, confidence, tag} or None |
| **Exit Manager** | `execution/exit_manager.py` | ← open positions + signals<br>→ list[{pocket, units, reason, tag}] |
| **Risk Guard** | `execution/risk_guard.py` | ← lot, SL/TP, pocket<br>→ bool (可否)・調整値 |
| **Order Manager** | `execution/order_manager.py` | ← units, sl, tp, client_order_id, tag<br>→ OANDA ticket ID |
| **Logger** | `logs/*.db` | 全コンポーネントが INSERT |

### 2.1 共通データスキーマ

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

class Tick(BaseModel):
    ts_ms: int
    instrument: Literal["USD_JPY"]
    bid: float
    ask: float
    mid: float
    volume: int

class Candle(BaseModel):
    ts_ms: int                # epoch ms (UTC)
    instrument: Literal["USD_JPY"]
    timeframe: Literal["M1","M5","H1","H4","D1"]
    o: float; h: float; l: float; c: float
    volume: int
    bid_close: Optional[float] = None
    ask_close: Optional[float] = None

class Factors(BaseModel):
    instrument: Literal["USD_JPY"]
    timeframe: Literal["M1","M5","H4","D1"]
    adx: float
    ma10: float
    ma20: float
    bbw: float
    atr_pips: float
    rsi: float
    vol_5m: float

class GPTDecision(BaseModel):
    focus_tag: Literal["micro","macro","hybrid","event"]
    weight_macro: float = Field(ge=0.0, le=1.0)
    weight_scalp: float = Field(ge=0.0, le=1.0)  # macro + scalp <= 1.0, remainder = micro
    ranked_strategies: List[str]
    reason: Optional[str] = None

class StrategyDecision(BaseModel):
    pocket: Literal["micro","macro","scalp"]
    action: Literal["OPEN_LONG","OPEN_SHORT","CLOSE","HOLD"]
    sl_pips: float = Field(gt=0)
    tp_pips: float = Field(gt=0)
    confidence: int = Field(ge=0, le=100)
    tag: str

class OrderIntent(BaseModel):
    instrument: Literal["USD_JPY"]
    units: int                      # +buy / -sell
    entry_price: float
    sl_price: float
    tp_price: float
    pocket: Literal["micro","macro","scalp"]
    client_order_id: str
```

> すべてのコンポーネントは上記モデル（もしくは互換 JSON Schema）で通信し、`logs/*.db` にも生 JSON を残す。

### 2.2 単位・用語の定義

| 用語 | 定義 |
|------|------|
| `pip` | USD/JPY の 1 pip は 0.01。入力/出力とも pip 単位を明記する。 |
| `point` | 0.001。OANDA REST 価格の丸め単位。 |
| `lot` | 1 lot = 100,000 units。`units = round(lot * 100000)`。 |
| `pocket` | `micro` = 短期テクニカル、`macro` = レジーム/ニュース。口座資金を `pocket_ratio` で按分。 |
| `weight_macro` | 0.0〜1.0。`pocket_macro = pocket_total * weight_macro` を意味する（運用では Macro を最大 30% に制限）。 |

- `price_from_pips("BUY", entry, sl_pips)` = `entry - sl_pips * 0.01` を `round(price, 3)`。
- `price_from_pips("SELL", entry, sl_pips)` = `entry + sl_pips * 0.01` を `round(price, 3)`。
- `client_order_id = f"qr-{ts_ms}-{focus_tag}-{tag}"`。9 桁以内のハッシュを付けて重複防止。

### 2.3 コンポーネント I/O 詳細

- DataFetcher: OANDA streaming で `Tick` を取得し 60s 終端で `Candle` を確定、欠損 tick は遅延として扱う (`lag_ms` を添付)。
- IndicatorEngine: 各 timeframe ごとに `Deque[Candle]` を維持し 300 本以上揃ったときに `Factors` を算出、入力欠損時は `stale=True` を返し Strategy を停止。
- Regime & Focus: macro=H4/D1, micro=M1 の `Factors` を消費し、`focus_decider` は `FocusDecision`(`focus_tag`,`weight_macro`) を返す。
- GPT Decider: 過去 15 分のニュース要約 + パフォーマンス指標を入力し `GPTDecision` を返す（`weight_macro` と `weight_scalp` を明示）。JSON Schema 不一致時はフォールバックを Raise。
- Strategy Plugin: `ranked_strategies` 順に呼び出し `StrategyDecision` または `None` を返す。必ず `confidence` と `tag` を含め、`None` は「ノートレード」。
- Exit Manager: 現在のポジションとシグナルを突き合わせ、逆方向シグナル・イベントロック・指標劣化の各条件でクローズ指示を組み立てる。
- Risk Guard: エントリー/クローズ双方の `StrategyDecision` と口座情報から `OrderIntent` を生成、拒否理由は `{"allow": False, "reason": ...}` としてロガーへ渡す。
- Order Manager: `OrderIntent` を OANDA REST へ送信、結果は `ticket_id` と `executed_price` を返し `logs/orders.db` に保存。

### 2.4 OANDA API マッピング

| Strategy action | REST 注文 | units 符号 | SL/TP 指定 | 備考 |
|-----------------|-----------|------------|------------|------|
| `OPEN_LONG` | `MARKET` | `+abs(units)` | `stopLossOnFill`, `takeProfitOnFill` | `timeInForce=FOK`, `positionFill=DEFAULT` |
| `OPEN_SHORT` | `MARKET` | `-abs(units)` | 同上 | ask/bid 逆転チェック後に送信 |
| `CLOSE` | `MARKET` | 既存ポジの反対売買 | SL/TP 指定なし | `OrderCancelReplace` で逆指値を削除 |
| `HOLD` | 送信なし | 0 | なし | Strategy ループ継続 |

- すべての注文に `clientExtensions = {"id": client_order_id, "tag": pocket}` を付与し、再試行時は同一 ID を再利用する。Exit 指示も同じ ID 命名規則に従い、`qr-{epoch_ms}-{focus_tag}-{strategy_tag}` 形式で 90 日間ユニークにする。
- OANDA 5xx/timeout 時は 0.5s, 1.5s の指数バックオフをかけ、3 回失敗で `Risk Guard` にエスカレーションする。

### 2.5 ログと永続化

- `logs/trades.db`: `trade_id`, `pocket`, `entry_ts`, `exit_ts`, `pl_pips`, `sl_pips`, `tp_pips`, `strategy_tag`, `client_order_id`, `event_mode`。
- `logs/news.db`: `published_at`, `source`, `headline`, `summary`, `url`, `tokens_used`。
- `logs/metrics.db`: `ts`, `metric`, `value`, `tags`。`decision_latency`, `data_lag`, `order_success_rate` 等を保存。
- **運用メモ**: 本番ログは VM (`fx-trader-vm`) 上 `/home/tossaki/QuantRabbit/logs/` にのみ保存。状況確認時は OS Login/IAP 経由で以下のように参照する：
  ```bash
  gcloud compute ssh fx-trader-vm \
    --project=quantrabbit --zone=asia-northeast1-a \
    --tunnel-through-iap \
    --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit \
    --command "sudo -u tossaki sqlite3 /home/tossaki/QuantRabbit/logs/trades.db \"SELECT DATE(close_time), COUNT(*), ROUND(SUM(pl_pips),2) FROM trades WHERE DATE(close_time)=DATE('now') GROUP BY 1;\""
  ```
  推奨（デフォルト設定不要のヘルパ）:
  ```bash
  scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
    sql -f /home/tossaki/QuantRabbit/logs/trades.db \
    -q "SELECT DATE(close_time), COUNT(*), ROUND(SUM(pl_pips),2) FROM trades WHERE DATE(close_time)=DATE('now') GROUP BY 1;" -t
  ```

---

## 3. ライフサイクル

1. **Startup (`main.py`)**
   1. env.toml 読込 → Secrets 確認
   2. WebSocket 接続確立
2. **Every 60 s**
   1. 新ローソク → factors 更新  
   2. regime + focus → GPT decision  
   3. pocket lot 配分 → Strategy loop（confidence スケーリング + ステージ判定）  
   4. Exit manager → Risk guard → order_manager でクローズ/新規発注  
   5. trades.db / news.db / metrics.db にログ
3. **Background Jobs**
   - `news_fetcher` RSS → GCS raw/  
   - Cloud Run `news‑summarizer`  raw → summary/  
   - `summary_ingestor` summary/ → news.db  
   - nightly `backup_to_gcs.sh` logs/ → backup bucket

### 3.1 60 秒タクトの運用要件

- サイクル開始は正秒同期 (`datetime.utcnow().replace(second=0, microsecond=0)`)、許容誤差 ±500 ms。
- 処理締切は 55 s。締切超過時は当該サイクルを捨て、次のサイクルで再計算する (バックログ禁止)。
- `monotonic()` ベースで `decision_latency_ms` を測定し、遅延は `logs/metrics.db` に記録する。

### 3.2 データ鮮度と完全性

- `max_data_lag_ms = 3000`。これを超える遅延は `DataFetcher` が `stale=True` を返し `Risk Guard` は発注を拒否する。
- Candle 確定は `tick.ts_ms // 60000` の変化で判定し、終値は最後の mid。`volume=0` のローソク足は `missing_bar` としてログ。
- ニュースは `summary_ingestor` が 30 秒毎にポーリング。最新記事が 120 分超なら `news_status=stale` をセット。

### 3.3 発注冪等性とリトライ

- `client_order_id` は 90 日間ユニーク。OANDA `POST /orders` 失敗時は同一 ID で最大 3 回まで再送。
- REST 429/5xx は指数バックオフ (0.5s→1.5s→3.5s) とジッター 100 ms を加える。
- 発注中に WebSocket 停止を検知した場合は `Order Manager` が `halt_reason="stream_disconnected"` を残して停止。

### 3.4 検証パイプライン

- **Record**: `DataFetcher` は全 Tick を `logs/replay/*.jsonl` に保存しテストで再生できる状態を担保。
- **Backtest**: Strategy Plugin は記録データを用い同一 `StrategyDecision` を再現できることを CI で検証。
- **Shadow**: 本番 tick + 仮想アカウントで `OrderIntent` を生成し、`risk_guard` の拒否理由を比較。

### 3.5 エントリー/クローズ制御

- **Confidence スケーリング**: Strategy の `confidence` (0–100) をポケット割当 lot に掛け、最低 0.2 倍〜最大 1.0 倍のレンジで段階的エントリーを行う。  
- **ステージ比率**: `STAGE_RATIOS` で定義されたフラクションに従い、各ステージ条件 (`_stage_conditions_met`) を通過した場合のみ追撃。  
- **Exit Manager**: 逆方向シグナルが閾値 (既定 70) を超えた場合やイベントロック、RSI/ADX 劣化などでクローズ。`allow_reentry` が False の場合は当該サイクル内の再参入を禁止する。
- **Release gate**: PF>1.1, 勝率>52%、最大 DD<5% を 2 週間連続で満たしたら実弾に昇格。

#### 3.5.1 レンジモード強化（2025-10）
- 判定: `analysis/range_guard.detect_range_mode` が M1 の `ADX<=22`, `BBW<=0.20`, `ATR<=6` の同時充足、または H4 トレンド弱含み＋複合スコア閾値超で `range_mode` を返す。`metrics.composite` と `reason` をログ出力。
- エントリー制御: `range_mode=True` の間はマクロ新規を抑制し、許可戦略を BB 逆張り（`BB_RSI`）等に限定。`focus_tag` を `micro` へ縮退、`weight_macro` を上限 0.15 に制限。
- 利確/損切り: レンジ中は TP/SL をタイトに調整（目安 1.5〜2.0 pips の RR≒1:1）。`execution/exit_manager` は含み益が+1.6pips 以上で利確、+0.4pips超はホールド、−1.0pips で早期撤退。
- 分割利確: `execution/order_manager.plan_partial_reductions` はレンジ中のしきい値を（macro 16/22, micro 10/16, scalp 6/10 pips）に低減し早めにヘッジ。
- ステージ/再入場: `execution/stage_tracker` が方向別クールダウンとステージ永続化を提供。強制クローズや連続 3 敗で 15 分ブロック。勝ち負けに応じてロット係数を自動縮小（マーチン禁止）。

実装差分の主な入口
- レンジ判定: `analysis/range_guard.py`
- エントリー選別/SLTP調整/レンジ抑制: `main.py` のシグナル評価・ロット配分周辺
- 早期利確/撤退: `execution/exit_manager.py`
- 分割利確しきい値(レンジ対応): `execution/order_manager.py`
- ステージ永続化/クールダウン/ロット係数: `execution/stage_tracker.py`

### 3.6 オンライン自動チューニング運用
- 5〜15 分間隔で `scripts/run_online_tuner.py` を呼び出し、Exit 感度や入口ゲート、quiet_low_vol の配分を**小幅**に調整する。リスクのあるホットパス（tick 判定・即時 Exit）は対象外。
- 既定はシャドウ運用（`TUNER_ENABLE=true`, `TUNER_SHADOW_MODE=true`）。`config/tuning_history/` に履歴だけを残し、本番パラメータ (`config/tuning_overrides.yaml`) は書き換えない。
- 本適用時は `TUNER_SHADOW_MODE=false` に切り替え、`scripts/apply_override.py` で `config/tuning_overlay.yaml` を生成してランタイムへ読み込ませる。
- 現在の検証タスクと実行手順は `docs/autotune_taskboard.md` に集約。定期実行の有無・評価観点（EV, hazard 比率, decision_latency）もここで管理する。
- オンラインチューニング関連の ToDo は必ず `docs/autotune_taskboard.md` に追記し、対応中はここを参照しながら進める。完了後は同ファイル内で状態をアーカイブ（チェック済み / メモ欄）として残す。

---

## 4. 環境変数 / Secret 一覧

| Key | 説明 |
|-----|------|
| `OPENAI_API_KEY` | GPT 呼び出し用 (Decider / Summarizer 共通) |
| `OPENAI_MODEL_DECIDER` | GPT デシジョン用モデル (例: gpt-5-mini) |
| `OPENAI_MODEL_SUMMARIZER` | ニュース要約用モデル (例: gpt-5-nano) |
| `OANDA_TOKEN` / `OANDA_ACCOUNT` | REST / Stream |
| `GCP_PROJECT` / `GOOGLE_APPLICATION_CREDENTIALS` | GCS・Pub/Sub |
| `GCS_BACKUP_BUCKET` | logs バックアップ先 |

---

## 5. トークン & コストガード

* `.cache/token_usage.json` に月累計。  
* `openai.max_month_tokens` (env.toml) で上限設定。  
* 超過時：`news_fetcher` は継続、`gpt_decider` はフォールバック JSON を返す。  
* フォールバック JSON: `{"focus_tag":"hybrid","weight_macro":0.5,"weight_scalp":0.15,"ranked_strategies":["TrendMA","H1Momentum","Donchian55","BB_RSI","NewsSpikeReversal"],"reason":"fallback"}`。  
* GPT 失敗時は過去 5 分の決定を再利用 (`reason="reuse_previous"`) し、`decision_latency_ms` を 9,000 で固定計上する。

---

## 6. 安全装置

* **Pocket DD**	: micro 5 %、macro 15 % → 該当 pocket の新規取引停止  
* **Global DD**	: 20 % → プロセス自動終了 (`risk_guard`)  
* **Event モード**	: 指標 ±30 min → micro 新規禁止  
* **Timeout**	: GPT 7 s、OANDA REST 5 s → 再試行 / フォールバック  
* **Healthbeat**	: `main.py` が 5 min ping を Cloud Logging に残す

### 6.1 リスク計算とロット配分

- `pocket_equity = account_equity * pocket_ratio`。`pocket_ratio` は `weight_macro` と `pocket` 固有の上限 (`micro<=0.6`, `macro<=0.3`) を掛け合わせる。
- 1 トレードの許容損失は `risk_pct = 0.02`。`risk_amount = pocket_equity * risk_pct`。
- USD/JPY の 1 lot 当たり pip 価値は 1000 JPY。従って `lot = min(MAX_LOT, round(risk_amount / (sl_pips * 1000), 3))`。
- `units = int(round(lot * 100000))`。`abs(units) < 1000` はノイズ扱いで発注しない。
- `clamp_sl_tp(price, sl, tp, side)` は 0.001 単位で丸め、SL/TP 逆転時は 0.1 のバッファを確保。

### 6.2 状態遷移

| 状態 | 遷移条件 | 動作 |
|------|----------|------|
| `NORMAL` | 初期状態 | 全 pocket 取引許可 |
| `EVENT_LOCK` | 経済指標 ±30 min | `micro` 新規停止、建玉縮小ロジック発動 |
| `MICRO_STOP` | `micro` pocket DD ≥5% または `news_status=stale` | `micro` 決済のみ、`macro` 継続 |
| `GLOBAL_STOP` | Global DD ≥20% または `Healthbeat` 欠損>10 min | 全取引停止、プロセス終了 |
| `RECOVERY` | DD が閾値の 80% 未満、24h 経過 | 新規建玉再開前に `main.py` がドライラン |

### 6.3 ニュース・イベント劣化運転

- `news_age_min > 120` で `focus_tag` を強制的に `micro` / `hybrid` へ縮退、`weight_macro` は指数減衰 (`weight_macro *= 0.5`)。
- RSS 取得失敗が 5 回連続した場合は `news_fetcher` が削除せずリトライを継続しつつ Slack へ通知。
- 週末・祝日ギャップは 金曜 21:55Z〜日曜 21:35Z を取引禁止 window とし、自動復帰時に `stale` フラグをクリアする。

### 6.4 観測指標とアラート

- **SLI**: `decision_latency_ms`, `data_lag_ms`, `order_success_rate`, `reject_rate`, `gpt_timeout_rate`, `pnl_day_pips`, `drawdown_pct`。
- **SLO**: `decision_latency_ms p95 < 2000`, `order_success_rate ≥ 0.995`, `data_lag_ms p95 < 1500`, `drawdown_pct max < 0.18`, `gpt_timeout_rate < 0.05`。
- **Alert**: SLO 違反、`healthbeat` 欠損 5 分超、`token_usage ≥ 0.8 * max_month_tokens`, `news_status=stale 10 min`, `order reject` 連続 3 件。

---

## 7. デプロイ手順（更新版 / 要約）

```bash
# Cloud Run ニュース要約のビルド（プロジェクトを明示）
gcloud --project=quantrabbit builds submit --tag gcr.io/quantrabbit/news-summarizer

# IaC（必要に応じて）
cd infra/terraform && terraform init && terraform apply

# VM へアプリをデプロイ（IAP 経由）
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm \
  deploy -b main -i --restart quantrabbit.service -t
```

- デプロイ前に `terraform plan` を CI で実行し差分確認、サービスアカウントは最小権限 (`roles/storage.objectAdmin`, `roles/logging.logWriter`, 必要な Pub/Sub Roles)。
- Cloud Build 成功時に `cosign sign` でイメージ署名、SBOM (`gcloud artifacts sbom export`) を保存。
- 予算アラート: `GCP Budget Alert ≥ 80%` で Slack 通知、IAP トンネルは `roles/iap.tunnelResourceAccessor` を必須化。
- ロールバック: `scripts/vm.sh ... exec -- 'cd ~/QuantRabbit && git checkout <release-tag> && ./startup.sh --dry-run'` を実施し、検証後に `--apply`。
  - 旧 `gcloud compute ssh ... --command` でも可だが、`vm.sh` でプロジェクト/ゾーン/IAP を一元化することを推奨。

### 7.1 VM デプロイ（Git フロー標準）

ローカル → リモート（origin）へ push → VM で `git pull` → systemd 再起動の流れをスクリプト化しています。

前提条件
- OS Login が有効、`roles/compute.osLogin` と（外部 IP 無しの場合）`roles/iap.tunnelResourceAccessor` 付与済み
- VM 側のリポジトリは `origin` が到達可能（例: GitHub）

コマンド例（新標準）
```bash
# 現在のブランチをデプロイし、VM の venv も依存更新
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm deploy -i -t

# 明示的にブランチを指定
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm deploy -b feature/exit-manager -i -t

# ログの追尾（systemd）
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm tail -s quantrabbit.service -t
```

Makefile ショートカット（`scripts/vm.env` を設定すると便利）
```bash
make vm-deploy BRANCH=main
make vm-tail VM_SERVICE=quantrabbit.service
make vm-logs
make vm-sql Q="SELECT COUNT(*) FROM trades;"
```

オプション
- `-b <BRANCH>`: デプロイ対象ブランチ（既定はローカルの現在ブランチ）
- `-i`: VM の venv で `pip install -r requirements.txt` を実行
- `-p <PROJECT>` / `-z <ZONE>` / `-m <INSTANCE>` / `-d <REPO_DIR>` / `-s <SERVICE>`: 環境に応じて上書き可能
- `-k <KEYFILE>`: OS Login 用 SSH 鍵を明示
- `-t`: IAP トンネルを使用（外部 IP 無しでも SSH）

注意点
- ローカルの未コミット変更は push されません。必ずコミットしてから実行してください。
- 直接 SCP での差し替えは緊急時のみ。通常運用は本スクリプト経由の Git ベース反映を推奨します。

OS Login 鍵準備（初回のみ）
```bash
ssh-keygen -t ed25519 -f ~/.ssh/gcp_oslogin_quantrabbit -N '' -C 'oslogin-quantrabbit'
gcloud compute os-login ssh-keys add --key-file ~/.ssh/gcp_oslogin_quantrabbit.pub --ttl 30d
```
デプロイ例（鍵指定/IAP併用）
```bash
scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm deploy -i -k ~/.ssh/gcp_oslogin_quantrabbit -t
```

## 7.2 デプロイトラブルシュート（OS Login / 権限 / systemd）

- 症状
  - IAP/OS Login で VM には入れるが、`sudo` 不可・`/home/tossaki/QuantRabbit` へアクセス不可・`systemctl` での再起動ができない。
  - `scripts/vm.sh ... deploy -i --restart ...` が `sudo` 権限不足で停止。

- 原因
  - 対象の OS Login ユーザに `roles/compute.osAdminLogin` が付与されていない。

- 解決（プロジェクト IAM に 1 コマンドで付与）
  - 実行者: プロジェクト IAM 管理者
  - 例:
    - `gcloud projects add-iam-policy-binding quantrabbit \
       --member="user:www.tosakiweb.net@gmail.com" \
       --role="roles/compute.osAdminLogin"`

- 検証
  - `gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a \
     --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit \
     --command "sudo -n true && echo SUDO_OK || echo NO_SUDO"`
  - `SUDO_OK` が出れば権限 OK。

- デプロイ実行（標準）
  - `scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t \
     -k ~/.ssh/gcp_oslogin_quantrabbit -d /home/tossaki/QuantRabbit \
     deploy -i --restart quantrabbit.service`

- デプロイ実行（フォールバック直書き）
  - vm.sh の remote 実行でシェルの quoting 事情により失敗する場合は、下記の直接実行に退避:
    - `gcloud compute ssh fx-trader-vm --project quantrabbit --zone asia-northeast1-a \
       --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit \
       --command "sudo -u tossaki -H bash -lc 'cd /home/tossaki/QuantRabbit && git fetch --all -q || true && git checkout -q main || git checkout -b main origin/main || true && git pull --ff-only && if [ -d .venv ]; then source .venv/bin/activate && pip install -r requirements.txt; fi'"`
    - `gcloud compute ssh fx-trader-vm --project quantrabbit --zone asia-northeast1-a \
       --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit \
       --command "sudo systemctl daemon-reload && sudo systemctl restart quantrabbit.service && sudo systemctl status --no-pager -l quantrabbit.service || true"`

- 稼働確認
  - ステータス: `scripts/vm.sh ... tail -s quantrabbit.service -t`
  - 直接: `gcloud compute ssh ... --command "journalctl -u quantrabbit.service -n 200 -f --output=short-iso"`

### 7.2.1 既知事象: vm.sh の remote コマンド quoting
- まれに `scripts/vm.sh deploy` の内部で base64+eval を用いたリモート実行の quoting が壊れ、
  期待通りに複数行コマンドが評価されないことがあります（`set` の出力が流れる等）。
- その場合は上記「フォールバック直書き」を用いるか、単純な `--command` で小さな単位に分けて実行してください。

### 7.2.2 直近のエラー履歴（参考）
- 旧ビルドで `main.py` に SyntaxError があり、`journalctl` に以下が記録されました。
  - `File "/home/tossaki/QuantRabbit/main.py", line 349` の `if os.getenv(SCALP_TACTICAL, 0).strip().lower() not in {, 0, false, no}:`
- 現行ブランチでは修正済み。類似箇所では以下を推奨します。
  - `if str(os.getenv("SCALP_TACTICAL", "0")).strip().lower() not in {"", "0", "false", "no"}: ...`


8. チーム運用ルール
    1. 1 ファイル = 1 PR、Squash Merge、CI green 必須
    2. コード規約：black / ruff / mypy (optional)
    3. 秘匿情報は 絶対に Git に push しない
    4. 不具合・改善は GitHub Issue で管理（ラベル: bug/feat/doc/ops）
    5. ポジション問い合わせ対応: ユーザが「今のポジ？」「これ何？」などと聞いたら、当該会話文脈と開いているログを優先し、最新の fills / open trades を即答する。基本手順:
        - 直近ログ（例: `logs/oanda/transactions_*.jsonl`, `logs/orders.db`, `logs/trades.db`）から最新の建玉とサイズ/向き/TP/SL/時刻を抜粋
        - オープンが無ければ「今はフラット」で、直近クローズと理由（SL/TP/手動）を短く添える
        - サイズ異常や最小単位クランプが原因なら、どの設定（`ORDER_MIN_UNITS_*`, `_MIN_ORDER_UNITS` など）で決まったかも一言で示す
        - 過去日の無関係なトレードには飛ばず、ユーザが示したファイル/タイムスタンプを優先（「今の話」であることを前提に回答）
        - ローカルで情報が欠ける/古い場合は必ず VM ログ（`/home/tossaki/QuantRabbit/logs/*`）か OANDA API を直接確認して最新状態を答える（手元データだけで推測しない）
        - 日付意識: まず `date` などで今日の日付を確認し、開いているログが今日より古ければ「手元は<日付>まで、最新はVM/OANDAを確認する」と明示してから最新データを取りに行く（「今の話」がデフォルト）
        - 代表的な確認コマンド（必要に応じて都度実行）
            - VM ログ: `scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm sql -f /home/tossaki/QuantRabbit/logs/trades.db -q "select ticket_id,pocket,client_order_id,units,entry_time,close_time,pl_pips from trades order by entry_time desc limit 5;" -t`
            - 取引履歴: `gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command "sqlite3 /home/tossaki/QuantRabbit/logs/orders.db 'select ts,pocket,side,units,client_order_id,status from orders order by ts desc limit 5;'"` 
            - OANDA 現在ポジ: `curl -s -H "Authorization: Bearer $OANDA_TOKEN" "https://api-fxtrade.oanda.com/v3/accounts/$OANDA_ACCOUNT/openTrades" | jq '.trades[] | {id, instrument, currentUnits, price, takeProfit, stopLoss}'`
        - ポジ/タグ欠損トラブルシュート（運用メモ）
            - オープン確認は VM で `.venv` 有効化後 `python - <<'PY' ... PositionManager().get_open_positions()`。trades.db はクローズ済みのみなので、openTrades API と突合する。
            - thesis/fast_cut/kill が空の micro/scalp はオーファン扱い。2025-12-09 以降の MomentumBurst は fast_cut/kill メタ付与済みだが、それ以前の建玉は空のまま残るので必要なら手動クローズする。
            - manual ポケットは自動エグジット対象外。意図しない manual が残っていれば明示的にクローズする。

⸻

9. 参考ドキュメント
    - README.md – 🍵 ユーザ向け概観
    - パッチ適用の推奨シーケンス.pdf – 開発手順ガイド
    - 全体仕様まとめ（最終版）.pdf – アーキテクチャ詳細
    - OFL.txt + ZenOldMincho-*.ttf – 付属フォントライセンス

---

## 10. GCP アクセス / デプロイ指針（最新版）

- 原則 OS Login + IAP を使用（メタデータ `ssh-keys` は OS Login 有効時に無視）。
- まずは Doctor で前提を自動検診し、鍵の生成/登録まで一括実施：
  - `scripts/gcloud_doctor.sh -p <PROJECT> -z asia-northeast1-a -m fx-trader-vm -E -S -G [-t -k ~/.ssh/gcp_oslogin_qr] -c`
- デプロイは `scripts/deploy_to_vm.sh` を使用：
  - 例（IAP 経由/依存更新込み）: `scripts/deploy_to_vm.sh -i -t -k ~/.ssh/gcp_oslogin_qr -p <PROJECT>`
- 詳細手順・背景は `docs/GCP_DEPLOY_SETUP.md` を参照。

### 10.1 事前健診（Doctor）

- gcloud 未導入時は `scripts/install_gcloud.sh` で導入。
- 推奨実行: `scripts/gcloud_doctor.sh -p <PROJECT> -z asia-northeast1-a -m fx-trader-vm -E -S -G [-t -k ~/.ssh/gcp_oslogin_qr] -c`
  - `-E`: Compute API 自動有効化、`-S`: OS Login 鍵登録、`-G`: SSH鍵が無い場合に生成。
- `scripts/deploy_to_vm.sh` は内部で Doctor を呼び出し、前提不備は早期失敗＋対処ガイドを表示。
- 詳細は `docs/GCP_DEPLOY_SETUP.md` を参照。

### 10.2 ヘッドレス（サービスアカウント）運用

- アクティブなユーザーアカウントが無い環境でも、Service Account(SA) で gcloud を操作できる。
- `scripts/gcloud_doctor.sh` は `-K <SA_KEYFILE>` 指定時、アカウント不在なら SA キーで自動有効化する。
- `scripts/deploy_to_vm.sh` は `-K <SA_KEYFILE> / -A <SA_ACCOUNT>` を受け取り、Compute/IAP/OS Login を SA で実行可能。
- 必須ロール例: `roles/compute.osAdminLogin`, `roles/compute.instanceAdmin.v1`, （IAP利用時）`roles/iap.tunnelResourceAccessor`。

### 10.3 クイックコマンド（quantrabbit 固定）

```bash
# プロジェクト/ゾーン/インスタンス（実値）
export PROJ=quantrabbit
export ZONE=asia-northeast1-a
export INST=fx-trader-vm

# 0) gcloud が無い場合の導入
scripts/install_gcloud.sh

# 1) 事前健診（Compute API / OS Login / IAP / SSH 検証と鍵登録まで）
scripts/gcloud_doctor.sh -p "$PROJ" -z "$ZONE" -m "$INST" \
  -E -S -G -t -k ~/.ssh/gcp_oslogin_qr -c

# 2) IAP 経由の疎通確認（単体）
gcloud compute ssh "$INST" --project "$PROJ" --zone "$ZONE" \
  --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_qr \
  --command 'echo [vm] hello'

# 3) デプロイ（venv 依存更新あり / IAP 経由）
scripts/deploy_to_vm.sh -i -t -k ~/.ssh/gcp_oslogin_qr -p "$PROJ"

# 4) ログ追尾（IAP 経由）
gcloud compute ssh "$INST" --project "$PROJ" --zone "$ZONE" \
  --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_qr \
  --command 'journalctl -u quantrabbit.service -f'

# 5) ヘッドレス（サービスアカウント）で診断/デプロイ
export SA=qr-deployer@${PROJ}.iam.gserviceaccount.com
export SA_KEY=~/.gcp/qr-deployer.json

# 診断（ユーザー非アクティブでも可）
scripts/gcloud_doctor.sh -p "$PROJ" -z "$ZONE" -m "$INST" \
  -K "$SA_KEY" -A "$SA" -E -S -G -t -k ~/.ssh/gcp_oslogin_qr -c

# デプロイ（SA インパーソネート）
scripts/deploy_to_vm.sh -p "$PROJ" -t -k ~/.ssh/gcp_oslogin_qr \
  -K "$SA_KEY" -A "$SA" -i
```

### 10.4 VM ブランチ運用と再起動ガイド
- 本番 VM (`fx-trader-vm`) は原則 `main` を稼働ブランチとする。検証や一時対応で別ブランチを動かす場合は必ず記録し、元に戻すときも明示する。SL/TP などの挙動差分はブランチ依存なので、切替時に注意。
- 稼働中のコードは「作業ツリーのブランチ＋最後に `systemctl restart quantrabbit` を実行した時点の内容」。`git checkout` だけではプロセスは変わらない。
- ブランチ切替前に `git status` でクリーンを確認し、未コミット・未 stash のまま切替えない。`scripts/deploy_to_vm.sh -b <branch>` を使うと自動 stash 付きで安全。
- 再起動前に `git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD` でブランチ/HEAD を確認し、journal にも再起動時のブランチをメモする。
- スタッシュが溜まると意図しないロールバックや SL/TP 設定の再旧化を招くため、不要になったら `git stash drop` で整理する。
- config/tuning_overrides.yaml・fixtures/*.json などのローカル調整を維持したい場合は stash/commit で保全し、ブランチ切替後に明示的に `git stash pop` してから再起動する。

---

## 11. タスク運用ルール（共通）

- タスクファイル: 本リポの全タスクは `docs/TASKS.md` を単一の台帳として管理する（正本）。
- 適用範囲: 機能開発/バグ修正/運用改善/ドキュメント更新など、すべての作業タスク。
- 位置付け: オンライン自動チューニング関連は従来どおり `docs/autotune_taskboard.md` を使用しつつ、必要に応じて `docs/TASKS.md` からリンクする。

### 11.1 運用フロー

1. タスク発生時: `docs/TASKS.md` の「Open Tasks」に新規エントリを追加する。
2. 作業中: 当該エントリを逐次更新し、進め方は同ファイルの計画（Plan）を参照しながら進行する。
3. 完了時: エントリを「Archive」に移し、完了日・対応 PR/コミット・要約を追記してアーカイブする。

### 11.2 記載項目（推奨）

- ID（例: `T-YYYYMMDD-###`）
- Title（簡潔な件名）
- Status（`todo | in-progress | review | done`）
- Priority（`P1 | P2 | P3`）
- Owner（担当）
- Scope/Paths（対象ファイルやディレクトリ）
- Context（関連 Issue/PR、参考リンク、仕様箇所）
- Acceptance Criteria（受入条件）
- Plan（主要ステップ。エージェントの `update_plan` と整合）
- Notes（補足、決定メモ）

### 11.3 テンプレート

以下テンプレートを `docs/TASKS.md` に記載済み。新規タスクはこれを複製して使用する。

```md
- [ ] ID: T-YYYYMMDD-001
  Title: <短い件名>
  Status: todo | in-progress | review | done
  Priority: P1 | P2 | P3
  Owner: <担当>
  Scope/Paths: <例> AGENTS.md, docs/TASKS.md
  Context: <Issue/PR/仕様リンク>
  Acceptance:
    - <受入条件1>
    - <受入条件2>
  Plan:
    - <主要ステップ1>
    - <主要ステップ2>
  Notes:
    - <補足>
```

運用メモ
- `docs/TASKS.md` は頻繁に更新されるため、コミットメッセージに `[Task:<ID>]` を含めて追跡性を確保する。
- 1 ファイル = 1 PR の原則は維持するが、台帳更新（`docs/TASKS.md`）は同時反映可。
- 自動チューニング関連 ToDo は引き続き `docs/autotune_taskboard.md` に追記し、完了後は同ファイル内で状態をアーカイブする。

## 12. 調査用ローソクの取得手順（VM/OANDA）
- OANDAから任意期間のローソクを取得する（例: M1 2025-12-08 03:40Z〜05:30Z）
  ```bash
  gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a \
    --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command \
    "sudo -u tossaki -H bash -lc 'cd /home/tossaki/QuantRabbit && source .venv/bin/activate && PYTHONPATH=. \
      python scripts/fetch_candles.py --instrument USD_JPY --granularity M1 \
      --start 2025-12-08T03:40:00Z --end 2025-12-08T05:30:00Z \
      --out logs/candles_USDJPY_M1_20251208_0340_0530.json'"
  ```
- ローカルへ持ち帰る（/home/tossaki 以下に直接 scp できない場合は /tmp 経由）
  ```bash
  gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a \
    --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command \
    "sudo cp /home/tossaki/QuantRabbit/logs/candles_USDJPY_M1_20251208_0340_0530.json /tmp/"

  gcloud compute scp --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap \
    --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit \
    fx-trader-vm:/tmp/candles_USDJPY_M1_20251208_0340_0530.json ./remote_logs/
  ```
- 取得済みファイル（例）: `remote_logs/candles_USDJPY_M1_20251208_0340_0530.json`

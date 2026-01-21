# AGENTS.md – QuantRabbit Agent Specification（整理版）

## 1. ミッション / 運用前提
> 狙い: USD/JPY で 1 日 +100 pips を狙う 24/7 無裁量トレーディング・エージェント。  
> 境界: 発注・リスクは機械的、曖昧判断は GPT‑5 系（既定 gpt-5-mini）。
- ニュース連動パイプラインは撤去済み（`news_fetcher` / `summary_ingestor` / NewsSpike は無効）。
- 現行デフォルト: `WORKER_ONLY_MODE=true` / `MAIN_TRADING_ENABLED=0`。共通 `exit_manager` はスタブ化され、エントリー/EXIT は各戦略ワーカー＋専用 `exit_worker` が担当。
- 発注経路はワーカー → `utils/signal_bus.py`（`logs/signals.db`）→ main 関所で confidence 順に選抜・ロット配分 → OANDA の一本化が既定（`SIGNAL_GATE_ENABLED=1` / `ORDER_FORWARD_TO_SIGNAL_GATE=1`）。ワーカー直接発注に戻す場合のみ両フラグを 0 にする。
- 運用モード（2025-12 攻め設定）: マージン活用を 85–92% 目安に引き上げ、ロット上限を拡大（`RISK_MAX_LOT` 既定 10.0 lot）。手動ポジションを含めた総エクスポージャでガードし、PF/勝率の悪い戦略は自動ブロック。必要に応じて `PERF_GUARD_GLOBAL_ENABLED=0` で解除する。
- エージェントの役割:
  - VM 上で常時ログ・オーダーを監視
  - 手動玉を含めたエクスポージャを高水準で維持
  - PF/勝率が悪化した戦略・時間帯を自動ブロック
  - マージン拒否やタグ欠損を検知したら即パラメータ更新＆デプロイ
  - ユーザ手動トレードと併走し、ポジション総量を管理
  - 最優先ゴールは「資産を劇的に増やす」。必要なリスクテイクと調整を即断・即実行
- 運用/デプロイ手順は `README.md` と `docs/` を参照。
- マージン余力判定の落とし穴（2025-12-29対応済み）: fast_scalp が shared_state 欠落時に余力0と誤判定し全シグナルをスキップした事象あり。`workers/fast_scalp/worker.py` で `get_account_snapshot()` フォールバックを追加済み。再発時はログに `refreshed account snapshot equity=... margin_available=...` が出ることを確認し、0判定が続く場合は OANDA snapshot 取得を先に疑う。

## 2. システム概要とフロー
- データ → 判定 → 発注: Tick 取得 → Candle 確定 → Factors 算出 → Regime/Focus → GPT Decider → Strategy Plugins → Risk Guard → Order Manager → ログ。
- コンポーネントと I/O

  | レイヤ | 担当 | 主な入出力 |
  |--------|------|------------|
  | DataFetcher | `market_data/*` | Tick JSON, Candle dict |
  | IndicatorEngine | `indicators/*` | Candle deque → Factors dict {ma10, rsi, …} |
  | Regime & Focus | `analysis/regime_classifier.py` / `analysis/focus_decider.py` | Factors → macro/micro レジーム・`weight_macro` |
  | GPT Decider | `analysis/gpt_decider.py` | focus + perf → `GPTDecision` |
  | Strategy Plugin | `strategies/*` | Factors → `StrategyDecision` または None |
  | Exit (専用ワーカー) | `workers/*/exit_worker.py` | pocket 別 open positions → exit 指示（PnL>0 決済が原則） |
  | Signal Gate | `utils/signal_bus.py` / `main.py` | ワーカー enqueue → confidence 順に選抜・ロット配分 → OrderIntent |
  | Risk Guard | `execution/risk_guard.py` | lot/SL/TP/pocket → 可否・調整値 |
  | Order Manager | `execution/order_manager.py` | units/sl/tp/client_order_id/tag → OANDA ticket |
  | Logger | `logs/*.db` | 全コンポーネントが INSERT |
- ライフサイクル
  - Startup: `env.toml` 読込 → Secrets 確認 → WebSocket 接続。
  - 60s タクト（main 有効時のみ）: 新ローソク → Factors 更新 → Regime/Focus → GPT decision → Strategy loop（confidence スケーリング + ステージ判定）→ exit_worker → risk_guard → order_manager → `trades.db` / `metrics.db` ログ。ワーカーは発注前に `signal_bus` へ enqueue、main の関所が confidence 順に選択・lot配分して発注。
  - タクト要件: 正秒同期（±500 ms）、締切 55 s 超でサイクル破棄（バックログ禁止）、`monotonic()` で `decision_latency_ms` 計測。
- Background: `utils/backup_to_gcs.sh` による nightly logs バックアップ + `/etc/cron.hourly/qr-gcs-backup-core` による GCS 退避（自動）。

## 3. データスキーマと単位
- 共通スキーマ（pydantic 互換）

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

- 単位と用語

  | 用語 | 定義 |
  |------|------|
  | `pip` | USD/JPY の 1 pip = 0.01。入力/出力は pip 単位を明記。 |
  | `point` | 0.001。OANDA REST 価格の丸め単位。 |
  | `lot` | 1 lot = 100,000 units。`units = round(lot * 100000)`。 |
  | `pocket` | `micro` = 短期テクニカル、`macro` = レジーム、`scalp` = スカルプ。口座資金を `pocket_ratio` で按分。 |
  | `weight_macro` | 0.0〜1.0。`pocket_macro = pocket_total * weight_macro`（運用では macro 上限 30%）。 |

- 価格計算: `price_from_pips("BUY", entry, sl_pips) = round(entry - sl_pips * 0.01, 3)`、`price_from_pips("SELL", entry, sl_pips) = round(entry + sl_pips * 0.01, 3)`。
- `client_order_id = f"qr-{ts_ms}-{focus_tag}-{tag}"`（9 桁以内のハッシュで重複防止）。Exit も同形式で 90 日ユニーク。

## 4. エントリー / Exit / リスク制御
- Strategy フロー: Focus/GPT decision → `ranked_strategies` 順に Strategy Plugin を呼び、`StrategyDecision` または None を返す。`None` はノートレード。
- Confidence スケーリング: `confidence`(0–100) を pocket 割当 lot に掛け、最小 0.2〜最大 1.0 の段階的エントリー。`STAGE_RATIOS` に従い `_stage_conditions_met` を通過したステージのみ追撃。
- Exit: 各戦略の `exit_worker` が最低保有時間とテクニカル/レンジ判定を踏まえ、PnL>0 決済が原則（強制 DD/ヘルスのみ例外）。共通 `execution/exit_manager.py` は常に空を返す互換スタブ。`execution/stage_tracker` がクールダウンと方向別ブロックを管理。
- エントリー詰まり対策（必要時のみ）: `ENTRY_TECH_FAILOPEN=1` で tech ブロック時に小ロットで通す（`ENTRY_TECH_FAILOPEN_MIN_SCORE`, `ENTRY_TECH_FAILOPEN_SIZE_MULT` で緩和幅を制御）
- Release gate: PF>1.1、勝率>52%、最大 DD<5% を 2 週間連続で満たすと実弾へ昇格。
- リスク計算とロット:
  - `pocket_equity = account_equity * pocket_ratio`
  - `POCKET_MAX_RATIOS` は macro/micro/scalp/scalp_fast すべて 0.85 を起点に ATR・PF・free_margin で動的スケールし、最終値を 0.92〜1.0 にクランプ（scalp_fast は scalp から 0.35 割合で分岐）
  - `risk_pct = 0.02`、`risk_amount = pocket_equity * risk_pct`
  - 1 lot の pip 価値は 1000 JPY → `lot = min(MAX_LOT, round(risk_amount / (sl_pips * 1000), 3))` → `units = int(round(lot * 100000))`、`abs(units) < 1000` は発注しない
  - 最小ロット下限: macro 0.1, micro 0.0, scalp 0.05（env で上書き可）
  - `clamp_sl_tp(price, sl, tp, side)` で 0.001 丸め、SL/TP 逆転時は 0.1 バッファ
- OANDA API マッピング

  | Strategy action | REST 注文 | units 符号 | SL/TP | 備考 |
  |-----------------|-----------|------------|-------|------|
  | `OPEN_LONG` | `MARKET` | `+abs(units)` | `stopLossOnFill`, `takeProfitOnFill` | `timeInForce=FOK`, `positionFill=DEFAULT` |
  | `OPEN_SHORT` | `MARKET` | `-abs(units)` | 同上 | ask/bid 逆転チェック後送信 |
  | `CLOSE` | `MARKET` | 既存ポジの反対売買 | 指定なし | `OrderCancelReplace` で逆指値削除 |
  | `HOLD` | 送信なし | 0 | なし | Strategy ループ継続 |

- 発注冪等性とリトライ: `client_order_id` は 90 日ユニーク。OANDA 429/5xx/timeout は 0.5s→1.5s→3.5s（+100 ms ジッター）で最大 3 回リトライ。同一 ID を再利用。WebSocket 停止検知時は `halt_reason="stream_disconnected"` を残して停止。

## 5. データ鮮度・ログ・検証
- データ鮮度: `max_data_lag_ms = 3000` 超は `DataFetcher` が `stale=True` を返し Risk Guard が発注拒否。Candle 確定は `tick.ts_ms // 60000` 変化で判定し、終値は最後の mid。`volume=0` は `missing_bar` としてログ。
- ログ永続化: 本番ログは VM `/home/tossaki/QuantRabbit/logs/` のみを真とする。ローカル `logs/*.db` は参考扱い。
  - GCS 自動退避: `GCS_BACKUP_BUCKET` を `/etc/quantrabbit.env` に設定し、`/etc/cron.hourly/qr-gcs-backup-core` で毎時アップロード。保存先は `gs://$GCS_BACKUP_BUCKET/qr-logs/<hostname>/core_*.tar`。
  - Storage から読むとき（VM 上）:
    - 一覧: `sudo -u tossaki -H /usr/local/bin/qr-gcs-fetch-core list`
    - 最新を展開: `sudo -u tossaki -H /usr/local/bin/qr-gcs-fetch-core latest /tmp/qr_gcs_restore`
    - 指定取得: `sudo -u tossaki -H /usr/local/bin/qr-gcs-fetch-core get gs://<bucket>/qr-logs/<host>/core_YYYYmmddTHHMMSSZ.tar /tmp/qr_gcs_restore`
  - 参照時の補助: `QR_BACKUP_HOST=<hostname>` を指定すると別ホストの退避も読める。
  - 日次集計例
    ```bash
    scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm sql -f /home/tossaki/QuantRabbit/logs/trades.db -q "SELECT DATE(close_time), COUNT(*), ROUND(SUM(pl_pips),2) FROM trades WHERE DATE(close_time)=DATE('now') GROUP BY 1;" -t
    ```
  - 直近オーダー例
    ```bash
    gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command "sqlite3 /home/tossaki/QuantRabbit/logs/orders.db 'select ts,pocket,side,units,client_order_id,status from orders order by ts desc limit 5;'"
    ```
- 検証パイプライン: `logs/replay/*.jsonl` で Record、Strategy Plugin は Backtest で再現性確認、Shadow では本番 tick + 仮想アカウントで `OrderIntent` と `risk_guard` 拒否理由を比較。
- 観測指標: `decision_latency_ms`, `data_lag_ms`, `order_success_rate`, `reject_rate`, `gpt_timeout_rate`, `pnl_day_pips`, `drawdown_pct`。SLO: `decision_latency_ms p95 < 2000`, `order_success_rate ≥ 0.995`, `data_lag_ms p95 < 1500`, `drawdown_pct max < 0.18`, `gpt_timeout_rate < 0.05`。Alert: SLO 違反、`healthbeat` 欠損 5 分超、`token_usage ≥ 0.8 * max_month_tokens`, `order reject` 連続 3 件。

## 6. 安全装置と状態遷移
- 安全装置: Pocket DD micro 5% / macro 15% / scalp 3% / scalp_fast 2% で該当 pocket 新規停止、Global DD 20% でプロセス終了、Event モード（指標 ±30 min）は micro 新規禁止、Timeout: GPT 7 s / OANDA REST 5 s 再試行、Healthbeat は `main.py` から 5 min ping。
- 状態遷移

  | 状態 | 遷移条件 | 動作 |
  |------|----------|------|
  | `NORMAL` | 初期 | 全 pocket 取引許可 |
  | `EVENT_LOCK` | 経済指標 ±30 min | `micro` 新規停止、建玉縮小ロジック発動 |
  | `MICRO_STOP` | `micro` pocket DD ≥5% | `micro` 決済のみ、`macro` 継続 |
  | `GLOBAL_STOP` | Global DD ≥20% または `healthbeat` 欠損>10 min | 全取引停止、プロセス終了 |
  | `RECOVERY` | DD が閾値の 80% 未満、24h 経過 | 新規建玉再開前に `main.py` ドライラン |

## 7. トークン & コストガード
- `.cache/token_usage.json` に月累計。`env.toml` の `openai_max_month_tokens` で上限を設定。
- 超過時のフォールバック JSON:
  ```json
  {"focus_tag":"hybrid","weight_macro":0.5,"weight_scalp":0.15,"ranked_strategies":["TrendMA","H1Momentum","Donchian55","BB_RSI"],"reason":"fallback"}
  ```
- GPT 失敗時は過去 5 分の決定を再利用し、`reason="reuse_previous"` / `decision_latency_ms=9000` として記録。フォールバックは最後の手段とし、影響（focus 固定・重複注文リスクなど）を共有して限定的に許可する。

## 8. レンジモードとオンラインチューニング
- レンジモード強化（2025-10）
  - 判定: `analysis/range_guard.detect_range_mode` が M1 の `ADX<=24`, `BBW<=0.24`, `ATR<=7` を主に見つつ圧縮/ボラ比の複合スコア（0.66 以上）や `compression_trigger` で `range_mode` を返す。`metrics.composite` と `reason` をログ。
  - エントリー制御: `range_mode=True` 中は macro 新規を抑制し、許可戦略を BB 逆張り（`BB_RSI` など）に限定。`focus_tag` を `micro` へ縮退、`weight_macro` 上限 0.15。
  - 利確/損切り: レンジ中は各 `exit_worker` が TP/トレイル/lock をタイトに（目安 1.5〜2.0 pips の RR≒1:1、fast_scalp/micro/macro で閾値別設定）。共通 `exit_manager` は使用しない。
  - 分割利確: `execution/order_manager.plan_partial_reductions` はレンジ中にしきい値を macro 16/22, micro 10/16, scalp 6/10 pips に低減。ステージ/再入場は `execution/stage_tracker` が方向別クールダウン、連続 3 敗で 15 分ブロック、勝敗に応じてロット係数縮小（マーチン禁止）。
- オンライン自動チューニング
  - 5〜15 分間隔で `scripts/run_online_tuner.py` を呼び、Exit 感度や入口ゲート・quiet_low_vol 配分を小幅調整（ホットパスは対象外）。
  - 既定: `TUNER_ENABLE=true`, `TUNER_SHADOW_MODE=true`。`config/tuning_history/` に履歴だけを残し、本番パラメータ (`config/tuning_overrides.yaml`) は書き換えない。
  - 本適用時: `TUNER_SHADOW_MODE=false` → `scripts/apply_override.py` で `config/tuning_overlay.yaml` を生成しランタイム読み込み。
  - ToDo/検証タスクは `docs/autotune_taskboard.md` に集約し、完了後は同ファイルでアーカイブ。

## 9. デプロイ / GCP アクセス
- 原則 OS Login + IAP。`scripts/gcloud_doctor.sh` で前提検診（Compute API 有効化 / OS Login 鍵登録 / IAP 確認）→ `scripts/deploy_to_vm.sh` でデプロイ。
- クイックコマンド（proj/zone/inst は適宜置換）

```bash
# Doctor（一括検診 + 鍵登録）
scripts/gcloud_doctor.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -E -S -G -t -k ~/.ssh/gcp_oslogin_qr -c
# デプロイ（venv 依存更新付き/IAP）
scripts/deploy_to_vm.sh -i -t -k ~/.ssh/gcp_oslogin_qr -p quantrabbit
# ログ追尾
gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_qr --command 'journalctl -u quantrabbit.service -f'
```

- フォールバック（vm.sh が失敗する場合の直書き）
  1) `gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command "sudo -u tossaki -H bash -lc 'cd /home/tossaki/QuantRabbit && git fetch --all -q || true && git checkout -q main || git checkout -b main origin/main || true && git pull --ff-only && if [ -d .venv ]; then source .venv/bin/activate && pip install -r requirements.txt; fi'"`
  2) `gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command "sudo systemctl daemon-reload && sudo systemctl restart quantrabbit.service && sudo systemctl status --no-pager -l quantrabbit.service || true"`
- IAP/SSH が不安定な場合の無SSH反映（metadata 経由・推奨）
  - 目的: `failed to connect to backend` 等で SSH/IAP が落ちても反映を止めない
  - 手順: `scripts/deploy_via_metadata.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -b main -i`
  - 健康レポート取得: `-r` を付ける（serial に status/trades/signal を出力）
  - 仕組み: `startup-script` に `deploy_id` を埋め込み、`/var/lib/quantrabbit/deploy_id` で重複実行を抑止
  - 後片付け（任意）: `gcloud compute instances remove-metadata fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --keys startup-script`
- IAP/SSH 落ちが続く場合の予防
  - 目的: SSH サービス/guest agent 停止で port 22 が閉じる事象を自動復旧
  - 監視: `systemd/quant-ssh-watchdog.timer`（1分間隔）→ `scripts/ssh_watchdog.sh`
  - 導入: `sudo bash scripts/install_trading_services.sh --units "quant-ssh-watchdog.service quant-ssh-watchdog.timer"`
  - 備考: `/home/tossaki/QuantRabbit` 前提。ユーザが異なる場合は unit を編集。
- 情報不足対策（無SSHでも状態確認）
  - 目的: trades/signal の最終時刻を GCS に書き出し、外部から即確認
  - 監視: `systemd/quant-health-snapshot.timer`（1分間隔）→ `scripts/publish_health_snapshot.py`
  - 出力先: `ui_bucket_name` 優先、未設定なら `GCS_BACKUP_BUCKET`（`HEALTH_OBJECT_PATH` で上書き）
  - アップロード: `google-cloud-storage` → `gcloud/gsutil` → metadata REST の順にフォールバック（CLI 未導入でも送信可能）
  - 取得例: `gcloud storage cat gs://<bucket>/<object>`
  - ローカル: `/home/tossaki/QuantRabbit/logs/health_snapshot.json` にも保存（バックアップ対象）
- OS Login 権限不足時は `roles/compute.osAdminLogin` を付与（検証: `sudo -n true && echo SUDO_OK`）。本番 VM `fx-trader-vm` は原則 `main` ブランチ稼働。スタッシュ/未コミットはブランチ切替前に解消。
- VM 削除禁止。再起動やブランチ切替で代替し、`gcloud compute instances delete` 等には触れない。
- IAP/SSH 不調時の反映/確認（代替フロー）
  - 反映: `scripts/deploy_via_metadata.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -b main -i -e local/vm_env_overrides.env`
  - GitHub 到達不安定時: `scripts/deploy_bundle_to_gcs.sh -b quantrabbit-logs` → `scripts/deploy_via_metadata.sh ... -g gs://quantrabbit-logs/deploy/qr_bundle_*.tar.gz`
  - 確認: `gcloud compute instances get-serial-port-output fx-trader-vm --zone=asia-northeast1-a --project=quantrabbit --port=1 | rg 'startup-script|deploy_id'`
  - SSH 自己回復: `quant-ssh-watchdog.timer` が `ssh/sshd` と `google-guest-agent` を 1 分ごとに再起動監視
  - ヘルス可視化: `quant-health-snapshot.timer` が `/home/tossaki/QuantRabbit/logs/health_snapshot.json` を 1 分ごとに更新し、`ui_bucket_name`（未設定なら `GCS_BACKUP_BUCKET`）の `realtime/health_<hostname>.json` へ送信（orders/signals/trades/サービス状態を同梱）
  - UI 可視化: `quant-ui-snapshot.timer` が `realtime/ui_state.json` を 1 分ごとに更新（orders/signals/healthbeat を metrics に同梱）
  - まだ復帰しない場合: cloud-init の `user-data` で openssh/guest-agent を再投入 → `gcloud compute instances add-metadata fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --metadata-from-file user-data=local/cloudinit_ssh_repair.yaml` → reset

### 9.1 GCP 基盤/VM（API・IAM・ネットワーク）
- 有効化 API: `compute.googleapis.com` / `storage.googleapis.com` / `secretmanager.googleapis.com` / `logging.googleapis.com`（任意: `bigquery.googleapis.com`, Firestore 使用時は Firestore API）
- IAM ロール（運用者/SA）
  - OS Login/IAP: `roles/compute.osAdminLogin`, `roles/compute.instanceAdmin.v1`, `roles/iap.tunnelResourceAccessor`（IAP 経由時）
  - VM 実行権限: `roles/storage.objectAdmin`, `roles/logging.logWriter`, `roles/secretmanager.secretAccessor`（BQ 利用時は `roles/bigquery.dataEditor`/`roles/bigquery.jobUser`）
  - Firestore 利用時: `roles/datastore.user`（読み取りのみなら `roles/datastore.viewer`）
- VM/Compute
  - 例: e2-small / Ubuntu 22.04
  - OS Login 有効 + metadata `enable-oslogin=TRUE`
  - IAP 経由の場合は tcp:22 を 35.235.240.0/20 から許可
  - VM の Service Account を付与（Secret Manager/Storage/Logging/BQ の権限を持たせる）
- 認証 (ADC): VM SA が基本。SA キー運用時は `GOOGLE_APPLICATION_CREDENTIALS` を設定（または `gcloud auth application-default login`）
- VM 上で `gsutil`/`bq` を使う運用（バックアップ/Loooker/UI 初期化）では gcloud SDK を導入する
- OS Login 鍵: `scripts/gcloud_doctor.sh -S -G` で鍵生成/登録（TTL 30d）
- ブートストラップ手段
  - Terraform: `infra/terraform/main.tf`（metadata startup script が repo clone + venv 構築。`analysis/gpt_decider.py` / `indicators/calc_core.py` を上書きするため事前確認）
  - 単体VM: `startup_script.sh`（Secret Manager から `/etc/quantrabbit.env` を生成、OANDA 欠損時は `MOCK_TICK_STREAM=1` を付与して `quantrabbit.service` を起動）
  - 手動: 下記の最小ブートストラップ
- Cloud Logging 連携: `startup_script.sh` が Ops Agent 用の `config.yaml` を書き出す（エージェント導入済みの場合に有効）

### 9.2 VM サービス/環境ファイル
- main systemd: `ops/systemd/quantrabbit.service`（`EnvironmentFile=/etc/quantrabbit.env`）
- ワーカー/タイマー: `systemd/*.service` / `systemd/*.timer`（cleanup/autotune/level-map/各戦略）
  - まとめて導入: `scripts/install_trading_services.sh --all`
  - 個別導入: `scripts/install_trading_services.sh --units <unit>`
- `systemd/quant-level-map.service` は `PROJECT`/`BQ_PROJECT`/`GCS_BUCKET`/`GOOGLE_APPLICATION_CREDENTIALS` が埋め込みのため環境ごとに上書き
- `systemd/quant-autotune.service` は `AUTOTUNE_BQ_TABLE`/`AUTOTUNE_BQ_SETTINGS_TABLE` が固定値のため環境ごとに上書き
- `systemd/quant-autotune-ui.service` は 8088/TCP で待受け。外部公開する場合は FW/IAP 前提で設計
- `systemd/*.service` / `ops/systemd/quantrabbit.service` は `User=tossaki` と `/home/tossaki/QuantRabbit` 前提のため、VM ユーザが異なる場合は編集する
- `/etc/quantrabbit.env` と `config/env.toml` は内容を一致させる（systemd とアプリの参照元が異なる）
- `scripts/vm.sh` を使う場合は `scripts/vm.env` に PROJECT/ZONE/INSTANCE を固定
- `startup_script.sh` は OpenAI/OANDA の最小キーと `TUNER_*` を `/etc/quantrabbit.env` に固定で書き込むため、GCS/BQ/リスク系は手動追記が必要
- VM 内ブートストラップ（最小）

```bash
sudo -u <user> -H bash -lc '
  cd /home/<user>
  git clone https://github.com/Tosaaaki/QuantRabbit.git
  cd QuantRabbit
  python3 -m venv .venv && . .venv/bin/activate
  pip install -r requirements.txt
  cp config/env.example.toml config/env.toml
'
```

- `config/env.toml` 最低限: `gcp_project_id`, `gcp_location`, `GCS_BACKUP_BUCKET`, `ui_bucket_name`, `BQ_PROJECT/BQ_DATASET/BQ_TRADES_TABLE`, `oanda_account_id`, `oanda_token`, `oanda_practice`, `openai_model_decider`  
  Secret Manager を使わない場合は `openai_api_key` を追加
- デプロイは上記の `gcloud_doctor.sh` / `deploy_to_vm.sh` を使用

### 9.3 Storage / GCS
- バケット
  - `GCS_BACKUP_BUCKET`（ログ退避）, `ui_bucket_name`（UI）, 任意で `analytics_bucket_name`
  - env マップ: `GCS_UI_BUCKET`（= `ui_bucket_name`）, `GCS_ANALYTICS_BUCKET`（= `analytics_bucket_name`）
- バックアップ
  - 日次: `utils/backup_to_gcs.sh`
  - 定時: `/etc/cron.hourly/qr-gcs-backup-core`
  - 軽量整理: `scripts/cleanup_logs.sh`（`systemd/cleanup-qr-logs.timer`）
- UI スナップショット: `analytics/gcs_publisher.py` → `ui_state_object_path`（既定 `realtime/ui_state.json`）
- ロットインサイト: `analytics/lot_pattern_analyzer.py` → `analytics/lot_insights.json`（analytics bucket か UI bucket）
- エクスカーションレポート: `scripts/run_excursion_report.sh` → `gs://<ui_bucket>/excursion/`（`systemd/quant-excursion-report.timer`）
- 既定バケット: `utils/backup_to_gcs.sh` は `GCS_BACKUP_BUCKET` 未設定時 `fx-backups` に退避するため必ず指定
- UI の例外パス: `excursion_reports_dir`（ローカル参照先）、`excursion_latest_object_path`（GCS 参照先, 既定 `excursion/latest.txt`）
- `gsutil` がない場合は GCS アップロードがスキップされる（バックアップ/エクスカーション）

### 9.4 BigQuery / Firestore / UI
- BigQuery 環境変数（`/etc/quantrabbit.env` 側で指定）
  - 必須: `BQ_PROJECT`（未設定時は `GOOGLE_CLOUD_PROJECT`）, `BQ_DATASET`, `BQ_TRADES_TABLE`, `BQ_LOCATION`
  - 任意: `BQ_MAX_EXPORT`, `BQ_CANDLES_TABLE`, `BQ_REALTIME_METRICS_TABLE`, `BQ_RECOMMENDATION_TABLE`, `BQ_STRATEGY_MODEL`
- BQ ML を使う場合は `CREATE MODEL` 権限が必要（`analytics/strategy_optimizer_job.py`）
- 同期パイプライン: `scripts/run_sync_pipeline.py`（`analytics/bq_exporter.py` が dataset/table を自動作成）
- Looker/UI 初期化: `scripts/setup_looker_sources.sh`（UI 用 SA, bucket, dataset, view 作成）
- オートチューン: `AUTOTUNE_BQ_TABLE`, `AUTOTUNE_BQ_SETTINGS_TABLE`（`systemd/quant-autotune.service`）
- Realtime KPI: `analytics/realtime_metrics_job.py`（`BQ_REALTIME_METRICS_TABLE`, `REALTIME_METRICS_LOOKBACK_MIN`, `REALTIME_METRICS_RETENTION_H`）
- 戦略レコメンド: `analytics/strategy_optimizer_job.py`（`BQ_RECOMMENDATION_TABLE`, `BQ_STRATEGY_MODEL`）
- ローソク export: `analytics/candle_export_job.py`（`BQ_CANDLES_TABLE` と OANDA 認証）
- Firestore（任意）: `FIRESTORE_PROJECT`, `FIRESTORE_COLLECTION`, `FIRESTORE_DOCUMENT`, `FIRESTORE_STRATEGY_ENABLE`, `FIRESTORE_STRATEGY_TTL_SEC`, `FIRESTORE_STRATEGY_APPLY_SLTP`
- 戦略スコア書き込み: `scripts/generate_strategy_scores.py`（Firestore へ `strategy_scores/current` を上書き）
- Level map（任意）: `LEVEL_MAP_ENABLE`, `LEVEL_MAP_OBJECT_PATH`, `LEVEL_MAP_TTL_SEC`, `LEVEL_MAP_BUCKET`（未設定なら analytics/UI bucket）
- Cloud Run UI（任意）: `cloudrun/autotune_ui/`（Cloud Build/Run API 有効化、BQ 読み取り権限 + UI bucket 参照権限）
- 参考値: `gcp_pubsub_topic`, `ui_service_account_email` は現行コード未参照（設定しても動作には影響なし）
- Realtime/レコメンド/ローソク/スコアは systemd では未配備。Cloud Scheduler/Cloud Run か cron/systemd で起動する
- `RealtimeMetricsClient` の適用閾値は `REALTIME_METRICS_TTL` / `CONF_POLICY_*` で調整可能

### 9.5 Secret Manager / 認証情報
- 参照順: `openai_api_key` は Secret Manager を優先、その他は `PREFER_GCP_SECRET_MANAGER=1` で優先化
- Secret 名は `config/env.toml` のキー名と同一（例: `openai_api_key`, `oanda_token`, `oanda_account_id`）
- 推奨シークレット（最低限）
  - `openai_api_key`, `oanda_token`, `oanda_account_id`
  - 任意: `openai_model_decider`, `openai_model_summarizer`, `openai_max_month_tokens`
- Secret Manager の参照プロジェクトは `GCP_PROJECT` / `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_PROJECT_NUMBER` で決定（未設定だと `quantrabbit`）
- `scripts/refresh_env_from_gcp.py` で Secret/環境変数から `config/env.toml` を生成できる
- 例（作成）

```bash
gcloud secrets create openai_api_key --replication-policy="automatic"
printf "sk-..." | gcloud secrets versions add openai_api_key --data-file=-
```

- Secret Manager を使わない場合は `DISABLE_GCP_SECRET_MANAGER=1`

### 9.6 GPT 設定（Decider/Advisors）
- 参照順: GCP Secret Manager → 環境変数 → `config/env.toml`
- 必須キー（`config/env.toml` 例）

```toml
openai_api_key = "sk-..."
openai_model_decider = "gpt-5-mini"
openai_model_summarizer = "gpt-4o-mini"
openai_max_month_tokens = "300000"
```

- 環境変数マップ: `OPENAI_API_KEY`, `OPENAI_DECIDER_MODEL`, `OPENAI_SUMMARIZER_MODEL`, `OPENAI_MAX_MONTH_TOKENS`
- モデル解決: `openai_model_decider` → `openai_model` → `gpt-4o-mini`（`analysis/gpt_prompter.py`）
- GPT Decider 主要環境変数: `GPT_DECIDER_MAX_TOKENS`, `GPT_MAX_MODEL_ATTEMPTS`, `GPT_RETRY_BASE_SEC`, `GPT_RETRY_JITTER_SEC`, `GPT_MIN_CALL_INTERVAL_SEC`, `GPT_DECIDER_TEMPERATURE`
- Advisors の個別モデル: `OPENAI_MODEL_EXIT_ADVISOR`, `OPENAI_MODEL_STAGE_PLAN`, `OPENAI_MODEL_VOLATILITY`, `OPENAI_MODEL_PARTIAL_ADVISOR`, `OPENAI_MODEL_STRATEGY_CONF`, `OPENAI_MODEL_FOCUS_ADVISOR`, `OPENAI_MODEL_RR_ADVISOR`
- その他: `OPENAI_MODEL`（共通フォールバック）, `OPENAI_MODEL_OPTIMIZER`（戦略最適化の要約用）
- いずれも Secret Manager / `config/env.toml` に同名キーで設定可
- `OPENAI_API_KEY` が無い場合は Advisors が自動無効化される
- 強制優先は `PREFER_GCP_SECRET_MANAGER=1`
- GPT 失敗時の挙動・フォールバックは「7. トークン & コストガード」に準拠

## 10. チーム / タスク運用ルール
- 変更は必ず `git commit` → `git push` → VM 反映（`scripts/vm.sh ... deploy -i -t` 推奨）で行う。未コミット状態やローカル差し替えでの運用は不可。
- チームルール: 1 ファイル = 1 PR、Squash Merge、CI green。コード規約 black / ruff / mypy(optional)。秘匿情報は Git に置かない。Issue 管理: bug/feat/doc/ops ラベル。
- タスク台帳: `docs/TASKS.md` を正本とし、Open→進行→Archive の流れで更新。テンプレート・Plan 記載済み。オンラインチューニング ToDo は `docs/autotune_taskboard.md` に追記し完了後アーカイブ。
- ポジション問い合わせ対応: 直近ログを優先し最新建玉/サイズ/向き/TP/SL/時刻を即答。オープン無しなら「今はフラット」＋直近クローズ理由。サイズ異常時は決定した設定（`ORDER_MIN_UNITS_*` など）を明示。
  - 代表コマンド:
    ```bash
    scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm sql -f /home/tossaki/QuantRabbit/logs/trades.db -q "select ticket_id,pocket,client_order_id,units,entry_time,close_time,pl_pips from trades order by entry_time desc limit 5;" -t
    ```
  - OANDA open trades:
    ```bash
    curl -s -H "Authorization: Bearer $OANDA_TOKEN" "https://api-fxtrade.oanda.com/v3/accounts/$OANDA_ACCOUNT/openTrades" | jq '.trades[] | {id, instrument, currentUnits, price, takeProfit, stopLoss}'
    ```
- ローソク/チャート確認依頼: 指定時刻の市況を VM ログ（candles/transactions/orders）や OANDA API から取得し、MA/RSI/ADX/BBW なども含めて短く要約。手元が古い場合は VM/OANDA を必ず参照。
- 回答言語: すべて日本語。動作確認・テストは担当者が自前で実施し、ユーザへ依頼しない。
- 調査用ローソク取得例（VM/OANDA）

```bash
gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a \
  --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command \
  "sudo -u tossaki -H bash -lc 'cd /home/tossaki/QuantRabbit && source .venv/bin/activate && PYTHONPATH=. \
    python scripts/fetch_candles.py --instrument USD_JPY --granularity M1 \
    --start 2025-12-08T03:40:00Z --end 2025-12-08T05:30:00Z \
    --out logs/candles_USDJPY_M1_20251208_0340_0530.json'"
gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a \
  --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command \
  "sudo cp /home/tossaki/QuantRabbit/logs/candles_USDJPY_M1_20251208_0340_0530.json /tmp/"
gcloud compute scp --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap \
  --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit \
  fx-trader-vm:/tmp/candles_USDJPY_M1_20251208_0340_0530.json ./remote_logs/
```

## 11. 他GCP運用時のトレース要件
- 目的: 複数 GCP/複数 VM で発生した注文・ログ・バックアップを即時に相互参照できる状態を維持する。
- 環境識別子（必須・台帳管理）

  | 項目 | 例 | 用途 |
  |------|----|------|
  | `gcp_project_id` | quantrabbit | GCP 環境の一意識別 |
  | `gcp_location` / zone | asia-northeast1 / asia-northeast1-a | VM の所属・リージョン判別 |
  | instance name | fx-trader-vm | SSH/運用対象の特定 |
  | hostname | fx-trader-vm-01 | GCS 退避/復元のキー（`QR_BACKUP_HOST`） |
  | service account | qr-deployer@... | 権限起点の追跡 |
  | `GCS_BACKUP_BUCKET` | fx-backups-prod | ログ退避先 |
  | `ui_bucket_name` | fx-ui-realtime-prod | UI 連携先 |
  | `BQ_PROJECT` / `BQ_DATASET` / `BQ_TRADES_TABLE` | quantrabbit / quantrabbit / trades_raw | 集計・可視化 |
  | `oanda_account_id` / `oanda_practice` | 000-... / true | 取引口座の紐付け |
  | deploy branch / commit | main / abc1234 | 事象発生時のコード版 |
- 運用ルール
  - hostname は環境ごとにユニークにする（GCS 退避/復元のパス衝突防止）。
  - OANDA 口座は GCP 環境と 1 対 1 で固定し、混在運用はしない。
  - バケット/BQ は環境分離を原則とし、共有する場合は dataset/table 名に環境名を含める。
  - `config/env.toml` と `/etc/quantrabbit.env` の値を一致させる。
  - トレース時は `logs/*.db` → GCS 退避 → OANDA の順で突合。`QR_BACKUP_HOST` で別ホストを参照。
- クイック取得（VM 内）

```bash
hostname
cat /etc/quantrabbit.env | sed -n '1,120p'
cd /home/tossaki/QuantRabbit && git rev-parse --short HEAD
```

- 詳細手順は `docs/README_GCP_MULTI.md` と `docs/GCP_DEPLOY_SETUP.md` を参照。

## 12. 参考ドキュメント
- `README.md` – ユーザ向け概観
- `docs/GCP_DEPLOY_SETUP.md` – GCP/IAP/OS Login 設定詳細
- `パッチ適用の推奨シーケンス.pdf` – 開発手順ガイド
- `全体仕様まとめ（最終版）.pdf` – アーキテクチャ詳細
- `OFL.txt` + `ZenOldMincho-*.ttf` – 付属フォントライセンス

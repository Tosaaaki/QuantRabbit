# AGENT.me – QuantRabbit Agent Specification（整理版）

## 1. ミッション / 運用前提
> 狙い: USD/JPY で 1 日 +100 pips を狙う 24/7 無裁量トレーディング・エージェント。  
> 境界: 発注・リスクは機械的、曖昧判断は GPT‑5 系（既定 gpt-5-mini）。
- ニュース連動パイプラインは撤去済み（`news_fetcher` / `summary_ingestor` / NewsSpike は無効）。
 - 現行デフォルト: `WORKER_ONLY_MODE=true` / `MAIN_TRADING_ENABLED=0`。共通 `exit_manager` はスタブ化され、エントリー/EXIT は各戦略ワーカー＋専用 `exit_worker` が担当。
 - 運用モード（2025-12 攻め設定）: マージン活用を 85–92% 目安に引き上げ、ロット上限を拡大（`RISK_MAX_LOT` 既定 10.0lot）。手動ポジションを含めた総エクスポージャでガードし、PF/勝率の悪い戦略は自動ブロック。必要に応じて `PERF_GUARD_GLOBAL_ENABLED=0` で解除する。
- 運用/デプロイ手順は `README.md` と `docs/` を参照。

## 2. システム概要とフロー
- データ → 判定 → 発注: Tick 取得 → Candle 確定 → Factors 算出 → Regime/Focus → GPT Decider → Strategy Plugins → Risk Guard → Order Manager → ログ。
- コンポーネントと I/O

  | レイヤ | 担当 | 主な入出力 |
  |--------|------|------------|
  | DataFetcher | `market_data/*` | Tick JSON, Candle dict |
  | IndicatorEngine | `indicators/*` | Candle deque → Factors dict {ma10, rsi, …} |
  | Regime & Focus | `analysis/regime_classifier.py` / `focus_decider.py` | Factors → macro/micro レジーム・`weight_macro` |
  | GPT Decider | `analysis/gpt_decider.py` | focus + perf → `GPTDecision` |
  | Strategy Plugin | `strategies/*` | Factors → `StrategyDecision` または None |
  | Exit (専用ワーカー) | `workers/*/exit_worker.py` | pocket 別 open positions → exit 指示（PnL>0 決済が原則） |
  | Risk Guard | `execution/risk_guard.py` | lot/SL/TP/pocket → 可否・調整値 |
  | Order Manager | `execution/order_manager.py` | units/sl/tp/client_order_id/tag → OANDA ticket |
  | Logger | `logs/*.db` | 全コンポーネントが INSERT |
- ライフサイクル
  - Startup: `env.toml` 読込 → Secrets 確認 → WebSocket 接続。
  - 60s タクト（main 有効時のみ）: 新ローソク → Factors 更新 → Regime/Focus → GPT decision → Strategy loop（confidence スケーリング + ステージ判定）→ exit_worker → risk_guard → order_manager → `trades.db` / `metrics.db` ログ。
  - タクト要件: 正秒同期（±500 ms）、締切 55 s 超でサイクル破棄（バックログ禁止）、`monotonic()` で `decision_latency_ms` 計測。
  - Background: `backup_to_gcs.sh` による nightly logs バックアップ。

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
  | `weight_macro` | 0.0〜1.0。`pocket_macro = pocket_total * weight_macro`（運用では Macro 上限 30%）。 |

- 価格計算: `price_from_pips("BUY", entry, sl_pips) = round(entry - sl_pips * 0.01, 3)`、`price_from_pips("SELL", entry, sl_pips) = round(entry + sl_pips * 0.01, 3)`。
- `client_order_id = f"qr-{ts_ms}-{focus_tag}-{tag}"`（9 桁以内のハッシュで重複防止）。Exit も同形式で 90 日ユニーク。

## 4. エントリー / Exit / リスク制御
- Strategy フロー: Focus/GPT decision → `ranked_strategies` 順に Strategy Plugin を呼び、`StrategyDecision` または None を返す。`None` はノートレード。
- Confidence スケーリング: `confidence`(0–100) を pocket 割当 lot に掛け、最小 0.2〜最大 1.0 の段階的エントリー。`STAGE_RATIOS` に従い `_stage_conditions_met` を通過したステージのみ追撃。
- Exit: 各戦略の `exit_worker` が最低保有時間とテクニカル/レンジ判定を踏まえ、PnL>0 決済が原則（強制 DD/ヘルスのみ例外）。共通 `execution/exit_manager.py` は常に空を返す互換スタブ。`execution/stage_tracker` がクールダウンと方向別ブロックを管理。
- Release gate: PF>1.1、勝率>52%、最大 DD<5% を 2 週間連続で満たすと実弾へ昇格。
- リスク計算とロット: `pocket_equity = account_equity * pocket_ratio`。`POCKET_MAX_RATIOS` は macro/micro/scalp/scalp_fast すべて 0.85 を起点に ATR・PF・free_margin で動的スケールし、下限 0.92〜上限 1.0 にクランプ（scalp_fast は scalp から 0.35 割合で分岐）。`risk_pct = 0.02`、`risk_amount = pocket_equity * risk_pct`。1 lot の pip 価値は 1000 JPY → `lot = min(MAX_LOT, round(risk_amount / (sl_pips * 1000), 3))`、`units = int(round(lot * 100000))`。`abs(units) < 1000` は発注しない。最小ロット下限: macro 0.1, micro 0.0, scalp 0.05（env で上書き可）。`clamp_sl_tp(price, sl, tp, side)` で 0.001 丸め、SL/TP 逆転時は 0.1 バッファ。
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
  - 日次集計例  
    `scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm sql -f /home/tossaki/QuantRabbit/logs/trades.db -q "SELECT DATE(close_time), COUNT(*), ROUND(SUM(pl_pips),2) FROM trades WHERE DATE(close_time)=DATE('now') GROUP BY 1;" -t`
  - 直近オーダー例  
    `gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --ssh-key-file ~/.ssh/gcp_oslogin_quantrabbit --command "sqlite3 /home/tossaki/QuantRabbit/logs/orders.db 'select ts,pocket,side,units,client_order_id,status from orders order by ts desc limit 5;'"` 
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
- `.cache/token_usage.json` に月累計。`env.toml` の `openai.max_month_tokens` で上限を設定。
- 超過時のフォールバック JSON: `{"focus_tag":"hybrid","weight_macro":0.5,"weight_scalp":0.15,"ranked_strategies":["TrendMA","H1Momentum","Donchian55","BB_RSI"],"reason":"fallback"}`。
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
- OS Login 権限不足時は `roles/compute.osAdminLogin` を付与（検証: `sudo -n true && echo SUDO_OK`）。本番 VM `fx-trader-vm` は原則 `main` ブランチ稼働。スタッシュ/未コミットはブランチ切替前に解消。
- VM 削除禁止。再起動やブランチ切替で代替し、`gcloud compute instances delete` 等には触れない。

## 10. チーム / タスク運用ルール
- 変更は必ず `git commit` → `git push` → VM 反映（`scripts/vm.sh ... deploy -i -t` 推奨）で行う。未コミット状態やローカル差し替えでの運用は不可。
- チームルール: 1 ファイル = 1 PR、Squash Merge、CI green。コード規約 black / ruff / mypy(optional)。秘匿情報は Git に置かない。Issue 管理: bug/feat/doc/ops ラベル。
- タスク台帳: `docs/TASKS.md` を正本とし、Open→進行→Archive の流れで更新。テンプレート・Plan 記載済み。オンラインチューニング ToDo は `docs/autotune_taskboard.md` に追記し完了後アーカイブ。
- ポジション問い合わせ対応: 直近ログを優先し最新建玉/サイズ/向き/TP/SL/時刻を即答。オープン無しなら「今はフラット」＋直近クローズ理由。サイズ異常時は決定した設定（`ORDER_MIN_UNITS_*` など）を明示。
  - 代表コマンド: `scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm sql -f /home/tossaki/QuantRabbit/logs/trades.db -q "select ticket_id,pocket,client_order_id,units,entry_time,close_time,pl_pips from trades order by entry_time desc limit 5;" -t`
  - OANDA open trades: `curl -s -H "Authorization: Bearer $OANDA_TOKEN" "https://api-fxtrade.oanda.com/v3/accounts/$OANDA_ACCOUNT/openTrades" | jq '.trades[] | {id, instrument, currentUnits, price, takeProfit, stopLoss}'`
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

## 11. 参考ドキュメント
- `README.md` – ユーザ向け概観
- `docs/GCP_DEPLOY_SETUP.md` – GCP/IAP/OS Login 設定詳細
- `パッチ適用の推奨シーケンス.pdf` – 開発手順ガイド
- `全体仕様まとめ（最終版）.pdf` – アーキテクチャ詳細
- `OFL.txt` + `ZenOldMincho-*.ttf` – 付属フォントライセンス

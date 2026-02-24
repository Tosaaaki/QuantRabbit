# Architecture Overview

## 1. システム概要とフロー
- データ → 判定 → 発注: Tick 取得 → Candle 確定 → Factors 算出 → strategy control 制約 → Strategy Plugins（ENTRY/EXIT ワーカー別）→ Risk Guard → `quant-order-manager` → ログ。
- V2 では monolithic 主制御（`main.py`起動）を本番から外し、`quantrabbit.service` は廃止対象。

## 2. コンポーネントと I/O

| レイヤ | 担当 | 主な入出力 |
|--------|------|------------|
| DataFetcher | `market_data/*` | Tick JSON, Candle dict |
| IndicatorEngine | `indicators/*` | Candle deque → Factors dict {ma10, rsi, …} |
| Regime & Focus | `analysis/regime_classifier.py` / `analysis/focus_decider.py` | Factors → macro/micro レジーム・`weight_macro` |
| Regime Router (opt-in) | `workers/regime_router/worker.py` + `quant-regime-router.service` | macro/micro レジーム → `strategy_control` の strategy別 `entry_enabled` |
| Local Decider | `analysis/local_decider.py` | focus + perf → ローカル判定 |
| Strategy Plugin | `strategies/*` | Factors → `StrategyDecision` または None |
| Strategy Feedback | `analysis/strategy_feedback.py` / `analysis/strategy_feedback_worker.py` | 取引実績 + 戦略検知から per-strategy の調整係数を `strategy_feedback.json` に出力 |
| Brain Gate (optional) | `workers/common/brain.py` | order preflight → allow/reduce/block |
| Exit (専用ワーカー) | `workers/*/exit_worker.py` | pocket 別 open positions → exit 指示（PnL>0 決済が原則） |
| Forecast Service | `workers/forecast/worker.py` / `workers/common/forecast_gate.py` / `quant-forecast.service` | `strategy_tag/pocket/side/units` を入力として予測判定（allow/reduce/block）を返却。回帰系に加えてトレンドライン傾き（20/50）とサポレジ/ブレイク圧力（`sr_balance` / `breakout_bias` / `squeeze`）を予測因子へ統合し、`breakout_bias` は直近一致スキルで適応重み化。`expected_pips` + `anchor/target` + 分位レンジ上下帯（`range_low/high_pips`, `range_low/high_price`）+ `tp/sl/rr` を `order_manager`/`entry_intent_board` へ連携。戦略ごとの主TFに対して補助TF整合（`tf_confluence_*`）も監査メタとして返し、`entry_thesis` 欠損時は `strategy_tag` から主TFを補完する |
| Strategy Control | `workers/common/strategy_control.py` / `workers/strategy_control/worker.py` | 戦略 `entry/exit` 可否、`global_lock`、環境変数上書き |
| Risk Guard | `execution/risk_guard.py` | lot/SL/TP/pocket → 可否・調整値 |
| Order Manager | `execution/order_manager.py` + `quant-order-manager.service` | units/sl/tp/client_order_id/tag → OANDA ticket。`env_prefix` は `entry_thesis` 優先で解決し、`entry` の意図を優先反映 |
| Position Manager | `execution/position_manager.py` + `quant-position-manager.service` | 決済/保有/実績の取得、`trades.db`/`orders.db` を参照 |
| Logger | `logs/*.db` | 全コンポーネントが INSERT |

## 3. ライフサイクル
- Startup: `env.toml` 読込 → Secrets 確認 → 各サービス起動。
- 戦略ワーカー: 新ローソク → Factors 更新 → Regime/Focus → Local decision → `strategy_control` 参照 → risk_guard → order_manager → `trades.db` / `metrics.db` ログ。
- `scalp_ping_5s_{b,c,d}` は strategy local env（`ops/env/scalp_ping_5s_*.env`）で
  `MIN_TICK_RATE` / `LOOKAHEAD_GATE_ENABLED` / `REVERT_MIN_TICK_RATE` を管理し、
  方向フィルタ（C/D は `SIDE_FILTER=long`）を維持したまま entry 密度を動的に調整する。
- `quant-regime-router`（有効化時）は `M1/H4` のレジームを定期評価し、
  `strategy_control` の strategy別 `entry_enabled` を上書きする。
  対象は `REGIME_ROUTER_MANAGED_STRATEGIES` で限定し、`exit_enabled` には介入しない。
- `quant-strategy-control` は戦略フラグ同期に加えて、市場オープン時に
  `data_lag_ms` / `decision_latency_ms` を `metrics.db` へ定期発行する（V2 SLO 観測の基準）。
- `execution/stage_tracker` は `strategy_reentry_state` 更新時に
  `trade_id` だけでなく `close_time` も比較し、`trades.db` の id 系列が
  restore 等で巻き戻った場合でも reentry 基準の stale 固定化を防ぐ。
- `signal_bus` を使う場合は `SIGNAL_GATE_ENABLED=1` 時のみ、戦略ワーカー起点で `signal_bus` enqueue の運用を許可。
- タクト要件: 正秒同期（±500ms）、締切 55s 超でサイクル破棄（バックログ禁止）、`monotonic()` で `decision_latency_ms` 計測。
- `quant-strategy-feedback.service`（`analysis/strategy_feedback_worker.py`）は一定間隔で
  `trades.db` / ENTRYワーカー稼働中の戦略を再評価し、`strategy_feedback.json` を更新。
- Background: `utils/backup_to_gcs.sh` による nightly logs バックアップ + `/etc/cron.hourly/qr-gcs-backup-core` による GCS 退避（自動）。
- Background: `quant-bq-sync.service`（`scripts/run_sync_pipeline.py`）は
  `--limit` と `BQ_EXPORT_BATCH_SIZE` で送信件数を上限化し、`BQ_RETRY_TIMEOUT_SEC`
  を超える長時間 retry を避ける（停止/再起動時のハング回避）。
  連続失敗時は `BQ_FAILURE_BACKOFF_*` に従って cooldown し、再送連打を抑止する。
- Background: `quant-forecast-watchdog.timer` は `quant-forecast.service` の
  `/health` を監視し、連続失敗時に forecast を再起動する。復旧不能時は
  `quant-bq-sync.service` を停止して予測APIの可用性を優先する。

## 4. データスキーマと単位

### 共通スキーマ（pydantic 互換）

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

class LocalDecision(BaseModel):
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

### 単位と用語

| 用語 | 定義 |
|------|------|
| `pip` | USD/JPY の 1 pip = 0.01。入力/出力は pip 単位を明記。 |
| `point` | 0.001。OANDA REST 価格の丸め単位。 |
| `lot` | 1 lot = 100,000 units。`units = round(lot * 100000)`。 |
| `pocket` | `micro` = 短期テクニカル、`macro` = レジーム、`scalp` = スカルプ。口座資金を `pocket_ratio` で按分。 |
| `weight_macro` | 0.0〜1.0。`pocket_macro = pocket_total * weight_macro`（運用では macro 上限 30%）。 |

### 価格計算
- `price_from_pips("BUY", entry, sl_pips) = round(entry - sl_pips * 0.01, 3)`
- `price_from_pips("SELL", entry, sl_pips) = round(entry + sl_pips * 0.01, 3)`

### client_order_id
- `client_order_id = f"qr-{ts_ms}-{focus_tag}-{tag}"`（9桁以内のハッシュで重複防止）。
- Exit も同形式で 90 日ユニーク。

## 5. リプレイ品質ゲート（内部テスト）

- 目的: 戦略ワーカーのリプレイ結果を walk-forward で評価し、過学習や不安定な調整を早期検知する。
- 実行: `scripts/replay_quality_gate.py` が backend 切替で以下を複数 tick ファイルに対して実行し、fold 単位で `train/test` を評価する。
  - `exit_workers_groups` → `scripts/replay_exit_workers_groups.py`
  - `exit_workers_main` → `scripts/replay_exit_workers.py`
- `exit_workers_main` では `replay.intraday_start_utc` / `replay.intraday_end_utc` を指定すると、
  ファイル名の日付（`YYYYMMDD`）に合わせて日内 UTC 窓を自動適用できる。
- `config/replay_quality_gate_main.yaml` はデフォルトで intraday 窓を無効化し、
  フルデイ再生を品質ゲートの基準とする。
- tick 入力は `ticks_globs`（config 配列）または `--ticks-glob` の
  カンマ区切り複数指定に対応し、複数 root の同日ファイルは basename で重複排除する。
  重複時はサイズが大きいファイルを優先して採用する。
- `min_tick_lines` を指定すると、閾値未満の tick ファイルを
  walk-forward 対象から自動除外できる。
- `exclude_end_of_replay=true` と短い intraday 窓の組み合わせは、
  close の大半が `end_of_replay` となり `trade_count=0` を作りやすい点に注意。
- `replay.main_only=true` で main 戦略（TrendMA/BB_RSI）経路の再生に限定できる。
- 判定指標: `trade_count`, `profit_factor`, `win_rate`, `total_pips`, `max_drawdown_pips`, `pf_stability_ratio`
  に加え、`total_jpy`, `jpy_per_hour`, `max_drawdown_jpy` をサポート。
- 閾値管理: `config/replay_quality_gate*.yaml`（`gates.default` + `gates.workers.<worker>`）。
- `replay.env` で replay 子プロセスの環境変数を固定化できる（strategy variant 切替や replay 向け safety override をconfig管理）。
- `config/replay_quality_gate_ping5s_c.yaml` / `config/replay_quality_gate_ping5s_d.yaml` は
  それぞれ `scalp_ping_5s_c_live` / `scalp_ping_5s_d_live` 専用の
  profit-lock プロファイルとして運用する
  （`SCALP_REPLAY_PING_VARIANT=C|D`）。
- `exit_workers_main` の `ScalpPing5SB` は replay 時に `FORCE_EXIT_*_MAX_HOLD_SEC` を `timeout_sec` として
  signal/thesis へ伝播し、`SimBroker.check_timeouts()` が毎 tick で `time_stop` close を再現する。
  これにより `end_of_replay` クローズ偏重を抑え、live の force-exit hold と replay の整合性を保つ。
- `exit_workers_main` の `ScalpReplayEntryEngine` は
  signal の `entry_units_intent` を優先して replay units を決定する。
  `ScalpPing5SB` は `BASE_ENTRY_UNITS/MIN_UNITS/MAX_UNITS` と `confidence` を使って
  `entry_units_intent` を生成するため、fixed `10000` entry にならず
  C/D 運用ロット（`ops/env/scalp_ping_5s_c.env`, `ops/env/scalp_ping_5s_d.env`）を
  replay 側でも再現できる。
- 成果物:
  - `quality_gate_report.json`（fold 詳細 + pass/fail）
  - `quality_gate_report.md`（運用サマリ）
  - `commands.json`（再現用コマンドログ）
- 定期ワーカー:
  - `quant-replay-quality-gate.service`（oneshot）+
    `quant-replay-quality-gate.timer`（1h 周期）
  - 実装: `analysis/replay_quality_gate_worker.py`
  - 監査出力:
    - `logs/replay_quality_gate_latest.json`
    - `logs/replay_quality_gate_history.jsonl`

## 6. トレード反実仮想レビュー（事後判定）

- 目的: 実トレード履歴から「この条件なら見送る/縮小/増額すべきだった」を
  多要因（side/hour/spread/probability）で推定し、運用調整へ反映する。
- 実装: `analysis/trade_counterfactual_worker.py`
  - 入力: `logs/trades.db` + `logs/orders.db`
    - 追加入力（任意）: `COUNTERFACTUAL_REPLAY_JSON_GLOBS` で replay 出力
      （`replay_exit_workers.json`）を直接集計可能
  - 解析: 5fold 一貫性 (`fold_consistency`) と 95%下限 (`lb95_pips`) を併用
    し、さらに fold 外疑似 OOS 検証（action一致率/正の uplift 比率/`oos_lb95_uplift_pips`）
    を満たした提案だけを採用
    - stuck 判定: `hold_sec >= COUNTERFACTUAL_STUCK_HOLD_SEC` かつ
      `pl_pips <= COUNTERFACTUAL_STUCK_LOSS_PIPS` または
      `reason in COUNTERFACTUAL_STUCK_REASONS`
      を `stuck_rate` として評価し、`block/reduce` 判定へ反映
  - 出力: `logs/trade_counterfactual_latest.json` / `logs/trade_counterfactual_history.jsonl`
- 定期ワーカー:
  - `quant-trade-counterfactual.service`（oneshot）+
    `quant-trade-counterfactual.timer`（30min 周期）

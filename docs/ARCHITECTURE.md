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
- `quant-scalp-extrema-reversal` は `workers/scalp_extrema_reversal/worker.py` で
  高値/安値の極値帯（M1レンジ上端/下端）+ tick 反転を同時に満たしたときのみ
  両方向（short/long）を出す専用ワーカーとして運用する。
  `entry_thesis` には `entry_probability_raw` と `extrema` 文脈を記録し、
  最終許可/縮小/拒否は `quant-order-manager` 側 preflight に委譲する。
- `scalp_ping_5s_{b,c,d}` は strategy local env（`ops/env/scalp_ping_5s_*.env`）で
  `MIN_TICK_RATE` / `LOOKAHEAD_GATE_ENABLED` / `REVERT_MIN_TICK_RATE` を管理し、
  `SIDE_FILTER` / spread stale guard（`spread_guard_stale_*`）を含めて
  entry 密度を動的に調整する。
  また B/C では `MIN_UNITS` と `ORDER_MIN_UNITS_*` を 30 まで緩和し、
  `REVERT_WINDOW` 拡張と `REVERT_*` 閾値緩和で no-signal 偏りを抑える。
  さらに B は `SL_STREAK_DIRECTION_FLIP` / `SIDE_METRICS_DIRECTION_FLIP` を停止し、
  `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER` を 0.24 へ引き下げて
  逆方向 short 化と確率reject連鎖を回避する。
  C も strategy 別の reject/min-scale を `0.35/0.40` へ緩和し、
  flow 偏重時の entry 取りこぼしを抑える。
  続く運用調整として B/C の `PERF_GUARD_*` は sample 要件を引き上げ、
  `MAX_ORDERS_PER_MINUTE` と `MIN_UNITS` を緩和して
  `perf_block`/`rate_limited` 偏重の時間帯で entry 密度を回復させる。
  なお B は実運用観測で `perf_block` 連発が残ったため、
  `PERF_GUARD_MODE=reduce` に切り替え、同ガードは警告運用とした。
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
- Background: `quant-ops-policy.service`（`scripts/gpt_ops_report.py`）は
  LLMを使わない deterministic 集計で `logs/gpt_ops_report.json`（+任意 Markdown）を生成し、
  `market_context / driver_breakdown / break_points / scenarios / if_then_rules / risk_protocol`
  を運用レビュー用に出力する。
  判定手順は `主因特定 → 壊れる点特定 → A/B/C シナリオ化 → 条件式` の固定フローとし、
  参照元は `factor_cache / trades.db / orders.db / policy_overlay / market_events / macro_snapshot / optional external snapshot` に限定する。
  実行制御は `scripts/run_market_playbook_cycle.py` が担当し、
  平常15分・指標前後5分・アクティブ1分の可変周期を `logs/ops_playbook_cycle_state.json` で管理する。
  外部市況の自動取得は `scripts/fetch_market_snapshot.py` で実行し、
  `market_external_snapshot`（価格・DXY・米日10年）と `market_events`（主要イベント）を更新する。
  `gpt_ops_report` は `OPS_PLAYBOOK_FACTOR_MAX_AGE_SEC` を超える stale `factor_cache` を検知した場合、
  `snapshot.current_price` を外部 `USD/JPY` へフォールバックし、`direction_confidence_pct` を減衰する。
  手動メモの取り込みは `scripts/import_market_brief.py` で `market_external_snapshot` と `market_events` へ変換し、
  `scripts/build_market_context.py` で `market_context` を再生成してからレポートへ反映する。
  `--policy` 導線はプレイブック結果（bias/confidence/event/factor freshness/reject_rate）を
  `policy_diff`（`entry_gates.allow_new` / pocket `bias` / `range_mode` / `uncertainty`）へ自動変換する。
  `--apply-policy` 指定時は `analytics.policy_apply.apply_policy_diff_to_paths` で
  `policy_overlay` へ反映し、同一状態は `no_change` 判定で version 連番更新を抑止する。
- Background: `utils/backup_to_gcs.sh` による nightly logs バックアップ +
  `quant-core-backup.timer`（`/usr/local/bin/qr-gcs-backup-core`）による
  guarded GCS 退避（低優先度 + 負荷ガード、legacy `cron.hourly` は無効化）。
- Background: `quant-bq-sync.service`（`scripts/run_sync_pipeline.py`）は
  `--limit` と `BQ_EXPORT_BATCH_SIZE` で送信件数を上限化し、`BQ_RETRY_TIMEOUT_SEC`
  を超える長時間 retry を避ける（停止/再起動時のハング回避）。
  連続失敗時は `BQ_FAILURE_BACKOFF_*` に従って cooldown し、再送連打を抑止する。
- Background: `quant-forecast-watchdog.timer` は `quant-forecast.service` の
  `/health` を監視し、連続失敗時に forecast を再起動する。復旧不能時は
  `quant-bq-sync.service` を停止して予測APIの可用性を優先する。
- Background: `quant-forecast-improvement-audit.timer` は `vm_forecast_snapshot.py` と
  `eval_forecast_before_after.py` を定期実行し、`logs/reports/forecast_improvement/latest.md`
  と `logs/forecast_improvement_latest.json` へ before/after 判定を保存する。

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
    `quant-replay-quality-gate.timer`（3h 周期）
  - 実装: `analysis/replay_quality_gate_worker.py`
  - `REPLAY_QUALITY_GATE_AUTO_IMPROVE_ENABLED=1` の場合、
    replay 直後に `analysis.trade_counterfactual_worker` を戦略単位で実行し、
    `policy_hints.reentry_overrides`（`cooldown_* / same_dir_reentry_pips / return_wait_bias`）
    を `config/worker_reentry.yaml` へ自動反映する。
    `block_jst_hours` は `REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_BLOCK_HOURS=1`
    を明示した場合のみ反映する（既定は 0）。
    `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_REENTRY_CONFIDENCE` と
    `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_REENTRY_LCB_UPLIFT_PIPS` を下回る候補は解析のみで不採用とする。
    `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_APPLY_INTERVAL_SEC` を使い、
    反映間隔をレート制限する（間隔内は解析のみ）。
    反映の成否・理由は `replay_quality_gate_latest.json.auto_improve` に格納する。
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
    - ノイズ補正: spread カバレッジ・spread 超過・stuck 率・OOS 不確実性を
      `noise_penalty` として uplift から控除し、`noise_lcb_uplift_pips` で保守判定する。
    - 過去パターン統合: `config/pattern_book_deep.json`（`top_robust`/`top_weak`）を
      strategy+side の事前確率として合成し、`pattern_adjusted_lcb_uplift_pips` を算出する。
    - 政策ヒント: `reentry_overrides`（tighten/loosen + multiplier）を出力し、
      replay auto-improve が `worker_reentry` の非時間帯パラメータへ反映する。
  - 出力: `logs/trade_counterfactual_latest.json` / `logs/trade_counterfactual_history.jsonl`
- 定期ワーカー:
  - `quant-trade-counterfactual.service`（oneshot）+
    `quant-trade-counterfactual.timer`（20min 周期）
  - replay quality gate 側の auto-improve でも同 worker を再利用し、
    replay run 固有 JSON (`runs/*/replay_exit_workers.json` 等) を入力に実行できる。

## 7. 2026-02-24 運用補足（ENTRY 詰まり解除）

- `quant-order-manager` は `ops/env/quant-order-manager.env` のみを読むため、
  5秒スキャ（B/C）の実効ガード値は worker env ではなく同ファイルを正とする。
- `scalp_ping_5s_c_live` は `meta.env_prefix=SCALP_PING_5S` で評価されるため、
  `SCALP_PING_5S_C_*` だけでなく `SCALP_PING_5S_*` のガード値も整合させる。
- さらに `ORDER_MANAGER_SERVICE_ENABLED=0` の local fallback（worker 内 preflight）でも
  同じ判定が走るため、`ops/env/scalp_ping_5s_c.env` 側にも
  `SCALP_PING_5S_PERF_GUARD_*` を持たせて差異を無くす。
- 5秒スキャの preflight 詰まり解除として、B/C の
  `ORDER_MANAGER_PRESERVE_INTENT_*` と `SCALP_PING_5S_*_PERF_GUARD_*`
  を緩和し、`PERF_GUARD_MODE=reduce` に統一した。
- `worker_reentry` の時間帯ブロックは、`scalp_ping_5s_b_live` /
  `scalp_ping_5s_d_live` で `20/21/22 JST` を解除し、
  深夜帯 `3/5/6 JST` のみ維持した。
- TickImbalance は `TICK_IMB_REENTRY_LOOKBACK_SEC=0` /
  `TICK_IMB_REENTRY_MIN_PRICE_GAP_PIPS=0` /
  `TICK_IMB_REENTRY_REQUIRE_LAST_PROFIT=0` とし、
  strategy ローカルの reentry 距離ゲートを無効化した。

## 8. 2026-02-24 運用補足（損切り肥大の抑制）

- `execution/order_manager.py` は `stopLossOnFill` の判定で、
  `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_<STRATEGY_TAG>` を最優先で評価する。
  - 例: `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICROPULLBACKEMA=1`
  - global `ORDER_FIXED_SL_MODE` を変更せずに strategy 単位で broker SL attach を切り替えられる。
- 同時に `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_<STRATEGY_TAG>` で
  strategy 単位の entry SL 上限を制御する。
  - 例: `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_MICROPULLBACKEMA=6.0`
- `scalp_ping_5s_b` は `ops/env/scalp_ping_5s_b.env` で
  `SL_BASE/SHORT_SL_BASE` と `FORCE_EXIT_MAX_FLOATING_LOSS_PIPS` を圧縮し、
  `PERF_GUARD_SL_LOSS_RATE_MAX` を `0.62` へ引き下げる。
- `MicroPullbackEMA` は `ops/env/quant-micro-pullbackema.env` で
  `MICRO_MULTI_BASE_UNITS` と `MICRO_MULTI_MAX_MARGIN_USAGE` を下げ、
  margin closeout 尾部を抑制する。
- `scalp_ping_5s_b` は `ops/env/scalp_ping_5s_b.env` で
  `FORCE_EXIT_RECOVERY_WINDOW_SEC` / `FORCE_EXIT_RECOVERABLE_LOSS_PIPS` /
  `FORCE_EXIT_GIVEBACK_*` を使った
  「DD後リカバリー待ち -> 戻らなければカット」運用を許可する。
  broker SL は `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_SCALP_PING_5S_B_LIVE`
  で上限管理し、hard-stop を無効化せずに戻り余地のみを拡張する。

# Current Mechanisms

このファイルは、QuantRabbit の「今ある仕組み」を 1 枚で見られるようにするための棚卸しです。詳細仕様は `docs/ARCHITECTURE.md` と `docs/WORKER_ROLE_MATRIX_V2.md` を正本とし、この文書は「どの仕組みがあり、どこを見れば状態確認できるか」の早見表として使います。

## 0. 保守ルール

- 新しい仕組みを local-v2 の運用導線へ追加したら、このファイルの該当セクションも同じ変更で更新する。
- 既存の仕組みを停止・削除したら、行を単純削除せず、末尾の `Archive` セクションへ移して日付と理由を残す。
- `trade_min` / `trade_cover` / `trade_all` の profile 変更、または `mechanism_integrity` の監査対象変更が入った場合も、このファイルを更新対象に含める。
- 実装詳細は正本ドキュメントへ譲り、このファイルでは「現行かどうか」「どこで確認するか」「何として動いているか」を優先して保守する。

## 1. 作成時点スナップショット（2026-03-12 JST）

- 市況チェック
  - USD/JPY: `bid=159.034 / ask=159.042 / spread=0.8 pips`
  - `M5 ATR14=7.707 pips`
  - `M5 直近60本レンジ=49.6 pips`
  - OANDA API: `pricing=248ms(200)`, `summary=279ms(200)`, `openTrades=220ms(200)`
  - `openTrades=0`
- ローカル stack
  - `trade_min` で core + strategy worker が起動中
  - `logs/health_snapshot.json` の `mechanism_integrity.ok=true`
  - `orders.db` 直近2h は `filled=69`, `entry_probability_reject=10`, `rejected=4`, `perf_block=2`

## 2. Core Runtime

`trade_min` で常駐させる中核サービスは次の 6 つです。

| 仕組み | 役割 | 主な実装 |
| --- | --- | --- |
| Market data feed | OANDA pricing から tick/candle/factor を更新する | `workers/market_data_feed/worker.py`, `market_data/*`, `indicators/*` |
| Strategy control | `global_lock`, `entry_enabled`, `exit_enabled`, `data_lag_ms`, `decision_latency_ms` を扱う | `workers/strategy_control/worker.py`, `workers/common/strategy_control.py` |
| Order manager | preserve-intent 前提で preflight, risk, 発注, protection を担う | `execution/order_manager.py`, `workers/order_manager/worker.py` |
| Position manager | open positions / close / trades 同期を集約する | `execution/position_manager.py`, `workers/position_manager/worker.py` |
| Forecast service | `allow/reduce/block` の予測ゲートを返す | `workers/forecast/worker.py`, `workers/common/forecast_gate.py` |
| Strategy feedback | live trades を再評価して `strategy_feedback.json` を更新する | `analysis/strategy_feedback_worker.py`, `analysis/strategy_feedback.py` |

`trade_min` で現在起動中の worker ペアは次の通りです。

- `quant-scalp-ping-5s-b` / `quant-scalp-ping-5s-b-exit`
- `quant-scalp-trend-breakout` / `quant-scalp-trend-breakout-exit`
- `quant-scalp-rangefader` / `quant-scalp-rangefader-exit`
- `quant-scalp-precision-lowvol` / `quant-scalp-precision-lowvol-exit`
- `quant-scalp-vwap-revert` / `quant-scalp-vwap-revert-exit`
- `quant-scalp-drought-revert` / `quant-scalp-drought-revert-exit`
- `quant-micro-rangebreak` / `quant-micro-rangebreak-exit`
- `quant-micro-momentumburst` / `quant-micro-momentumburst-exit`
- `quant-micro-levelreactor` / `quant-micro-levelreactor-exit`
- `quant-micro-trendretest` / `quant-micro-trendretest-exit`
- `quant-m1scalper` / `quant-m1scalper-exit`
- `quant-session-open` / `quant-session-open-exit`

`scripts/local_v2_stack.sh` 上では profile ごとの広がりも定義されています。

- `trade_min`: 上記 12 worker ペアまで
- `trade_cover`: `macd_rsi_div_b`, `pullback_continuation`, `extrema_reversal`, `failed_break_reverse`, `false_break_fade`, `tick_imbalance`, `squeeze_pulse_break`, `momentumpulse`, `momentumstack`, `trendmomentum`, `vwapbound`, `vwaprevert` を追加
- `trade_all`: `precision_lowvol`, `ping_5s_c`, `ping_5s_d`, `ping_5s_flow`, `level_reject`, `wick_reversal_blend`, `wick_reversal_pro`, `micro_pullbackema`, `micro_compressionrevert` を含む全展開

## 3. Entry / Exit / Execution Mechanisms

主要な実行系の仕組みは次の順でつながっています。

| 層 | 何をするか | 主な実装 |
| --- | --- | --- |
| Regime / focus / local decision | 市況と戦略優先度をローカル判定する | `analysis/regime_classifier.py`, `analysis/focus_decider.py`, `analysis/local_decider.py` |
| Strategy-local entry | 各 worker が `entry_probability` と `entry_units_intent` を持つ `entry_thesis` を作る | `workers/*/worker.py`, `strategies/*` |
| Strategy entry shaping | feedback, forecast fusion, net-edge, leading profile, blackboard coordination を順に適用する | `execution/strategy_entry.py` |
| Blackboard coordination | `entry_intent_board` で同時 entry の衝突を調整する | `execution/strategy_entry.py`, `logs/orders.db` |
| Risk / preflight | margin, drawdown, RR, optional Brain / pattern / forecast を見る | `execution/risk_guard.py`, `execution/order_manager.py`, `workers/common/brain.py`, `workers/common/pattern_gate.py`, `workers/common/forecast_gate.py` |
| Order submit | OANDA 発注と protection の realign を行う | `execution/order_manager.py` |
| Dedicated exit | 共通 exit manager ではなく strategy ごとの `exit_worker` が決済する | `workers/*/exit_worker.py` |

補足:

- 共通 `exit_manager` はスタブで、exit 判定の主体ではありません。
- Brain は optional で、現行 safe canary は `brain-ollama-safe.env` 合成前提です。
- common layer は preserve-intent 方針で、戦略の方向意図を後付けで再採点しません。
- `PrecisionLowVol` / `DroughtRevert` は `workers/scalp_wick_reversal_blend/worker.py`
  の thin wrapper で、
  `PrecisionLowVol`
  については current live で
  short `volatility_compression`
  の
  `gap:up_flat` shallow lane,
  mid-RSI continuation-headwind lane,
  `gap:down_flat` low-score lane
  を worker-local guard で個別に抑える運用です。
- 同じ family では
  `PrecisionLowVol` / `DroughtRevert` / `WickReversalBlend`
  の stop band も strategy-local に調整しており、
  `scalp_extrema_reversal_live` と `scalp_ping_5s_d_live`
  は dedicated env 側で SL/TP 帯を別管理しています。
- `workers/scalp_level_reject/exit_worker.py`
  と
  `workers/scalp_wick_reversal_blend/exit_worker.py`
  は、
  `config/strategy_exit_protections.yaml`
  の
  `be_profile / tp_move`
  を opt-in で読み、
  `scalp_extrema_reversal_live`,
  `WickReversalBlend`,
  `PrecisionLowVol`,
  `DroughtRevert`
  の open trade に対して broker
  `SL/TP`
  を live 更新する。
  current local-v2 では
  base profile に加えて
  `ATR / spread / setup_quality / continuation_pressure / reversion_support / extrema setup pressure`
  から
  `trigger / lock / TP buffer`
  を補正する market-aware multiplier を持つ。
  共通 exit manager を増やすのではなく、
  dedicated exit worker 内で strategy-local に完結させる。

## 4. Live Optimization / Feedback / Guard Mechanisms

`logs/health_snapshot.json.mechanism_integrity` が現在監視対象にしている live 仕組みは次の通りです。

| 仕組み | 役割 | 主な artifact / 実装 |
| --- | --- | --- |
| `strategy_feedback` | live trades を strategy / setup 単位で再評価して bounded overlay を返す | `logs/strategy_feedback.json`, `analysis/strategy_feedback_worker.py` |
| `dynamic_alloc` | strategy / setup ごとの lot multiplier を動的更新する | `config/dynamic_alloc.json`, `workers/common/dynamic_alloc.py` |
| `pattern_book` | 勝ち負けパターンの集計と deep pattern を保持する | `config/pattern_book.json`, `config/pattern_book_deep.json`, `scripts/pattern_book_worker.py` |
| `entry_path_summary` | recent order path を戦略 / setup 単位で集計する | `logs/entry_path_summary_latest.json`, `scripts/entry_path_aggregator.py` |
| `participation_alloc` | winner lane の boost と loser lane の trim/probability offset を出す。`2 fills + negative realized` の fresh loser lane も fast trim 対象に含む | `config/participation_alloc.json`, `scripts/participation_allocator.py` |
| `forecast_runtime` | forecast runtime override と改善監査結果を保持する | `logs/forecast_improvement_latest.json`, `workers/common/forecast_gate.py` |
| `forecast_service` | `:8302/health` を持つ dedicated forecast 判定サービス | `workers/forecast/worker.py` |
| `blackboard` | `entry_intent_board` の存在と直近 row を監査する | `logs/orders.db`, `execution/strategy_entry.py` |
| `loser_cluster` | loser lane の setup clustering を出す | `logs/loser_cluster_latest.json`, `scripts/loser_cluster_worker.py` |
| `auto_canary` | loser cluster / replay / counterfactual を使った小さい override を出す | `config/auto_canary_overrides.json`, `scripts/auto_canary_improver.py` |
| `macro_news_context` | caution window を含むマクロ文脈を供給する | `logs/macro_news_context.json`, `scripts/macro_news_context_worker.py`, `scripts/build_market_context.py` |

slow loop 側で更新されるが、上の `mechanism_integrity` から間接監査されるもの:

- `trade_counterfactual`
- `replay_quality_gate`
- `worker_reentry`
- `ops policy / market_context`

2026-03-13 current:
- `run_local_feedback_cycle`
  の既定は
  `dynamic_alloc=1d lookback / min_trades=8 / setup_min_trades=2 / half_life=6h / min_lot_multiplier=0.20`
  と
  `participation_allocator=3h lookback / min_attempts=4 / setup_min_attempts=1 / max_units_cut=0.35 / max_units_boost=0.30 / max_probability_boost=0.15`
  を正とする。
- `dynamic_alloc`
  は setup override だけでなく strategy-level でも
  low-sample severe loser /
  fast burst loser
  を current 窓で clamp し、
  `TickImbalance`
  のような single hard loser や
  `WickReversalBlend`
  のような few-trade burst loser を historical floor のまま残さない。

## 5. Strategy Coverage

`logs/health_snapshot.json` で active discovery されている戦略キーは 30 本です。

- `DroughtRevert`
- `FailedBreakReverse`
- `M1Scalper-M1`
- `MicroCompressionRevert`
- `MicroLevelReactor`
- `MicroMomentumStack`
- `MicroPullbackEMA`
- `MicroRangeBreak`
- `MicroTrendRetest`
- `MicroVWAPBound`
- `MicroVWAPRevert`
- `MomentumBurst`
- `MomentumPulse`
- `PrecisionLowVol`
- `PullbackContinuation`
- `RangeFader`
- `TrendBreakout`
- `TrendMomentumMicro`
- `VwapRevertS`
- `scalp_extrema_reversal_live`
- `scalp_macd_rsi_div_b_live`
- `scalp_macd_rsi_div_live`
- `scalp_ping_5s_b_live`
- `scalp_ping_5s_c_live`
- `scalp_ping_5s_d_live`
- `scalp_ping_5s_flow_live`
- `squeeze_pulse_break`
- `tick_imbalance`
- `wick_reversal_blend`
- `wick_reversal_pro`

このうち、現時点で `strategy_feedback` が eligible active として扱っている current lane は 9 本です。

- `DroughtRevert`
- `MicroLevelReactor`
- `MicroTrendRetest`
- `MomentumBurst`
- `PrecisionLowVol`
- `RangeFader`
- `VwapRevertS`
- `scalp_extrema_reversal_live`
- `scalp_ping_5s_d_live`

実装上の worker inventory は概ね次の構成です。

- core / special: `forecast`, `market_data_feed`, `order_manager`, `position_manager`, `regime_router`, `session_open`, `strategy_control`
- micro family: `micro_compressionrevert`, `micro_levelreactor`, `micro_momentumburst`, `micro_momentumpulse`, `micro_momentumstack`, `micro_pullbackema`, `micro_rangebreak`, `micro_runtime`, `micro_trendmomentum`, `micro_trendretest`, `micro_vwapbound`, `micro_vwaprevert`
- scalp family: `scalp_drought_revert`, `scalp_extrema_reversal`, `scalp_failed_break_reverse`, `scalp_false_break_fade`, `scalp_level_reject`, `scalp_m1scalper`, `scalp_macd_rsi_div`, `scalp_macd_rsi_div_b`, `scalp_ping_5s`, `scalp_ping_5s_b`, `scalp_ping_5s_c`, `scalp_ping_5s_d`, `scalp_ping_5s_flow`, `scalp_precision_lowvol`, `scalp_pullback_continuation`, `scalp_rangefader`, `scalp_squeeze_pulse_break`, `scalp_tick_imbalance`, `scalp_trend_breakout`, `scalp_vwap_revert`, `scalp_wick_reversal_blend`, `scalp_wick_reversal_pro`

構造メモ:

- `PrecisionLowVol` と `DroughtRevert` は dedicated service 名を持つが、signal family としては `scalp_wick_reversal_blend` 系の thin wrapper です。
- `MicroLevelReactor`, `MomentumBurst`, `MicroTrendRetest` は individual service で起動されるが、runtime family としては `micro_runtime` の allowlist 運用と密接です。
- `scalp_ping_5s_b/c/d/flow` は shared ping family の派生 lane です。
- `trade_min` の lean set に入っていても、`strategy_feedback` の eligible active に乗るかどうかは別管理です。

## 6. まず見る場所

状態確認の入口はこの 5 つで足ります。

| 何を見るか | 参照先 |
| --- | --- |
| stack の起動状態 | `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` |
| live mechanism の欠落 | `logs/health_snapshot.json` |
| 発注・拒否・協調 | `logs/orders.db` |
| 約定・損益・entry_thesis | `logs/trades.db` |
| service ごとの最新ログ | `logs/local_v2_stack/*.log` |

詳細仕様の正本:

- `docs/ARCHITECTURE.md`
- `docs/WORKER_ROLE_MATRIX_V2.md`
- `docs/OBSERVABILITY.md`
- `docs/RISK_AND_EXECUTION.md`

## 7. 更新方法

この一覧を更新したいときは、最低限次を見直します。

```bash
scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env
bash scripts/collect_local_health.sh
sqlite3 logs/orders.db "select status, count(*) from orders where ts >= datetime('now','-2 hours') group by status order by count(*) desc;"
python3 - <<'PY'
import json
from pathlib import Path
obj = json.loads(Path('logs/health_snapshot.json').read_text())
print(obj["mechanism_integrity"]["strategy_feedback"]["active_strategies"])
print(obj["mechanism_integrity"]["strategy_feedback"]["eligible_active_strategies"])
PY
```

## 8. Archive

現時点では archive 対象なし。

削除・停止した仕組みは、次の形式でここへ追記する。

- `YYYY-MM-DD`: `mechanism_name`
  - 前の役割:
  - 現在の扱い: archived / removed / replaced
  - 置き換え先:
  - 備考:

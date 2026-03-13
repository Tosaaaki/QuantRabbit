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
- `session_open_breakout` は
  `min_hold_sec=300`
  の negative/candle exit を維持したまま、
  positive PnL 中だけ
  `workers/session_open/exit_worker.py`
  から broker
  `set_trade_protections`
  を前倒しで更新する。
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
- `DroughtRevert`
  については
  long `volatility_compression`
  の
  `macro:trend_long + gap:down_flat`
  でも
  `projection deeply negative`
  かつ
  `di support が弱い`
  weak reclaim probe だけを
  `DROUGHT_WEAK_TREND_LONG_PROBE_*`
  で worker-local に reject する。
  2026-03-13 21:55 JST からは、
  `rsi 42-46 / adx<=12.5 / projection<=0.10 / flat gap`
  の soft-trend long も
  `DROUGHT_FLAT_GAP_SOFT_TREND_LONG_*`
  で reject 対象に含める。
  `tight_thin`
  の recovery winner まで broad に落とさないため、
  shared gate ではなく
  exact setup 条件で扱う。
- `WickReversalBlend`
  は
  `volatility_compression|adx_squeeze`
  の long について、
  `0.35 <= gap_ratio < 1.20`
  の lean-gap reclaim を
  `WICK_BLEND_LEAN_GAP_LONG_*`
  で worker-local に reject する。
  `gap:up_flat`
  の winner lane を broad stop しないため、
  flat-gap ではなく lean-gap だけを先に落とす。
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
- `scalp_extrema_reversal_live`
  は
  `config/strategy_exit_protections.yaml`
  で
  `min_profit_ratio=0.60`
  を持ち、
  `take_profit / range_timeout / candle_*`
  の positive market close では
  broker TP の
  60%
  未満を
  `order_manager`
  が
  `close_reject_profit_ratio`
  で拒否する。
  一方で
  `lock_floor`
  は protective close として
  near-BE で通し、
  `min_profit_pips=0.1`
  だけを floor にする。
  早すぎる soft TP だけを抑えつつ、
  seen-profit の giveback は減らす設計で、
  shared blanket hold 延長ではない。
- `order_manager`
  の
  `profit_guard`
  は
  既定では pocket scope だが、
  `ORDER_PROFIT_GUARD_SCOPE_STRATEGY_*`
  で strategy ごとに
  `strategy`
  へ切り替えられる。
  現行 local-v2 では
  `PrecisionLowVol`
  だけが
  `ORDER_PROFIT_GUARD_SCOPE_STRATEGY_PRECISIONLOWVOL=strategy`
  で
  loser scalp pocket の giveback から切り離されている。
- `TickImbalance` / `TickImbalanceRRPlus`
  は
  `workers/scalp_tick_imbalance/worker.py`
  内で
  `trend exhaustion`
  guard を持ち、
  `TREND`
  文脈の
  side-aligned extreme
  `RSI + ADX + VWAP gap + ema_slope + MACD hist`
  が揃う伸び切り entry を
  strategy-local に reject する。
  判定結果は
  `entry_thesis.tick_imbalance.exhaustion_guard`
  に残し、
  shared gate の追加 tightening では扱わない。

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
  `dynamic_alloc=1d lookback / min_trades=8 / setup_min_trades=2 / half_life=6h / min_lot_multiplier=0.30`
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
- `participation_alloc`
  は high-turnover scalp の small positive winner を
  `profit_per_fill>=1.5-2.0 JPY`
  帯でも setup-level
  `boost_participation`
  へ上げる。
  `dynamic_alloc`
  も current loser strategy の中にある exact winner setup については
  single winner は
  `0.70-0.82`,
  2-trade winner は
  `0.82-1.00`
  の bounded winner-relief override を emit し、
  strategy-wide trim が
  `PrecisionLowVol / DroughtRevert`
  の current winner long まで
  `0.20`
  に潰し切らないようにする。

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
- `scalp_extrema_reversal_live` の dedicated exit は `scalp_level_reject` 系の thin wrapper で、extrema の stale-loser / negative-exit 改善は common `exit_manager` ではなく `workers/scalp_level_reject/exit_worker.py` 側に入る。
- `MicroLevelReactor`, `MomentumBurst`, `MicroTrendRetest` は individual service で起動されるが、runtime family としては `micro_runtime` の allowlist 運用と密接です。
- `scalp_ping_5s_b/c/d/flow` は shared ping family の派生 lane です。
- `PrecisionLowVol`, `WickReversalBlend`, `scalp_extrema_reversal_live` の dedicated exit worker は `config/strategy_exit_protections.yaml` の `exit_profile.inventory_stress` を読み、account stress（`health_buffer / free_margin_ratio / margin_usage_ratio / unrealized_dd_ratio`）と strategy-local stale-loser 条件が同時に立ったときだけ negative close を実行する。common の一律 exit judge は増やしていない。
- `workers/scalp_wick_reversal_blend/worker.py` は current local-v2 で、`DroughtRevert` の flat-gap soft-trend long と `WickReversalBlend` の lean-gap long に加えて、`WickReversalBlend` の `volatility_compression` weak countertrend short も worker-local に reject する。条件は `projection_score>=0.10`, `wick_quality<=0.78`, `rsi<=58`, `adx<=20`, `macd_hist_pips>=0.12` の同時成立に限定し、stronger short lane や shared gate へは広げない。
- 同じ `workers/scalp_wick_reversal_blend/worker.py` の `PrecisionLowVol` は、`short_up_lean` かつ `macro:trend_long / align:countertrend` の weak fade を worker-local に reject する。条件は `setup_quality<0.30`, `reversion_support<0.58`, `continuation_pressure>=0.22`, `rsi<=55`, `projection_score<=0.30` の同時成立に限定し、up-lean short 全体は止めない。
- `PrecisionLowVol` は加えて、`long_up_flat` かつ `volatility_compression / macro:trend_long` の shallow reclaim も worker-local に reject する。条件は `continuation_pressure<=0.28`, `rsi>=44`, `projection_score<=0.30`, `setup_quality<0.52`, `reversion_support<0.72` で、`strong_reclaim_probe` が立つ long は維持する。
- 収益/リスク/ENTRY/EXIT 改善の着手前レビューは `scripts/change_preflight.sh "<strategy_tag or hypothesis_key or close_reason>"` を正とする。wrapper は `collect_local_health.sh` による local health refresh、`tick_cache/factor_cache/health_snapshot` を使った USD/JPY 市況要約、`trade_findings_review.py` による同系改善 review をまとめて実行する。
- commit 前 enforcement は `.githooks/pre-commit` の `scripts/preflight_guard.py` が担う。`execution/`, `workers/`, `strategies/`, `analysis/`, 主要 risk config/env の staged 変更では、fresh `logs/change_preflight_latest.json` と staged `docs/TRADE_FINDINGS.md` を必須にする。
- `scripts/trade_findings_lint.py` は strict-since 以降の entry に required fields と `Hypothesis Key` の format を要求し、`scripts/trade_findings_index.py` は latest hypothesis index / unresolved list / dominant loss driver summary を `logs/trade_findings_index_latest.{json,md}` へ出す。
- `scalp_ping_5s_c_live` は current local-v2 で
  `LOOKAHEAD_NEGATIVE_EDGE_RESCUE_*`
  を持ち、
  `edge_negative_block`
  のうち
  `recent fills == 0`
  かつ
  `pred/momentum/range`
  が最低条件を満たす候補だけを
  `0.18-0.42x`
  units で strategy-local に救済する。
  blanket loosening ではなく、
  low-activity window 専用の cadence 回復レバーとして扱う。
  あわせて execution 側の
  `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)`
  は worker の `MIN_UNITS=5`
  に整合させ、
  rescued 5-6 units 候補を
  order-manager で落とさない。
  それでも `entry_probability`
  scale 後に `<5`
  へ落ちる rescued candidate には、
  worker-local の post-probability floor を入れて
  `7-10 units`
  帯まで持ち上げる。
- `scalp_ping_5s_d_live`
  は
  `non-neutral horizon + m1_opposite`
  を worker-local に reject する。
  この guard は
  `signal.side != horizon_side`
  の countertrend lane だけでなく、
  `fast_flip`
  で horizon 側へ寄せた
  `relation=align`
  lane も例外にしない。
  さらに
  `market_order`
  直前でも同じ contract を再評価し、
  route/sizing 後の late send を防ぐ。
- `execution/order_manager.py`
  は
  `entry_thesis.entry_units_intent`
  と
  `strategy_tag`
  を missing のときだけ補うのではなく、
  最終送信
  `units`
  / 明示
  `strategy_tag`
  に強制整合する。
  stale intent が残っても
  request payload 側で self-heal される。
- さらに
  `market_order / limit_order`
  の status log は、
  その status 時点の actual
  `units / side / strategy_tag`
  で nested
  `entry_thesis`
  まで canonicalize する。
  `probability_scaled / submit_attempt / filled`
  で contract がズレたときは
  `entry_contract_corrections`
  と
  `entry_units_intent_raw / strategy_tag_raw`
  を残して self-heal する。
- `execution/position_manager.py`
  は trade ingest 時に
  `entry_thesis.entry_units_intent`
  の stale 値を actual trade units へ上書きし、
  explicit
  `strategy_tag`
  を thesis より優先して正規化する。
- `trade_min` の lean set に入っていても、`strategy_feedback` の eligible active に乗るかどうかは別管理です。
- `workers/micro_runtime/worker.py`
  の history gate は family-level score を基本に維持するが、
  same lane 内で winner / loser setup が混在する current 運用に合わせて、
  recent winner
  `setup_fingerprint`
  は exact setup history で
  `winner_setup_override`
  を付けて
  `skip`
  だけ解除できる。
  sizing は family history multiplier を維持し、
  loser family 全体を broad に緩めない。

- `workers/micro_runtime/worker.py`
  の dynamic alloc / participation profile lookup は、
  base strategy key だけでなく exact
  `signal_tag`
  も first key として解決する。
  `MicroLevelReactor-bounce-lower`
  のような winner lane は、
  family fallback ではなく exact key の
  `lot_multiplier`
  を使って sizing / cadence へ反映する。

- micro dedicated winner runner で margin full を狙うときは、
  worker local knob に加えて service env の
  `MAX_MARGIN_USAGE`
  /
  `MAX_MARGIN_USAGE_HARD`
  を override する。
  `allowed_lot()`
  がこの global cap を使うため、
  actual margin pressure はここで決まる。

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

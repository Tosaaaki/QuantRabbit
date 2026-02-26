# Risk & Execution

## 1. エントリー/EXIT/リスク制御

### Strategy フロー
- Focus/Local decision → `ranked_strategies` 順に Strategy Plugin を呼び、`StrategyDecision` または None を返す。
- `None` はノートレード。

### Confidence スケーリング
- `confidence`(0–100) を pocket 割当 lot に掛け、最小 0.2〜最大 1.0 の段階的エントリー。
- `STAGE_RATIOS` に従い `_stage_conditions_met` を通過したステージのみ追撃。

### Brainゲート（任意）
- `execution/order_manager.py` で LLM 判断を実行し、**ALLOW/REDUCE/BLOCK** を返す。
- `REDUCE` は units 縮小のみ（増加は禁止）。

### AddonLive 経路の strategy_tag 契約（2026-02-26）
- `workers/common/addon_live.py` は `strategy_tag` を必須で解決し、
  `order -> intent -> meta -> worker_id` の順で補完する。
- `strategy_tag` を解決できない注文は送信前に skip し、
  `missing_strategy_tag` 注文を `order_manager` へ流さない。
- `entry_thesis.strategy_tag` と `client_order_id` は同一戦略タグ由来で揃える
  （EXITルーティング、監査、集計整合性の維持）。

### Forecast 段階導入（2026-02-25）
- forecast 強化パラメータは、いきなり全戦略へ適用せず
  `FORECAST_GATE_STRATEGY_ALLOWLIST` で対象を限定して段階導入する。
- 現行の段階導入対象:
  - `MicroRangeBreak`
  - `MicroVWAPBound`
- `quant-v2-runtime.env` の運用値（cand1_mid）:
  - `FORECAST_TECH_FEATURE_EXPANSION_GAIN=0.20`
  - `FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP=1m=0.16,5m=0.22,10m=0.24`
  - `FORECAST_TECH_SESSION_BIAS_WEIGHT_MAP=1m=0.0,5m=0.22,10m=0.28`
- 受け入れガード（before/after 同一期間比較）:
  - `hit_delta < -0.002` を悪化
  - `mae_delta > +0.020` を悪化
  - `range_coverage_delta < -0.030` を悪化

### Exit
- 各戦略の `exit_worker` が最低保有時間とテクニカル/レンジ判定を踏まえ、PnL>0 決済が原則。
- 例外は強制 DD / ヘルス / マージン使用率 / 余力 / 未実現DDの総合判定のみ。
- 共通 `execution/exit_manager.py` は常に空を返す互換スタブ。
- `execution/stage_tracker` がクールダウンと方向別ブロックを管理。

### scalp_ping_5s_b 運用補足（取り残し抑制）
- `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE` を
  `quant-order-manager` の環境で運用し、低確率シグナルを `order_manager` 側で reject する。
- `scalp_ping_5s_b*` は entry 時の SL 欠損を防ぐため、
  `order_manager` 側で `stopLossOnFill` を戦略別に許可する。
  - `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_B=1`
  - `ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_B=0`
  - Bワーカー env は `SCALP_PING_5S_B_USE_SL=1` / `SCALP_PING_5S_B_DISABLE_ENTRY_HARD_STOP=0` を維持する。
  - `workers/scalp_ping_5s_b.worker` は fail-safe として上記2値を起動時に強制補正する
    （`SCALP_PING_5S_B_ALLOW_UNPROTECTED_ENTRY=1` のときのみ無効化）。
- `scalp_ping_5s_b*` は方向転換遅延による逆張り連発を抑えるため、
  `direction_bias` と `horizon_bias` が同方向で強く一致した場合に
  エントリーをブロックせず `fast_direction_flip` で side を即時リルートする。
  - 既定は B で有効（`SCALP_PING_5S_B_FAST_DIRECTION_FLIP_ENABLED=1`）。
  - `horizon=neutral` でも `direction_bias` が十分強ければ反転を許可する
    （`SCALP_PING_5S_B_FAST_DIRECTION_FLIP_NEUTRAL_HORIZON_BIAS_SCORE_MIN`）。
  - 頻度維持のため、reject ではなく side リライト＋confidence 加算のみ行う。
- `scalp_ping_5s_b*` は「同方向の `STOP_LOSS_ORDER` 連発」を方向ミスマッチとして扱い、
  直近クローズ履歴に同方向SLが規定回数（既定2連）続いた場合、
  次エントリーの side を `sl_streak_direction_flip` で反転する。
  - 既定は B で有効（`SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_ENABLED=1`）。
  - 発注拒否ではなく side リライトで頻度を維持し、`entry_thesis` に
    `sl_streak_*`（side/count/age/applied/reason）を記録する。
  - 過去トレード参照は `strategy_tag + pocket` のクローズ履歴に限定し、
    `SL_STREAK_DIRECTION_FLIP_MAX_AGE_SEC` を超える古い連敗は反転対象外にする。
  - 2026-02-19 追記:
    - `SL回数` と `MARKET_ORDER_TRADE_CLOSE でのプラス回数` を追加条件にした。
      - `SL_STREAK_DIRECTION_FLIP_MIN_SIDE_SL_HITS`（既定2）
      - `SL_STREAK_DIRECTION_FLIP_MIN_TARGET_MARKET_PLUS`（既定1）
      - 集計窓は `SL_STREAK_DIRECTION_FLIP_METRICS_LOOKBACK_TRADES`（既定24）
    - さらに `direction_bias/horizon` のテクニカル一致がない場合は反転しない
      （`SL_STREAK_DIRECTION_FLIP_REQUIRE_TECH_CONFIRM=1`）。
    - `fast_direction_flip` が同ループで発火した場合は `fast_flip` を優先し、
      `sl_streak` 側の逆上書きを禁止する
      （`SL_STREAK_DIRECTION_FLIP_ALLOW_WITH_FAST_FLIP=0` 既定）。
    - 連敗数が `SL_STREAK_DIRECTION_FLIP_FORCE_STREAK` 以上のときは、
      `MIN_TARGET_MARKET_PLUS` 条件のみをバイパス可能にして反転遅延を抑える
      （`direction_bias/horizon` の tech 一致要件は維持）。
- `scalp_ping_5s_b*` の extrema は 2026-02-19 以降、ショート側のみ非対称チューニング。
  - `short_bottom_soft` は専用倍率
    `EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT`（B既定 0.42）で縮小。
  - `mtf_balanced` かつ short非優勢では
    `EXTREMA_SHORT_BOTTOM_SOFT_BALANCED_UNITS_MULT`（B既定 0.30）へ追加縮小し、
    `entry_thesis.extrema_gate_reason=short_bottom_soft_balanced` で監査する。
  - extrema reversal は B既定で `long -> short` を無効化
    （`EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=0`）し、
    `long_top_soft_reverse` クラスタの誤反転を抑制する。
- 2026-02-21 追記（逆方向抑制）:
  - `short` 側の誤エントリー抑制を優先し、Bローカルで short 判定閾値を厳格化:
    - `SHORT_MIN_TICKS=4`
    - `SHORT_MIN_SIGNAL_TICKS=4`
    - `SHORT_MIN_TICK_RATE=0.62`
    - `SHORT_MOMENTUM_TRIGGER_PIPS=0.11`
  - 方向逆行時のサイズ抑制を強化:
    - `DIRECTION_BIAS_BLOCK_SCORE=0.52`
    - `DIRECTION_BIAS_OPPOSITE_UNITS_MULT=0.60`
    - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.42`
    - `SIDE_BIAS_SCALE_GAIN=0.50`
    - `SIDE_BIAS_SCALE_FLOOR=0.18`
    - `SIDE_BIAS_BLOCK_THRESHOLD=0.30`
  - `entry_probability` の counter 罰則を強め、counter 優勢局面のサイズを縮小:
    - `ENTRY_PROBABILITY_ALIGN_PENALTY_MAX=0.55`
    - `ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX=0.32`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER=0.24`
    - `ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT=0.45`
- 2026-02-21 追記（ケース別総合チューニング）:
  - `scalp_ping_5s_b_live` は単一要因ではなく
    「上昇押し目局面の逆張り」「同方向クラスター」「劣化時間帯の継続稼働」が
    重なって悪化しやすいため、strategy local + order_manager の両面で対処する。
  - strategy local（`ops/env/scalp_ping_5s_b.env`）:
    - 過剰エントリー抑制:
      - `MAX_ACTIVE_TRADES=14`
      - `MAX_PER_DIRECTION=8`
      - `MAX_ORDERS_PER_MINUTE=10`
      - `ENTRY_CHASE_MAX_PIPS=1.0`
      - `MIN_TICKS=5`, `MIN_SIGNAL_TICKS=4`, `MIN_TICK_RATE=0.85`
      - `IMBALANCE_MIN=0.55`
    - 方向反転の品質改善:
      - `FAST_DIRECTION_FLIP_*` は閾値を引き上げて誤反転を抑制
      - `SL_STREAK_DIRECTION_FLIP_*` は `min_streak=2` と tech confirm 必須化
      - `SIDE_METRICS_DIRECTION_FLIP_*` は sample/差分条件を緩和して、
        side劣化時の反転追従を増やす
    - 同方向逆行スタック抑制:
      - `SIDE_ADVERSE_STACK_UNITS_ACTIVE_START=2`
      - `SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.20`
      - `SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.18`
      - `SIDE_ADVERSE_STACK_DD_START_PIPS=0.45`
      - `SIDE_ADVERSE_STACK_DD_FULL_PIPS=1.60`
      - `SIDE_ADVERSE_STACK_DD_MIN_MULT=0.22`
    - ケース追従:
      - `SIGNAL_WINDOW_ADAPTIVE_ENABLED=1`
      - `LOOKAHEAD_GATE_ENABLED=1`
      - `LONG_MOMENTUM_TRIGGER_PIPS=0.12`
      - `SHORT_MIN_TICK_RATE=0.72`
      - `MOMENTUM_TRIGGER_PIPS=0.11`
  - order manager（`ops/env/quant-order-manager.env`）:
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.48`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.65`
    - `SCALP_PING_5S_B_PERF_GUARD_MODE=block` +
      `PERF_GUARD_HOURLY=1` / `PERF_GUARD_SPLIT_DIRECTIONAL=1`
      を有効にし、劣化ケースで fail-fast させる。
- `RANGEFADER_EXIT_NEW_POLICY_START_TS` を `quant-scalp-ping-5s-b-exit` の環境で固定し、
  service再起動時も既存建玉が legacy 扱いで loss-cut 系ルールから外れないようにする。
  - `workers/scalp_ping_5s_b.exit_worker` は同キーを float として読むため、
    値は ISO ではなく Unix秒（例: `1771286400`）で指定する。
- `config/strategy_exit_protections.yaml` では
  `scalp_ping_5s_b_live: *SCALP_PING_5S_EXIT_PROFILE` を維持し、
  `ALLOWED_TAGS=scalp_ping_5s_b_live` の建玉が default EXITプロファイルへ落ちないようにする。

### scalp_ping_5s（C/D 含む）side filter 運用補足
- `SCALP_PING_5S_*_SIDE_FILTER` は初期シグナルだけでなく、
  MTF/flip など後段ルーティング後の最終シグナルにも強制適用する。
- 最終 side が filter と不一致の場合は
  `side_filter_final_block` として entry を拒否する。
- replay 経路（`scripts/replay_exit_workers.py`）でも同様に、
  `MTF` 調整後の side で再判定して不一致を破棄する。
- 目的は、`SIDE_FILTER=long` 運用時に `mtf_reversion_fade` 等で
  short が混入する経路をなくし、実運用と検証の整合性を保つこと。

### scalp_ping_5s_c 運用補足（動的同方向cap + side別EXIT）
- `workers/scalp_ping_5s.worker` は `SCALP_PING_5S_DYNAMIC_DIRECTION_CAP_*` を使い、
  `direction_bias` / `horizon` / `side_adverse_stack` が弱い・逆行圧のときだけ
  同方向 cap を自動縮小する。
  - 既定（C運用）:
    - `SCALP_PING_5S_C_DYNAMIC_DIRECTION_CAP_ENABLED=1`
    - `...WEAK_CAP=2`
    - `...ADVERSE_CAP=2`
    - `...METRICS_ADVERSE_CAP=1`
- `quant-scalp-ping-5s-c-exit`（`workers/scalp_ping_5s_c.exit_worker`）は
  `config/strategy_exit_protections.yaml` の side別キーを適用する。
  - `non_range_max_hold_sec_short`
  - `direction_flip.short_*` / `direction_flip.long_*`
- これにより C 系は「高確度局面の建玉維持」と
  「弱一致局面のクラスター抑制」を同一戦略内で両立する。
- 2026-02-25 追記（entry_probability reject 緩和）:
  - `quant-order-manager` で C の preserve-intent を次に更新した。
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.25`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.70`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE)=1.00`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=20`
  - 目的は `entry_probability_below_min_units` による未約定連発を減らし、
    反転局面で小ロットでも約定させること。
- 2026-02-25 追記（forecast-first 再稼働）:
  - `quant-scalp-ping-5s-c` は次を運用値とする。
    - `SCALP_PING_5S_C_ENABLED=1`
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`
    - `SCALP_PING_5S_C_CONF_FLOOR=68`
  - `entry_probability` 再補正を予測優勢側へ寄せる。
    - `ENTRY_PROBABILITY_ALIGN_DIRECTION/HORIZON/M1_WEIGHT=0.20/0.70/0.10`
    - `ENTRY_PROBABILITY_ALIGN_BOOST_MAX=0.14`
    - `ENTRY_PROBABILITY_ALIGN_PENALTY_MAX=0.42`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN/FLOOR=0.62/0.48`
    - `ENTRY_PROBABILITY_ALIGN_UNITS_MIN/MAX_MULT=0.65/1.20`
  - `quant-order-manager` の C preserve-intent はさらに通過寄りへ更新する。
    - `REJECT_UNDER=0.18`
    - `MIN_SCALE=0.80`
    - `MAX_SCALE=1.15`
    - `BOOST_PROBABILITY=0.65`
  - 目的は、予測優勢局面で「入らない/小さすぎる」を抑え、
    strategy local の方向判断をより直接に発注へ反映すること。
- 2026-02-25 追記（preserve-intent 再調整）:
  - 並行調整で aggressive に寄った C preserve-intent を、
    反転取りこぼし解消を主目的とした中立値へ戻す。
    - `REJECT_UNDER=0.25`
    - `MIN_SCALE=0.70`
    - `MAX_SCALE=1.00`
    - `BOOST_PROBABILITY=0.80`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=20` は維持
  - 目的は、低確率帯の過通過を抑えつつ
    `entry_probability_below_min_units` 由来の取りこぼしを減らすこと。
- 2026-02-25 追記（preserve-intent aggressive 再適用）:
  - VM 実測で `scalp_ping_5s_c_live` の `sell` 候補が
    `OPEN_SKIP note=entry_probability:entry_probability_below_min_units`
    （`02:28:13Z`, `02:28:41Z`）で落ちるケースを再確認したため、
    C の preserve-intent を再び forecast-first 側へ寄せた。
  - `quant-order-manager` の運用値:
    - `REJECT_UNDER=0.18`
    - `MIN_SCALE=0.80`
    - `MAX_SCALE=1.15`
    - `BOOST_PROBABILITY=0.65`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=20` は維持
  - 目的は、下落/上昇どちらでも予測優勢側の意図を通しやすくし、
    `entry_probability_reject` と `below_min_units` を同時に減らすこと。
- 2026-02-25 追記（逆向きショート抑制）:
  - VM 実測で `scalp_ping_5s_c_live` の `sell` 約定
    （`2026-02-25 04:01:37Z`, `client_order_id=...s6e04304b`）が
    `horizon_composite_side=long` の局面でも通過していたため、
    C ローカルで horizon 逆向き抑制を強化した。
  - `ops/env/scalp_ping_5s_c.env` の運用値:
    - `SCALP_PING_5S_C_HORIZON_BLOCK_SCORE=0.38`
    - `SCALP_PING_5S_C_HORIZON_OPPOSITE_UNITS_MULT=0.28`
    - `SCALP_PING_5S_C_HORIZON_OPPOSITE_BLOCK_MIN_MULT=0.08`
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_PENALTY_MAX=0.55`
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX=0.35`
  - 目的は「予測は使っているが減点止まり」で通っていた逆向きショートを、
    `block/scaled` 側に寄せて実発注を抑えること。
- 2026-02-26 追記（SL反映の根本修正）:
  - `execution/order_manager.py` の family判定を修正し、
    `scalp_ping_5s_c(_live)` / `scalp_ping_5s_d(_live)` でも
    次の env を正しく参照するようにした。
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_C`
    - `ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_C`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_D`
    - `ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_D`
  - 従来は B 以外が legacy 判定へ落ち、
    C で `MARKET_ORDER_TRADE_CLOSE` 偏重（`STOP_LOSS_ORDER` 不在）になる経路が残っていた。
  - 本修正で C/D も strategy別 hard-stop 方針を `order_manager` preflight へ反映する。

### scalp_ping_5s_flow 運用補足（stale closeout 抑止）
- `quant-order-manager` 環境で
  `SCALP_PING_5S_FLOW_PERF_GUARD_LOOKBACK_DAYS=1` を運用する。
- service timeout 時の local fallback でも同条件を維持するため、
  `quant-scalp-ping-5s-flow` worker 環境
  （`ops/env/quant-scalp-ping-5s-flow.env` と `ops/env/scalp_ping_5s_flow.env`）
  にも同じ `SCALP_PING_5S_FLOW_PERF_GUARD_LOOKBACK_DAYS=1` を設定する。
- `margin_closeout` は `hard/soft` の二段判定で扱う。
  - hard: `PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_TRADES` 未満の小標本、
    または `PERF_GUARD_MARGIN_CLOSEOUT_HARD_RATE` 超過（かつ min count 条件成立）。
  - soft: closeout が存在しても hard 条件を満たさない場合。
- `PERF_GUARD_MODE=reduce` 戦略では soft 判定を `warn` として通し、
  stale closeout だけで長時間全停止する経路を避ける。
- `strategy_entry` / `order_manager` の `env_prefix` 推論は
  `scalp_ping_5s_flow_* -> SCALP_PING_5S_FLOW` を優先し、
  `SCALP_PING_5S` への丸め込みをしない。
  - flowタグが `SCALP_PING_5S` に正規化されると、
    flow専用 `PERF_GUARD_*` が効かず、意図せず stale closeout block が継続するため。
- `config/strategy_exit_protections.yaml` には
  `scalp_ping_5s_flow` / `scalp_ping_5s_flow_live` を必ず定義し、
  `exit_profile` が default（`loss_cut_enabled=false`）へ落ちないようにする。
- `ops/env/quant-scalp-ping-5s-flow-exit.env` は次を運用値とする。
  - `RANGEFADER_EXIT_LOSS_CUT_ENABLED=1`
  - `RANGEFADER_EXIT_LOSS_CUT_REQUIRE_SL=0`
  - `RANGEFADER_EXIT_LOSS_CUT_HARD_PIPS=12.0`
  - `RANGEFADER_EXIT_LOSS_CUT_MAX_HOLD_SEC=900`
  - `SCALP_PRECISION_EXIT_OPEN_POSITIONS_RETRY_COUNT=1`
  - `SCALP_PRECISION_EXIT_OPEN_POSITIONS_RETRY_DELAY_SEC=0.35`
- 2026-02-26 追記（entry stale lock 修正）:
  - `market_data/spread_monitor.py:get_state()` は、
    in-process snapshot が `MAX_AGE_MS` を超えた場合に
    `tick_cache` fallback を優先評価する。
  - fallback が非 stale（または snapshot より新しい）なら
    fallback state を採用し、`snapshot_age_ms` を監査用に付与する。
  - stale cooldown 中でも fallback が fresh なら
    `is_blocked()` で `spread_stale` ブロックを即時解除する。
  - `workers/scalp_ping_5s/worker.py` は
    `spread_stale` 判定中に `continue` する前に snapshot fallback を再取得し、
    復帰できた場合は同サイクルで entry 判定へ戻す。
  - これにより、`spread_stale age=...` が継続して
    `entry-skip summary ... spread_blocked=...` に張り付く
    偽ブロック経路を回避する。
- 目的は、legacy建玉でも loss-cut/time-stop が機能する状態を維持しつつ、
  一過性の `position_manager` timeout で EXIT 判定サイクルが欠落する頻度を下げること。

### strategy_entry 先行プロファイル（2026-02-26）
- `execution/strategy_entry.py` は `market_order` / `limit_order` の共通経路で、
  黒板協調（`coordinate_entry_intent`）の直前に
  `entry_probability` の戦略ローカル再計算を行う。
- 計算は `entry_thesis.env_prefix`（または strategy_tag 正規化）をキーに
  `*_ENTRY_LEADING_PROFILE_*` 環境変数を参照する。
  - 主キー:
    - `ENTRY_LEADING_PROFILE_ENABLED`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW(_LONG/_SHORT)`
    - `ENTRY_LEADING_PROFILE_BOOST_MAX` / `...PENALTY_MAX`
    - `ENTRY_LEADING_PROFILE_WEIGHT_FORECAST/TECH/RANGE/MICRO/PATTERN`
    - `ENTRY_LEADING_PROFILE_UNITS_MIN_MULT` / `...MAX_MULT`
- 方向判定は strategy local の `entry_probability` を基準に、
  forecast/tech/range/micro/pattern の成分で補正する。
  共通レイヤは戦略の方向意図を反転させず、最終拒否は `REJECT_BELOW` 設定時のみ。
- 補正結果は `entry_thesis.entry_probability_leading_profile` に監査記録する。
- `limit_order` 経路でも黒板協調後の最終 `units` で
  `entry_thesis.entry_units_intent` を再同期する。
- 運用は `quant-v2-runtime.env` で
  `STRATEGY_ENTRY_LEADING_PROFILE_ENABLED=0` を既定にし、
  各 worker env のみで `*_ENTRY_LEADING_PROFILE_ENABLED=1` を設定する。
  （全戦略一括有効化はしない）
- 初期導入の strategy-local 設定先:
  - `ops/env/scalp_ping_5s_{b,c,d,flow}.env`
  - `ops/env/quant-m1scalper.env`
  - `ops/env/quant-scalp-{macd-rsi-div,rangefader,level-reject,false-break-fade,extrema-reversal}.env`
  - `ops/env/quant-scalp-{wick-reversal-blend,wick-reversal-pro,squeeze-pulse-break,tick-imbalance}.env`
  - `ops/env/quant-micro-rangebreak.env`

### orders.db ログ運用補足（lock耐性）
- `execution/order_manager.py` の orders logger は lock 検知時に
  `ORDER_DB_LOG_RETRY_*` の短時間 backoff 再試行を行う。
- 既定運用値:
  - `ORDER_DB_BUSY_TIMEOUT_MS=1500`
  - `ORDER_DB_LOG_RETRY_ATTEMPTS=6`
  - `ORDER_DB_LOG_RETRY_SLEEP_SEC=0.04`
  - `ORDER_DB_LOG_RETRY_BACKOFF=1.8`
  - `ORDER_DB_LOG_RETRY_MAX_SLEEP_SEC=0.30`
  - `ORDER_DB_FILE_LOCK_ENABLED=1`
  - `ORDER_DB_FILE_LOCK_TIMEOUT_SEC=0.30`
  - `ORDER_DB_FILE_LOCK_FAST_TIMEOUT_SEC=0.05`
- `execution/order_manager.py` は `orders.db.lock` (`flock`) で
  cross-process write を直列化する。
  - 2026-02-25 以降は service/fallback を同値
    `0.30s / 0.05s` で運用する。
  - さらに `execution/order_manager.py` は process 内 `RLock` を追加し、
    同一プロセス内同時ログ書き込みの自己競合を抑制する。
  - `scripts/cleanup_logs.sh` は hot DB (`orders.db/trades.db/metrics.db`) の
    `VACUUM` を既定で禁止し（checkpoint のみ）、
    cleanup timer 実行中の lock 競合を回避する。
- `ORDER_STATUS_CACHE_TTL_SEC`（既定 180）で
  `client_order_id` ごとの直近 status をメモリ保持する。
  - pre-service DB 記録を抑制する経路でも、
    `get_last_order_status_by_client_id` が local reject 理由を返せる。
  - `order_manager_none` の過剰表示を避け、実際の reject reason を優先する。
- `close_trade` 経路の orders 監査ログは fast-fail モードで記録する。
  - DB lock 中は `/order/close_trade` 応答遅延を避けることを優先し、
    ログ書き込みは短い retry budget で打ち切る。
  - 既定値:
    - `ORDER_DB_LOG_FAST_RETRY_ATTEMPTS=1`
    - `ORDER_DB_LOG_FAST_RETRY_SLEEP_SEC=0.0`
    - `ORDER_DB_LOG_FAST_RETRY_BACKOFF=1.0`
    - `ORDER_DB_LOG_FAST_RETRY_MAX_SLEEP_SEC=0.0`
- `quant-order-manager.service` は
  `ops/env/quant-v2-runtime.env` に加えて
  `ops/env/quant-order-manager.env` も読むため、
  両ファイルで `ORDER_DB_*` を同値に揃える。
  `quant-order-manager.env` 側が runtime より後段で上書きされる点に注意する。
  - 固定運用: `quant-order-manager.env` の
    `ORDER_MANAGER_SERVICE_ENABLED=1` を維持する。
    ここが `0` だと service unit は active 表示でも `:8300` が bind されない。
  - 同様に service-mode の最小ロット制御（`ORDER_MIN_UNITS_STRATEGY_*`）も
    `quant-order-manager.env` 側へ明示する。worker 個別 env のみで設定すると、
    order-manager サービスには反映されず、runtime 側の pocket 既定
    （例: `ORDER_MIN_UNITS_SCALP=900`）へフォールバックする。
- `ORDER_DB_LOG_PRESERVICE_IN_SERVICE_MODE=0` を維持し、
  service mode worker（`ORDER_MANAGER_SERVICE_ENABLED=1`）では
  `entry_probability_reject` / `probability_scaled` など
  pre-service 状態を `orders.db` に直接記録しない。
  - order-manager service を主ライターに寄せて同時書き込み競合を抑える。
  - service call 失敗時の fallback は journald の
    `order_manager service call failed` で監査する。
  - service応答の `result=null` は「正規の reject/skip」として扱い、
    local fallback を行わない（`_ORDER_MANAGER_SERVICE_UNHANDLED` sentinel）。
    これにより `OPEN_SKIP` 後の同一 `client_order_id` 再発注を防止する。
- 2026-02-25 追記（service timeout 連鎖の抑止）:
  - `quant-v2-runtime.env` は次を運用値とする。
    - `ORDER_MANAGER_SERVICE_TIMEOUT=20.0`
    - `ORDER_MANAGER_SERVICE_CONNECT_TIMEOUT=1.0`
    - `ORDER_MANAGER_SERVICE_POOL_CONNECTIONS=8`
    - `ORDER_MANAGER_SERVICE_POOL_MAXSIZE=32`
    - `ORDER_OANDA_REQUEST_TIMEOUT_SEC=8.0`
    - `ORDER_OANDA_REQUEST_POOL_CONNECTIONS=16`
    - `ORDER_OANDA_REQUEST_POOL_MAXSIZE=32`
  - `workers/order_manager/worker.py` は `ORDER_MANAGER_SERVICE_WORKERS` キーを解釈する。
    ただし現行 unit 起動互換を優先し、`quant-order-manager.env` は
    `ORDER_MANAGER_SERVICE_WORKERS=1` を運用値とする。
  - 2026-02-25 再追記（service 経路の長時間ブロッキング抑止）:
    - `ORDER_SUBMIT_MAX_ATTEMPTS=1`（`quant-order-manager.env`）
    - `ORDER_PROTECTION_FALLBACK_MAX_RETRIES=0`（`quant-order-manager.env`）
    - `quant-order-manager.env` と `quant-v2-runtime.env` の
      `ORDER_DB_LOG_RETRY_*` / `ORDER_DB_BUSY_TIMEOUT_MS` / `ORDER_DB_FILE_LOCK_*`
      は同値（`1500ms/6`, `0.30s/0.05s`）へ統一する。
      service/fallback 間の lock 耐性差で監査ログ品質が揺れないようにする。
    - `coordinate_entry_intent` は同一 `client_order_id` の事前削除 write を行わず、
      board 参照前の不要 write を減らして lock競合を抑える。
      `entry_intent_board` の整理は purge + expire 窓で行う。
    - `execution/order_manager.py` は `ORDER_SUBMIT_MAX_ATTEMPTS=1` を許容する
      （最小値を 2 固定にしない）。
    - 目的: `/order/market_order` が protection retry で 20 秒超に膨らみ、
      strategy 側の service timeout -> `order_manager_none` -> local fallback を連鎖させる経路を抑える。
  - `execution/order_manager.py` の service 失敗ログは payload 要約のみを記録し、
    巨大 `entry_thesis` 全量出力で journald/CPU を圧迫しない。
- 目的は「発注判断を変えずに」orders 監査ログ欠損を減らすこと。
  発注可否ロジック（perf/risk/policy/coordination）には影響しない。

### replay quality gate 運用補足（本番干渉抑制）
- `quant-replay-quality-gate.service` は replay/backtest を本番VMで実行するため、
  systemd で低優先度実行を固定する。
  - `Nice=15`
  - `IOSchedulingClass=idle`
  - `CPUWeight=20`
- `analysis/replay_quality_gate_worker.py` は
  `REPLAY_QUALITY_GATE_ENABLED=0` の場合、replay 本体を実行せず正常終了する。
  本番既定は `ops/env/quant-replay-quality-gate.env` で `1` とし、
  replay 品質監査を 1h 周期で継続する。
  一時的に本番負荷を抑える必要がある場合のみ、運用判断で `0` に切り替える。
- 目的は replay 品質検証を継続しつつ、稼働中の strategy/order/position worker を
  CPU 競合で阻害しないこと。

### cleanup 運用補足（hot DB 保護）
- `scripts/cleanup_logs.sh` は hot DB（`orders.db` / `trades.db` / `metrics.db`）への
  `VACUUM` を既定で実行しない。
  - `DB_VACUUM_SKIP_FILES=orders.db trades.db metrics.db`
  - `DB_VACUUM_ALLOW_HOT_DBS=0`
- 上記 DB は `wal_checkpoint(TRUNCATE)` のみ実行し、
  本番発注経路との lock 競合を避ける。

### position_manager タイムアウト運用補足（2026-02-20）
- `open_positions` の read timeout 連発を避けるため、`quant-v2-runtime.env` は次を運用値とする。
  - 固定運用: `quant-position-manager.env` の
    `POSITION_MANAGER_SERVICE_ENABLED=1` を維持する。
    ここが `0` だと service unit は active 表示でも `:8301` が bind されない。
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT=6.0`
  - `POSITION_MANAGER_SERVICE_PERFORMANCE_SUMMARY_TIMEOUT=25.0`
  - `POSITION_MANAGER_HTTP_RETRY_TOTAL=0`
  - `POSITION_MANAGER_OPEN_TRADES_HTTP_TIMEOUT=2.8`
  - `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_STALE_MAX_AGE_SEC=60.0`
  - `POSITION_MANAGER_ORDERS_DB_READ_TIMEOUT_SEC=0.08`
  - `POSITION_MANAGER_WORKER_OPEN_POSITIONS_TIMEOUT_SEC=5.0`
  - `POSITION_MANAGER_WORKER_SYNC_TRADES_TIMEOUT_SEC=8.0`
  - `POSITION_MANAGER_WORKER_PERFORMANCE_SUMMARY_TIMEOUT_SEC=20.0`
  - `POSITION_MANAGER_WORKER_FETCH_RECENT_TRADES_TIMEOUT_SEC=8.0`
  - `POSITION_MANAGER_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC=60.0`
  - `POSITION_MANAGER_WORKER_SYNC_TRADES_MAX_FETCH=1000`
  - `POSITION_MANAGER_SYNC_MIN_INTERVAL_SEC=2.0`
  - `POSITION_MANAGER_SYNC_CACHE_WINDOW_SEC=1.5`
  - `POSITION_MANAGER_ENTRY_META_CACHE_MAX_ENTRIES=20000`
  - `POSITION_MANAGER_POCKET_CACHE_MAX_ENTRIES=50000`
  - `POSITION_MANAGER_CLIENT_CACHE_MAX_ENTRIES=50000`
- 目的は、`position_manager` が不調時でも stale cache へ早めにフォールバックし、
  strategy worker 側の `position_manager_timeout` による entry skip を減らすこと。

### bq-sync 運用補足（performance summary 負荷平準化）
- 本番既定は `PIPELINE_PERF_SUMMARY_ENABLED=0` とし、
  `quant-bq-sync` の通常ループでは `pm.get_performance_summary()` を実行しない。
- `scripts/run_sync_pipeline.py` は `PIPELINE_PERF_SUMMARY_REFRESH_SEC`（既定 180s）で
  `pm.get_performance_summary()` をTTLキャッシュする。
- `PIPELINE_PERF_SUMMARY_STALE_MAX_AGE_SEC`（既定 900s）以内は、
  一時的な取得失敗時でも stale cache を使って GCS snapshot を継続する。
- 目的は `quant-bq-sync` の重い集計呼び出し頻度を下げ、
  `decision_latency_ms` / `data_lag_ms` への干渉を抑えること。

### scalp_macd_rsi_div 運用補足（legacy tag 互換）
- `quant-scalp-macd-rsi-div-exit` は `SCALP_PRECISION_EXIT_TAGS=scalp_macd_rsi_div_live` で運用する。
- `workers/scalp_macd_rsi_div.exit_worker` では
  `scalpmacdrsi*`（例: `scalpmacdrsic7c3e9c1`）を `scalp_macd_rsi_div_live` に正規化し、
  旧 client_id 由来タグでも EXIT 監視から外れないようにする。
- `config/strategy_exit_protections.yaml` に
  `scalp_macd_rsi_div_live` の `exit_profile` を必ず定義し、
  default（`loss_cut_enabled=false`）フォールバックで負け玉が残留しないようにする。

### scalp_macd_rsi_div_b 運用補足（精度優先プロファイル）
- `ops/env/quant-scalp-macd-rsi-div-b.env` は 2026-02-19 以降、
  looser mode ではなく precision-biased を既定とする。
- 主要ゲートは次を維持する:
  - `SCALP_MACD_RSI_DIV_B_REQUIRE_RANGE_ACTIVE=1`
  - `SCALP_MACD_RSI_DIV_B_RANGE_MIN_SCORE=0.35`
  - `SCALP_MACD_RSI_DIV_B_MAX_ADX=30`
  - `SCALP_MACD_RSI_DIV_B_MIN_DIV_SCORE=0.08`
  - `SCALP_MACD_RSI_DIV_B_MIN_DIV_STRENGTH=0.12`
  - `SCALP_MACD_RSI_DIV_B_MAX_DIV_AGE_BARS=24`
  - `SCALP_MACD_RSI_DIV_B_TECH_FAILOPEN=0`
- `workers/scalp_macd_rsi_div.worker` は `MIN_ENTRY_CONF` を実効評価し、
  `confidence < MIN_ENTRY_CONF` のエントリーを reject する
  （`gate_block confidence` ログで監査）。

### dynamic_alloc 運用補足（サイズ過多抑止）
- `scripts/dynamic_alloc_worker.py` は PF ガードを持ち、
  `pf < 1.0` の戦略を `lot_multiplier <= 0.95`、
  `pf < 0.7` を `lot_multiplier <= 0.90` に制限する。
- `trades < min_trades` の戦略は `lot_multiplier <= 1.00` とし、
  サンプル不足での過剰増量を防ぐ。
- `quant-dynamic-alloc.service` の `--target-use` は `0.88` を基準とし、
  `account.margin_usage_ratio` が高止まりする局面での margin block 連発を抑える。
- 2026-02-26 以降は `allocation_policy.soft_participation=true` を既定とし、
  `dynamic_alloc.json` から worker へ `allow_loser_block=false` /
  `allow_winner_only=false` を配布する。
  - 目的: 低品質戦略を hard stop せず、`lot_multiplier` の縮小で継続稼働させる。
  - `workers/common/dynamic_alloc.load_strategy_profile()` は
    `allocation_policy` を読み込み、`allow_loser_block` /
    `allow_winner_only` / `soft_participation` を返す。
  - `micro_runtime` / `M1Scalper` / `scalp_macd_rsi_div` は
    上記フラグを優先し、dyn alloc 由来の hard block を抑制する。

### ドローダウン恒久対策（2026-02-25 追記）
- 方針:
  - 「全停止」ではなく、低品質シグナルのみを機械的に拒否する。
  - EXIT経路は常時維持し、ENTRYは `perf_guard` / forecast contra / strategy local guard で絞る。
- 2026-02-25 以降の運用値:
  - `STRATEGY_CONTROL_GLOBAL_ENTRY_ENABLED=1`
  - `STRATEGY_CONTROL_GLOBAL_EXIT_ENABLED=1`
  - `STRATEGY_CONTROL_GLOBAL_LOCK=0`
  - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED=1`
- `perf_guard` 運用補足:
  - `mode=reduce` でも `sl_loss_rate` は hard block 扱い。
  - `margin_closeout` は hard/soft を分離し、
    hard 以外は `warn:margin_closeout_soft_*` として通す。
    - hard しきい値:
      `PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_TRADES` /
      `PERF_GUARD_MARGIN_CLOSEOUT_HARD_RATE` /
      `PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_COUNT`
  - `failfast` も hard/soft を分離し、
    hard 以外は `warn:failfast_soft:*` として通す。
    - hard しきい値:
      `PERF_GUARD_FAILFAST_HARD_PF` /
      `PERF_GUARD_FAILFAST_HARD_REQUIRE_BOTH`
  - 既定 `PERF_GUARD_RELAX_TAGS` から `M1Scalper` を除外し、
    劣化時の bypass を防ぐ。
  - `B/C/D/M1` は `PERF_GUARD_MODE=block` と failfast 閾値で運用し、
    PF/勝率が崩れた戦略だけ自動停止する。
  - `strategy_tag` が `-l<hex>` / `-s<hex>` 付きで保存される場合でも、
    `perf_guard` は同一戦略として集計する（hash suffix 互換）。
    これにより `warmup_n` 取りこぼしで劣化戦略が通過する経路を防ぐ。
  - `regime` フィルタ運用時でも、`margin_closeout / failfast / sl_loss_rate` は
    全体集計を併用して hard block 判定する（regime局所の良化で回避不可）。
  - `MicroLevelReactor` は専用 env で局所絞り込みを行う。
    - base units縮小、1サイクル信号数を1に制限、dyn alloc の loser block を早期適用、
      failfast を `n>=4` で有効化して低品質局面を早めに遮断する。
  - forecast gate は strategy別に品質閾値を持てる。
    - `FORECAST_GATE_EDGE_BLOCK_STRATEGY_<TAG>`
    - `FORECAST_GATE_EXPECTED_PIPS_GUARD_ENABLED_STRATEGY_<TAG>`
    - `FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_<TAG>`
    - `FORECAST_GATE_EXPECTED_PIPS_CONTRA_MAX_STRATEGY_<TAG>`
    - `FORECAST_GATE_TARGET_REACH_GUARD_ENABLED_STRATEGY_<TAG>`
    - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_<TAG>`
    - `<TAG>` は `SCALP_PING_5S_B_LIVE` のような underscore 形式でも指定可能。
      （hash suffix 付き `-l<hex>` / `-s<hex>` は base tag 解決で同一扱い）
  - 2026-02-25 時点の運用では、`FORECAST_GATE_STRATEGY_ALLOWLIST` を
    `MicroRangeBreak,MicroVWAPBound,MicroLevelReactor,scalp_ping_5s_b_live,scalp_ping_5s_c_live,scalp_ping_5s_flow_live`
    へ拡張し、毀損戦略の entry を「期待値・到達確率」基準で機械的に選別する。
- 再開条件:
  - 戦略ごとに直近ウィンドウで `PF>=1.0` かつ `win_rate>=0.50`
    （または戦略固有閾値）へ回復し、failfast理由が解消したことを確認して解除する。

### `+2000円/h` 目標向け scalp_fast 再配線（2026-02-26）
- 方針:
  - 損失寄与が継続する戦略を機械停止し、`scalp_ping_5s_c_live` の
    勝ち時間・勝ち方向へロットを集中する。
- 運用値:
  - 停止:
    - `SCALP_PING_5S_B_ENABLED=0`
    - `SCALP_PING_5S_D_ENABLED=0`
    - `SCALP_PING_5S_FLOW_ENABLED=0`
    - `M1SCALP_ENABLED=0`
    - `M1SCALP_BLOCK_HOURS_ENABLED=1`
    - `M1SCALP_BLOCK_HOURS_UTC=0-23`
  - 集中:
    - `SCALP_PING_5S_C_SIDE_FILTER=`（long固定を外し、方向は戦略ローカル判定に委譲）
    - `SCALP_PING_5S_C_ALLOW_HOURS_JST=18,19,22`
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=3000`
    - `SCALP_PING_5S_C_MAX_UNITS=7500`
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.70`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_C_LIVE=0.55`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_C_LIVE=1.00`
- 監査:
  - 反映後は `orders.db` の `strategy_tag/status` で
    `scalp_ping_5s_b_live` / `scalp_ping_5s_d_live` / `M1Scalper-M1` の
    新規 `filled` が止まっていることを確認する。
  - `scalp_ping_5s_c_live` は `JST 18,19,22` 以外の `filled` が無いことを確認する。

### `ALLOW_HOURS` soft-mode 運用（2026-02-26）
- 方針:
  - 時間帯優先（`ALLOW_HOURS_JST`）は維持しつつ、許可時間外を hard stop せず、
    strategy local の確率・信頼度・ロット縮小で継続稼働させる。
- 運用値:
  - `SCALP_PING_5S_C_ALLOW_HOURS_JST=18,19,22`
  - `SCALP_PING_5S_C_ALLOW_HOURS_SOFT_ENABLED=1`（C/D prefix は既定有効）
  - `SCALP_PING_5S_C_ALLOW_HOURS_OUTSIDE_UNITS_MULT=0.55`（既定）
  - `SCALP_PING_5S_C_ALLOW_HOURS_OUTSIDE_MIN_CONFIDENCE=72`（既定）
  - `SCALP_PING_5S_C_ALLOW_HOURS_OUTSIDE_MIN_ENTRY_PROBABILITY=0.62`（既定）
- 監査:
  - `orders.db` に `outside_allow_soft_confidence` /
    `outside_allow_soft_probability` が記録されること。
  - `entry_thesis.allow_hour_*` が保存され、`outside_allow_hour_jst` 単独で
    全停止しないことを確認する。

### 常時運転プロファイル（2026-02-26 更新）
- 方針:
  - 時間帯を entry 条件に使わず、全時間帯で稼働する。
  - 方向選別は戦略ローカルの `SIDE_FILTER` と `perf_guard/forecast` で行う。
- 運用値（`scalp_ping_5s_c_live`）:
  - `SCALP_PING_5S_C_ALLOW_HOURS_JST=`（空）
  - `SCALP_PING_5S_C_ALLOW_HOURS_SOFT_ENABLED=0`
  - `SCALP_PING_5S_C_SIDE_FILTER=`（空。long/short を自動判定）
  - `SCALP_PING_5S_C_MAX_ACTIVE_TRADES=4`
  - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=10`
  - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=1600`
  - `SCALP_PING_5S_C_MAX_UNITS=3000`
- 運用値（`M1Scalper-M1`）:
  - `M1SCALP_ENABLED=1`
  - `M1SCALP_BLOCK_HOURS_ENABLED=0`
  - `M1SCALP_BLOCK_HOURS_UTC=`
  - `M1SCALP_SIDE_FILTER=long`
- 監査:
  - `orders.db` で `outside_allow_hour_jst` が entry-stop 理由の主因になっていないこと。
  - `trades.db` で `C/M1 sell` の比率と損益が縮小していること。

### C系 `env_prefix` / perf_guard 運用補足（2026-02-26）
- `scalp_ping_5s_c_live` / `scalp_ping_5s_d_live` は、
  `order_manager` preflight 時に `SCALP_PING_5S_C` / `SCALP_PING_5S_D`
  prefix として解決する（`SCALP_PING_5S` へのフォールバック誤判定を禁止）。
- `scalp_ping_5s_c_live` の margin closeout hard しきい値:
  - `SCALP_PING_5S_C_PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_TRADES=1`
  - `SCALP_PING_5S_C_PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_COUNT=3`
  - `SCALP_PING_5S_C_PERF_GUARD_MARGIN_CLOSEOUT_HARD_RATE=0.50`
- 目的:
  - `mode=reduce` 戦略で軽微な closeout 履歴を soft 扱いにし、
    `perf_block:hard:margin_closeout_n=*` の過剰発生を防ぐ。
- 実装補足:
  - `quant-order-manager.service` では worker 個別 env が読まれないため、
    C戦略の `SCALP_PING_5S_C_PERF_GUARD_*` は
    `ops/env/quant-v2-runtime.env` にも同値を置いて preflight 判定へ反映する。
  - 互換運用として `SCALP_PING_5S_PERF_GUARD_MARGIN_CLOSEOUT_HARD_*` も同値化し、
    旧 prefix 解決経路が残っても hard reject へ偏らないようにする。
  - `client_order_id` 解析は
    `qr-<ts>-<strategy_tag>-<digest>` / `qr-<ts>-<pocket>-<strategy_tag>-<digest>`
    の両形式を許容し、`strategy_tag` 解決失敗で strategy 個別しきい値が
    無効化されないようにする。
  - `market_order` / `limit_order` の hard SL 判定は
    `ORDER_FIXED_SL_MODE` を基準にしつつも、
    `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_*` と
    `ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_{B,C,D}` が true の戦略では
    `sl_disabled` を解除して `stopLossOnFill` を有効化する。
    （fixed-mode OFF 時に strategy override が無視される不整合を防止）
  - 現行 runtime baseline は `ops/env/quant-v2-runtime.env` の
    `ORDER_FIXED_SL_MODE=0`（global OFF）で固定し、
    実運用の attach 可否は `ops/env/quant-order-manager.env` の
    strategy/family override で明示的に管理する。
  - `margin_closeout_soft_warmup_n=*` を運用監査し、停止ではなく縮小継続へ遷移させる。

### C tail-loss clamp（2026-02-26 追加）
- 背景（VM実測, UTC 05:15-06:15）:
  - `scalp_ping_5s_c_live` の勝率は高め（約 69%）だが、
    大口の `MARKET_ORDER_TRADE_CLOSE` が -6〜-9 pips を作り、
    1h 実現損益がマイナスへ傾く。
- 変更（current）:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_MAX_ACTIVE_TRADES=4`
    - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=10`
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=1600`
    - `SCALP_PING_5S_C_MAX_UNITS=3000`
    - `SCALP_PING_5S_C_SL_BASE_PIPS=1.3`
    - `SCALP_PING_5S_C_FORCE_EXIT_MAX_HOLD_SEC=75`
    - `SCALP_PING_5S_C_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=1.0`
    - `SCALP_PING_5S_C_SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=1.0`
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.76`
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR=0.62`
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_UNITS_MAX_MULT=1.00`
    - `SCALP_PING_5S_C_ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MAX_MULT=1.00`
    - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.40`
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.55`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_C_LIVE=0.55`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_C_LIVE=1.00`
    - `ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY_STRATEGY_SCALP_PING_5S_C_LIVE=0.85`
    - `SCALP_PING_5S_PERF_SCALE_ENABLED=0`
    - `SCALP_PING_5S_C_PERF_SCALE_ENABLED=0`
- 意図:
  - 停止/時間帯ブロックなしで、低品質エントリー通過とロット上振れを抑制し、
    通過注文の尾損失を縮小する。

### C negative-close policy（2026-02-26 追加）
- 背景:
  - `scalp_ping_5s_c_live` で `close_reject_no_negative` が多発し、
    EXIT シグナル後も負けを保持し続けるケースが確認された。
- 変更:
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_c` / `scalp_ping_5s_c_live` に
      `neg_exit.allow_reasons: ["*"]` を追加。
- 意図:
  - C は no-block 方針で EXIT 機動性を優先し、
    no-negative ガードによる引っ張り損失を抑制する。

### C lot overrun clamp（2026-02-26 追加）
- 背景:
  - VM 実測で `SCALP_PING_5S_C_MAX_UNITS=4500` 設定下でも、
    `scalp_ping_5s_c_live` の実送信ユニットが 4500 超に上振れするケースがあった。
- 実装:
  - `workers/scalp_ping_5s/worker.py`
    - TECH sizing 適用後と `market_order` 送信直前の 2 点で
      `units_risk` / `MAX_UNITS` / `MIN_UNITS` クランプを強制。
  - `execution/strategy_entry.py`
    - 協調後 `final_units` が strategy 要求量（raw `units`）を超えないよう上限化。
- 運用意図:
  - 戦略が出したサイズ意図を「最大値」として扱い、
    協調/補正レイヤ由来の上振れでリスクが膨らむ経路を遮断する。

### Release gate
- PF>1.1、勝率>52%、最大 DD<5% を 2 週間連続で満たすと実弾へ昇格。

## 2. リスク計算とロット
- `pocket_equity = account_equity * pocket_ratio`
- `POCKET_MAX_RATIOS` は macro/micro/scalp/scalp_fast すべて 0.85 を起点に ATR・PF・free_margin で動的スケールし、最終値を 0.92〜1.0 にクランプ（scalp_fast は scalp から 0.35 割合で分岐）
- `risk_pct = 0.02`、`risk_amount = pocket_equity * risk_pct`
- 1 lot の pip 価値は 1000 JPY
- `lot = min(MAX_LOT, round(risk_amount / (sl_pips * 1000), 3))` → `units = int(round(lot * 100000))`
- `abs(units) < 1000` は発注しない
- 最小ロット下限: macro 0.1, micro 0.0, scalp 0.05（env で上書き可）
- `clamp_sl_tp(price, sl, tp, side)` で 0.001 丸め、SL/TP 逆転時は 0.1 バッファ

### 円換算優先の判定軸（2026-02-26）
- `execution/risk_guard.py`:
  - `check_global_drawdown()` は `realized_pl`（JPY）を優先して DD 比率を計算する。
  - DD母数は `GLOBAL_DD_EQUITY_BASE_JPY` または `update_dd_context()` で更新した口座 equity ヒントを使う。
  - 集計窓は `GLOBAL_DD_LOOKBACK_DAYS`（既定 7 日）。
  - `loss_cooldown_status()` の連敗判定も `realized_pl` 優先（`LOSS_COOLDOWN_MIN_ABS_JPY` で微小ノイズを除外可能）。
- `workers/common/perf_guard.py`:
  - PF/勝率の集計列は `PERF_GUARD_VALUE_COLUMN` で選択可能（既定は `realized_pl` 優先、`pl_pips` はフォールバック）。
  - `avg_pips` は値幅品質の補助指標として維持し、資金管理の主軸には使わない。

## 3. OANDA API マッピング

| Strategy action | REST 注文 | units 符号 | SL/TP | 備考 |
|-----------------|-----------|------------|-------|------|
| `OPEN_LONG` | `MARKET` | `+abs(units)` | `stopLossOnFill`, `takeProfitOnFill` | `timeInForce=FOK`, `positionFill=DEFAULT` |
| `OPEN_SHORT` | `MARKET` | `-abs(units)` | 同上 | ask/bid 逆転チェック後送信 |
| `CLOSE` | `MARKET` | 既存ポジの反対売買 | 指定なし | `OrderCancelReplace` で逆指値削除 |
| `HOLD` | 送信なし | 0 | なし | Strategy ループ継続 |

## 4. 発注冪等性とリトライ
- `client_order_id` は 90 日ユニーク。
- OANDA 429/5xx/timeout は 0.5s→1.5s→3.5s（+100ms ジッター）で最大 3 回リトライ。
- 同一 ID を再利用。
- WebSocket 停止検知時は `halt_reason="stream_disconnected"` を残して停止。

## 5. 安全装置と状態遷移

### 安全装置
- Pocket DD: micro 5% / macro 15% / scalp 3% / scalp_fast 2% で該当 pocket 新規停止。
- Global DD 20% でプロセス終了。
- Event モード（指標 ±30min）は micro 新規禁止。
- Timeout: OANDA REST 5s 再試行。
- Healthbeat は `main.py` から 5min ping。

### 状態遷移

| 状態 | 遷移条件 | 動作 |
|------|----------|------|
| `NORMAL` | 初期 | 全 pocket 取引許可 |
| `EVENT_LOCK` | 経済指標 ±30min | `micro` 新規停止、建玉縮小ロジック発動 |
| `MICRO_STOP` | `micro` pocket DD ≥5% | `micro` 決済のみ、`macro` 継続 |
| `GLOBAL_STOP` | Global DD ≥20% または `healthbeat` 欠損>10min | 全取引停止、プロセス終了 |
| `RECOVERY` | DD が閾値の 80% 未満、24h 経過 | 新規建玉再開前に `main.py` ドライラン |

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

### Forecast/Perf 再配線（2026-02-26）
- 目的:
  - レンジ局面での「range専用戦略が止まり、scalp_fast が低EVで通る」偏りを補正する。
- 変更要点:
  - `quant-v2-runtime.env`:
    - `RangeFader` / `MicroLevelReactor` / `MicroVWAP{Revert,Bound}` の
      `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE_*` を `0.40` へ緩和。
    - `RangeFader` の `FORECAST_GATE_EDGE_BLOCK_*` を `0.32` へ緩和。
    - `MicroLevelReactor` の expected/target しきい値を
      `0.20/-0.02/0.28` -> `0.12/-0.03/0.22` へ調整。
  - `quant-order-manager.env`:
    - `scalp_ping_5s_{b,c,flow}_live` の
      `FORECAST_GATE_EDGE_BLOCK` / `EXPECTED_PIPS_MIN` / `EXPECTED_PIPS_CONTRA_MAX` /
      `TARGET_REACH_MIN` を引き上げ、
      低EVエントリーを preflight で追加遮断。
    - `scalp_ping_5s_{b,c}` の preserve-intent / perf-guard
      （`REJECT_UNDER`, `MAX_SCALE`, `FAILFAST_*`, `SL_LOSS_RATE_MAX`）を tighten。
- 運用意図:
  - レンジ専用戦略の入口復帰と、`scalp_fast` の損失主導エントリー抑制を同時に行う。

### B/C no-stop 改善チューニング（2026-02-26）
- 背景（VM 直近24h）:
  - `scalp_ping_5s_b_live`: `481 trades / -3144.3 JPY / PF 0.359`
  - `scalp_ping_5s_c_live`: `587 trades / -7025.8 JPY / PF 0.434`
  - `B` は `entry_probability=0.85-0.92` 帯の long で劣化が集中。
- 実装方針:
  - 戦略停止ではなく `PERF_GUARD_MODE=reduce` を基本にし、hard条件のみ拒否する。
  - `B` は `preserve-intent` / `forecast-gate` / `probability-band` を同時に tighten し、
    通過時ロットも `MAX_SCALE` と `MAX_UNITS` で圧縮する。
  - `C` は再有効化するが、`BASE_ENTRY_UNITS` と `MAX_UNITS` を抑えた縮小運転にする。
- 主要パラメータ（反映先: `ops/env/scalp_ping_5s_{b,c}.env`, `ops/env/quant-order-manager.env`）:
  - `B`: `REJECT_UNDER=0.68`, `MIN/MAX_SCALE=0.25/0.55`, `MAX_UNITS=900`
  - `B`: `FORECAST_GATE_EDGE_BLOCK=0.72`, `EXPECTED_PIPS_MIN=0.24`, `TARGET_REACH_MIN=0.34`
  - `C`: `ENABLED=1`, `BASE_ENTRY_UNITS=260`, `MAX_UNITS=420`
  - `C`: `PERF_GUARD_MODE=reduce`
  - `FORECAST_GATE_STRATEGY_ALLOWLIST` に `scalp_ping_5s_b_live` を追加し、B の forecast preflight を実効化。
  - `quant-v2-runtime` 側の `FORECAST_GATE_*_SCALP_PING_5S_{B,C}_LIVE` もしきい値を同値化し、
    worker runtime と order-manager preflight の判定差を縮小。

### B/C 方向精度リセット（2026-02-27）
- 背景（VM実測）:
  - post-check で B/C 約定が `sell` 偏重となり、方向一致率が 50% を下回った。
  - `entry_probability` 低帯の通過が継続し、B/C の pips がマイナス寄与。
- 反映先:
  - `ops/env/scalp_ping_5s_b.env`
  - `ops/env/scalp_ping_5s_c.env`
  - `ops/env/quant-order-manager.env`
  - `systemd/quant-scalp-ping-5s-b.service`
- 変更要点:
  - `SCALP_PING_5S_{B,C}_SIDE_FILTER=none`
  - `SCALP_PING_5S_{B,C}_ALLOW_NO_SIDE_FILTER=1`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER` を
    `B=0.64`, `C=0.62` へ引き上げ
  - `SCALP_PING_5S_{B,C}_ENTRY_LEADING_PROFILE_ENABLED=1`
  - leading profile の `REJECT_BELOW` を
    `B=0.64/0.70(short)`, `C=0.62/0.68(short)` へ引き上げ
- 運用意図:
  - side固定を解除して方向選択を戦略ローカル判定へ戻す。
  - 低 edge エントリーは preserve-intent と leading profile の二段で遮断する。

### B/C 負け筋遮断（2026-02-26 追加）
- 14日集計で `scalp_ping_5s_b_live` と `scalp_ping_5s_c_live` の赤字継続を確認し、
  全時間稼働を中止して「勝っている条件のみ稼働」へ変更。
- `ops/env/scalp_ping_5s_b.env`:
  - `SIDE_FILTER=buy`
  - `ALLOW_HOURS_JST=16,17,18,23`
  - `PERF_GUARD_MODE=block`
- `ops/env/scalp_ping_5s_c.env`:
  - `SIDE_FILTER=buy`
  - `ALLOW_HOURS_JST=18,19,22`
  - `PERF_GUARD_MODE=block`
- `config/worker_reentry.yaml` でも同条件を block リストで二重化し、
  env差し替え時でも損失時間帯へ再流入しないよう固定。
- `scripts/dynamic_alloc_worker.py` は `margin_closeout_rate` と
  `realized_jpy` 悪化への罰則を追加し、`lot_multiplier` を早期縮小する。
- 当日回収モード（2026-02-26）:
  - `ORDER_MANUAL_MARGIN_GUARD_*` は `ENABLED=1` を維持したまま、
    `0.05 / 0.07 / 3000` へ緩和して「手動玉1本で全停止」を回避。
  - `b/c` は `buy + allow_hours` 制約を維持しつつ
    `BASE_ENTRY_UNITS` / `MAX_UNITS` / `MAX_ORDERS_PER_MINUTE` を引き上げる。

### 停止なし・時間帯停止なしへの再構成（2026-02-26 追加）
- 運用方針を「常時動的トレード」に戻し、停止/時間帯ブロック前提の設定を解除。
- 反映点:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_ALLOW_HOURS_JST=`（時間帯制約解除）
    - `SCALP_PING_5S_B_PERF_GUARD_MODE=reduce`
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ALLOW_HOURS_JST=`（時間帯制約解除）
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`
    - `SCALP_PING_5S_PERF_GUARD_MODE=reduce`（fallback経路同値）
  - `config/worker_reentry.yaml`
    - `M1Scalper`, `MicroPullbackEMA`, `MicroLevelReactor`,
      `scalp_ping_5s_{b,c,d}_live` の `block_jst_hours` を `[]` へ更新。
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_{c,c_live}.neg_exit` に
      `strict_no_negative: false`, `allow_reasons: ['*']`, `deny_reasons: []` を明示。
- 意図:
  - 時間帯で止めず、strategyローカル判定 + preflightリスクでサイズ/通過を動的制御する。
  - `close_reject_no_negative` による EXIT 詰まりを減らし、損失玉滞留を避ける。

### Exit
- 各戦略の `exit_worker` が最低保有時間とテクニカル/レンジ判定を踏まえ、PnL>0 決済が原則。
- 例外は強制 DD / ヘルス / マージン使用率 / 余力 / 未実現DDの総合判定のみ。
- 共通 `execution/exit_manager.py` は常に空を返す互換スタブ。
- `execution/stage_tracker` がクールダウンと方向別ブロックを管理。

### Manual Margin Pressure Guard（2026-02-26）
- `execution/order_manager.py` は、`manual/unknown` 建玉が残っている状態で
  口座余力が閾値未達の場合、非manualの新規ENTRYを `manual_margin_pressure` で拒否する。
- 判定キー（`ops/env/quant-order-manager.env`）:
  - `ORDER_MANUAL_MARGIN_GUARD_ENABLED`
  - `ORDER_MANUAL_MARGIN_GUARD_MIN_TRADES`
  - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO`
  - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER`
  - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY`
- 監査:
  - `orders.db` の `status=manual_margin_pressure`
  - メトリクス `order_manual_margin_block`
- 運用意図:
  - 手動建玉が余力を占有する局面で戦略建玉を積み増ししない。
  - `MARKET_ORDER_MARGIN_CLOSEOUT` 連鎖を入口側で遮断する。

### Quote Stability Guard（2026-02-26）
- `execution/order_manager.py` は quote を `order_manager` 側で健全性チェックしてから利用する。
  - cross（`ask<=bid`）、負/ゼロ spread、異常に広い spread は不健全として破棄。
  - `ORDER_QUOTE_MAX_SPREAD_PIPS`（既定 `4.0`）で異常 spread 上限を制御する。
- 新規ENTRYでは `ORDER_REQUIRE_HEALTHY_QUOTE_FOR_ENTRY=1`（既定）時に、
  健全 quote が取得できない場合 `status=quote_unavailable` で skip する。
- quote 系 reject（`OFF_QUOTES` 等）では、`ORDER_QUOTE_RETRY_*` 設定に従い
  再クォートして再送する（`status=quote_retry`）。
  - `ORDER_SUBMIT_MAX_ATTEMPTS=1` 運用でも、quote 系 reject に限っては
    retry 予算を別枠で確保し、`market_order` / `limit_order` の両方で再送する。
- 主要運用キー（`ops/env/quant-order-manager.env`）:
  - `ORDER_REQUIRE_HEALTHY_QUOTE_FOR_ENTRY`
  - `ORDER_QUOTE_MAX_SPREAD_PIPS`
  - `ORDER_QUOTE_FETCH_ATTEMPTS`, `ORDER_QUOTE_FETCH_SLEEP_SEC`
  - `ORDER_QUOTE_RETRY_MAX_RETRIES`, `ORDER_QUOTE_RETRY_SLEEP_SEC`
- 2026-02-26 運用プロファイル（再取得耐性強化）:
  - `ORDER_TICK_QUOTE_MAX_AGE_SEC=1.2`
  - `ORDER_QUOTE_FETCH_ATTEMPTS=4`
  - `ORDER_QUOTE_FETCH_SLEEP_SEC=0.10`
  - `ORDER_QUOTE_FETCH_MAX_SLEEP_SEC=0.50`
  - `ORDER_QUOTE_RETRY_MAX_RETRIES=2`
  - `ORDER_QUOTE_RETRY_SLEEP_SEC=0.45`
  - `ORDER_QUOTE_RETRY_SLEEP_BACKOFF=1.8`
  - `ORDER_QUOTE_RETRY_MAX_SLEEP_SEC=1.5`

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
  replay 品質監査を 3h 周期で継続する。
  `REPLAY_QUALITY_GATE_AUTO_IMPROVE_MIN_APPLY_INTERVAL_SEC`（既定 10800）により、
  `worker_reentry` 反映は最小間隔を守って実施する（間隔内は解析のみ）。
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
  - 時間帯ブロックの因果効果（A/B）は未実証のため、`ALLOW_HOURS_JST` は使わない。
- 運用値（`scalp_ping_5s_c_live`）:
  - `SCALP_PING_5S_C_ALLOW_HOURS_JST=`（空）
  - `SCALP_PING_5S_C_ALLOW_HOURS_SOFT_ENABLED=0`
  - `SCALP_PING_5S_C_SIDE_FILTER=buy`
  - `SCALP_PING_5S_C_MAX_ACTIVE_TRADES=1`
  - `SCALP_PING_5S_C_MAX_PER_DIRECTION=1`
  - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=4`
  - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=700`
  - `SCALP_PING_5S_C_MAX_UNITS=1200`
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

### C flip de-risk（2026-02-26 追加）
- 背景（VM実測, UTC 07:05 前後）:
  - `scalp_ping_5s_c_live` の直近60件で `flip_any=19`（fast/sl/side）、
    反転あり平均 `pl_pips=-3.12` と反転なし（`-0.84`）より劣後。
  - `STOP_LOSS_ORDER` が直近12件で11件とクラスター化。
- 運用値（`ops/env/scalp_ping_5s_c.env`）:
  - fast flip を厳格化:
    - `FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN=0.58`
    - `FAST_DIRECTION_FLIP_HORIZON_SCORE_MIN=0.38`
    - `FAST_DIRECTION_FLIP_HORIZON_AGREE_MIN=3`
    - `FAST_DIRECTION_FLIP_NEUTRAL_HORIZON_BIAS_SCORE_MIN=0.82`
    - `FAST_DIRECTION_FLIP_MOMENTUM_MIN_PIPS=0.16`
    - `FAST_DIRECTION_FLIP_COOLDOWN_SEC=1.2`
    - `FAST_DIRECTION_FLIP_CONFIDENCE_ADD=1`
  - sl-streak flip を厳格化:
    - `SL_STREAK_DIRECTION_FLIP_MIN_SIDE_SL_HITS=3`
    - `SL_STREAK_DIRECTION_FLIP_MIN_TARGET_MARKET_PLUS=2`
    - `SL_STREAK_DIRECTION_FLIP_FORCE_STREAK=4`
    - `SL_STREAK_DIRECTION_FLIP_METRICS_OVERRIDE_ENABLED=0`
    - `SL_STREAK_DIRECTION_FLIP_DIRECTION_SCORE_MIN=0.55`
    - `SL_STREAK_DIRECTION_FLIP_HORIZON_SCORE_MIN=0.42`
  - `SIDE_METRICS_DIRECTION_FLIP_ENABLED=0`（Cは停止）
  - 併せてロット/密度を圧縮:
    - `MAX_ORDERS_PER_MINUTE=4`
    - `BASE_ENTRY_UNITS=700`
    - `MAX_UNITS=1200`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.60`
    - `...MIN_SCALE...=0.40`, `...MAX_SCALE...=0.85`
- 意図:
  - 反転ロジックは維持しつつ、SL連発帯での反転起点エントリーと過大ロットを抑える。

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

### C emergency stop + wildcard reasonless close（2026-02-26 追記）
- 背景（VM実測）:
  - `scalp_ping_5s_c_live` は `perf_block` が大量発生する一方で `filled` も継続し、
    直近24hで `close_reject_no_negative` が 50 件超発生。
  - `orders.db` には `close_reject_no_negative` で `error_code` が空のケースが多く、
    `exit_reason` 欠損時に `neg_exit.allow_reasons=["*"]` の意図が反映されない経路が残っていた。
- 変更:
  - `ops/env/scalp_ping_5s_c.env`, `ops/env/quant-scalp-ping-5s-c.env`:
    - `SCALP_PING_5S_C_ENABLED=0`（緊急停止）。
  - `execution/order_manager.py`:
    - `_reason_matches_tokens()` を修正し、token が `*` の場合は `exit_reason` 欠損時も一致扱いに変更。
- 意図:
  - 最大損失源のエントリーを即停止して流出を止める。
  - reason欠損での `close_reject_no_negative` を減らし、含み損ポジションの解放を優先する。

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

### C/B emergency de-risk（2026-02-26 追加）
- 背景（VM実測, UTC 07:24 集計）:
  - 直近24h:
    - `scalp_ping_5s_c_live`: `587 trades / -7025.8 JPY / -749.1 pips / win 53.2% / PF 0.434`
    - `scalp_ping_5s_b_live`: `481 trades / -3144.3 JPY / -414.7 pips / win 24.9% / PF 0.359`
  - 直近6h: `scalp_ping_5s_c_live long` が `188 trades / -3258.0 JPY / -231.4 pips`。
- 変更値（`ops/env/scalp_ping_5s_c.env`）:
  - 取引密度/サイズ圧縮:
    - `MAX_ORDERS_PER_MINUTE=2`
    - `BASE_ENTRY_UNITS=450`
    - `MAX_UNITS=700`
  - entry品質:
    - `CONF_FLOOR=78`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR=0.70`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.52`
    - `ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.60`
    - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=0.85`
  - failfast:
    - `PERF_GUARD_FAILFAST_MIN_TRADES=6`
    - `PERF_GUARD_FAILFAST_PF=0.90`
    - `PERF_GUARD_FAILFAST_WIN=0.48`
    - `PERF_GUARD_SL_LOSS_RATE_MAX=0.55`
- 変更値（`ops/env/quant-order-manager.env`）:
  - C preserve-intent:
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.72`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.30`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.55`
  - B preserve-intent:
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.62`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.35`
    - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.65`
  - fallback `perf_guard` は `SCALP_PING_5S*` / `SCALP_PING_5S_C*` とも `block` に統一。
- 変更値（`systemd/quant-scalp-ping-5s-b.service`）:
  - `BASE_ENTRY_UNITS=600`
  - `MAX_UNITS=1200`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.64`
  - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.65`
- 意図:
  - C の悪化帯では「件数×ロット」を同時に落として尾損失を抑制する。
  - B は再悪化防止を優先し、実効ロット上限を unit override 側でも固定する。

### C/B setup-perf gate 強化（2026-02-26 追加）
- 背景（VM実測, UTC 2026-02-26 08:47 集計）:
  - 直近1h（時刻正規化後）:
    - `scalp_ping_5s_c_live`: `28 trades / -143.4 JPY / -19.4 pips`
  - 同時間帯の `orders.db` で `perf_block` と `filled` が併存し、
    failfast 単体では劣化クラスター停止が遅れる局面が残存。
- 変更値（`ops/env/scalp_ping_5s_c.env`）:
  - 取引密度/サイズ:
    - `MAX_ORDERS_PER_MINUTE=1`
    - `BASE_ENTRY_UNITS=320`
    - `MAX_UNITS=520`
  - `PERF_GUARD` の短期反応化:
    - `LOOKBACK_DAYS=1`
    - `MIN_TRADES=16`
    - `PF_MIN=0.90`
    - `WIN_MIN=0.47`
    - `HOURLY_MIN_TRADES=6`
    - `REGIME_FILTER=1`
    - `REGIME_MIN_TRADES=10`
  - setup 判定（hour×side×regime）:
    - `SETUP_ENABLED=1`
    - `SETUP_USE_HOUR=1`
    - `SETUP_USE_DIRECTION=1`
    - `SETUP_USE_REGIME=1`
    - `SETUP_MIN_TRADES=6`
    - `SETUP_PF_MIN=0.95`
    - `SETUP_WIN_MIN=0.50`
    - `SETUP_AVG_PIPS_MIN=0.00`
  - fallback prefix `SCALP_PING_5S_*` へ同値を併記。
- 変更値（`ops/env/scalp_ping_5s_b.env`）:
  - `LOOKBACK_DAYS=1`
  - `MIN_TRADES=20`
  - `PF_MIN=0.88`
  - `WIN_MIN=0.46`
  - `HOURLY_MIN_TRADES=6`
  - setup 判定:
    - `SETUP_MIN_TRADES=6`
    - `SETUP_PF_MIN=0.92`
    - `SETUP_WIN_MIN=0.48`
    - `SETUP_AVG_PIPS_MIN=0.00`
- 変更値（`ops/env/quant-order-manager.env`）:
  - `SCALP_PING_5S_B_*` / `SCALP_PING_5S_C_*` / `SCALP_PING_5S_*`
    の `PERF_GUARD` と setup しきい値を上記へ同値化。
- 意図:
  - 静的時間帯ブロックに頼らず、`hour×side×regime` の直近実績で
    preflight を動的に選別する。

### B short隔離 + C一時停止（2026-02-26 追加）
- 背景（VM実測, UTC 2026-02-26 08:05）:
  - 直近日次:
    - `scalp_ping_5s_c_live`: 2/24 `-1574.2JPY`, 2/25 `-3628.7JPY`, 2/26 `-5351.1JPY`
    - `scalp_ping_5s_b_live`: 2/24 `-2246.6JPY`, 2/25 `-3716.2JPY`, 2/26 `-615.9JPY`
  - 90日（実データ範囲）:
    - `B`: `4978 trades / -41156.5JPY / -5288.6pips / PF 0.427`
    - `C`: `855 trades / -10554.0JPY / -1006.7pips / PF 0.440`
  - B の短期悪化クラスター（JST x short）:
    - `00: n=62, EV=-13.2pips`
    - `23: n=125, EV=-8.1888pips`
    - `15: n=47, EV=-7.2915pips`
- 変更値:
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_ENABLED=0`（再検証まで一時停止）
- 意図:
  - 負けの主因フローを先に隔離し、エッジ確認済み導線へ資源を戻す。

### サイド固定撤回（2026-02-26 追加）
- 背景:
  - 固定 `long-only/short-only` は応急停止としては有効でも、
    市況依存の運用設計としては不十分。
- 変更値:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_SIDE_FILTER` を削除。
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_SIDE_FILTER` を削除（C停止解除時の固定方向化を防止）。
- 以後の方針:
  - 方向選別は `PERF_GUARD_SETUP`（`USE_HOUR=1/USE_DIRECTION=1/USE_REGIME=1`）、
    `DIRECTION_BIAS`、`HORIZON_BIAS` で動的に行う。

### order preflight SLO ガード（2026-02-26）
- 目的:
  - 実行遅延（`data_lag_ms` / `decision_latency_ms`）が悪化した局面で新規 entry を止め、
    `MARKET_ORDER_MARGIN_CLOSEOUT` の連鎖を抑える。
- 実装:
  - `execution/order_manager.py` preflight に `workers/common/slo_guard.py` を追加。
  - 判定は `metrics.db` の mode=`strategy_control` を対象に、latest + p95 を評価する。
  - reject 時は `status=slo_block` で orders 監査に残し、
    `order_slo_block` metric を必須送信する。
- 運用キー（`ops/env/quant-order-manager.env`）:
  - `ORDER_SLO_GUARD_ENABLED=1`
  - `ORDER_SLO_GUARD_APPLY_POCKETS=scalp_fast,scalp,micro`
  - `ORDER_SLO_GUARD_LOOKBACK_SEC=180`
  - `ORDER_SLO_GUARD_SAMPLE_MIN=8`
  - `ORDER_SLO_GUARD_DATA_LAG_MAX_MS=3500`
  - `ORDER_SLO_GUARD_DECISION_LATENCY_MAX_MS=2500`
  - `ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS=5000`
  - `ORDER_SLO_GUARD_DECISION_LATENCY_P95_MAX_MS=4000`

### 市場時間中の重処理スキップ（2026-02-26）
- `analysis/replay_quality_gate_worker.py`:
  - `REPLAY_QUALITY_GATE_SKIP_WHEN_MARKET_OPEN=1` のとき market open 中は skip。
- `analysis/trade_counterfactual_worker.py`:
  - `COUNTERFACTUAL_SKIP_WHEN_MARKET_OPEN=1` のとき market open 中は skip。
- 意図:
  - replay/counterfactual の重処理と実トレード導線（market-data / strategy / order-manager）
    の資源競合を避け、preflight latency 劣化を抑える。

### intraday攻め設定（2026-02-26 追加）
- 目的:
  - `scalp_ping_5s_c_live` と `MicroCompressionRevert-short` の当日負け寄与を縮小し、
    利益寄与のある時間帯/戦略へロットを再配分する。
- エントリー制御:
  - `scalp_ping_5s_b`
    - 稼働時間制限は使わず常時稼働（`ALLOW_HOURS_JST=`）
    - `BASE_ENTRY_UNITS=1800`, `MAX_UNITS=3600`
    - `PERF_GUARD_MODE=reduce`（停止ではなく縮小運転）
    - local fallback 整合: `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER...B_LIVE=0.64`
    - order-manager 側は通過閾値を `0.64` へ緩和し、利益時間帯で約定回復を優先。
  - `scalp_ping_5s_c`
    - 稼働時間制限は使わず常時稼働（`ALLOW_HOURS_JST=`）
    - `BASE_ENTRY_UNITS=400`, `MAX_UNITS=900`, `CONF_FLOOR=82`
    - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`
    - `SCALP_PING_5S_PERF_GUARD_MODE=reduce`（fallback整合）
    - local fallback 整合: `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER...C_LIVE=0.76`
    - order-manager 側は低EV通過を厳格化:
      - `REJECT_UNDER=0.76`
      - `EDGE_BLOCK=0.78`
      - `EXPECTED_PIPS_MIN=0.32`
      - `TARGET_REACH_MIN=0.42`
- 共通runtime整合:
  - `ops/env/quant-v2-runtime.env` の forecast gate も同値へ更新。
    - B: `0.70 / 0.20 / 0.30`
    - C: `0.78 / 0.32 / 0.42`
- micro配分:
  - `MicroCompressionRevert` 専用workerは停止せず、縮小配分で継続運転。
  - `MicroVWAPRevert` / `MicroRangeBreak` / `MicroTrendRetest` / `MomentumBurstMicro` は
    `MICRO_MULTI_BASE_UNITS=42000` へ増量（rangebreak/trendretest/momentumburst は loop 4.0s）。
- 運用意図:
  - 低期待値フローを block/reduce で先に切り、同日中は高寄与フローへサイズを寄せる。

### no-stop再調整（2026-02-26 追記）
- `scalp_ping_5s_b` の `hard failfast` 連発で新規が全面拒否される状態を回避するため、
  `quant-order-manager` 自身の `ORDER_MANAGER_SERVICE_ENABLED=0` に戻し、
  service内の自己HTTP再入を停止。
  `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_PF/WIN` を `0.58/0.27` へ更新。
  併せて `SCALP_PING_5S_B_PERF_GUARD_SL_LOSS_RATE_MAX` を `0.75` へ更新し、
  `sl_loss_rate` 理由での hard stop 常態化を回避。
  さらに低確率帯のスケールゼロ拒否を減らすため、
  `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER` を `B=0.48`, `C=0.62` へ緩和。
  併せて `strategy_entry` 側の forecast 逆行一律拒否を解除するため、
  `STRATEGY_FORECAST_FUSION_STRONG/WEAK_CONTRA_REJECT_ENABLED=0` を適用。
  加えて `SCALP_PING_5S_B/C_ENTRY_LEADING_PROFILE_ENABLED=0` とし、
  strategy_entry の二重拒否（leading profile 起点）を無効化。
  - 反映先: `ops/env/quant-order-manager.env`, `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`, `ops/env/quant-v2-runtime.env`
- 方向固定での機会損失を減らすため、`scalp_ping_5s_b/c` の `SIDE_FILTER=buy` を解除。
  - 反映先: `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
- 目的:
  - 時間帯停止なし・恒久停止なしのまま、`reduce` 運用で約定回復と方向適応を両立する。

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

### manual margin guard（2026-02-26 no-stop プロファイル）
- 適用箇所: `execution/order_manager.py` preflight（manual exposure 併走時の新規抑制）
- 有効化: `ORDER_MANUAL_MARGIN_GUARD_ENABLED=1`（維持）
- 反映値（`ops/env/quant-v2-runtime.env` と `ops/env/quant-order-manager.env` を同値同期）:
  - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO=0.00`
  - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER=0.00`
  - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY=0`
- 目的:
  - no-stop運用で `manual_margin_pressure` の連続 reject を抑制し、strategy 側の通過を優先する。
  - 強制ブロックは `margin_usage_projected_cap` / `order_margin_block` 系に集約して破綻側を防ぐ。

### no-stop 配分再調整（2026-02-26）
- 目的: サービス停止なしで、負け寄与戦略のサイズを即圧縮し、勝ち寄与戦略へ配分を寄せる。
- 実装:
  - `scalp_ping_5s_b/c` と `M1Scalper` は `base/max units` と同時保有・発注頻度を縮小。
  - `scalp_ping_5s_b` は縮小後の通過維持のため `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=20` を適用。
  - `scalp_ping_5s_b` / `M1Scalper` は `hard:failfast` 固着を避けるため、
    no-stop 向けに `PERF_GUARD_FAILFAST_*` を soft 警告側へ再調整。
  - それでも `perf_block` が連発したため、B/M1 は `PERF_GUARD_ENABLED=0`（prefix: `SCALP_PING_5S_B`, `M1SCALP`）で hard reject ループを解除。
  - 同値は `quant-v2-runtime.env` にも置き、worker local `order_manager` 経路へ確実に反映する。
  - 24h 実損で B/C とも long 側の負け寄与が卓越したため、
    `SCALP_PING_5S_B_SIDE_FILTER=sell`, `SCALP_PING_5S_C_SIDE_FILTER=sell` を適用して方向逆風を遮断。
  - `slo_block(data_lag_p95_exceeded)` が連続したため、
    `ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS=9000` へ緩和して遅延スパイク時の過剰拒否を抑制。
  - `M1Scalper` は 直近で損失勾配が縮小したため、`M1SCALP_BASE_UNITS=4500` へ増量して寄与を引き上げ。
  - 24h 実損の下位（B/C）は `BASE/MAX_UNITS` を圧縮し、上位（MomentumBurst/MicroVWAPRevert）は
    `MICRO_MULTI_BASE_UNITS` を増量して利益寄与側へ配分を寄せる。
  - `MicroVWAPRevert` は `MICRO_MULTI_LOOP_INTERVAL_SEC=4.0` へ短縮して、機会検知を増やす。
  - `MicroVWAPRevert` は breakout 発火閾値（`MIN_ADX`, `MIN_RANGE_SCORE`, `MIN_ATR`）を
    MomentumBurst 相当まで緩和し、約定頻度を引き上げる。
  - `TickImbalance` は `SCALP_PRECISION_COOLDOWN_SEC=120` に短縮し、連続機会の取りこぼしを減らす。
  - `MicroRangeBreak` / `MomentumBurstMicro` は `MICRO_MULTI_BASE_UNITS` を増量。
  - 同2戦略の breakout 発火閾値を緩和（`MIN_ADX`, `MIN_RANGE_SCORE`, `MIN_ATR`）し、
    `LOOP_INTERVAL_SEC=3.0` で検知頻度を引き上げる。
  - `RangeFader` は `ORDER_MIN_UNITS_STRATEGY_SCALP_RANGEFAD=300` を追加し、
    `entry_probability_below_min_units` での連続 reject を抑制。
- 監視指標:
  - `orders.db`: `entry_probability_reject`（rangefader の内訳）と `filled` 増減。
  - `trades.db`: `realized_pl` の strategy 別増分（B/C/M1 の下振れ勾配、MicroRangeBreak/MomentumBurst の上振れ確認）。

### no-stop short発火補正（2026-02-26 追記）
- 背景:
  - `SIDE_FILTER=sell` 適用後、longは遮断できたが `revert_not_found` と `short units_below_min` が支配的となり B/C の約定が枯渇。
- 変更点（B/C 共通方針）:
  - short最小通過ロットを引き下げ（`SCALP_PING_5S_{B,C}_MIN_UNITS=5`、`ORDER_MIN_UNITS_STRATEGY_* = 5`）。
  - `revert` 判定を緩和（`REVERT_RANGE/SWEEP/BOUNCE/CONFIRM_RATIO` 緩和、`REVERT_SHORT_WINDOW_SEC=1.20`）。
  - long→short 変換の導線を有効化（`EXTREMA_GATE_ENABLED=1`、`EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=1`）。
- C 固有の補正:
  - `BASE_ENTRY_UNITS=180` へ増量し、縮小後でも short が最小ロットを割り込みにくい設計へ変更。
  - `SHORT_MIN_TICKS/SHORT_MIN_SIGNAL_TICKS=3`、`FAST_DIRECTION_FLIP_*` 緩和、`SIDE_METRICS_DIRECTION_FLIP_ENABLED=1`。
  - extrema short側ブロックの過剰抑制を緩和（`SHORT_BOTTOM_BLOCK_POS/SOFT_POS`, `TECH_FILTER_SHORT_BOTTOM_RSI_MIN`）。
- 期待効果:
  - long遮断を維持したまま short 約定密度を回復し、`orders.db filled` を再増加させる。

### no-stop 無約定化の通過率補正（2026-02-26 追記）
- 背景:
  - `orders.db` で直近30分の発注イベントが枯渇し、B/C は local 判定で `revert_not_found` と `units_below_min` が残存。
  - 共通 `POLICY_HEURISTIC_PERF_BLOCK_ENABLED=1` により、他戦略側も hard reject が再燃しやすい状態だった。
- 実装:
  - `ops/env/quant-v2-runtime.env`
    - `POLICY_HEURISTIC_PERF_BLOCK_ENABLED=0`
  - `ops/env/scalp_ping_5s_b.env`
    - `MIN_UNITS=1`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=1`
    - `SHORT_MOMENTUM_TRIGGER_PIPS=0.08`
    - `MOMENTUM_TRIGGER_PIPS=0.08`
    - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.58`
    - `SIDE_BIAS_SCALE_GAIN/FLOOR=0.35/0.28`
  - `ops/env/scalp_ping_5s_c.env`
    - `SIDE_FILTER=none`（片側固定を解除）
    - `ALLOW_NO_SIDE_FILTER=1`
    - `MIN_UNITS=1`
    - `MAX_ORDERS_PER_MINUTE=3`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=1`
    - `SHORT/LONG_MOMENTUM_TRIGGER_PIPS=0.08/0.18`
    - `MOMENTUM_TRIGGER_PIPS=0.08`
    - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.62`
    - `SIDE_BIAS_SCALE_GAIN/FLOOR=0.35/0.28`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN/FLOOR=0.68/0.58`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER=0.38`
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.52`
    - `PERF_GUARD_FAILFAST_MIN_TRADES=30`（C prefix / base prefix 両方）
    - `PERF_GUARD_FAILFAST_PF/WIN=0.20/0.20`（C prefix / base prefix 両方）
    - `PERF_GUARD_ENABLED=0`（C prefix / base prefix 両方）
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.46`
  - `ops/env/quant-order-manager.env` 同期:
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.46`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN/MAX_SCALE_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.40/0.85`
    - `ORDER_MANAGER_PRESERVE_INTENT_BOOST_PROBABILITY_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.85`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=1`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_EXTREMA_REVERSAL(_LIVE)=30`
    - `SCALP_PING_5S(_C)_PERF_GUARD_MODE=off`, `SCALP_PING_5S(_C)_PERF_GUARD_ENABLED=0`
  - `ops/env/quant-v2-runtime.env` 同期:
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_EXTREMA_REVERSAL(_LIVE)=30`
- 運用意図:
  - 時間帯停止・恒久停止を使わず、発注通路を再開するための最小通過ユニットとシグナル閾値を補正。
  - hard block を減らしつつ、単位当たりリスクは `MIN_UNITS`/bias scale で抑えたまま再稼働させる。

### `strategy_guard` の DBロック耐性（2026-02-26 追記）
- 背景:
  - `market_order` preflight 後に `database is locked` で落ちるケースが発生し、
    実約定まで進めない事象を確認。
  - 主要経路は `execution/strategy_guard.py` の `stage_state.db` 参照（`strategy_cooldown` 判定）。
- 実装:
  - `STRATEGY_GUARD_DB_BUSY_TIMEOUT_MS` / lock retry を追加。
  - 接続PRAGMAを `WAL` + `busy_timeout` に設定し、接続を autocommit 化。
  - `set_block` / `is_blocked` / `clear_expired` で lock時は fail-open して
    order-manager 上位へ `OperationalError` を伝播しない。
- 運用意図:
  - `stage_state.db` 共有アクセス競合を理由にエントリー導線を止めない。
  - 停止ではなく改善優先方針を維持しつつ、発注可否は戦略ローカル + リスクガードで判定する。

### B/C no-signal 緩和（2026-02-26 追記）
- 背景:
  - lock障害解消後、B/C は `revert_not_found` / `units_below_min` で entry が詰まりやすい状態を確認。
- 実装:
  - B/C とも `SIDE_FILTER=none` へ変更（C は `ALLOW_NO_SIDE_FILTER=1`）。
  - B/C とも `REVERT_ENABLED=0` で revert 依存を外し、momentum主導の通過を優先。
  - `MAX_ORDERS_PER_MINUTE` / `BASE_ENTRY_UNITS` を引き上げ、`MIN_UNITS_RESCUE` の閾値を緩和。
  - `CONF_FLOOR` を下げ、低頻度化しすぎる条件を緩和。
- 運用意図:
  - 時間帯停止なしで発注密度を回復し、実取引ログで改善を再評価できる状態を作る。
  - 主要ブロックは `order_manager` 側の margin/probability/risk guard に集約して制御する。

### B/C `SIDE_FILTER` wrapper 正規化（2026-02-26 追記）
- 背景:
  - `ops/env/scalp_ping_5s_b.env` / `ops/env/scalp_ping_5s_c.env` で
    `SIDE_FILTER=none` を指定しても、wrapper の fail-closed により `sell` へ上書きされ、
    `ALLOW_NO_SIDE_FILTER=1` が実効していなかった。
- 実装:
  - `workers/scalp_ping_5s_b/worker.py`
  - `workers/scalp_ping_5s_c/worker.py`
    - `ALLOW_NO_SIDE_FILTER=1` かつ `SIDE_FILTER in {"", "none", "off", "disabled"}` の場合は
      `SCALP_PING_5S_SIDE_FILTER=""` を許可。
    - それ以外の未設定/不正値は従来どおり `sell` へ fail-closed。
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1` を追加。
- 運用意図:
  - 無効化したい時だけ明示 opt-in で side filter を外し、
    デフォルトは fail-closed を維持して意図しない方向ドリフトを防ぐ。

### B `profit-first` 即効デリスク（2026-02-27 追記）
- 背景:
  - VM 実測で直近15分の全体損益はプラスでも、自動戦略のみではマイナスが残存。
  - `scalp_ping_5s_b_live` の long 側で stop 系損失が先行し、当日損失寄与の主因となっていた。
- 実装（`ops/env/scalp_ping_5s_b.env`）:
  - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=720`（from `900`）
  - `SCALP_PING_5S_B_CONF_FLOOR=75`（from `72`）
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.74`（from `0.70`）
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR=0.60`（from `0.54`）
- 運用意図:
  - 停止なし方針を維持したまま、B の低品質エントリーを抑制し損失勾配を圧縮する。
  - 低品質シグナルの通過率を下げ、`filled` 継続下で1トレード期待値の改善を狙う。

### 勝ち筋寄せ再配分（2026-02-27 追記）
- 背景:
  - 24h 実績で `WickReversalBlend` が唯一のプラス寄与、B/C が大幅マイナス寄与だった。
  - 直近15分の自動損益もマイナスが継続し、勝ち戦略の寄与不足が明確だった。
- 実装:
  - `ops/env/quant-scalp-wick-reversal-blend.env`
    - `SCALP_PRECISION_UNIT_BASE_UNITS=9500`（from `7000`）
    - `SCALP_PRECISION_UNIT_CAP_MAX=0.65`（from `0.55`）
    - `SCALP_PRECISION_COOLDOWN_SEC=8`（from `12`）
    - `SCALP_PRECISION_MAX_OPEN_TRADES=2`（from `1`）
    - `WICK_BLEND_RANGE_SCORE_MIN=0.40`（from `0.45`）
    - `WICK_BLEND_ADX_MIN/MAX=14.0/28.0`（from `16.0/24.0`）
    - `WICK_BLEND_BB_TOUCH_RATIO=0.18`（from `0.22`）
    - `WICK_BLEND_TICK_MIN_STRENGTH=0.30`（from `0.40`）
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=600`（from `720`）
    - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=5`（from `6`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=220`（from `260`）
    - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=5`（from `6`）
- 運用意図:
  - 停止なしで運転を維持しつつ、利益寄与戦略へ配分を寄せる。
  - B/C は「継続稼働」前提のまま、サイズと頻度の上限を圧縮して損失勾配を下げる。

### `StageTracker` ロック耐性（2026-02-27 追記）
- 背景:
  - `quant-scalp-wick-reversal-blend` が起動時に
    `execution/stage_tracker.py` の DB初期化で `database is locked` となり停止した。
- 実装:
  - `execution/stage_tracker.py`
    - `STAGE_DB_BUSY_TIMEOUT_MS` / `STAGE_DB_LOCK_RETRY` /
      `STAGE_DB_LOCK_RETRY_SLEEP_SEC` を追加。
    - `stage_state.db` 接続で `busy_timeout` / `journal_mode=WAL` を適用。
    - schema作成を lock retry 付き実行へ変更し、起動時ロックを吸収。
- 運用意図:
  - 勝ち筋戦略の起動失敗を防ぎ、停止なし運用での導線断をなくす。

### `StageTracker` 起動DDL最小化（2026-02-27 追記）
- 背景:
  - lock retry 追加後も `StageTracker.__init__` の `CREATE TABLE IF NOT EXISTS` 実行時に
    `database is locked` が再発した。
  - 共有 `stage_state.db` に対して「既存テーブルでも毎回DDL」を投げる設計が、
    起動時のスキーマロック競合を残していた。
- 実装:
  - `execution/stage_tracker.py`
    - `_table_exists` / `_column_exists` を追加。
    - `_ensure_table` / `_ensure_column` を追加し、
      テーブル/列が不足している時のみ DDL を実行。
    - 既存スキーマ環境では起動時DDLを書き込みしない導線へ変更。
- 運用意図:
  - 共有DBロックの瞬間風速で勝ち寄与ワーカーが停止する経路を削減する。

### `quant-order-manager` service worker 増強（2026-02-27 追記）
- 背景:
  - strategy worker 側で `order_manager service call failed ... Read timed out (20s)` が発生し、
    service応答待ち起点で entry/close の遅延が出ていた。
  - `quant-order-manager` は稼働中だったが、`ORDER_MANAGER_SERVICE_WORKERS=1` の単一並列だった。
- 実装:
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_SERVICE_WORKERS=4`（from `1`, staged `2 -> 4`）
- 運用意図:
  - order_manager の同時処理能力を上げ、localhost API timeout を減らす。
  - timeout由来の重複リクエストと reject ノイズを抑制する。

### WickBlend `StageTracker` fail-open（2026-02-27 追記）
- 背景:
  - `StageTracker` 初期化時の `database is locked` が再発し、
    `quant-scalp-wick-reversal-blend` がプロセス終了していた。
- 実装:
  - `workers/scalp_wick_reversal_blend/worker.py`
    - `StageTracker()` を `try/except` で包み、初期化失敗時は `_NoopStageTracker` を使用。
- 運用意図:
  - shared DBロックの瞬間的な競合で、勝ち寄与 worker が停止し続ける状態を防ぐ。

### `position_manager sync_trades` runtime tuning（2026-02-27 追記）
- 背景:
  - `quant-position-manager` で `sync_trades timeout` と `position manager busy` が高頻度発生し、
    strategy 側のポジ同期呼び出しが不安定化していた。
- 実装:
  - `ops/env/quant-v2-runtime.env`
    - `POSITION_MANAGER_MAX_FETCH=600`
    - `POSITION_MANAGER_SYNC_MIN_INTERVAL_SEC=4.0`
    - `POSITION_MANAGER_SYNC_CACHE_WINDOW_SEC=4.0`
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_TIMEOUT_SEC=12.0`
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_CACHE_TTL_SEC=3.0`
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC=120.0`
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_MAX_FETCH=600`
- 運用意図:
  - sync負荷を平準化し、position manager 起点の timeout 連鎖を抑制する。

### B/C 追加圧縮（2026-02-27 追記）
- 背景:
  - 直近窓で `scalp_ping_5s_b_live` / `scalp_ping_5s_c_live` の負け寄与が継続。
- 実装:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=450`（from `600`）
    - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=4`（from `5`）
  - `ops/env/scalp_ping_5s_c.env`
    - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=170`（from `220`）
    - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=4`（from `5`）
- 運用意図:
  - 停止なしで稼働を維持しつつ、B/C の損失勾配を追加で低減する。

### duplicate CID 回収 + order-manager RPC詰まり緩和（2026-02-27 追記）
- 背景:
  - `order_manager service call failed ... Read timed out (20.0)` 後に
    同一 `client_order_id` 再送が発生し、`CLIENT_TRADE_ID_ALREADY_EXISTS` が連鎖。
  - `orders.db` では同一CIDに `filled` と `rejected` が並ぶケースを確認。
- 実装:
  - `execution/order_manager.py`
    - `CLIENT_TRADE_ID_ALREADY_EXISTS` 発生時に同一CIDの最新 `filled` を逆引きし、
      `trade_id` を回収できた場合は `status=duplicate_recovered` として成功返却。
    - service timeout 後の local fallback 前に、同一CIDの `orders.db` 状態を
      最大10秒ポーリングし、終端状態を先に回収して二重送信を抑止。
  - `ops/env/quant-v2-runtime.env`
    - `ORDER_MANAGER_SERVICE_TIMEOUT=45.0`（from `8.0`）
    - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_WAIT_SEC=10.0`
    - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_POLL_SEC=0.5`
  - `ops/env/quant-order-manager.env`
    - `ORDER_MANAGER_SERVICE_WORKERS=6`（from `4`）
- 運用意図:
  - timeout起点の重複CID rejectを減らし、既存約定の取りこぼしを抑制する。

### 状態遷移

| 状態 | 遷移条件 | 動作 |
|------|----------|------|
| `NORMAL` | 初期 | 全 pocket 取引許可 |
| `EVENT_LOCK` | 経済指標 ±30min | `micro` 新規停止、建玉縮小ロジック発動 |
| `MICRO_STOP` | `micro` pocket DD ≥5% | `micro` 決済のみ、`macro` 継続 |
| `GLOBAL_STOP` | Global DD ≥20% または `healthbeat` 欠損>10min | 全取引停止、プロセス終了 |
| `RECOVERY` | DD が閾値の 80% 未満、24h 経過 | 新規建玉再開前に `main.py` ドライラン |

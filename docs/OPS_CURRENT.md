# Ops Current (2026-02-11 JST)

## 0-2. 2026-02-19 UTC `scalp_macd_rsi_div_b_live` 精度優先チューニング
- 背景:
  - VM `trades.db`（UTC 2026-02-18 02:22〜2026-02-19 01:33）で
    `scalp_macd_rsi_div_b_live` が `4 trades / PF=0.046 / -32.9 pips`。
  - `tick_entry_validate`（ticket `365759`）で
    `TP_touch<=600s = 0/1` を確認し、逆行局面での誤発火を優先是正。
- 対応（`ops/env/quant-scalp-macd-rsi-div-b.env`）:
  - レンジ限定:
    - `SCALP_MACD_RSI_DIV_B_REQUIRE_RANGE_ACTIVE=1`
    - `SCALP_MACD_RSI_DIV_B_RANGE_MIN_SCORE=0.35`
    - `SCALP_MACD_RSI_DIV_B_MAX_ADX=30`
  - シグナル品質の引き締め:
    - `SCALP_MACD_RSI_DIV_B_MIN_DIV_SCORE=0.08`
    - `SCALP_MACD_RSI_DIV_B_MIN_DIV_STRENGTH=0.12`
    - `SCALP_MACD_RSI_DIV_B_MAX_DIV_AGE_BARS=24`
    - `SCALP_MACD_RSI_DIV_B_RSI_LONG_ENTRY=36`
    - `SCALP_MACD_RSI_DIV_B_RSI_SHORT_ENTRY=62`
  - エクスポージャ抑制:
    - `SCALP_MACD_RSI_DIV_B_MAX_OPEN_TRADES=1`
    - `SCALP_MACD_RSI_DIV_B_COOLDOWN_SEC=45`
    - `SCALP_MACD_RSI_DIV_B_BASE_ENTRY_UNITS=5000`
  - fail-open 経路を停止:
    - `SCALP_MACD_RSI_DIV_B_TECH_FAILOPEN=0`

## 0-1. 2026-02-18 UTC MicroCompressionRevert デリスク（専用調整）
- 背景:
  - 直近24hで `MicroCompressionRevert-short` が `PF<1`（特に同時多発エントリー後の損失クラスター）を確認。
- 対応（`ops/env/quant-micro-compressionrevert.env`）:
  - サイズ縮小:
    - `MICRO_MULTI_BASE_UNITS=14000`（従来 28000）
    - `MICRO_MULTI_STRATEGY_UNITS_MULT=MicroCompressionRevert:0.45`
  - 同時多発抑制:
    - `MICRO_MULTI_MAX_SIGNALS_PER_CYCLE=1`
    - `MICRO_MULTI_MULTI_SIGNAL_MIN_SCALE=0.55`
    - `MICRO_MULTI_STRATEGY_COOLDOWN_SEC=120`
  - 低成績ブロックを前倒し:
    - `MICRO_MULTI_HIST_MIN_TRADES=8`
    - `MICRO_MULTI_HIST_SKIP_SCORE=0.55`
    - `MICRO_MULTI_DYN_ALLOC_MIN_TRADES=8`
    - `MICRO_MULTI_DYN_ALLOC_LOSER_SCORE=0.45`
- EXIT調整（`config/strategy_exit_protections.yaml` `MicroCompressionRevert.exit_profile`）:
  - `range_max_hold_sec=900`
  - `loss_cut_soft_sl_mult=0.95`
  - `loss_cut_hard_sl_mult=1.20`（従来 1.60）
  - `loss_cut_max_hold_sec=900`（従来 2400）
  - `loss_cut_cooldown_sec=4`
  - 追加: `profit/trail/lock` の明示値（`profit_pips=1.1`, `trail_start_pips=1.5` など）

## 0. 2026-02-17 UTC 5秒スキャをB専用へ固定
- 無印5秒スキャ（`scalp_ping_5s_live`）の運用導線を削除。
  - 削除: `quant-scalp-ping-5s.service`, `quant-scalp-ping-5s-exit.service`
  - 削除: `ops/env/quant-scalp-ping-5s.env`, `ops/env/quant-scalp-ping-5s-exit.env`, `ops/env/scalp_ping_5s.env`
- 5秒スキャは B版（`scalp_ping_5s_b_live`）のみ稼働。
  - 使用env: `ops/env/quant-scalp-ping-5s-b.env`, `ops/env/quant-scalp-ping-5s-b-exit.env`, `ops/env/scalp_ping_5s_b.env`
- 2026-02-17 UTC 追加: 現況追従のエントリー頻度回復チューニング
  - `scalp_ping_5s_b`
    - `SCALP_PING_5S_B_REVERT_MIN_TICKS=2`
    - `SCALP_PING_5S_B_REVERT_CONFIRM_TICKS=1`
    - `SCALP_PING_5S_B_SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW=1`
    - `SCALP_PING_5S_B_MIN_UNITS=150`
    - `SCALP_PING_5S_B_EXTREMA_REQUIRE_M1_M5_AGREE_SHORT=1`
    - `SCALP_PING_5S_B_EXTREMA_SHORT_BOTTOM_BLOCK_POS=0.10`
    - `SCALP_PING_5S_B_EXTREMA_SHORT_BOTTOM_SOFT_POS=0.18`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=150`
  - `scalp_ping_5s_flow`
    - `SCALP_PING_5S_FLOW_MIN_TICKS=4`
    - `SCALP_PING_5S_FLOW_MIN_SIGNAL_TICKS=3`
    - `SCALP_PING_5S_FLOW_MIN_TICK_RATE=0.50`
    - `SCALP_PING_5S_FLOW_SHORT_MIN_TICKS=3`
    - `SCALP_PING_5S_FLOW_SHORT_MIN_SIGNAL_TICKS=3`
    - `SCALP_PING_5S_FLOW_SHORT_MIN_TICK_RATE=0.50`
    - `SCALP_PING_5S_FLOW_IMBALANCE_MIN=0.50`
    - `SCALP_PING_5S_FLOW_MOMENTUM_TRIGGER_PIPS=0.10`
    - `SCALP_PING_5S_FLOW_SHORT_MOMENTUM_TRIGGER_PIPS=0.10`
    - `SCALP_PING_5S_FLOW_MOMENTUM_SPREAD_MULT=0.12`
    - `SCALP_PING_5S_FLOW_SIGNAL_WINDOW_FALLBACK_ALLOW_FULL_WINDOW=1`
    - `SCALP_PING_5S_FLOW_LOOKAHEAD_GATE_ENABLED=0`
    - `SCALP_PING_5S_FLOW_REVERT_WINDOW_SEC=1.60`
    - `SCALP_PING_5S_FLOW_REVERT_SHORT_WINDOW_SEC=0.55`
    - `SCALP_PING_5S_FLOW_REVERT_MIN_TICKS=2`
    - `SCALP_PING_5S_FLOW_REVERT_CONFIRM_TICKS=1`
    - `SCALP_PING_5S_FLOW_REVERT_MIN_TICK_RATE=0.50`
    - `SCALP_PING_5S_FLOW_REVERT_RANGE_MIN_PIPS=0.35`
    - `SCALP_PING_5S_FLOW_REVERT_SWEEP_MIN_PIPS=0.22`
    - `SCALP_PING_5S_FLOW_REVERT_BOUNCE_MIN_PIPS=0.08`
    - `SCALP_PING_5S_FLOW_REVERT_CONFIRM_RATIO_MIN=0.50`
    - `SCALP_PING_5S_FLOW_DROP_FLOW_MIN_PIPS=0.15`
    - `SCALP_PING_5S_FLOW_DROP_FLOW_MIN_TICKS=3`
- 2026-02-17 UTC 追加: 極値反転ルーティング（件数維持 + 方向補正）
  - `workers/scalp_ping_5s` に `EXTREMA_REVERSAL_*` を追加。
  - `SCALP_PING_5S_B` は既定で `EXTREMA_REVERSAL_ENABLED=1`。
  - `short_bottom_*` / `long_top_*` で反転根拠が揃う場合、
    block ではなく side 反転 (`*_extrev`) で注文継続。
  - `entry_thesis` 監査項目:
    - `extrema_reversal_applied`
    - `extrema_reversal_score`
- 2026-02-19 UTC 追加: ショート偏重クラスタ抑制（方向補正の非対称化）
  - `scalp_ping_5s_b`
    - `SCALP_PING_5S_B_EXTREMA_SHORT_BOTTOM_SOFT_UNITS_MULT=0.42`
    - `SCALP_PING_5S_B_EXTREMA_SHORT_BOTTOM_SOFT_BALANCED_UNITS_MULT=0.30`
    - `SCALP_PING_5S_B_EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=0`
    - `SCALP_PING_5S_B_EXTREMA_REVERSAL_LONG_TO_SHORT_MIN_SCORE=2.10`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_FORCE_STREAK=3`
- 2026-02-19 UTC 追加: flip過発火の緊急デリスク（方向上書き厳格化）
  - `scalp_ping_5s_b`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN=0.52`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_HORIZON_SCORE_MIN=0.32`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_HORIZON_AGREE_MIN=3`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_NEUTRAL_HORIZON_BIAS_SCORE_MIN=0.82`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_MOMENTUM_MIN_PIPS=0.18`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_CONFIDENCE_ADD=2`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_COOLDOWN_SEC=1.2`
    - `SCALP_PING_5S_B_FAST_DIRECTION_FLIP_REGIME_BLOCK_SCORE=0.60`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_LOOKBACK_TRADES=10`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_MIN_SIDE_SL_HITS=3`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_MIN_TARGET_MARKET_PLUS=2`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_FORCE_STREAK=5`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_DIRECTION_SCORE_MIN=0.55`
    - `SCALP_PING_5S_B_SL_STREAK_DIRECTION_FLIP_HORIZON_SCORE_MIN=0.42`
- 2026-02-19 UTC 追加: `quant-order-manager` 側の `perf_block` 是正（件数維持）
  - `ops/env/quant-order-manager.env`
    - `SCALP_PING_5S_B_PERF_GUARD_MODE=warn`
  - 目的:
    - `OPEN_REJECT note=perf_block:margin_closeout_n=...` を防ぎ、
      scalp_ping_5s_b の方向改善ロジックを発注欠落なく評価する。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b` 利伸ばし設定（exit最適化）
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_b` / `scalp_ping_5s_b_live` を個別exit_profile化
    - `profit_pips=2.0`
    - `trail_start_pips=2.3`
    - `trail_backoff_pips=0.95`
    - `lock_buffer_pips=0.70`
    - `lock_floor_min_hold_sec=45`
    - `range_profit_pips=1.6`
    - `range_trail_start_pips=2.0`
    - `range_trail_backoff_pips=0.80`
    - `range_lock_buffer_pips=0.55`
  - 目的:
    - `lock_floor` での早取り（平均 +0.6p）を減らし、
      `take_profit` 側での利伸ばし比率を上げる。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b` ロット逆転是正（確率スケール平坦化）
  - 観測（直近24h, VM）:
    - `win3` 平均ロット `620.6` に対し、`loss24` 平均ロット `864.7`
    - `ep>=0.90` は `avg_units=1464.4` なのに `avg_pips=-0.95`
  - 反映:
    - `ops/env/quant-order-manager.env`
      - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.45`
      - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.75`
      - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=1.00`
    - `ops/env/scalp_ping_5s_b.env`
      - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.75`
      - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=1.00`
  - 目的:
    - 高確率時の過大ロットを抑えつつ、低確率側の過小ロットを補正する。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b` 高確率遅行補正（方向整合prob再校正）
  - 背景（VM, 2026-02-18 17:00 JST 以降）:
    - `ep>=0.90`: `367件`, `avg_pips=-0.59`
    - ショート `ep>=0.90`: `123件`, `avg_pips=-1.07`, `SL率=63.4%`
    - 発注遅延は主要因でなく `preflight->fill ≒ 381ms`
  - 反映:
    - `workers/scalp_ping_5s/config.py`
      - `ENTRY_PROBABILITY_ALIGN_*`（direction/horizon/m1 加重、penalty/boost、units follow）
    - `workers/scalp_ping_5s/worker.py`
      - `entry_probability` を `confidence` 直結から整合再校正へ変更
      - `probability_units_mult` をサイズ計算に追加
      - `entry_thesis` に `entry_probability_raw` と `entry_probability_alignment.*` を追加
    - `ops/env/scalp_ping_5s_b.env`
      - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_*` を追加
  - 目的:
    - 高確率の過大評価による逆行ロット集中を抑え、
      エントリー頻度を大きく落とさずに損失振れ幅を下げる。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b_live` short反転撤退の高速化（サイド別EXIT）
  - 背景（VM, 2026-02-18 17:00 JST 以降）:
    - `short + MARKET_ORDER_TRADE_CLOSE`: `n=831`, `avg=-2.666p`, `avg_hold=625s`
    - `900s+` 保有の short MARKET close: `n=278`, `PL=-12,537.1 JPY`（主損失塊）
  - 反映:
    - `workers/scalp_ping_5s/exit_worker.py`
      - `non_range_max_hold_sec_<side>` と `direction_flip.<side>_*` オーバーライドを追加
    - `config/strategy_exit_protections.yaml`（`scalp_ping_5s_b_live`）
      - `non_range_max_hold_sec_short=300`
      - `direction_flip.short_*` を追加
        - `min_hold_sec=45`
        - `min_adverse_pips=1.0`
        - `score_threshold=0.56`
        - `release_threshold=0.42`
        - `confirm_hits=2`
        - `confirm_window_sec=18`
        - `de_risk_threshold=0.50`
        - `forecast_weight=0.45`
  - 目的:
    - short逆行を長時間抱える tail を圧縮し、反転時の微益/小損撤退を前倒しする。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b_live` 確率帯ロット再配分（逆配分是正）
  - 背景（VM, 2026-02-18 17:00 JST 以降）:
    - 高確率帯（`ep>=0.90`）の成績劣後に対してロットが重く、
      低確率帯（`ep<0.70`）の優位局面でロットが不足していた。
  - 反映:
    - `workers/scalp_ping_5s/config.py`
      - `ENTRY_PROBABILITY_BAND_ALLOC_*` を追加。
    - `workers/scalp_ping_5s/worker.py`
      - `trades.db` の帯別統計（`<0.70` / `>=0.90`）で
        `probability_band_units_mult` を算出し、最終ロットへ反映。
      - side別 `SL hit` と `MARKET_ORDER_TRADE_CLOSE +` 件数を `side_mult` に反映。
      - `entry_thesis` に `entry_probability_band_units_mult` と
        `entry_probability_band_allocation.*` を記録。
    - `ops/env/scalp_ping_5s_b.env`
      - `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_*` を追加。
  - 目的:
    - エントリー頻度は維持しつつ、負け筋への過大ロット集中を抑え、
      勝ち筋への配分を増やす。
- 2026-02-19 UTC 追加: `scalp_ping_5s_b_live` 利小損大是正（EXIT非対称の強化）
  - 背景（VM, 直近12h）:
    - 平均勝ち `+2.021p` に対し平均負け `-3.634p`。
    - `MARKET_ORDER_TRADE_CLOSE` の負け平均が `-6.343p` と大きく、
      特に short 側で長時間保有後の深いマイナスが残っていた。
  - 反映（`config/strategy_exit_protections.yaml`）:
    - `scalp_ping_5s_b` / `scalp_ping_5s_b_live` の exit profile を再調整
      - 利益側: `profit_pips/trail_start` を引き上げ（利を伸ばす）
      - 損失側: `loss_cut_hard_pips=6.0`, `loss_cut_hard_cap_pips=6.2`,
        `loss_cut_max_hold_sec=900` に圧縮
      - `non_range_max_hold_sec_short=240` を `b_live` にも明示
      - `direction_flip` の short 閾値を前倒し（早期 de-risk/close）
  - 目的:
    - 「勝ちをやや伸ばす」よりも先に「負けを深くしない」を強化し、
      1トレード当たりの損益非対称を改善する。

## 1. 2026-02-12 JST 追加チューニング（稼働戦略のみ）
- `TickImbalance` / `LevelReject` / `M1Scalper` だけを対象に EXIT の time-stop を短縮。
  - `TickImbalance`: `range_max_hold_sec=600`, `loss_cut_max_hold_sec=600`
  - `LevelReject`: `range_max_hold_sec=1200`, `loss_cut_max_hold_sec=1200`
  - `M1Scalper`: `range_max_hold_sec=300`, `loss_cut_max_hold_sec=300`
- `M1Scalper` は本番上書きでサイズを半減。
  - `config/vm_env_overrides_aggressive.env`: `M1SCALP_BASE_UNITS=7500`, `M1SCALP_EXIT_MAX_HOLD_SEC=300`
- `micro_multistrat` に戦略別サイズ倍率を追加。
  - 新規 env: `MICRO_MULTI_STRATEGY_UNITS_MULT`（例: `TickImbalance:0.70,LevelReject:0.70`）

## 2. 運用モード（2025-12 攻め設定）
- マージン活用を 85–92% 目安に引き上げ。
- ロット上限を拡大（`RISK_MAX_LOT` 既定 10.0 lot）。
- 手動ポジションを含めた総エクスポージャでガード。
- PF/勝率の悪い戦略は自動ブロック。
- 必要に応じて `PERF_GUARD_GLOBAL_ENABLED=0` で解除。
- 2026-02-09 以降、`env_prefix` を渡す worker の設定解決は「`<PREFIX>_UNIT_*` → `<PREFIX>_*` のみ」。グローバル `*` へのフォールバックは無効。

## 3. 2026-02-06 JST 時点の `fx-trader-vm` mask 済みユニット
```
quant-scalp-impulseretrace.service
quant-scalp-impulseretrace-exit.service
quant-scalp-multi.service
quant-scalp-multi-exit.service
quant-pullback-s5.service
quant-pullback-s5-exit.service
quant-pullback-runner-s5.service
quant-pullback-runner-s5-exit.service
quant-range-comp-break.service
quant-range-comp-break-exit.service
quant-scalp-reversal-nwave.service
quant-scalp-reversal-nwave-exit.service
quant-vol-spike-rider.service
quant-vol-spike-rider-exit.service
quant-tech-fusion.service
quant-tech-fusion-exit.service
quant-macro-tech-fusion.service
quant-macro-tech-fusion-exit.service
quant-micro-pullback-fib.service
quant-micro-pullback-fib-exit.service
quant-manual-swing.service
quant-manual-swing-exit.service
```

- 2026-02-06 JST: `quant-m1scalper*.service` は一度 VM からアンインストール（ユニット削除）。
- 2026-02-11 JST: `quant-m1scalper.service` は再導入され、`enabled/active` を確認。

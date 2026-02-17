# Ops Current (2026-02-11 JST)

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
    - `SCALP_PING_5S_B_MIN_UNITS=300`
    - `workers/scalp_ping_5s` は `ENV_PREFIX=SCALP_PING_5S_B` 時、
      `SCALP_PING_5S_EXTREMA_REQUIRE_M1_M5_AGREE_SHORT=true` を既定適用
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=150`
  - `scalp_ping_5s_flow`
    - `SCALP_PING_5S_FLOW_MIN_TICKS=3`
    - `SCALP_PING_5S_FLOW_MIN_SIGNAL_TICKS=2`
    - `SCALP_PING_5S_FLOW_MIN_TICK_RATE=0.35`
    - `SCALP_PING_5S_FLOW_DROP_FLOW_MIN_PIPS=0.20`
    - `SCALP_PING_5S_FLOW_DROP_FLOW_MIN_TICKS=4`

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

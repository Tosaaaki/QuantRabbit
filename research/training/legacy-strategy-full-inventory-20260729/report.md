# Legacy strategy full inventory

- Generated: 2026-07-29T03:02:02.848560+00:00
- Authority: NONE; live_permission=false
- Scope: local/Git/archive static read-only discovery
- Found: 82 normalized strategy families
- Evaluated: 4
- Unevaluated: 78
- Replay-ready unevaluated: 10
- Notion/Slack corpus: connector unavailable in this execution; not counted as searched

## Replay priority

- `failed_break_reverse`: score=165, runtime_evidence=8
- `pullback_continuation`: score=165, runtime_evidence=8
- `session_open`: score=165, runtime_evidence=8
- `trend_breakout`: score=165, runtime_evidence=8
- `impulse_break_s5`: score=125, runtime_evidence=0
- `impulse_momentum_s5`: score=125, runtime_evidence=0
- `impulse_retest_s5`: score=125, runtime_evidence=0
- `pullback_s5`: score=125, runtime_evidence=0
- `stop_run_reversal`: score=125, runtime_evidence=0
- `vwap_magnet_s5`: score=125, runtime_evidence=0

## GCP/VM trace recovery

- Instance identifiers: 4
- Disk/snapshot/image identifiers: 0
- systemd units: 218
- Worker-linked evidence families: 73
- Current GCP was not queried or changed. Secret values and command bodies are excluded.

## Normalized inventory

| strategy_id | aliases | runtime | result | replay | reproducibility |
|---|---:|---:|---|---|---|
| `basic` | 1 | 0 | unevaluated | no | evidence_only |
| `bb_rsi` | 2 | 0 | unevaluated | no | implementation_recoverable |
| `bb_rsi_fast` | 1 | 0 | unevaluated | no | implementation_recoverable |
| `compression_revert` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `donchian55` | 2 | 0 | unevaluated | no | implementation_recoverable |
| `failed_break_reverse` | 1 | 8 | unevaluated | yes | offline_replay_ready |
| `fast_scalp` | 1 | 0 | unevaluated | no | evidence_only |
| `h1_momentum` | 3 | 0 | unevaluated | no | implementation_recoverable |
| `impulse_break_s5` | 1 | 0 | unevaluated | yes | offline_replay_ready |
| `impulse_momentum_s5` | 1 | 0 | unevaluated | yes | offline_replay_ready |
| `impulse_retest_s5` | 1 | 0 | unevaluated | yes | offline_replay_ready |
| `impulse_retrace` | 2 | 0 | unevaluated | no | implementation_recoverable |
| `level_reactor` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `london_momentum` | 1 | 0 | unevaluated | no | evidence_only |
| `m1_scalper` | 2 | 0 | evaluated | no | implementation_recoverable |
| `ma_rsi_macd` | 1 | 0 | unevaluated | no | evidence_only |
| `macro_core` | 1 | 0 | unevaluated | no | evidence_only |
| `macro_tech_fusion` | 1 | 0 | unevaluated | no | evidence_only |
| `manual_spike` | 1 | 0 | unevaluated | no | evidence_only |
| `manual_swing` | 1 | 0 | unevaluated | no | evidence_only |
| `micro_adaptive_revert` | 1 | 0 | unevaluated | no | implementation_recoverable |
| `micro_core` | 1 | 0 | unevaluated | no | evidence_only |
| `micro_multistrat` | 1 | 0 | unevaluated | no | evidence_only |
| `micro_pullback_fib` | 1 | 0 | unevaluated | no | evidence_only |
| `micro_range_revert_lite` | 1 | 0 | unevaluated | no | evidence_only |
| `micro_vwap_revert` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `mirror_spike` | 1 | 0 | unevaluated | no | evidence_only |
| `mirror_spike_s5` | 1 | 0 | unevaluated | no | evidence_only |
| `mirror_spike_tight` | 1 | 0 | unevaluated | no | evidence_only |
| `mm_lite` | 1 | 0 | unevaluated | no | evidence_only |
| `momentum_burst` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `momentum_pulse` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `momentum_stack` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `mtf_breakout` | 1 | 0 | unevaluated | no | evidence_only |
| `onepip_maker_s1` | 1 | 0 | unevaluated | no | evidence_only |
| `pullback_continuation` | 1 | 8 | unevaluated | yes | offline_replay_ready |
| `pullback_ema` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `pullback_runner_s5` | 1 | 0 | unevaluated | no | evidence_only |
| `pullback_s5` | 1 | 0 | unevaluated | yes | offline_replay_ready |
| `pullback_scalp` | 1 | 0 | unevaluated | no | evidence_only |
| `pulse_break` | 3 | 8 | evaluated | no | implementation_recoverable |
| `range_bounce` | 1 | 0 | unevaluated | no | evidence_only |
| `range_break` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `range_compression_break` | 1 | 0 | unevaluated | no | evidence_only |
| `range_fader` | 2 | 8 | evaluated | no | implementation_recoverable |
| `range_revert_lite` | 1 | 0 | unevaluated | no | implementation_recoverable |
| `scalp_core` | 1 | 0 | unevaluated | no | evidence_only |
| `scalp_drought_revert` | 1 | 0 | unevaluated | no | implementation_recoverable |
| `scalp_extrema_reversal` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `scalp_false_break_fade` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `scalp_level_reject` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `scalp_macd_rsi_div` | 1 | 16 | unevaluated | no | implementation_recoverable |
| `scalp_macd_rsi_div_b` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `scalp_multistrat` | 1 | 0 | unevaluated | no | evidence_only |
| `scalp_ping_5s` | 1 | 32 | unevaluated | no | implementation_recoverable |
| `scalp_ping_5s_b` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `scalp_ping_5s_c` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `scalp_ping_5s_d` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `scalp_ping_5s_flow` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `scalp_precision` | 1 | 0 | unevaluated | no | evidence_only |
| `scalp_precision_lowvol` | 1 | 0 | unevaluated | no | implementation_recoverable |
| `scalp_reversal_nwave` | 1 | 0 | unevaluated | no | evidence_only |
| `scalp_tick_imbalance` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `scalp_vwap_revert` | 1 | 0 | unevaluated | no | implementation_recoverable |
| `scalp_wick_reversal_blend` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `scalp_wick_reversal_pro` | 1 | 8 | unevaluated | no | implementation_recoverable |
| `session_open` | 1 | 8 | unevaluated | yes | offline_replay_ready |
| `spike_reversal` | 1 | 0 | unevaluated | no | evidence_only |
| `squeeze_break_s5` | 1 | 0 | unevaluated | no | evidence_only |
| `stop_run_reversal` | 1 | 0 | unevaluated | yes | offline_replay_ready |
| `tech_fusion` | 1 | 0 | unevaluated | no | evidence_only |
| `trend_breakout` | 1 | 8 | unevaluated | yes | offline_replay_ready |
| `trend_ma` | 2 | 0 | evaluated | no | implementation_recoverable |
| `trend_momentum` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `trend_pullback` | 1 | 0 | unevaluated | no | evidence_only |
| `trend_reclaim` | 2 | 0 | unevaluated | no | implementation_recoverable |
| `trend_retest` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `vol_compression_break` | 1 | 0 | unevaluated | no | implementation_recoverable |
| `vol_spike_rider` | 1 | 0 | unevaluated | no | evidence_only |
| `vol_squeeze` | 1 | 0 | unevaluated | no | evidence_only |
| `vwap_bound_revert` | 2 | 8 | unevaluated | no | implementation_recoverable |
| `vwap_magnet_s5` | 1 | 0 | unevaluated | yes | offline_replay_ready |

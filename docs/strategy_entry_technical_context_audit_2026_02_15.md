# 戦略別 technical_context / evaluate_entry_techniques 監査（2026-02-15 更新）

- 対象: `workers/*/worker.py`（最終更新: 2026-02-15)
- `market_order/limit_order` 直呼び: 38
- そのうち戦略ワーカーによる直呼び: 37
- `order_manager` は実行経路側インフラであり戦略側監査対象外
- 戦略ワーカー直呼びのうち `evaluate_entry_techniques` あり: 37
- `technical_context_tfs/ticks/candle_counts` 3要素明示あり: 37

| 戦略（worker） | market_order/limit_order | evaluate_entry_techniques | technical_context_tfs | technical_context_ticks | technical_context_candle_counts | require_fib | require_median | require_nwave | require_candle | size_scale | size_min | size_max | tech_policy_locked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hedge_balancer | 1 | 1 | 1 | 1 | 1 |  |  |  |  |  |  |  |  |
| impulse_break_s5 | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| impulse_momentum_s5 | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| impulse_retest_s5 | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| london_momentum | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| macro_h1momentum | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| macro_tech_fusion | 1 | 1 | 1 | 1 | 1 | True | True | True | True | 0.2 | 0.7 | 1.25 |  |
| manual_swing | 1 | 1 | 1 | 1 | 1 |  |  |  |  |  |  |  |  |
| market_data_feed | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| micro_adaptive_revert | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| micro_bbrsi | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| micro_levelreactor | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| micro_momentumburst | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| micro_momentumstack | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| micro_multistrat | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| micro_pullback_fib | 1 | 1 | 1 | 1 | 1 | True | True | True | True | 0.18 | 0.6 | 1.25 |  |
| micro_pullbackema | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| micro_range_revert_lite | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| micro_rangebreak | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| micro_trendmomentum | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| micro_vwapbound | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| mm_lite | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| mtf_breakout | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| order_manager | 1 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| position_manager | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| pullback_runner_s5 | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| pullback_s5 | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| range_compression_break | 1 | 1 | 1 | 1 | 1 | False | False | True | False | 0.12 | 0.6 | 1.2 |  |
| scalp_impulseretrace | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| scalp_m1scalper | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| scalp_macd_rsi_div | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| scalp_multistrat | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| scalp_ping_5s | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| scalp_ping_5s_b | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| scalp_precision | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| scalp_pulsebreak | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| scalp_rangefader | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| scalp_reversal_nwave | 1 | 1 | 1 | 1 | 1 | False | False | True | True | 0.2 | 0.7 | 1.25 |  |
| scalp_squeeze_pulse_break | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| scalp_tick_imbalance | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| scalp_trend_reclaim | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| scalp_wick_reversal_blend | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| scalp_wick_reversal_pro | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| session_open | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| stop_run_reversal | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| strategy_control | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| tech_fusion | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.18 | 0.6 | 1.2 |  |
| trend_h1 | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| vol_spike_rider | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |
| vol_squeeze | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |  |  |  |
| vwap_magnet_s5 | 1 | 1 | 1 | 1 | 1 | False | False | False | False | 0.15 | 0.6 | 1.25 |  |

- 補足: `entry_thesis` の `technical_context_*` は市場エントリー実装の戦略側実装可否を優先集計。
- `macro_tech_fusion` / `micro_pullback_fib` / `range_compression_break` / `scalp_reversal_nwave` は、従来 `tech_tfs` 運用のままでも `technical_context_tfs` を明示化しました。

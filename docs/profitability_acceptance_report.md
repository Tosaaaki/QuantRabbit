# Profitability Acceptance

- Status: `PROFITABILITY_ACCEPTANCE_BLOCKED`
- Generated: `2026-06-26T06:40:25.577826+00:00`
- Findings: `10`

## Findings

| Priority | Code | Message |
| --- | --- | --- |
| `P0` | `SELF_IMPROVEMENT_P0_PRESENT` | self-improvement audit still has 1 P0 finding(s) |
| `P0` | `NEGATIVE_EXPECTANCY_ACTIVE` | capture economics is still NEGATIVE_EXPECTANCY |
| `P0` | `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE` | 1 TP-proven segment(s) are still net-damaged by MARKET_ORDER_TRADE_CLOSE leakage |
| `P0` | `RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK` | 1 loss-side gateway MARKET_ORDER_TRADE_CLOSE event(s) remain inside the 7-day acceptance window without contained-risk timing evidence |
| `P0` | `LOSS_CLOSE_GATE_EVIDENCE_MISSING` | 1 recent GPT loss-side market close(s) lack durable close_gate_evidence in verification_observations |
| `P0` | `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE` | 30-day OANDA candle replay says the current TP-progress repair improves loss-side closes, but the replayed loss-close P/L is still net negative |
| `P1` | `PROJECTION_HEADLINE_PRECISION_ECONOMIC_GAP` | 8 projection bucket(s) clear headline precision but fail economic precision after TIMEOUT/no-touch penalties |
| `P1` | `BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN` | S5 bid/ask replay has price truth for loaded samples, but pair-direction sample coverage is too thin to claim all-currency high-turn readiness |
| `P1` | `NO_LIVE_READY_TARGET_COVERAGE` | daily target is open but there are no LIVE_READY lanes |
| `P1` | `REPAIR_FRONTIER_BLOCKED` | 8 repair-mode candidate(s) exist, but none currently clear live gates |

## Metrics

```json
{
  "bidask_replay_rules": {
    "adoption_summary": {
      "has_live_grade_support": true,
      "has_rank_only_support": true,
      "live_grade_support_rules": 11,
      "negative_block_rules": 52,
      "rank_only_blocker_counts": {
        "DAILY_PNL_UNSTABLE": 2,
        "DAILY_SAMPLE_CONCENTRATED": 2,
        "NEEDS_HIGHER_POSITIVE_DAY_RATE": 2,
        "NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION": 2
      },
      "rank_only_support_rules": 4
    },
    "contrarian_edge_rules": 15,
    "daily_stability_requirements": {
      "max_daily_sample_share": 0.7,
      "min_active_days": 3,
      "min_positive_day_rate": 0.6666666666666666
    },
    "daily_stable_contrarian_edge_rules": 11,
    "daily_stable_edge_rules": 0,
    "daily_stable_support_rules": 11,
    "edge_rules": 0,
    "history_dirs": [
      "logs/replay/oanda_history/20260623T082438Z",
      "logs/replay/oanda_history/20260623T082500Z",
      "logs/replay/oanda_history/20260623T200805Z",
      "logs/replay/oanda_history/20260623T205017Z",
      "logs/replay/oanda_history/20260623T225355Z",
      "logs/replay/oanda_history/20260624T005035Z",
      "logs/replay/oanda_history/20260624T010246Z",
      "logs/replay/oanda_history/20260624T011450Z",
      "logs/replay/oanda_history/20260624T012845Z"
    ],
    "history_fetch_command": "python3 scripts/oanda_history_fetch.py --pairs AUD_USD,EUR_CAD,NZD_CAD --granularities S5 --price BA --days 120 --output-dir logs/replay/oanda_history",
    "negative_rules": 52,
    "path": "/Users/tossaki/App/QuantRabbit/src/quant_rabbit/bidask_replay_precision_rules.json",
    "price_truth_coverage": {
      "adoption_level": "PAIR_LOCAL_RANK_ONLY",
      "all_currency_sample_coverage_status": "UNDER_SAMPLED",
      "evaluated_rows": 39680,
      "global_currency_validation_blocked": true,
      "history_fetch_command": null,
      "history_fetch_command_count": 0,
      "history_fetch_command_mode": "WINDOWED",
      "missing_pair_directions": [],
      "missing_pairs": [],
      "missing_price_truth_samples": 0,
      "missing_price_window_group_count": 0,
      "status": "PRICE_TRUTH_OK",
      "under_sampled_missing_evaluated_samples": 0,
      "under_sampled_pair_direction_count": 52,
      "under_sampled_pair_directions": [
        "AUD_CAD:DOWN",
        "AUD_CAD:UP",
        "AUD_CHF:UP",
        "AUD_JPY:DOWN",
        "AUD_JPY:UP",
        "AUD_NZD:DOWN",
        "AUD_NZD:UP",
        "AUD_USD:DOWN",
        "AUD_USD:UP",
        "CAD_CHF:UP",
        "CAD_JPY:DOWN",
        "CAD_JPY:UP",
        "CHF_JPY:DOWN",
        "CHF_JPY:UP",
        "EUR_AUD:DOWN",
        "EUR_AUD:UP",
        "EUR_CAD:DOWN",
        "EUR_CAD:UP",
        "EUR_CHF:DOWN",
        "EUR_CHF:UP",
        "EUR_GBP:DOWN",
        "EUR_GBP:UP",
        "EUR_JPY:DOWN",
        "EUR_JPY:UP"
      ],
      "warnings": [
        "FORECAST_ROWS_DURING_BROKER_NO_MARKET_WINDOW",
        "FORECAST_ROWS_WITH_PENDING_FUTURE_TRUTH_WINDOW"
      ]
    },
    "rank_only_contrarian_edge_rules": 4,
    "rank_only_edge_rules": 0,
    "rank_only_examples": [
      {
        "active_days": 3,
        "adoption_blockers": [
          "DAILY_SAMPLE_CONCENTRATED",
          "NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION"
        ],
        "adoption_status": "RANK_ONLY_NOT_DAILY_STABLE",
        "daily_stability_gap": {
          "current_max_daily_sample_share": 0.7467,
          "current_positive_day_rate": 0.6667,
          "max_allowed_daily_sample_share": 0.7,
          "missing_active_days": 0,
          "missing_positive_days_at_current_requirement": 0,
          "reasons": [
            "NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION"
          ],
          "required_active_days": 3,
          "required_positive_day_rate": 0.6667,
          "required_positive_days_at_current_requirement": 2,
          "status": "DAILY_SAMPLE_CONCENTRATED"
        },
        "daily_stability_status": "DAILY_SAMPLE_CONCENTRATED",
        "direction": "DOWN",
        "forecast_direction": "UP",
        "granularity": "S5",
        "max_daily_sample_share": 0.7467,
        "name": "AUD_USD_UP_H61_240m_FADE_TO_DOWN_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7",
        "optimized_profit_factor": 2.5897,
        "optimized_stop_loss_pips": 7.0,
        "optimized_take_profit_pips": 10.0,
        "optimized_win_rate": 0.72,
        "pair": "AUD_USD",
        "positive_day_rate": 0.6667,
        "positive_days": 2,
        "samples": 75
      },
      {
        "active_days": 3,
        "adoption_blockers": [
          "DAILY_PNL_UNSTABLE",
          "NEEDS_HIGHER_POSITIVE_DAY_RATE"
        ],
        "adoption_status": "RANK_ONLY_NOT_DAILY_STABLE",
        "daily_stability_gap": {
          "current_max_daily_sample_share": 0.6164,
          "current_positive_day_rate": 0.3333,
          "max_allowed_daily_sample_share": 0.7,
          "missing_active_days": 0,
          "missing_positive_days_at_current_requirement": 1,
          "reasons": [
            "NEEDS_HIGHER_POSITIVE_DAY_RATE"
          ],
          "required_active_days": 3,
          "required_positive_day_rate": 0.6667,
          "required_positive_days_at_current_requirement": 2,
          "status": "DAILY_PNL_UNSTABLE"
        },
        "daily_stability_status": "DAILY_PNL_UNSTABLE",
        "direction": "UP",
        "forecast_direction": "DOWN",
        "granularity": "S5",
        "max_daily_sample_share": 0.6164,
        "name": "AUD_USD_DOWN_H61_240m_CLT0p50_FADE_TO_UP_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7",
        "optimized_profit_factor": 2.1414,
        "optimized_stop_loss_pips": 7.0,
        "optimized_take_profit_pips": 10.0,
        "optimized_win_rate": 0.5753,
        "pair": "AUD_USD",
        "positive_day_rate": 0.3333,
        "positive_days": 1,
        "samples": 73
      },
      {
        "active_days": 4,
        "adoption_blockers": [
          "DAILY_SAMPLE_CONCENTRATED",
          "NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION"
        ],
        "adoption_status": "RANK_ONLY_NOT_DAILY_STABLE",
        "daily_stability_gap": {
          "current_max_daily_sample_share": 0.7179,
          "current_positive_day_rate": 0.75,
          "max_allowed_daily_sample_share": 0.7,
          "missing_active_days": 0,
          "missing_positive_days_at_current_requirement": 0,
          "reasons": [
            "NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION"
          ],
          "required_active_days": 3,
          "required_positive_day_rate": 0.6667,
          "required_positive_days_at_current_requirement": 3,
          "status": "DAILY_SAMPLE_CONCENTRATED"
        },
        "daily_stability_status": "DAILY_SAMPLE_CONCENTRATED",
        "direction": "DOWN",
        "forecast_direction": "UP",
        "granularity": "S5",
        "max_daily_sample_share": 0.7179,
        "name": "NZD_CAD_UP_H61_240m_CLT0p50_FADE_TO_DOWN_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7",
        "optimized_profit_factor": 1.7857,
        "optimized_stop_loss_pips": 7.0,
        "optimized_take_profit_pips": 10.0,
        "optimized_win_rate": 0.5641,
        "pair": "NZD_CAD",
        "positive_day_rate": 0.75,
        "positive_days": 3,
        "samples": 39
      },
      {
        "active_days": 5,
        "adoption_blockers": [
          "DAILY_PNL_UNSTABLE",
          "NEEDS_HIGHER_POSITIVE_DAY_RATE"
        ],
        "adoption_status": "RANK_ONLY_NOT_DAILY_STABLE",
        "daily_stability_gap": {
          "current_max_daily_sample_share": 0.3333,
          "current_positive_day_rate": 0.6,
          "max_allowed_daily_sample_share": 0.7,
          "missing_active_days": 0,
          "missing_positive_days_at_current_requirement": 1,
          "reasons": [
            "NEEDS_HIGHER_POSITIVE_DAY_RATE"
          ],
          "required_active_days": 3,
          "required_positive_day_rate": 0.6667,
          "required_positive_days_at_current_requirement": 4,
          "status": "DAILY_PNL_UNSTABLE"
        },
        "daily_stability_status": "DAILY_PNL_UNSTABLE",
        "direction": "UP",
        "forecast_direction": "DOWN",
        "granularity": "S5",
        "max_daily_sample_share": 0.3333,
        "name": "EUR_CAD_DOWN_H61_240m_CLT0p50_FADE_TO_UP_S5_BIDASK_CONTRARIAN_HARVEST_TP5_SL7",
        "optimized_profit_factor": 1.6071,
        "optimized_stop_loss_pips": 7.0,
        "optimized_take_profit_pips": 5.0,
        "optimized_win_rate": 0.6923,
        "pair": "EUR_CAD",
        "positive_day_rate": 0.6,
        "positive_days": 3,
        "samples": 39
      }
    ],
    "rank_only_support_rules": 4,
    "replay_validation_command": "python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --granularity S5 --history-dir logs/replay/oanda_history/20260623T082438Z --history-dir logs/replay/oanda_history/20260623T082500Z --history-dir logs/replay/oanda_history/20260623T200805Z --history-dir logs/replay/oanda_history/20260623T205017Z --history-dir logs/replay/oanda_history/20260623T225355Z --history-dir logs/replay/oanda_history/20260624T005035Z --history-dir logs/replay/oanda_history/20260624T010246Z --history-dir logs/replay/oanda_history/20260624T011450Z --history-dir logs/replay/oanda_history/20260624T012845Z --auto-history-min-days 30 --stable-min-active-days 3 --stable-max-daily-sample-share 0.7 --stable-min-positive-day-rate 0.6666666667",
    "support_rules": 15
  },
  "capture_economics": {
    "market_close": {
      "expectancy_jpy_per_trade": -768.7,
      "net_jpy": -74564.8,
      "trades": 97
    },
    "overall": {
      "expectancy_jpy_per_trade": -164.6,
      "net_jpy": -37031.0,
      "payoff_ratio": 0.396,
      "profit_factor": null,
      "trades": 225,
      "win_rate": 0.6044
    },
    "status": "NEGATIVE_EXPECTANCY",
    "take_profit": {
      "expectancy_jpy_per_trade": 508.4,
      "net_jpy": 48804.8,
      "trades": 96
    },
    "tp_proven_market_close_leak_segments": 1
  },
  "execution_ledger_close_leak": {
    "execution_timing_audit": {
      "generated_at_utc": "2026-06-26T06:22:19.531654+00:00",
      "label_counts": {
        "LOSS_CLOSE_CONTAINED_RISK": 19,
        "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE": 13
      },
      "loaded": true,
      "loss_close_actual_pl_jpy": -35276.4462,
      "loss_close_counterfactual_profit_capture_delta_jpy": 18724.2477,
      "loss_close_counterfactual_profit_capture_jpy": 3540.1531,
      "loss_close_counterfactual_profit_capture_pl_jpy": -16552.1985,
      "loss_close_repair_replay_block_reasons": {
        "BELOW_NOISE_FLOOR": 1
      },
      "loss_close_repair_replay_counterfactual_pl_jpy": -16505.9883,
      "loss_close_repair_replay_delta_jpy": 18770.4579,
      "loss_close_repair_replay_profit_capture_jpy": 3825.8424,
      "loss_closes_profit_capture_missed": 14,
      "loss_closes_repair_replay_triggered": 13,
      "loss_market_close_rows": 32,
      "path": "/Users/tossaki/App/QuantRabbit-live/data/execution_timing_audit.json",
      "post_repair_live_evidence_loss_closes_audited": 2,
      "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
      "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
      "pre_repair_historical_loss_closes_audited": 34,
      "pre_repair_historical_loss_closes_profit_capture_missed": 14,
      "pre_repair_historical_loss_closes_repair_replay_triggered": 13,
      "read_error": null,
      "repair_replay_contract": "TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1",
      "repair_replay_contract_present": true,
      "top_entry_quality_residual_groups": [
        {
          "actual_pl_jpy": -2981.8961,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -2981.8961,
              "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -2981.8961,
              "repair_replay_triggered": false,
              "trade_id": "472071"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "BREAKOUT_FAILURE",
          "pair": "GBP_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -2981.8961,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        },
        {
          "actual_pl_jpy": -2333.8215,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -2333.8215,
              "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -2333.8215,
              "repair_replay_triggered": false,
              "trade_id": "471817"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "EUR_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -2333.8215,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        },
        {
          "actual_pl_jpy": -2181.1565,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -2181.1565,
              "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -2181.1565,
              "repair_replay_triggered": false,
              "trade_id": "471711"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "EUR_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -2181.1565,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        },
        {
          "actual_pl_jpy": -2044.4543,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 2
          },
          "examples": [
            {
              "actual_pl_jpy": -548.9268,
              "lane_id": "range_trader:NZD_CAD:SHORT:RANGE_ROTATION:MARKET",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -548.9268,
              "repair_replay_triggered": false,
              "trade_id": "472380"
            },
            {
              "actual_pl_jpy": -1495.5275,
              "lane_id": "range_trader:NZD_CAD:SHORT:RANGE_ROTATION:MARKET",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1495.5275,
              "repair_replay_triggered": false,
              "trade_id": "472312"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 2,
          "method": "RANGE_ROTATION",
          "pair": "NZD_CAD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -2044.4543,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        },
        {
          "actual_pl_jpy": -1705.6738,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -1705.6738,
              "lane_id": "range_trader:AUD_USD:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1705.6738,
              "repair_replay_triggered": false,
              "trade_id": "472834"
            }
          ],
          "exit_reason": "STOP_LOSS_ORDER",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "AUD_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -1705.6738,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        },
        {
          "actual_pl_jpy": -1380.8008,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -1380.8008,
              "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1380.8008,
              "repair_replay_triggered": false,
              "trade_id": "472743"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "NZD_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -1380.8008,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        },
        {
          "actual_pl_jpy": -1272.0771,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 2
          },
          "examples": [
            {
              "actual_pl_jpy": -1019.7829,
              "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -1019.7829,
              "repair_replay_triggered": false,
              "trade_id": "472445"
            },
            {
              "actual_pl_jpy": -252.2942,
              "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -252.2942,
              "repair_replay_triggered": false,
              "trade_id": "472174"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 2,
          "method": "TREND_CONTINUATION",
          "pair": "EUR_CHF",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -1272.0771,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        },
        {
          "actual_pl_jpy": -1071.9,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -1071.9,
              "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1071.9,
              "repair_replay_triggered": false,
              "trade_id": "472094"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "EUR_JPY",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -1071.9,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        },
        {
          "actual_pl_jpy": -971.0121,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -971.0121,
              "lane_id": "range_trader:GBP_USD:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -971.0121,
              "repair_replay_triggered": false,
              "trade_id": "472837"
            }
          ],
          "exit_reason": "STOP_LOSS_ORDER",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "GBP_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -971.0121,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        },
        {
          "actual_pl_jpy": -955.691,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -955.691,
              "lane_id": "trend_trader:GBP_CHF:LONG:TREND_CONTINUATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -955.691,
              "repair_replay_triggered": false,
              "trade_id": "472190"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "TREND_CONTINUATION",
          "pair": "GBP_CHF",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -955.691,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        }
      ],
      "top_entry_quality_residual_method_rollups": [
        {
          "actual_pl_jpy": -12945.8682,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 4,
            "NO_PROFIT_CANDIDATE": 8
          },
          "examples": [
            {
              "actual_pl_jpy": -971.0121,
              "lane_id": "range_trader:GBP_USD:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "GBP_USD",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -971.0121,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472837"
            },
            {
              "actual_pl_jpy": -1705.6738,
              "lane_id": "range_trader:AUD_USD:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "AUD_USD",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1705.6738,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472834"
            },
            {
              "actual_pl_jpy": -280.8,
              "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "USD_JPY",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -280.8,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472775"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 9,
            "STOP_LOSS_ORDER": 3
          },
          "group_count": 11,
          "loss_closes": 12,
          "method": "RANGE_ROTATION",
          "pair_count": 9,
          "pairs": [
            "AUD_USD",
            "CHF_JPY",
            "EUR_JPY",
            "EUR_USD",
            "GBP_USD",
            "NZD_CAD",
            "NZD_CHF",
            "NZD_USD",
            "USD_JPY"
          ],
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -12945.8682,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side_count": 2,
          "sides": [
            "LONG",
            "SHORT"
          ]
        },
        {
          "actual_pl_jpy": -4081.1334,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 4
          },
          "examples": [
            {
              "actual_pl_jpy": -38.054,
              "lane_id": "failure_trader:EUR_CHF:LONG:BREAKOUT_FAILURE:LIMIT",
              "method": "BREAKOUT_FAILURE",
              "pair": "EUR_CHF",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -38.054,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472252"
            },
            {
              "actual_pl_jpy": -891.0833,
              "lane_id": "failure_trader:EUR_GBP:SHORT:BREAKOUT_FAILURE:MARKET",
              "method": "BREAKOUT_FAILURE",
              "pair": "EUR_GBP",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -891.0833,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472208"
            },
            {
              "actual_pl_jpy": -170.1,
              "lane_id": "failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE:LIMIT",
              "method": "BREAKOUT_FAILURE",
              "pair": "EUR_JPY",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -170.1,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472156"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 4
          },
          "group_count": 4,
          "loss_closes": 4,
          "method": "BREAKOUT_FAILURE",
          "pair_count": 4,
          "pairs": [
            "EUR_CHF",
            "EUR_GBP",
            "EUR_JPY",
            "GBP_USD"
          ],
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -4081.1334,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side_count": 2,
          "sides": [
            "LONG",
            "SHORT"
          ]
        },
        {
          "actual_pl_jpy": -3065.35,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 4,
            "NO_PROFIT_CANDIDATE": 2
          },
          "examples": [
            {
              "actual_pl_jpy": -1019.7829,
              "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
              "method": "TREND_CONTINUATION",
              "pair": "EUR_CHF",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -1019.7829,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472445"
            },
            {
              "actual_pl_jpy": -252.2942,
              "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
              "method": "TREND_CONTINUATION",
              "pair": "EUR_CHF",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -252.2942,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472174"
            },
            {
              "actual_pl_jpy": -620.0993,
              "lane_id": "trend_trader:USD_CAD:LONG:TREND_CONTINUATION:MARKET",
              "method": "TREND_CONTINUATION",
              "pair": "USD_CAD",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -620.0993,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472427"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 6
          },
          "group_count": 5,
          "loss_closes": 6,
          "method": "TREND_CONTINUATION",
          "pair_count": 5,
          "pairs": [
            "EUR_CHF",
            "EUR_GBP",
            "GBP_AUD",
            "GBP_CHF",
            "USD_CAD"
          ],
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -3065.35,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side_count": 2,
          "sides": [
            "LONG",
            "SHORT"
          ]
        }
      ],
      "top_profit_capture_misses": [
        {
          "counterfactual_delta_jpy": 446.04,
          "counterfactual_exit": "TP_PROGRESS_CAPTURE",
          "counterfactual_jpy": 105.84,
          "exit_reason": "STOP_LOSS_ORDER",
          "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
          "pair": "USD_JPY",
          "realized_pl_jpy": -340.2,
          "repair_replay_block_reason": null,
          "repair_replay_candidate_noise_floor_pips": null,
          "repair_replay_candidate_profit_pips": null,
          "repair_replay_max_profit_pips": null,
          "repair_replay_max_tp_progress": null,
          "side": "SHORT",
          "tp_progress_before_loss_close": 0.6071,
          "trade_id": "472792"
        },
        {
          "counterfactual_delta_jpy": 603.3741,
          "counterfactual_exit": "TP_PROGRESS_CAPTURE",
          "counterfactual_jpy": 363.895,
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
          "pair": "AUD_NZD",
          "realized_pl_jpy": -239.4791,
          "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
          "repair_replay_candidate_noise_floor_pips": 4.9667,
          "repair_replay_candidate_profit_pips": 4.7,
          "repair_replay_max_profit_pips": 4.7,
          "repair_replay_max_tp_progress": 0.3507,
          "side": "SHORT",
          "tp_progress_before_loss_close": 0.3507,
          "trade_id": "472632"
        },
        {
          "counterfactual_delta_jpy": 778.6774,
          "counterfactual_exit": "TP_PROGRESS_CAPTURE",
          "counterfactual_jpy": 613.508,
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
          "pair": "USD_CAD",
          "realized_pl_jpy": -165.1694,
          "repair_replay_block_reason": null,
          "repair_replay_candidate_noise_floor_pips": null,
          "repair_replay_candidate_profit_pips": null,
          "repair_replay_max_profit_pips": null,
          "repair_replay_max_tp_progress": null,
          "side": "LONG",
          "tp_progress_before_loss_close": 0.3571,
          "trade_id": "472318"
        },
        {
          "counterfactual_delta_jpy": 340.7987,
          "counterfactual_exit": "TP_PROGRESS_CAPTURE",
          "counterfactual_jpy": 311.922,
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
          "pair": "EUR_USD",
          "realized_pl_jpy": -28.8767,
          "repair_replay_block_reason": null,
          "repair_replay_candidate_noise_floor_pips": null,
          "repair_replay_candidate_profit_pips": null,
          "repair_replay_max_profit_pips": null,
          "repair_replay_max_tp_progress": null,
          "side": "LONG",
          "tp_progress_before_loss_close": 0.9159,
          "trade_id": "472280"
        },
        {
          "counterfactual_delta_jpy": 1265.9911,
          "counterfactual_exit": "TP_PROGRESS_CAPTURE",
          "counterfactual_jpy": 284.1969,
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "lane_id": "trend_trader:GBP_CHF:LONG:TREND_CONTINUATION",
          "pair": "GBP_CHF",
          "realized_pl_jpy": -981.7942,
          "repair_replay_block_reason": null,
          "repair_replay_candidate_noise_floor_pips": null,
          "repair_replay_candidate_profit_pips": null,
          "repair_replay_max_profit_pips": null,
          "repair_replay_max_tp_progress": null,
          "side": "LONG",
          "tp_progress_before_loss_close": 0.3138,
          "trade_id": "472222"
        }
      ],
      "top_repair_replay_blocks": [
        {
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
          "pair": "AUD_NZD",
          "realized_pl_jpy": -239.4791,
          "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
          "repair_replay_candidate_m1_atr_pips": 4.9667,
          "repair_replay_candidate_noise_floor_pips": 4.9667,
          "repair_replay_candidate_profit_pips": 4.7,
          "repair_replay_candidate_spread_pips": 2.8,
          "repair_replay_candidate_tp_progress": 0.3507,
          "repair_replay_max_profit_pips": 4.7,
          "repair_replay_max_tp_progress": 0.3507,
          "side": "SHORT",
          "tp_progress_before_loss_close": 0.3507,
          "trade_id": "472632"
        }
      ],
      "top_repair_replay_residual_groups": [
        {
          "actual_pl_jpy": -2981.8961,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -2981.8961,
              "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -2981.8961,
              "repair_replay_triggered": false,
              "trade_id": "472071"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "BREAKOUT_FAILURE",
          "pair": "GBP_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -2981.8961,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        },
        {
          "actual_pl_jpy": -2333.8215,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -2333.8215,
              "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -2333.8215,
              "repair_replay_triggered": false,
              "trade_id": "471817"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "EUR_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -2333.8215,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        },
        {
          "actual_pl_jpy": -2181.1565,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -2181.1565,
              "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -2181.1565,
              "repair_replay_triggered": false,
              "trade_id": "471711"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "EUR_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -2181.1565,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        },
        {
          "actual_pl_jpy": -2044.4543,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 2
          },
          "examples": [
            {
              "actual_pl_jpy": -548.9268,
              "lane_id": "range_trader:NZD_CAD:SHORT:RANGE_ROTATION:MARKET",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -548.9268,
              "repair_replay_triggered": false,
              "trade_id": "472380"
            },
            {
              "actual_pl_jpy": -1495.5275,
              "lane_id": "range_trader:NZD_CAD:SHORT:RANGE_ROTATION:MARKET",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1495.5275,
              "repair_replay_triggered": false,
              "trade_id": "472312"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 2,
          "method": "RANGE_ROTATION",
          "pair": "NZD_CAD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -2044.4543,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        },
        {
          "actual_pl_jpy": -1705.6738,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -1705.6738,
              "lane_id": "range_trader:AUD_USD:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1705.6738,
              "repair_replay_triggered": false,
              "trade_id": "472834"
            }
          ],
          "exit_reason": "STOP_LOSS_ORDER",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "AUD_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -1705.6738,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        },
        {
          "actual_pl_jpy": -1380.8008,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -1380.8008,
              "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1380.8008,
              "repair_replay_triggered": false,
              "trade_id": "472743"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "NZD_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -1380.8008,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        },
        {
          "actual_pl_jpy": -1272.0771,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 2
          },
          "examples": [
            {
              "actual_pl_jpy": -1019.7829,
              "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -1019.7829,
              "repair_replay_triggered": false,
              "trade_id": "472445"
            },
            {
              "actual_pl_jpy": -252.2942,
              "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -252.2942,
              "repair_replay_triggered": false,
              "trade_id": "472174"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 2,
          "method": "TREND_CONTINUATION",
          "pair": "EUR_CHF",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -1272.0771,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        },
        {
          "actual_pl_jpy": -1071.9,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -1071.9,
              "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1071.9,
              "repair_replay_triggered": false,
              "trade_id": "472094"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "EUR_JPY",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -1071.9,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        },
        {
          "actual_pl_jpy": -971.0121,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -971.0121,
              "lane_id": "range_trader:GBP_USD:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -971.0121,
              "repair_replay_triggered": false,
              "trade_id": "472837"
            }
          ],
          "exit_reason": "STOP_LOSS_ORDER",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "GBP_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -971.0121,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        },
        {
          "actual_pl_jpy": -955.691,
          "block_reasons": {
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -955.691,
              "lane_id": "trend_trader:GBP_CHF:LONG:TREND_CONTINUATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -955.691,
              "repair_replay_triggered": false,
              "trade_id": "472190"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "TREND_CONTINUATION",
          "pair": "GBP_CHF",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -955.691,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
        }
      ],
      "top_repair_replay_residual_method_rollups": [
        {
          "actual_pl_jpy": -12945.8682,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 4,
            "NO_PROFIT_CANDIDATE": 8
          },
          "examples": [
            {
              "actual_pl_jpy": -971.0121,
              "lane_id": "range_trader:GBP_USD:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "GBP_USD",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -971.0121,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472837"
            },
            {
              "actual_pl_jpy": -1705.6738,
              "lane_id": "range_trader:AUD_USD:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "AUD_USD",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1705.6738,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472834"
            },
            {
              "actual_pl_jpy": -280.8,
              "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "USD_JPY",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -280.8,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472775"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 9,
            "STOP_LOSS_ORDER": 3
          },
          "group_count": 11,
          "loss_closes": 12,
          "method": "RANGE_ROTATION",
          "pair_count": 9,
          "pairs": [
            "AUD_USD",
            "CHF_JPY",
            "EUR_JPY",
            "EUR_USD",
            "GBP_USD",
            "NZD_CAD",
            "NZD_CHF",
            "NZD_USD",
            "USD_JPY"
          ],
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -12945.8682,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side_count": 2,
          "sides": [
            "LONG",
            "SHORT"
          ]
        },
        {
          "actual_pl_jpy": -4081.1334,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 4
          },
          "examples": [
            {
              "actual_pl_jpy": -38.054,
              "lane_id": "failure_trader:EUR_CHF:LONG:BREAKOUT_FAILURE:LIMIT",
              "method": "BREAKOUT_FAILURE",
              "pair": "EUR_CHF",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -38.054,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472252"
            },
            {
              "actual_pl_jpy": -891.0833,
              "lane_id": "failure_trader:EUR_GBP:SHORT:BREAKOUT_FAILURE:MARKET",
              "method": "BREAKOUT_FAILURE",
              "pair": "EUR_GBP",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -891.0833,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472208"
            },
            {
              "actual_pl_jpy": -170.1,
              "lane_id": "failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE:LIMIT",
              "method": "BREAKOUT_FAILURE",
              "pair": "EUR_JPY",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -170.1,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472156"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 4
          },
          "group_count": 4,
          "loss_closes": 4,
          "method": "BREAKOUT_FAILURE",
          "pair_count": 4,
          "pairs": [
            "EUR_CHF",
            "EUR_GBP",
            "EUR_JPY",
            "GBP_USD"
          ],
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -4081.1334,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side_count": 2,
          "sides": [
            "LONG",
            "SHORT"
          ]
        },
        {
          "actual_pl_jpy": -3065.35,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 4,
            "NO_PROFIT_CANDIDATE": 2
          },
          "examples": [
            {
              "actual_pl_jpy": -1019.7829,
              "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
              "method": "TREND_CONTINUATION",
              "pair": "EUR_CHF",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -1019.7829,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472445"
            },
            {
              "actual_pl_jpy": -252.2942,
              "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
              "method": "TREND_CONTINUATION",
              "pair": "EUR_CHF",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -252.2942,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472174"
            },
            {
              "actual_pl_jpy": -620.0993,
              "lane_id": "trend_trader:USD_CAD:LONG:TREND_CONTINUATION:MARKET",
              "method": "TREND_CONTINUATION",
              "pair": "USD_CAD",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -620.0993,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472427"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 6
          },
          "group_count": 5,
          "loss_closes": 6,
          "method": "TREND_CONTINUATION",
          "pair_count": 5,
          "pairs": [
            "EUR_CHF",
            "EUR_GBP",
            "GBP_AUD",
            "GBP_CHF",
            "USD_CAD"
          ],
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -3065.35,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side_count": 2,
          "sides": [
            "LONG",
            "SHORT"
          ]
        },
        {
          "actual_pl_jpy": -239.4791,
          "block_reasons": {
            "BELOW_NOISE_FLOOR": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -239.4791,
              "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "AUD_NZD",
              "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
              "repair_replay_pl_jpy": -239.4791,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472632"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 1
          },
          "group_count": 1,
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair_count": 1,
          "pairs": [
            "AUD_NZD"
          ],
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -239.4791,
          "repair_replay_triggered": 0,
          "residual_scope": "TP_PROGRESS_DIAGNOSTIC_BLOCKED",
          "side_count": 1,
          "sides": [
            "SHORT"
          ]
        }
      ],
      "top_repair_replay_triggers": [
        {
          "exit_reason": "STOP_LOSS_ORDER",
          "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
          "pair": "USD_JPY",
          "realized_pl_jpy": -340.2,
          "repair_counterfactual_delta_jpy": 466.2,
          "repair_counterfactual_jpy": 126.0,
          "repair_noise_floor_pips": 1.6,
          "repair_profit_pips": 2.0,
          "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
          "repair_tp_progress": 0.3571,
          "repair_trigger_at_utc": "2026-06-22T06:45:00+00:00",
          "side": "SHORT",
          "tp_progress_before_loss_close": 0.6071,
          "trade_id": "472792"
        },
        {
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
          "pair": "USD_CAD",
          "realized_pl_jpy": -165.1694,
          "repair_counterfactual_delta_jpy": 831.6289,
          "repair_counterfactual_jpy": 666.4595,
          "repair_noise_floor_pips": 1.7,
          "repair_profit_pips": 7.3,
          "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
          "repair_tp_progress": 0.3259,
          "repair_trigger_at_utc": "2026-06-12T09:09:00+00:00",
          "side": "LONG",
          "tp_progress_before_loss_close": 0.3571,
          "trade_id": "472318"
        },
        {
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
          "pair": "EUR_USD",
          "realized_pl_jpy": -28.8767,
          "repair_counterfactual_delta_jpy": 388.4129,
          "repair_counterfactual_jpy": 359.5362,
          "repair_noise_floor_pips": 1.6,
          "repair_profit_pips": 3.7,
          "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
          "repair_tp_progress": 0.3458,
          "repair_trigger_at_utc": "2026-06-11T22:45:00+00:00",
          "side": "LONG",
          "tp_progress_before_loss_close": 0.9159,
          "trade_id": "472280"
        },
        {
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "lane_id": "trend_trader:GBP_CHF:LONG:TREND_CONTINUATION",
          "pair": "GBP_CHF",
          "realized_pl_jpy": -981.7942,
          "repair_counterfactual_delta_jpy": 1279.0713,
          "repair_counterfactual_jpy": 297.2771,
          "repair_noise_floor_pips": 2.8,
          "repair_profit_pips": 7.5,
          "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
          "repair_tp_progress": 0.3138,
          "repair_trigger_at_utc": "2026-06-11T17:34:00+00:00",
          "side": "LONG",
          "tp_progress_before_loss_close": 0.3138,
          "trade_id": "472222"
        },
        {
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "lane_id": "trend_trader:EUR_AUD:LONG:TREND_CONTINUATION",
          "pair": "EUR_AUD",
          "realized_pl_jpy": -6.7428,
          "repair_counterfactual_delta_jpy": 122.8493,
          "repair_counterfactual_jpy": 116.1065,
          "repair_noise_floor_pips": 3.7,
          "repair_profit_pips": 5.2,
          "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
          "repair_tp_progress": 0.65,
          "repair_trigger_at_utc": "2026-06-11T17:03:00+00:00",
          "side": "LONG",
          "tp_progress_before_loss_close": 0.65,
          "trade_id": "472230"
        }
      ],
      "top_tp_progress_repair_residual_groups": [
        {
          "actual_pl_jpy": -239.4791,
          "block_reasons": {
            "BELOW_NOISE_FLOOR": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -239.4791,
              "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
              "repair_replay_pl_jpy": -239.4791,
              "repair_replay_triggered": false,
              "trade_id": "472632"
            }
          ],
          "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "AUD_NZD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -239.4791,
          "repair_replay_triggered": 0,
          "residual_scope": "TP_PROGRESS_DIAGNOSTIC_BLOCKED",
          "side": "SHORT"
        }
      ],
      "top_tp_progress_repair_residual_method_rollups": [
        {
          "actual_pl_jpy": -239.4791,
          "block_reasons": {
            "BELOW_NOISE_FLOOR": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -239.4791,
              "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "AUD_NZD",
              "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
              "repair_replay_pl_jpy": -239.4791,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472632"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 1
          },
          "group_count": 1,
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair_count": 1,
          "pairs": [
            "AUD_NZD"
          ],
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -239.4791,
          "repair_replay_triggered": 0,
          "residual_scope": "TP_PROGRESS_DIAGNOSTIC_BLOCKED",
          "side_count": 1,
          "sides": [
            "SHORT"
          ]
        }
      ],
      "tp_progress_repair_live_evidence_boundary_utc": "2026-06-22T17:54:26Z",
      "tp_progress_repair_live_evidence_split_present": true,
      "tp_progress_repair_live_evidence_status": "POST_REPAIR_REPLAY_CLEAN",
      "window_from_utc": "2026-05-25T19:16:37.296403+00:00",
      "window_lookback_hours": 744.0,
      "window_to_utc": "2026-06-26T06:22:19.531654+00:00"
    },
    "gateway_event_stream_events": 31506,
    "gateway_event_stream_lag_minutes": 14.009,
    "gateway_event_stream_latest_ts_utc": "2026-06-26T06:26:25.013983+00:00",
    "gateway_event_stream_market_close_gap_minutes": 0.0,
    "gateway_event_stream_stale": false,
    "gateway_market_closes": 98,
    "latest_gateway_market_close_ts_utc": "2026-06-25T19:16:35.131073+00:00",
    "latest_loss_close_ts_utc": "2026-06-19T14:22:08.785628+00:00",
    "ledger_exists": true,
    "lookback_days": 7,
    "path": "/Users/tossaki/App/QuantRabbit-live/data/execution_ledger.db",
    "recent_close_gate_missing_loss_closes": 1,
    "recent_close_gate_missing_loss_examples": [
      {
        "accepted_gpt_close_ts_utc": "2026-06-19T14:20:02.545196+00:00",
        "accepted_receipt_close_gate_evidence_count": 0,
        "accepted_receipt_has_any_close_gate_evidence": false,
        "accepted_receipt_has_close_gate_evidence": false,
        "close_provenance": "GATEWAY_TRADE_CLOSE_SENT",
        "gateway_exit_reason": "GPT_CLOSE",
        "has_close_gate_evidence": false,
        "has_passing_close_gate_evidence": false,
        "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
        "order_id": "472750",
        "pair": "NZD_USD",
        "realized_pl_jpy": -1380.8008,
        "side": "LONG",
        "timing_path_label": "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE",
        "trade_id": "472743",
        "ts_utc": "2026-06-19T14:22:08.785628+00:00"
      }
    ],
    "recent_close_gate_missing_loss_net_jpy": -1380.8008,
    "recent_close_gate_missing_receipt_evidence_absent_loss_closes": 1,
    "recent_close_gate_missing_receipt_evidence_absent_loss_net_jpy": -1380.8008,
    "recent_close_gate_missing_receipt_evidence_present_loss_closes": 0,
    "recent_close_gate_missing_receipt_evidence_present_loss_net_jpy": 0,
    "recent_close_gate_not_passing_loss_closes": 0,
    "recent_close_gate_not_passing_loss_examples": [],
    "recent_close_gate_not_passing_loss_net_jpy": 0,
    "recent_close_gate_unverified_loss_closes": 1,
    "recent_close_gate_unverified_loss_examples": [
      {
        "close_provenance": "GATEWAY_TRADE_CLOSE_SENT",
        "gateway_exit_reason": "GPT_CLOSE",
        "has_close_gate_evidence": false,
        "has_passing_close_gate_evidence": false,
        "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
        "order_id": "472750",
        "pair": "NZD_USD",
        "realized_pl_jpy": -1380.8008,
        "side": "LONG",
        "timing_path_label": "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE",
        "trade_id": "472743",
        "ts_utc": "2026-06-19T14:22:08.785628+00:00"
      }
    ],
    "recent_close_gate_unverified_loss_net_jpy": -1380.8008,
    "recent_contained_risk_loss_closes": 0,
    "recent_contained_risk_loss_examples": [],
    "recent_contained_risk_loss_net_jpy": 0,
    "recent_gateway_market_closes": 6,
    "recent_leak_loss_by_lane": [
      {
        "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "net_jpy": -1380.8008,
        "pair": "NZD_USD",
        "side": "LONG"
      }
    ],
    "recent_leak_loss_closes": 1,
    "recent_leak_loss_examples": [
      {
        "close_provenance": "GATEWAY_TRADE_CLOSE_SENT",
        "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
        "order_id": "472750",
        "pair": "NZD_USD",
        "realized_pl_jpy": -1380.8008,
        "side": "LONG",
        "timing_path_label": "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE",
        "trade_id": "472743",
        "ts_utc": "2026-06-19T14:22:08.785628+00:00"
      }
    ],
    "recent_leak_loss_net_jpy": -1380.8008,
    "recent_loss_by_lane": [
      {
        "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "net_jpy": -1380.8008,
        "pair": "NZD_USD",
        "side": "LONG"
      }
    ],
    "recent_loss_closes": 1,
    "recent_loss_examples": [
      {
        "close_provenance": "GATEWAY_TRADE_CLOSE_SENT",
        "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
        "order_id": "472750",
        "pair": "NZD_USD",
        "realized_pl_jpy": -1380.8008,
        "side": "LONG",
        "timing_path_label": "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE",
        "trade_id": "472743",
        "ts_utc": "2026-06-19T14:22:08.785628+00:00"
      }
    ],
    "recent_loss_net_jpy": -1380.8008,
    "recent_loss_timing_label_counts": {
      "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE": 1
    },
    "recent_premature_loss_closes": 1,
    "recent_unclassified_loss_closes": 0,
    "recent_unverified_loss_closes": 0,
    "recent_unverified_loss_examples": [],
    "recent_unverified_loss_net_jpy": 0
  },
  "finding_counts": {
    "P0": 6,
    "P1": 4,
    "P2": 0
  },
  "generated_at_utc": "2026-06-26T06:40:25.577826+00:00",
  "oanda_campaign_firepower": {
    "contract": "audit-only firepower estimate from validation expectancy R at the configured per-trade risk lens; it does not grant live permission, size orders, or waive forecast, spread, strategy-profile, risk, broker-truth, or gateway gates",
    "evidence_queue": {
      "estimated_return_pct_per_active_day_at_observed_frequency": 11.156301,
      "observed_attempts_per_active_day": 17.466666,
      "pair_count": 1,
      "top_vehicle_keys": [
        "EUR_JPY|SHORT|range_reversion|tp1.25_sl1",
        "EUR_JPY|SHORT|range_reversion|tp1_sl0.75",
        "EUR_JPY|SHORT|range_reversion|tp0.75_sl0.75"
      ],
      "trades_needed_for_minimum_5pct_at_weighted_expectancy": 8.0,
      "trades_needed_for_target_10pct_at_weighted_expectancy": 16.0,
      "unique_vehicle_count": 6,
      "weighted_return_pct_per_trade_at_risk_lens": 0.63872
    },
    "generated_at_utc": "2026-06-25T00:41:15.541769Z",
    "high_precision": {
      "estimated_return_pct_per_active_day_at_observed_frequency": 5.299476,
      "observed_attempts_per_active_day": 11.666666,
      "pair_count": 1,
      "top_vehicle_keys": [
        "EUR_JPY|SHORT|trend_continuation|tp1.25_sl1",
        "EUR_JPY|SHORT|trend_continuation|tp1_sl1",
        "EUR_JPY|SHORT|range_reversion|tp1.25_sl1"
      ],
      "trades_needed_for_minimum_5pct_at_weighted_expectancy": 12.0,
      "trades_needed_for_target_10pct_at_weighted_expectancy": 23.0,
      "unique_vehicle_count": 4,
      "weighted_return_pct_per_trade_at_risk_lens": 0.454241
    },
    "minimum_return_pct": 5.0,
    "path": "/Users/tossaki/App/QuantRabbit/logs/reports/forecast_improvement/oanda_universal_rotation_mining_latest.json",
    "per_trade_risk_pct_lens": 1.0,
    "report_exists": true,
    "status": "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED",
    "target_open": true,
    "target_return_pct": 10.0
  },
  "order_capture_freshness": {
    "capture_economics_generated_at_utc": "2026-06-26T06:39:35.134906+00:00",
    "capture_generated_after_order_intents": false,
    "capture_trades": 225,
    "intent_capture_economics_trades": [
      225
    ],
    "metadata_trade_count_mismatch": false,
    "mismatch_examples": [],
    "order_intents_generated_at_utc": "2026-06-26T06:40:12.298577+00:00"
  },
  "order_intents": {
    "candidate_count": 82,
    "dry_run_blocked_lanes": 82,
    "generated_at_utc": "2026-06-26T06:40:12.298577+00:00",
    "live_ready_lanes": 0,
    "repair_frontier": {
      "blocked_count": 8,
      "candidate_count": 8,
      "examples": [
        {
          "blocker_codes": [
            "REWARD_RISK_TOO_LOW"
          ],
          "blocker_count": 1,
          "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "LIMIT",
          "pair": "EUR_USD",
          "positive_rotation_pessimistic_expectancy_jpy": 326.2315,
          "repair_mode": "TP_PROVEN_HARVEST",
          "risk_allowed": false,
          "side": "LONG",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH"
          ],
          "blocker_count": 2,
          "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "LIMIT",
          "pair": "EUR_USD",
          "positive_rotation_pessimistic_expectancy_jpy": 305.7481,
          "repair_mode": "TP_PROOF_COLLECTION_HARVEST",
          "risk_allowed": false,
          "side": "SHORT",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE",
            "PATTERN_REVERSAL_CHASE",
            "EXHAUSTION_RANGE_CHASE"
          ],
          "blocker_count": 3,
          "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "STOP-ENTRY",
          "pair": "EUR_USD",
          "positive_rotation_pessimistic_expectancy_jpy": 326.2315,
          "repair_mode": "TP_PROVEN_HARVEST",
          "risk_allowed": false,
          "side": "LONG",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH"
          ],
          "blocker_count": 3,
          "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "STOP-ENTRY",
          "pair": "EUR_USD",
          "positive_rotation_pessimistic_expectancy_jpy": 305.7481,
          "repair_mode": "TP_PROOF_COLLECTION_HARVEST",
          "risk_allowed": false,
          "side": "SHORT",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH"
          ],
          "blocker_count": 3,
          "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "LIMIT",
          "pair": "GBP_USD",
          "positive_rotation_pessimistic_expectancy_jpy": 85.3139,
          "repair_mode": "TP_PROOF_COLLECTION_HARVEST",
          "risk_allowed": false,
          "side": "LONG",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "HARVEST_TP_STRUCTURE_MISSING",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
            "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"
          ],
          "blocker_count": 4,
          "lane_id": "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "LIMIT",
          "pair": "AUD_JPY",
          "positive_rotation_pessimistic_expectancy_jpy": 193.4726,
          "repair_mode": "TP_PROOF_COLLECTION_HARVEST",
          "risk_allowed": false,
          "side": "SHORT",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "PATTERN_REVERSAL_CHASE",
            "EXHAUSTION_RANGE_CHASE",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
            "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"
          ],
          "blocker_count": 5,
          "lane_id": "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "STOP-ENTRY",
          "pair": "AUD_JPY",
          "positive_rotation_pessimistic_expectancy_jpy": 193.4726,
          "repair_mode": "TP_PROOF_COLLECTION_HARVEST",
          "risk_allowed": false,
          "side": "SHORT",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE",
            "EXHAUSTION_RANGE_CHASE",
            "MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH"
          ],
          "blocker_count": 5,
          "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "STOP-ENTRY",
          "pair": "GBP_USD",
          "positive_rotation_pessimistic_expectancy_jpy": 85.3139,
          "repair_mode": "TP_PROOF_COLLECTION_HARVEST",
          "risk_allowed": false,
          "side": "LONG",
          "status": "DRY_RUN_BLOCKED"
        }
      ],
      "live_ready_count": 0,
      "top_remaining_blockers": [
        {
          "code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
          "count": 6
        },
        {
          "code": "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
          "count": 6
        },
        {
          "code": "BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE",
          "count": 3
        },
        {
          "code": "EXHAUSTION_RANGE_CHASE",
          "count": 3
        },
        {
          "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
          "count": 2
        },
        {
          "code": "MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED",
          "count": 2
        },
        {
          "code": "PATTERN_REVERSAL_CHASE",
          "count": 2
        },
        {
          "code": "HARVEST_TP_STRUCTURE_MISSING",
          "count": 1
        },
        {
          "code": "REWARD_RISK_TOO_LOW",
          "count": 1
        }
      ]
    },
    "top_blockers": [
      {
        "code": "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
        "count": 80
      },
      {
        "code": "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
        "count": 74
      },
      {
        "code": "RANGE_ROTATION_BROADER_LOCATION_CHASE",
        "count": 40
      },
      {
        "code": "FORECAST_WATCH_ONLY",
        "count": 31
      },
      {
        "code": "EXHAUSTION_RANGE_CHASE",
        "count": 30
      },
      {
        "code": "SPREAD_TOO_WIDE",
        "count": 27
      },
      {
        "code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
        "count": 24
      },
      {
        "code": "STRATEGY_PROFILE_MISSING",
        "count": 24
      },
      {
        "code": "STRATEGY_NOT_ELIGIBLE",
        "count": 22
      },
      {
        "code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
        "count": 18
      }
    ]
  },
  "profit_capture_replay_repair": {
    "active_counterfactual_profit_capture_pl_jpy": -16505.9883,
    "capture_market_close_net_jpy": -74564.8,
    "capture_take_profit_net_jpy": 48804.8,
    "clearance_condition": "execution-timing-audit must report zero post-repair live-evidence loss_closes_repair_replay_triggered with the current production-gate replay contract after TP-progress TAKE_PROFIT_MARKET / guardian repair has run on live broker truth; pre-repair historical misses remain diagnostic unless a post-repair production gate also proves an executable profit capture",
    "counterfactual_profit_capture_delta_jpy": 18770.4579,
    "counterfactual_profit_capture_jpy": 3825.8424,
    "execution_timing_generated_at_utc": "2026-06-26T06:22:19.531654+00:00",
    "execution_timing_loaded": true,
    "execution_timing_window_from_utc": "2026-05-25T19:16:37.296403+00:00",
    "execution_timing_window_lookback_hours": 744.0,
    "execution_timing_window_to_utc": "2026-06-26T06:22:19.531654+00:00",
    "guardian_profit_capture_inactive": false,
    "loss_close_repair_replay_block_reasons": {
      "BELOW_NOISE_FLOOR": 1
    },
    "loss_closes_profit_capture_missed": 14,
    "loss_closes_repair_replay_triggered": 13,
    "month_scale_replay_loaded": true,
    "month_scale_replay_min_hours": 720.0,
    "month_scale_replay_required": true,
    "post_repair_live_evidence_loss_closes_audited": 2,
    "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
    "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
    "pre_repair_historical_loss_closes_profit_capture_missed": 14,
    "pre_repair_historical_loss_closes_repair_replay_triggered": 13,
    "raw_counterfactual_profit_capture_pl_jpy": -16552.1985,
    "repair_replay_contract": "TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1",
    "repair_replay_contract_present": true,
    "repair_replay_counterfactual_pl_jpy": -16505.9883,
    "replay_repair_proved": false,
    "self_improvement_p0_codes": [
      "TARGET_OPEN_NO_LIVE_READY_LANES"
    ],
    "self_improvement_profit_capture_context": false,
    "top_entry_quality_residual_groups": [
      {
        "actual_pl_jpy": -2981.8961,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -2981.8961,
            "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -2981.8961,
            "repair_replay_triggered": false,
            "trade_id": "472071"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "BREAKOUT_FAILURE",
        "pair": "GBP_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -2981.8961,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      },
      {
        "actual_pl_jpy": -2333.8215,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -2333.8215,
            "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -2333.8215,
            "repair_replay_triggered": false,
            "trade_id": "471817"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "EUR_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -2333.8215,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      },
      {
        "actual_pl_jpy": -2181.1565,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -2181.1565,
            "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -2181.1565,
            "repair_replay_triggered": false,
            "trade_id": "471711"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "EUR_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -2181.1565,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "SHORT"
      },
      {
        "actual_pl_jpy": -2044.4543,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 2
        },
        "examples": [
          {
            "actual_pl_jpy": -548.9268,
            "lane_id": "range_trader:NZD_CAD:SHORT:RANGE_ROTATION:MARKET",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -548.9268,
            "repair_replay_triggered": false,
            "trade_id": "472380"
          },
          {
            "actual_pl_jpy": -1495.5275,
            "lane_id": "range_trader:NZD_CAD:SHORT:RANGE_ROTATION:MARKET",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -1495.5275,
            "repair_replay_triggered": false,
            "trade_id": "472312"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 2,
        "method": "RANGE_ROTATION",
        "pair": "NZD_CAD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -2044.4543,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "SHORT"
      },
      {
        "actual_pl_jpy": -1705.6738,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -1705.6738,
            "lane_id": "range_trader:AUD_USD:SHORT:RANGE_ROTATION",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -1705.6738,
            "repair_replay_triggered": false,
            "trade_id": "472834"
          }
        ],
        "exit_reason": "STOP_LOSS_ORDER",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "AUD_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -1705.6738,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "SHORT"
      },
      {
        "actual_pl_jpy": -1380.8008,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -1380.8008,
            "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -1380.8008,
            "repair_replay_triggered": false,
            "trade_id": "472743"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "NZD_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -1380.8008,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      },
      {
        "actual_pl_jpy": -1272.0771,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 2
        },
        "examples": [
          {
            "actual_pl_jpy": -1019.7829,
            "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -1019.7829,
            "repair_replay_triggered": false,
            "trade_id": "472445"
          },
          {
            "actual_pl_jpy": -252.2942,
            "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -252.2942,
            "repair_replay_triggered": false,
            "trade_id": "472174"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 2,
        "method": "TREND_CONTINUATION",
        "pair": "EUR_CHF",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -1272.0771,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      },
      {
        "actual_pl_jpy": -1071.9,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -1071.9,
            "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -1071.9,
            "repair_replay_triggered": false,
            "trade_id": "472094"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "EUR_JPY",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -1071.9,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      },
      {
        "actual_pl_jpy": -971.0121,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -971.0121,
            "lane_id": "range_trader:GBP_USD:SHORT:RANGE_ROTATION",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -971.0121,
            "repair_replay_triggered": false,
            "trade_id": "472837"
          }
        ],
        "exit_reason": "STOP_LOSS_ORDER",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "GBP_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -971.0121,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "SHORT"
      },
      {
        "actual_pl_jpy": -955.691,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -955.691,
            "lane_id": "trend_trader:GBP_CHF:LONG:TREND_CONTINUATION",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -955.691,
            "repair_replay_triggered": false,
            "trade_id": "472190"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "TREND_CONTINUATION",
        "pair": "GBP_CHF",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -955.691,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      }
    ],
    "top_entry_quality_residual_method_rollups": [
      {
        "actual_pl_jpy": -12945.8682,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 4,
          "NO_PROFIT_CANDIDATE": 8
        },
        "examples": [
          {
            "actual_pl_jpy": -971.0121,
            "lane_id": "range_trader:GBP_USD:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "GBP_USD",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -971.0121,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472837"
          },
          {
            "actual_pl_jpy": -1705.6738,
            "lane_id": "range_trader:AUD_USD:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "AUD_USD",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -1705.6738,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472834"
          },
          {
            "actual_pl_jpy": -280.8,
            "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "USD_JPY",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -280.8,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472775"
          }
        ],
        "exit_reasons": {
          "MARKET_ORDER_TRADE_CLOSE": 9,
          "STOP_LOSS_ORDER": 3
        },
        "group_count": 11,
        "loss_closes": 12,
        "method": "RANGE_ROTATION",
        "pair_count": 9,
        "pairs": [
          "AUD_USD",
          "CHF_JPY",
          "EUR_JPY",
          "EUR_USD",
          "GBP_USD",
          "NZD_CAD",
          "NZD_CHF",
          "NZD_USD",
          "USD_JPY"
        ],
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -12945.8682,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side_count": 2,
        "sides": [
          "LONG",
          "SHORT"
        ]
      },
      {
        "actual_pl_jpy": -4081.1334,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 4
        },
        "examples": [
          {
            "actual_pl_jpy": -38.054,
            "lane_id": "failure_trader:EUR_CHF:LONG:BREAKOUT_FAILURE:LIMIT",
            "method": "BREAKOUT_FAILURE",
            "pair": "EUR_CHF",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -38.054,
            "repair_replay_triggered": false,
            "side": "LONG",
            "trade_id": "472252"
          },
          {
            "actual_pl_jpy": -891.0833,
            "lane_id": "failure_trader:EUR_GBP:SHORT:BREAKOUT_FAILURE:MARKET",
            "method": "BREAKOUT_FAILURE",
            "pair": "EUR_GBP",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -891.0833,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472208"
          },
          {
            "actual_pl_jpy": -170.1,
            "lane_id": "failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE:LIMIT",
            "method": "BREAKOUT_FAILURE",
            "pair": "EUR_JPY",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -170.1,
            "repair_replay_triggered": false,
            "side": "LONG",
            "trade_id": "472156"
          }
        ],
        "exit_reasons": {
          "MARKET_ORDER_TRADE_CLOSE": 4
        },
        "group_count": 4,
        "loss_closes": 4,
        "method": "BREAKOUT_FAILURE",
        "pair_count": 4,
        "pairs": [
          "EUR_CHF",
          "EUR_GBP",
          "EUR_JPY",
          "GBP_USD"
        ],
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -4081.1334,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side_count": 2,
        "sides": [
          "LONG",
          "SHORT"
        ]
      },
      {
        "actual_pl_jpy": -3065.35,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 4,
          "NO_PROFIT_CANDIDATE": 2
        },
        "examples": [
          {
            "actual_pl_jpy": -1019.7829,
            "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
            "method": "TREND_CONTINUATION",
            "pair": "EUR_CHF",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -1019.7829,
            "repair_replay_triggered": false,
            "side": "LONG",
            "trade_id": "472445"
          },
          {
            "actual_pl_jpy": -252.2942,
            "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
            "method": "TREND_CONTINUATION",
            "pair": "EUR_CHF",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -252.2942,
            "repair_replay_triggered": false,
            "side": "LONG",
            "trade_id": "472174"
          },
          {
            "actual_pl_jpy": -620.0993,
            "lane_id": "trend_trader:USD_CAD:LONG:TREND_CONTINUATION:MARKET",
            "method": "TREND_CONTINUATION",
            "pair": "USD_CAD",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -620.0993,
            "repair_replay_triggered": false,
            "side": "LONG",
            "trade_id": "472427"
          }
        ],
        "exit_reasons": {
          "MARKET_ORDER_TRADE_CLOSE": 6
        },
        "group_count": 5,
        "loss_closes": 6,
        "method": "TREND_CONTINUATION",
        "pair_count": 5,
        "pairs": [
          "EUR_CHF",
          "EUR_GBP",
          "GBP_AUD",
          "GBP_CHF",
          "USD_CAD"
        ],
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -3065.35,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side_count": 2,
        "sides": [
          "LONG",
          "SHORT"
        ]
      }
    ],
    "top_profit_capture_misses": [
      {
        "counterfactual_delta_jpy": 446.04,
        "counterfactual_exit": "TP_PROGRESS_CAPTURE",
        "counterfactual_jpy": 105.84,
        "exit_reason": "STOP_LOSS_ORDER",
        "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
        "pair": "USD_JPY",
        "realized_pl_jpy": -340.2,
        "repair_replay_block_reason": null,
        "repair_replay_candidate_noise_floor_pips": null,
        "repair_replay_candidate_profit_pips": null,
        "repair_replay_max_profit_pips": null,
        "repair_replay_max_tp_progress": null,
        "side": "SHORT",
        "tp_progress_before_loss_close": 0.6071,
        "trade_id": "472792"
      },
      {
        "counterfactual_delta_jpy": 603.3741,
        "counterfactual_exit": "TP_PROGRESS_CAPTURE",
        "counterfactual_jpy": 363.895,
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
        "pair": "AUD_NZD",
        "realized_pl_jpy": -239.4791,
        "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
        "repair_replay_candidate_noise_floor_pips": 4.9667,
        "repair_replay_candidate_profit_pips": 4.7,
        "repair_replay_max_profit_pips": 4.7,
        "repair_replay_max_tp_progress": 0.3507,
        "side": "SHORT",
        "tp_progress_before_loss_close": 0.3507,
        "trade_id": "472632"
      },
      {
        "counterfactual_delta_jpy": 778.6774,
        "counterfactual_exit": "TP_PROGRESS_CAPTURE",
        "counterfactual_jpy": 613.508,
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
        "pair": "USD_CAD",
        "realized_pl_jpy": -165.1694,
        "repair_replay_block_reason": null,
        "repair_replay_candidate_noise_floor_pips": null,
        "repair_replay_candidate_profit_pips": null,
        "repair_replay_max_profit_pips": null,
        "repair_replay_max_tp_progress": null,
        "side": "LONG",
        "tp_progress_before_loss_close": 0.3571,
        "trade_id": "472318"
      },
      {
        "counterfactual_delta_jpy": 340.7987,
        "counterfactual_exit": "TP_PROGRESS_CAPTURE",
        "counterfactual_jpy": 311.922,
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
        "pair": "EUR_USD",
        "realized_pl_jpy": -28.8767,
        "repair_replay_block_reason": null,
        "repair_replay_candidate_noise_floor_pips": null,
        "repair_replay_candidate_profit_pips": null,
        "repair_replay_max_profit_pips": null,
        "repair_replay_max_tp_progress": null,
        "side": "LONG",
        "tp_progress_before_loss_close": 0.9159,
        "trade_id": "472280"
      },
      {
        "counterfactual_delta_jpy": 1265.9911,
        "counterfactual_exit": "TP_PROGRESS_CAPTURE",
        "counterfactual_jpy": 284.1969,
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "lane_id": "trend_trader:GBP_CHF:LONG:TREND_CONTINUATION",
        "pair": "GBP_CHF",
        "realized_pl_jpy": -981.7942,
        "repair_replay_block_reason": null,
        "repair_replay_candidate_noise_floor_pips": null,
        "repair_replay_candidate_profit_pips": null,
        "repair_replay_max_profit_pips": null,
        "repair_replay_max_tp_progress": null,
        "side": "LONG",
        "tp_progress_before_loss_close": 0.3138,
        "trade_id": "472222"
      }
    ],
    "top_repair_replay_blocks": [
      {
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
        "pair": "AUD_NZD",
        "realized_pl_jpy": -239.4791,
        "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
        "repair_replay_candidate_m1_atr_pips": 4.9667,
        "repair_replay_candidate_noise_floor_pips": 4.9667,
        "repair_replay_candidate_profit_pips": 4.7,
        "repair_replay_candidate_spread_pips": 2.8,
        "repair_replay_candidate_tp_progress": 0.3507,
        "repair_replay_max_profit_pips": 4.7,
        "repair_replay_max_tp_progress": 0.3507,
        "side": "SHORT",
        "tp_progress_before_loss_close": 0.3507,
        "trade_id": "472632"
      }
    ],
    "top_repair_replay_residual_groups": [
      {
        "actual_pl_jpy": -2981.8961,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -2981.8961,
            "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -2981.8961,
            "repair_replay_triggered": false,
            "trade_id": "472071"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "BREAKOUT_FAILURE",
        "pair": "GBP_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -2981.8961,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      },
      {
        "actual_pl_jpy": -2333.8215,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -2333.8215,
            "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -2333.8215,
            "repair_replay_triggered": false,
            "trade_id": "471817"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "EUR_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -2333.8215,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      },
      {
        "actual_pl_jpy": -2181.1565,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -2181.1565,
            "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -2181.1565,
            "repair_replay_triggered": false,
            "trade_id": "471711"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "EUR_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -2181.1565,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "SHORT"
      },
      {
        "actual_pl_jpy": -2044.4543,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 2
        },
        "examples": [
          {
            "actual_pl_jpy": -548.9268,
            "lane_id": "range_trader:NZD_CAD:SHORT:RANGE_ROTATION:MARKET",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -548.9268,
            "repair_replay_triggered": false,
            "trade_id": "472380"
          },
          {
            "actual_pl_jpy": -1495.5275,
            "lane_id": "range_trader:NZD_CAD:SHORT:RANGE_ROTATION:MARKET",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -1495.5275,
            "repair_replay_triggered": false,
            "trade_id": "472312"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 2,
        "method": "RANGE_ROTATION",
        "pair": "NZD_CAD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -2044.4543,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "SHORT"
      },
      {
        "actual_pl_jpy": -1705.6738,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -1705.6738,
            "lane_id": "range_trader:AUD_USD:SHORT:RANGE_ROTATION",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -1705.6738,
            "repair_replay_triggered": false,
            "trade_id": "472834"
          }
        ],
        "exit_reason": "STOP_LOSS_ORDER",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "AUD_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -1705.6738,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "SHORT"
      },
      {
        "actual_pl_jpy": -1380.8008,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -1380.8008,
            "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -1380.8008,
            "repair_replay_triggered": false,
            "trade_id": "472743"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "NZD_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -1380.8008,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      },
      {
        "actual_pl_jpy": -1272.0771,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 2
        },
        "examples": [
          {
            "actual_pl_jpy": -1019.7829,
            "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -1019.7829,
            "repair_replay_triggered": false,
            "trade_id": "472445"
          },
          {
            "actual_pl_jpy": -252.2942,
            "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -252.2942,
            "repair_replay_triggered": false,
            "trade_id": "472174"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 2,
        "method": "TREND_CONTINUATION",
        "pair": "EUR_CHF",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -1272.0771,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      },
      {
        "actual_pl_jpy": -1071.9,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -1071.9,
            "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -1071.9,
            "repair_replay_triggered": false,
            "trade_id": "472094"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "EUR_JPY",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -1071.9,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      },
      {
        "actual_pl_jpy": -971.0121,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -971.0121,
            "lane_id": "range_trader:GBP_USD:SHORT:RANGE_ROTATION",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -971.0121,
            "repair_replay_triggered": false,
            "trade_id": "472837"
          }
        ],
        "exit_reason": "STOP_LOSS_ORDER",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "GBP_USD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -971.0121,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "SHORT"
      },
      {
        "actual_pl_jpy": -955.691,
        "block_reasons": {
          "NO_PROFIT_CANDIDATE": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -955.691,
            "lane_id": "trend_trader:GBP_CHF:LONG:TREND_CONTINUATION",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -955.691,
            "repair_replay_triggered": false,
            "trade_id": "472190"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "TREND_CONTINUATION",
        "pair": "GBP_CHF",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -955.691,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side": "LONG"
      }
    ],
    "top_repair_replay_residual_method_rollups": [
      {
        "actual_pl_jpy": -12945.8682,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 4,
          "NO_PROFIT_CANDIDATE": 8
        },
        "examples": [
          {
            "actual_pl_jpy": -971.0121,
            "lane_id": "range_trader:GBP_USD:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "GBP_USD",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -971.0121,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472837"
          },
          {
            "actual_pl_jpy": -1705.6738,
            "lane_id": "range_trader:AUD_USD:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "AUD_USD",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -1705.6738,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472834"
          },
          {
            "actual_pl_jpy": -280.8,
            "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "USD_JPY",
            "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
            "repair_replay_pl_jpy": -280.8,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472775"
          }
        ],
        "exit_reasons": {
          "MARKET_ORDER_TRADE_CLOSE": 9,
          "STOP_LOSS_ORDER": 3
        },
        "group_count": 11,
        "loss_closes": 12,
        "method": "RANGE_ROTATION",
        "pair_count": 9,
        "pairs": [
          "AUD_USD",
          "CHF_JPY",
          "EUR_JPY",
          "EUR_USD",
          "GBP_USD",
          "NZD_CAD",
          "NZD_CHF",
          "NZD_USD",
          "USD_JPY"
        ],
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -12945.8682,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side_count": 2,
        "sides": [
          "LONG",
          "SHORT"
        ]
      },
      {
        "actual_pl_jpy": -4081.1334,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 4
        },
        "examples": [
          {
            "actual_pl_jpy": -38.054,
            "lane_id": "failure_trader:EUR_CHF:LONG:BREAKOUT_FAILURE:LIMIT",
            "method": "BREAKOUT_FAILURE",
            "pair": "EUR_CHF",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -38.054,
            "repair_replay_triggered": false,
            "side": "LONG",
            "trade_id": "472252"
          },
          {
            "actual_pl_jpy": -891.0833,
            "lane_id": "failure_trader:EUR_GBP:SHORT:BREAKOUT_FAILURE:MARKET",
            "method": "BREAKOUT_FAILURE",
            "pair": "EUR_GBP",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -891.0833,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472208"
          },
          {
            "actual_pl_jpy": -170.1,
            "lane_id": "failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE:LIMIT",
            "method": "BREAKOUT_FAILURE",
            "pair": "EUR_JPY",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -170.1,
            "repair_replay_triggered": false,
            "side": "LONG",
            "trade_id": "472156"
          }
        ],
        "exit_reasons": {
          "MARKET_ORDER_TRADE_CLOSE": 4
        },
        "group_count": 4,
        "loss_closes": 4,
        "method": "BREAKOUT_FAILURE",
        "pair_count": 4,
        "pairs": [
          "EUR_CHF",
          "EUR_GBP",
          "EUR_JPY",
          "GBP_USD"
        ],
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -4081.1334,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side_count": 2,
        "sides": [
          "LONG",
          "SHORT"
        ]
      },
      {
        "actual_pl_jpy": -3065.35,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 4,
          "NO_PROFIT_CANDIDATE": 2
        },
        "examples": [
          {
            "actual_pl_jpy": -1019.7829,
            "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
            "method": "TREND_CONTINUATION",
            "pair": "EUR_CHF",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -1019.7829,
            "repair_replay_triggered": false,
            "side": "LONG",
            "trade_id": "472445"
          },
          {
            "actual_pl_jpy": -252.2942,
            "lane_id": "trend_trader:EUR_CHF:LONG:TREND_CONTINUATION",
            "method": "TREND_CONTINUATION",
            "pair": "EUR_CHF",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -252.2942,
            "repair_replay_triggered": false,
            "side": "LONG",
            "trade_id": "472174"
          },
          {
            "actual_pl_jpy": -620.0993,
            "lane_id": "trend_trader:USD_CAD:LONG:TREND_CONTINUATION:MARKET",
            "method": "TREND_CONTINUATION",
            "pair": "USD_CAD",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -620.0993,
            "repair_replay_triggered": false,
            "side": "LONG",
            "trade_id": "472427"
          }
        ],
        "exit_reasons": {
          "MARKET_ORDER_TRADE_CLOSE": 6
        },
        "group_count": 5,
        "loss_closes": 6,
        "method": "TREND_CONTINUATION",
        "pair_count": 5,
        "pairs": [
          "EUR_CHF",
          "EUR_GBP",
          "GBP_AUD",
          "GBP_CHF",
          "USD_CAD"
        ],
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -3065.35,
        "repair_replay_triggered": 0,
        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
        "side_count": 2,
        "sides": [
          "LONG",
          "SHORT"
        ]
      },
      {
        "actual_pl_jpy": -239.4791,
        "block_reasons": {
          "BELOW_NOISE_FLOOR": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -239.4791,
            "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "AUD_NZD",
            "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
            "repair_replay_pl_jpy": -239.4791,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472632"
          }
        ],
        "exit_reasons": {
          "MARKET_ORDER_TRADE_CLOSE": 1
        },
        "group_count": 1,
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair_count": 1,
        "pairs": [
          "AUD_NZD"
        ],
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -239.4791,
        "repair_replay_triggered": 0,
        "residual_scope": "TP_PROGRESS_DIAGNOSTIC_BLOCKED",
        "side_count": 1,
        "sides": [
          "SHORT"
        ]
      }
    ],
    "top_repair_replay_triggers": [
      {
        "exit_reason": "STOP_LOSS_ORDER",
        "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
        "pair": "USD_JPY",
        "realized_pl_jpy": -340.2,
        "repair_counterfactual_delta_jpy": 466.2,
        "repair_counterfactual_jpy": 126.0,
        "repair_noise_floor_pips": 1.6,
        "repair_profit_pips": 2.0,
        "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
        "repair_tp_progress": 0.3571,
        "repair_trigger_at_utc": "2026-06-22T06:45:00+00:00",
        "side": "SHORT",
        "tp_progress_before_loss_close": 0.6071,
        "trade_id": "472792"
      },
      {
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
        "pair": "USD_CAD",
        "realized_pl_jpy": -165.1694,
        "repair_counterfactual_delta_jpy": 831.6289,
        "repair_counterfactual_jpy": 666.4595,
        "repair_noise_floor_pips": 1.7,
        "repair_profit_pips": 7.3,
        "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
        "repair_tp_progress": 0.3259,
        "repair_trigger_at_utc": "2026-06-12T09:09:00+00:00",
        "side": "LONG",
        "tp_progress_before_loss_close": 0.3571,
        "trade_id": "472318"
      },
      {
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
        "pair": "EUR_USD",
        "realized_pl_jpy": -28.8767,
        "repair_counterfactual_delta_jpy": 388.4129,
        "repair_counterfactual_jpy": 359.5362,
        "repair_noise_floor_pips": 1.6,
        "repair_profit_pips": 3.7,
        "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
        "repair_tp_progress": 0.3458,
        "repair_trigger_at_utc": "2026-06-11T22:45:00+00:00",
        "side": "LONG",
        "tp_progress_before_loss_close": 0.9159,
        "trade_id": "472280"
      },
      {
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "lane_id": "trend_trader:GBP_CHF:LONG:TREND_CONTINUATION",
        "pair": "GBP_CHF",
        "realized_pl_jpy": -981.7942,
        "repair_counterfactual_delta_jpy": 1279.0713,
        "repair_counterfactual_jpy": 297.2771,
        "repair_noise_floor_pips": 2.8,
        "repair_profit_pips": 7.5,
        "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
        "repair_tp_progress": 0.3138,
        "repair_trigger_at_utc": "2026-06-11T17:34:00+00:00",
        "side": "LONG",
        "tp_progress_before_loss_close": 0.3138,
        "trade_id": "472222"
      },
      {
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "lane_id": "trend_trader:EUR_AUD:LONG:TREND_CONTINUATION",
        "pair": "EUR_AUD",
        "realized_pl_jpy": -6.7428,
        "repair_counterfactual_delta_jpy": 122.8493,
        "repair_counterfactual_jpy": 116.1065,
        "repair_noise_floor_pips": 3.7,
        "repair_profit_pips": 5.2,
        "repair_replay_exit": "TP_PROGRESS_PRODUCTION_GATE_REPLAY",
        "repair_tp_progress": 0.65,
        "repair_trigger_at_utc": "2026-06-11T17:03:00+00:00",
        "side": "LONG",
        "tp_progress_before_loss_close": 0.65,
        "trade_id": "472230"
      }
    ],
    "top_tp_progress_repair_residual_groups": [
      {
        "actual_pl_jpy": -239.4791,
        "block_reasons": {
          "BELOW_NOISE_FLOOR": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -239.4791,
            "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
            "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
            "repair_replay_pl_jpy": -239.4791,
            "repair_replay_triggered": false,
            "trade_id": "472632"
          }
        ],
        "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair": "AUD_NZD",
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -239.4791,
        "repair_replay_triggered": 0,
        "residual_scope": "TP_PROGRESS_DIAGNOSTIC_BLOCKED",
        "side": "SHORT"
      }
    ],
    "top_tp_progress_repair_residual_method_rollups": [
      {
        "actual_pl_jpy": -239.4791,
        "block_reasons": {
          "BELOW_NOISE_FLOOR": 1
        },
        "examples": [
          {
            "actual_pl_jpy": -239.4791,
            "lane_id": "range_trader:AUD_NZD:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "AUD_NZD",
            "repair_replay_block_reason": "BELOW_NOISE_FLOOR",
            "repair_replay_pl_jpy": -239.4791,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472632"
          }
        ],
        "exit_reasons": {
          "MARKET_ORDER_TRADE_CLOSE": 1
        },
        "group_count": 1,
        "loss_closes": 1,
        "method": "RANGE_ROTATION",
        "pair_count": 1,
        "pairs": [
          "AUD_NZD"
        ],
        "repair_replay_delta_jpy": 0.0,
        "repair_replay_pl_jpy": -239.4791,
        "repair_replay_triggered": 0,
        "residual_scope": "TP_PROGRESS_DIAGNOSTIC_BLOCKED",
        "side_count": 1,
        "sides": [
          "SHORT"
        ]
      }
    ],
    "tp_progress_repair_live_evidence_boundary_utc": "2026-06-22T17:54:26Z",
    "tp_progress_repair_live_evidence_split_present": true,
    "tp_progress_repair_live_evidence_status": "POST_REPAIR_REPLAY_CLEAN",
    "waiting_for_post_repair_sample": false
  },
  "projection_precision": {
    "economic_precision_edges": 7,
    "economic_precision_gaps": 8,
    "ledger_exists": true,
    "top_edges": [
      {
        "bucket": "EUR_USD:UNCLEAR",
        "economic_hit_rate": 1.0,
        "economic_hit_rate_wilson_lower": 0.963,
        "economic_samples": 100,
        "hit_rate": 1.0,
        "hit_rate_wilson_lower": 0.963,
        "pair": "EUR_USD",
        "passes_economic_precision": true,
        "regime": "UNCLEAR",
        "samples": 100,
        "signal_name": "session_expansion_london",
        "timeout_count": 0,
        "timeout_rate": 0.0
      },
      {
        "bucket": "EUR_USD:_all_regimes",
        "economic_hit_rate": 1.0,
        "economic_hit_rate_wilson_lower": 0.963,
        "economic_samples": 100,
        "hit_rate": 1.0,
        "hit_rate_wilson_lower": 0.963,
        "pair": "EUR_USD",
        "passes_economic_precision": true,
        "regime": "_all_regimes",
        "samples": 100,
        "signal_name": "session_expansion_london",
        "timeout_count": 0,
        "timeout_rate": 0.0
      },
      {
        "bucket": "_all_pairs:_all_regimes",
        "economic_hit_rate": 0.98,
        "economic_hit_rate_wilson_lower": 0.93,
        "economic_samples": 100,
        "hit_rate": 0.98,
        "hit_rate_wilson_lower": 0.93,
        "pair": "_all_pairs",
        "passes_economic_precision": true,
        "regime": "_all_regimes",
        "samples": 100,
        "signal_name": "session_expansion_ny",
        "timeout_count": 0,
        "timeout_rate": 0.0
      },
      {
        "bucket": "AUD_JPY:UNCLEAR",
        "economic_hit_rate": 0.977,
        "economic_hit_rate_wilson_lower": 0.9191,
        "economic_samples": 86,
        "hit_rate": 1.0,
        "hit_rate_wilson_lower": 0.9563,
        "pair": "AUD_JPY",
        "passes_economic_precision": true,
        "regime": "UNCLEAR",
        "samples": 84,
        "signal_name": "liquidity_sweep_high_up",
        "timeout_count": 2,
        "timeout_rate": 0.0233
      },
      {
        "bucket": "AUD_JPY:_all_regimes",
        "economic_hit_rate": 0.977,
        "economic_hit_rate_wilson_lower": 0.9191,
        "economic_samples": 86,
        "hit_rate": 1.0,
        "hit_rate_wilson_lower": 0.9563,
        "pair": "AUD_JPY",
        "passes_economic_precision": true,
        "regime": "_all_regimes",
        "samples": 84,
        "signal_name": "liquidity_sweep_high_up",
        "timeout_count": 2,
        "timeout_rate": 0.0233
      }
    ],
    "top_gaps": [
      {
        "bucket": "_all_pairs:UNCLEAR",
        "economic_hit_rate": 0.43,
        "economic_hit_rate_wilson_lower": 0.3373,
        "economic_samples": 100,
        "hit_rate": 1.0,
        "hit_rate_wilson_lower": 0.918,
        "pair": "_all_pairs",
        "regime": "UNCLEAR",
        "samples": 43,
        "signal_name": "session_expansion_ny",
        "timeout_count": 57,
        "timeout_rate": 0.57
      },
      {
        "bucket": "EUR_USD:_all_regimes",
        "economic_hit_rate": 0.57,
        "economic_hit_rate_wilson_lower": 0.4722,
        "economic_samples": 100,
        "hit_rate": 1.0,
        "hit_rate_wilson_lower": 0.9369,
        "pair": "EUR_USD",
        "regime": "_all_regimes",
        "samples": 57,
        "signal_name": "session_expansion_ny",
        "timeout_count": 43,
        "timeout_rate": 0.43
      },
      {
        "bucket": "EUR_USD:TREND",
        "economic_hit_rate": 0.62,
        "economic_hit_rate_wilson_lower": 0.5221,
        "economic_samples": 100,
        "hit_rate": 1.0,
        "hit_rate_wilson_lower": 0.9417,
        "pair": "EUR_USD",
        "regime": "TREND",
        "samples": 62,
        "signal_name": "bb_squeeze_expansion_imminent",
        "timeout_count": 38,
        "timeout_rate": 0.38
      },
      {
        "bucket": "EUR_USD:_all_regimes",
        "economic_hit_rate": 0.62,
        "economic_hit_rate_wilson_lower": 0.5221,
        "economic_samples": 100,
        "hit_rate": 1.0,
        "hit_rate_wilson_lower": 0.9417,
        "pair": "EUR_USD",
        "regime": "_all_regimes",
        "samples": 62,
        "signal_name": "bb_squeeze_expansion_imminent",
        "timeout_count": 38,
        "timeout_rate": 0.38
      },
      {
        "bucket": "_all_pairs:UNCLEAR",
        "economic_hit_rate": 0.81,
        "economic_hit_rate_wilson_lower": 0.7222,
        "economic_samples": 100,
        "hit_rate": 0.964,
        "hit_rate_wilson_lower": 0.9002,
        "pair": "_all_pairs",
        "regime": "UNCLEAR",
        "samples": 84,
        "signal_name": "bb_squeeze_expansion_imminent",
        "timeout_count": 16,
        "timeout_rate": 0.16
      }
    ]
  },
  "self_improvement": {
    "p0_codes": [
      "TARGET_OPEN_NO_LIVE_READY_LANES"
    ],
    "p0_findings": 1,
    "p1_findings": 5,
    "status": "SELF_IMPROVEMENT_BLOCKED"
  },
  "target": {
    "remaining_target_jpy": 17394.49,
    "status": "PURSUE_TARGET",
    "target_open": true
  }
}
```

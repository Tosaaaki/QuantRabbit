# Payoff Shape Diagnosis Report

- Generated at UTC: `2026-07-08T07:06:55.452302+00:00`
- Status: `OK`
- Verdict: `MIXED_HARVEST_PRIMARY`
- Live promotion allowed: `False`
- Live side effects: `[]`

## 4x Payoff Shape Verdict

4x requires a mixed shape with HARVEST as the accounting base and only evidence-scoped runners. RUNNER-only promotion is not supported while market-close leakage and month-scale replay remain negative.

## HARVEST Candidates

| shape | class | TP n | TP exp | proof gap | market close net | live permission |
|---|---|---:|---:|---:|---:|---|
| `EUR_USD|SHORT|BREAKOUT_FAILURE` | `HARVEST_PROOF_FLOOR_REACHED_EVIDENCE_ONLY` | 20 | 643.2912 | 0 | -7636.3 | `False` |
| `EUR_USD|LONG|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_TP_PROVEN` | 20 | 591.5 | 0 | -15091.7 | `False` |
| `GBP_USD|LONG|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_THIN_SAMPLE` | 10 | 523.3 | 10 | -22478.7 | `False` |
| `EUR_USD|SHORT|RANGE_ROTATION` | `HARVEST_POSITIVE_THIN_SAMPLE` | 10 | 399.6 | 10 | -2151.0 | `False` |
| `AUD_JPY|SHORT|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_THIN_SAMPLE` | 6 | 992.7 | 14 | -3016.3 | `False` |
| `EUR_USD|SHORT|TREND_CONTINUATION` | `HARVEST_POSITIVE_THIN_SAMPLE` | 5 | 282.7 | 15 | -1202.1 | `False` |
| `GBP_USD|LONG|RANGE_ROTATION` | `HARVEST_POSITIVE_THIN_SAMPLE` | 3 | 331.6 | 17 | None | `False` |
| `EUR_USD|LONG|RANGE_ROTATION` | `HARVEST_POSITIVE_THIN_SAMPLE` | 3 | 164.0 | 17 | -2633.6 | `False` |
| `AUD_CAD|LONG|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_THIN_SAMPLE` | 2 | 482.4 | 18 | None | `False` |
| `EUR_USD|LONG|TREND_CONTINUATION` | `HARVEST_POSITIVE_THIN_SAMPLE` | 2 | 155.9 | 18 | -3307.4 | `False` |
| `AUD_JPY|LONG|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_THIN_SAMPLE` | 2 | 150.1 | 18 | -3077.3 | `False` |
| `GBP_USD|SHORT|BREAKOUT_FAILURE` | `HARVEST_POSITIVE_THIN_SAMPLE` | 2 | 21.7 | 18 | None | `False` |

## RUNNER Candidates

| shape | n | exp | avg win | avg loss | tail cases | tail JPY | live permission |
|---|---:|---:|---:|---:|---:|---:|---|

## Partial TP + Runner

| shape | TP exp | runner cases | runner tail JPY | market close leak |
|---|---:|---:|---:|---:|
| `EUR_USD|SHORT|BREAKOUT_FAILURE` | 643.2912 | 1 | 2099.4487 | -7636.3 |
| `GBP_USD|LONG|BREAKOUT_FAILURE` | 523.3 | 0 | 0.0 | -22478.7 |
| `EUR_USD|LONG|BREAKOUT_FAILURE` | 591.5 | 0 | 0.0 | -15091.7 |
| `EUR_USD|LONG|TREND_CONTINUATION` | 155.9 | 0 | 0.0 | -3307.4 |
| `AUD_JPY|LONG|BREAKOUT_FAILURE` | 150.1 | 0 | 0.0 | -3077.3 |
| `AUD_JPY|SHORT|BREAKOUT_FAILURE` | 992.7 | 0 | 0.0 | -3016.3 |
| `EUR_USD|LONG|RANGE_ROTATION` | 164.0 | 0 | 0.0 | -2633.6 |
| `EUR_USD|SHORT|RANGE_ROTATION` | 399.6 | 0 | 0.0 | -2151.0 |
| `EUR_USD|SHORT|TREND_CONTINUATION` | 282.7 | 0 | 0.0 | -1202.1 |

## NO_TRADE Shapes

| shape | reason | net | live permission |
|---|---|---:|---|
| `AUD_JPY|SHORT|RANGE_ROTATION` | `REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE` | -3171.0 | `False` |
| `AUD_USD|LONG|RANGE_ROTATION` | `MONTH_SCALE_REPLAY_NEGATIVE` | -2690.7 | `False` |
| `NZD_CAD|SHORT|RANGE_ROTATION` | `MONTH_SCALE_REPLAY_NEGATIVE` | -2044.5 | `False` |
| `GBP_CHF|LONG|TREND_CONTINUATION` | `REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE` | -1937.5 | `False` |
| `GBP_USD|LONG|TREND_CONTINUATION` | `REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE` | -1857.3 | `False` |
| `AUD_USD|SHORT|RANGE_ROTATION` | `MONTH_SCALE_REPLAY_NEGATIVE` | -1705.7 | `False` |
| `EUR_JPY|SHORT|BREAKOUT_FAILURE` | `REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE` | -1414.2 | `False` |
| `NZD_USD|LONG|RANGE_ROTATION` | `MONTH_SCALE_REPLAY_NEGATIVE` | -1380.8 | `False` |
| `EUR_CHF|LONG|TREND_CONTINUATION` | `MONTH_SCALE_REPLAY_NEGATIVE` | -1044.5 | `False` |
| `GBP_USD|SHORT|RANGE_ROTATION` | `MONTH_SCALE_REPLAY_NEGATIVE` | -971.0 | `False` |
| `AUD_CHF|LONG|BREAKOUT_FAILURE` | `REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE` | -897.9 | `False` |
| `EUR_GBP|SHORT|BREAKOUT_FAILURE` | `REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE` | -878.3 | `False` |

## MFE / MAE Summary

```json
{
  "execution_timing": {
    "canceled_order_mfe": {
      "audited": 4,
      "entry_touched": 2,
      "estimated_missed_mfe_jpy": 714.3022,
      "tp_touched": 1
    },
    "loss_close_mfe": {
      "audited": 8,
      "estimated_mfe_jpy": 606.7484,
      "had_positive_mfe": 5
    },
    "post_close_runner": {
      "audited": 6,
      "estimated_followthrough_jpy": 3150.6487,
      "left_runner_upside": 2
    },
    "summary": {
      "avg_decision_lag_minutes_after_first_positive": 93.57,
      "canceled_entry_touched_after_cancel": 2,
      "canceled_entry_touched_after_cancel_rate": 0.5,
      "canceled_estimated_missed_mfe_jpy": 714.3022,
      "canceled_orders_audited": 4,
      "canceled_positive_after_cancel_entry": 2,
      "canceled_positive_after_cancel_entry_rate": 0.5,
      "canceled_tp_touched_after_cancel": 1,
      "canceled_tp_touched_after_cancel_rate": 0.25,
      "historical_pre_repair_loss_closes_profit_capture_missed": 1,
      "historical_pre_repair_loss_closes_repair_replay_triggered": 1,
      "loss_close_actual_pl_jpy": -8677.3834,
      "loss_close_counterfactual_profit_capture_delta_jpy": 446.04,
      "loss_close_counterfactual_profit_capture_jpy": 105.84,
      "loss_close_counterfactual_profit_capture_pl_jpy": -8231.3434,
      "loss_close_estimated_capture_gap_jpy": 214.2,
      "loss_close_estimated_mfe_jpy": 606.7484,
      "loss_close_repair_replay_actual_pl_jpy": -8677.3834,
      "loss_close_repair_replay_block_reasons": {},
      "loss_close_repair_replay_counterfactual_pl_jpy": -8211.1834,
      "loss_close_repair_replay_delta_jpy": 466.2,
      "loss_close_repair_replay_profit_capture_jpy": 126.0,
      "loss_closes_audited": 8,
      "loss_closes_had_positive_mfe": 5,
      "loss_closes_had_positive_mfe_rate": 0.625,
      "loss_closes_profit_capture_missed": 1,
      "loss_closes_profit_capture_missed_rate": 0.125,
      "loss_closes_repair_replay_triggered": 1,
      "loss_closes_repair_replay_triggered_rate": 0.125,
      "loss_closes_tp_touched_before_close": 0,
      "loss_closes_tp_touched_before_close_rate": 0.0,
      "loss_market_closes_audited": 1,
      "loss_market_closes_contained_risk": 0,
      "loss_market_closes_may_have_been_premature": 1,
      "market_close_estimated_avoided_adverse_jpy": 7808.0,
      "market_close_estimated_followthrough_jpy": 5158.3377,
      "market_closes_audited": 6,
      "market_closes_post_close_adverse": 3,
      "market_closes_post_close_adverse_rate": 0.5,
      "market_closes_post_close_continued": 3,
      "market_closes_post_close_continued_rate": 0.5,
      "market_closes_sl_touched_after_close": 2,
      "market_closes_sl_touched_after_close_rate": 0.3333,
      "market_closes_tp_touched_after_close": 3,
      "market_closes_tp_touched_after_close_rate": 0.5,
      "max_decision_lag_minutes_after_first_positive": 300.28,
      "post_repair_live_evidence_loss_closes_audited": 5,
      "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
      "post_repair_live_evidence_loss_closes_profit_capture_missed_rate": 0.0,
      "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
      "post_repair_live_evidence_loss_closes_repair_replay_triggered_rate": 0.0,
      "post_repair_live_evidence_stop_loss_closes_profit_capture_missed": 0,
      "pre_repair_historical_loss_closes_audited": 3,
      "pre_repair_historical_loss_closes_profit_capture_missed": 1,
      "pre_repair_historical_loss_closes_profit_capture_missed_rate": 0.3333,
      "pre_repair_historical_loss_closes_repair_replay_triggered": 1,
      "pre_repair_historical_loss_closes_repair_replay_triggered_rate": 0.3333,
      "profit_market_closes_audited": 5,
      "profit_market_closes_avoided_giveback": 3,
      "profit_market_closes_left_runner_upside": 2,
      "stop_loss_closes_profit_capture_missed": 1,
      "top_entry_quality_residual_groups": [
        {
          "actual_pl_jpy": -2690.6967,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -2690.6967,
              "lane_id": "range_trader:AUD_USD:LONG:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -2690.6967,
              "repair_replay_triggered": false,
              "trade_id": "472952"
            }
          ],
          "exit_reason": "STOP_LOSS_ORDER",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "AUD_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -2690.6967,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
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
          "actual_pl_jpy": -844.2,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -844.2,
              "lane_id": "range_trader:EUR_JPY:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -844.2,
              "repair_replay_triggered": false,
              "trade_id": "472903"
            }
          ],
          "exit_reason": "STOP_LOSS_ORDER",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "EUR_JPY",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -844.2,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        },
        {
          "actual_pl_jpy": -744.8,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1,
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -464.0,
              "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -464.0,
              "repair_replay_triggered": false,
              "trade_id": "472900"
            },
            {
              "actual_pl_jpy": -280.8,
              "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -280.8,
              "repair_replay_triggered": false,
              "trade_id": "472775"
            }
          ],
          "exit_reason": "STOP_LOSS_ORDER",
          "loss_closes": 2,
          "method": "RANGE_ROTATION",
          "pair": "USD_JPY",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -744.8,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        }
      ],
      "top_entry_quality_residual_method_rollups": [
        {
          "actual_pl_jpy": -8337.1834,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 4,
            "NO_PROFIT_CANDIDATE": 3
          },
          "examples": [
            {
              "actual_pl_jpy": -2690.6967,
              "lane_id": "range_trader:AUD_USD:LONG:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "AUD_USD",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -2690.6967,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472952"
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
              "actual_pl_jpy": -1380.8008,
              "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "NZD_USD",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1380.8008,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472743"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 1,
            "STOP_LOSS_ORDER": 6
          },
          "group_count": 6,
          "loss_closes": 7,
          "method": "RANGE_ROTATION",
          "pair_count": 5,
          "pairs": [
            "AUD_USD",
            "EUR_JPY",
            "GBP_USD",
            "NZD_USD",
            "USD_JPY"
          ],
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -8337.1834,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side_count": 2,
          "sides": [
            "LONG",
            "SHORT"
          ]
        }
      ],
      "top_repair_replay_residual_groups": [
        {
          "actual_pl_jpy": -2690.6967,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -2690.6967,
              "lane_id": "range_trader:AUD_USD:LONG:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -2690.6967,
              "repair_replay_triggered": false,
              "trade_id": "472952"
            }
          ],
          "exit_reason": "STOP_LOSS_ORDER",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "AUD_USD",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -2690.6967,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "LONG"
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
          "actual_pl_jpy": -844.2,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -844.2,
              "lane_id": "range_trader:EUR_JPY:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -844.2,
              "repair_replay_triggered": false,
              "trade_id": "472903"
            }
          ],
          "exit_reason": "STOP_LOSS_ORDER",
          "loss_closes": 1,
          "method": "RANGE_ROTATION",
          "pair": "EUR_JPY",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -844.2,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        },
        {
          "actual_pl_jpy": -744.8,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 1,
            "NO_PROFIT_CANDIDATE": 1
          },
          "examples": [
            {
              "actual_pl_jpy": -464.0,
              "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -464.0,
              "repair_replay_triggered": false,
              "trade_id": "472900"
            },
            {
              "actual_pl_jpy": -280.8,
              "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -280.8,
              "repair_replay_triggered": false,
              "trade_id": "472775"
            }
          ],
          "exit_reason": "STOP_LOSS_ORDER",
          "loss_closes": 2,
          "method": "RANGE_ROTATION",
          "pair": "USD_JPY",
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -744.8,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side": "SHORT"
        }
      ],
      "top_repair_replay_residual_method_rollups": [
        {
          "actual_pl_jpy": -8337.1834,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 4,
            "NO_PROFIT_CANDIDATE": 3
          },
          "examples": [
            {
              "actual_pl_jpy": -2690.6967,
              "lane_id": "range_trader:AUD_USD:LONG:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "AUD_USD",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -2690.6967,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472952"
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
              "actual_pl_jpy": -1380.8008,
              "lane_id": "range_trader:NZD_USD:LONG:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "NZD_USD",
              "repair_replay_block_reason": "NO_PROFIT_CANDIDATE",
              "repair_replay_pl_jpy": -1380.8008,
              "repair_replay_triggered": false,
              "side": "LONG",
              "trade_id": "472743"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 1,
            "STOP_LOSS_ORDER": 6
          },
          "group_count": 6,
          "loss_closes": 7,
          "method": "RANGE_ROTATION",
          "pair_count": 5,
          "pairs": [
            "AUD_USD",
            "EUR_JPY",
            "GBP_USD",
            "NZD_USD",
            "USD_JPY"
          ],
          "repair_replay_delta_jpy": 0.0,
          "repair_replay_pl_jpy": -8337.1834,
          "repair_replay_triggered": 0,
          "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
          "side_count": 2,
          "sides": [
            "LONG",
            "SHORT"
          ]
        }
      ],
      "top_tp_progress_repair_residual_groups": [],
      "top_tp_progress_repair_residual_method_rollups": [],
      "tp_progress_repair_live_evidence_boundary_reason": "Full TP-progress production replay contract deployed; 2026-06-22T09:35:39Z was TP-progress banking only.",
      "tp_progress_repair_live_evidence_boundary_utc": "2026-06-22T17:54:26Z",
      "tp_progress_repair_live_evidence_status": "POST_REPAIR_REPLAY_CLEAN"
    }
  },
  "order_intent_bidask_replay": {
    "by_family": {
      "BREAKOUT_FAILURE": {
        "avg_final_pips": -5.1515,
        "avg_mae_pips": 14.3483,
        "avg_mfe_pips": 5.394,
        "blocks_live_support_packets": 21,
        "negative_avg_final_packets": 21,
        "packets": 45,
        "sample_sum": 23697
      },
      "RANGE_ROTATION": {
        "avg_final_pips": -3.7901,
        "avg_mae_pips": 11.745,
        "avg_mfe_pips": 5.3146,
        "blocks_live_support_packets": 33,
        "negative_avg_final_packets": 33,
        "packets": 70,
        "sample_sum": 29238
      },
      "TREND_CONTINUATION": {
        "avg_final_pips": -4.817,
        "avg_mae_pips": 14.3892,
        "avg_mfe_pips": 5.4041,
        "blocks_live_support_packets": 17,
        "negative_avg_final_packets": 17,
        "packets": 36,
        "sample_sum": 17924
      }
    },
    "by_pair": {
      "AUD_CHF": {
        "avg_final_pips": -2.0411,
        "avg_mae_pips": 5.7157,
        "avg_mfe_pips": 2.2821,
        "blocks_live_support_packets": 2,
        "negative_avg_final_packets": 2,
        "packets": 4,
        "sample_sum": 1512
      },
      "AUD_JPY": {
        "avg_final_pips": -3.6274,
        "avg_mae_pips": 11.4669,
        "avg_mfe_pips": 5.5591,
        "blocks_live_support_packets": 6,
        "negative_avg_final_packets": 6,
        "packets": 12,
        "sample_sum": 7936
      },
      "AUD_USD": {
        "avg_final_pips": -2.1798,
        "avg_mae_pips": 9.1379,
        "avg_mfe_pips": 5.7485,
        "blocks_live_support_packets": 4,
        "negative_avg_final_packets": 4,
        "packets": 8,
        "sample_sum": 2734
      },
      "CHF_JPY": {
        "avg_final_pips": -6.3458,
        "avg_mae_pips": 16.1307,
        "avg_mfe_pips": 5.0983,
        "blocks_live_support_packets": 7,
        "negative_avg_final_packets": 7,
        "packets": 14,
        "sample_sum": 7889
      },
      "EUR_AUD": {
        "avg_final_pips": -5.3594,
        "avg_mae_pips": 17.4313,
        "avg_mfe_pips": 10.7586,
        "blocks_live_support_packets": 2,
        "negative_avg_final_packets": 2,
        "packets": 4,
        "sample_sum": 1988
      },
      "EUR_CHF": {
        "avg_final_pips": -1.9732,
        "avg_mae_pips": 5.0864,
        "avg_mfe_pips": 2.1696,
        "blocks_live_support_packets": 2,
        "negative_avg_final_packets": 2,
        "packets": 4,
        "sample_sum": 954
      },
      "EUR_JPY": {
        "avg_final_pips": -4.0908,
        "avg_mae_pips": 11.4948,
        "avg_mfe_pips": 4.9026,
        "blocks_live_support_packets": 14,
        "negative_avg_final_packets": 14,
        "packets": 28,
        "sample_sum": 18991
      },
      "EUR_USD": {
        "avg_final_pips": -2.8936,
        "avg_mae_pips": 7.6197,
        "avg_mfe_pips": 3.2954,
        "blocks_live_support_packets": 4,
        "negative_avg_final_packets": 4,
        "packets": 15,
        "sample_sum": 5532
      },
      "GBP_AUD": {
        "avg_final_pips": -5.7043,
        "avg_mae_pips": 18.0423,
        "avg_mfe_pips": 6.7586,
        "blocks_live_support_packets": 7,
        "negative_avg_final_packets": 7,
        "packets": 14,
        "sample_sum": 3724
      },
      "GBP_JPY": {
        "avg_final_pips": -4.0513,
        "avg_mae_pips": 16.7497,
        "avg_mfe_pips": 8.4708,
        "blocks_live_support_packets": 4,
        "negative_avg_final_packets": 4,
        "packets": 8,
        "sample_sum": 3142
      },
      "GBP_NZD": {
        "avg_final_pips": -7.7174,
        "avg_mae_pips": 22.3776,
        "avg_mfe_pips": 7.1256,
        "blocks_live_support_packets": 7,
        "negative_avg_final_packets": 7,
        "packets": 14,
        "sample_sum": 5257
      },
      "GBP_USD": {
        "avg_final_pips": -5.9085,
        "avg_mae_pips": 11.9463,
        "avg_mfe_pips": 3.6146,
        "blocks_live_support_packets": 4,
        "negative_avg_final_packets": 4,
        "packets": 10,
        "sample_sum": 5680
      },
      "NZD_USD": {
        "avg_final_pips": -3.6204,
        "avg_mae_pips": 7.2687,
        "avg_mfe_pips": 2.1251,
        "blocks_live_support_packets": 2,
        "negative_avg_final_packets": 2,
        "packets": 4,
        "sample_sum": 1098
      },
      "USD_JPY": {
        "avg_final_pips": -1.9361,
        "avg_mae_pips": 9.1855,
        "avg_mfe_pips": 4.4112,
        "blocks_live_support_packets": 6,
        "negative_avg_final_packets": 6,
        "packets": 12,
        "sample_sum": 4422
      }
    },
    "by_session": {
      "LONDON_KILLZONE": {
        "avg_final_pips": -4.4387,
        "avg_mae_pips": 13.1481,
        "avg_mfe_pips": 5.3595,
        "blocks_live_support_packets": 71,
        "negative_avg_final_packets": 71,
        "packets": 151,
        "sample_sum": 70859
      }
    },
    "overall": {
      "avg_final_pips": -4.4387,
      "avg_mae_pips": 13.1481,
      "avg_mfe_pips": 5.3595,
      "blocks_live_support_packets": 71,
      "negative_avg_final_packets": 71,
      "packets": 151,
      "sample_sum": 70859
    },
    "top_negative_packets": [
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -7.7174,
        "avg_mae_pips": 22.3776,
        "avg_mfe_pips": 7.1256,
        "lane_id": "range_trader:GBP_NZD:LONG:RANGE_ROTATION",
        "method": "RANGE_ROTATION",
        "name": "GBP_NZD_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "GBP_NZD",
        "samples": 751,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -7.7174,
        "avg_mae_pips": 22.3776,
        "avg_mfe_pips": 7.1256,
        "lane_id": "range_trader:GBP_NZD:LONG:RANGE_ROTATION:MARKET",
        "method": "RANGE_ROTATION",
        "name": "GBP_NZD_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "GBP_NZD",
        "samples": 751,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -7.7174,
        "avg_mae_pips": 22.3776,
        "avg_mfe_pips": 7.1256,
        "lane_id": "failure_trader:GBP_NZD:LONG:BREAKOUT_FAILURE:LIMIT",
        "method": "BREAKOUT_FAILURE",
        "name": "GBP_NZD_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "GBP_NZD",
        "samples": 751,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -7.7174,
        "avg_mae_pips": 22.3776,
        "avg_mfe_pips": 7.1256,
        "lane_id": "failure_trader:GBP_NZD:LONG:BREAKOUT_FAILURE",
        "method": "BREAKOUT_FAILURE",
        "name": "GBP_NZD_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "GBP_NZD",
        "samples": 751,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -7.7174,
        "avg_mae_pips": 22.3776,
        "avg_mfe_pips": 7.1256,
        "lane_id": "failure_trader:GBP_NZD:LONG:BREAKOUT_FAILURE:MARKET",
        "method": "BREAKOUT_FAILURE",
        "name": "GBP_NZD_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "GBP_NZD",
        "samples": 751,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -7.7174,
        "avg_mae_pips": 22.3776,
        "avg_mfe_pips": 7.1256,
        "lane_id": "trend_trader:GBP_NZD:LONG:TREND_CONTINUATION",
        "method": "TREND_CONTINUATION",
        "name": "GBP_NZD_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "GBP_NZD",
        "samples": 751,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -7.7174,
        "avg_mae_pips": 22.3776,
        "avg_mfe_pips": 7.1256,
        "lane_id": "trend_trader:GBP_NZD:LONG:TREND_CONTINUATION:MARKET",
        "method": "TREND_CONTINUATION",
        "name": "GBP_NZD_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "GBP_NZD",
        "samples": 751,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -6.3458,
        "avg_mae_pips": 16.1307,
        "avg_mfe_pips": 5.0983,
        "lane_id": "range_trader:CHF_JPY:LONG:RANGE_ROTATION",
        "method": "RANGE_ROTATION",
        "name": "CHF_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "CHF_JPY",
        "samples": 1127,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -6.3458,
        "avg_mae_pips": 16.1307,
        "avg_mfe_pips": 5.0983,
        "lane_id": "range_trader:CHF_JPY:LONG:RANGE_ROTATION:MARKET",
        "method": "RANGE_ROTATION",
        "name": "CHF_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "CHF_JPY",
        "samples": 1127,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -6.3458,
        "avg_mae_pips": 16.1307,
        "avg_mfe_pips": 5.0983,
        "lane_id": "failure_trader:CHF_JPY:LONG:BREAKOUT_FAILURE:LIMIT",
        "method": "BREAKOUT_FAILURE",
        "name": "CHF_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "CHF_JPY",
        "samples": 1127,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -6.3458,
        "avg_mae_pips": 16.1307,
        "avg_mfe_pips": 5.0983,
        "lane_id": "failure_trader:CHF_JPY:LONG:BREAKOUT_FAILURE",
        "method": "BREAKOUT_FAILURE",
        "name": "CHF_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "CHF_JPY",
        "samples": 1127,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      },
      {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "avg_final_pips": -6.3458,
        "avg_mae_pips": 16.1307,
        "avg_mfe_pips": 5.0983,
        "lane_id": "failure_trader:CHF_JPY:LONG:BREAKOUT_FAILURE:MARKET",
        "method": "BREAKOUT_FAILURE",
        "name": "CHF_JPY_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
        "pair": "CHF_JPY",
        "samples": 1127,
        "session": "LONDON_KILLZONE",
        "side": "LONG"
      }
    ]
  }
}
```

## Recommendations

- `NO_LIVE_PERMISSION_FROM_DIAGNOSIS`: keep diagnosis read-only; do not send, cancel, close, modify launchd, or relax gates
- `PROOF_QUEUE_COUNT_IS_NOT_PERMISSION`: treat proof_queue_count=0 or any firepower estimate as evidence status only, never as a live permission
- `NO_4X_DEFICIT_LOT_BACKSOLVE`: do not derive lot size from remaining_to_4x; this diagnosis contains no unit sizing output
- `NEGATIVE_EXPECTANCY_VISIBLE`: keep fresh entries blocked except exact TP-proven HARVEST shapes that pass every existing current gate
- `MONTH_SCALE_REPLAY_BLOCKS_PROMOTION`: do not promote matching family/pair/session while month-scale replay remains negative
- `HARVEST_BASE_KEEP_NARROW`: preserve positive broker-TP HARVEST shapes, but only as exact-shape, spread-included evidence candidates
- `PARTIAL_TP_RUNNER_REPLAY_NEXT`: simulate TP1 bank plus small runner leg for top HARVEST shapes before changing live behavior
- `RUNNER_ONLY_UNPROVED`: do not switch to RUNNER-only; require average win greater than average loss plus MFE-extension evidence
- `NO_TRADE_SHAPES_STAY_BLOCKED`: keep spread/slippage-included negative and month-scale-negative shapes as NO_TRADE until replay clears
- `FOUR_X_PAYOFF_SHAPE_VERDICT`: 4x requires a mixed shape with HARVEST as the accounting base and only evidence-scoped runners. RUNNER-only promotion is not supported while market-close leakage and month-scale replay remain negative.

## Next Evidence Actions

- `REFRESH_MONTH_SCALE_TIMING_AUDIT` - `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80`
- `REFRESH_CAPTURE_ECONOMICS_AFTER_LEDGER_SYNC` - `PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics`
- `RECHECK_MONTH_SCALE_RESIDUALS` - `PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance`
- `COLLECT_EXACT_HARVEST_TP_PROOF_GAPS`
- `NO_TRADE_ESCAPE_CONDITION`

## Safety

- No live orders were sent.
- No orders were canceled.
- No positions were closed.
- No launchd state was changed.
- No gate was relaxed.
- Negative expectancy and month-scale replay blockers remain visible.

# Profitability Acceptance

- Status: `PROFITABILITY_ACCEPTANCE_BLOCKED`
- Generated: `2026-07-06T01:58:50.390861+00:00`
- Findings: `9`

## Findings

| Priority | Code | Message |
| --- | --- | --- |
| `P0` | `SELF_IMPROVEMENT_P0_PRESENT` | self-improvement audit still has 1 P0 finding(s) |
| `P0` | `NEGATIVE_EXPECTANCY_ACTIVE` | capture economics is still NEGATIVE_EXPECTANCY |
| `P0` | `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE` | 1 TP-proven segment(s) are still net-damaged by MARKET_ORDER_TRADE_CLOSE leakage |
| `P0` | `MARKET_CLOSE_LEAK_FAMILY_BLOCKED` | EUR_USD LONG BREAKOUT_FAILURE system-gateway MARKET_ORDER_TRADE_CLOSE loss family remains blocked from fresh-entry and repair-exit live routing until the exact exception proof stack exists. |
| `P0` | `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE` | 30-day OANDA candle replay says the current TP-progress repair improves loss-side closes, but the replayed loss-close P/L is still net negative |
| `P1` | `PROJECTION_HEADLINE_PRECISION_ECONOMIC_GAP` | 5 projection bucket(s) clear headline precision but fail economic precision after TIMEOUT/no-touch penalties |
| `P1` | `BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN` | S5 bid/ask replay has price truth for loaded samples, but pair-direction sample coverage is too thin to claim all-currency high-turn readiness |
| `P1` | `NO_LIVE_READY_TARGET_COVERAGE` | daily target is open but there are no LIVE_READY lanes |
| `P1` | `REPAIR_FRONTIER_BLOCKED` | 11 repair-mode candidate(s) exist, but none currently clear live gates |

## Metrics

```json
{
  "bidask_replay_rules": {
    "adoption_summary": {
      "has_live_grade_support": true,
      "has_rank_only_support": true,
      "live_grade_support_rules": 12,
      "negative_block_rules": 52,
      "rank_only_blocker_counts": {
        "DAILY_PNL_UNSTABLE": 4,
        "DAILY_SAMPLE_CONCENTRATED": 1,
        "NEEDS_HIGHER_POSITIVE_DAY_RATE": 4,
        "NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION": 1
      },
      "rank_only_support_rules": 5
    },
    "contrarian_edge_rules": 17,
    "daily_stability_requirements": {
      "max_daily_sample_share": 0.7,
      "min_active_days": 3,
      "min_positive_day_rate": 0.6666666666666666
    },
    "daily_stable_contrarian_edge_rules": 12,
    "daily_stable_edge_rules": 0,
    "daily_stable_support_rules": 12,
    "edge_rules": 0,
    "forecast_sample_collection_required": true,
    "forecast_sample_coverage_summary": {
      "min_active_days_for_daily_stability": 3,
      "min_directional_samples_for_precision_rule": 30,
      "pair_count": 28,
      "pair_coverage": [
        {
          "evaluated_samples": 3004,
          "forecast_samples": 3109,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_USD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 105
        },
        {
          "evaluated_samples": 2713,
          "forecast_samples": 2747,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 34
        },
        {
          "evaluated_samples": 2624,
          "forecast_samples": 2727,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_USD",
          "pending_future_truth_samples": 3,
          "unscorable_no_market_samples": 100
        },
        {
          "evaluated_samples": 2607,
          "forecast_samples": 2713,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_JPY",
          "pending_future_truth_samples": 3,
          "unscorable_no_market_samples": 103
        },
        {
          "evaluated_samples": 1600,
          "forecast_samples": 1631,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "CHF_JPY",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 29
        },
        {
          "evaluated_samples": 1571,
          "forecast_samples": 1600,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_JPY",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 28
        },
        {
          "evaluated_samples": 1540,
          "forecast_samples": 1675,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_CAD",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 134
        },
        {
          "evaluated_samples": 1538,
          "forecast_samples": 1568,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 30
        },
        {
          "evaluated_samples": 1510,
          "forecast_samples": 1535,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 25
        },
        {
          "evaluated_samples": 1497,
          "forecast_samples": 1506,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_GBP",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 9
        },
        {
          "evaluated_samples": 1444,
          "forecast_samples": 1510,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 66
        },
        {
          "evaluated_samples": 1406,
          "forecast_samples": 1435,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_CHF",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 28
        },
        {
          "evaluated_samples": 1393,
          "forecast_samples": 1400,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "CAD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 7
        },
        {
          "evaluated_samples": 1382,
          "forecast_samples": 1430,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_AUD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 48
        },
        {
          "evaluated_samples": 1368,
          "forecast_samples": 1388,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_JPY",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 19
        },
        {
          "evaluated_samples": 1367,
          "forecast_samples": 1497,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_USD",
          "pending_future_truth_samples": 3,
          "unscorable_no_market_samples": 127
        },
        {
          "evaluated_samples": 1359,
          "forecast_samples": 1420,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_USD",
          "pending_future_truth_samples": 6,
          "unscorable_no_market_samples": 55
        },
        {
          "evaluated_samples": 1358,
          "forecast_samples": 1429,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_AUD",
          "pending_future_truth_samples": 24,
          "unscorable_no_market_samples": 47
        },
        {
          "evaluated_samples": 1345,
          "forecast_samples": 1360,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_NZD",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 14
        },
        {
          "evaluated_samples": 1306,
          "forecast_samples": 1341,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 35
        },
        {
          "evaluated_samples": 1304,
          "forecast_samples": 1332,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 28
        },
        {
          "evaluated_samples": 1285,
          "forecast_samples": 1296,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CHF",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 9
        },
        {
          "evaluated_samples": 1216,
          "forecast_samples": 1232,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 16
        },
        {
          "evaluated_samples": 1176,
          "forecast_samples": 1224,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_NZD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 48
        },
        {
          "evaluated_samples": 1173,
          "forecast_samples": 1187,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_NZD",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 13
        },
        {
          "evaluated_samples": 1163,
          "forecast_samples": 1287,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "CAD_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 124
        },
        {
          "evaluated_samples": 1145,
          "forecast_samples": 1182,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_CAD",
          "pending_future_truth_samples": 3,
          "unscorable_no_market_samples": 34
        },
        {
          "evaluated_samples": 1120,
          "forecast_samples": 1155,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_CAD",
          "pending_future_truth_samples": 7,
          "unscorable_no_market_samples": 28
        }
      ],
      "pair_coverage_count": 28,
      "pair_coverage_examples": [
        {
          "evaluated_samples": 3004,
          "forecast_samples": 3109,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_USD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 105
        },
        {
          "evaluated_samples": 2713,
          "forecast_samples": 2747,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 34
        },
        {
          "evaluated_samples": 2624,
          "forecast_samples": 2727,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_USD",
          "pending_future_truth_samples": 3,
          "unscorable_no_market_samples": 100
        },
        {
          "evaluated_samples": 2607,
          "forecast_samples": 2713,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_JPY",
          "pending_future_truth_samples": 3,
          "unscorable_no_market_samples": 103
        },
        {
          "evaluated_samples": 1600,
          "forecast_samples": 1631,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "CHF_JPY",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 29
        },
        {
          "evaluated_samples": 1571,
          "forecast_samples": 1600,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_JPY",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 28
        },
        {
          "evaluated_samples": 1540,
          "forecast_samples": 1675,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_CAD",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 134
        },
        {
          "evaluated_samples": 1538,
          "forecast_samples": 1568,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 30
        },
        {
          "evaluated_samples": 1510,
          "forecast_samples": 1535,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 25
        },
        {
          "evaluated_samples": 1497,
          "forecast_samples": 1506,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_GBP",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 9
        },
        {
          "evaluated_samples": 1444,
          "forecast_samples": 1510,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 66
        },
        {
          "evaluated_samples": 1406,
          "forecast_samples": 1435,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_CHF",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 28
        },
        {
          "evaluated_samples": 1393,
          "forecast_samples": 1400,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "CAD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 7
        },
        {
          "evaluated_samples": 1382,
          "forecast_samples": 1430,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_AUD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 48
        },
        {
          "evaluated_samples": 1368,
          "forecast_samples": 1388,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_JPY",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 19
        },
        {
          "evaluated_samples": 1367,
          "forecast_samples": 1497,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_USD",
          "pending_future_truth_samples": 3,
          "unscorable_no_market_samples": 127
        },
        {
          "evaluated_samples": 1359,
          "forecast_samples": 1420,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_USD",
          "pending_future_truth_samples": 6,
          "unscorable_no_market_samples": 55
        },
        {
          "evaluated_samples": 1358,
          "forecast_samples": 1429,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_AUD",
          "pending_future_truth_samples": 24,
          "unscorable_no_market_samples": 47
        },
        {
          "evaluated_samples": 1345,
          "forecast_samples": 1360,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_NZD",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 14
        },
        {
          "evaluated_samples": 1306,
          "forecast_samples": 1341,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 35
        },
        {
          "evaluated_samples": 1304,
          "forecast_samples": 1332,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 28
        },
        {
          "evaluated_samples": 1285,
          "forecast_samples": 1296,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CHF",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 9
        },
        {
          "evaluated_samples": 1216,
          "forecast_samples": 1232,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 16
        },
        {
          "evaluated_samples": 1176,
          "forecast_samples": 1224,
          "missing_evaluated_samples_to_min_directional": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_NZD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 48
        }
      ],
      "pair_coverage_examples_omitted": 4,
      "pair_direction_count": 56,
      "pending_future_truth_samples": 59,
      "under_sampled_gap_reason_counts": {
        "NO_MARKET_SESSION_UNSCORABLE": 52,
        "PENDING_FUTURE_TRUTH_WINDOW": 18
      },
      "under_sampled_pair_direction_detail_count": 53,
      "under_sampled_pair_direction_details": [
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 31,
          "evaluated_samples": 620,
          "forecast_active_days": 32,
          "forecast_samples": 637,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 17
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 32,
          "evaluated_samples": 890,
          "forecast_active_days": 33,
          "forecast_samples": 898,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 8
        },
        {
          "coverage_gap_reasons": [
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 32,
          "evaluated_samples": 756,
          "forecast_active_days": 32,
          "forecast_samples": 758,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CHF",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 0
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 31,
          "evaluated_samples": 529,
          "forecast_active_days": 33,
          "forecast_samples": 538,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 9
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 39,
          "evaluated_samples": 1361,
          "forecast_active_days": 40,
          "forecast_samples": 1377,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_JPY",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 15
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 34,
          "evaluated_samples": 1246,
          "forecast_active_days": 35,
          "forecast_samples": 1336,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_JPY",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 88
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 25,
          "evaluated_samples": 488,
          "forecast_active_days": 26,
          "forecast_samples": 499,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_NZD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 11
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 34,
          "evaluated_samples": 857,
          "forecast_active_days": 34,
          "forecast_samples": 861,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_NZD",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 3
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 33,
          "evaluated_samples": 972,
          "forecast_active_days": 36,
          "forecast_samples": 1095,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_USD",
          "pending_future_truth_samples": 3,
          "unscorable_no_market_samples": 120
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 22,
          "evaluated_samples": 395,
          "forecast_active_days": 23,
          "forecast_samples": 402,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_USD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 7
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 29,
          "evaluated_samples": 547,
          "forecast_active_days": 29,
          "forecast_samples": 554,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "CAD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 7
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 30,
          "evaluated_samples": 790,
          "forecast_active_days": 33,
          "forecast_samples": 906,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "CAD_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 116
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 31,
          "evaluated_samples": 373,
          "forecast_active_days": 31,
          "forecast_samples": 381,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "CAD_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 8
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 25,
          "evaluated_samples": 473,
          "forecast_active_days": 25,
          "forecast_samples": 479,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "CHF_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 6
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 32,
          "evaluated_samples": 1127,
          "forecast_active_days": 34,
          "forecast_samples": 1152,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "CHF_JPY",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 23
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 25,
          "evaluated_samples": 388,
          "forecast_active_days": 27,
          "forecast_samples": 406,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_AUD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 18
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 32,
          "evaluated_samples": 994,
          "forecast_active_days": 33,
          "forecast_samples": 1024,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_AUD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 30
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 27,
          "evaluated_samples": 563,
          "forecast_active_days": 27,
          "forecast_samples": 581,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 18
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 31,
          "evaluated_samples": 743,
          "forecast_active_days": 31,
          "forecast_samples": 760,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 17
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 22,
          "evaluated_samples": 477,
          "forecast_active_days": 23,
          "forecast_samples": 485,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 8
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 29,
          "evaluated_samples": 1061,
          "forecast_active_days": 30,
          "forecast_samples": 1083,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 22
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 28,
          "evaluated_samples": 708,
          "forecast_active_days": 28,
          "forecast_samples": 716,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_GBP",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 8
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 27,
          "evaluated_samples": 789,
          "forecast_active_days": 27,
          "forecast_samples": 790,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_GBP",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 1
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 37,
          "evaluated_samples": 1147,
          "forecast_active_days": 38,
          "forecast_samples": 1157,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 10
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 33,
          "evaluated_samples": 1566,
          "forecast_active_days": 34,
          "forecast_samples": 1590,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 24
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 34,
          "evaluated_samples": 762,
          "forecast_active_days": 34,
          "forecast_samples": 776,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_NZD",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 13
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 35,
          "evaluated_samples": 1621,
          "forecast_active_days": 38,
          "forecast_samples": 1722,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_USD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 101
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 32,
          "evaluated_samples": 1383,
          "forecast_active_days": 32,
          "forecast_samples": 1387,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_USD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 4
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 34,
          "evaluated_samples": 532,
          "forecast_active_days": 34,
          "forecast_samples": 548,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_AUD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 16
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 33,
          "evaluated_samples": 826,
          "forecast_active_days": 33,
          "forecast_samples": 881,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_AUD",
          "pending_future_truth_samples": 24,
          "unscorable_no_market_samples": 31
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 32,
          "evaluated_samples": 734,
          "forecast_active_days": 34,
          "forecast_samples": 771,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_CAD",
          "pending_future_truth_samples": 3,
          "unscorable_no_market_samples": 34
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 27,
          "evaluated_samples": 447,
          "forecast_active_days": 27,
          "forecast_samples": 450,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 3
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 33,
          "evaluated_samples": 959,
          "forecast_active_days": 34,
          "forecast_samples": 985,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_CHF",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 25
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 31,
          "evaluated_samples": 657,
          "forecast_active_days": 31,
          "forecast_samples": 665,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_JPY",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 7
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 31,
          "evaluated_samples": 914,
          "forecast_active_days": 33,
          "forecast_samples": 935,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 21
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 26,
          "evaluated_samples": 425,
          "forecast_active_days": 27,
          "forecast_samples": 432,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_NZD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 7
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 36,
          "evaluated_samples": 751,
          "forecast_active_days": 37,
          "forecast_samples": 792,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_NZD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 41
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 37,
          "evaluated_samples": 1204,
          "forecast_active_days": 39,
          "forecast_samples": 1288,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_USD",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 83
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 33,
          "evaluated_samples": 1420,
          "forecast_active_days": 33,
          "forecast_samples": 1439,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "GBP_USD",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 17
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 30,
          "evaluated_samples": 492,
          "forecast_active_days": 31,
          "forecast_samples": 513,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 21
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 25,
          "evaluated_samples": 628,
          "forecast_active_days": 27,
          "forecast_samples": 642,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_CAD",
          "pending_future_truth_samples": 7,
          "unscorable_no_market_samples": 7
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 31,
          "evaluated_samples": 433,
          "forecast_active_days": 32,
          "forecast_samples": 437,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 4
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 30,
          "evaluated_samples": 783,
          "forecast_active_days": 32,
          "forecast_samples": 795,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 12
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 34,
          "evaluated_samples": 794,
          "forecast_active_days": 34,
          "forecast_samples": 810,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 16
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 31,
          "evaluated_samples": 510,
          "forecast_active_days": 32,
          "forecast_samples": 522,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 12
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 31,
          "evaluated_samples": 810,
          "forecast_active_days": 34,
          "forecast_samples": 862,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_USD",
          "pending_future_truth_samples": 4,
          "unscorable_no_market_samples": 48
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 25,
          "evaluated_samples": 549,
          "forecast_active_days": 26,
          "forecast_samples": 558,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "NZD_USD",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 7
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 25,
          "evaluated_samples": 290,
          "forecast_active_days": 25,
          "forecast_samples": 298,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_CAD",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 7
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 35,
          "evaluated_samples": 1250,
          "forecast_active_days": 38,
          "forecast_samples": 1377,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 127
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 27,
          "evaluated_samples": 459,
          "forecast_active_days": 28,
          "forecast_samples": 466,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 7
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 36,
          "evaluated_samples": 985,
          "forecast_active_days": 38,
          "forecast_samples": 1044,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 59
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 30,
          "evaluated_samples": 525,
          "forecast_active_days": 30,
          "forecast_samples": 526,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 1
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 37,
          "evaluated_samples": 843,
          "forecast_active_days": 39,
          "forecast_samples": 862,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "USD_JPY",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 18
        }
      ],
      "under_sampled_pair_direction_examples": [
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 31,
          "evaluated_samples": 620,
          "forecast_active_days": 32,
          "forecast_samples": 637,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 17
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 32,
          "evaluated_samples": 890,
          "forecast_active_days": 33,
          "forecast_samples": 898,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 8
        },
        {
          "coverage_gap_reasons": [
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 32,
          "evaluated_samples": 756,
          "forecast_active_days": 32,
          "forecast_samples": 758,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CHF",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 0
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 31,
          "evaluated_samples": 529,
          "forecast_active_days": 33,
          "forecast_samples": 538,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 9
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 39,
          "evaluated_samples": 1361,
          "forecast_active_days": 40,
          "forecast_samples": 1377,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_JPY",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 15
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 34,
          "evaluated_samples": 1246,
          "forecast_active_days": 35,
          "forecast_samples": 1336,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_JPY",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 88
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 25,
          "evaluated_samples": 488,
          "forecast_active_days": 26,
          "forecast_samples": 499,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_NZD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 11
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 34,
          "evaluated_samples": 857,
          "forecast_active_days": 34,
          "forecast_samples": 861,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_NZD",
          "pending_future_truth_samples": 1,
          "unscorable_no_market_samples": 3
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 33,
          "evaluated_samples": 972,
          "forecast_active_days": 36,
          "forecast_samples": 1095,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_USD",
          "pending_future_truth_samples": 3,
          "unscorable_no_market_samples": 120
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 22,
          "evaluated_samples": 395,
          "forecast_active_days": 23,
          "forecast_samples": 402,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "AUD_USD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 7
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 29,
          "evaluated_samples": 547,
          "forecast_active_days": 29,
          "forecast_samples": 554,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "CAD_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 7
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 30,
          "evaluated_samples": 790,
          "forecast_active_days": 33,
          "forecast_samples": 906,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "CAD_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 116
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 31,
          "evaluated_samples": 373,
          "forecast_active_days": 31,
          "forecast_samples": 381,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "CAD_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 8
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 25,
          "evaluated_samples": 473,
          "forecast_active_days": 25,
          "forecast_samples": 479,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "CHF_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 6
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE",
            "PENDING_FUTURE_TRUTH_WINDOW"
          ],
          "direction": "UP",
          "evaluated_active_days": 32,
          "evaluated_samples": 1127,
          "forecast_active_days": 34,
          "forecast_samples": 1152,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "CHF_JPY",
          "pending_future_truth_samples": 2,
          "unscorable_no_market_samples": 23
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 25,
          "evaluated_samples": 388,
          "forecast_active_days": 27,
          "forecast_samples": 406,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_AUD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 18
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 32,
          "evaluated_samples": 994,
          "forecast_active_days": 33,
          "forecast_samples": 1024,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_AUD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 30
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 27,
          "evaluated_samples": 563,
          "forecast_active_days": 27,
          "forecast_samples": 581,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 18
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 31,
          "evaluated_samples": 743,
          "forecast_active_days": 31,
          "forecast_samples": 760,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CAD",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 17
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 22,
          "evaluated_samples": 477,
          "forecast_active_days": 23,
          "forecast_samples": 485,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 8
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 29,
          "evaluated_samples": 1061,
          "forecast_active_days": 30,
          "forecast_samples": 1083,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_CHF",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 22
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 28,
          "evaluated_samples": 708,
          "forecast_active_days": 28,
          "forecast_samples": 716,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_GBP",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 8
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "UP",
          "evaluated_active_days": 27,
          "evaluated_samples": 789,
          "forecast_active_days": 27,
          "forecast_samples": 790,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_GBP",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 1
        },
        {
          "coverage_gap_reasons": [
            "NO_MARKET_SESSION_UNSCORABLE"
          ],
          "direction": "DOWN",
          "evaluated_active_days": 37,
          "evaluated_samples": 1147,
          "forecast_active_days": 38,
          "forecast_samples": 1157,
          "missing_active_days": 0,
          "missing_evaluated_samples": 0,
          "missing_price_truth_samples": 0,
          "pair": "EUR_JPY",
          "pending_future_truth_samples": 0,
          "unscorable_no_market_samples": 10
        }
      ],
      "under_sampled_pair_direction_examples_omitted": 29,
      "under_sampled_pair_directions": 53,
      "unscorable_no_market_samples": 1343
    },
    "generated_at_utc": "2026-07-03T14:52:18.653002Z",
    "history_dirs": [
      "logs/replay/oanda_history/20260703T072439Z",
      "logs/replay/oanda_history/20260703T080331Z",
      "logs/replay/oanda_history/20260703T120929Z",
      "logs/replay/oanda_history/20260703T123013Z",
      "logs/replay/oanda_history/20260703T134642Z",
      "logs/replay/oanda_history/20260703T135559Z",
      "logs/replay/oanda_history/20260703T142126Z",
      "logs/replay/oanda_history/20260703T142653Z",
      "logs/replay/oanda_history/20260703T143956Z"
    ],
    "history_fetch_command": null,
    "negative_rules": 52,
    "packaged_by": "scripts/package_bidask_replay_precision_rules.py",
    "path": "/Users/tossaki/App/QuantRabbit/src/quant_rabbit/bidask_replay_precision_rules.json",
    "price_truth_coverage": {
      "adoption_level": "PAIR_LOCAL_RANK_ONLY",
      "all_currency_sample_coverage_status": "UNDER_SAMPLED",
      "evaluated_rows": 43514,
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
      "under_sampled_pair_direction_count": 53,
      "under_sampled_pair_directions": [
        "AUD_CAD:DOWN",
        "AUD_CAD:UP",
        "AUD_CHF:DOWN",
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
        "EUR_JPY:DOWN"
      ],
      "warnings": [
        "FORECAST_ROWS_DURING_BROKER_NO_MARKET_WINDOW",
        "FORECAST_ROWS_WITH_PENDING_FUTURE_TRUTH_WINDOW"
      ]
    },
    "price_truth_fetch_required": false,
    "rank_only_contrarian_edge_rules": 5,
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
          "missing_positive_days_at_current_requirement": 1,
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
          "missing_positive_days_at_current_requirement": 2,
          "reasons": [
            "NEEDS_HIGHER_POSITIVE_DAY_RATE"
          ],
          "required_active_days": 3,
          "required_positive_day_rate": 0.6667,
          "required_positive_days_at_current_requirement": 3,
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
          "DAILY_PNL_UNSTABLE",
          "NEEDS_HIGHER_POSITIVE_DAY_RATE"
        ],
        "adoption_status": "RANK_ONLY_NOT_DAILY_STABLE",
        "daily_stability_gap": {
          "current_max_daily_sample_share": 0.5625,
          "current_positive_day_rate": 0.5,
          "max_allowed_daily_sample_share": 0.7,
          "missing_active_days": 0,
          "missing_positive_days_at_current_requirement": 1,
          "reasons": [
            "NEEDS_HIGHER_POSITIVE_DAY_RATE"
          ],
          "required_active_days": 3,
          "required_positive_day_rate": 0.6667,
          "required_positive_days_at_current_requirement": 3,
          "status": "DAILY_PNL_UNSTABLE"
        },
        "daily_stability_status": "DAILY_PNL_UNSTABLE",
        "direction": "DOWN",
        "forecast_direction": "UP",
        "granularity": "S5",
        "max_daily_sample_share": 0.5625,
        "name": "NZD_USD_UP_H61_240m_CLT0p50_FADE_TO_DOWN_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7",
        "optimized_profit_factor": 1.7099,
        "optimized_stop_loss_pips": 7.0,
        "optimized_take_profit_pips": 10.0,
        "optimized_win_rate": 0.5625,
        "pair": "NZD_USD",
        "positive_day_rate": 0.5,
        "positive_days": 2,
        "samples": 32
      },
      {
        "active_days": 4,
        "adoption_blockers": [
          "DAILY_PNL_UNSTABLE",
          "NEEDS_HIGHER_POSITIVE_DAY_RATE"
        ],
        "adoption_status": "RANK_ONLY_NOT_DAILY_STABLE",
        "daily_stability_gap": {
          "current_max_daily_sample_share": 0.5625,
          "current_positive_day_rate": 0.5,
          "max_allowed_daily_sample_share": 0.7,
          "missing_active_days": 0,
          "missing_positive_days_at_current_requirement": 1,
          "reasons": [
            "NEEDS_HIGHER_POSITIVE_DAY_RATE"
          ],
          "required_active_days": 3,
          "required_positive_day_rate": 0.6667,
          "required_positive_days_at_current_requirement": 3,
          "status": "DAILY_PNL_UNSTABLE"
        },
        "daily_stability_status": "DAILY_PNL_UNSTABLE",
        "direction": "DOWN",
        "forecast_direction": "UP",
        "granularity": "S5",
        "max_daily_sample_share": 0.5625,
        "name": "NZD_USD_UP_H61_240m_FADE_TO_DOWN_S5_BIDASK_CONTRARIAN_HARVEST_TP10_SL7",
        "optimized_profit_factor": 1.7099,
        "optimized_stop_loss_pips": 7.0,
        "optimized_take_profit_pips": 10.0,
        "optimized_win_rate": 0.5625,
        "pair": "NZD_USD",
        "positive_day_rate": 0.5,
        "positive_days": 2,
        "samples": 32
      },
      {
        "active_days": 7,
        "adoption_blockers": [
          "DAILY_PNL_UNSTABLE",
          "NEEDS_HIGHER_POSITIVE_DAY_RATE"
        ],
        "adoption_status": "RANK_ONLY_NOT_DAILY_STABLE",
        "daily_stability_gap": {
          "current_max_daily_sample_share": 0.3023,
          "current_positive_day_rate": 0.5714,
          "max_allowed_daily_sample_share": 0.7,
          "missing_active_days": 0,
          "missing_positive_days_at_current_requirement": 1,
          "reasons": [
            "NEEDS_HIGHER_POSITIVE_DAY_RATE"
          ],
          "required_active_days": 3,
          "required_positive_day_rate": 0.6667,
          "required_positive_days_at_current_requirement": 5,
          "status": "DAILY_PNL_UNSTABLE"
        },
        "daily_stability_status": "DAILY_PNL_UNSTABLE",
        "direction": "UP",
        "forecast_direction": "DOWN",
        "granularity": "S5",
        "max_daily_sample_share": 0.3023,
        "name": "EUR_CAD_DOWN_H61_240m_CLT0p50_FADE_TO_UP_S5_BIDASK_CONTRARIAN_HARVEST_TP5_SL7",
        "optimized_profit_factor": 1.6297,
        "optimized_stop_loss_pips": 7.0,
        "optimized_take_profit_pips": 5.0,
        "optimized_win_rate": 0.6977,
        "pair": "EUR_CAD",
        "positive_day_rate": 0.5714,
        "positive_days": 4,
        "samples": 43
      }
    ],
    "rank_only_support_rules": 5,
    "replay_validation_command": "python3 scripts/oanda_history_replay_validate.py --forecast-history data/forecast_history.jsonl --granularity S5 --history-dir logs/replay/oanda_history/20260703T072439Z --history-dir logs/replay/oanda_history/20260703T080331Z --history-dir logs/replay/oanda_history/20260703T120929Z --history-dir logs/replay/oanda_history/20260703T123013Z --history-dir logs/replay/oanda_history/20260703T134642Z --history-dir logs/replay/oanda_history/20260703T135559Z --history-dir logs/replay/oanda_history/20260703T142126Z --history-dir logs/replay/oanda_history/20260703T142653Z --history-dir logs/replay/oanda_history/20260703T143956Z --auto-history-min-days 30 --stable-min-active-days 3 --stable-max-daily-sample-share 0.7 --stable-min-positive-day-rate 0.6666666667",
    "source_report": "logs/reports/forecast_improvement/oanda_history_replay_validate_20260703T145218Z.json",
    "stale_history_fetch_command_suppressed": false,
    "support_rules": 17
  },
  "capture_economics": {
    "market_close": {
      "expectancy_jpy_per_trade": -756.7,
      "net_jpy": -74151.8,
      "trades": 98
    },
    "overall": {
      "expectancy_jpy_per_trade": -177.4,
      "net_jpy": -40616.9,
      "payoff_ratio": 0.393,
      "profit_factor": null,
      "trades": 229,
      "win_rate": 0.5983
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
    "audit_now_utc": "2026-07-06T01:58:50.390861+00:00",
    "execution_timing_audit": {
      "generated_at_utc": "2026-07-06T01:07:00.756897+00:00",
      "label_counts": {
        "LOSS_CLOSE_CONTAINED_RISK": 19,
        "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE": 13
      },
      "loaded": true,
      "loss_close_actual_pl_jpy": -39275.3429,
      "loss_close_counterfactual_profit_capture_delta_jpy": 18731.1971,
      "loss_close_counterfactual_profit_capture_jpy": 3547.1025,
      "loss_close_counterfactual_profit_capture_pl_jpy": -20544.1458,
      "loss_close_repair_replay_block_reasons": {
        "BELOW_NOISE_FLOOR": 1
      },
      "loss_close_repair_replay_counterfactual_pl_jpy": -20500.5457,
      "loss_close_repair_replay_delta_jpy": 18774.7972,
      "loss_close_repair_replay_profit_capture_jpy": 3830.1817,
      "loss_closes_profit_capture_missed": 14,
      "loss_closes_repair_replay_triggered": 13,
      "loss_market_close_rows": 32,
      "path": "/Users/tossaki/App/QuantRabbit-live/data/execution_timing_audit.json",
      "post_repair_live_evidence_loss_closes_audited": 5,
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
        }
      ],
      "top_entry_quality_residual_method_rollups": [
        {
          "actual_pl_jpy": -16944.7649,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 7,
            "NO_PROFIT_CANDIDATE": 8
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
              "actual_pl_jpy": -844.2,
              "lane_id": "range_trader:EUR_JPY:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "EUR_JPY",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -844.2,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472903"
            },
            {
              "actual_pl_jpy": -464.0,
              "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "USD_JPY",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -464.0,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472900"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 9,
            "STOP_LOSS_ORDER": 6
          },
          "group_count": 13,
          "loss_closes": 15,
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
          "repair_replay_pl_jpy": -16944.7649,
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
          "counterfactual_delta_jpy": 606.5364,
          "counterfactual_exit": "TP_PROGRESS_CAPTURE",
          "counterfactual_jpy": 367.0573,
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
          "counterfactual_delta_jpy": 777.8047,
          "counterfactual_exit": "TP_PROGRESS_CAPTURE",
          "counterfactual_jpy": 612.6353,
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
          "counterfactual_delta_jpy": 340.6732,
          "counterfactual_exit": "TP_PROGRESS_CAPTURE",
          "counterfactual_jpy": 311.7965,
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
          "counterfactual_delta_jpy": 1267.7279,
          "counterfactual_exit": "TP_PROGRESS_CAPTURE",
          "counterfactual_jpy": 285.9337,
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
        }
      ],
      "top_repair_replay_residual_method_rollups": [
        {
          "actual_pl_jpy": -16944.7649,
          "block_reasons": {
            "BELOW_TP_PROGRESS_GATE": 7,
            "NO_PROFIT_CANDIDATE": 8
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
              "actual_pl_jpy": -844.2,
              "lane_id": "range_trader:EUR_JPY:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "EUR_JPY",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -844.2,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472903"
            },
            {
              "actual_pl_jpy": -464.0,
              "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
              "method": "RANGE_ROTATION",
              "pair": "USD_JPY",
              "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
              "repair_replay_pl_jpy": -464.0,
              "repair_replay_triggered": false,
              "side": "SHORT",
              "trade_id": "472900"
            }
          ],
          "exit_reasons": {
            "MARKET_ORDER_TRADE_CLOSE": 9,
            "STOP_LOSS_ORDER": 6
          },
          "group_count": 13,
          "loss_closes": 15,
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
          "repair_replay_pl_jpy": -16944.7649,
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
          "repair_counterfactual_delta_jpy": 830.681,
          "repair_counterfactual_jpy": 665.5116,
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
          "repair_counterfactual_delta_jpy": 388.2683,
          "repair_counterfactual_jpy": 359.3916,
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
          "repair_counterfactual_delta_jpy": 1280.8881,
          "repair_counterfactual_jpy": 299.0939,
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
          "repair_counterfactual_delta_jpy": 123.4724,
          "repair_counterfactual_jpy": 116.7296,
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
      "window_from_utc": "2026-05-26T07:03:46.222763+00:00",
      "window_lookback_hours": 744.0,
      "window_to_utc": "2026-07-06T01:07:00.756897+00:00"
    },
    "gateway_event_stream_events": 34718,
    "gateway_event_stream_lag_minutes": 45.073,
    "gateway_event_stream_latest_ts_utc": "2026-07-06T01:13:45.999239+00:00",
    "gateway_event_stream_market_close_gap_minutes": 0.0,
    "gateway_event_stream_stale": false,
    "gateway_market_closes": 99,
    "latest_gateway_market_close_ts_utc": "2026-06-26T07:03:45.928869+00:00",
    "latest_loss_close_ts_utc": null,
    "ledger_exists": true,
    "lookback_days": 7,
    "path": "/Users/tossaki/App/QuantRabbit-live/data/execution_ledger.db",
    "recent_close_gate_missing_loss_closes": 0,
    "recent_close_gate_missing_loss_examples": [],
    "recent_close_gate_missing_loss_net_jpy": 0,
    "recent_close_gate_missing_receipt_evidence_absent_loss_closes": 0,
    "recent_close_gate_missing_receipt_evidence_absent_loss_net_jpy": 0,
    "recent_close_gate_missing_receipt_evidence_present_loss_closes": 0,
    "recent_close_gate_missing_receipt_evidence_present_loss_net_jpy": 0,
    "recent_close_gate_not_passing_loss_closes": 0,
    "recent_close_gate_not_passing_loss_examples": [],
    "recent_close_gate_not_passing_loss_net_jpy": 0,
    "recent_close_gate_unverified_loss_closes": 0,
    "recent_close_gate_unverified_loss_examples": [],
    "recent_close_gate_unverified_loss_net_jpy": 0,
    "recent_contained_risk_loss_closes": 0,
    "recent_contained_risk_loss_examples": [],
    "recent_contained_risk_loss_net_jpy": 0,
    "recent_cutoff_utc": "2026-06-29T01:58:50.390861+00:00",
    "recent_gateway_market_closes": 0,
    "recent_leak_loss_by_lane": [],
    "recent_leak_loss_closes": 0,
    "recent_leak_loss_examples": [],
    "recent_leak_loss_net_jpy": 0,
    "recent_loss_by_lane": [],
    "recent_loss_closes": 0,
    "recent_loss_examples": [],
    "recent_loss_net_jpy": 0,
    "recent_loss_timing_label_counts": {},
    "recent_premature_loss_closes": 0,
    "recent_unclassified_loss_closes": 0,
    "recent_unverified_loss_closes": 0,
    "recent_unverified_loss_examples": [],
    "recent_unverified_loss_net_jpy": 0
  },
  "finding_counts": {
    "P0": 5,
    "P1": 4,
    "P2": 0
  },
  "generated_at_utc": "2026-07-06T01:58:50.390861+00:00",
  "oanda_campaign_firepower": {
    "contract": "audit-only merged firepower estimate from pair-shard validation evidence; it does not grant live permission, size orders, or waive gateway gates",
    "evidence_queue": {
      "estimated_return_pct_per_active_day_at_observed_frequency": 61.172817,
      "observed_attempts_per_active_day": 99.170958,
      "pair_count": 6,
      "top_vehicle_keys": [
        "EUR_JPY|SHORT|trend_continuation|tp1_sl0.75",
        "EUR_JPY|SHORT|trend_continuation|tp1.25_sl1",
        "USD_JPY|SHORT|range_reversion|tp1_sl0.75"
      ],
      "trades_needed_for_minimum_5pct_at_weighted_expectancy": 9.0,
      "trades_needed_for_target_10pct_at_weighted_expectancy": 17.0,
      "unique_vehicle_count": 49,
      "weighted_return_pct_per_trade_at_risk_lens": 0.616842
    },
    "generated_at_utc": "2026-07-03T08:29:33.892648Z",
    "high_precision": {
      "estimated_return_pct_per_active_day_at_observed_frequency": 59.015709,
      "observed_attempts_per_active_day": 89.49192,
      "pair_count": 8,
      "top_vehicle_keys": [
        "AUD_USD|SHORT|range_reversion|tp1.25_sl1",
        "GBP_USD|SHORT|range_reversion|tp1.25_sl1",
        "AUD_USD|SHORT|range_reversion|tp1.25_sl1"
      ],
      "trades_needed_for_minimum_5pct_at_weighted_expectancy": 8.0,
      "trades_needed_for_target_10pct_at_weighted_expectancy": 16.0,
      "unique_vehicle_count": 45,
      "weighted_return_pct_per_trade_at_risk_lens": 0.659453
    },
    "minimum_return_pct": 5.0,
    "path": "/Users/tossaki/App/QuantRabbit/src/quant_rabbit/oanda_universal_rotation_precision_rules.json",
    "per_trade_risk_pct_lens": 1.0,
    "report_exists": true,
    "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
    "target_open": true,
    "target_return_pct": 10.0
  },
  "order_capture_freshness": {
    "capture_economics_generated_at_utc": "2026-07-06T01:57:35.731924+00:00",
    "capture_generated_after_order_intents": false,
    "capture_trades": 229,
    "intent_capture_economics_trades": [
      229
    ],
    "metadata_trade_count_mismatch": false,
    "mismatch_examples": [],
    "order_intents_generated_at_utc": "2026-07-06T01:58:22.732022+00:00"
  },
  "order_intents": {
    "candidate_count": 82,
    "dry_run_blocked_lanes": 82,
    "generated_at_utc": "2026-07-06T01:58:22.732022+00:00",
    "live_ready_lanes": 0,
    "repair_frontier": {
      "blocked_count": 11,
      "candidate_count": 11,
      "examples": [
        {
          "blocker_codes": [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
          ],
          "blocker_count": 4,
          "lane_id": "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "LIMIT",
          "pair": "AUD_JPY",
          "positive_rotation_pessimistic_expectancy_jpy": 189.9205,
          "repair_mode": "TP_PROOF_COLLECTION_HARVEST",
          "risk_allowed": false,
          "side": "SHORT",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "REWARD_RISK_TOO_LOW",
            "EXHAUSTION_RANGE_CHASE",
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
          ],
          "blocker_count": 4,
          "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "LIMIT",
          "pair": "EUR_USD",
          "positive_rotation_pessimistic_expectancy_jpy": 324.7652,
          "repair_mode": "TP_PROVEN_HARVEST",
          "risk_allowed": false,
          "side": "LONG",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
          ],
          "blocker_count": 4,
          "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "STOP-ENTRY",
          "pair": "EUR_USD",
          "positive_rotation_pessimistic_expectancy_jpy": 304.0708,
          "repair_mode": "TP_PROOF_COLLECTION_HARVEST",
          "risk_allowed": false,
          "side": "SHORT",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
          ],
          "blocker_count": 4,
          "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "LIMIT",
          "pair": "EUR_USD",
          "positive_rotation_pessimistic_expectancy_jpy": 304.0708,
          "repair_mode": "TP_PROOF_COLLECTION_HARVEST",
          "risk_allowed": false,
          "side": "SHORT",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "REWARD_RISK_TOO_LOW",
            "PATTERN_REVERSAL_CHASE",
            "EXHAUSTION_RANGE_CHASE",
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
          ],
          "blocker_count": 5,
          "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "STOP-ENTRY",
          "pair": "EUR_USD",
          "positive_rotation_pessimistic_expectancy_jpy": 324.7652,
          "repair_mode": "TP_PROVEN_HARVEST",
          "risk_allowed": false,
          "side": "LONG",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "RANGE_COUNTERTREND_RR_TOO_LOW",
            "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
          ],
          "blocker_count": 5,
          "lane_id": "range_trader:GBP_JPY:SHORT:RANGE_ROTATION",
          "method": "RANGE_ROTATION",
          "oanda_campaign_matching_vehicle_key": "GBP_JPY|SHORT|range_reversion|tp1_sl1",
          "order_type": "LIMIT",
          "pair": "GBP_JPY",
          "positive_rotation_pessimistic_expectancy_jpy": null,
          "repair_mode": "OANDA_CAMPAIGN_FIREPOWER_HARVEST",
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
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
          ],
          "blocker_count": 6,
          "lane_id": "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE",
          "method": "BREAKOUT_FAILURE",
          "oanda_campaign_matching_vehicle_key": null,
          "order_type": "STOP-ENTRY",
          "pair": "AUD_JPY",
          "positive_rotation_pessimistic_expectancy_jpy": 189.9205,
          "repair_mode": "TP_PROOF_COLLECTION_HARVEST",
          "risk_allowed": false,
          "side": "SHORT",
          "status": "DRY_RUN_BLOCKED"
        },
        {
          "blocker_codes": [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "REWARD_RISK_TOO_LOW",
            "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED",
            "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
          ],
          "blocker_count": 6,
          "lane_id": "trend_trader:EUR_JPY:SHORT:TREND_CONTINUATION",
          "method": "TREND_CONTINUATION",
          "oanda_campaign_matching_vehicle_key": "EUR_JPY|SHORT|trend_continuation|tp1_sl1",
          "order_type": "STOP-ENTRY",
          "pair": "EUR_JPY",
          "positive_rotation_pessimistic_expectancy_jpy": null,
          "repair_mode": "OANDA_CAMPAIGN_FIREPOWER_HARVEST",
          "risk_allowed": false,
          "side": "SHORT",
          "status": "DRY_RUN_BLOCKED"
        }
      ],
      "live_ready_count": 0,
      "top_remaining_blockers": [
        {
          "code": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
          "count": 11
        },
        {
          "code": "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
          "count": 9
        },
        {
          "code": "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
          "count": 8
        },
        {
          "code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
          "count": 8
        },
        {
          "code": "EXHAUSTION_RANGE_CHASE",
          "count": 5
        },
        {
          "code": "REWARD_RISK_TOO_LOW",
          "count": 5
        },
        {
          "code": "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED",
          "count": 3
        },
        {
          "code": "OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED",
          "count": 3
        },
        {
          "code": "MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED",
          "count": 2
        },
        {
          "code": "PATTERN_REVERSAL_CHASE",
          "count": 2
        }
      ]
    },
    "top_blockers": [
      {
        "code": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
        "count": 82
      },
      {
        "code": "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
        "count": 80
      },
      {
        "code": "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
        "count": 73
      },
      {
        "code": "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
        "count": 71
      },
      {
        "code": "LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT",
        "count": 38
      },
      {
        "code": "EXHAUSTION_RANGE_CHASE",
        "count": 37
      },
      {
        "code": "RANGE_ROTATION_BROADER_LOCATION_CHASE",
        "count": 27
      },
      {
        "code": "SPREAD_TOO_WIDE",
        "count": 27
      },
      {
        "code": "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
        "count": 26
      },
      {
        "code": "STRATEGY_NOT_ELIGIBLE",
        "count": 23
      }
    ]
  },
  "profit_capture_replay_repair": {
    "active_counterfactual_profit_capture_pl_jpy": -20500.5457,
    "capture_market_close_net_jpy": -74151.8,
    "capture_take_profit_net_jpy": 48804.8,
    "clearance_condition": "execution-timing-audit must report zero post-repair live-evidence loss_closes_repair_replay_triggered with the current production-gate replay contract after TP-progress TAKE_PROFIT_MARKET / guardian repair has run on live broker truth; pre-repair historical misses remain diagnostic unless a post-repair production gate also proves an executable profit capture",
    "counterfactual_profit_capture_delta_jpy": 18774.7972,
    "counterfactual_profit_capture_jpy": 3830.1817,
    "execution_timing_generated_at_utc": "2026-07-06T01:07:00.756897+00:00",
    "execution_timing_loaded": true,
    "execution_timing_window_from_utc": "2026-05-26T07:03:46.222763+00:00",
    "execution_timing_window_lookback_hours": 744.0,
    "execution_timing_window_to_utc": "2026-07-06T01:07:00.756897+00:00",
    "guardian_profit_capture_inactive": false,
    "loss_close_repair_replay_block_reasons": {
      "BELOW_NOISE_FLOOR": 1
    },
    "loss_closes_profit_capture_missed": 14,
    "loss_closes_repair_replay_triggered": 13,
    "month_scale_replay_cleared_by_residual_family_filters": false,
    "month_scale_replay_loaded": true,
    "month_scale_replay_min_hours": 720.0,
    "month_scale_replay_required": true,
    "month_scale_residual_family_filters": {
      "all_negative_families_can_create_live_permission_false": true,
      "baseline_pl_jpy": -39246.4662,
      "clearance_basis": "table absent, stale, unsafe, or filtered replay remains negative",
      "clears_month_scale_tp_progress_replay_still_negative": false,
      "current_against_execution_timing": false,
      "excluded_family_count": 23,
      "excluded_trade_ids": [
        "471711",
        "471817",
        "472071",
        "472088",
        "472094",
        "472125",
        "472156",
        "472174",
        "472190",
        "472208",
        "472233",
        "472252",
        "472312",
        "472380",
        "472427",
        "472445",
        "472497",
        "472530",
        "472632",
        "472743",
        "472775",
        "472834",
        "472837",
        "472900",
        "472903",
        "472952"
      ],
      "generated_at_utc": "2026-07-05T17:21:46Z",
      "improved_pl_jpy": -20863.5316,
      "loaded": true,
      "manual_eurusd_472987_excluded": true,
      "market_close_leak_family_gate_active": true,
      "no_unproven_fresh_entry_promotion": true,
      "remaining_residual_groups": [],
      "residual_family_filters_active": true,
      "residual_pl_jpy": 2984.1927,
      "safety_ok": true,
      "source_execution_timing_generated_at_utc": "2026-07-03T20:08:53.084075+00:00",
      "tp_progress_harvest_gate_active": true
    },
    "post_repair_live_evidence_loss_closes_audited": 5,
    "post_repair_live_evidence_loss_closes_profit_capture_missed": 0,
    "post_repair_live_evidence_loss_closes_repair_replay_triggered": 0,
    "pre_repair_historical_loss_closes_profit_capture_missed": 14,
    "pre_repair_historical_loss_closes_repair_replay_triggered": 13,
    "raw_counterfactual_profit_capture_pl_jpy": -20544.1458,
    "repair_replay_contract": "TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1",
    "repair_replay_contract_present": true,
    "repair_replay_counterfactual_pl_jpy": -20500.5457,
    "replay_repair_proved": false,
    "self_improvement_p0_codes": [
      "EXECUTION_LEDGER_STALE"
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
      }
    ],
    "top_entry_quality_residual_method_rollups": [
      {
        "actual_pl_jpy": -16944.7649,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 7,
          "NO_PROFIT_CANDIDATE": 8
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
            "actual_pl_jpy": -844.2,
            "lane_id": "range_trader:EUR_JPY:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "EUR_JPY",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -844.2,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472903"
          },
          {
            "actual_pl_jpy": -464.0,
            "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "USD_JPY",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -464.0,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472900"
          }
        ],
        "exit_reasons": {
          "MARKET_ORDER_TRADE_CLOSE": 9,
          "STOP_LOSS_ORDER": 6
        },
        "group_count": 13,
        "loss_closes": 15,
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
        "repair_replay_pl_jpy": -16944.7649,
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
        "counterfactual_delta_jpy": 606.5364,
        "counterfactual_exit": "TP_PROGRESS_CAPTURE",
        "counterfactual_jpy": 367.0573,
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
        "counterfactual_delta_jpy": 777.8047,
        "counterfactual_exit": "TP_PROGRESS_CAPTURE",
        "counterfactual_jpy": 612.6353,
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
        "counterfactual_delta_jpy": 340.6732,
        "counterfactual_exit": "TP_PROGRESS_CAPTURE",
        "counterfactual_jpy": 311.7965,
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
        "counterfactual_delta_jpy": 1267.7279,
        "counterfactual_exit": "TP_PROGRESS_CAPTURE",
        "counterfactual_jpy": 285.9337,
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
      }
    ],
    "top_repair_replay_residual_method_rollups": [
      {
        "actual_pl_jpy": -16944.7649,
        "block_reasons": {
          "BELOW_TP_PROGRESS_GATE": 7,
          "NO_PROFIT_CANDIDATE": 8
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
            "actual_pl_jpy": -844.2,
            "lane_id": "range_trader:EUR_JPY:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "EUR_JPY",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -844.2,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472903"
          },
          {
            "actual_pl_jpy": -464.0,
            "lane_id": "range_trader:USD_JPY:SHORT:RANGE_ROTATION",
            "method": "RANGE_ROTATION",
            "pair": "USD_JPY",
            "repair_replay_block_reason": "BELOW_TP_PROGRESS_GATE",
            "repair_replay_pl_jpy": -464.0,
            "repair_replay_triggered": false,
            "side": "SHORT",
            "trade_id": "472900"
          }
        ],
        "exit_reasons": {
          "MARKET_ORDER_TRADE_CLOSE": 9,
          "STOP_LOSS_ORDER": 6
        },
        "group_count": 13,
        "loss_closes": 15,
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
        "repair_replay_pl_jpy": -16944.7649,
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
        "repair_counterfactual_delta_jpy": 830.681,
        "repair_counterfactual_jpy": 665.5116,
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
        "repair_counterfactual_delta_jpy": 388.2683,
        "repair_counterfactual_jpy": 359.3916,
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
        "repair_counterfactual_delta_jpy": 1280.8881,
        "repair_counterfactual_jpy": 299.0939,
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
        "repair_counterfactual_delta_jpy": 123.4724,
        "repair_counterfactual_jpy": 116.7296,
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
    "economic_precision_edges": 10,
    "economic_precision_gaps": 5,
    "ledger_exists": true,
    "top_edges": [
      {
        "bucket": "_all_pairs:_all_regimes",
        "economic_hit_rate": 1.0,
        "economic_hit_rate_wilson_lower": 0.963,
        "economic_samples": 100,
        "hit_rate": 1.0,
        "hit_rate_wilson_lower": 0.963,
        "pair": "_all_pairs",
        "passes_economic_precision": true,
        "regime": "_all_regimes",
        "samples": 100,
        "signal_name": "bb_squeeze_expansion_imminent",
        "timeout_count": 0,
        "timeout_rate": 0.0
      },
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
        "bucket": "_all_pairs:TREND",
        "economic_hit_rate": 0.99,
        "economic_hit_rate_wilson_lower": 0.9455,
        "economic_samples": 100,
        "hit_rate": 0.99,
        "hit_rate_wilson_lower": 0.9455,
        "pair": "_all_pairs",
        "passes_economic_precision": true,
        "regime": "TREND",
        "samples": 100,
        "signal_name": "bb_squeeze_expansion_imminent",
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
        "bucket": "AUD_JPY:UNCLEAR",
        "economic_hit_rate": 0.946,
        "economic_hit_rate_wilson_lower": 0.879,
        "economic_samples": 92,
        "hit_rate": 0.967,
        "hit_rate_wilson_lower": 0.9065,
        "pair": "AUD_JPY",
        "regime": "UNCLEAR",
        "samples": 90,
        "signal_name": "liquidity_sweep_high",
        "timeout_count": 2,
        "timeout_rate": 0.0217
      }
    ]
  },
  "self_improvement": {
    "p0_codes": [
      "EXECUTION_LEDGER_STALE"
    ],
    "p0_findings": 1,
    "p1_findings": 8,
    "status": "SELF_IMPROVEMENT_BLOCKED"
  },
  "target": {
    "remaining_target_jpy": 28718.6,
    "status": "PURSUE_TARGET",
    "target_open": true
  }
}
```

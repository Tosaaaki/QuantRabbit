# Execution Timing Audit

- Generated at UTC: `2026-07-06T01:58:38.527953+00:00`
- Status: `OK`
- Window UTC: `2026-06-19T07:03:46.222763+00:00` to `2026-07-06T01:58:38.527953+00:00`
- Precision: `OANDA_M1_BID_ASK_CANDLES`

## Summary

- `canceled_orders_audited`: `4`
- `canceled_entry_touched_after_cancel`: `2`
- `canceled_entry_touched_after_cancel_rate`: `0.5`
- `canceled_positive_after_cancel_entry`: `2`
- `canceled_positive_after_cancel_entry_rate`: `0.5`
- `canceled_tp_touched_after_cancel`: `1`
- `canceled_tp_touched_after_cancel_rate`: `0.25`
- `canceled_estimated_missed_mfe_jpy`: `714.3022`
- `loss_closes_audited`: `8`
- `loss_closes_had_positive_mfe`: `5`
- `loss_closes_had_positive_mfe_rate`: `0.625`
- `loss_closes_tp_touched_before_close`: `0`
- `loss_closes_tp_touched_before_close_rate`: `0.0`
- `loss_close_estimated_mfe_jpy`: `606.7484`
- `loss_closes_profit_capture_missed`: `1`
- `loss_closes_profit_capture_missed_rate`: `0.125`
- `stop_loss_closes_profit_capture_missed`: `1`
- `loss_close_estimated_capture_gap_jpy`: `214.2`
- `loss_close_actual_pl_jpy`: `-8677.3834`
- `loss_close_counterfactual_profit_capture_pl_jpy`: `-8231.3434`
- `loss_close_counterfactual_profit_capture_delta_jpy`: `446.04`
- `loss_close_counterfactual_profit_capture_jpy`: `105.84`
- `loss_closes_repair_replay_triggered`: `1`
- `loss_closes_repair_replay_triggered_rate`: `0.125`
- `tp_progress_repair_live_evidence_boundary_utc`: `2026-06-22T17:54:26Z`
- `tp_progress_repair_live_evidence_boundary_reason`: `Full TP-progress production replay contract deployed; 2026-06-22T09:35:39Z was TP-progress banking only.`
- `tp_progress_repair_live_evidence_status`: `POST_REPAIR_REPLAY_CLEAN`
- `pre_repair_historical_loss_closes_audited`: `3`
- `pre_repair_historical_loss_closes_profit_capture_missed`: `1`
- `pre_repair_historical_loss_closes_profit_capture_missed_rate`: `0.3333`
- `pre_repair_historical_loss_closes_repair_replay_triggered`: `1`
- `pre_repair_historical_loss_closes_repair_replay_triggered_rate`: `0.3333`
- `post_repair_live_evidence_loss_closes_audited`: `5`
- `post_repair_live_evidence_loss_closes_profit_capture_missed`: `0`
- `post_repair_live_evidence_loss_closes_profit_capture_missed_rate`: `0.0`
- `post_repair_live_evidence_stop_loss_closes_profit_capture_missed`: `0`
- `post_repair_live_evidence_loss_closes_repair_replay_triggered`: `0`
- `post_repair_live_evidence_loss_closes_repair_replay_triggered_rate`: `0.0`
- `loss_close_repair_replay_profit_capture_jpy`: `126.0`
- `loss_close_repair_replay_actual_pl_jpy`: `-8677.3834`
- `loss_close_repair_replay_counterfactual_pl_jpy`: `-8211.1834`
- `loss_close_repair_replay_delta_jpy`: `466.2`
- `loss_close_repair_replay_block_reasons`: `{}`
- `top_repair_replay_residual_groups`: `[{'pair': 'AUD_USD', 'side': 'LONG', 'method': 'RANGE_ROTATION', 'exit_reason': 'STOP_LOSS_ORDER', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 1, 'actual_pl_jpy': -2690.6967, 'repair_replay_pl_jpy': -2690.6967, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'BELOW_TP_PROGRESS_GATE': 1}, 'examples': [{'trade_id': '472952', 'lane_id': 'range_trader:AUD_USD:LONG:RANGE_ROTATION', 'actual_pl_jpy': -2690.6967, 'repair_replay_pl_jpy': -2690.6967, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'BELOW_TP_PROGRESS_GATE'}]}, {'pair': 'AUD_USD', 'side': 'SHORT', 'method': 'RANGE_ROTATION', 'exit_reason': 'STOP_LOSS_ORDER', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 1, 'actual_pl_jpy': -1705.6738, 'repair_replay_pl_jpy': -1705.6738, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'NO_PROFIT_CANDIDATE': 1}, 'examples': [{'trade_id': '472834', 'lane_id': 'range_trader:AUD_USD:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -1705.6738, 'repair_replay_pl_jpy': -1705.6738, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'NO_PROFIT_CANDIDATE'}]}, {'pair': 'NZD_USD', 'side': 'LONG', 'method': 'RANGE_ROTATION', 'exit_reason': 'MARKET_ORDER_TRADE_CLOSE', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 1, 'actual_pl_jpy': -1380.8008, 'repair_replay_pl_jpy': -1380.8008, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'NO_PROFIT_CANDIDATE': 1}, 'examples': [{'trade_id': '472743', 'lane_id': 'range_trader:NZD_USD:LONG:RANGE_ROTATION', 'actual_pl_jpy': -1380.8008, 'repair_replay_pl_jpy': -1380.8008, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'NO_PROFIT_CANDIDATE'}]}, {'pair': 'GBP_USD', 'side': 'SHORT', 'method': 'RANGE_ROTATION', 'exit_reason': 'STOP_LOSS_ORDER', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 1, 'actual_pl_jpy': -971.0121, 'repair_replay_pl_jpy': -971.0121, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'BELOW_TP_PROGRESS_GATE': 1}, 'examples': [{'trade_id': '472837', 'lane_id': 'range_trader:GBP_USD:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -971.0121, 'repair_replay_pl_jpy': -971.0121, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'BELOW_TP_PROGRESS_GATE'}]}, {'pair': 'EUR_JPY', 'side': 'SHORT', 'method': 'RANGE_ROTATION', 'exit_reason': 'STOP_LOSS_ORDER', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 1, 'actual_pl_jpy': -844.2, 'repair_replay_pl_jpy': -844.2, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'BELOW_TP_PROGRESS_GATE': 1}, 'examples': [{'trade_id': '472903', 'lane_id': 'range_trader:EUR_JPY:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -844.2, 'repair_replay_pl_jpy': -844.2, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'BELOW_TP_PROGRESS_GATE'}]}, {'pair': 'USD_JPY', 'side': 'SHORT', 'method': 'RANGE_ROTATION', 'exit_reason': 'STOP_LOSS_ORDER', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 2, 'actual_pl_jpy': -744.8, 'repair_replay_pl_jpy': -744.8, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'BELOW_TP_PROGRESS_GATE': 1, 'NO_PROFIT_CANDIDATE': 1}, 'examples': [{'trade_id': '472900', 'lane_id': 'range_trader:USD_JPY:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -464.0, 'repair_replay_pl_jpy': -464.0, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'BELOW_TP_PROGRESS_GATE'}, {'trade_id': '472775', 'lane_id': 'range_trader:USD_JPY:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -280.8, 'repair_replay_pl_jpy': -280.8, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'NO_PROFIT_CANDIDATE'}]}]`
- `top_tp_progress_repair_residual_groups`: `[]`
- `top_entry_quality_residual_groups`: `[{'pair': 'AUD_USD', 'side': 'LONG', 'method': 'RANGE_ROTATION', 'exit_reason': 'STOP_LOSS_ORDER', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 1, 'actual_pl_jpy': -2690.6967, 'repair_replay_pl_jpy': -2690.6967, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'BELOW_TP_PROGRESS_GATE': 1}, 'examples': [{'trade_id': '472952', 'lane_id': 'range_trader:AUD_USD:LONG:RANGE_ROTATION', 'actual_pl_jpy': -2690.6967, 'repair_replay_pl_jpy': -2690.6967, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'BELOW_TP_PROGRESS_GATE'}]}, {'pair': 'AUD_USD', 'side': 'SHORT', 'method': 'RANGE_ROTATION', 'exit_reason': 'STOP_LOSS_ORDER', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 1, 'actual_pl_jpy': -1705.6738, 'repair_replay_pl_jpy': -1705.6738, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'NO_PROFIT_CANDIDATE': 1}, 'examples': [{'trade_id': '472834', 'lane_id': 'range_trader:AUD_USD:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -1705.6738, 'repair_replay_pl_jpy': -1705.6738, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'NO_PROFIT_CANDIDATE'}]}, {'pair': 'NZD_USD', 'side': 'LONG', 'method': 'RANGE_ROTATION', 'exit_reason': 'MARKET_ORDER_TRADE_CLOSE', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 1, 'actual_pl_jpy': -1380.8008, 'repair_replay_pl_jpy': -1380.8008, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'NO_PROFIT_CANDIDATE': 1}, 'examples': [{'trade_id': '472743', 'lane_id': 'range_trader:NZD_USD:LONG:RANGE_ROTATION', 'actual_pl_jpy': -1380.8008, 'repair_replay_pl_jpy': -1380.8008, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'NO_PROFIT_CANDIDATE'}]}, {'pair': 'GBP_USD', 'side': 'SHORT', 'method': 'RANGE_ROTATION', 'exit_reason': 'STOP_LOSS_ORDER', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 1, 'actual_pl_jpy': -971.0121, 'repair_replay_pl_jpy': -971.0121, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'BELOW_TP_PROGRESS_GATE': 1}, 'examples': [{'trade_id': '472837', 'lane_id': 'range_trader:GBP_USD:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -971.0121, 'repair_replay_pl_jpy': -971.0121, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'BELOW_TP_PROGRESS_GATE'}]}, {'pair': 'EUR_JPY', 'side': 'SHORT', 'method': 'RANGE_ROTATION', 'exit_reason': 'STOP_LOSS_ORDER', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 1, 'actual_pl_jpy': -844.2, 'repair_replay_pl_jpy': -844.2, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'BELOW_TP_PROGRESS_GATE': 1}, 'examples': [{'trade_id': '472903', 'lane_id': 'range_trader:EUR_JPY:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -844.2, 'repair_replay_pl_jpy': -844.2, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'BELOW_TP_PROGRESS_GATE'}]}, {'pair': 'USD_JPY', 'side': 'SHORT', 'method': 'RANGE_ROTATION', 'exit_reason': 'STOP_LOSS_ORDER', 'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'loss_closes': 2, 'actual_pl_jpy': -744.8, 'repair_replay_pl_jpy': -744.8, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'BELOW_TP_PROGRESS_GATE': 1, 'NO_PROFIT_CANDIDATE': 1}, 'examples': [{'trade_id': '472900', 'lane_id': 'range_trader:USD_JPY:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -464.0, 'repair_replay_pl_jpy': -464.0, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'BELOW_TP_PROGRESS_GATE'}, {'trade_id': '472775', 'lane_id': 'range_trader:USD_JPY:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -280.8, 'repair_replay_pl_jpy': -280.8, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'NO_PROFIT_CANDIDATE'}]}]`
- `top_repair_replay_residual_method_rollups`: `[{'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'method': 'RANGE_ROTATION', 'group_count': 6, 'pair_count': 5, 'pairs': ['AUD_USD', 'EUR_JPY', 'GBP_USD', 'NZD_USD', 'USD_JPY'], 'side_count': 2, 'sides': ['LONG', 'SHORT'], 'exit_reasons': {'STOP_LOSS_ORDER': 6, 'MARKET_ORDER_TRADE_CLOSE': 1}, 'loss_closes': 7, 'actual_pl_jpy': -8337.1834, 'repair_replay_pl_jpy': -8337.1834, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'BELOW_TP_PROGRESS_GATE': 4, 'NO_PROFIT_CANDIDATE': 3}, 'examples': [{'trade_id': '472952', 'lane_id': 'range_trader:AUD_USD:LONG:RANGE_ROTATION', 'actual_pl_jpy': -2690.6967, 'repair_replay_pl_jpy': -2690.6967, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'BELOW_TP_PROGRESS_GATE', 'pair': 'AUD_USD', 'side': 'LONG', 'method': 'RANGE_ROTATION'}, {'trade_id': '472834', 'lane_id': 'range_trader:AUD_USD:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -1705.6738, 'repair_replay_pl_jpy': -1705.6738, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'NO_PROFIT_CANDIDATE', 'pair': 'AUD_USD', 'side': 'SHORT', 'method': 'RANGE_ROTATION'}, {'trade_id': '472743', 'lane_id': 'range_trader:NZD_USD:LONG:RANGE_ROTATION', 'actual_pl_jpy': -1380.8008, 'repair_replay_pl_jpy': -1380.8008, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'NO_PROFIT_CANDIDATE', 'pair': 'NZD_USD', 'side': 'LONG', 'method': 'RANGE_ROTATION'}]}]`
- `top_tp_progress_repair_residual_method_rollups`: `[]`
- `top_entry_quality_residual_method_rollups`: `[{'residual_scope': 'ENTRY_QUALITY_OR_CLOSE_RESIDUAL', 'method': 'RANGE_ROTATION', 'group_count': 6, 'pair_count': 5, 'pairs': ['AUD_USD', 'EUR_JPY', 'GBP_USD', 'NZD_USD', 'USD_JPY'], 'side_count': 2, 'sides': ['LONG', 'SHORT'], 'exit_reasons': {'STOP_LOSS_ORDER': 6, 'MARKET_ORDER_TRADE_CLOSE': 1}, 'loss_closes': 7, 'actual_pl_jpy': -8337.1834, 'repair_replay_pl_jpy': -8337.1834, 'repair_replay_delta_jpy': 0.0, 'repair_replay_triggered': 0, 'block_reasons': {'BELOW_TP_PROGRESS_GATE': 4, 'NO_PROFIT_CANDIDATE': 3}, 'examples': [{'trade_id': '472952', 'lane_id': 'range_trader:AUD_USD:LONG:RANGE_ROTATION', 'actual_pl_jpy': -2690.6967, 'repair_replay_pl_jpy': -2690.6967, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'BELOW_TP_PROGRESS_GATE', 'pair': 'AUD_USD', 'side': 'LONG', 'method': 'RANGE_ROTATION'}, {'trade_id': '472834', 'lane_id': 'range_trader:AUD_USD:SHORT:RANGE_ROTATION', 'actual_pl_jpy': -1705.6738, 'repair_replay_pl_jpy': -1705.6738, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'NO_PROFIT_CANDIDATE', 'pair': 'AUD_USD', 'side': 'SHORT', 'method': 'RANGE_ROTATION'}, {'trade_id': '472743', 'lane_id': 'range_trader:NZD_USD:LONG:RANGE_ROTATION', 'actual_pl_jpy': -1380.8008, 'repair_replay_pl_jpy': -1380.8008, 'repair_replay_triggered': False, 'repair_replay_block_reason': 'NO_PROFIT_CANDIDATE', 'pair': 'NZD_USD', 'side': 'LONG', 'method': 'RANGE_ROTATION'}]}]`
- `avg_decision_lag_minutes_after_first_positive`: `93.57`
- `max_decision_lag_minutes_after_first_positive`: `300.28`
- `market_closes_audited`: `6`
- `market_closes_post_close_continued`: `3`
- `market_closes_post_close_continued_rate`: `0.5`
- `market_closes_post_close_adverse`: `3`
- `market_closes_post_close_adverse_rate`: `0.5`
- `market_closes_tp_touched_after_close`: `3`
- `market_closes_tp_touched_after_close_rate`: `0.5`
- `market_closes_sl_touched_after_close`: `2`
- `market_closes_sl_touched_after_close_rate`: `0.3333`
- `market_close_estimated_followthrough_jpy`: `5158.3377`
- `market_close_estimated_avoided_adverse_jpy`: `7808.0`
- `profit_market_closes_audited`: `5`
- `profit_market_closes_left_runner_upside`: `2`
- `profit_market_closes_avoided_giveback`: `3`
- `loss_market_closes_audited`: `1`
- `loss_market_closes_may_have_been_premature`: `1`
- `loss_market_closes_contained_risk`: `0`

## Canceled Order Regret By Shape

| pair | side | method | type | priority | orders | entry touch | TP touch | missed MFE JPY |
|---|---|---|---|---|---:|---:|---:|---:|
| `GBP_JPY` | `SHORT` | `RANGE_ROTATION` | `LIMIT_ORDER` | `PRESERVE_PENDING_THESIS_TP_TOUCHED` | 1 | 1 | 1 | 630.0 |
| `EUR_USD` | `LONG` | `BREAKOUT_FAILURE` | `LIMIT_ORDER` | `REPRICE_OR_EXTEND_TTL_ENTRY_TOUCHED` | 3 | 1 | 0 | 84.3022 |

## Top Canceled Order Regrets

| Order | Lane | Pair | Side | Entry touch min | TP touch min | MFE pips | Est MFE JPY |
|---|---|---|---|---:|---:|---:|---:|
| `472922` | `range_trader:GBP_JPY:SHORT:RANGE_ROTATION` | `GBP_JPY` | `SHORT` | `1.39` | `156.39` | `21.0` | `630.0` |
| `472916` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `LONG` | `31.0` | `None` | `1.3` | `84.3022` |
| `472895` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `LONG` | `None` | `None` | `0.0` | `0.0` |
| `472892` | `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `LONG` | `None` | `None` | `0.0` | `0.0` |

## Top Loss Close Timing Regrets

| Trade | Lane | Pair | Side | Exit | PL JPY | First plus min | Lag min | MFE pips | TP progress | Capture missed | Repair replay | Repair block | Repair JPY | Repair delta | Est MFE JPY | TP touched |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|
| `472837` | `range_trader:GBP_USD:SHORT:RANGE_ROTATION` | `GBP_USD` | `SHORT` | `STOP_LOSS_ORDER` | `-971.0121` | `19.8` | `41.42` | `2.5` | `0.1799` | `False` | `False` | `BELOW_TP_PROGRESS_GATE` | `None` | `None` | `218.8614` | `False` |
| `472792` | `range_trader:USD_JPY:SHORT:RANGE_ROTATION` | `USD_JPY` | `SHORT` | `STOP_LOSS_ORDER` | `-340.2` | `2.88` | `42.99` | `3.4` | `0.6071` | `True` | `True` | `` | `126.0` | `466.2` | `214.2` | `False` |
| `472952` | `range_trader:AUD_USD:LONG:RANGE_ROTATION` | `AUD_USD` | `LONG` | `STOP_LOSS_ORDER` | `-2690.6967` | `50.87` | `300.28` | `0.4` | `0.0342` | `False` | `False` | `BELOW_TP_PROGRESS_GATE` | `None` | `None` | `90.787` | `False` |
| `472900` | `range_trader:USD_JPY:SHORT:RANGE_ROTATION` | `USD_JPY` | `SHORT` | `STOP_LOSS_ORDER` | `-464.0` | `5.28` | `64.32` | `0.8` | `0.1379` | `False` | `False` | `BELOW_TP_PROGRESS_GATE` | `None` | `None` | `64.0` | `False` |
| `472903` | `range_trader:EUR_JPY:SHORT:RANGE_ROTATION` | `EUR_JPY` | `SHORT` | `STOP_LOSS_ORDER` | `-844.2` | `13.83` | `18.84` | `0.3` | `0.0207` | `False` | `False` | `BELOW_TP_PROGRESS_GATE` | `None` | `None` | `18.9` | `False` |
| `472834` | `range_trader:AUD_USD:SHORT:RANGE_ROTATION` | `AUD_USD` | `SHORT` | `STOP_LOSS_ORDER` | `-1705.6738` | `None` | `None` | `0.0` | `0.0` | `False` | `False` | `NO_PROFIT_CANDIDATE` | `None` | `None` | `0.0` | `False` |
| `472775` | `range_trader:USD_JPY:SHORT:RANGE_ROTATION` | `USD_JPY` | `SHORT` | `STOP_LOSS_ORDER` | `-280.8` | `None` | `None` | `0.0` | `0.0` | `False` | `False` | `NO_PROFIT_CANDIDATE` | `None` | `None` | `0.0` | `False` |
| `472743` | `range_trader:NZD_USD:LONG:RANGE_ROTATION` | `NZD_USD` | `LONG` | `MARKET_ORDER_TRADE_CLOSE` | `-1380.8008` | `None` | `None` | `0.0` | `0.0` | `False` | `False` | `NO_PROFIT_CANDIDATE` | `None` | `None` | `0.0` | `False` |

## Top Market Close Counterfactuals

| Trade | Lane | Pair | Side | Gateway | PL JPY | Label | Fav pips | Fav JPY | Adv pips | Adv JPY | TP after | SL after |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| `472876` | `range_trader:EUR_JPY:SHORT:RANGE_ROTATION` | `EUR_JPY` | `SHORT` | `` | `413.0` | `PROFIT_CLOSE_AVOIDED_GIVEBACK` | `3.4` | `238.0` | `90.2` | `6314.0` | `False` | `True` |
| `472819` | `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT` | `EUR_USD` | `SHORT` | `` | `218.0282` | `PROFIT_CLOSE_LEFT_RUNNER_UPSIDE` | `25.9` | `2099.4487` | `14.8` | `1199.685` | `True` | `False` |
| `472743` | `range_trader:NZD_USD:LONG:RANGE_ROTATION` | `NZD_USD` | `LONG` | `` | `-1380.8008` | `LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE` | `12.9` | `2007.689` | `0.7` | `108.9444` | `False` | `False` |
| `472840` | `range_trader:USD_JPY:LONG:RANGE_ROTATION` | `USD_JPY` | `LONG` | `` | `432.0` | `PROFIT_CLOSE_LEFT_RUNNER_UPSIDE` | `14.6` | `1051.2` | `4.1` | `295.2` | `True` | `False` |
| `472848` | `range_trader:EUR_JPY:SHORT:RANGE_ROTATION` | `EUR_JPY` | `SHORT` | `` | `441.0` | `PROFIT_CLOSE_AVOIDED_GIVEBACK` | `4.9` | `343.0` | `13.0` | `910.0` | `False` | `False` |
| `472861` | `range_trader:USD_JPY:SHORT:RANGE_ROTATION` | `USD_JPY` | `SHORT` | `` | `224.0` | `PROFIT_CLOSE_AVOIDED_GIVEBACK` | `3.8` | `304.0` | `7.3` | `584.0` | `True` | `True` |

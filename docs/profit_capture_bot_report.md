# Profit Capture Bot Report

- Generated at UTC: `2026-07-05T18:09:16.879182+00:00`
- Status: `PROFIT_CAPTURE_BLOCKED`
- Read only: `True`
- Live side effects: `0`

## Metrics

| Metric | Value |
|---|---:|
| `open_trader_positions` | `0` |
| `bankable_positions` | `0` |
| `blocked_positions` | `0` |
| `watch_positions` | `0` |
| `historical_missed_loss_closes` | `14` |
| `historical_estimated_gap_jpy` | `5485.238` |
| `historical_actual_loss_close_pl_jpy` | `-39275.343` |
| `historical_counterfactual_profit_capture_pl_jpy` | `-20546.009` |
| `historical_counterfactual_profit_capture_delta_jpy` | `18729.334` |
| `historical_counterfactual_profit_capture_jpy` | `3545.239` |
| `historical_repair_replay_contract` | `TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1` |
| `historical_repair_replay_contract_present` | `True` |
| `historical_repair_replay_triggered` | `13` |
| `historical_repair_replay_profit_capture_jpy` | `3827.122` |
| `historical_repair_replay_counterfactual_pl_jpy` | `-20503.606` |
| `historical_repair_replay_delta_jpy` | `18771.737` |

## Open Trader Positions

- none

## Historical Misses

- Missed loss closes: `14`
- Estimated gap JPY: `5485.238`
- Actual loss-close PL JPY: `-39275.343`
- Counterfactual profit-capture PL JPY: `-20546.009`
- Counterfactual profit-capture delta JPY: `18729.334`
- Production-gate replay triggers: `13`
- Production-gate replay contract: `TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1`
- Production-gate replay contract present: `True`
- Production-gate replay delta JPY: `18771.737`
- Production-gate replay block reasons: `{'BELOW_NOISE_FLOOR': 1}`
- `472792` `USD_JPY` `SHORT` STOP_LOSS_ORDER repair_at=`2026-06-22T06:45:00+00:00` repair_jpy=`126.0` delta=`466.2`
- `472318` `USD_CAD` `LONG` MARKET_ORDER_TRADE_CLOSE repair_at=`2026-06-12T09:09:00+00:00` repair_jpy=`665.02` delta=`830.189`
- `472280` `EUR_USD` `LONG` MARKET_ORDER_TRADE_CLOSE repair_at=`2026-06-11T22:45:00+00:00` repair_jpy=`358.987` delta=`387.863`
- `472222` `GBP_CHF` `LONG` MARKET_ORDER_TRADE_CLOSE repair_at=`2026-06-11T17:34:00+00:00` repair_jpy=`298.896` delta=`1280.69`
- `472230` `EUR_AUD` `LONG` MARKET_ORDER_TRADE_CLOSE repair_at=`2026-06-11T17:03:00+00:00` repair_jpy=`116.701` delta=`123.444`
- `472632` `AUD_NZD` `SHORT` MARKET_ORDER_TRADE_CLOSE repair_block=`BELOW_NOISE_FLOOR` candidate=`4.7`p noise=`4.9667`p
- `472792` `USD_JPY` `SHORT` STOP_LOSS_ORDER realized=`-340.2` counterfactual=`105.84` delta=`446.04`
- `472632` `AUD_NZD` `SHORT` MARKET_ORDER_TRADE_CLOSE realized=`-239.479` counterfactual=`367.691` delta=`607.17`
- `472318` `USD_CAD` `LONG` MARKET_ORDER_TRADE_CLOSE realized=`-165.169` counterfactual=`612.183` delta=`777.352`
- `472280` `EUR_USD` `LONG` MARKET_ORDER_TRADE_CLOSE realized=`-28.877` counterfactual=`311.445` delta=`340.322`
- `472222` `GBP_CHF` `LONG` MARKET_ORDER_TRADE_CLOSE realized=`-981.794` counterfactual=`285.745` delta=`1267.539`

## Blockers

- `P0` `HISTORICAL_PROFIT_CAPTURE_MISSED`: 14 recent loss close(s) missed executable profit capture; production-gate replay triggers=13 delta=18771.737 JPY

## Operator Actions

- `REFRESH_PROFIT_CAPTURE_BOT` approval=`no`: `PYTHONPATH=src python3 -m quant_rabbit.cli profit-capture-bot`
- `REFRESH_EXECUTION_TIMING_AUDIT` approval=`no`: `PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80`

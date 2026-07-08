# EUR_USD SHORT BREAKOUT_FAILURE STOP HARVEST Replay

Generated: `2026-07-08T09:27:13.917942Z`

## Verdict

- Status: `STOP_HARVEST_EXACT_S5_BIDASK_REPLAY_FAILED_BLOCKED`
- Target shape: `EUR_USD|SHORT|BREAKOUT_FAILURE|STOP|HARVEST`
- Read-only: `True`
- Live permission allowed: `False`
- Scout candidate after replay: `False`
- S5 trigger/TP path passed: `False`

The STOP-only observed OANDA transaction replay is positive: `7` wins, `0` losses, net `6307.2357` JPY, expectancy `901.0337` JPY/trade. Independent S5 bid/ask path replay confirms `7/7` STOP triggers but only `4/7` active TP touches. This is not SCOUT-ready and does not create live permission.

## Boundary

- STOP samples only: `True`
- MARKET samples mixed into STOP proof: `False`
- LIMIT samples mixed into STOP proof: `False`
- Market-close losses mixed into HARVEST proof: `False`

## Replay Basis

SHORT STOP trigger is checked on S5 `bid_low <= trigger`. SHORT attached TP is checked on S5 `ask_low <= active TAKE_PROFIT_ORDER`. TP replacements are replayed as a schedule; the final TP is not assumed to have been active from entry.

## S5 Summary

- Trigger touches: `7/7`
- TP touches after trigger: `4/7`
- S5 wins/losses: `4/3`
- TP path missing trade IDs: `['471306', '471492', '471495']`
- Transaction/S5 mismatch status: `OANDA_TRANSACTION_TP_FILLS_NOT_FULLY_RECONSTRUCTED_BY_S5_BA_CANDLES`
- S5 TP miss distance pips: `{'min': 0.2, 'median': 0.2, 'max': 0.4}`
- Post-trigger MAE pips: `{'min': 1.6, 'median': 10.4, 'max': 61.1}`
- Post-trigger MFE pips: `{'min': 1.7, 'median': 6.1, 'max': 13.3}`
- Max observed path adverse JPY: `2628.0174`

## Samples

| trade_id | stop_order | trigger | S5 trigger | S5 TP | realized JPY | MAE pips | MFE pips |
|---|---:|---:|---|---|---:|---:|---:|
| 471292 | 471291 | 1.16077 | 2026-05-20T00:38:00.000000000Z | 2026-05-20T05:47:30.000000000Z | 4572.2661 | 1.6 | 13.3 |
| 471306 | 471305 | 1.1596 | 2026-05-20T11:07:40.000000000Z | None | 784.1109 | 10.4 | 3.4 |
| 471345 | 471344 | 1.16244 | 2026-05-20T23:18:45.000000000Z | 2026-05-21T02:19:45.000000000Z | 44.4033 | 11.8 | 1.7 |
| 471403 | 471402 | 1.16007 | 2026-05-21T12:56:25.000000000Z | 2026-05-21T13:08:00.000000000Z | 117.5561 | 2.0 | 3.7 |
| 471460 | 471459 | 1.16144 | 2026-05-22T05:15:50.000000000Z | 2026-05-22T05:58:15.000000000Z | 92.1091 | 3.4 | 6.1 |
| 471492 | 471491 | 1.16013 | 2026-05-22T15:30:45.000000000Z | None | 348.3951 | 61.1 | 7.8 |
| 471495 | 471494 | 1.16013 | 2026-05-22T15:49:55.000000000Z | None | 348.3951 | 61.1 | 7.8 |

## Slippage Sensitivity

| extra adverse pips | net JPY | expectancy JPY/trade | wins | losses | min trade JPY |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 6307.2357 | 901.0337 | 7 | 0 | 44.4033 |
| 0.1 | 6235.1591 | 890.737 | 7 | 0 | 41.2316 |
| 0.2 | 6163.0825 | 880.4404 | 7 | 0 | 38.06 |
| 0.5 | 5946.8528 | 849.5504 | 7 | 0 | 28.545 |
| 1.0 | 5586.4699 | 798.0671 | 7 | 0 | 12.6867 |
| 1.5 | 5226.0869 | 746.5838 | 6 | 1 | -3.1717 |
| 2.0 | 4865.704 | 695.1006 | 6 | 1 | -19.03 |

## Invalidation

- Status: `S5_PATH_REPLAYED_BUT_SCOUT_INVALIDATION_NOT_DEFINED`
- Scout gap: The 7 historical OANDA transaction winners are useful broker-side evidence, but they do not by themselves prove a complete independent S5 trigger-to-TP path or define a production pre-trigger stop-chase cutoff, max trigger slippage, post-trigger invalidation level, or max loss cap.

## Remaining Blockers

- `STOP_S5_TRIGGER_OR_TP_PATH_REPLAY_FAILED`: BLOCKING_SCOUT_AND_LIVE_PERMISSION - At least one STOP sample lacks independent S5 trigger touch or active TP touch.
- `S5_TP_PATH_DOES_NOT_RECONSTRUCT_OBSERVED_TP_FILLS`: BLOCKING_SCOUT_AND_LIVE_PERMISSION - OANDA transaction fullPrice confirms TP exits, but independent S5 bid/ask candles miss active TP touches for trades ['471306', '471492', '471495'].
- `STOP_SAMPLE_COUNT_THIN_FOR_LIVE_GRADE`: BLOCKING_LIVE_PERMISSION - STOP_HARVEST has only 7 STOP samples; even a fully reconstructed replay would not be live-grade proof.
- `STOP_TRIGGER_INVALIDATION_NOT_SCOUT_READY`: BLOCKING_SCOUT - Pre-trigger stop-chase, max trigger slippage, post-trigger invalidation, and max loss cap are not production-defined.
- `GLOBAL_NEGATIVE_EXPECTANCY_ACTIVE`: VISIBLE_GLOBAL_BLOCKER - This STOP packet does not clear global capture-economics or self-improvement blockers.
- `MARKET_CLOSE_LEAK_PRESENT_EXCLUDED`: VISIBLE_TARGET_SHAPE_BLOCKER - Market-close leakage remains excluded from this STOP/TP proof and must not be mixed into HARVEST proof.
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`: VISIBLE_GLOBAL_BLOCKER - Existing profitability/harvest artifacts keep the month-scale residual blocker visible.
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`: BLOCKING_ROUTING - Normal routing remains blocked until guardian receipt/operator-review state is cleared by its own contract.
- `NO_LIVE_READY_PORTFOLIO`: BLOCKING_LIVE_PERMISSION - Portfolio path still has no live-ready permission for STOP_HARVEST.
- `NO_FRESH_GATEWAY_PERMISSION`: BLOCKING_LIVE_PERMISSION - No verifier/gateway packet exists for STOP_HARVEST.

## Validation

```bash
python3 -m json.tool data/eurusd_short_breakout_failure_stop_harvest_replay.json >/dev/null
PYTHONPATH=src python3 -m unittest tests.test_eurusd_stop_harvest_replay -v
PYTHONPATH=src python3 -m unittest tests.test_live_runtime_sync -v
```

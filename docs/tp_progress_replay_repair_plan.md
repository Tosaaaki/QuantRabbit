# TP Progress Replay Repair Plan

- Generated: `2026-07-05T15:18:41Z`
- Blocker: `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`
- Fresh entries blocked: `True`

## Current Evidence

- `window_lookback_hours`: `744.0`
- `loss_closes_profit_capture_missed`: `14`
- `loss_closes_repair_replay_triggered`: `13`
- `counterfactual_profit_capture_delta_jpy`: `18770.7603`
- `counterfactual_profit_capture_jpy`: `3826.1448`
- `repair_replay_counterfactual_pl_jpy`: `-20504.5826`
- `active_counterfactual_profit_capture_pl_jpy`: `-20504.5826`
- `raw_counterfactual_profit_capture_pl_jpy`: `-20547.4015`

## Residual Groups

| pair | side | strategy | exit | diagnosis | replay P/L JPY | block reasons |
|---|---|---|---|---|---:|---|
| GBP_USD | LONG | BREAKOUT_FAILURE | MARKET_ORDER_TRADE_CLOSE | entry_quality_or_premature_exit_below_tp_progress_gate | -2981.8961 | {"BELOW_TP_PROGRESS_GATE": 1} |
| AUD_USD | LONG | RANGE_ROTATION | STOP_LOSS_ORDER | entry_quality_or_premature_exit_below_tp_progress_gate | -2690.6967 | {"BELOW_TP_PROGRESS_GATE": 1} |
| EUR_USD | LONG | RANGE_ROTATION | MARKET_ORDER_TRADE_CLOSE | entry_quality_or_premature_exit_below_tp_progress_gate | -2333.8215 | {"BELOW_TP_PROGRESS_GATE": 1} |
| EUR_USD | SHORT | RANGE_ROTATION | MARKET_ORDER_TRADE_CLOSE | bad_entry_or_no_positive_excursion | -2181.1565 | {"NO_PROFIT_CANDIDATE": 1} |
| NZD_CAD | SHORT | RANGE_ROTATION | MARKET_ORDER_TRADE_CLOSE | bad_entry_or_no_positive_excursion | -2044.4543 | {"NO_PROFIT_CANDIDATE": 2} |
| AUD_USD | SHORT | RANGE_ROTATION | STOP_LOSS_ORDER | bad_entry_or_no_positive_excursion | -1705.6738 | {"NO_PROFIT_CANDIDATE": 1} |
| NZD_USD | LONG | RANGE_ROTATION | MARKET_ORDER_TRADE_CLOSE | bad_entry_or_no_positive_excursion | -1380.8008 | {"NO_PROFIT_CANDIDATE": 1} |
| EUR_CHF | LONG | TREND_CONTINUATION | MARKET_ORDER_TRADE_CLOSE | entry_quality_or_premature_exit_below_tp_progress_gate | -1272.0771 | {"BELOW_TP_PROGRESS_GATE": 2} |
| EUR_JPY | LONG | RANGE_ROTATION | MARKET_ORDER_TRADE_CLOSE | bad_entry_or_no_positive_excursion | -1071.9 | {"NO_PROFIT_CANDIDATE": 1} |
| GBP_USD | SHORT | RANGE_ROTATION | STOP_LOSS_ORDER | entry_quality_or_premature_exit_below_tp_progress_gate | -971.0121 | {"BELOW_TP_PROGRESS_GATE": 1} |

## Clearing Filter

- Required replay improvement: `20504.5826` JPY
- Filter: Block or repair matching pair/side/method residual groups with BELOW_TP_PROGRESS_GATE, NO_PROFIT_CANDIDATE, or BELOW_NOISE_FLOOR until the 744h replay is non-negative.
- Manual EUR_USD `472987` excluded: `True`

## Proof Before Live

- rerun execution-timing-audit --lookback-hours 744 --post-close-hours 6
- rerun profitability-acceptance and confirm MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE cleared or matching residual group disappeared
- show exact local TP proof for the candidate pair/side/method/exit shape
- show close-gate evidence for any loss-side market close path
- show RiskEngine, LiveOrderGateway, and fresh GPT-5.5 TRADE/ADD receipt after evidence clears

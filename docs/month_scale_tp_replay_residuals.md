# Month-Scale TP Replay Residuals

- Generated: `2026-07-05T16:24:00Z`
- Blocker: `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`
- Replay window: `{"from_utc": "2026-05-26T07:03:46.222763+00:00", "lookback_hours": 744.0, "post_close_hours": 6.0, "to_utc": "2026-07-03T20:08:53.084075+00:00"}`
- Baseline actual loss-close P/L JPY: `-39275.3429`
- Improved P/L after missed-capture repair JPY: `-20504.5826`
- Current residual P/L JPY: `-20504.5826`
- Manual EUR_USD `472987` excluded: `True`

## Residual Groups

| pair | side | strategy | exit | repair P/L JPY | loss closes | family | block reasons | examples |
|---|---|---|---|---:|---:|---|---|---|
| GBP_USD | LONG | BREAKOUT_FAILURE | MARKET_ORDER_TRADE_CLOSE | -2981.8961 | 1 | BAD_ENTRY_OR_BAD_EXIT_BELOW_TP_PROGRESS_GATE | {"BELOW_TP_PROGRESS_GATE": 1} | 472071 |
| AUD_USD | LONG | RANGE_ROTATION | STOP_LOSS_ORDER | -2690.6967 | 1 | BAD_ENTRY_OR_BAD_EXIT_BELOW_TP_PROGRESS_GATE | {"BELOW_TP_PROGRESS_GATE": 1} | 472952 |
| EUR_USD | LONG | RANGE_ROTATION | MARKET_ORDER_TRADE_CLOSE | -2333.8215 | 1 | BAD_ENTRY_OR_BAD_EXIT_BELOW_TP_PROGRESS_GATE | {"BELOW_TP_PROGRESS_GATE": 1} | 471817 |
| EUR_USD | SHORT | RANGE_ROTATION | MARKET_ORDER_TRADE_CLOSE | -2181.1565 | 1 | BAD_ENTRY_NO_PROFIT_CANDIDATE | {"NO_PROFIT_CANDIDATE": 1} | 471711 |
| NZD_CAD | SHORT | RANGE_ROTATION | MARKET_ORDER_TRADE_CLOSE | -2044.4543 | 2 | BAD_ENTRY_NO_PROFIT_CANDIDATE | {"NO_PROFIT_CANDIDATE": 2} | 472380,472312 |
| AUD_USD | SHORT | RANGE_ROTATION | STOP_LOSS_ORDER | -1705.6738 | 1 | BAD_ENTRY_NO_PROFIT_CANDIDATE | {"NO_PROFIT_CANDIDATE": 1} | 472834 |
| NZD_USD | LONG | RANGE_ROTATION | MARKET_ORDER_TRADE_CLOSE | -1380.8008 | 1 | BAD_ENTRY_NO_PROFIT_CANDIDATE | {"NO_PROFIT_CANDIDATE": 1} | 472743 |
| EUR_CHF | LONG | TREND_CONTINUATION | MARKET_ORDER_TRADE_CLOSE | -1272.0771 | 2 | BAD_ENTRY_OR_BAD_EXIT_BELOW_TP_PROGRESS_GATE | {"BELOW_TP_PROGRESS_GATE": 2} | 472445,472174 |
| EUR_JPY | LONG | RANGE_ROTATION | MARKET_ORDER_TRADE_CLOSE | -1071.9 | 1 | BAD_ENTRY_NO_PROFIT_CANDIDATE | {"NO_PROFIT_CANDIDATE": 1} | 472094 |
| GBP_USD | SHORT | RANGE_ROTATION | STOP_LOSS_ORDER | -971.0121 | 1 | BAD_ENTRY_OR_BAD_EXIT_BELOW_TP_PROGRESS_GATE | {"BELOW_TP_PROGRESS_GATE": 1} | 472837 |

## Rollup

- Bad entry / bad exit / missed capture: `{"BAD_ENTRY_NO_PROFIT_CANDIDATE": {"examples": ["471711", "472380", "472312", "472834", "472743"], "groups": 5, "loss_closes": 6, "repair_replay_pl_jpy": -8383.9854}, "BAD_ENTRY_OR_BAD_EXIT_BELOW_TP_PROGRESS_GATE": {"examples": ["472071", "472952", "471817", "472445", "472174"], "groups": 5, "loss_closes": 6, "repair_replay_pl_jpy": -10249.5035}}`
- Required improvement to non-negative: `20504.5826` JPY
- Known top residual abs loss: `18633.4889` JPY
- Additional tail loss to cover: `1871.0937` JPY
- Filter: Block or repair every matching pair/side/method residual group with NO_PROFIT_CANDIDATE, BELOW_TP_PROGRESS_GATE, or BELOW_NOISE_FLOOR until the full 744h replay is non-negative; the top residual table alone is not permission evidence.

## Gate Definitions

- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`: Rerun execution-timing-audit --lookback-hours 744 --post-close-hours 6 and profitability-acceptance; blocker clears only when replay P/L is non-negative or matching residual groups disappear. Can create A/S permission: `False`

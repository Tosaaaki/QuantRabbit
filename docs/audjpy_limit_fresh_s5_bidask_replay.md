# AUD_JPY LIMIT Fresh S5 Bid/Ask Replay

- Generated: `2026-07-06T10:10:06.406105Z`
- Classification: `EVIDENCE_GAP`
- Pair/side/method/order type/exit shape: `AUD_JPY` `SHORT` `BREAKOUT_FAILURE` `LIMIT` `TP_PROOF_COLLECTION_HARVEST`
- Replay window: `2026-05-14T15:15:47.378998Z` to `2026-06-22T05:47:46.506094Z`
- Samples: `135`
- Active days: `6`
- Max daily sample share: `0.9111`
- Positive day rate: `0.3333`
- Spread-included net P/L: `116.3` pips
- Expectancy: `0.861481` pips/trade
- Avg win/loss: `7.967164` / `6.231343` pips
- Max loss: `-7.0` pips
- Wilson 95% lower win rate: `0.413231`
- Pessimistic expectancy: `-0.364079` pips/trade
- Meets exact S5 proof thresholds: `False`

## Thresholds

- `sample_count_floor`: `True`
- `active_day_floor`: `True`
- `daily_stability_floor`: `False`
- `positive_day_rate_floor`: `False`
- `spread_included_expectancy_positive`: `True`

## Daily Distribution

- `2026-05-15` samples `123` realized `146.7` pips avg `1.1927` win_rate `0.5203`
- `2026-05-21` samples `6` realized `-15.0` pips avg `-2.5` win_rate `0.1667`
- `2026-06-02` samples `2` realized `-4.4` pips avg `-2.2` win_rate `0.5`
- `2026-06-05` samples `1` realized `-7.0` pips avg `-7.0` win_rate `0.0`
- `2026-06-08` samples `1` realized `-7.0` pips avg `-7.0` win_rate `0.0`
- `2026-06-22` samples `2` realized `3.0` pips avg `1.5` win_rate `0.5`

## Remaining Blockers

- `S5_DAILY_SAMPLE_CONCENTRATED`
- `S5_POSITIVE_DAY_RATE_LOW`
- `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`
- `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`
- `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- `FRESH_GPT_VERIFIER_TRADE_RECEIPT_MISSING`
- `LIVE_ORDER_GATEWAY_PREFLIGHT_MISSING`
- `RISK_ENGINE_PASS_MISSING`
- `S5_BIDASK_SPREAD_INCLUDED_REPLAY_NOT_PROVEN_FOR_LIVE`
- `FORECAST_EXECUTABLE_PROOF_STILL_REQUIRED_BEFORE_LIVE`
- `PROFITABILITY_ACCEPTANCE_BLOCKED`

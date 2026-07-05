# TP Progress Harvest Gate Evidence

- generated_at_utc: `2026-07-05T15:01:28.977313+00:00`
- missed captures: `14`
- current-rule triggers: `13`
- executable before loss close: `13`
- below noise floor: `1`
- attribution: system_gateway=`13`, unknown_lane=`1`, manual=`0`

## Contract

- Historical repair evidence is allowed as evidence.
- Historical repair evidence does not by itself create live A/S permission.
- Future TP-progress harvest requires trigger-before-loss-close, above-noise-floor proof, and production replay gate status.

## Month-Scale Replay

- current baseline P/L: `-39275.3429` JPY
- current improved P/L: `-20504.5826` JPY
- proposed-gate baseline P/L: `-39246.4662` JPY
- proposed-gate improved P/L: `-20863.5316` JPY
- proposed-gate residual P/L: `-20863.5316` JPY
- MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE clears: `False`

## Residual Groups

| pair | side | method | rows | residual_jpy | blockers |
| --- | --- | --- | ---: | ---: | --- |
| GBP_USD | LONG | BREAKOUT_FAILURE | 1 | -2981.8961 | BELOW_TP_PROGRESS_GATE:1 |
| AUD_USD | LONG | RANGE_ROTATION | 1 | -2690.6967 | BELOW_TP_PROGRESS_GATE:1 |
| EUR_USD | LONG | RANGE_ROTATION | 1 | -2333.8215 | BELOW_TP_PROGRESS_GATE:1 |
| EUR_USD | SHORT | RANGE_ROTATION | 1 | -2181.1565 | NO_PROFIT_CANDIDATE:1 |
| NZD_CAD | SHORT | RANGE_ROTATION | 2 | -2044.4543 | NO_PROFIT_CANDIDATE:2 |
| AUD_USD | SHORT | RANGE_ROTATION | 1 | -1705.6738 | NO_PROFIT_CANDIDATE:1 |
| NZD_USD | LONG | RANGE_ROTATION | 1 | -1380.8008 | NO_PROFIT_CANDIDATE:1 |
| EUR_CHF | LONG | TREND_CONTINUATION | 2 | -1272.0771 | BELOW_TP_PROGRESS_GATE:2 |
| EUR_JPY | LONG | RANGE_ROTATION | 1 | -1071.9 | NO_PROFIT_CANDIDATE:1 |
| GBP_USD | SHORT | RANGE_ROTATION | 1 | -971.0121 | BELOW_TP_PROGRESS_GATE:1 |

## Trigger Evidence

| trade_id | pair | side | method | attribution | status | executable | actual_jpy | replay_jpy |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 472792 | USD_JPY | SHORT | RANGE_ROTATION | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -340.2 | 126.0 |
| 472632 | AUD_NZD | SHORT | RANGE_ROTATION | SYSTEM_GATEWAY | BELOW_NOISE_FLOOR | False | -239.4791 | -239.4791 |
| 472318 | USD_CAD | LONG | BREAKOUT_FAILURE | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -165.1694 | 664.9147 |
| 472280 | EUR_USD | LONG | BREAKOUT_FAILURE | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -28.8767 | 358.949 |
| 472222 | GBP_CHF | LONG | TREND_CONTINUATION | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -981.7942 | 298.7055 |
| 472230 | EUR_AUD | LONG | TREND_CONTINUATION | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -6.7428 | 116.6358 |
| 472149 | EUR_GBP | SHORT | TREND_CONTINUATION | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -116.1114 | 58.2976 |
| 472109 | USD_CAD | LONG | BREAKOUT_FAILURE | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -172.863 | 210.6322 |
| 472037 | AUD_NZD | SHORT | TREND_CONTINUATION | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -462.3796 | 95.9401 |
| 471923 | GBP_JPY | LONG | TREND_CONTINUATION | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -238.0 | 62.0 |
| 471414 | EUR_USD | SHORT | BREAKOUT_FAILURE | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -2642.1057 | 198.8771 |
| 471979 | AUD_CHF | LONG | BREAKOUT_FAILURE | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -1215.4046 | 398.274 |
| 471240 | EUR_USD | LONG | UNKNOWN | SYSTEM_GATEWAY_UNKNOWN_LANE | CURRENT_RULE_TRIGGER | True | -5267.5461 | 897.3725 |
| 471232 | EUR_USD | LONG | TREND_CONTINUATION | SYSTEM_GATEWAY | CURRENT_RULE_TRIGGER | True | -3307.422 | 339.5463 |

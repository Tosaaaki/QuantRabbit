# Fast-bot raw-shock / exit-architecture replay — 2026-08-29

## Authority and truth

- Diagnostic shadow evidence only. `execution_authority=NONE`, broker mutation 0, external order attempts 0, external orders 0.
- EUR/USD OANDA M1 bid/ask, 2020-01-01 22:00 UTC through 2026-07-09 plus a GET-only 2026-08-28 UTC slice.
- Historical rows: 2,398,149 (97.901% expected-market-minute coverage in the existing inventory). 2026-08-28 rows: 1,260. Combined deduplicated rows: 2,399,409.
- The onset detector is causal raw price truth: 15-minute displacement of at least 18 pips plus at least two confirmations from three-minute velocity, acceleration, spread expansion, and prior-swing break. ATR is not an onset trigger. It is retained only for normalization and an auxiliary catastrophe-width upper bound.
- Raw detection produced 3,001 non-overlapping episodes. Thirty lacked a causal completed-M5 ATR needed only for the ATR comparison arm, so all four arms use the same remaining 2,971 episodes. Volume was not used as a liquidity proxy. Spread is included through executable bid/ask.

## Side-relative episodes

| Measure | Result |
|---|---:|
| Compared episodes | 2,971 |
| UP / DOWN | 1,495 / 1,476 |
| Failed continuation at 5m | 750 (25.24%) |
| 30m retrace >= 50% | 1,143 (38.47%) |

The 2026-08-28 episode remains 14:03 UTC, DOWN, -21.0 pips over 15 minutes, causal completed-M5 ATR 3.207143 pips, not failed at five minutes, not retraced 50% within 30 minutes, and +5.1 pips in the original direction at 60 minutes. The raw detector reaches the same classification without ATR as an onset condition.

## Defensive baseline

| Arm | Trades | Net pips | PF | Hit | Average loss | P05 | Max loss streak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Immediate continuation baseline | 2,971 | -4,918.000 | 0.800593 | 43.79% | -14.830 | -32.650 | 13 |
| New-entry shock freeze | 0 | 0 | n/a | n/a | n/a | n/a | 0 |
| Shock freeze + 50% bot-owned drain proxy | 2,971 | -2,459.000 | 0.800593 | 43.79% | -7.415 | -16.325 | 13 |

Freeze is loss avoidance, not profit. It rejects only entries inside the detected shock band in this replay. The drain arm is paper/shadow only and manual/tagless inventory remains `NO_TOUCH`. Current `EUR_USD/RANGE_ROTATION/LONG` quarantine is not reconstructable from price-only M1 files because proposal method labels are absent; no directional proxy was fabricated.

## Exit architecture comparison

All arms retain the existing 2.4-pip target for controlled comparison. Stop hits are scored before targets in the same minute. Gap-through uses executable open and records slippage; no stop is described as guaranteed. Risk-scaled values multiply pips by the inverse-size fraction implied by each protective width. The no-SL shadow arm uses the mandatory 50-pip campaign cap as its sizing denominator; this is a comparison convention, not live permission.

| Arm | Net / PF | Avg loss / P05 | Max MAE / P95 MAE | Loss streak | Avg/max hold | Margin unit-min | Risk-scaled net / P05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Catastrophe SL + structure exit | -5,990.275 / 0.387107 | -7.098 / -12.400 | 66.7 / 15.2 | 9 | 3.94 / 15m | 0.482 | -701.206 / -1.406 |
| ATR 1.5 only | -5,828.777 / 0.441001 | -9.884 / -13.465 | 44.7 / 16.6 | 7 | 7.31 / 60m | 2.186 | -1,913.523 / -3.200 |
| Fixed 3.2 only | -5,096.400 / 0.272649 | -3.222 / -3.200 | 30.9 / 9.8 | 21 | 1.59 / 32m | 1.591 | -5,096.400 / -3.200 |
| No-SL + structure exit, shadow only | -5,917.500 / 0.390313 | -7.059 / -12.300 | 66.7 / 15.2 | 9 | 3.94 / 15m | 0.252 | -378.720 / -0.787 |

Exit reasons for catastrophe SL + structure exit were: target 1,572; failed continuation 819; adverse swing break 383; adverse velocity 102; adverse acceleration 77; catastrophe stop 16; spread expansion 1; time stop 1. No gap stop occurred in the observed cohort. Fixed 3.2 produced 2,145 ordinary stops, 30 gap stops, and 46.8 pips of gap slippage. ATR 1.5 produced 1,030 ordinary stops, 6 gap stops, and 16.009 pips of gap slippage.

With a simulated five-minute runtime disconnect, the server-side catastrophe arm remained protected: net -5,829.370, PF 0.400491, raw P05 -12.7, risk-scaled P05 -1.489, maximum MAE 76.4. The no-SL shadow comparison reached net -5,812.6, PF 0.401479, maximum MAE 76.4, and five campaign-cap exits, but it has no server-side protection and is permanently live-ineligible.

## Same-truth bounded structure-exit sensitivity

The operational architecture was replayed on the identical 2,971-episode cohort with three preregistered-style bounded timing cells. `Early` used velocity/acceleration thresholds 0.35/0.20 pips per minute, a three-minute swing lookback, and a ten-minute time stop. `Central` used 0.50/0.25, five minutes, and fifteen minutes. `Late` used 0.75/0.35, seven minutes, and twenty minutes. No row changes shock detection, catastrophe protection, direction symmetry, or live authority.

| Cell | Net / PF / hit | Risk-scaled net / PF / P05 | Loss streak | Margin unit-min / average hold |
|---|---:|---:|---:|---:|
| Early | -5,878.839 / 0.373795 / 49.51% | -690.492 / 0.375820 / -1.247 | 10 | 0.380 / 3.12m |
| Central | -5,990.275 / 0.387107 / 53.55% | -701.206 / 0.390841 / -1.406 | 9 | 0.482 / 3.94m |
| Late | -6,017.775 / 0.390431 / 54.49% | -707.013 / 0.393482 / -1.419 | 9 | 0.508 / 4.14m |

No cell dominates. Early reduces risk-scaled loss, tail, and inventory residence but lowers PF and hit rate and raises the maximum loss streak. Late slightly raises PF and hit rate but worsens loss, tail, and inventory residence. The central cell therefore remains the fixed shadow setting rather than selecting a post-hoc winner. All three PF values remain below one, so none is live-admissible.

## Adoption decision

- Keep `CONSERVATIVE_CATASTROPHE_PLUS_STRUCTURE_EXIT` as the only operational architecture eligible for future zero-authority shadow observation. Normal exit evaluates failed continuation, adverse swing break, raw velocity, raw acceleration, spread expansion, and time stop before relying on the server-side catastrophe stop.
- Keep ATR 1.5, fixed 3.2, and no-SL only as diagnostic comparison arms. ATR never triggers onset. Fixed 3.2 recreates 21 consecutive stops. No-SL can look smaller only after the imposed low-unit comparison convention and cannot survive a real communication/runtime failure contract.
- Promote no arm to live. Every PF is below 1. The catastrophe architecture reduces risk-scaled tail and margin occupancy relative to fixed and ATR-only, but it does not create positive expectancy.

## Reproduction

Run `tools/analyze_fast_bot_shock_guard_replay.py` with the seven existing `EUR_USD_M1_BA_2020...20260710` gzip files and the GET-only 2026-08-28 M1 BA file. The tool records input SHA-256, uses no broker client, writes no policy, and reports `external_order_attempts=0` and `external_orders=0`.

Fresh verification starting from commit `4e59662bcf4a1eb1f0f25c91f33093977e348627` reproduced the prior replay exactly after excluding only `generated_at_utc` and the equivalent `/tmp` versus `/private/tmp` path spelling. Replay-focused regression passed 40 tests and 8 subtests; the added profitability frontier/gate suite passed 18 tests. Full regression passed 5,247 tests and 1,235 subtests with one skip and two pre-existing pytest collection warnings.

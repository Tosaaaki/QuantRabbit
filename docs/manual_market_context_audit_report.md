# Manual Market Context Audit

- Generated at UTC: `2026-06-12T04:44:52.117777+00:00`
- Status: `MANUAL_MARKET_CONTEXT_PASS`
- Pair: `USD_JPY`
- Analyzed trades: `411` / `411` (`100.0`%)
- Guidance basis: `bounded_replay_lt_12h_excluding_margin_closeout`
- Best H1 alignment bucket: `AGAINST_H1_TREND`
- Best session bucket: `LONDON_AM`
- Conflict bucket requiring extra current reason: `WITH_H1_TREND`

## Bounded H1 Alignment

Bounded replay excludes >=12h holds and margin-closeout exits, because those are the same unbounded carry tail this runtime must avoid.

| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |
|---|---:|---:|---:|---:|---:|---:|
| `AGAINST_H1_TREND` | `131` | `96729.5` | `0.466` | `738.4` | `0.23` | `27.5` |
| `WITH_H1_TREND` | `192` | `8891.7` | `0.479` | `46.3` | `0.14` | `26.8` |

## Bounded Side x H1 Alignment

| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |
|---|---:|---:|---:|---:|---:|---:|
| `LONG_AGAINST_H1_TREND` | `35` | `65663.5` | `0.571` | `1876.1` | `0.51` | `32.5` |
| `LONG_WITH_H1_TREND` | `136` | `55886.7` | `0.544` | `410.9` | `0.11` | `25.4` |
| `SHORT_AGAINST_H1_TREND` | `96` | `31066.0` | `0.427` | `323.6` | `0.18` | `25.6` |
| `SHORT_WITH_H1_TREND` | `56` | `-46995.0` | `0.321` | `-839.2` | `0.21` | `30.4` |

## Bounded Side x 24h Location

| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |
|---|---:|---:|---:|---:|---:|---:|
| `LONG_LOWER_THIRD_24H` | `46` | `99877.9` | `0.652` | `2171.3` | `0.46` | `29.4` |
| `SHORT_UPPER_THIRD_24H` | `55` | `34735.0` | `0.6` | `631.5` | `0.18` | `28.8` |
| `LONG_UPPER_THIRD_24H` | `74` | `16091.0` | `0.459` | `217.4` | `0.06` | `26.0` |
| `LONG_MIDDLE_THIRD_24H` | `51` | `5581.3` | `0.588` | `109.4` | `0.7` | `25.8` |
| `SHORT_LOWER_THIRD_24H` | `37` | `-6930.0` | `0.324` | `-187.3` | `0.18` | `29.6` |
| `SHORT_MIDDLE_THIRD_24H` | `60` | `-43734.0` | `0.233` | `-728.9` | `0.24` | `24.6` |

## Raw H1 Alignment

| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |
|---|---:|---:|---:|---:|---:|---:|
| `AGAINST_H1_TREND` | `179` | `416768.7` | `0.559` | `2328.3` | `0.62` | `29.2` |
| `WITH_H1_TREND` | `232` | `-149952.8` | `0.474` | `-646.3` | `0.31` | `27.2` |

## Bounded Session

| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |
|---|---:|---:|---:|---:|---:|---:|
| `OFF_HOURS` | `27` | `58085.0` | `0.444` | `2151.3` | `0.13` | `25.0` |
| `TOKYO` | `82` | `57901.5` | `0.451` | `706.1` | `0.25` | `30.5` |
| `LONDON_AM` | `56` | `23016.3` | `0.625` | `411.0` | `0.63` | `28.2` |
| `NY_OVERLAP` | `158` | `-33381.6` | `0.437` | `-211.3` | `0.09` | `25.3` |

## Excluded Tail

| bucket | trades | net JPY | win rate | expectancy | median hold h | avg H1 ADX |
|---|---:|---:|---:|---:|---:|---:|
| `GE_12H` | `76` | `186253.7` | `0.75` | `2450.7` | `121.38` | `29.7` |
| `<30M` | `1` | `-3120.0` | `0.0` | `-3120.0` | `0.31` | `57.1` |
| `2H_12H` | `11` | `-21939.0` | `0.0` | `-1994.5` | `5.5` | `43.6` |

## Contract

- Advisory only: this audit gates only whether the 2025 manual precedent may be cited as an aggression/ranking reason.
- It cannot override RiskEngine, LiveOrderGateway, forecast, spread, event, broker-truth, or close Gate A/B checks.

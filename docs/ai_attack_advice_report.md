# AI Attack Advice Report

- Generated at UTC: `2026-05-06T15:12:16.362213+00:00`
- Status: `ATTACK_PARTIAL`
- Read only: `True`
- Live permission: `False`
- Live-ready lanes: `22`
- Live-ready reward: `565 JPY` (`2.7%`)
- Recommended now: `14` lanes, reward=`565 JPY`, risk=`256 JPY`
- Required additional reward: `20329 JPY`
- Required additional live-ready lanes: `473`

## Recommended Now

- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET` score=`58.8` reward=`42` risk=`18` rr=`2.27` hist_edge=`16375.5786` archive_edge=`2780.1475`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION:MARKET` score=`55.3` reward=`31` risk=`18` rr=`1.68` hist_edge=`4823.5725` archive_edge=`14096.4296`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` score=`53.8` reward=`41` risk=`18` rr=`2.27` hist_edge=`16375.5786` archive_edge=`2780.1475`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` score=`52.0` reward=`36` risk=`18` rr=`1.96` hist_edge=`4823.5725` archive_edge=`14096.4296`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` score=`51.4` reward=`34` risk=`18` rr=`1.88` hist_edge=`16375.5786` archive_edge=`2057.4571`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET` score=`32.3` reward=`52` risk=`18` rr=`2.84` hist_edge=`4823.5725` archive_edge=`-425.3698`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` score=`27.3` reward=`52` risk=`18` rr=`2.84` hist_edge=`4823.5725` archive_edge=`-425.3698`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:MARKET` score=`4.2` reward=`28` risk=`18` rr=`1.51` hist_edge=`-220.11` archive_edge=`85.0231`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION:MARKET` score=`3.2` reward=`25` risk=`18` rr=`1.34` hist_edge=`-715.4634` archive_edge=`598.0`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` score=`0.2` reward=`31` risk=`18` rr=`1.68` hist_edge=`-715.4634` archive_edge=`598.0`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` score=`-0.8` reward=`28` risk=`18` rr=`1.51` hist_edge=`-220.11` archive_edge=`85.0231`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION:MARKET` score=`-11.7` reward=`70` risk=`18` rr=`3.83` hist_edge=`-715.4634` archive_edge=`-860.0`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` score=`-16.7` reward=`70` risk=`18` rr=`3.83` hist_edge=`-715.4634` archive_edge=`-860.0`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` score=`-30.9` reward=`27` risk=`18` rr=`1.50` hist_edge=`-220.11` archive_edge=`-878.4849`

## Watchlist

- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:MARKET`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:MARKET`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION:MARKET`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION`

## Blockers

- none

## Action Items

- build additional LIVE_READY receipts for 20329 JPY of target coverage
- use recommended_now as the first verified basket, then keep generating sequential ladder receipts
- resolve coverage optimizer status COVERAGE_GAP before treating attack advice as certified

## Contract

- This advice is read-only and never places, stages, or resizes broker orders.
- `LiveOrderGateway` remains the final broker-truth and risk authority.
- Do not raise loss caps from this report; fix coverage by adding validated lanes or improving execution evidence.

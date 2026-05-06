# AI Attack Advice Report

- Generated at UTC: `2026-05-06T12:27:26.336244+00:00`
- Status: `ATTACK_PARTIAL`
- Read only: `True`
- Live permission: `False`
- Live-ready lanes: `20`
- Live-ready reward: `506 JPY` (`2.4%`)
- Recommended now: `12` lanes, reward=`506 JPY`, risk=`219 JPY`
- Required additional reward: `20388 JPY`
- Required additional live-ready lanes: `460`

## Recommended Now

- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION:MARKET` score=`47.3` reward=`52` risk=`18` rr=`2.84` hist_edge=`4823.5725`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET` score=`43.8` reward=`41` risk=`18` rr=`2.27` hist_edge=`16375.5786`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` score=`42.3` reward=`52` risk=`18` rr=`2.84` hist_edge=`4823.5725`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` score=`38.8` reward=`41` risk=`18` rr=`2.26` hist_edge=`16375.5786`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` score=`36.6` reward=`35` risk=`18` rr=`1.91` hist_edge=`4823.5725`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` score=`36.4` reward=`34` risk=`18` rr=`1.88` hist_edge=`16375.5786`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION:MARKET` score=`3.3` reward=`70` risk=`18` rr=`3.83` hist_edge=`-715.4634`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` score=`-1.7` reward=`70` risk=`18` rr=`3.83` hist_edge=`-715.4634`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION:MARKET` score=`-10.8` reward=`27` risk=`18` rr=`1.50` hist_edge=`-220.11`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` score=`-15.7` reward=`28` risk=`18` rr=`1.53` hist_edge=`-715.4634`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` score=`-15.8` reward=`27` risk=`18` rr=`1.50` hist_edge=`-220.11`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` score=`-15.9` reward=`27` risk=`18` rr=`1.50` hist_edge=`-220.11`

## Watchlist

- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:MARKET`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE:MARKET`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:MARKET`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE`

## Blockers

- none

## Action Items

- build additional LIVE_READY receipts for 20388 JPY of target coverage
- use recommended_now as the first verified basket, then keep generating sequential ladder receipts
- resolve coverage optimizer status COVERAGE_GAP before treating attack advice as certified

## Contract

- This advice is read-only and never places, stages, or resizes broker orders.
- `LiveOrderGateway` remains the final broker-truth and risk authority.
- Do not raise loss caps from this report; fix coverage by adding validated lanes or improving execution evidence.

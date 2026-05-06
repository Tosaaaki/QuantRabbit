# Strategy Mining Report

- Generated at UTC: `2026-05-06T03:21:56.712623+00:00`
- History DB: `/Users/tossaki/App/QuantRabbit/data/legacy_history.db`
- Strategy profile JSON: `/Users/tossaki/App/QuantRabbit/data/strategy_profile.json`
- Per-trade loss cap: `1051 JPY` (`daily target state /Users/tossaki/App/QuantRabbit/data/daily_target_state.json`)

## Evidence Coverage

- Source files indexed: `11998`
- `chunks` rows: `768`
- `market_events` rows: `13`
- `pretrade_outcomes` rows: `704`
- `seat_outcomes` rows: `2145`
- `trades` rows: `6728`
- `user_calls` rows: `11`
- `audit_history` events: `987`
- `s_hunt_ledger` events: `448`
- `trader_journal` events: `37`

## Candidate Edges

- None. No pair/direction passed the evidence gate.

## Risk-Repair Candidates

- `EUR_USD SHORT` status=`RISK_REPAIR_CANDIDATE` pretrade n=34 net=1605.1 avg=47.2; live n=46 net=767.4 worst=-2076.7; seats missed/captured=94/147; fix: edge exists but old sizing broke the loss cap; require <=1051 JPY dry-run receipt before live use

## Mine Missed Edges Before Live Use

- `EUR_USD LONG` status=`MINE_MISSED_EDGE` pretrade n=49 net=16174.0 avg=330.1; live n=34 net=839.3 worst=-798.0; seats missed/captured=34/37; fix: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution
- `EUR_JPY LONG` status=`MINE_MISSED_EDGE` pretrade n=23 net=3158.5 avg=137.3; live n=19 net=-1757.5 worst=-1272.0; seats missed/captured=92/39; fix: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
- `AUD_JPY LONG` status=`MINE_MISSED_EDGE` pretrade n=19 net=3152.2 avg=165.9; live n=22 net=2129.0 worst=-856.0; seats missed/captured=46/2; fix: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution
- `GBP_USD LONG` status=`MINE_MISSED_EDGE` pretrade n=21 net=1341.7 avg=63.9; live n=25 net=-3935.3 worst=-2583.1; seats missed/captured=86/9; fix: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap

## Blocked Until New Evidence

- `EUR_GBP LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=5 net=-908.9 worst=-392.7; seats missed/captured=0/0; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `NZD_USD SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=8 net=-503.6 worst=-103.8; seats missed/captured=0/0; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `EUR_CHF SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=3 net=-209.1 worst=-86.9; seats missed/captured=0/0; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `AUD_USD LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=9 net=-18.1 avg=-2.0; live n=9 net=-1668.2 worst=-1016.0; seats missed/captured=19/0; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
  - block reason: 4x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
  - block reason: 1x AUD_USD planned SL 3.7pip is only 2.6x spread (1.4pip); widen the stop or improve the entry first
- `USD_JPY LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=13 net=-29.0 avg=-2.2; live n=21 net=-6954.8 worst=-3360.0; seats missed/captured=135/10; fix: historical live loss exceeded the 1051 JPY cap; only risk-resized dry-run receipts can reopen it
  - block reason: 1x USD_JPY planned SL 25.0pip is still inside the recent noise floor 43.0pip (recent stop-loss regret: 3/3 recovered in 6h, avg loss 9.2pip p75 loss 11.6pip -> avg later favorable 28.6pip); widen the stop or improve the entry first
  - block reason: 1x USD_JPY planned SL 7.1pip is still inside the recent noise floor 11.6pip (recent stop-loss regret: 3/3 recovered in 6h, avg loss 9.2pip p75 loss 11.6pip -> avg later favorable 28.6pip); widen the stop or improve the entry first
  - block reason: 1x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
- `EUR_JPY SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=8 net=-569.0 avg=-71.1; live n=18 net=-2284.5 worst=-930.0; seats missed/captured=110/23; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
  - block reason: 8x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
  - block reason: 2x planned reward/risk 0.90x (reward 360 JPY / risk 402 JPY) is below 1.2x; improve geometry or skip
  - block reason: 2x C size cap is 1000u but the order asks for 2000u
- `GBP_USD SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=14 net=-837.8 avg=-59.8; live n=17 net=-1501.6 worst=-815.2; seats missed/captured=77/5; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
  - block reason: 2x B0 size cap is 3000u but the order asks for 4000u
  - block reason: 2x exact pretrade blocks this order: learning cap or final allocation says this seat is not worth real risk
  - block reason: 1x B- size cap is 2000u but the order asks for 3000u
- `USD_JPY SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=17 net=-2544.0 avg=-149.6; live n=12 net=-554.0 worst=-334.0; seats missed/captured=57/15; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
  - block reason: 7x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
  - block reason: 2x planned reward/risk 0.69x (reward 200 JPY / risk 288 JPY) is below 1.2x; improve geometry or skip
  - block reason: 1x USD_JPY planned SL 1.3pip is still inside the recent noise floor 11.6pip (recent stop-loss regret: 3/3 recovered in 6h, avg loss 9.2pip p75 loss 11.6pip -> avg later favorable 28.6pip); widen the stop or improve the entry first
- `GBP_JPY SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=5 net=-2590.0 avg=-518.0; live n=6 net=-1269.5 worst=-654.0; seats missed/captured=47/0; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
  - block reason: 1x C size cap is 1000u but the order asks for 2000u
  - block reason: 1x GBP_JPY planned SL 10.5pip is only 3.8x spread (2.8pip); widen the stop or improve the entry first
  - block reason: 1x GBP_JPY planned TP 12.2pip is only 3.9x spread (3.1pip); the move is too thin to pay live friction; GBP_JPY planned SL 14.6pip is still inside the recent noise floor 15.5pip (recent stop-loss regret: 2/3 recovered in 6h, avg loss 12.5pip -> avg later favorable 11.7pip); widen the stop or improve the entry first
- `AUD_USD SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=3 net=-3016.0 avg=-1005.3; live n=10 net=-1158.7 worst=-397.0; seats missed/captured=44/0; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
  - block reason: 2x OANDA immediate cancel
  - block reason: 1x AUD_USD planned SL 3.4pip is only 2.4x spread (1.4pip); widen the stop or improve the entry first
  - block reason: 1x AUD_USD planned SL 5.1pip is only 3.6x spread (1.4pip); widen the stop or improve the entry first
- `GBP_JPY LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=19 net=-3396.0 avg=-178.7; live n=15 net=-474.0 worst=-876.0; seats missed/captured=68/2; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
  - block reason: 1x B+ size floor is 3000u but the order asks for 2000u
  - block reason: 1x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
  - block reason: 1x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: repeat audit / missed-seat pressure is too strong to leave this pass-cap seat as prose; re-open it as a thin trigger-first scout (audit repeated 3x/6h + strongest high shelf on the board and the last candle re-ignited the squeeze.); live tape stays messy (mixed / friction-dominated (move +0.2pip, range 0.2pip, spread 2.8/2.8pip, mode=?)), so trigger proof remains mandatory
- `AUD_JPY SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=20 net=-4533.9 avg=-226.7; live n=28 net=-2660.0 worst=-1190.0; seats missed/captured=17/12; fix: historical live loss exceeded the 1051 JPY cap; only risk-resized dry-run receipts can reopen it
  - block reason: 3x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
  - block reason: 2x B+ size cap is 4500u but the order asks for 5000u
  - block reason: 1x A size floor is 4000u but the order asks for 3000u

## Generated System Rules

- A strategy candidate is not an order; it becomes an order intent only after current tape supplies entry, TP, SL, and thesis.
- Any pair/direction with a historical live loss worse than -1051 JPY needs fresh risk-resized dry-run receipts before live use, even when expectancy is positive.
- Missed directional seats are not chased at market; they require trigger or pending-entry receipts.
- Mixed or weak evidence remains watch-only even if the latest prompt wants action.

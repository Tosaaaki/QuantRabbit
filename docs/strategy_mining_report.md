# Strategy Mining Report

- Generated at UTC: `2026-06-26T06:36:22.770849+00:00`
- History DB: `/Users/tossaki/App/QuantRabbit/data/legacy_history.db`
- Strategy profile JSON: `/Users/tossaki/App/QuantRabbit/data/strategy_profile.json`
- Per-trade loss cap: `1739 JPY` (`daily target state /Users/tossaki/App/QuantRabbit/data/daily_target_state.json`)

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
- current execution ledger merged outcomes: `226` (path `/Users/tossaki/App/QuantRabbit/data/execution_ledger.db`)

## Candidate Edges

- `EUR_JPY LONG BREAKOUT_FAILURE` status=`CANDIDATE` pretrade n=23 net=3158.5 avg=137.3; live n=22 net=-2344.3 worst=-1272.0; seats missed/captured=92/39 net=12720.0 n=44 win=88.6%; fix: promoted from MINE_MISSED_EDGE by missed edge converted into STOP-ENTRY trigger receipt; source lane failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE; receipt_at_utc=2026-06-08T09:24:34.330983+00:00
- `EUR_JPY LONG RANGE_ROTATION` status=`CANDIDATE` pretrade n=23 net=3158.5 avg=137.3; live n=22 net=-2344.3 worst=-1272.0; seats missed/captured=92/39 net=12720.0 n=44 win=88.6%; fix: promoted from MINE_MISSED_EDGE by missed edge converted into LIMIT trigger receipt; source lane range_trader:EUR_JPY:LONG:RANGE_ROTATION; receipt_at_utc=2026-06-05T17:06:44.782924+00:00
- `EUR_USD SHORT BREAKOUT_FAILURE` status=`CANDIDATE` pretrade n=34 net=1605.1 avg=47.2; live n=105 net=6770.4 worst=-2891.1; seats missed/captured=94/147 net=-33883.1 n=273 win=53.8%; fix: promoted from RISK_REPAIR_CANDIDATE by loss-cap geometry repaired by current dry-run receipt; source lane failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT; receipt_at_utc=2026-06-08T12:10:14.433507+00:00
- `EUR_USD SHORT TREND_CONTINUATION` status=`CANDIDATE` pretrade n=34 net=1605.1 avg=47.2; live n=105 net=6770.4 worst=-2891.1; seats missed/captured=94/147 net=-33883.1 n=273 win=53.8%; fix: promoted from RISK_REPAIR_CANDIDATE by loss-cap geometry repaired by current dry-run receipt; source lane trend_trader:EUR_USD:SHORT:TREND_CONTINUATION; receipt_at_utc=2026-06-08T17:07:25.829931+00:00
- `CAD_JPY LONG` status=`CANDIDATE` pretrade n=0 net=0.0 avg=0.0; live n=12 net=961.3 worst=-51.8; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: current execution ledger shows repeated positive trader-owned live outcomes; eligible for dry-run order-intent generation, still behind risk gateway
- `USD_CAD LONG` status=`CANDIDATE` pretrade n=0 net=0.0 avg=0.0; live n=8 net=1019.9 worst=-620.1; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: current execution ledger shows repeated positive trader-owned live outcomes; eligible for dry-run order-intent generation, still behind risk gateway
- `AUD_CAD LONG` status=`CANDIDATE` pretrade n=0 net=0.0 avg=0.0; live n=3 net=1104.4 worst=139.7; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: current execution ledger shows repeated positive trader-owned live outcomes; eligible for dry-run order-intent generation, still behind risk gateway

## Risk-Repair Candidates

- `EUR_USD SHORT` status=`RISK_REPAIR_CANDIDATE` pretrade n=34 net=1605.1 avg=47.2; live n=105 net=6770.4 worst=-2891.1; seats missed/captured=94/147 net=-33883.1 n=273 win=53.8%; fix: edge exists but old sizing broke the loss cap; require <=1739 JPY dry-run receipt before live use

## Mine Missed Edges Before Live Use

- `EUR_JPY LONG` status=`MINE_MISSED_EDGE` pretrade n=23 net=3158.5 avg=137.3; live n=22 net=-2344.3 worst=-1272.0; seats missed/captured=92/39 net=12720.0 n=44 win=88.6%; fix: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution

## Blocked Until New Evidence

- `EUR_USD LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=49 net=16174.0 avg=330.1; live n=74 net=-7909.3 worst=-8378.5; seats missed/captured=34/37 net=-2881.5 n=113 win=32.7%; fix: historical live loss exceeded the 1739 JPY cap; only risk-resized dry-run receipts can reopen it
  - block reason: 18x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
  - block reason: 4x EUR_USD live tape probe is friction-dominated (range 0.0pip vs avg spread 0.8pip)
  - block reason: 2x C size cap is 1000u but the order asks for 4000u
- `AUD_JPY LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=19 net=3152.2 avg=165.9; live n=36 net=-1059.2 worst=-1740.0; seats missed/captured=46/2 net=-11873.0 n=63 win=3.2%; fix: historical live loss exceeded the 1739 JPY cap; only risk-resized dry-run receipts can reopen it
  - block reason: 2x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
  - block reason: 2x size asymmetry guard: dirty seat B0 at 3000u is larger than the recent paid winner size 2000u
  - block reason: 1x AUD_JPY planned SL 6.2pip is still inside the recent noise floor 9.0pip (recent stop-loss regret: 3/3 recovered in 6h, avg loss 7.2pip p75 loss 7.4pip -> avg later favorable 22.3pip); widen the stop or improve the entry first
- `GBP_USD LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=21 net=1341.7 avg=63.9; live n=54 net=-24407.6 worst=-11986.9; seats missed/captured=86/9 net=-8787.1 n=60 win=15.0%; fix: historical live loss exceeded the 1739 JPY cap; only risk-resized dry-run receipts can reopen it
  - block reason: 8x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
  - block reason: 1x A size floor is 4000u but the order asks for 3000u
  - block reason: 1x GBP_USD planned SL 3.3pip is only 2.5x spread (1.3pip); widen the stop or improve the entry first
- `GBP_CHF LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=4 net=-1569.8 worst=-981.8; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `EUR_GBP LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=5 net=-908.9 worst=-392.7; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `EUR_GBP SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=5 net=-702.4 worst=-891.1; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `EUR_CHF LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=5 net=-648.9 worst=-1019.8; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `AUD_NZD SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=3 net=-631.5 worst=-462.4; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `NZD_USD SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=8 net=-503.6 worst=-103.8; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `AUD_CHF LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=4 net=-462.2 worst=-1215.4; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `EUR_CHF SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=3 net=-209.1 worst=-86.9; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `AUD_CAD SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=3 net=-106.3 worst=-204.7; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `AUD_NZD LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=0 net=0.0 avg=0.0; live n=3 net=-91.1 worst=-65.1; seats missed/captured=0/0 net=0.0 n=0 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
- `AUD_USD LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=9 net=-18.1 avg=-2.0; live n=9 net=-1668.2 worst=-1016.0; seats missed/captured=19/0 net=-84.6 n=1 win=0.0%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
  - block reason: 4x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
  - block reason: 1x AUD_USD planned SL 3.7pip is only 2.6x spread (1.4pip); widen the stop or improve the entry first
- `USD_JPY LONG` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=13 net=-29.0 avg=-2.2; live n=22 net=-6522.8 worst=-3360.0; seats missed/captured=135/10 net=-7884.0 n=52 win=19.2%; fix: historical live loss exceeded the 1739 JPY cap; only risk-resized dry-run receipts can reopen it
  - block reason: 1x USD_JPY planned SL 25.0pip is still inside the recent noise floor 43.0pip (recent stop-loss regret: 3/3 recovered in 6h, avg loss 9.2pip p75 loss 11.6pip -> avg later favorable 28.6pip); widen the stop or improve the entry first
  - block reason: 1x USD_JPY planned SL 7.1pip is still inside the recent noise floor 11.6pip (recent stop-loss regret: 3/3 recovered in 6h, avg loss 9.2pip p75 loss 11.6pip -> avg later favorable 28.6pip); widen the stop or improve the entry first
  - block reason: 1x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
- `EUR_JPY SHORT` status=`BLOCK_UNTIL_NEW_EVIDENCE` pretrade n=8 net=-569.0 avg=-71.1; live n=23 net=-3257.7 worst=-930.0; seats missed/captured=110/23 net=-16227.0 n=134 win=17.2%; fix: both live execution and pretrade feedback are negative; require a new vehicle or market-structure proof
  - block reason: 8x exact pretrade allocation C is watch-only; do not turn a weak/pass-cap thesis into a 1000u live receipt: learning cap or final allocation says this seat is not worth real risk
  - block reason: 2x planned reward/risk 0.90x (reward 360 JPY / risk 402 JPY) is below 1.2x; improve geometry or skip
  - block reason: 2x C size cap is 1000u but the order asks for 2000u

## Generated System Rules

- A strategy candidate is not an order; it becomes an order intent only after current tape supplies entry, TP, SL, and thesis.
- Any pair/direction with a historical live loss worse than -1739 JPY needs fresh risk-resized dry-run receipts before live use, even when expectancy is positive.
- Missed directional seats are not chased at market; they require trigger or pending-entry receipts.
- Mixed or weak evidence remains watch-only even if the latest prompt wants action.

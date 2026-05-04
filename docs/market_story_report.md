# Market Story Report

- Generated at UTC: `2026-05-04T23:52:22.721976+00:00`
- Archive: `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z`
- Market story profile JSON: `/Users/tossaki/App/QuantRabbit/data/market_story_profile.json`
- Story artifacts read: `19`
- Narrative/chart lines mined: `2146`

## Artifacts

- `logs/news_digest.md` kind=`news_digest` lines=90
- `logs/news_flow_log.md` kind=`news_flow` lines=240
- `logs/quality_audit.md` kind=`quality_audit` lines=105
- `collab_trade/state.md` kind=`state` lines=294
- `collab_trade/strategy_memory.md` kind=`strategy_memory` lines=347
- `news/news_digest.md` kind=`news_digest` lines=55
- `news/news_flow_log.md` kind=`news_flow` lines=240
- `collab_trade/daily/2026-04-18/state.md` kind=`daily_state` lines=181
- `collab_trade/daily/2026-04-19/state.md` kind=`daily_state` lines=230
- `collab_trade/daily/2026-04-20/state.md` kind=`daily_state` lines=188
- `collab_trade/daily/2026-04-21/state.md` kind=`daily_state` lines=238
- `collab_trade/daily/2026-04-22/state.md` kind=`daily_state` lines=217
- `collab_trade/daily/2026-04-23/state.md` kind=`daily_state` lines=197
- `collab_trade/daily/2026-04-24/state.md` kind=`daily_state` lines=206
- `collab_trade/daily/2026-04-26/state.md` kind=`daily_state` lines=162
- `collab_trade/daily/2026-04-27/state.md` kind=`daily_state` lines=269
- `collab_trade/daily/2026-04-28/state.md` kind=`daily_state` lines=232
- `collab_trade/daily/2026-04-29/state.md` kind=`daily_state` lines=343
- `collab_trade/daily/2026-04-30/state.md` kind=`daily_state` lines=319

## Global Themes

- `breakout_failure`: `565`
- `range_rail`: `563`
- `intervention`: `371`
- `central_bank`: `296`
- `spread_liquidity`: `267`
- `event_risk`: `154`
- `position_risk`: `99`
- `momentum`: `92`

## Method Pressure

- `BREAKOUT_FAILURE`: `611`
- `RANGE_ROTATION`: `514`
- `EVENT_RISK`: `320`
- `TREND_CONTINUATION`: `190`
- `POSITION_MANAGEMENT`: `142`

## Pair Story Profiles

- `USD_JPY` methods: BREAKOUT_FAILURE=104, EVENT_RISK=96, RANGE_ROTATION=89, POSITION_MANAGEMENT=40; themes: intervention=143, breakout_failure=98, range_rail=96, central_bank=63, spread_liquidity=31
  - news_digest: USD/JPY has been pressing near 160, the threshold where authorities previously intervened (July 2024).
  - news_digest: Trade implication**: Long USD/JPY or short JPY crosses carry real intervention risk. Tight SLs on JPY shorts = getting hunted. If already long JPY via rate-check pop, thesis is asymmetric upside.
  - news_digest: USD/JPY**: Near 160. Rate check = intervention warning. Avoid being short JPY with tight SLs. If BOJ intervenes, move is fast (-200–300 pip in minutes). Rollover guard critical tonight.
- `EUR_USD` methods: RANGE_ROTATION=142, BREAKOUT_FAILURE=127, TREND_CONTINUATION=43, POSITION_MANAGEMENT=22; themes: range_rail=167, breakout_failure=106, spread_liquidity=40, intervention=28, event_risk=17
  - news_digest: Implication**: EUR caught between hot inflation (can't cut) and weak growth (needs cut). EUR directional bias remains murky. EUR/USD resistance likely firm below 1.1700.
  - news_digest: EUR/USD**: Capped near 1.1700. Hot EU CPI vs weak GDP = directionless. Below 1.1600 = EUR weakness thesis. NFP break could set direction.
  - news_flow: WATCH: EUR/USD ~1.1725
- `EUR_JPY` methods: RANGE_ROTATION=87, BREAKOUT_FAILURE=84, TREND_CONTINUATION=23, POSITION_MANAGEMENT=8; themes: intervention=108, range_rail=91, breakout_failure=80, spread_liquidity=34, central_bank=5
  - news_digest: EUR/JPY / GBP/JPY / AUD/JPY**: All carry intervention risk. JPY crosses can gap violently on rate check → actual intervention. Size down on all JPY shorts.
  - quality_audit: | EUR_JPY | TREND-BULL | TREND-BEAR | TREND-BEAR | heavy red flush, then a narrow repair shelf of small mixed-to-green bodies under the EMA cluster = corrective bounce inside larger bear control | NO: repair, not honest rotation |
  - quality_audit: No range trades: the live scanner rails on `EUR_USD`, `GBP_JPY`, `EUR_JPY`, and `AUD_JPY` are not visually confirmed as stable two-way boxes. The direct-USD charts are breakout-continuation structures, and the JPY crosses are repair-first,
- `AUD_JPY` methods: BREAKOUT_FAILURE=56, RANGE_ROTATION=46, TREND_CONTINUATION=34, EVENT_RISK=12; themes: intervention=58, breakout_failure=52, range_rail=49, spread_liquidity=17, event_risk=12
  - news_digest: EUR/JPY / GBP/JPY / AUD/JPY**: All carry intervention risk. JPY crosses can gap violently on rate check → actual intervention. Size down on all JPY shorts.
  - quality_audit: | AUD_JPY | TREND-BULL | TREND-BEAR | TREND-BEAR | flush to the low, then an orderly green recovery and late higher close, but the move is still a rebound inside the broader breakdown | NO: rebound inside trend, not range |
  - quality_audit: No range trades: the live scanner rails on `EUR_USD`, `GBP_JPY`, `EUR_JPY`, and `AUD_JPY` are not visually confirmed as stable two-way boxes. The direct-USD charts are breakout-continuation structures, and the JPY crosses are repair-first,
- `GBP_USD` methods: RANGE_ROTATION=66, BREAKOUT_FAILURE=48, TREND_CONTINUATION=28, EVENT_RISK=10; themes: range_rail=68, breakout_failure=47, spread_liquidity=28, momentum=15, intervention=9
  - news_digest: BoE held as expected, no new guidance. Mortgage data today (15:00 JST) may give GBP/USD a nudge.
  - news_digest: GBP/USD**: Neutral around 1.3500. BoE gave nothing new. GBP follows USD/risk tone.
  - quality_audit: | GBP_USD | TREND-BULL | TREND-BULL | TREND-BULL | a short mid-chart pause resolves into two strong green thrusts and a final push to highs, with almost no counter-wick rejection = trend walk | NO: breakout continuation, not range rotation
- `AUD_USD` methods: RANGE_ROTATION=69, BREAKOUT_FAILURE=40, TREND_CONTINUATION=18, EVENT_RISK=7; themes: range_rail=71, breakout_failure=39, spread_liquidity=17, intervention=10, event_risk=8
  - news_digest: AUD/USD**: At 100/200h MA convergence ~0.7159 — breakout setup. AU PPI today (10:30) could nudge direction. Net-long specs declined — positioning lighter, easier to move.
  - quality_audit: | AUD_USD | TREND-BULL | TREND-BULL | TREND-BULL | orderly green ladder after a shallow pullback, closes stay near the highs and EMA12 stays above EMA20 = controlled trend climb | NO: still a trend walk |
  - quality_audit: AUD_USD: Predicted `0.71435` SHORT. Actual: `0.7191` (about `+48` pip against). WRONG.
- `GBP_JPY` methods: RANGE_ROTATION=50, BREAKOUT_FAILURE=46, TREND_CONTINUATION=27, POSITION_MANAGEMENT=15; themes: spread_liquidity=78, intervention=60, range_rail=51, breakout_failure=46, position_risk=11
  - news_digest: EUR/JPY / GBP/JPY / AUD/JPY**: All carry intervention risk. JPY crosses can gap violently on rate check → actual intervention. Size down on all JPY shorts.
  - quality_audit: | GBP_JPY | TREND-BULL | TREND-BEAR | TREND-BEAR | waterfall lower followed by a shallow shelf and a cautious rebound; the right edge is still below the old breakdown zone = dead-cat repair | NO: recovery leg, not a clean box |
  - quality_audit: No range trades: the live scanner rails on `EUR_USD`, `GBP_JPY`, `EUR_JPY`, and `AUD_JPY` are not visually confirmed as stable two-way boxes. The direct-USD charts are breakout-continuation structures, and the JPY crosses are repair-first,

## Method Switching Contract

- `TREND_CONTINUATION`: use only when chart story names staircase, band-walk, trend walk, impulse continuation, or shallow-pullback continuation.
- `RANGE_ROTATION`: use only at exact rail/box prices; if the story says impulse, vertical leg, band-walk, or trend extension, this method is wrong.
- `BREAKOUT_FAILURE`: requires trapped side, failed reclaim/break, rejection price, and body-based invalidation.
- `EVENT_RISK`: central-bank, NFP, intervention, and spread-window stories reduce size or block tight-stop participation.
- `POSITION_MANAGEMENT`: live unprotected exposure, margin pressure, or stale protection overrides fresh-entry hunting.
- Every order intent must carry `market_context` so the system can judge method-vs-regime consistency before risk leaves the trader's hand.

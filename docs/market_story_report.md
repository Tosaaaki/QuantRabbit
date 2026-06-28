# Market Story Report

- Generated at UTC: `2026-06-26T03:45:03.155674+00:00`
- Archive: `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z`
- Market story profile JSON: `/Users/tossaki/App/QuantRabbit/data/market_story_profile.json`
- Story artifacts read: `19`
- Narrative/chart lines mined: `2152`

## Artifacts

- `news/news_digest.md` kind=`news_digest` lines=48
- `news/news_flow_log.md` kind=`news_flow` lines=240
- `logs/news_digest.md` kind=`news_digest` lines=90
- `logs/news_flow_log.md` kind=`news_flow` lines=240
- `logs/quality_audit.md` kind=`quality_audit` lines=105
- `collab_trade/state.md` kind=`state` lines=294
- `collab_trade/strategy_memory.md` kind=`strategy_memory` lines=347
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

- `breakout_failure`: `563`
- `range_rail`: `558`
- `intervention`: `296`
- `spread_liquidity`: `260`
- `central_bank`: `182`
- `event_risk`: `159`
- `position_risk`: `97`
- `momentum`: `92`

## Method Pressure

- `BREAKOUT_FAILURE`: `609`
- `RANGE_ROTATION`: `509`
- `EVENT_RISK`: `206`
- `TREND_CONTINUATION`: `188`
- `POSITION_MANAGEMENT`: `140`

## Pair Story Profiles

- `EUR_USD` methods: RANGE_ROTATION=141, BREAKOUT_FAILURE=126, TREND_CONTINUATION=41, POSITION_MANAGEMENT=22; themes: range_rail=166, breakout_failure=105, spread_liquidity=40, intervention=28, event_risk=17
  - news_flow [day=2026-04-30 pnl=-4061]: WATCH: EUR/USD below 200-day MA
  - news_flow [day=2026-04-28 pnl=+3127]: WATCH: EUR/USD ~1.1725
  - daily_state [day=2026-04-29 pnl=-1681]: NO-TRADE ACCOUNTABILITY: the book is not flat because `EUR_USD SHORT` trade id=`469898` remains live. The old `USD_JPY LONG` lower-box seat died on tape and was closed; there is no live backup trigger left behind.
- `USD_JPY` methods: BREAKOUT_FAILURE=103, RANGE_ROTATION=85, POSITION_MANAGEMENT=40, EVENT_RISK=31; themes: breakout_failure=97, range_rail=92, intervention=82, spread_liquidity=27, position_risk=25
  - news_flow [day=2026-04-30 pnl=-4061]: WATCH: USD/JPY near 159.0
  - daily_state [day=2026-04-28 pnl=+3127]: Entries today: 3 fills / 6 new entry orders / 43 rejects. Broker truth this session: EUR_USD SHORT id=469806 live 5000u @1.17082 TP=1.17067, GBP_JPY LONG id=469802 live 3000u @215.721 TP=215.960, AUD_JPY LONG id=469795 live 2000u @114.552 T
  - daily_state [day=2026-04-29 pnl=-1681]: NO-TRADE ACCOUNTABILITY: the book is not flat because `EUR_USD SHORT` trade id=`469898` remains live. The old `USD_JPY LONG` lower-box seat died on tape and was closed; there is no live backup trigger left behind.
- `EUR_JPY` methods: RANGE_ROTATION=86, BREAKOUT_FAILURE=83, TREND_CONTINUATION=23, POSITION_MANAGEMENT=8; themes: intervention=108, range_rail=90, breakout_failure=79, spread_liquidity=34, position_risk=5
  - quality_audit [day=2026-04-30 pnl=-4061]: | EUR_JPY | TREND-BULL | TREND-BEAR | TREND-BEAR | heavy red flush, then a narrow repair shelf of small mixed-to-green bodies under the EMA cluster = corrective bounce inside larger bear control | NO: repair, not honest rotation |
  - daily_state [day=2026-04-28 pnl=+3127]: Entries today: 3 fills / 6 new entry orders / 43 rejects. Broker truth this session: EUR_USD SHORT id=469806 live 5000u @1.17082 TP=1.17067, GBP_JPY LONG id=469802 live 3000u @215.721 TP=215.960, AUD_JPY LONG id=469795 live 2000u @114.552 T
  - daily_state [day=2026-04-29 pnl=-1681]: What would make me trade next session: `EUR_USD` rejects from the upper rail again, `AUD_USD` fails a higher retest back under the broken shelf, or `EUR_JPY` reprints `187.267/30` and rejects.
- `GBP_USD` methods: RANGE_ROTATION=66, BREAKOUT_FAILURE=48, TREND_CONTINUATION=28, EVENT_RISK=16; themes: range_rail=68, breakout_failure=47, spread_liquidity=28, momentum=13, central_bank=11
  - quality_audit [day=2026-04-30 pnl=-4061]: | GBP_USD | TREND-BULL | TREND-BULL | TREND-BULL | a short mid-chart pause resolves into two strong green thrusts and a final push to highs, with almost no counter-wick rejection = trend walk | NO: breakout continuation, not range rotation
  - daily_state [day=2026-04-28 pnl=+3127]: Driving force: Fed wait-state and FOMC risk keep direct-USD pairs compressed; ADP weakness and euro/GBP bid are pushing against fresh USD longs, while Iran/Hormuz relief keeps JPY offered intraday. vs last session: rollover has passed, but
  - daily_state [day=2026-04-29 pnl=-1681]: Exploration candidate #2: `GBP_USD SHORT`
- `AUD_JPY` methods: BREAKOUT_FAILURE=56, RANGE_ROTATION=46, TREND_CONTINUATION=34, EVENT_RISK=12; themes: intervention=58, breakout_failure=52, range_rail=49, spread_liquidity=17, event_risk=12
  - quality_audit [day=2026-04-30 pnl=-4061]: | AUD_JPY | TREND-BULL | TREND-BEAR | TREND-BEAR | flush to the low, then an orderly green recovery and late higher close, but the move is still a rebound inside the broader breakdown | NO: rebound inside trend, not range |
  - daily_state [day=2026-04-28 pnl=+3127]: Entries today: 3 fills / 6 new entry orders / 43 rejects. Broker truth this session: EUR_USD SHORT id=469806 live 5000u @1.17082 TP=1.17067, GBP_JPY LONG id=469802 live 3000u @215.721 TP=215.960, AUD_JPY LONG id=469795 live 2000u @114.552 T
  - daily_state [day=2026-04-29 pnl=-1681]: Regimes: `USD_JPY=M1 TREND-BEAR/M5 TREND-BEAR/H1 TREND-BULL`, `EUR_USD=M1 TREND-BULL/M5 TREND-BULL/H1 TREND-BEAR`, `GBP_USD=M1 TREND-BULL/M5 TREND-BULL/H1 TREND-BEAR`, `AUD_USD=M1 TREND-BULL/M5 TREND-BULL/H1 TREND-BEAR`, `EUR_JPY=M1 RANGE/M
- `AUD_USD` methods: RANGE_ROTATION=69, BREAKOUT_FAILURE=40, TREND_CONTINUATION=18, EVENT_RISK=7; themes: range_rail=71, breakout_failure=39, spread_liquidity=17, intervention=10, momentum=9
  - quality_audit [day=2026-04-30 pnl=-4061]: | AUD_USD | TREND-BULL | TREND-BULL | TREND-BULL | orderly green ladder after a shallow pullback, closes stay near the highs and EMA12 stays above EMA20 = controlled trend climb | NO: still a trend walk |
  - daily_state [day=2026-04-28 pnl=+3127]: Entries today: 3 fills / 6 new entry orders / 43 rejects. Broker truth this session: EUR_USD SHORT id=469806 live 5000u @1.17082 TP=1.17067, GBP_JPY LONG id=469802 live 3000u @215.721 TP=215.960, AUD_JPY LONG id=469795 live 2000u @114.552 T
  - daily_state [day=2026-04-29 pnl=-1681]: Best untraded seat: `AUD_USD SHORT` only after a fresh sell-high failure, not at the current lifted print.
- `GBP_JPY` methods: RANGE_ROTATION=50, BREAKOUT_FAILURE=46, TREND_CONTINUATION=27, POSITION_MANAGEMENT=15; themes: spread_liquidity=78, intervention=60, range_rail=51, breakout_failure=46, position_risk=11
  - quality_audit [day=2026-04-30 pnl=-4061]: | GBP_JPY | TREND-BULL | TREND-BEAR | TREND-BEAR | waterfall lower followed by a shallow shelf and a cautious rebound; the right edge is still below the old breakdown zone = dead-cat repair | NO: recovery leg, not a clean box |
  - daily_state [day=2026-04-28 pnl=+3127]: Entries today: 3 fills / 6 new entry orders / 43 rejects. Broker truth this session: EUR_USD SHORT id=469806 live 5000u @1.17082 TP=1.17067, GBP_JPY LONG id=469802 live 3000u @215.721 TP=215.960, AUD_JPY LONG id=469795 live 2000u @114.552 T
  - daily_state [day=2026-04-29 pnl=-1681]: Long-term trend (2h-1d): `GBP_JPY LONG` / floor-defense idea / `PASS` / expected about `+1,000 JPY` if a real defense candle and sane spread printed / action = dead because spread `3.7pip` plus no defense candle means there is no live long-
- `EUR_GBP` methods: EVENT_RISK=6, BREAKOUT_FAILURE=2, RANGE_ROTATION=1; themes: central_bank=6, breakout_failure=2, range_rail=1
  - daily_state [day=2026-04-21 pnl=-878]: AUD_USD: SQUEEZE | weaker rebound than EUR/GBP and already stalling. NOW none. RELOAD short only on 0.71535/45 failure. OTHER SIDE long only on 0.71395/71405 defense. -> Edge C / Allocation C / SKIP.
  - daily_state [day=2026-04-24 pnl=-472]: Macro chain: USD offered intraday -> supports EUR/GBP/AUD vs USD but does not justify bad prices | EUR soft data caps upside -> favors tactical box | GBP strongest -> long-only on pullback | JPY selectively bid -> USD_JPY sell-cap better th
  - news_flow [day=2026-06-17 pnl=?]: HOT: EUR_GBP, GBP_USD: British Pound: Political risks and BoE stance – RaboResearch
- `GBP_AUD` methods: none; themes: intervention=1
  - daily_state [day=2026-04-22 pnl=-689]: vs last session: EUR stayed weak, JPY stayed selective, GBP/AUD stayed compressed, the EUR_JPY short receipt has already closed, and EUR_USD remains trigger-only.
- `NZD_USD` methods: none; themes: none
  - news_digest [day=2026-06-26 pnl=?]: NZD_USD** — NZD/USD Price Forecast: RSI flashes oversold as the pair hovers near seven-month lows. Thesis driver: inflation. https://www.fxstreet.com/news/nzd-usd-price-forecast-rsi-flashes-oversold-as-the-pair-hovers-near-seven-month-lows-
  - news_flow [day=2026-05-30 pnl=?]: HOT: AUD_USD, NZD_USD: Asia open: S&P 500 nabs records on US - Iran ceasefire extension amid Hot PCE inflation shock
  - news_flow [day=2026-06-01 pnl=?]: HOT: AUD_USD, NZD_USD: Asia open: S&P 500 nabs records on US - Iran ceasefire extension amid Hot PCE inflation shock
- `USD_CHF` methods: none; themes: none
  - news_digest [day=2026-06-26 pnl=?]: USD_CHF** — USD/CHF Price Forecast: US Dollar retreats below 0.8100 as ‘tweezer top’ forms. Trade read: inflation, yields; do not fade this without chart confirmation. Source: FXStreet News; https://www.fxstreet.com/news/usd-chf-price-forec
  - news_flow [day=2026-06-23 pnl=?]: HOT: USD_CHF: Asia open: US stock futures and Asia Pacific equities wobbled on a firmer US dollar
  - news_flow [day=2026-06-24 pnl=?]: THEME: EUR_USD, GBP_USD, USD_JPY / USD_CHF

## Method Switching Contract

- `TREND_CONTINUATION`: use only when chart story names staircase, band-walk, trend walk, impulse continuation, or shallow-pullback continuation.
- `RANGE_ROTATION`: use rail/box prices, or low-vol M5 RANGE/QUIET directional scalps when the side bias is explicit; if the story says impulse, vertical leg, band-walk, or trend extension, this method is wrong.
- `BREAKOUT_FAILURE`: requires trapped side, failed reclaim/break, rejection price, and body-based invalidation.
- `EVENT_RISK`: central-bank, NFP, intervention, and spread-window stories reduce size or block tight-stop participation.
- `POSITION_MANAGEMENT`: live unprotected exposure, margin pressure, or stale protection overrides fresh-entry hunting.
- Every order intent must carry `market_context` so the system can judge method-vs-regime consistency before risk leaves the trader's hand.

# Trader Decision Report

- Generated at UTC: `2026-05-04T17:01:31.812149+00:00`
- Action: `SEND_ENTRY`
- Selected lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
- Selected lane score: `200.85`
- Selected lane size multiple: `1.0`
- Positions: `1`
- Orders: `2`
- Pending cancel ids: `none`
- Reason: Selected highest-scoring live-ready lane: failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE

## Ranked Lanes

- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` score=`200.85` action=`SEND_ENTRY` `EUR_USD LONG BREAKOUT_FAILURE`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; old worst loss repaired only by current sizing: -798 JPY; market-story method pressure 126; breakout-failure theme supports trap/reclaim; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` score=`196.85` action=`SEND_ENTRY` `EUR_USD LONG RANGE_ROTATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; old worst loss repaired only by current sizing: -798 JPY; market-story method pressure 141; range rail theme supports rotation; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` score=`186.44` action=`NO_TRADE` `EUR_USD SHORT BREAKOUT_FAILURE`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 126; breakout-failure theme supports trap/reclaim; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: historical live worst loss is large: -2077 JPY
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` score=`182.44` action=`NO_TRADE` `EUR_USD SHORT RANGE_ROTATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 141; range rail theme supports rotation; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
  - blockers: historical live worst loss is large: -2077 JPY
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` score=`130.27` action=`NO_TRADE` `GBP_USD LONG BREAKOUT_FAILURE`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1342 JPY; old worst loss repaired only by current sizing: -2583 JPY; market-story method pressure 48; breakout-failure theme supports trap/reclaim; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: negative live execution history -3935 JPY; low capture rate=5% (9/167); historical live worst loss is large: -2583 JPY
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` score=`111.21` action=`NO_TRADE` `AUD_JPY LONG BREAKOUT_FAILURE`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; old worst loss repaired only by current sizing: -856 JPY; market-story method pressure 56; breakout-failure theme supports trap/reclaim; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: low capture rate=2% (2/126)
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` score=`104.77` action=`NO_TRADE` `GBP_USD LONG RANGE_ROTATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1342 JPY; old worst loss repaired only by current sizing: -2583 JPY; market-story method pressure 66; range rail theme supports rotation; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
  - blockers: negative live execution history -3935 JPY; low capture rate=5% (9/167); historical live worst loss is large: -2583 JPY
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` score=`103.71` action=`NO_TRADE` `AUD_JPY LONG TREND_CONTINUATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; old worst loss repaired only by current sizing: -856 JPY; market-story method pressure 34; momentum theme supports trend; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for TREND_CONTINUATION
  - blockers: low capture rate=2% (2/126)
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` score=`89.69` action=`NO_TRADE` `EUR_JPY LONG BREAKOUT_FAILURE`
  - size_multiple: `0.93`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3158 JPY; old worst loss repaired only by current sizing: -1272 JPY; market-story method pressure 83; breakout-failure theme supports trap/reclaim; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: negative live execution history -1758 JPY; historical live worst loss is large: -1272 JPY
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` score=`78.71` action=`NO_TRADE` `AUD_JPY LONG RANGE_ROTATION`
  - size_multiple: `0.9`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; old worst loss repaired only by current sizing: -856 JPY; market-story method pressure 46; range rail theme supports rotation; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
  - blockers: low capture rate=2% (2/126)
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` score=`72.69` action=`NO_TRADE` `EUR_JPY LONG TREND_CONTINUATION`
  - size_multiple: `0.9`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3158 JPY; old worst loss repaired only by current sizing: -1272 JPY; market-story method pressure 23; momentum theme supports trend; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for TREND_CONTINUATION
  - blockers: negative live execution history -1758 JPY; historical live worst loss is large: -1272 JPY
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` score=`60.44` action=`NO_TRADE` `EUR_JPY LONG RANGE_ROTATION`
  - size_multiple: `0.9`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3158 JPY; old worst loss repaired only by current sizing: -1272 JPY; market-story method pressure 86; range rail theme supports rotation; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
  - blockers: negative live execution history -1758 JPY; historical live worst loss is large: -1272 JPY

## Trader-Brain Contract

- This layer must compare lanes; it must not send the first live-ready candidate mechanically.
- Scores rank attention only; live entry requires explicit discretionary gates, not a single score threshold.
- Pending entry or non-layerable exposure makes fresh-entry action monitor-only.
- JPY-cross long trades are penalized when intervention / thin-liquidity themes are active.
- The execution gateway remains the final authority for live risk.

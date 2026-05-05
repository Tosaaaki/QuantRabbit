# Trader Decision Report

- Generated at UTC: `2026-05-05T03:40:04.610997+00:00`
- Action: `SEND_ENTRY`
- Selected lane: `range_trader:EUR_USD:LONG:RANGE_ROTATION`
- Selected lane score: `210.19`
- Selected lane size multiple: `1.0`
- Positions: `1`
- Orders: `2`
- Pending cancel ids: `none`
- Loss cap: `1050.5334` (`strategy profile system_contract.loss_cap_jpy`)
- Reason: Selected highest-scoring live-ready lane: range_trader:EUR_USD:LONG:RANGE_ROTATION

## Ranked Lanes

- `range_trader:EUR_USD:LONG:RANGE_ROTATION` score=`210.19` action=`SEND_ENTRY` `EUR_USD LONG RANGE_ROTATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; market-story method pressure 142; range rail theme supports rotation; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` score=`207.19` action=`SEND_ENTRY` `EUR_USD LONG BREAKOUT_FAILURE`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; market-story method pressure 126; breakout-failure theme supports trap/reclaim; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` score=`185.56` action=`NO_TRADE` `EUR_USD SHORT BREAKOUT_FAILURE`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 126; breakout-failure theme supports trap/reclaim; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: historical live worst loss is large: -2077 JPY
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` score=`185.56` action=`NO_TRADE` `EUR_USD SHORT RANGE_ROTATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 142; range rail theme supports rotation; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
  - blockers: historical live worst loss is large: -2077 JPY
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` score=`185.44` action=`SEND_ENTRY` `EUR_USD LONG TREND_CONTINUATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; market-story method pressure 41; momentum theme supports trend; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for TREND_CONTINUATION
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` score=`163.81` action=`NO_TRADE` `EUR_USD SHORT TREND_CONTINUATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 41; momentum theme supports trend; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for TREND_CONTINUATION
  - blockers: historical live worst loss is large: -2077 JPY
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` score=`131.83` action=`NO_TRADE` `GBP_USD LONG BREAKOUT_FAILURE`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1342 JPY; old worst loss repaired only by current sizing: -2583 JPY; market-story method pressure 48; breakout-failure theme supports trap/reclaim; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: negative live execution history -3935 JPY; low capture rate=5% (9/167); historical live worst loss is large: -2583 JPY
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` score=`124.83` action=`NO_TRADE` `GBP_USD LONG TREND_CONTINUATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1342 JPY; old worst loss repaired only by current sizing: -2583 JPY; market-story method pressure 28; momentum theme supports trend; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for TREND_CONTINUATION
  - blockers: negative live execution history -3935 JPY; low capture rate=5% (9/167); historical live worst loss is large: -2583 JPY
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` score=`116.83` action=`NO_TRADE` `AUD_JPY LONG BREAKOUT_FAILURE`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; market-story method pressure 56; breakout-failure theme supports trap/reclaim; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: low capture rate=2% (2/126)
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` score=`113.08` action=`NO_TRADE` `GBP_USD LONG RANGE_ROTATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1342 JPY; old worst loss repaired only by current sizing: -2583 JPY; market-story method pressure 67; range rail theme supports rotation; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
  - blockers: negative live execution history -3935 JPY; low capture rate=5% (9/167); historical live worst loss is large: -2583 JPY
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` score=`109.33` action=`NO_TRADE` `AUD_JPY LONG TREND_CONTINUATION`
  - size_multiple: `1.0`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; market-story method pressure 34; momentum theme supports trend; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for TREND_CONTINUATION
  - blockers: low capture rate=2% (2/126)
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` score=`88.33` action=`NO_TRADE` `AUD_JPY LONG RANGE_ROTATION`
  - size_multiple: `0.92`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; market-story method pressure 46; range rail theme supports rotation; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: ORDER_INTENT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
  - blockers: low capture rate=2% (2/126)

## Trader-Brain Contract

- This layer must compare lanes; it must not send the first live-ready candidate mechanically.
- Scores rank attention only; live entry requires explicit discretionary gates, not a single score threshold.
- Pending entry or non-layerable exposure makes fresh-entry action monitor-only.
- JPY-cross long trades are penalized when intervention / thin-liquidity themes are active.
- The execution gateway remains the final authority for live risk.

# Trader Decision Report

- Generated at UTC: `2026-05-04T09:12:57.691407+00:00`
- Action: `SEND_ENTRY`
- Selected lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
- Selected lane score: `192.85`
- Selected lane size multiple: `1.65`
- Positions: `0`
- Orders: `0`
- Pending cancel ids: `none`
- Reason: Selected highest-scoring live-ready lane: failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE

## Ranked Lanes

- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` score=`192.85` action=`SEND_ENTRY` `EUR_USD LONG BREAKOUT_FAILURE`
  - size_multiple: `1.65`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; old worst loss repaired only by current sizing: -798 JPY; market-story method pressure 126; breakout-failure theme supports trap/reclaim; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: RISK_REPAIR_DRY_RUN; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` score=`188.85` action=`SEND_ENTRY` `EUR_USD LONG RANGE_ROTATION`
  - size_multiple: `1.62`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; old worst loss repaired only by current sizing: -798 JPY; market-story method pressure 142; range rail theme supports rotation; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: RISK_REPAIR_DRY_RUN; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` score=`178.44` action=`NO_TRADE` `EUR_USD SHORT BREAKOUT_FAILURE`
  - size_multiple: `1.55`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 126; breakout-failure theme supports trap/reclaim; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: RISK_REPAIR_DRY_RUN; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: historical live worst loss is large: -2077 JPY
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` score=`174.44` action=`NO_TRADE` `EUR_USD SHORT RANGE_ROTATION`
  - size_multiple: `1.52`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 142; range rail theme supports rotation; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: RISK_REPAIR_DRY_RUN; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
  - blockers: historical live worst loss is large: -2077 JPY
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` score=`171.6` action=`SEND_ENTRY` `EUR_USD LONG TREND_CONTINUATION`
  - size_multiple: `1.5`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; old worst loss repaired only by current sizing: -798 JPY; market-story method pressure 43; momentum theme supports trend; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: RISK_REPAIR_DRY_RUN; mined or repaired edge evidence is positive; current story contains method pressure for TREND_CONTINUATION
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` score=`157.19` action=`NO_TRADE` `EUR_USD SHORT TREND_CONTINUATION`
  - size_multiple: `1.4`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 43; momentum theme supports trend; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: RISK_REPAIR_DRY_RUN; mined or repaired edge evidence is positive; current story contains method pressure for TREND_CONTINUATION
  - blockers: historical live worst loss is large: -2077 JPY
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` score=`115.27` action=`NO_TRADE` `GBP_USD LONG BREAKOUT_FAILURE`
  - size_multiple: `1.11`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1342 JPY; old worst loss repaired only by current sizing: -2583 JPY; market-story method pressure 52; breakout-failure theme supports trap/reclaim; event risk requires restraint; spread/liquidity theme reduces urgency
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: TRIGGER_RECEIPT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: negative live execution history -3935 JPY; low capture rate=5% (9/167); historical live worst loss is large: -2583 JPY
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` score=`103.21` action=`NO_TRADE` `AUD_JPY LONG BREAKOUT_FAILURE`
  - size_multiple: `1.02`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; old worst loss repaired only by current sizing: -856 JPY; market-story method pressure 56; breakout-failure theme supports trap/reclaim; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: RISK_REPAIR_DRY_RUN; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: JPY-cross long faces intervention/rate-check narrative risk; low capture rate=2% (2/126)
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` score=`95.71` action=`NO_TRADE` `AUD_JPY LONG TREND_CONTINUATION`
  - size_multiple: `0.97`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; old worst loss repaired only by current sizing: -856 JPY; market-story method pressure 34; momentum theme supports trend; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: RISK_REPAIR_DRY_RUN; mined or repaired edge evidence is positive; current story contains method pressure for TREND_CONTINUATION
  - blockers: JPY-cross long faces intervention/rate-check narrative risk; low capture rate=2% (2/126)
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` score=`70.71` action=`NO_TRADE` `AUD_JPY LONG RANGE_ROTATION`
  - size_multiple: `0.9`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; old worst loss repaired only by current sizing: -856 JPY; market-story method pressure 46; range rail theme supports rotation; event risk requires restraint
  - judgment: fresh live-ready receipt exists; strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: RISK_REPAIR_DRY_RUN; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
  - blockers: JPY-cross long faces intervention/rate-check narrative risk; visual story explicitly rejected range rotation; low capture rate=2% (2/126)
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` score=`-379.11` action=`NO_TRADE` `EUR_JPY LONG BREAKOUT_FAILURE`
  - size_multiple: `0.9`
  - why: strategy profile candidate; positive pretrade evidence 3158 JPY; old worst loss repaired only by current sizing: -1272 JPY; market-story method pressure 83; breakout-failure theme supports trap/reclaim; event risk requires restraint; spread/liquidity theme reduces urgency; JPY liquidity theme requires smaller/fewer entries
  - judgment: strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: TRIGGER_RECEIPT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for BREAKOUT_FAILURE
  - blockers: intent status is DRY_RUN_BLOCKED; negative live execution history -1758 JPY; JPY-cross long faces intervention/rate-check narrative risk; wide spread for fresh edge=2.2pip; historical live worst loss is large: -1272 JPY; EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip; receipt is not live-ready: DRY_RUN_BLOCKED
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` score=`-408.36` action=`NO_TRADE` `EUR_JPY LONG RANGE_ROTATION`
  - size_multiple: `0.9`
  - why: strategy profile candidate; positive pretrade evidence 3158 JPY; old worst loss repaired only by current sizing: -1272 JPY; market-story method pressure 86; range rail theme supports rotation; event risk requires restraint; spread/liquidity theme reduces urgency; JPY liquidity theme requires smaller/fewer entries
  - judgment: strategy profile is live-eligible; thesis, narrative, chart story, method, and invalidation are explicit; campaign lane is executable after receipts: TRIGGER_RECEIPT_REQUIRED; mined or repaired edge evidence is positive; current story contains method pressure for RANGE_ROTATION
  - blockers: intent status is DRY_RUN_BLOCKED; negative live execution history -1758 JPY; JPY-cross long faces intervention/rate-check narrative risk; visual story explicitly rejected range rotation; wide spread for fresh edge=2.2pip; historical live worst loss is large: -1272 JPY; EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip; receipt is not live-ready: DRY_RUN_BLOCKED

## Trader-Brain Contract

- This layer must compare lanes; it must not send the first live-ready candidate mechanically.
- Scores rank attention only; live entry requires explicit discretionary gates, not a single score threshold.
- Pending entry or non-layerable exposure makes fresh-entry action monitor-only.
- JPY-cross long trades are penalized when intervention / thin-liquidity themes are active.
- The execution gateway remains the final authority for live risk.

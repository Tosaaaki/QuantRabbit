# Trader Decision Report

- Generated at UTC: `2026-04-30T17:01:46.514907+00:00`
- Action: `MONITOR_EXISTING_EXPOSURE`
- Selected lane: `None`
- Positions: `1`
- Orders: `2`
- Pending cancel ids: `none`
- Reason: Existing broker exposure is open; evaluate but do not stack fresh risk.

## Ranked Lanes

- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` score=`164.85` action=`NO_TRADE` `EUR_USD LONG BREAKOUT_FAILURE`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; old worst loss repaired only by current sizing: -798 JPY; market-story method pressure 126; breakout-failure theme supports trap/reclaim; event risk requires restraint
  - blockers: open position exists: EUR_USD LONG id=470024
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` score=`162.85` action=`NO_TRADE` `EUR_USD LONG RANGE_ROTATION`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; old worst loss repaired only by current sizing: -798 JPY; market-story method pressure 141; range rail theme supports rotation; event risk requires restraint
  - blockers: open position exists: EUR_USD LONG id=470024
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` score=`150.14` action=`NO_TRADE` `EUR_USD SHORT BREAKOUT_FAILURE`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 126; breakout-failure theme supports trap/reclaim; event risk requires restraint
  - blockers: open position exists: EUR_USD LONG id=470024
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` score=`148.14` action=`NO_TRADE` `EUR_USD SHORT RANGE_ROTATION`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 141; range rail theme supports rotation; event risk requires restraint
  - blockers: open position exists: EUR_USD LONG id=470024
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` score=`143.1` action=`NO_TRADE` `EUR_USD LONG TREND_CONTINUATION`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 16174 JPY; positive live evidence 839 JPY; old worst loss repaired only by current sizing: -798 JPY; market-story method pressure 41; momentum theme supports trend; event risk requires restraint
  - blockers: open position exists: EUR_USD LONG id=470024
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` score=`128.39` action=`NO_TRADE` `EUR_USD SHORT TREND_CONTINUATION`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1605 JPY; positive live evidence 767 JPY; old worst loss repaired only by current sizing: -2077 JPY; market-story method pressure 41; momentum theme supports trend; event risk requires restraint
  - blockers: open position exists: EUR_USD LONG id=470024
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` score=`96.47` action=`NO_TRADE` `GBP_USD LONG BREAKOUT_FAILURE`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 1342 JPY; old worst loss repaired only by current sizing: -2583 JPY; market-story method pressure 48; breakout-failure theme supports trap/reclaim; event risk requires restraint; spread/liquidity theme reduces urgency
  - blockers: open position exists: EUR_USD LONG id=470024; negative live execution history -3935 JPY
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` score=`83.41` action=`NO_TRADE` `AUD_JPY LONG BREAKOUT_FAILURE`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; old worst loss repaired only by current sizing: -856 JPY; market-story method pressure 56; breakout-failure theme supports trap/reclaim; event risk requires restraint
  - blockers: open position exists: EUR_USD LONG id=470024; JPY-cross long faces intervention/rate-check narrative risk
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` score=`75.91` action=`NO_TRADE` `AUD_JPY LONG TREND_CONTINUATION`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; old worst loss repaired only by current sizing: -856 JPY; market-story method pressure 34; momentum theme supports trend; event risk requires restraint
  - blockers: open position exists: EUR_USD LONG id=470024; JPY-cross long faces intervention/rate-check narrative risk
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` score=`54.39` action=`NO_TRADE` `EUR_JPY LONG BREAKOUT_FAILURE`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3158 JPY; old worst loss repaired only by current sizing: -1272 JPY; market-story method pressure 83; breakout-failure theme supports trap/reclaim; event risk requires restraint; spread/liquidity theme reduces urgency
  - blockers: open position exists: EUR_USD LONG id=470024; negative live execution history -1758 JPY; JPY-cross long faces intervention/rate-check narrative risk
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` score=`50.91` action=`NO_TRADE` `AUD_JPY LONG RANGE_ROTATION`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3152 JPY; positive live evidence 2129 JPY; old worst loss repaired only by current sizing: -856 JPY; market-story method pressure 46; range rail theme supports rotation; event risk requires restraint
  - blockers: open position exists: EUR_USD LONG id=470024; JPY-cross long faces intervention/rate-check narrative risk; visual story explicitly rejected range rotation
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` score=`25.14` action=`NO_TRADE` `EUR_JPY LONG RANGE_ROTATION`
  - why: live-ready risk/profile receipt; strategy profile candidate; positive pretrade evidence 3158 JPY; old worst loss repaired only by current sizing: -1272 JPY; market-story method pressure 86; range rail theme supports rotation; event risk requires restraint; spread/liquidity theme reduces urgency
  - blockers: open position exists: EUR_USD LONG id=470024; negative live execution history -1758 JPY; JPY-cross long faces intervention/rate-check narrative risk; visual story explicitly rejected range rotation

## Trader-Brain Contract

- This layer must compare lanes; it must not send the first live-ready candidate mechanically.
- Existing broker exposure makes fresh-entry action monitor-only.
- JPY-cross long trades are penalized when intervention / thin-liquidity themes are active.
- The execution gateway remains the final authority for live risk.

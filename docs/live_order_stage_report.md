# Live Order Stage Report

- Generated at UTC: `2026-05-06T11:43:48.372457+00:00`
- Status: `BLOCKED`
- Lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET`
- Lanes: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET`
- Requested units: `None` size multiple: `None` scaled units:`None`
- Send requested: `True`
- Sent: `False`
- Sent count: `0`

## Order Request

- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET` status=`BLOCKED` sent=`False`
  - `EUR_USD` `MARKET` units=`1153`
  - takeProfitOnFill: `1.17956`
  - stopLossOnFill: `1.17782`
  - broker-truth risk: `14.4 JPY` reward=`299.3 JPY` rr=`20.75` margin=`8493.8 JPY`

## Issues

- `BLOCK` MARKET_ENTRY_DRIFT: MARKET expected entry is stale versus broker quote: expected=1.17835 executable=1.1779 drift=4.5pip > 1.6pip
- `BLOCK` STOP_TOO_THIN_FOR_SPREAD: stop 0.8pip is less than 5.0x spread 0.8pip
- `BLOCK` BASKET_MARGIN_UTILIZATION_CAP_EXCEEDED: basket candidate margin 37940 JPY exceeds remaining 92.0% margin room 37936 JPY
- `WARN` BASKET_DOWNSIZED_FOR_CAPACITY: EUR_USD LONG downsized to fit basket margin room 8490 JPY

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

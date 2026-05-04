# Live Order Stage Report

- Generated at UTC: `2026-05-04T16:32:44.113272+00:00`
- Status: `SENT`
- Lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Requested units: `11000` size multiple: `1.0` scaled units:`11000`
- Send requested: `True`
- Sent: `True`

## Order Request

- `EUR_USD` `STOP` units=`-11000`
- price: `1.16890`
- takeProfitOnFill: `1.16530`
- stopLossOnFill: `1.16950`
- broker-truth risk: `1037.8 JPY` reward=`6227.0 JPY` rr=`6.00` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

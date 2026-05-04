# Live Order Stage Report

- Generated at UTC: `2026-05-04T10:59:07.377202+00:00`
- Status: `SENT`
- Lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Requested units: `12000` size multiple: `1.0` scaled units:`12000`
- Send requested: `True`
- Sent: `True`

## Order Request

- `EUR_USD` `STOP` units=`-12000`
- price: `1.17016`
- takeProfitOnFill: `1.16702`
- stopLossOnFill: `1.17068`
- broker-truth risk: `980.5 JPY` reward=`5920.7 JPY` rr=`6.04` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

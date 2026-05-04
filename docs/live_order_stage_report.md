# Live Order Stage Report

- Generated at UTC: `2026-05-04T13:39:59.670608+00:00`
- Status: `SENT`
- Lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Requested units: `13000` size multiple: `1.0` scaled units:`13000`
- Send requested: `True`
- Sent: `True`

## Order Request

- `EUR_USD` `STOP` units=`-13000`
- price: `1.17139`
- takeProfitOnFill: `1.16851`
- stopLossOnFill: `1.17187`
- broker-truth risk: `979.7 JPY` reward=`5878.3 JPY` rr=`6.00` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

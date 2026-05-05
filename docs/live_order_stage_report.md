# Live Order Stage Report

- Generated at UTC: `2026-05-05T01:32:39.604656+00:00`
- Status: `SENT`
- Lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Requested units: `13000` size multiple: `1.0` scaled units:`13000`
- Send requested: `True`
- Sent: `True`

## Order Request

- `EUR_USD` `STOP` units=`-13000`
- price: `1.16812`
- takeProfitOnFill: `1.16524`
- stopLossOnFill: `1.16860`
- broker-truth risk: `981.3 JPY` reward=`5887.7 JPY` rr=`6.00` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

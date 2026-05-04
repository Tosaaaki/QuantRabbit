# Live Order Stage Report

- Generated at UTC: `2026-05-04T02:53:37.071307+00:00`
- Status: `SENT`
- Lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
- Send requested: `True`
- Sent: `True`

## Order Request

- `EUR_USD` `STOP` units=`3000`
- price: `1.17290`
- takeProfitOnFill: `1.17670`
- stopLossOnFill: `1.17210`
- broker-truth risk: `377.3 JPY` reward=`1792.0 JPY` rr=`4.75` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

# Live Order Stage Report

- Generated at UTC: `2026-04-30T16:55:02.944338+00:00`
- Status: `SENT`
- Lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
- Send requested: `True`
- Sent: `True`

## Order Request

- `EUR_USD` `STOP` units=`1000`
- price: `1.17290`
- takeProfitOnFill: `1.17410`
- stopLossOnFill: `1.17210`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

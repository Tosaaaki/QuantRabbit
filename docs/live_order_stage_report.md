# Live Order Stage Report

- Generated at UTC: `2026-04-30T16:36:35.959164+00:00`
- Status: `STAGED`
- Lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
- Send requested: `False`
- Sent: `False`

## Order Request

- `EUR_USD` `STOP` units=`1000`
- price: `1.17262`
- takeProfitOnFill: `1.17382`
- stopLossOnFill: `1.17182`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

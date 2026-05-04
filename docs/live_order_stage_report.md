# Live Order Stage Report

- Generated at UTC: `2026-05-04T12:15:56.727076+00:00`
- Status: `STAGED`
- Lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Requested units: `13000` size multiple: `1.0` scaled units:`13000`
- Send requested: `False`
- Sent: `False`

## Order Request

- `EUR_USD` `STOP` units=`-13000`
- price: `1.16986`
- takeProfitOnFill: `1.16687`
- stopLossOnFill: `1.17036`
- broker-truth risk: `1021.3 JPY` reward=`6107.5 JPY` rr=`5.98` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

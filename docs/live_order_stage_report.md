# Live Order Stage Report

- Generated at UTC: `2026-05-04T13:21:46.245780+00:00`
- Status: `STAGED`
- Lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Requested units: `13000` size multiple: `1.0` scaled units:`13000`
- Send requested: `False`
- Sent: `False`

## Order Request

- `EUR_USD` `STOP` units=`-13000`
- price: `1.17064`
- takeProfitOnFill: `1.16776`
- stopLossOnFill: `1.17112`
- broker-truth risk: `980.2 JPY` reward=`5881.0 JPY` rr=`6.00` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

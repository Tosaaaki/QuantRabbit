# Live Order Stage Report

- Generated at UTC: `2026-05-04T12:28:23.101462+00:00`
- Status: `SENT`
- Lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Requested units: `13000` size multiple: `1.0` scaled units:`13000`
- Send requested: `True`
- Sent: `True`

## Order Request

- `EUR_USD` `STOP` units=`-13000`
- price: `1.17049`
- takeProfitOnFill: `1.16748`
- stopLossOnFill: `1.17099`
- broker-truth risk: `1021.0 JPY` reward=`6146.1 JPY` rr=`6.02` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

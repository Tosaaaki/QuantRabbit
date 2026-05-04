# Live Order Stage Report

- Generated at UTC: `2026-05-04T15:33:19.468818+00:00`
- Status: `SENT`
- Lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Requested units: `12000` size multiple: `1.0` scaled units:`12000`
- Send requested: `True`
- Sent: `True`

## Order Request

- `EUR_USD` `STOP` units=`-12000`
- price: `1.16958`
- takeProfitOnFill: `1.16635`
- stopLossOnFill: `1.17012`
- broker-truth risk: `1018.6 JPY` reward=`6092.5 JPY` rr=`5.98` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

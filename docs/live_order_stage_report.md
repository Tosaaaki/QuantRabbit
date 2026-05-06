# Live Order Stage Report

- Generated at UTC: `2026-05-06T05:40:22.574632+00:00`
- Status: `STAGED`
- Lane: `trend_trader:EUR_USD:LONG:TREND_CONTINUATION`
- Requested units: `13000` size multiple: `1.0` scaled units:`13000`
- Send requested: `False`
- Sent: `False`

## Order Request

- `EUR_USD` `STOP` units=`13000`
- price: `1.17349`
- takeProfitOnFill: `1.17457`
- stopLossOnFill: `1.17301`
- broker-truth risk: `977.6 JPY` reward=`2199.7 JPY` rr=`2.25` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

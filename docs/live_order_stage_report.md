# Live Order Stage Report

- Generated at UTC: `2026-05-05T03:39:40.189104+00:00`
- Status: `STAGED`
- Lane: `range_trader:EUR_USD:LONG:RANGE_ROTATION`
- Requested units: `13000` size multiple: `1.0` scaled units:`13000`
- Send requested: `False`
- Sent: `False`

## Order Request

- `EUR_USD` `LIMIT` units=`13000`
- price: `1.16847`
- takeProfitOnFill: `1.16906`
- stopLossOnFill: `1.16799`
- broker-truth risk: `983.2 JPY` reward=`1208.5 JPY` rr=`1.23` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

# Live Order Stage Report

- Generated at UTC: `2026-05-06T03:27:01.748145+00:00`
- Status: `STAGED`
- Lane: `range_trader:EUR_USD:LONG:RANGE_ROTATION`
- Requested units: `13000` size multiple: `1.0` scaled units:`13000`
- Send requested: `False`
- Sent: `False`

## Order Request

- `EUR_USD` `LIMIT` units=`13000`
- price: `1.17160`
- takeProfitOnFill: `1.17224`
- stopLossOnFill: `1.17112`
- broker-truth risk: `987.1 JPY` reward=`1316.1 JPY` rr=`1.33` spread=`0.8pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

# Live Order Stage Report

- Generated at UTC: `2026-05-04T23:29:37.114323+00:00`
- Status: `SENT`
- Lane: `range_trader:AUD_JPY:LONG:RANGE_ROTATION`
- Requested units: `10000` size multiple: `0.9` scaled units:`9000`
- Send requested: `True`
- Sent: `True`

## Order Request

- `AUD_JPY` `LIMIT` units=`9000`
- price: `112.648`
- takeProfitOnFill: `113.416`
- stopLossOnFill: `112.552`
- broker-truth risk: `864.0 JPY` reward=`6912.0 JPY` rr=`8.00` spread=`1.6pip`

## Issues

- none

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

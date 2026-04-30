# Live Order Stage Report

- Generated at UTC: `2026-04-30T16:40:59.818509+00:00`
- Status: `BLOCKED`
- Lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
- Send requested: `True`
- Sent: `False`

## Order Request

- `EUR_USD` `STOP` units=`1000`
- price: `1.17282`
- takeProfitOnFill: `1.17402`
- stopLossOnFill: `1.17202`

## Issues

- `BLOCK` PENDING_ENTRY_ORDER_OPEN: pending entry order is already open: AUD_JPY STOP id=470021; resolve it before new entries

## Send Contract

- This command stages by default and sends nothing.
- A live send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, a lane id, fresh broker truth, RiskEngine live validation, and StrategyProfile live validation.

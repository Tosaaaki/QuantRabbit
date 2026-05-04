# Position Execution Report

- Generated at UTC: `2026-05-04T11:54:21.459714+00:00`
- Status: `STAGED`
- Send requested: `False`
- Sent: `False`

## Actions

- `470140` `EUR_USD` management=`PROFIT_PROTECT_REQUIRED` request=`DEPENDENT_ORDER_REPLACE` sent=`False`
  - order_request: `{"stopLoss": {"price": "1.17016", "timeInForce": "GTC"}}`

## Execution Contract

- Position writes are risk-reducing only: close the trade, create missing protection, or tighten an existing SL.
- Existing SL cannot be widened. Existing TP is not moved by this gateway.
- Live execution requires the autotrade send path and `QR_LIVE_ENABLED=1`.

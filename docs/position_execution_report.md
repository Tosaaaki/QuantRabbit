# Position Execution Report

- Generated at UTC: `2026-05-04T14:46:31.684577+00:00`
- Status: `SENT`
- Send requested: `True`
- Sent: `True`

## Actions

- `470154` `EUR_USD` management=`PROFIT_PROTECT_REQUIRED` request=`DEPENDENT_ORDER_REPLACE` sent=`True`
  - order_request: `{"stopLoss": {"price": "1.17126", "timeInForce": "GTC"}}`

## Execution Contract

- Position writes are risk-reducing only: close the trade, create missing protection, or tighten an existing SL.
- Existing SL cannot be widened. Existing TP is not moved by this gateway.
- Live execution requires the autotrade send path and `QR_LIVE_ENABLED=1`.

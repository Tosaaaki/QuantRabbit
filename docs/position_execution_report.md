# Position Execution Report

- Generated at UTC: `2026-05-04T06:15:09.632684+00:00`
- Status: `BLOCKED`
- Send requested: `True`
- Sent: `False`

## Actions

- `470130` `EUR_USD` management=`REPAIR_PROTECTION_REQUIRED` request=`none` sent=`False`
  - `BLOCK` NON_TRADER_POSITION: refusing to modify unknown position id=470130

## Execution Contract

- Position writes are risk-reducing only: close the trade, create missing protection, or tighten an existing SL.
- Existing SL cannot be widened. Existing TP is not moved by this gateway.
- Live execution requires the autotrade send path and `QR_LIVE_ENABLED=1`.

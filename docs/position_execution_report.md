# Position Execution Report

- Generated at UTC: `2026-05-17T23:03:50.691645+00:00`
- Status: `NO_ACTION`
- Send requested: `False`
- Sent: `False`

## Actions

- `471255` `EUR_USD` management=`HOLD_SL_FREE` request=`none` sent=`False`
- `471232` `EUR_USD` management=`HOLD_SL_FREE` request=`none` sent=`False`

## Execution Contract

- Trader-owned position writes are risk-reducing only: close the trade, create missing protection, tighten an existing SL, or update TP.
- Manual/tagless position writes are TP-only profit management; SL writes and market closes are forbidden.
- Existing SL cannot be widened. Existing TP may be moved only by TP-management actions.
- Live execution requires the autotrade send path and `QR_LIVE_ENABLED=1`.

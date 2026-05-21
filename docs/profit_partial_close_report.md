# Profit Partial Close Report

- Generated at UTC: `2026-05-20T01:41:42.537074+00:00`
- Status: `NO_ACTION`
- Send requested: `False`
- Sent count: `0`

## Actions

- none

## Contract

- Profit partial close only reduces already-profitable trader-owned or manual/tagless exposure.
- Manual/tagless profit partials never realize an adverse P/L loss and never write stop-loss orders.
- Same trade milestone is persisted in state after a successful send to avoid repeat closes.
- Live send requires `--send --confirm-live` and `QR_LIVE_ENABLED=1`.

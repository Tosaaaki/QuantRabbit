# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T23:36:20.456953+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT. Daily target already exceeded (11.9% vs 10%). Golden Week thin market. All lanes blocked by stale quotes. EUR_USD SHORT protected at break-even, running toward TP. AUD_JPY LIMIT 470194 pending from prior cycle. Protection-first stance per contract §5. No new risk warranted.

## Verification Issues

- `BLOCK` BAD_METHOD: unsupported method ''

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

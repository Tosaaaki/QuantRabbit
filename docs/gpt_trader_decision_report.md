# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T17:58:41.221776+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT: all 15 lanes blocked by STALE_QUOTE (45-47s vs 20s threshold). Market technically open (Mon 17:54 UTC) but Golden Week thin liquidity. Campaign 8.6% progress, 19196 JPY remaining. Cannot trade without fresh broker truth per contract §9. Persistent condition from previous cycle.

## Verification Issues

- `BLOCK` WAIT_MISSING_LIVE_READY_REJECTION: WAIT must cite at least one current LIVE_READY lane evidence ref when clean tradeable lanes exist and the daily target is still open

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

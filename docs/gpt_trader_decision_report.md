# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T23:52:22.734395+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT decision. Market OFF_HOURS (Sunday 19:49 NY, pre-Tokyo Monday open). All quotes STALE (50-63s > 20s), blocking all 15 lane intents. Daily target exceeded (11.55% > 10%), triggering protection-first behavior per contract §5. Existing EUR_USD SHORT position protected at breakeven (+611.81 JPY unrealized, remaining_risk=0.0). No action until fresh quotes available. Next cycle: refresh broker-snapshot when Tokyo opens (~00:00 UTC Monday).

## Verification Issues

- `BLOCK` WAIT_MISSING_LIVE_READY_REJECTION: WAIT must cite at least one current LIVE_READY lane evidence ref when clean tradeable lanes exist and the daily target is still open

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

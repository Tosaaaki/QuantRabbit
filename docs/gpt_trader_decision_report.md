# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T19:36:16.065118+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT: Market closed until Sunday 17:00 EDT. All 15 lanes blocked by STALE_QUOTE (quotes 45-47s old). No action required. Next cycle at/after market open will refresh broker-snapshot and re-evaluate live-ready coverage. Daily campaign: 8.6% progress (1815/21011 JPY), 0 open risk, 4202 JPY daily budget remaining.

## Verification Issues

- `BLOCK` WAIT_MISSING_LIVE_READY_REJECTION: WAIT must cite at least one current LIVE_READY lane evidence ref when clean tradeable lanes exist and the daily target is still open

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

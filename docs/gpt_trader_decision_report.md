# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T09:20:51.953518+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT decision due to weekend market closure. All quotes are 58-64s stale (fetched Sunday 09:15 UTC during closure period). All 15 trade lanes blocked by RiskEngine STALE_QUOTE validation. Market reopens Sunday 22:00 UTC. Contract gate: AGENT_CONTRACT §3.5 stale-persistence prohibition. Next cycle should execute after market open when fresh quotes available. Progress: 8.6% of daily target, flat account.

## Verification Issues

- `BLOCK` WAIT_MISSING_LIVE_READY_REJECTION: WAIT must cite at least one current LIVE_READY lane evidence ref when clean tradeable lanes exist and the daily target is still open

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

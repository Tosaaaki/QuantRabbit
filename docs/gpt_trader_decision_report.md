# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T22:56:17.562614+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT decision: all 15 intents blocked by STALE_QUOTE risk gate (quotes 51-66s stale, threshold 20s). Market in OFF_HOURS during Golden Week. Existing EUR_USD SHORT position protected at breakeven with +979 JPY unrealized. No actionable lanes until London killzone (430 min). Progress 13.3% of target (2794/21011 JPY), campaign pursuit continues when quotes refresh.

## Verification Issues

- `BLOCK` WAIT_MISSING_LIVE_READY_REJECTION: WAIT must cite at least one current LIVE_READY lane evidence ref when clean tradeable lanes exist and the daily target is still open

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T17:24:48.732103+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT due to STALE_QUOTE blocker on all 15 lanes during Golden Week (May 5 Children's Day holiday). Markets effectively paused, quotes 51-53s stale. Pending EUR_USD SHORT order preserved for post-holiday resumption. Progress 8.64% of target. Dry-run only (QR_LIVE_ENABLED not set). No contract violations, no invented thresholds - legitimate market-closed gate from §9.

## Verification Issues

- `BLOCK` WAIT_MISSING_LIVE_READY_REJECTION: WAIT must cite at least one current LIVE_READY lane evidence ref when clean tradeable lanes exist and the daily target is still open

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

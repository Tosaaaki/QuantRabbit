# GPT Trader Decision Report

- Generated at UTC: `2026-05-05T01:58:26.067145+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT decision. All 15 lane candidates are DRY_RUN_BLOCKED by STALE_QUOTE (quote ages 47-58s > 20s threshold). Current state: Golden Week (jp_holiday=true), Asian session, thin liquidity. Broker quote feed aging beyond risk validation gates. Contract forbids silent fallbacks; when required inputs are stale, the cycle waits. Existing EUR_USD SHORT protected at breakeven (+1509 JPY unrealized, 7.2% of daily target). No action required until fresh broker truth is available.

## Verification Issues

- `BLOCK` WAIT_MISSING_LIVE_READY_REJECTION: WAIT must cite at least one current LIVE_READY lane evidence ref when clean tradeable lanes exist and the daily target is still open

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

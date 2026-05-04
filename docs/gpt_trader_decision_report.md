# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T19:59:58.035535+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT due to STALE_QUOTE blocking all 15 lanes. Broker-snapshot quotes (fetched 19:55:08 UTC) aged to 50-54s by the time intents were validated (19:55:58 UTC), exceeding the 20s contract threshold. Market is open (NY PM session, Golden Week holiday thin liquidity). EUR_USD SHORT -225 JPY is protected and within noise; no action needed. Coverage gap (0 LIVE_READY, 0 reward vs 19421 JPY target) is a product blocker. Next cycle: consider refreshing broker-snapshot immediately before intents, or accept dry-run staleness since live sends re-fetch anyway. Progress 7.57%, 3221 JPY risk budget remaining.

## Verification Issues

- `BLOCK` WAIT_MISSING_LIVE_READY_REJECTION: WAIT must cite at least one current LIVE_READY lane evidence ref when clean tradeable lanes exist and the daily target is still open

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T17:32:19.397394+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT decision at JST 02:27 Monday early Asia session. All 15 lanes blocked by STALE_QUOTE (quotes aged 49-51s during sequential command execution, exceeding 20s threshold). Market structure shows SHORT bias across EUR_USD (0.82), AUD_JPY (0.814), GBP_USD (0.752). Currency strength confirms USD strongest, AUD weakest. EUR_JPY additionally blocked by wide spread (2.3p vs 0.8p normal). Campaign progress 8.64% of 10% target. Next cycle should consider refreshing broker-snapshot immediately before intent generation to minimize staleness gap.

## Verification Issues

- `BLOCK` WAIT_MISSING_LIVE_READY_REJECTION: WAIT must cite at least one current LIVE_READY lane evidence ref when clean tradeable lanes exist and the daily target is still open

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T12:38:05.338856+00:00`
- Status: `ACCEPTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT. Pending EUR_USD SHORT order 470147 (@1.17049) blocks new trade. Current bid 1.17088 (3.9p above order). M5 pullback in M15/H1 downtrend - order remains valid. MONITOR mode per AGENT_CONTRACT §9. Progress 8.64% of target.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

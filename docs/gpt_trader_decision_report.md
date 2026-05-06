# GPT Trader Decision Report

- Generated at UTC: `2026-05-06T03:27:01.746217+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `range_trader:EUR_USD:LONG:RANGE_ROTATION`
- Cancel order ids: `none`
- Confidence: `MEDIUM`
- Operator summary: TRADE the deterministic EUR_USD support-rail range rotation lane. It is LIVE_READY, flat-account eligible, and specifically addresses the weak range-trading path without forcing market entry or bypassing risk gates.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

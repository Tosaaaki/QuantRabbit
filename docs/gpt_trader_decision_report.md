# GPT Trader Decision Report

- Generated at UTC: `2026-05-06T11:43:48.356833+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET`
- Selected basket lanes: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET`
- Cancel order ids: `none`
- Confidence: `MEDIUM`
- Operator summary: TRADE one additional reduced EUR_USD MARKET long through the verified gateway path while keeping existing pending order 470204 counted in basket risk/margin. Do not send blocked alternatives.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires known `LIVE_READY` lane(s); pending entries are counted by gateway basket validation.
- Evidence refs must come from the input packet; invented refs reject the decision.

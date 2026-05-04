# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T12:46:32.967942+00:00`
- Status: `ACCEPTED`
- Action: `CANCEL_PENDING`
- Selected lane: `None`
- Cancel order ids: `470147`
- Confidence: `HIGH`
- Operator summary: CANCEL order 470147. Golden Week thin liquidity + STALE_QUOTE blocker (quotes 46-55s old > 20s threshold) on all 15 lanes. Cannot validate pending order risk with 55s-stale broker truth. EUR_JPY spread 2.625x normal confirms stress. Better to cancel and re-enter when market provides fresh pricing. Progress 8.64% of target.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

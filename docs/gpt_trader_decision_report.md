# GPT Trader Decision Report

- Generated at UTC: `2026-05-06T04:37:24.205318+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT one cycle. The book is flat and the campaign remains active, but the current LIVE_READY lanes are clustered in EUR_USD and GBP_USD immediately after M5 jump events, so no-send is the disciplined choice until the next refresh.

## Verification Issues

- `BLOCK` UNKNOWN_EVIDENCE_REF: unknown evidence refs: intent:trend_trader:EUR_USD:LONG:TREND_CONTINUATION, intent:trend_trader:GBP_USD:LONG:TREND_CONTINUATION

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

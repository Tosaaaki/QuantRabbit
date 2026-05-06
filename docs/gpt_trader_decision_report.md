# GPT Trader Decision Report

- Generated at UTC: `2026-05-06T05:40:22.572513+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `trend_trader:EUR_USD:LONG:TREND_CONTINUATION`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: TRADE one dry-run cycle with trend_trader:EUR_USD:LONG:TREND_CONTINUATION. The lane is live-ready, market-derived risk stays under the 1050.5334 JPY per-trade cap, and current tape supports USD-weak continuation more cleanly than the other available lanes.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

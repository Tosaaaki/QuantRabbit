# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T12:28:23.099572+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: EUR_USD SHORT at 1.17056 via failure_trader desk. Currency strength divergence (USD #1, EUR #6) + DXY momentum (+0.42%) + MTF downtrend structure + excellent spread (0.8p) + no event risk = HIGH conviction. Risk 1021 JPY, reward 6146 JPY, RR 6.02. This represents 32% of remaining daily target in a single trade with per-trade risk budget compliance.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

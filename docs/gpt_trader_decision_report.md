# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T13:39:59.668661+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: EUR_USD SHORT via breakout failure: M15 TREND_DOWN (ADX=27.1) + H1 structure BOS_DOWN + DXY strength + COT flow shift (EUR longs reducing) + currency strength alignment (USD #1, EUR #7). STOP-ENTRY at 1.17139 captures failed upside breakout attempts before continuation to 1.16851. 5.99 RR, 5,880 JPY reward, 980 JPY risk. High conviction on multi-timeframe + cross-asset + institutional flow confluence.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

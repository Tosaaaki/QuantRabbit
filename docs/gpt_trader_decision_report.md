# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T09:12:57.693365+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: TRADE decision: EUR_USD SHORT via failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE. M5 bullish momentum (AroonOsc=79) exhausted at structure (BOS_DOWN@1.1724), M15/H1 bearish momentum + structure. DXY up supports EUR down. STOP-ENTRY at 1.17191 (below current 1.17228) waits for breakdown confirmation. 6.00R, 2712 JPY reward, 452 JPY risk. All gates clear: no event window, spread NORMAL, currency strength aligns. Progress 8.64%, 15 LIVE_READY lanes available. Pursuing 19,196 JPY remaining target.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

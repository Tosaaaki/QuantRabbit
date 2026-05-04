# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T13:21:46.243853+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: TRADE failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE. Entry 1.17039 STOP, TP 1.16751 (28.8p), SL 1.17087 (4.8p), units 13k, risk 980 JPY, reward 5,881 JPY (5.99 RR). Currency strength (USD #1, EUR #7) + chart short bias (0.62) + M15 TREND_DOWN + DXY UP + spread NORMAL + no event risk = high confluence. STOP-ENTRY waits for breakout failure trigger. Progress 8.6% of target, this trade targets +30.6% of remaining gap.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

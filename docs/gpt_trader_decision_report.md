# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T12:15:56.724984+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: EUR_USD SHORT shows highest conviction across all pairs (score 0.943). TREND_DOWN regime confirmed M5/M15 with strong ADX (31.9/32.0). Recent BOS_DOWN@1.1696 structure event confirms breakdown. Currency strength fully aligned (USD rank 1 strongest, EUR rank 6 weak). DXY rising +0.47% adds USD tailwind. COT funds reducing EUR longs (-8723 Δw). No calendar risk. Spread normal 0.8p. Entry via STOP at 1.1697 to confirm breakdown, TP 1.16671 (S1 pivot zone), SL 1.1702. Risk 1021.2 JPY (0.48% equity), reward 6106.6 JPY (5.98 R:R, 31.8% of remaining target). Account flat (previous position closed). Executing failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

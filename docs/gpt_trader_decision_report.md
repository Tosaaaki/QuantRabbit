# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T18:11:56.109041+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `MEDIUM`
- Operator summary: Selected failure_trader EUR_USD SHORT STOP-ENTRY at 1.16928 (3.3p below current) for breakdown confirmation. Regime is UNCLEAR/RANGE making BREAKOUT_FAILURE strategy appropriate. All timeframes show bearish structure (ST=-, BOS_DOWN) but low ADX confirms non-trending consolidation. DXY strength and declining COT EUR long positioning support SHORT thesis. Risk 981 JPY for 5,886 JPY potential reward (rr=6.00, 30.6% of remaining daily target). STOP-ENTRY filters noise better than LIMIT in this regime.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

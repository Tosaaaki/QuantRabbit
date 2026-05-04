# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T15:33:19.466765+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: TRADE failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE. Selling EUR_USD on break below 1.16976 support. DXY rally +0.44%, USD#1 vs EUR#4 strength divergence, and M5/M15/H1 CHOCH_DOWN/BOS_DOWN structure support SHORT bias within RANGE regime. STOP-ENTRY at -2.0p from current bid, TP 1.16653 (-32.3p = 2.2×ATR), SL 1.1703 (+5.4p = 0.37×ATR). Risk 1018.5 JPY (96.9% of per-trade budget), RR 5.98. Potential reward 6092 JPY = 31.7% of remaining daily target. Campaign progress 8.64%, this is trade #1 of target 4/day. No contract gate fires: spread NORMAL, no event window, risk within budget. Among available LIVE_READY lanes, EUR_USD SHORT shows strongest confluence (regime, structure, currency strength, DXY, COT flow).

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

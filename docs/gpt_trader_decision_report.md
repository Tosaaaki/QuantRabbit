# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T16:32:44.111428+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: TRADE EUR_USD SHORT via failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE. Macro: USD strength (DXY +0.36%, rank #1), EUR weakness (rank #4, COT reducing longs), risk-off (equities down, gold down, oil spiking). Technical: M5 TREND_DOWN (ADX=27.5, AroonOsc=-93, ST=-), H1 RANGE but bearish bias (%R=-80, MFI=27.7 weak). Breakout-failure method waits for failed bounce near S1 1.1696, then SHORT. Spread tight (0.8p), calendar clear, per-trade risk 1,050.53 JPY caps loss, RR 6.00 offers 6,225 JPY (29.6% of remaining target). Campaign 8.64% progress, aggressive pursuit needed. autotrade-cycle will refresh snapshot and re-validate geometry before send.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

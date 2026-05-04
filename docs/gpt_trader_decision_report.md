# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T09:29:35.916468+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `range_trader:AUD_JPY:LONG:RANGE_ROTATION`
- Cancel order ids: `none`
- Confidence: `MEDIUM`
- Operator summary: TRADE decision: AUD_JPY LONG via range_trader LIMIT entry at 112.948, TP 113.764, SL 112.846, risk 408 JPY (RR=8.00). Thesis: currency strength differential (AUD rank 2, JPY rank 8) + M15 uptrend (ADX=44.4) + M5 overbought pullback setup. Risk: MTF conflict (H1 downtrend ADX=34.8), trading with COT crowd (LONG AUD +48k, SHORT JPY -76k). Only lane under per-trade cap 420 JPY; other lanes blocked by sizing. Progress 8.64% of target, 15 LIVE_READY lanes, no event windows — TRADE justified.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

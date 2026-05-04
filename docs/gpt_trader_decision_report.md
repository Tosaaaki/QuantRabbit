# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T23:29:37.112500+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `range_trader:AUD_JPY:LONG:RANGE_ROTATION`
- Cancel order ids: `none`
- Confidence: `MEDIUM`
- Operator summary: TRADE: AUD_JPY LONG LIMIT mean-reversion from M5 oversold (%R -71, mean_rev 1.73). Counter-trend in H1 downtrend (trend -0.79, ADX 46) but M5 bounce setup valid. Entry 112.637, TP 113.405 (RR 8.00), SL 112.541 (960 JPY risk << 1050 budget). ASIA Golden Week thin session acknowledged; size conservative, LIMIT order reduces chase. Best available setup, campaign requires action (progress 12.52%).

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

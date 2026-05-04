# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T08:45:14.295365+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `range_trader:AUD_JPY:LONG:RANGE_ROTATION`
- Cancel order ids: `none`
- Confidence: `MEDIUM`
- Operator summary: TRADE range_trader:AUD_JPY:LONG:RANGE_ROTATION. M5 overbought bounce (%R=-2.1) setup; LIMIT at 112.900 = support buy 3.4 pips below bid, 2.5 pips above M5 structure break. Only AUD_JPY lanes within per_trade_risk_budget (420 JPY); actual risk=408 JPY (0.19% equity). Counter-trend H1/M15 acknowledged but sized accordingly. 8.00 RR targeting 3,264 JPY (17% of remaining 19,196 JPY target). Progress 8.6%, campaign requires action not defense. 15 LIVE_READY lanes available; no contract gate blocks execution. Previous WAIT (08:27 UTC) cited spread blocker (2.1p); current snapshot shows spread normalized to 1.7p, removing blocker. Operator: Claude Sonnet 4.5.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

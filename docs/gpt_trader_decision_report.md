# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T16:22:33.732437+00:00`
- Status: `REJECTED`
- Action: `WAIT`
- Selected lane: `None`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: WAIT due to market closure (STALE_QUOTE gate). All 15 generated intents blocked by quotes aged 49+ seconds exceeding 20s freshness threshold. Current time 01:20 JST Tuesday in FX weekend gap between NY close and Sydney open (~5h 40m). Chart structure shows strong directional bias (EUR/AUD/GBP shorts, USD_JPY long) with NORMAL spreads. Campaign at 8.64% of 10% target with full risk budget intact. Next cycle after Sydney 07:00 JST open when quotes refresh.

## Verification Issues

- `BLOCK` WAIT_MISSING_LIVE_READY_REJECTION: WAIT must cite at least one current LIVE_READY lane evidence ref when clean tradeable lanes exist and the daily target is still open

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

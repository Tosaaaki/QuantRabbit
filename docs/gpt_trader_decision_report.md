# GPT Trader Decision Report

- Generated at UTC: `2026-05-04T10:04:15.866509+00:00`
- Status: `ACCEPTED`
- Action: `TRADE`
- Selected lane: `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE`
- Cancel order ids: `none`
- Confidence: `HIGH`
- Operator summary: Executing AUD_JPY LONG BREAKOUT_FAILURE at 113.014 with 8.00R setup (3840 JPY reward, 480 JPY risk). Strongest currency divergence signal in current market (AUD #1 vs JPY #8). COT shows leveraged funds heavily short JPY (-75802) and long AUD (+47855), we're fading the JPY shorts on a recovery structure. M5 CHOCH_UP@112.968 confirms bottom, M15 ADX=41.7 TREND_UP shows momentum building. Entry above consolidation confirms breakout. No calendar risk, spread NORMAL (1.6p), risk well-bounded at 45.7% of per-trade cap. Campaign needs 19196 JPY more (91.4% of target remaining), this setup offers 19.9% of remaining target in single trade if TP hits. STALE_QUOTE blockers will clear when autotrade-cycle refreshes snapshot at send time.

## Verification Issues

- none

## Decision Contract

- GPT is the discretionary reasoning layer; deterministic verification remains the execution gate.
- `TRADE` requires a known `LIVE_READY` lane and no pending-entry or non-layerable exposure.
- Evidence refs must come from the input packet; invented refs reject the decision.

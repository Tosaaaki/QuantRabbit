# Autotrade Cycle Report

- Generated at UTC: `2026-05-06T11:43:48.348393+00:00`
- Status: `BLOCKED`
- Positions: `1`
- Orders: `1`
- Live-ready intents: `19`
- Receipt promotions: `0`
- Decision source: `deterministic_basket`
- Deterministic lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET`
- Selected lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET`
- Selected basket lanes: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:MARKET`
- Selected lane score: `211.0`
- Selected lane size multiple: `1.0`
- Sent: `False`
- Sent count: `0`
- Canceled orders: `none`
- Position management: `NO_POSITION`
- Position execution: `NO_ACTION` sent=`False`
- Daily target: `PURSUE_TARGET` remaining=`20894.58` progress_pct=`0.0`
- GPT trader: status=`ACCEPTED` action=`TRADE` allowed=`True` issues=`0`
- GPT error: `none`
- GPT wait recovery attempts: `0`
- GPT recovery source: `none`
- Campaign exposure required: `False`
- Market artifact mode: `reuse_existing`
- Market story refresh: `True` (source: `/Users/tossaki/App/QuantRabbit/logs`)

## Cycle Contract

- Protected trader-owned positions and trader-owned pending entries may add only through basket portfolio risk validation.
- If basket portfolio validation has no capacity, pending entries remain monitor-only.
- Open positions are handed to PositionManager first, then the protection gateway may close, repair protection, or tighten SL when the action is risk-reducing.
- If a pending entry came from a now-vetoed lane, the cycle may cancel it before waiting for the next cycle.
- If flat, risk-repair or trigger receipts may promote the strategy profile before TraderBrain compares lanes.
- If the daily target is open, the trader is flat, and LIVE_READY lanes survive prefiltering, the cycle must recover to a lane instead of preserving discretionary flatness.
- If the daily target is already reached while flat, the cycle records protection-first no-send status and adds no fresh risk.
- If GPT trader handoff is enabled, the selected lane must also be an accepted GPT `TRADE` decision from the deterministic prefilter set.
- If flat, the cycle refreshes broker truth immediately before pricing intents unless `--reuse-market-artifacts` pins the already generated decision packet; the live gateway still refreshes broker truth before any stage/send.

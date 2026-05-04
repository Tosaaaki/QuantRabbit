# Autotrade Cycle Report

- Generated at UTC: `2026-05-04T13:21:45.159382+00:00`
- Status: `STAGED`
- Positions: `0`
- Orders: `0`
- Live-ready intents: `12`
- Receipt promotions: `0`
- Decision source: `gpt_trader`
- Deterministic lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
- Selected lane: `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE`
- Selected lane score: `186.44`
- Selected lane size multiple: `1.0`
- Sent: `False`
- Canceled orders: `none`
- Position management: `none`
- Position execution: `none` sent=`False`
- Daily target: `PURSUE_TARGET` remaining=`19195.67` progress_pct=`8.6385`
- GPT trader: status=`ACCEPTED` action=`TRADE` allowed=`True` issues=`0`
- GPT error: `none`
- GPT wait recovery attempts: `0`
- GPT recovery source: `none`
- Market story refresh: `True` (source: `/Users/tossaki/App/QuantRabbit/logs`)

## Cycle Contract

- Protected trader-owned positions may add only through portfolio risk validation; pending entries remain monitor-only.
- Open positions are handed to PositionManager first, then the protection gateway may close, repair protection, or tighten SL when the action is risk-reducing.
- If a pending entry came from a now-vetoed lane, the cycle may cancel it before waiting for the next cycle.
- If flat, risk-repair or trigger receipts may promote the strategy profile before TraderBrain compares lanes.
- If the daily target is already reached while flat, the cycle records protection-first no-send status and adds no fresh risk.
- If GPT trader handoff is enabled, the selected lane must also be an accepted GPT `TRADE` decision from the deterministic prefilter set.
- If flat, the cycle refreshes broker truth, regenerates intents, asks TraderBrain to compare lanes, and sends only the selected lane when live mode is explicitly enabled.

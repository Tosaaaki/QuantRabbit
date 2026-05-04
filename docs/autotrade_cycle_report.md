# Autotrade Cycle Report

- Generated at UTC: `2026-05-04T23:36:45.480310+00:00`
- Status: `CANCELED_CONTAMINATED_PENDING`
- Positions: `1`
- Orders: `3`
- Live-ready intents: `2`
- Receipt promotions: `0`
- Decision source: `deterministic`
- Deterministic lane: `None`
- Selected lane: `None`
- Selected lane score: `None`
- Selected lane size multiple: `None`
- Sent: `False`
- Canceled orders: `470194`
- Position management: `HOLD_PROTECTED`
- Position execution: `NO_ACTION` sent=`False`
- Daily target: `PURSUE_TARGET` remaining=`18604.2301` progress_pct=`11.4534`
- GPT trader: status=`not used` action=`None` allowed=`None` issues=`None`
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

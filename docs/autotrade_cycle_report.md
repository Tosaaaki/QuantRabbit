# Autotrade Cycle Report

- Generated at UTC: `2026-04-30T17:09:47.591498+00:00`
- Status: `MONITOR_ONLY_EXPOSURE_OPEN`
- Positions: `1`
- Orders: `2`
- Live-ready intents: `12`
- Selected lane: `None`
- Sent: `False`
- Canceled orders: `none`
- Position management: `HOLD_PROTECTED`
- Position execution: `NO_ACTION` sent=`False`

## Cycle Contract

- If any open position or pending order exists, the cycle is monitor-only for fresh entries and sends no new entry.
- Open positions are handed to PositionManager first, then the protection gateway may close, repair protection, or tighten SL when the action is risk-reducing.
- If a pending entry came from a now-vetoed lane, the cycle may cancel it before waiting for the next cycle.
- If flat, the cycle refreshes broker truth, regenerates intents, asks TraderBrain to compare lanes, and sends only the selected lane when live mode is explicitly enabled.

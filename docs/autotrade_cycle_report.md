# Autotrade Cycle Report

- Generated at UTC: `2026-04-30T17:01:45.842743+00:00`
- Status: `MONITOR_ONLY_EXPOSURE_OPEN`
- Positions: `1`
- Orders: `2`
- Live-ready intents: `12`
- Selected lane: `None`
- Sent: `False`
- Canceled orders: `none`
- Position management: `HOLD_PROTECTED`

## Cycle Contract

- If any open position or pending order exists, the cycle is monitor-only and sends no fresh entry.
- If a pending entry came from a now-vetoed lane, the cycle may cancel it before waiting for the next cycle.
- If flat, the cycle refreshes broker truth, regenerates intents, asks TraderBrain to compare lanes, and sends only the selected lane when live mode is explicitly enabled.

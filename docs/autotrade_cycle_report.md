# Autotrade Cycle Report

- Generated at UTC: `2026-04-30T16:42:26.564708+00:00`
- Status: `MONITOR_ONLY_EXPOSURE_OPEN`
- Positions: `0`
- Orders: `1`
- Live-ready intents: `0`
- Selected lane: `None`
- Sent: `False`

## Cycle Contract

- If any open position or pending order exists, the cycle is monitor-only and sends no fresh entry.
- If flat, the cycle refreshes broker truth, regenerates intents, selects the first live-ready lane, and sends only when live mode is explicitly enabled.

# Autotrade Cycle Report

- Generated at UTC: `2026-04-30T16:55:02.312353+00:00`
- Status: `SENT`
- Positions: `0`
- Orders: `0`
- Live-ready intents: `12`
- Selected lane: `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE`
- Sent: `True`
- Canceled orders: `none`

## Cycle Contract

- If any open position or pending order exists, the cycle is monitor-only and sends no fresh entry.
- If a pending entry came from a now-vetoed lane, the cycle may cancel it before waiting for the next cycle.
- If flat, the cycle refreshes broker truth, regenerates intents, asks TraderBrain to compare lanes, and sends only the selected lane when live mode is explicitly enabled.

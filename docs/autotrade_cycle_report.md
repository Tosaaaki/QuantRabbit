# Autotrade Cycle Report

- Generated at UTC: `2026-05-17T23:03:45.543279+00:00`
- Status: `MONITOR_ONLY_EXPOSURE_OPEN`
- Positions: `3`
- Orders: `0`
- Live-ready intents: `0`
- Receipt promotions: `0`
- Decision source: `deterministic`
- Deterministic lane: `None`
- Selected lane: `None`
- Selected basket lanes: `none`
- Selected lane score: `None`
- Selected lane size multiple: `None`
- Sent: `False`
- Sent count: `0`
- Canceled orders: `none`
- Position management: `HOLD_PROTECTED`
- Position execution: `NO_ACTION` sent=`False`
- Daily target: `REPAIR_REQUIRED` remaining=`25918.684` progress_pct=`-34.0995`
- GPT trader: status=`not used` action=`None` allowed=`None` issues=`None`
- GPT error: `none`
- GPT wait recovery attempts: `0`
- GPT recovery source: `none`
- Campaign exposure required: `False`
- Market artifact mode: `refresh_and_reprice`
- Market story refresh: `False` (source: `/Users/tossaki/App/QuantRabbit/logs`)

## Cycle Contract

- Protected trader-owned positions and trader-owned pending entries may add only through basket portfolio risk validation.
- If basket portfolio validation has no capacity, pending entries remain monitor-only.
- Open positions are handed to PositionManager first; trader-owned positions may close/repair/tighten when gated, while manual/tagless positions are TP-only.
- If a pending entry came from a now-vetoed lane, the cycle may cancel it before waiting for the next cycle.
- A verified GPT `CANCEL_PENDING` cancels only current trader-owned pending entry ids and sends no fresh entry in that cycle.
- If flat, risk-repair or trigger receipts may promote the strategy profile before TraderBrain compares lanes.
- If the daily target is open, the trader is flat, and LIVE_READY lanes survive prefiltering, the cycle must recover to a lane instead of preserving discretionary flatness.
- If the daily target is already reached while flat, the cycle records protection-first no-send status and adds no fresh risk; trader-owned pending entries are canceled instead of left fillable.
- If GPT trader handoff is enabled, the selected lane must also be an accepted GPT `TRADE` decision from the deterministic prefilter set.
- If flat, the cycle refreshes broker truth immediately before pricing intents unless `--reuse-market-artifacts` pins the already generated decision packet; the live gateway still refreshes broker truth before any stage/send.

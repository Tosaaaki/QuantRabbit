# Memory Health Report

- Generated at UTC: `2026-07-05T18:09:03.987825+00:00`
- Status: `MEMORY_HEALTH_BLOCKED`

## Layers

- `short_term`: `BLOCK`
- `medium_term`: `PASS`
- `long_term`: `PASS`
- `position_memory`: `PASS`

## Issues

- `WARN` `short_term` `SHORT_FORECAST_HISTORY_STALE_WHILE_QUOTES_STALE`: forecast_history latest row predates broker snapshot (forecast=2026-07-02T04:08:10.563128+00:00, snapshot=2026-07-05T18:08:49.470675+00:00) while latest quote is also stale (quote=2026-07-03T20:59:05.386059+00:00)
- `BLOCK` `short_term` `SHORT_ORDER_INTENTS_MEMORY_BLOCKERS`: order_intents contains 1 memory/telemetry blocker(s)

## Contract

- This audit does not grant permission to trade.
- BLOCK means a memory artifact is missing, stale, unreconciled, or internally inconsistent before routing.
- Final broker send remains governed by RiskEngine, IntentGenerator telemetry validation, and LiveOrderGateway.

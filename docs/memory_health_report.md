# Memory Health Report

- Generated at UTC: `2026-07-06T15:02:44.342933+00:00`
- Status: `MEMORY_HEALTH_BLOCKED`

## Layers

- `short_term`: `BLOCK`
- `medium_term`: `PASS`
- `long_term`: `PASS`
- `position_memory`: `PASS`

## Issues

- `BLOCK` `short_term` `SHORT_ORDER_INTENTS_MEMORY_BLOCKERS`: order_intents contains 2 memory/telemetry blocker(s)

## Contract

- This audit does not grant permission to trade.
- BLOCK means a memory artifact is missing, stale, unreconciled, or internally inconsistent before routing.
- Final broker send remains governed by RiskEngine, IntentGenerator telemetry validation, and LiveOrderGateway.

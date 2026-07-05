# Remaining Profitability P0 Decomposition

- Generated: `2026-07-05T17:34:58Z`
- Rows: `5`
- Can create live permission: `False`

| blocker | row | class | trades/families | residual gate | market-close gate | live permission | action |
|---|---|---|---|---|---|---|---|
| `SELF_IMPROVEMENT_P0_PRESENT` | `MEMORY_HEALTH_BLOCKED` | `ACTIVE_BLOCKER` | none | `False` | `False` | `False` | Refresh forecast/market evidence, regenerate order_intents, rerun memory-health and self-improvement. |
| `SELF_IMPROVEMENT_P0_PRESENT` | `TARGET_OPEN_NO_LIVE_READY_LANES` | `ACTIVE_BLOCKER` | none | `False` | `False` | `False` | Use the firepower board and proof-pack queue to repair exact lane blockers; do not end with optimism. |
| `NEGATIVE_EXPECTANCY_ACTIVE` | `NEGATIVE_EXPECTANCY_ACTIVE` | `NEGATIVE_EXPECTANCY_REALIZED` | none | `False` | `False` | `False` | Repair market-close leakage and residual families, then rerun capture-economics and profitability-acceptance. |
| `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE` | `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE` | `MARKET_CLOSE_LEAK_FAMILY` | 470730,471089,470353,471174,470356 | `False` | `True` | `False` | Preserve attached TP/HARVEST shape; block or repair loss-side market-close path. |
| `MARKET_CLOSE_LEAK_FAMILY_BLOCKED` | `MARKET_CLOSE_LEAK_FAMILY_BLOCKED` | `MARKET_CLOSE_LEAK_FAMILY` | 470356,470353,470730,471174,471089,471255,472280 | `False` | `True` | `False` | Keep the family DRY_RUN_BLOCKED and non-permission until the full exception proof stack exists. |

## Dependency Graph

- `MEMORY_HEALTH_BLOCKED` -> blocks `A/S LIVE_READY, normal routing`; requires: memory-health must PASS after fresh forecast/quote/order-intent evidence; stale short forecast history cannot support live routing.
- `TARGET_OPEN_NO_LIVE_READY_LANES` -> blocks `A/S LIVE_READY, normal routing`; requires: regenerated order_intents must contain at least one LIVE_READY lane after profitability, telemetry, guardian, risk, and gateway gates pass.
- `NEGATIVE_EXPECTANCY_ACTIVE` -> blocks `A/S LIVE_READY, normal routing`; requires: Regenerated post-gate realized capture economics must be non-negative without relying only on excluded blocked historical families. Exact positive attached-TP HARVEST evidence may enter a proof pack, but cannot create live permission by itself.
- `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE` -> blocks `A/S LIVE_READY, normal routing`; requires: TP-proven segment must no longer be net-damaged by MARKET_ORDER_TRADE_CLOSE leakage, or every retained loss-side market close must have durable close-gate and contained-risk timing proof.
- `MARKET_CLOSE_LEAK_FAMILY_BLOCKED` -> blocks `A/S LIVE_READY, normal routing`; requires: Exact close-gate proof, contained-risk timing evidence, and TP-proven exception evidence must all exist for EUR_USD LONG BREAKOUT_FAILURE before the family can route.

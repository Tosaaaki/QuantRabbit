# Entry Decision

## Use When

- Daily target remains open.
- Account is flat or existing trader-owned exposure is layerable.
- Current `LIVE_READY` lane(s) exist in `data/order_intents.json`.

## Decision Order

1. Read `docs/trader_prompts/20_market_packet.md`.
2. List every current `LIVE_READY` lane.
3. Intersect `ai_attack_advice.recommended_now_lane_ids` with current tradeable lanes.
4. If the target is open and the first advised lane is tradeable, include it in the selected basket unless a named deterministic gate now blocks it.
5. If advice spans multiple distinct pairs, include one lane per advised pair up to portfolio capacity, or cite a named gate per skipped pair in `risk_notes`.
6. Prefer a `MARKET` variant for immediate participation when it is current `LIVE_READY`; pending entries are basket-counted by the gateway and are not blanket no-trade reasons.
7. Write exactly one `data/codex_trader_decision_response.json`.

## Valid Actions

- `TRADE`
- `WAIT`
- `REQUEST_EVIDENCE`

## WAIT Discipline

- WAIT is valid only when a named contract or packet gate fires.
- Generic market prose is not a gate.
- Past losses, low old capture rate, missing positive mined evidence, or stale rejection memory are audit context only after current risk geometry is `LIVE_READY`.
- If `progress_pct < 50` and at least three `LIVE_READY` lanes exist, WAIT must reject each lane with a specific M5 chart-story sentence and cite the exact gate.

## Required TRADE Content

- Selected lane id and basket lane ids.
- Thesis, method, narrative, chart story, invalidation, TP, SL, units, expected reward, worst-case loss.
- Rejected alternatives.
- Risk notes naming `per_trade_risk_budget_jpy`, spread state, calendar state, strength alignment or conflict, and any COT warning.
- Evidence refs for broker, target, intent, campaign, strategy, story, charts, cross-asset, strength, flow, levels, calendar, COT, option skew, and attack advice when used.

## Specialist Reviews

- Optional and read-only.
- Must declare `read_only=true` and `live_permission=false`.
- Must not select lanes, set units, stage orders, send orders, cancel orders, or change risk budgets.
- Strategy reviews are keyed by `lane_id` and `method`; a review for one method cannot authorize another.

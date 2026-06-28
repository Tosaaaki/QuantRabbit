# QuantRabbit 5% Loop Goal Prompt

Generated from local/live artifacts on 2026-06-28 10:26 UTC / 19:26 JST.

## Current State

- Market calendar: weekend window on Sunday JST. Do not send orders or mutate launchd from this prompt.
- Development/live baseline before this document commit: `50c5e510`.
- Current document commit should be verified with `git log -1` in both development and live before the next runtime-affecting loop.
- Live worktree has report-only runtime drift under `docs/*_report.md`.
- Latest broker snapshot: `2026-06-26T06:48:39Z`, still shows `EUR_JPY SHORT 472876` open with TP `183.859` and SL `184.128`.
- Execution ledger is newer than the broker snapshot: `2026-06-26T07:03:46Z` records `EUR_JPY SHORT 472876` closed by `TAKE_PROFIT_MARKET` for `+413.0 JPY`.
- Daily target packet before that close: start equity `173,944.9465 JPY`, 5% floor `8,697.25 JPY`, 10% target `17,394.49 JPY`, remaining 5% floor `8,809.25 JPY`, remaining target `17,506.49 JPY`.
- Capture economics remains negative: win rate `59.1%`, PF `0.851`, expectancy `-163.7 JPY/trade`, net `-37,640.0 JPY` in trader-attributed realized capture report.
- Positive edge is concentrated in `TAKE_PROFIT_ORDER` net `+48,804.8 JPY`; dominant loss leak is `MARKET_ORDER_TRADE_CLOSE` net `-74,564.8 JPY`.
- The latest `+413.0 JPY` EUR_JPY win is arithmetically meaningful: roughly thirty similar clean wins would exceed the last known 5% floor, but only if loss-side market-close leakage stays bounded.
- Current execution-ledger breadth is not single-pair: favorable realized pockets include `USD_JPY SHORT`, `USD_CHF LONG`, `AUD_CAD LONG`, `USD_CAD LONG`, and `EUR_JPY SHORT`; the current loop must scan every broker-enabled pair and side before concluding there is no turnover.
- Current all-currency `order_intents` packet had `96` results and `0` `LIVE_READY` lanes. The top blockers were `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`, `RANGE_COUNTERTREND_RR_TOO_LOW`, `LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT`, `RANGE_ROTATION_BROADER_LOCATION_CHASE`, `STRATEGY_NOT_ELIGIBLE`, and `EXHAUSTION_RANGE_CHASE`; `SPREAD_TOO_WIDE` is present but not the sole blocker.
- Operational 5% reachability is false even though audit firepower says a 5% route is estimated. The live blockers are no current `LIVE_READY` lanes, blocked profitability acceptance, and fresh-entry send not allowed.

## Loop Objective

Drive QuantRabbit toward the daily 5% minimum as an audit and repair obligation, not as a promise of market returns. Each loop must either:

- realize or protect current broker-truth profit through the existing gateways,
- produce at least one current, verifier-accepted, gateway-valid `LIVE_READY` attached-TP HARVEST or other approved lane,
- clear a named profitability/close-gate evidence blocker,
- or explicitly report that the blocker is waiting for live evidence and must not be reimplemented.

## All-Currency Spread-Aware Turnover Contract

- The loop must evaluate every broker-enabled currency pair and side in the fresh `order_intents` packet. Do not replay only the last winning EUR_JPY shape if other pairs have cleaner current geometry.
- When the daily target remains open, rank all `LIVE_READY` lanes, all spread-normalized candidates, and the top all-currency blockers before reporting no-trade.
- A fresh entry starts by paying the bid/ask spread. Initial unrealized red P/L inside current spread plus M1 noise is expected execution cost, not a failed thesis and not a market-close reason.
- A negative close is acceptable only when it is machine-checkable: current broker truth, RiskEngine geometry, and close Gate A/B or broker-side SL/attached-TP mechanics must prove the loss is the designed invalidation cost.
- In `NEGATIVE_EXPECTANCY`, planned loss must stay below the capture-economics loss-asymmetry cap unless the exact attached-TP HARVEST vehicle is TP-proven. One loss must not erase multiple average TP wins.
- Increasing turnover means basketing every eligible pair the gateway can fit, then cutting only invalidated exposure. It does not mean lowering min-lot, bypassing spread, chasing broad-location extremes, or turning protective blockers into prose exceptions.

## Causal Blocker Order

1. Broker truth and target ledger mismatch.
   - The ledger has a post-snapshot close for `472876`; any next loop starts by refreshing broker snapshot, execution ledger, and daily target state.
   - Do not reason from the stale open-position snapshot after the ledger proves a later close.

2. Profitability acceptance P0s.
   - `NEGATIVE_EXPECTANCY_ACTIVE`
   - `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`
   - `RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK`
   - `LOSS_CLOSE_GATE_EVIDENCE_MISSING`
   - `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`
   - The loop must preserve TP-proven capture and must not increase churn while market-close leakage is still the dominant drag.

3. Fresh opportunity availability.
   - Current `LIVE_READY` count is `0`.
   - Top blockers are `NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION`, `RANGE_COUNTERTREND_RR_TOO_LOW`, `LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT`, `RANGE_ROTATION_BROADER_LOCATION_CHASE`, `STRATEGY_NOT_ELIGIBLE`, and `EXHAUSTION_RANGE_CHASE`.
   - Correct protective guardrails are not code bugs. Only edit code when a failing regression proves the blocker is false for an allowed shape.
   - Spread-only red at entry is not a blocker-clearance signal by itself. Recheck spread-normalized candidates after fresh broker truth; repair non-spread blockers separately.

4. Repair frontier.
   - Current orchestrator selected `REPAIR_FRONTIER_LANE_BLOCKER` as actionable auxiliary work.
   - Waiting P0s stay in scope: `REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY` and `REVIEW_CLOSE_GATE_EVIDENCE_FAILURES`.
   - Auxiliary frontier work is not operational 5% proof until P0 clearance evidence changes.

5. Forecast/projection measurement hygiene.
   - Forecast history currently has no duplicate `cycle_id,pair` rows in the quick check.
   - Projection ledger has many expired `PENDING` rows; the next refresh must run `verify-projections` before intent generation and acceptance review.

## Loop Phases

### Phase 0 - Read Only Precheck

Run no order, close, cancel, or launchd mutation. Confirm:

```bash
date -u '+%Y-%m-%d %H:%M:%S UTC'
TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S JST'
git -C /Users/tossaki/App/QuantRabbit log -1 --oneline --decorate
git -C /Users/tossaki/App/QuantRabbit-live log -1 --oneline --decorate
git -C /Users/tossaki/App/QuantRabbit-live status --short
```

### Phase 1 - Refresh Truth

When the market/runtime window is appropriate, refresh broker truth before selecting a lane:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10
```

Acceptance condition:

- `data/broker_snapshot.json` no longer contradicts the latest execution ledger close.
- `data/daily_target_state.json` includes the realized `+413.0 JPY` close if broker truth confirms it.
- Any open trader-owned position has fresh TP/SL/owner state.

### Phase 2 - Full Evidence Refresh

Use the consolidated command, not dozens of separate shell turns:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10
```

Acceptance condition:

- `order_intents`, `capture_economics`, `memory_health`, `self_improvement_audit`, `profitability_acceptance`, `trader_support_bot`, and `trader_repair_orchestrator` are all generated from the same fresh broker/market packet.
- `verify-projections` has run before intent generation.
- `trader_repair_orchestrator` either selects an actionable code/evidence repair or classifies the top P0 as waiting for live evidence.

### Phase 3 - Decision And Gateway

If there is a current `LIVE_READY` lane and no hard close-first blocker:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli trader-draft-decision \
  --snapshot data/broker_snapshot.json \
  --output data/codex_trader_decision_response.json

PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json
```

Then hand off immediately through the wrapper when live trading is intentionally enabled by the runtime:

```bash
QR_LIVE_ENABLED=1 ./scripts/run-autotrade-live.sh \
  --reuse-market-artifacts \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send
```

Acceptance condition:

- `gpt-trader-decision` is `ACCEPTED`.
- The selected lane remains deterministic-prefiltered and gateway-valid.
- If the receipt is not `TRADE`, the gateway/sidecar path still maintains existing positions.
- No workaround send follows a rejected or stale decision.

### Phase 4 - Repair Branch

If `LIVE_READY=0` or profitability acceptance remains blocked, choose exactly one repair lane:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot
PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator
```

Decision rules:

- If the selected request is `READY_FOR_CODEX_IMPLEMENTATION`, read its evidence, add a focused regression test, implement the smallest code/test/doc change, run targeted tests and full unittest, commit, and sync live.
- If the selected request is `WAITING_FOR_LIVE_EVIDENCE_WINDOW`, do not rewrite the same guard. Run only the named read-only evidence command and compare whether the clearance metric changed.
- If a blocker is margin/min-lot/exposure capacity, do not lower production lot size or bypass risk. Wait for broker state to change or use an approved exposure action.
- If a blocker is protective market-structure logic, do not loosen it unless a regression proves a false positive while the original invalid chase stays blocked.

### Phase 5 - Verify Progress

Progress is accepted only when at least one of these changes:

- `remaining_minimum_jpy` decreases from fresh broker truth.
- `capture_economics.status` is no longer `NEGATIVE_EXPECTANCY`, or entries are limited to exact TP-proven positive-expectancy HARVEST shapes.
- `profitability_acceptance.status` clears P0 blockers or documents that they aged out without new leaks.
- `verification-ledger-audit` proves recent loss-side closes have durable `close_gate_evidence`.
- `execution-timing-audit --lookback-hours 744 --post-close-hours 6 --max-events 80` shows month-scale replay non-negative or residual groups removed.
- `order_intents` has at least one `LIVE_READY` lane that passes verifier and gateway validation.
- `order_intents` or `coverage_optimization` shows all-currency opportunity breadth improving: more spread-normalized candidates, fewer non-spread blockers, or more unique tradable pair/side lanes.
- Realized negative closes remain "reasonable": average loss stays below the current average winner/loss-asymmetry cap, and recent loss-side `MARKET_ORDER_TRADE_CLOSE` count does not grow without durable close-gate evidence.
- `trader_support_bot` reports operational 5% reachable true.

## Reusable Goal Prompt

```text
You are operating QuantRabbit from /Users/tossaki/App/QuantRabbit.

Goal:
Drive the system toward the current trading day's 5% minimum floor as an audit/repair obligation. Do not promise market returns. Do not bypass broker truth, RiskEngine, LiveOrderGateway, PositionProtectionGateway, close Gate A/B, or position-guardian rules.

Current known state:
- Use /Users/tossaki/App/QuantRabbit-live artifacts as production truth.
- Development and live HEAD should match before runtime-affecting work.
- First reconcile broker truth against execution_ledger because the latest ledger may be newer than data/broker_snapshot.json.
- The last known daily floor was 5% of 173,944.9465 JPY = 8,697.25 JPY, but refresh daily-target-state before using any remaining amount.
- Capture economics is negative; TAKE_PROFIT_ORDER is the positive edge and MARKET_ORDER_TRADE_CLOSE is the dominant loss leak.
- The last EUR_JPY +413 JPY clean close proves the arithmetic shape, not a single-pair mandate. Scan all broker-enabled currencies and sides for the same or better current attached-TP / HARVEST / gateway-valid shape.
- A new trade begins spread-negative. Treat initial red inside current spread plus M1 noise as execution cost, not thesis failure. Only accept negative closes that pass close Gate A/B or broker-side designed invalidation, and keep loss size below the capture-economics loss-asymmetry cap unless exact TP-proven HARVEST evidence relaxes it.
- Audit firepower may estimate a 5% route, but operational 5% is false until profitability acceptance, LIVE_READY, guardian/send-readiness, and gateway validation are clear.

Loop:
1. Read docs/AGENT_CONTRACT.md and docs/SKILL_trader.md.
2. Run read-only precheck: UTC/JST time, dev/live HEAD, live worktree status.
3. Refresh broker snapshot, execution ledger, and daily target state before making any decision.
4. Run cycle-refresh --daily-risk-pct 10 as the single evidence refresh command.
5. Inspect only targeted artifacts: daily_target_state, order_intents LIVE_READY count and top blockers, capture_economics, profitability_acceptance, trader_support_bot, trader_repair_orchestrator.
6. For order_intents, inspect all currencies: unique pair/side coverage, spread-normalized candidates, spread-only blockers versus non-spread blockers, and every current LIVE_READY lane the gateway could fit.
7. If an open trader-owned position exists, prioritize TP/profit capture, TP rebalance, protection, or verified close discipline through PositionProtectionGateway. Do not close from fear, margin pressure, stale prose, or soft evidence without Gate B.
8. If flat or layerable and at least one LIVE_READY lane exists, draft one basket decision across every eligible pair the gateway can fit, verify it with gpt-trader-decision, then hand it immediately to the live wrapper only through the approved gateway path.
9. If no LIVE_READY lane or profitability acceptance is blocked, select one repair from trader-repair-orchestrator. Implement only READY_FOR_CODEX_IMPLEMENTATION work. For WAITING_FOR_LIVE_EVIDENCE_WINDOW, run the named evidence command and report the wait instead of reimplementing the same guard.
10. Verify progress using fresh metrics: remaining_minimum_jpy, capture expectancy/PF, close_gate_evidence durability, month-scale replay status, LIVE_READY count, and trader_support_bot operational_minimum_5pct_reachable.
11. For any runtime-affecting code/doc change, add focused tests, run targeted and full unittest, commit with Codex attribution, run scripts/sync-live-runtime.sh, and verify live HEAD matches.

Anti-loop rules:
- Do not rerun profitability-acceptance as the fix unless an input artifact, gateway proof, or live evidence window changed.
- Do not treat audit-only firepower as operational reachability.
- Do not lower MIN_PRODUCTION_LOT_UNITS or bypass min-lot, margin, spread, forecast, close, or protective market-structure gates.
- Do not scale churn while capture economics is negative unless the lane is an exact TP-proven positive-expectancy HARVEST/repair shape.
- Do not send orders, cancel orders, close positions, mutate launchd, or call model APIs from QuantRabbit code outside the existing gateway or explicit operator approval boundary.

End each loop with:
- current UTC/JST,
- broker truth freshness,
- remaining 5% floor and 10% target,
- open trader positions and pending entries,
- LIVE_READY count and top blockers,
- profitability acceptance status,
- selected repair or selected lane,
- exact verification command that proved progress or the exact evidence wait blocker.
```

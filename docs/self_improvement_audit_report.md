# Self Improvement Audit Report

- Generated at UTC: `2026-07-05T18:09:03.864348+00:00`
- Status: `SELF_IMPROVEMENT_BLOCKED`
- Audit history DB: `/Users/tossaki/App/QuantRabbit/data/self_improvement_history.db`

## Runtime

- Target open: `True`
- Open trader positions: `0`
- Open trader pending entries: `0`
- LIVE_READY lanes: `0`
- Position guardian: required=`True` active=`True` source=`launchd+heartbeat` launchd_loaded=`True`
- GPT status/action: `ACCEPTED` / `CANCEL_PENDING`

## Profitability

- Window `168.0`h: trades `19`, net `14798.22` JPY, PF `4.701`, expectancy `778.854` JPY
- Last 24h: trades `0`, net `0.00` JPY, PF `n/a`, expectancy `n/a` JPY

## Execution Quality

- Pending entry lifecycle: accepted `23`, filled `19`, canceled_before_fill `4`, open_unfilled `0`, fill_rate `0.826`, cancel_before_fill_rate `0.174`
- Pending cancel timing regret: audited `80`, entry_touched `48`, tp_touched `4`, missed_mfe_jpy `16478.123`
- Top pending cancel regret shape: `timing:canceled_shape:GBP_JPY:SHORT:RANGE_ROTATION:LIMIT_ORDER`, priority `PRESERVE_PENDING_THESIS_TP_TOUCHED`, orders `4`, missed_mfe_jpy `2616.000`
- Pending entry reconcile: reviewed `0`, cancel_review `0`, ids `none`

## Root Cause Focus

- Primary: `OPPORTUNITY_COVERAGE` score `160.000` confidence `HIGH`
- Why: campaign coverage is too small for the open target; profit_factor=4.701; codes=MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE,PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP,TARGET_OPEN_NO_LIVE_READY_LANES,PROFITABLE_BACKTEST_EDGE_STRATEGY_GATED
- Goal adjustment: Shift the goal from single-lane selection to building enough current LIVE_READY coverage for the remaining floor/target while preserving risk gates.
- Next: Promote only the nearest named forecast/strategy blockers into additional HARVEST or RUNNER candidates, then rerun coverage metrics.
- Supporting codes: `MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE`, `PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP`, `TARGET_OPEN_NO_LIVE_READY_LANES`, `PROFITABLE_BACKTEST_EDGE_STRATEGY_GATED`
- Downstream symptoms: `data freshness/integrity score=155.000`, `realized P/L leak score=115.000`, `forecast adverse path score=105.000`, `process loop score=60.000`

## Findings

- `P0` `memory` `MEMORY_HEALTH_BLOCKED`: memory_health is blocked with 1 issue(s) Next: Repair the first memory-health BLOCK, then rerun trader-prompt-route.
- `P1` `execution_quality` `LOSS_CLOSE_PROFIT_CAPTURE_MISSED`: 14 raw TP-progress miss(es) remain, but post-repair production-gate replay found no executable profit-capture trigger Next: Keep these rows as diagnostic only; use tick replay or improved candle ordering to upgrade them before blocking high-turnover entries.
- `P1` `forecast` `PROJECTION_ECONOMIC_PRECISION_WEAK`: 5 projection bucket(s) clear headline Wilson 90% precision but fail economic precision after TIMEOUT/no-touch penalties Next: Do not use the named projection buckets as 90% high-turn live support. Mine pair/direction/regime variants or tighten target/horizon geometry until economic_hit_rate Wilson clears the same live precision floor.
- `P1` `learning` `LEARNING_AUDIT_WARN`: learning_audit has 3 warning(s) Next: Do not increase learning score impact until the effect window improves.
- `P1` `opportunity` `MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE`: 1 blocked profitable edge(s) have same-side market-context matrix support Next: Use the matrix-supported edges as the next discovery repair queue, but keep forecast confidence, spread, strategy-profile, RiskEngine, and gateway gates intact.
- `P1` `opportunity` `PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP`: 8 profitable backtest edge(s) are missing or blocked in current candidate coverage Next: Repair the named historical-profitable pair/directions in strategy_profile, candidate generation, or live blockers before widening the discovery universe.
- `P1` `opportunity` `TARGET_OPEN_NO_LIVE_READY_LANES`: daily target is open but order_intents has no LIVE_READY lanes Next: Refresh broker truth and regenerate intents after quotes/spreads become tradable; do not treat market-evidence noise as a strategy expansion defect yet.
  - opportunity modes: HARVEST lanes=`73` live=`0` reward=`26569.6375` live_codes=`STALE_QUOTE, SPREAD_TOO_WIDE, TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE` codes=`GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, STALE_QUOTE`; RUNNER lanes=`0` live=`0` reward=`0.0` live_codes=`STALE_QUOTE, SPREAD_TOO_WIDE, NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION` codes=`GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, STALE_QUOTE`
  - perspective alignment: status=`NO_RANGE_METHOD_MISMATCH`, groups=`0`, lanes=`0`
  - runner candidates: status=`RUNNER_CANDIDATES_DEMOTED_TO_HARVEST`, trend=`17`, runner_qualified=`0`, attached_harvest=`17`, demotions=`UNCLEAR regime is not a clean runner trend=7, ADX 13.7 below trend threshold 25.0=3, ADX 15.5 below trend threshold 25.0=2`
- `P1` `process` `REPEATED_SELF_IMPROVEMENT_LOOP`: same self-improvement finding `MEMORY_HEALTH_BLOCKED` has persisted for 6 non-duplicate audit run(s) Next: Stop repeating broad refresh/analysis for this finding. Execute the current finding's named next_action as a narrow repair, then verify with its target metric before cycling back to the same diagnosis.
- `P1` `verification` `VERIFICATION_LEDGER_LANE_BLOCKERS_RECORDED`: verification ledger recorded 82 protective order-intent lane blocker(s); ledger integrity itself is not the P0 Next: Repair the order_intents top blockers instead of treating the verification ledger as broken.
- `P2` `forecast` `FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS`: forecast_history has 57 old no-cycle same-second cluster(s); legacy forecast evaluation must dedupe them Next: Keep legacy no-cycle rows deduped by pair/second/direction/confidence/target/invalidation when measuring forecast improvement.
- `P2` `forecast` `PROFITABLE_BACKTEST_EDGE_FORECAST_GATED`: 2 profitable backtest edge(s) are visible, but current forecast gates block live expansion Next: Treat these as forecast-repair evidence, not live coverage expansion; keep RiskEngine and forecast-confidence gates intact until the current prediction packet supports the side.
- `P2` `opportunity` `PROFITABLE_BACKTEST_EDGE_STRATEGY_GATED`: 5 profitable backtest edge(s) remain blocked by strategy-profile repair gates Next: Do not amplify these historical buckets until a current risk-resized dry-run receipt or new market-structure proof reopens the strategy profile gate.

## Next Actions

- `P1` `ROOT_CAUSE_FOCUS:OPPORTUNITY_COVERAGE`: Promote only the nearest named forecast/strategy blockers into additional HARVEST or RUNNER candidates, then rerun coverage metrics.
- `P0` `MEMORY_HEALTH_BLOCKED`: Repair the first memory-health BLOCK, then rerun trader-prompt-route.
- `P1` `LOSS_CLOSE_PROFIT_CAPTURE_MISSED`: Keep these rows as diagnostic only; use tick replay or improved candle ordering to upgrade them before blocking high-turnover entries.

## Contract

- This audit does not grant permission to trade.
- P0 means the next trader route should repair or explicitly account for the hole before adding risk.
- Live sends remain governed by broker truth, RiskEngine, IntentGenerator telemetry validation, LiveOrderGateway, and Gate A/Gate B close discipline.

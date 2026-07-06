# Self Improvement Audit Report

- Generated at UTC: `2026-07-06T15:10:21.818842+00:00`
- Status: `SELF_IMPROVEMENT_BLOCKED`
- Audit history DB: `/Users/tossaki/App/QuantRabbit-live/data/self_improvement_history.db`

## Runtime

- Target open: `True`
- Open trader positions: `0`
- Open trader pending entries: `0`
- LIVE_READY lanes: `0`
- Position guardian: required=`True` active=`True` source=`launchd+heartbeat` launchd_loaded=`True`
- GPT status/action: `REJECTED` / `REQUEST_EVIDENCE`

## Profitability

- Window `168.0`h: trades `15`, net `13175.23` JPY, PF `5.897`, expectancy `878.349` JPY
- Last 24h: trades `0`, net `0.00` JPY, PF `n/a`, expectancy `n/a` JPY

## Execution Quality

- Pending entry lifecycle: accepted `16`, filled `14`, canceled_before_fill `2`, open_unfilled `0`, fill_rate `0.875`, cancel_before_fill_rate `0.125`
- Pending cancel timing regret: audited `80`, entry_touched `48`, tp_touched `4`, missed_mfe_jpy `16504.710`
- Top pending cancel regret shape: `timing:canceled_shape:GBP_JPY:SHORT:RANGE_ROTATION:LIMIT_ORDER`, priority `PRESERVE_PENDING_THESIS_TP_TOUCHED`, orders `4`, missed_mfe_jpy `2616.000`
- Pending entry reconcile: reviewed `0`, cancel_review `0`, ids `none`

## Root Cause Focus

- Primary: `OPPORTUNITY_COVERAGE` score `345.000` confidence `HIGH`
- Why: campaign coverage is too small for the open target; profit_factor=5.897; same finding repeated 65 audit run(s); codes=TARGET_OPEN_NO_LIVE_READY_LANES,MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE,PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP,PROFITABLE_BACKTEST_EDGE_STRATEGY_GATED
- Goal adjustment: Shift the goal from single-lane selection to building enough current LIVE_READY coverage for the remaining floor/target while preserving risk gates.
- Next: Promote only the nearest named forecast/strategy blockers into additional HARVEST or RUNNER candidates, then rerun coverage metrics.
- Supporting codes: `TARGET_OPEN_NO_LIVE_READY_LANES`, `MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE`, `PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP`, `PROFITABLE_BACKTEST_EDGE_STRATEGY_GATED`
- Downstream symptoms: `data freshness/integrity score=155.000`, `realized P/L leak score=155.000`, `process loop score=30.000`, `forecast adverse path score=20.000`

## Findings

- `P0` `memory` `MEMORY_HEALTH_BLOCKED`: memory_health is blocked with 1 issue(s) Next: Repair the first memory-health BLOCK, then rerun trader-prompt-route.
- `P0` `opportunity` `TARGET_OPEN_NO_LIVE_READY_LANES`: daily target is open but order_intents has no LIVE_READY lanes Next: Refresh market context and inspect top live blockers instead of ending flat without a named gate.
  - opportunity modes: HARVEST lanes=`99` live=`0` reward=`22048.0614` live_codes=`none` codes=`none`; RUNNER lanes=`0` live=`0` reward=`0.0` live_codes=`none` codes=`none`
  - perspective alignment: status=`NO_RANGE_METHOD_MISMATCH`, groups=`0`, lanes=`0`
  - runner candidates: status=`RUNNER_CANDIDATES_DEMOTED_TO_HARVEST`, trend=`18`, runner_qualified=`0`, attached_harvest=`18`, demotions=`ATR percentile 0.06 is small-wave tape=4, ATR percentile 0.19 is small-wave tape=4, ATR percentile 0.16 is small-wave tape=2`
- `P1` `decision_history` `LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS`: latest GPT decision was already rejected with blocking verification issue(s); it is not an unconsumed live permission Next: Do not reuse the rejected receipt; write and verify a fresh decision against the current packet.
- `P1` `execution_quality` `LOSS_CLOSE_PROFIT_CAPTURE_MISSED`: 14 raw TP-progress miss(es) remain, but post-repair production-gate replay found no executable profit-capture trigger Next: Keep these rows as diagnostic only; use tick replay or improved candle ordering to upgrade them before blocking high-turnover entries.
- `P1` `learning` `LEARNING_AUDIT_WARN`: learning_audit has 2 warning(s) Next: Do not increase learning score impact until the effect window improves.
- `P1` `opportunity` `MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE`: 2 blocked profitable edge(s) have same-side market-context matrix support Next: Use the matrix-supported edges as the next discovery repair queue, but keep forecast confidence, spread, strategy-profile, RiskEngine, and gateway gates intact.
- `P1` `opportunity` `PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP`: 8 profitable backtest edge(s) are missing or blocked in current candidate coverage Next: Repair the named historical-profitable pair/directions in strategy_profile, candidate generation, or live blockers before widening the discovery universe.
- `P1` `process` `REPEATED_SELF_IMPROVEMENT_LOOP`: same self-improvement finding `TARGET_OPEN_NO_LIVE_READY_LANES` has persisted for 65 non-duplicate audit run(s) Next: Stop repeating broad refresh/analysis for this finding. Execute the current finding's named next_action as a narrow repair, then verify with its target metric before cycling back to the same diagnosis.
- `P1` `profitability` `SMALL_WIN_LARGE_LOSS_ASYMMETRY`: average loss 2690.70 JPY is more than 2x average win 1133.28 JPY Next: Audit TP capture, giveback, and close discipline for the worst losing segment.
- `P1` `verification` `VERIFICATION_LEDGER_LANE_BLOCKERS_RECORDED`: verification ledger recorded 134 protective order-intent lane blocker(s); ledger integrity itself is not the P0 Next: Repair the order_intents top blockers instead of treating the verification ledger as broken.
- `P2` `assumption_ablation` `LEGACY_REVIEW_EXIT_HISTORICAL_DRAG`: Legacy REVIEW_EXIT market-close losses are historical and separated from current GPT_CLOSE Gate A/B evidence Next: Keep the historical REVIEW_EXIT loss cluster as audit evidence, but do not let it occupy the current close-gate P1 slot unless fresh 24h REVIEW_EXIT losses reappear.
- `P2` `forecast` `FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS`: forecast_history has 342 old no-cycle same-second cluster(s); legacy forecast evaluation must dedupe them Next: Keep legacy no-cycle rows deduped by pair/second/direction/confidence/target/invalidation when measuring forecast improvement.
- `P2` `forecast` `PROFITABLE_BACKTEST_EDGE_FORECAST_GATED`: 1 profitable backtest edge(s) are visible, but current forecast gates block live expansion Next: Treat these as forecast-repair evidence, not live coverage expansion; keep RiskEngine and forecast-confidence gates intact until the current prediction packet supports the side.
- `P2` `opportunity` `PROFITABLE_BACKTEST_EDGE_STRATEGY_GATED`: 5 profitable backtest edge(s) remain blocked by strategy-profile repair gates Next: Do not amplify these historical buckets until a current risk-resized dry-run receipt or new market-structure proof reopens the strategy profile gate.

## Next Actions

- `P0` `ROOT_CAUSE_FOCUS:OPPORTUNITY_COVERAGE`: Promote only the nearest named forecast/strategy blockers into additional HARVEST or RUNNER candidates, then rerun coverage metrics.
- `P0` `MEMORY_HEALTH_BLOCKED`: Repair the first memory-health BLOCK, then rerun trader-prompt-route.
- `P0` `TARGET_OPEN_NO_LIVE_READY_LANES`: Refresh market context and inspect top live blockers instead of ending flat without a named gate.

## Contract

- This audit does not grant permission to trade.
- P0 means the next trader route should repair or explicitly account for the hole before adding risk.
- Live sends remain governed by broker truth, RiskEngine, IntentGenerator telemetry validation, LiveOrderGateway, and Gate A/Gate B close discipline.

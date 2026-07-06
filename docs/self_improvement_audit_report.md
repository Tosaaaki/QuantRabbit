# Self Improvement Audit Report

- Generated at UTC: `2026-07-06T01:58:38.176453+00:00`
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

- Pending entry lifecycle: accepted `22`, filled `18`, canceled_before_fill `4`, open_unfilled `0`, fill_rate `0.818`, cancel_before_fill_rate `0.182`
- Pending cancel timing regret: audited `4`, entry_touched `2`, tp_touched `1`, missed_mfe_jpy `714.262`
- Top pending cancel regret shape: `timing:canceled_shape:GBP_JPY:SHORT:RANGE_ROTATION:LIMIT_ORDER`, priority `PRESERVE_PENDING_THESIS_TP_TOUCHED`, orders `1`, missed_mfe_jpy `630.000`
- Pending entry reconcile: reviewed `0`, cancel_review `0`, ids `none`

## Root Cause Focus

- Primary: `FORECAST_ADVERSE_PATH` score `215.000` confidence `HIGH`
- Why: forecast path or economic precision is failing before entry expansion; projection_economic_precision_gap_count=5.000; projection_economic_precision_edge_count=10.000; projection_worst_economic_wilson_lower=0.337; profit_factor=4.701; same finding repeated 65 audit run(s); codes=PROJECTION_ECONOMIC_PRECISION_WEAK,FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS,PROFITABLE_BACKTEST_EDGE_FORECAST_GATED
- Goal adjustment: Shift the immediate improvement goal from more entries to reducing invalidation-first directional forecasts, weak buckets, and timeout-heavy projection precision gaps; only then expand coverage.
- Next: Repair the named directional forecast or projection economic-precision buckets, then verify recent hit-rate, timeout, and invalidation-first metrics before increasing entry frequency.
- Supporting codes: `PROJECTION_ECONOMIC_PRECISION_WEAK`, `FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS`, `PROFITABLE_BACKTEST_EDGE_FORECAST_GATED`
- Downstream symptoms: `coverage shortfall score=160.000`, `realized P/L leak score=115.000`, `data freshness/integrity score=100.000`, `process loop score=60.000`

## Findings

- `P0` `execution_ledger` `EXECUTION_LEDGER_STALE`: execution ledger last transaction `472992` is behind broker snapshot `472994` Next: Run execution-ledger-sync before trusting decision or learning history.
- `P1` `execution_quality` `LOSS_CLOSE_PROFIT_CAPTURE_MISSED`: 1 raw TP-progress miss(es) remain, but post-repair production-gate replay found no executable profit-capture trigger Next: Keep these rows as diagnostic only; use tick replay or improved candle ordering to upgrade them before blocking high-turnover entries.
- `P1` `forecast` `PROJECTION_ECONOMIC_PRECISION_WEAK`: 5 projection bucket(s) clear headline Wilson 90% precision but fail economic precision after TIMEOUT/no-touch penalties Next: Do not use the named projection buckets as 90% high-turn live support. Mine pair/direction/regime variants or tighten target/horizon geometry until economic_hit_rate Wilson clears the same live precision floor.
- `P1` `learning` `LEARNING_AUDIT_WARN`: learning_audit has 3 warning(s) Next: Do not increase learning score impact until the effect window improves.
- `P1` `opportunity` `MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE`: 1 blocked profitable edge(s) have same-side market-context matrix support Next: Use the matrix-supported edges as the next discovery repair queue, but keep forecast confidence, spread, strategy-profile, RiskEngine, and gateway gates intact.
- `P1` `opportunity` `PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP`: 8 profitable backtest edge(s) are missing or blocked in current candidate coverage Next: Repair the named historical-profitable pair/directions in strategy_profile, candidate generation, or live blockers before widening the discovery universe.
- `P1` `opportunity` `TARGET_OPEN_NO_LIVE_READY_LANES`: daily target is open but order_intents has no LIVE_READY lanes Next: Refresh broker truth and regenerate intents after quotes/spreads become tradable; do not treat market-evidence noise as a strategy expansion defect yet.
  - opportunity modes: HARVEST lanes=`73` live=`0` reward=`26569.6375` live_codes=`none` codes=`none`; RUNNER lanes=`0` live=`0` reward=`0.0` live_codes=`none` codes=`none`
  - perspective alignment: status=`NO_RANGE_METHOD_MISMATCH`, groups=`0`, lanes=`0`
  - runner candidates: status=`RUNNER_CANDIDATES_DEMOTED_TO_HARVEST`, trend=`17`, runner_qualified=`0`, attached_harvest=`17`, demotions=`UNCLEAR regime is not a clean runner trend=7, ADX 13.7 below trend threshold 25.0=3, ADX 15.5 below trend threshold 25.0=2`
- `P1` `process` `REPEATED_SELF_IMPROVEMENT_LOOP`: same self-improvement finding `LEARNING_AUDIT_WARN` has persisted for 65 non-duplicate audit run(s) Next: Stop repeating broad refresh/analysis for this finding. Execute the current finding's named next_action as a narrow repair, then verify with its target metric before cycling back to the same diagnosis.
- `P1` `verification` `VERIFICATION_LEDGER_LANE_BLOCKERS_RECORDED`: verification ledger recorded 82 protective order-intent lane blocker(s); ledger integrity itself is not the P0 Next: Repair the order_intents top blockers instead of treating the verification ledger as broken.
- `P2` `forecast` `FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS`: forecast_history has 57 old no-cycle same-second cluster(s); legacy forecast evaluation must dedupe them Next: Keep legacy no-cycle rows deduped by pair/second/direction/confidence/target/invalidation when measuring forecast improvement.
- `P2` `forecast` `PROFITABLE_BACKTEST_EDGE_FORECAST_GATED`: 2 profitable backtest edge(s) are visible, but current forecast gates block live expansion Next: Treat these as forecast-repair evidence, not live coverage expansion; keep RiskEngine and forecast-confidence gates intact until the current prediction packet supports the side.
- `P2` `opportunity` `PROFITABLE_BACKTEST_EDGE_STRATEGY_GATED`: 5 profitable backtest edge(s) remain blocked by strategy-profile repair gates Next: Do not amplify these historical buckets until a current risk-resized dry-run receipt or new market-structure proof reopens the strategy profile gate.

## Next Actions

- `P1` `ROOT_CAUSE_FOCUS:FORECAST_ADVERSE_PATH`: Repair the named directional forecast or projection economic-precision buckets, then verify recent hit-rate, timeout, and invalidation-first metrics before increasing entry frequency.
- `P0` `EXECUTION_LEDGER_STALE`: Run execution-ledger-sync before trusting decision or learning history.
- `P1` `LOSS_CLOSE_PROFIT_CAPTURE_MISSED`: Keep these rows as diagnostic only; use tick replay or improved candle ordering to upgrade them before blocking high-turnover entries.

## Contract

- This audit does not grant permission to trade.
- P0 means the next trader route should repair or explicitly account for the hole before adding risk.
- Live sends remain governed by broker truth, RiskEngine, IntentGenerator telemetry validation, LiveOrderGateway, and Gate A/Gate B close discipline.

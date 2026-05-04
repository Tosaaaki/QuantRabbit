# Build Sequence

## Phase 1: Broker Truth And Risk Gateway

1. Import legacy history into `data/legacy_history.db`.
2. Read OANDA account, open trades, pending orders, and live quotes through a read-only client.
3. Validate proposed order intents through `RiskEngine`.
4. Keep live send unavailable until dry-run receipts prove the gateway blocks known loss paths.

## Phase 2: Strategy Mining

1. Mine `pretrade_outcomes`, `seat_outcomes`, `live_trade_log`, `trader_journal`, and daily state files.
2. Promote only pair/direction/vehicle contexts with positive expectancy and bounded loss.
3. Generate strategy candidates as order intents, not broker orders.
4. Treat `RISK_REPAIR_CANDIDATE` as not-live-ready until a dry-run receipt proves loss is capped under 500 JPY.
5. Feed the strategy profile into `risk-dry-run` so history status is enforced, not just reported.
6. Run `replay-backtest` to classify legacy days by target hit, target coverage, risk-repair requirement, and missed-edge capture gap.

## Phase 2B: Market Story And Method Switching

1. Mine `news_digest`, `news_flow_log`, `quality_audit`, `state.md`, daily state snapshots, and `strategy_memory.md`.
2. Convert narrative/chart evidence into a seven-pair `market_story_profile`.
3. Require each order intent to state `regime`, `narrative`, `chart_story`, `method`, and `invalidation`.
4. Reject live sends when the chosen method contradicts the stated market regime, such as range rotation inside a one-way trend impulse.

## Phase 2C: Multi-Desk Daily Campaign

1. Compute the daily campaign target as `day_start_balance * 10%`.
2. Run simultaneous desks: trend continuation, range rotation, breakout/failure, event-risk, and position management.
3. Let each desk propose, veto, or resize lanes from the same strategy and market-story evidence.
4. Portfolio Director must publish NOW / BACKUP / RUNNER candidates or an explicit coverage gap.
5. A 10% campaign target never overrides the risk gateway; it only forces missing coverage to be named.
6. `daily-target-state` records start equity, target JPY, realized/unrealized PnL, remaining target, open risk, and remaining risk budget.

## Phase 2D: Dry-Run Intent Generation

1. Read current broker truth to `data/broker_snapshot.json`.
2. Convert campaign lanes into priced order intents with entry, TP, SL, units, campaign role, and market context.
3. Batch-check the generated intents through `RiskEngine` and `StrategyProfile`.
4. Promote only `LIVE_READY` receipts to the future live gateway; `DRY_RUN_PASSED` still needs profile promotion, and `NEEDS_BROKER_SNAPSHOT` is a hard stop.

## Phase 2E: Verified GPT Trader Decision

1. Build the GPT input packet from broker snapshot, daily target state, order intents, campaign lanes, strategy profile, and market-story profile.
2. Require GPT output to match the strict trader decision schema and cite only packet evidence refs.
3. Reject `TRADE` when any broker exposure exists, the lane is unknown, the lane is not `LIVE_READY`, the method conflicts with the lane, or trade thesis fields are incomplete.
4. Keep the standalone decision command advisory by default; Codex automation writes the GPT decision receipt and `autotrade-cycle --use-gpt-trader --gpt-decision-response ...` may hand off only an ACCEPTED GPT `TRADE` whose lane is also in the deterministic TraderBrain prefilter set.

## Phase 3: Execution Gateway

1. Add one live gateway method behind `QR_LIVE_ENABLED=1`.
2. Require current broker snapshot, fresh quote, TP, SL, broker-truth risk JPY, reward/risk, spread, owner, market context, and campaign role.
3. Journal every accepted and rejected intent.

## Phase 4: Automation

1. Automation may observe and report first.
2. Automation may propose order intents after Phase 2.
3. Automation may execute only after Phase 3 has a tested live-send path.
4. GPT handoff is opt-in and no-send on rejected, hallucinated, non-live-ready, exposure-conflicting, or non-prefiltered decisions.

## Phase 5: Coverage And Gap Receipts

1. Run `optimize-coverage` after intent generation and receipt promotion.
2. Count only `LIVE_READY` reward as executable coverage.
3. Count `DRY_RUN_PASSED` reward only as potential coverage until profile blockers are promoted.
4. Emit action items for target gap, replay gap, missing risk budget, and blocked pairs.

## Phase 6: Execution Replay

1. Run `replay-execution` against a supplied quote path before unattended dry-run certification.
2. Fill pending entries only when the quote path crosses the order trigger.
3. Resolve TP/SL from quotes after fill and write replay PnL receipts.
4. Treat replay target hits as evidence, not live-profit assurance.

## Phase 7: Post-Trade Learning

1. Run `learn-post-trade` from execution/protection/outcome receipts.
2. Produce advisory profile update candidates only; do not mutate strategy memory silently.
3. Losses beyond the current risk cap become `BLOCK_UNTIL_NEW_EVIDENCE` candidates.

## Phase 8: Dry-Run Certification

1. Run `certify-dry-run` after coverage, replay, and learning receipts exist.
2. Block certification when any dry-run artifact requested or performed a live write.
3. Block certification when live-ready intents lack thesis or market context.
4. Certification is not live enablement; live still requires explicit gates and monitoring.

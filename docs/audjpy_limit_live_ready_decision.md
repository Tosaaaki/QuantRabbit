# AUD_JPY LIMIT LIVE_READY Decision

- Generated: `2026-07-06T09:42:39Z`
- Scope: `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`
- Mode: read-only; no orders, cancels, closes, SL/TP changes, execution flags, or broker mutation.

## Exact Answers

1. Can this candidate become PROOF_READY now?
   **No.**

2. If no, is the blocker sample shortage, forecast proof, geometry, RiskEngine, Gateway, GPT verifier, guardian/operator review, or profitability acceptance?
   **Blockers:** sample/daily-stability proof, forecast proof, RiskEngine, Gateway, GPT verifier, guardian/operator review, and profitability acceptance.
   **Not the primary blocker:** geometry. The LIMIT geometry exists, but it is diagnostic only and cannot create permission.

3. Is it REPAIR_REQUIRED, EVIDENCE_GAP, REJECTED, or HISTORICAL_ONLY?
   **REPAIR_REQUIRED.**

4. What is the single next proof needed?
   **Fresh S5 bid/ask spread-included replay proof for the exact AUD_JPY SHORT BREAKOUT_FAILURE LIMIT HARVEST vehicle that is daily-stable/live-grade:** at least 3 active days, max daily sample share <= 0.7, positive day rate >= 0.6667, and non-negative/positive expectancy after spread. It must replace the current rank-only contrarian support and not conflict with the current AUD_JPY DOWN negative replay block.

5. If rejected, name the next candidate.
   **Not rejected now.** If this lane later becomes REJECTED, the next proof-queue row is `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE` STOP-ENTRY, but it is farther from proof and also `REPAIR_REQUIRED`.

## Evidence

- A/S board: `LIVE_READY=0`, `as_live_ready_path_exists=false`, normal routing `BLOCKED`.
- Proof queue: closest candidate is this LIMIT lane, `REPAIR_REQUIRED`, proof distance `5`, `can_create_live_permission=false`.
- Current order intent: `DRY_RUN_BLOCKED`, `risk_allowed=false`.
- Current blockers: `RANGE_FORECAST_REQUIRES_RANGE_ROTATION`, `SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH`, `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`, `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`.
- Geometry: entry `112.502`, TP `112.165`, SL `112.627`, reward/risk `2.696`, risk `375.0` JPY, units `3000`.
- Attached TP proof: `AUD_JPY|SHORT|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER`, 6 wins / 6 trades, expectancy `992.7` JPY, but this is proof-collection support only.
- S5 replay conflict: direct `AUD_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY` has 1361 samples, 39 active days, 0 positive days, avg final `-3.365` pips.
- Positive contrarian support is rank-only: 40 samples, 2 active days, positive day rate `0.5`, max daily share `0.95`, daily stability `INSUFFICIENT_ACTIVE_DAYS`.
- Guardian/operator: `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`, `normal_routing_allowed=false`, unresolved issues `2`.
- Profitability: `PROFITABILITY_ACCEPTANCE_BLOCKED`, capture status `NEGATIVE_EXPECTANCY`.
- Manual safety: EUR_USD `472987` remains `OPERATOR_MANUAL` / `KEEP`; TP `472996` remains protected from automation.
- Notion reference: read-only scan/search found the 2026-07-03 QuantRabbit memory mirror, which is consistent with local state: no broker actions, EUR_USD `472987` manual keep, and no LIVE_READY gates cleared.

## Decision

Keep `LIVE_READY=0`, `PROOF_READY=0`, and normal routing `BLOCKED`. This candidate stays in the proof pack as a repair candidate only; it must not be traded or used to infer live permission.

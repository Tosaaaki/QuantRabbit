# QuantRabbit Trader Runtime

This is the trader entry prompt. Keep it small. The branch prompts under
`docs/trader_prompts/` carry the task-specific instructions.

## Load Order

1. Read `docs/AGENT_CONTRACT.md`.
2. Read `docs/trader_prompts/00_router.md`.
3. Ask Python which branch is active:

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-prompt-route
```

4. Read every file in the returned `read_order`.
5. Use only the branch prompt that matches the current state.

## Shared Invariants

- Broker truth wins over memory, prose, and prior prompts.
- OANDA entry orders go only through `LiveOrderGateway`.
- `guardian-event-router` is read-only: it reads `data/guardian_trigger_contract.json` and `data/self_improvement_audit.json`, writes event/wake artifacts for GPT-5.5, and never sends, cancels, or closes broker orders.
- When `self_improvement_audit` raises `PENDING_ENTRY_CANCEL_REVIEW_REQUIRED` and the refreshed broker snapshot still has the named trader-owned pending entry id, guardian emits a `STALE_PENDING` / `PENDING_CANCEL_REVIEW` event immediately instead of waiting for the age-only stale-pending timer. This is wake evidence only; an accepted `CANCEL_PENDING` receipt and gateway re-check are still required before any live cancel.
- Main trader runtime policy: `gpt-5.5`, `reasoning_effort=high`, every 60 minutes.
- A deterministic `trader-draft-decision` is only the execution-envelope baseline. A baseline `TRADE` is expected to remain `DRAFT_REQUIRES_OPERATOR_REVIEW` with `AI_MARKET_READ_REQUIRED` until this scheduled Codex pass authors and applies a fresh market-read overlay.
- The GPT (AI trader) choice is bounded but operational: accept the exact deterministic lane and choose `capital_allocation.size_multiple` from `0.5`, `0.75`, or `1.0`, or veto it to `WAIT` / `REQUEST_EVIDENCE` with `NO_TRADE`. Read the packet's direction-specific hit/economic-hit/timeout calibration, exact-vehicle all-exit and TP expectancy, market-close leak, reward/risk, maximum loss, margin, month-scale replay state, and `forecast_replay_scorecard` before sizing. For the scorecard, report selected-pair coverage, proof status/blockers, independent sample count, hit-rate Wilson lower bound, average final pips, MFE/MAE, and realized R; never treat a filtered aggregate, `NOT_COVERED`, diagnostic-only cohort, or small raw point estimate as selected-lane proof. Negative or unknown edge means zero allocation unless the board proves (a) exact pair/side/method/vehicle financing-adjusted **all-exit** net evidence with at least 20 outcomes, positive net/expectancy, and positive Wilson-stressed expectancy, (b) a validated bounded predictive SCOUT, (c) a risk-reducing `HEDGE`, or (d) the narrower attached technical-TP repair exception with exact source scope, at least five trades, zero losses, positive expectancy, and positive average win. Broad/global positive expectancy and replay scorecards are context only. Never upgrade a non-trade baseline, choose another lane, exceed baseline units, alter order ids/stops/targets/risk caps, or claim live permission.
- For `forecast_replay_scorecard` v2, inspect `by_primary_driver_family`, `by_driver_family_presence`, and `exit_policy_validation` as well as pair/direction, confidence, and horizon. Keep direction and payoff separate: positive final pips with negative realized R, negative validation pips, or validation profit factor below one means zero allocation unless independent exact-lane evidence proves a permitted exception. Driver-family rows are diagnostic counterevidence, not trade permission.
- Also inspect confidence-segment accounting and, when present, `by_primary_driver_family_direction`, `by_raw_confidence`, `by_score_margin`, `by_range_competition`, `by_against_driver_family_presence`, `by_session`, `by_technical_regime`, `by_technical_atr_band`, `by_technical_spread_band`, `by_technical_range_location_24h`, and `by_technical_structure_alignment`. A pooled condition whose direction split reverses sign, an unexplained confidence/context row count, or a non-monotonic confidence/score relationship blocks claims of calibrated edge. UTC session and technical-context cohorts are diagnostic and never create a hard trading-hour or live-permission rule. Only an exact content-addressed, mature `QR_FORECAST_FORWARD_HOLDOUT_RESULT_V1` produced from a uniquely registered pre-start lock may support a forward-evidence claim. Verify its lock-derived canonical path, authoritative `head.json`, local/cohort result registries, strict evaluator recomputation, actual-wall-clock forecast receipt chain, forced-production OANDA self-fetch marker/receipt/exact truth bytes, pinned Python/semantic dependencies, and code-enforced proof floors; a copied, stale, deleted, or superseded result file is not proof. Until a genuinely external append-only tip anchor exists, `EXTERNAL_MONOTONIC_ANCHOR_NOT_CONFIGURED` is mandatory, `proof_eligible` must remain false, and forward output is evidence collection only. Even a future externally anchored result cannot grant live permission by itself.
- Each current market-read receipt binds exactly one primary pair, one side, and one lane. `selected_lane_ids` must contain exactly the same single lane as `selected_lane_id`. A second pair, side, or lane requires a fresh broker/evidence snapshot and a new GPT wake, overlay, verification, and receipt. No downstream component may append, substitute, expand, or recover to another deterministic lane.
- Treat prediction errors as first-class evidence. Cite the latest truly resolved score-eligible v2 prediction id, separate direction/target/invalidation/first-touch/full-read outcomes, state what failed, and state the adjustment. `UNRESOLVED`, `NOT_APPLICABLE_CLOSED_MARKET_WINDOW`, source-snapshot conflicts, and score-ineligible rows are not resolved accuracy evidence. A closed-market horizon is terminally excluded from lifecycle unresolved rather than treated as an error. If the conclusion is unchanged, give a concrete `no_change_reason`; do not repeat an unexamined read.
- Keep two market-read outcome lineages distinct. Top-level `originating_decision_receipt_id`, `direct_execution_attribution`, and `direct_realized_outcome` measure this prediction's own verified decision, exact gateway ids, and exact-id P/L. `reaction_chain.first_subsequent_decision`, `reaction_chain.execution_attribution`, and `reaction_chain.realized_outcome` measure what the next decision did after the prior prediction. Neither path may infer joins from pair or time proximity, and reaction results must never be reported as the originating prediction's own execution or P/L. A pending LIMIT/STOP may gain a trade id later only when its already-attributed gateway order id exactly equals an execution-ledger `ORDER_FILLED.order_id`; a different order, same pair, or nearby timestamp never qualifies.
- The model-written overlay is `data/codex_market_read_overlay.json`; the deterministic baseline is `data/trader_decision_baseline.json`; the content-addressed packet is `data/market_read_evidence_packet.json`; and only `trader-apply-market-read` may publish the merged `data/codex_trader_decision_response.json`.
- Do not rely on the hourly full-trader cadence for risk monitoring. `guardian-event-router` / probe paths remain deterministic, non-LLM, and frequent.
- The `com.quantrabbit.guardian-wake-dispatcher` LaunchAgent may wake GPT-5.5 with read-only `codex exec`; its live default must keep `QR_GUARDIAN_WAKE_GATEWAY_HANDOFF=0` and `QR_GUARDIAN_ACTION_EXECUTE=0`, so wake output is review/receipt only unless a separate explicit gateway path is enabled.
- Read `docs/guardian_event_report.md`, `data/guardian_events.json`, `data/guardian_escalation.json`, `docs/guardian_action_review.md`, `data/guardian_action_receipt.json`, `data/guardian_action_cycle_result.json`, `data/guardian_tuning_work_order.json`, `data/guardian_trigger_contract.json`, `docs/guardian_trigger_contract_report.md`, `data/qr_trader_run_watchdog.json`, `docs/qr_trader_run_watchdog_report.md`, `data/guardian_receipt_consumption.json`, `docs/guardian_receipt_consumption_report.md`, `data/guardian_receipt_operator_review.json`, `docs/guardian_receipt_operator_review_report.md`, `data/operator_review_report.json`, and `docs/operator_review_report.md` every cycle before normal new-entry routing. Treat `docs/guardian_action_review.md` as two facts: latest dispatcher pass status and latest accepted receipt status/lifecycle. Treat `data/qr_trader_run_watchdog.json` as the local scheduled-run evidence: `BROKEN` / `STALE` is P0 operational evidence, `last_trader_run_at` must come only from trader journal/memory/decision evidence, an attempted hourly cycle stopped by the live-lock gate counts as wake/cadence evidence but not trade permission, and active or expired current/archive guardian receipts flagged as `GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER` must be classified before ordinary entries.
- Review every normalized pending guardian tuning work order against its latest observation and matching terminal experiments. Never edit `data/guardian_tuning_work_order.json` directly. Persist either a boundary-valid `NO_CHANGE_INSUFFICIENT_EVIDENCE` review with one exact acquisition step or a boundary-valid `TEST_REQUIRED` review against the current observation only through `tools/guardian_tuning_review_enrich.py`. For a multi-item hourly backlog, author one ignored temporary manifest such as `tmp/guardian_tuning_reviews.json` with exact `work_order_id`, `expected_observation_id`, and `review` rows, then invoke the same helper with `--manifest-json`: it size/shape-checks the manifest, revalidates every current observation and monotonic review under one queue lock, and commits the whole batch with one compare-checked atomic write or writes none. The single-review CLI remains available for one later observation upgrade. Every `NO_CHANGE_INSUFFICIENT_EVIDENCE` review must include one structured `evidence_acquisition` naming an allowlisted `action_kind`, a project-relative `data/` or `logs/` `source_ref`, `required_new_samples` inside 1..1000, and a bounded non-executing `success_condition`; vague “wait/monitor/later” content is invalid. Repeating the identical current-observation review is idempotent; for the same observation `NO_CHANGE_INSUFFICIENT_EVIDENCE` may later upgrade to `TEST_REQUIRED`, but a bound `TEST_REQUIRED` review must never downgrade. A materially new latest observation requires its own review and may validly be `NO_CHANGE_INSUFFICIENT_EVIDENCE` even when the prior observation was `TEST_REQUIRED`. `NO_CHANGE_INSUFFICIENT_EVIDENCE` stays pending and never frees a queue slot. Before ordinary routing, inspect `trader-support-bot`'s `guardian_tuning_acquisition`: report normalized pending plus `current_reviewed_count`, `current_unreviewed_count`, and per-work-order entry/pre-entry/resolved/complete counts. `WAITING_FOR_ENTRIES` and `WAITING_FOR_RESOLUTION` mean keep collecting the same first-N forward cohort; `COLLECTING_WITH_SIGNAL_DEFECT` / `ACQUISITION_SOURCE_DEFECT` require source repair; only `READY_FOR_GPT_REVIEW_UPGRADE` asks GPT to reconsider an exact monotonic `TEST_REQUIRED` upgrade. Replay/backtest is hypothesis context only and must not advance these counts. Do not claim the backlog was processed unless `current_unreviewed_count=0` or every failed writer result is named for retry. An invalid revision/counter/identity/safety boundary means queue counts are unknown, never zero, and requires `REPAIR_GUARDIAN_TUNING_QUEUE_INTEGRITY` before any cleared-backlog claim.
- Before other tuning work, run `tools/guardian_tuning_override_reconcile.py`. A staged override without matching revision-4 terminal evidence remains dormant and blocks that lane's live send; reconciliation may confirm only an exact `CONSUMED` + `ACCEPTED_IMPROVEMENT` work-order/experiment/evidence match.
- Then run `tools/guardian_tuning_post_activation_monitor.py`. It fixes the first 20 canonical attributed entries whose ledger row and raw OANDA entry time are both strictly after the immutable activation-ledger anchor, waits for those same 20 to resolve, and revalidates raw broker entry/close truth plus both the sealed ledger prefix and the current full-ledger first-20 truth before committing `KEEP` for a positive normalized metric or fail-closed `QUARANTINE` for a non-positive metric. Unrelated later rows may append, but a late earlier entry or financing/outcome correction invalidates the old seal and requires a new one. Once entry 20 exists, `RiskEngine` blocks that exact lane with `GUARDIAN_TUNING_POST_ACTIVATION_MONITOR_PENDING` even if this worker was skipped. A same-lane successor may follow only a prior `KEEP` whose activation, terminal, monitor, and current-ledger provenance still validates; unmonitored/quarantined/stale state cannot reset the first-20 boundary. Retry reuses an already-valid content-addressed monitor artifact. Deep override validation is cached only within one gateway phase and is cleared for a fresh final pre-POST read; it is never a process-persistent shortcut. Quarantine is not an automatic relaxation to the recorded previous value.
- Version 1 accepts only a `TEST_REQUIRED` `forecast_confidence_floor` tightening. The review must precommit one exact five-part `lane_id` in `desk:pair:side:method:vehicle` form and the actual active floor as `current_value` before any qualifying entry; unsupported score/lookback/weight/execution parameters stay pending as `NO_CHANGE_INSUFFICIENT_EVIDENCE`. Past trades may suggest the hypothesis but cannot prove it.
- Run `tools/guardian_tuning_cohort_builder.py --work-order-id ... --expected-observation-id ... --lane-id ...`; the CLI lane must equal the reviewed exact five-part lane. At preparation, its SQLite-ledger and append-only thesis/forecast source tips must match the current canonical source tips. The gateway entry-thesis writer must already have recorded the exact OANDA order id, canonical five-part lane, forecast timestamp/cycle, and broker fill timestamp; a local recorder timestamp or hand-filled link is invalid. `FAILED_ACCEPTANCE` also requires the side-matched explicit M5 predicate plus its reclaimed boundary; the method label alone is not evidence. The builder fixes the first 20 canonical attributed entries for that lane opened strictly after review, using canonical gateway-to-fill attribution. If fewer than 20 exist, or any of those first 20 remains unresolved, wait; later resolved entries cannot substitute for an earlier unresolved entry. The builder derives the cutoff, reads one canonical SQLite snapshot, verifies every forecast was observed strictly before entry through append-only thesis/forecast logs, includes losing rows, and normalizes post-financing JPY by entry units. Later source appends are allowed only when the frozen prefixes revalidate. There is no `--signal-evidence` input and no hand-authored outcome path.
- Run `tools/guardian_tuning_evidence_builder.py init-run ...` and then `run` with the exact work-order/observation/experiment IDs. The threshold is fixed at zero with strict `improvement > 0`; baseline metrics use all 20 actually executed trades in the frozen forward cohort, while the candidate alone applies the proposed hard floor. Acceptance also requires positive candidate expectancy and at least 80% retention against that complete baseline. The evaluator copy is hash/version provenance only; trusted `src` code recomputes the result and data-directory Python is never executed. If preparation/run fails, close it with `abort`; a new review needs a new post-review forward cohort.
- Transition only with `tools/guardian_tuning_work_order_lifecycle.py --work-order-id ... --expected-observation-id ... --status CONSUMED|SUPERSEDED --consumed-by qr-trader-hourly --experiment-id ... --experiment-result ... --experiment-evidence-ref 'data/...#sha256=<64hex>'`. The shared writer revalidates canonical source prefixes, hashes, metrics, timing, frequency, monotonic experiment identity, and no-repeat identity, captures the activation-ledger anchor, then stages→terminal-commits→confirms an accepted exact-lane tightening. `RiskEngine` blocks malformed, confirmation-pending, post-activation-monitor-pending, or quarantined state and enforces confirmed tightening at live send; rejected/superseded evidence changes no runtime setting and none of this grants order permission.
- `data/guardian_receipt_consumption.json` is the qr-trader durable acknowledgement record for watchdog guardian receipt issues. For each issue, write `issue_code`, `receipt_event_id`, `receipt_action`, `receipt_lifecycle`, `consumed_by_trader`, `classification`, `reason`, `normal_routing_allowed`, `generated_by`, and `generated_at_utc`. Allowed classifications are `CONSUMED`, `EXPIRED_ACKNOWLEDGED`, `STALE_ACKNOWLEDGED`, `REJECTED_ACKNOWLEDGED`, `HISTORICAL_ONLY`, and `NEEDS_OPERATOR_REVIEW`; expired unconsumed `REDUCE`, `HARVEST`, or `CANCEL_PENDING` must remain `NEEDS_OPERATOR_REVIEW` until valid operator review clears them or a previously fresh-cleared review is durably consumed with `operator_review_status=OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT`. `data/guardian_receipt_operator_review.json` rows classify reviewed receipts as `OPERATOR_ACKNOWLEDGED_HISTORICAL`, `OPERATOR_CONFIRMED_NO_ACTION`, `OPERATOR_CONFIRMED_MANUAL_OWNED`, `OPERATOR_REQUESTS_KEEP_BLOCKED`, or `OPERATOR_REQUESTS_FRESH_REVIEW`, include `no_live_side_effects=true`, and expire explicitly. Normal new-entry routing is blocked while watchdog has unclassified `GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER`, consumption has `normal_routing_allowed=false`, review is missing/stale and was never previously cleared, review requests keep-blocked/fresh-review, or any separate current P0/P1 watchdog issue remains. Initial clearance needs a fresh explicit operator decision plus broker-truth proof that no active emergency remains for the same event; later cycles may preserve that historical acknowledgement after the review handoff TTL expires only if current broker truth still clears the same reviewed event. Existing exposure may still be managed/protected, pending cleanup may proceed through its approved path, and an operator-requested fresh review may be written; ordinary `TRADE` / `ADD` / `campaign_exposure_recovery` stays blocked unless the durable reviewed-historical consumption row clears the receipt. The verifier, `RiskEngine.validate(..., for_live_send=True)`, and `LiveOrderGateway` must all surface `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` for review-required receipts that lack durable clearance instead of letting ordinary fresh entries proceed.
- Legacy operator-review artifacts that predate row-level `clearance_status` may still be durably consumed only when top-level review status says the historical receipt was cleared, the matching row has `normal_routing_allowed=true` and `no_live_side_effects=true`, timestamps are valid, current broker truth still clears the same event, and no separate current P0/P1 watchdog issue remains. Do not use this to clear missing review, active current guardian events, keep-blocked/fresh-review decisions, or broker-truth-uncleared exposure.
- Handle guardian wake runtime failures before normal entries: read the selected event, `docs/guardian_event_report.md`, `docs/guardian_action_review.md`, parse/preflight diagnostics, Codex binary path/version, requested model, raw stdout/stderr excerpts, Codex output-last-message / stdout / session JSONL source, and whether the dispatcher queued the event for the active trader; classify the failure (`CODEX_MODEL_UNSUPPORTED`, `CODEX_CLI_VERSION_UNSUPPORTED`, `CODEX_USAGE_LIMIT`, `CODEX_EMPTY_LAST_MESSAGE`, `CODEX_NO_ASSISTANT_MESSAGE`, `CODEX_NO_JSON_RECEIPT`, `CODEX_TIMEOUT`, `CODEX_AUTH_OR_SANDBOX_FAILURE`, or `SCHEMA_INVALID`), then either write a fresh valid receipt through the normal trader flow or record why the wake is stale/rejected. `CODEX_USAGE_LIMIT` is capacity failure, not malformed receipt evidence: keep the exact event in the durable pending queue, do not spend a schema-repair retry, and normally retry only after the quota backoff while the hourly trader records the blocked tuning obligation. A later accepted guardian receipt proves capacity recovery only for quota failures older than that receipt; those older quota-only backoffs may be released immediately, while schema/parse/runtime backoffs remain unchanged. The dispatcher prompt and any repair retry must contain only the authoritative selected event/pair; never ask Codex to choose among the full pending queue or unrelated trigger-contract/report context.
- A guardian wake receipt is executable only when its `event_id`, `pair`, compatible side/direction, any included `dedupe_key`, `receipt_status=ACCEPTED`, `receipt_lifecycle=ACTIVE`, and `expires_at_utc` match the selected event and current time. `RECEIPT_EVENT_MISMATCH` means the new wake output was rejected; resolve the queued event from `docs/guardian_action_review.md` before ordinary entries. Do not treat a later `NO_WAKE` or `SUPPRESSED` dispatcher pass as invalidating a prior accepted HOLD / NO_ACTION receipt.
- Accepted HOLD / NO_ACTION guardian receipts are resolved review evidence. They are preserved until `expires_at_utc`, superseded by a newer accepted receipt, or consumed by `guardian-action-cycle`; they must not trigger live orders, cancels, or closes.
- Queued `WAKE_PARSE_FAILURE` / `queued_for_active_trader=true` work must be resolved before ordinary new entries. Do not proceed to new-risk routing while the guardian action review names an unclassified parse failure for a current event.
- Resolve queued guardian wake actions before ordinary new entries: `queued_for_active_trader=true` means the dispatcher yielded to the active trader, so the trader must review the event/report/receipt first and either consume the receipt through the normal verifier/gateway path, recognize that `guardian-action-cycle` already executed/rejected it, or write the exact reason it is stale/rejected.
- Refresh `data/guardian_trigger_contract.json` on every hourly trader cycle after market read and position housekeeping. For each open position, especially manual/unknown USD_JPY exposure, map by trade_id and keep pair, side, owner, units, average entry, thesis, thesis_state, non-empty harvest / invalidation / emergency triggers, no-add triggers, next_review_reason, and a non-expired next_review_deadline_utc current. Every nested trigger for an open-position entry must either omit duplicated parent fields or match the parent `trade_id`, pair, side, owner, units, average entry, and thesis exactly enough for `validate_guardian_trigger_contract`; a parent such as trade_id `472931` must never carry trigger metadata from `472909`. Candidate/watch-only entries may omit invented triggers when no market thesis exists.
- A guardian trigger contract whose deadline expires while exposure exists is `CONTRACT_STALE`; refresh it before normal new-entry routing instead of letting the router infer triggers from prose.
- Target-path entry sends require `QR_TARGET_PATH_LIVE_ENABLED=1` in addition to `QR_LIVE_ENABLED=1`; default is dry-run/stage/LIVE-LEARNING receipt only.
- OANDA position changes go only through `PositionProtectionGateway`.
- Direct `OandaExecutionClient.close_trade()` is blocked; live market closes must use the provenance-aware gateway/partial-close paths and leave a position-execution receipt.
- Do not print secrets.
- Do not call `QR_OPENAI_API_KEY`, `OPENAI_API_KEY`, or any model API path from QuantRabbit code.
- Do not invent JPY caps, pip distances, reward/risk multipliers, stale defaults, or extra risk gates.
- Missing required evidence is a blocker, not a value to guess.
- Rolling target accounting uses `ROLLING_30D_4X` as the top KPI: 30 calendar days to 4x. Reports must separate `current_equity_raw` from `funding_adjusted_equity`; capital flows are not trading P/L. +5% is a pace marker / review trigger / protection milestone, and +10% is extension-only after an explicit favorable-market gate.
- One final decision receipt selects action; specialist and strategy prompts are read-only observation.
- A blocked, rejected, monitor-only, or no-trade cycle must not be followed by a workaround send.
- Do not stop solely because a decision receipt was written recently or stale local state disagrees with refreshed broker truth. Use `trader-prompt-route`: unconsumed receipts go to verify; rejected, consumed, or broker-stale receipts go back to fresh decision work.

## Branches

| Branch | Read |
|---|---|
| Refresh broker truth and market context | `docs/trader_prompts/10_precheck_refresh.md` |
| Read the current market packet | `docs/trader_prompts/20_market_packet.md` |
| Flat / layerable account entry decision | `docs/trader_prompts/30_entry_decision.md` |
| Open exposure, pending order, protection decision | `docs/trader_prompts/35_position_management.md` |
| Receipt verification and gateway execution | `docs/trader_prompts/40_verify_execute.md` |
| Post-trade learning, missed-edge, gap work | `docs/trader_prompts/50_learning_gap.md` |
| Shared decision JSON schema | `docs/trader_prompts/90_decision_receipt_schema.md` |

## Runtime Skeleton

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"

# 0. Export SL-free strategy env vars so generate-intents and risk validation
#    pick up market-derived geometry, suppressed SL repair, disabled
#    broker-side SL/trailing, advisory REVIEW_EXIT, and the expanded
#    portfolio cap. These mirror `scripts/run-autotrade-live.sh` defaults so
#    direct CLI invocations and the wrapper produce identical receipts.
export QR_GEOMETRY_ATR_MULT="${QR_GEOMETRY_ATR_MULT:-5.0}"
export QR_GEOMETRY_SPREAD_FLOOR_MULT="${QR_GEOMETRY_SPREAD_FLOOR_MULT:-12.0}"
export QR_TRADER_DISABLE_SL_REPAIR="${QR_TRADER_DISABLE_SL_REPAIR:-1}"
export QR_MAX_PORTFOLIO_POSITIONS="${QR_MAX_PORTFOLIO_POSITIONS:-10}"
# NAV-pct sizing: each new position locks % of current NAV as margin so
# unit count auto-scales with equity (feedback_use_nav_percent.md). 30%
# per position lands ≈10000u for EUR_USD at NAV 227k — three concurrent
# positions reach ~90% margin utilization, just inside the 92% cap.
# Mirrors scripts/run-autotrade-live.sh so direct CLI invocations and the
# wrapper produce equivalent sizing.
export QR_TRADER_POSITION_NAV_PCT="${QR_TRADER_POSITION_NAV_PCT:-30}"
# Deterministic loss-side REVIEW_EXIT is advisory by default in SL-free live
# mode. Loss closes must pass the gpt_trader close discipline and operator
# token; profit-only TAKE_PROFIT_MARKET remains a separate harvest path.
export QR_DISABLE_AUTO_CLOSE="${QR_DISABLE_AUTO_CLOSE:-1}"
# Broker-side SL/trailing are opt-in only. The live default keeps NEW
# entries SL-free because widened broker SLs were still harvested by
# thin-session noise on 2026-05-13.
export QR_NEW_ENTRY_INITIAL_SL="${QR_NEW_ENTRY_INITIAL_SL:-0}"
export QR_DISABLE_TRAILING_SL="${QR_DISABLE_TRAILING_SL:-1}"
# Disaster stop (2026-06-11, operator-approved 「SLの件もやっていい」).
# Every NEW entry carries a broker-side CATASTROPHE stop at
# H4 ATR × QR_DISASTER_SL_H4_ATR_MULT (2.5) × session widening —
# 60-120+ pips on majors, far beyond the noise band that hunted the
# 2026-05-13 stops. It is decoupled from intent.sl: sizing, reward/risk,
# and risk validation are unchanged; it never trails; existing positions
# are never retro-fitted. Its job is to cap the give-up-close tail and
# survive a flash move / intervention inside the full-trader blind window.
# Frequent guardian probe/router monitoring covers state-change risk between
# hourly trader cycles.
export QR_DISASTER_SL="${QR_DISASTER_SL:-1}"
export QR_DISASTER_SL_H4_ATR_MULT="${QR_DISASTER_SL_H4_ATR_MULT:-2.5}"
# Fresh entries need both executable forecast context and auditable telemetry.
# If forecast_history, projection_ledger, or execution_ledger sync is missing,
# generate-intents may diagnose the lane but must not emit LIVE_READY.
export QR_REQUIRE_FORECAST_FOR_LIVE="${QR_REQUIRE_FORECAST_FOR_LIVE:-1}"
export QR_REQUIRE_TELEMETRY_FOR_LIVE="${QR_REQUIRE_TELEMETRY_FOR_LIVE:-1}"
# Predictive SCOUT is a double gate. This env flag alone grants nothing;
# config/predictive_scout_policy.json, canonical digest, current bucket,
# LIMIT/GTD/attached TP+SL, current-NAV risk sizing to positive integer units
# (including 1-999u), atomic signal claim, concurrency and loss quarantine must pass.
export QR_PREDICTIVE_SCOUT_LIVE_ENABLED="${QR_PREDICTIVE_SCOUT_LIVE_ENABLED:-1}"
# Controlled target-path live is an extra gate. Leave it off unless the
# operator intentionally wants a LIVE-LEARNING target-path send through
# LiveOrderGateway after exact pretrade/spread/pricing/fill proofs pass.
export QR_TARGET_PATH_LIVE_ENABLED="${QR_TARGET_PATH_LIVE_ENABLED:-0}"

# Session-start read-only target block. This does not stage, send, cancel, or
# close orders. It persists the first-seen UTC day-start NAV under
# logs/day_start_nav/ and prints the rolling 30-calendar-day 4x KPI plus the
# +5% pace marker / extension +10% operating mode. It also prints the required
# FULL_TRADER pace board, attack stack, and 10% extension gate.
# Do not leave those fields blank in the working decision or end report: report
# rolling_30d_start_equity, current_equity_raw, capital_flows_30d,
# funding_adjusted_equity, rolling_30d_multiplier_raw,
# rolling_30d_multiplier_funding_adjusted, remaining_to_4x_raw,
# remaining_to_4x_funding_adjusted, required_calendar_daily_return_raw,
# required_active_day_return_raw, required_calendar_daily_return_funding_adjusted,
# required_active_day_return_funding_adjusted, required_calendar_daily_return,
# required_active_day_return, performance_basis, sizing_basis, and pace_state
# every cycle. Performance / 30d 4x uses funding_adjusted_equity; risk / margin
# / sizing uses current broker NAV / current_equity_raw. The legacy
# required_calendar_daily_return / required_active_day_return fields are
# funding-adjusted aliases. Do not claim 30d 4x pace from raw NAV when
# capital_flows_30d explains the increase. While remaining_to_5pct is above zero, fill a
# concrete A/S Path A / HERO route or write the exact blocker and next trigger.
# B/C churn is not a substitute for the HERO path, and a single distant pending
# order is not enough. "Trigger not printed yet" is an arm condition for a
# LIMIT/STOP thesis, not a dead thesis. The selected path must map to the
# ATTACK STACK. Before any fresh target-path order, run dry-run sizing with
# tools/position_sizing.py or tools/place_trader_order.py. If you need a
# handoff artifact, use tools/place_trader_order.py --gateway-intent-output;
# live sends still go only through LiveOrderGateway and require
# QR_TARGET_PATH_LIVE_ENABLED=1.
python3 tools/session_data.py

Required trader block:

```markdown
## MARKET READ FIRST
Naked read:
- Currency bought:
- Currency sold:
- Cleanest pair expression:
- Is this pair the cleanest currency theme: YES / NO / UNKNOWN
- 24h location: LOWER / MIDDLE / UPPER / UNKNOWN
- H1/H4 alignment:
- Tape state: TREND / RANGE / SQUEEZE / FADE / ROTATION
- Known winning trade-shape match: MATCH / PARTIAL / NO_MATCH / UNKNOWN
- Proposed building style allowed: YES / NO / UNKNOWN
- Thesis state: ALIVE / WOUNDED / INVALIDATED / EMERGENCY / UNKNOWN
- What price is trying to do now:

Next 30m prediction:
- Pair:
- Direction:
- Expected path:
- Target zone:
- Invalidation:

Next 2h prediction:
- Pair:
- Direction:
- Expected path:
- Target zone:
- Invalidation:

Best trade if forced:
- Pair:
- Direction:
- Vehicle: MARKET / LIMIT / STOP
- Entry:
- TP:
- SL:
- Why this pays:

Execution filters after the read:
- LIVE_READY lanes:
- Exact blockers:
- Negative expectancy / capture economics context:
- Final action:

## 5% PACE BOARD
Remaining to +5%:
Role: pace marker / review trigger / protection milestone, not forced churn.

Path A / HERO:
Pair / side / vehicle:
Expected pips:
Suggested units:
Expected contribution:
Entry:
TP:
SL:
Status: live / armable / blocked
Exact blocker if blocked:

Path B / SECOND SHOT:
Pair / side / vehicle:
Expected pips:
Suggested units:
Expected contribution:
Entry:
TP:
SL:
Status:
Exact blocker if blocked:

Path C / NO HONEST PATH:
Exact blocker:
Next trigger:
Shelf-life:

## ATTACK STACK
Hero thesis:
Why this thesis can still reach +5% today:

NOW:
Pair / side / vehicle:
Entry:
TP:
SL:
Units:
Why now:

RELOAD:
Pair / side / vehicle:
Entry:
TP:
SL:
Units:
Why this is better price, not hesitation:

SECOND SHOT:
Pair / side / vehicle:
Entry:
TP:
SL:
Units:
Why this is same theme, different expression:

If any slot is empty:
Exact blocker:
Next trigger:
Shelf-life:

## USER ALPHA CONTINUATION
Latest USER_ALPHA / OPERATOR_ALPHA:
Pair / side:
Entry:
TP:
Realized P/L:
MFE if available:
Time to TP:
Thesis if available:
Discovered by: system / operator / user
System TP-managed: YES / NO / UNKNOWN

What the user saw:
What the system missed:
Is thesis still alive: YES / NO / UNKNOWN
RELOAD candidate:
SECOND SHOT candidate:
5% PACE BOARD mapping:
Exact blocker if no continuation:
Next trigger:
```

Path rules:
- Start with `MARKET READ FIRST` every cycle before citing LIVE_READY count, blocker codes, negative expectancy, margin, target pressure, or pending-order repair.
- A blocker is not a market read. `LIVE_READY=0` is not a market read. Negative expectancy is not a market read. Predict price first, then filter execution.
- The naked read must classify cleanest theme expression, 24h location, H1/H4 alignment, known winning trade-shape match, proposed building style, thesis state, and SL/noise context before any blocker prose.
- A fresh `TRADE` must not contradict the naked read: SHORT market read with LONG target/invalidation geometry, LONG market read with SHORT geometry, or selected side vs market-read direction conflict requires rejection/request-evidence unless the final action explicitly resolves the contradiction.
- A blocked but correct read is discovery success / execution miss. A wrong read that passes filters is market-read failure.
- Final `TRADE` / `WAIT` text must reference the next 30m or next 2h prediction from `MARKET READ FIRST`.
- Codex-authored directional 30m/2h reads require a current broker quote plus numeric target and invalidation on the correct sides. `RANGE` requires numeric lower/upper target rails bracketing the quote and strictly wider lower/upper invalidation rails. `best_trade_if_forced` always requires pair, LONG/SHORT, and numeric entry/TP/SL geometry even when the final action is non-trade.
- Always write the strongest counterargument. Use `ACCEPT_BASELINE` only when the current numeric forecast supports the exact baseline lane; use `VETO_WAIT` / `VETO_REQUEST_EVIDENCE` when the forecast-backed counterargument makes that entry wrong. A veto reason such as session preference alone is not enough.
- For `ACCEPT_BASELINE` `TRADE`, author `capital_allocation` from the content-addressed board: cite direction-specific economic hit rate/sample count, ordinary hit rate/sample count, timeout rate, exact-vehicle all-exit net/expectancy/Wilson-stressed expectancy and sample count, any narrower TP proof, reward/risk, maximum loss, and the strongest loss/leak counter-evidence in the rationale. Choose only an advertised exact `0.5`, `0.75`, or `1.0`; `selected_units` is integer floor of `base_units × size_multiple`, and a sub-unit choice is never advertised. Never turn broad, negative, or unknown edge into size by reference to the monthly target. Missing or invalid allocation must remain blocked, not silently become `1.0`.
- Treat that numeric allocation as a prediction hypothesis, not durable order permission. Ordinary schema-v2 allocations execute only as MARKET and must survive the gateway's fresh quote/NAV/conversion RiskEngine remeasure, directionally favorable drift rule, economic-Wilson EV, quarter-Kelly floor-unit ceiling, and complete OANDA S5 no-touch path from forecast emission through the current quote. A target/TP/invalidation touch followed by a retrace is still consumed. The gateway fixes a side-aware worst-fill `priceBound` into the reservation and re-proves EV/Kelly/NAV/margin/basket/attached-stop truth at that bound after reservation, followed by an unchanged-transaction and non-worsening NAV/margin account fence. Do not remove or loosen these proofs to increase trade frequency; obtain a fresh forecast instead. Broker-carried entry/TP/SL/disaster prices must already be on the instrument tick grid.
- Read and cite the board's dynamic `execution_cost_floor`; never reason with `additional_cost_jpy=0` or a hand-entered spread/slippage constant. `QR_EXACT_VEHICLE_ALLOCATION_SURFACE_V2` binds strict MARKET entry p95, authoritative exact-order/trade TP/SL exit p95, and adverse-only unit-scaled financing (maximum global/exact Wilson-95 upper stress; credits do not offset debits). Each transport/global cohort is a latest-anchored 90-day window with a latest-age limit of 90 days and the required sample floor; exact financing is key-local on the same rolling basis. Missing/thin/stale/mismatched evidence means `NO_TRADE`/refresh, not zero cost. At the fresh quote the ordinary proof subtracts entry + protected-exit + financing stress from EV and adds the same total to downside/Kelly. At `priceBound`, entry slippage is embedded in the adverse fill and must not be charged twice, but the bound must cover at least entry p95; exit + financing outcome cost remains included exactly once in EV and every loss/portfolio/attached-stop cap. The receipt, early proof, reserved request, and final proof must carry the same cost SHA and exact pair/side/`market_context.method`/MARKET scope. If `metadata.method` exists it must match the market context. A row from another edge lane can still stale this proof when it enters the global cost cohort, so “unrelated vehicle” is non-material only when the cost digest also stays unchanged.
- The execution-ledger evidence source is a WAL-aware semantic snapshot, not a raw `.db` file hash. It binds the selected lane's exact pair/side/method/entry-vehicle all-exit and pure-TP rows plus the global execution-cost surface, including arithmetic and unresolved realized cash. Non-zero audited `DAILY_FINANCING` on a still-open attributed trade is unresolved cash; exact-zero and manual components do not become system edge, while account-total zero still preserves each non-zero system component. Missing raw financing transactions or broken gateway-fill trade identity make the surface unreadable. A WAL checkpoint alone is non-material. An unrelated edge-only row is non-material only if it changes neither the selected exact rows nor a global cost cohort/digest; a selected close/reduction, changed selected proof, or cost-cohort change stales the read. The gateway repeats the same-basis edge and cost proof after ledger/broker reconciliation immediately before POST; an all-exit authorization cannot fall back to TP, and TP is suppressed by any contradictory loss, unresolved lifecycle, or invalid arithmetic.
- Never hand-edit execution fields into the overlay. Unknown top-level overlay fields, stale baseline/evidence SHA, stale model timestamp, missing latest resolved prediction review, or a changed final envelope must fail closed and require a fresh baseline/AI pass.
- The final `TRADE` must keep `selected_lane_id` and the sole `selected_lane_ids` item identical to the exact baseline lane. If that lane is stale, missing, ineligible, or no longer fits, write a non-entry outcome and obtain a fresh receipt; do not use another current lane as a replacement.
- `campaign_exposure_recovery` and deterministic recovery are fresh-entry routes, not WAIT overrides. They require a fresh accepted GPT `TRADE`/`ADD` receipt naming the lane; stale accepted `WAIT` / `REQUEST_EVIDENCE`, `gpt_allowed=false`, fresh-receipt-required errors, unresolved guardian receipt issues, missing/stale guardian receipt operator review, watchdog `normal_routing_allowed=false`, and missing market-read confirmation block the send.
- `OANDA_CAMPAIGN_FIREPOWER_RELAXED` is capacity only. It cannot relax `NEGATIVE_EXPECTANCY`, stale GPT/non-TRADE receipts, accepted WAIT/REQUEST_EVIDENCE, missing fresh TRADE receipt, guardian hard blockers, or no current market-read confirmation.
- Under +5%, trader must name an A/S path, a +10% extension setup gate candidate, or exact blocker.
- A +5% miss must not force B/C churn.
- B/C trades cannot be the +5% pace path.
- One distant pending order is not enough.
- "Trigger not printed yet" is an arm condition for LIMIT/STOP, not a dead thesis.
- The path must map to ATTACK STACK.
- `tools/place_trader_order.py` is dry-run only. It may emit a gateway intent, but it must not send.
- Live target-path receipts must include daily target mode, remaining-to-5%, path role, attack-stack slot, grade, suggested/final units, risk, target contribution, and `LIVE_LEARNING` mode.
- A profitable manual/operator-discovered winner is `USER_ALPHA` / `OPERATOR_ALPHA`, not system-discovered bot edge. If `data/trader_overrides.json` carries active `user_alpha_continuation`, the trader must answer thesis-alive / RELOAD / SECOND SHOT / exact blocker and cite `user_alpha:continuation`; generic `NEGATIVE_EXPECTANCY` cannot erase it.
- Stale trader-owned pending entries must resolve as exactly one of `CANCEL_PENDING`, `REPRICE`, `REPLACE_WITH_NEW_INTENT`, or `KEEP_WITH_EXACT_REASON`. A replacement `TRADE` may include current pending `cancel_order_ids`; duplicate parent-lane occupancy must not block cancel/replace of the very pending id being replaced.

## 10% EXTENSION GATE
Default: NO
YES only if:
- Progress is strong, ideally +3.5%+, or protected S/A winner can carry past +5%.
- Hero thesis still paying.
- 3+ pairs confirm same currency theme, or hero pair has clean trend/band-walk.
- Spread stable.
- No major whipsaw event in next 30m.
- Last A/S trade green, protected, or structurally alive.
- Real reload/second-shot level exists, not chase.

Gate effect:
- EXTEND mode requires A/S grade risk.
- After +5%, Extension Gate NO blocks fresh B risk.
- Before any fresh target-path order, run dry-run sizing with tools/position_sizing.py or tools/place_trader_order.py.

# 1. Route to the right prompt branch
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-prompt-route

# 2. Refresh evidence when routed there — ONE consolidated command.
#
# `cycle-refresh` runs the full refresh step list (broker-snapshot →
# daily-target-state → execution-ledger-sync → import-legacy → mine-strategy →
# pair-charts → cross-asset/context/flow/strength/levels/calendar/COT/skew →
# market-context-matrix → news-snapshot → mine-market-stories → news-health --strict →
# daily-review → tp-rebalance → verify-projections → broker-snapshot →
# daily-target-state → capture-economics → qr-trader-run-watchdog →
# guardian-receipt-consumption → generate-intents --reuse-market-artifacts →
# optimize-coverage → ai-attack-advice →
# learning/execution-timing/manual-market-context/operator-precedent/verification audits →
# generate-predictive-limits → position sidecars → guardian-trigger-contract →
# guardian-event-router → qr-trader-run-watchdog → guardian-receipt-consumption →
# profit-capture-bot → memory-health → self-improvement-audit → profitability-acceptance →
# trader-support-bot → as-live-ready-evidence-loop → as-4x-proof-path →
# trader-repair-orchestrator → trader-goal-loop-orchestrator → active-trader-contract →
# active-opportunity-board → non-eurusd-proof-lane-mapper →
# non-eurusd-live-grade-frontier → active-trader-contract →
# entry-frequency-recovery → active-trader-contract →
# forecast-pattern-refresh → active-trader-contract →
# range-rail-geometry-repair → active-trader-contract →
# guardian-trigger-contract → guardian-event-router →
# operator-review-report → trader-support-bot →
# trader-repair-orchestrator → trader-goal-loop-orchestrator) in one
# process, in the same order and with the same arguments the per-step
# skeleton used (`cli._cycle_refresh_steps` is the canonical list), then
# prints ONE compact digest including the re-routed prompt branch.
#
# Token discipline (2026-06-10): the per-step skeleton burned ~3M tokens per
# former 20-minute cycle (one shell turn per command × full-context resend) and
# exhausted the scheduler's credits on 2026-06-09, stopping live trading.
# Read the digest, then drill into `data/order_intents.json`,
# `data/pair_charts.json`, `data/market_context_matrix.json` etc. with
# TARGETED queries (jq / python -c) only where the digest flags something.
# Never cat a multi-megabyte artifact into the conversation.
#
# Long-running commands (2026-06-11; generate-intents timeout widened
# 2026-06-25): `cycle-refresh`, the live wrapper, and `cycle-sidecars` take
# minutes. Invoke them with ONE long wait (shell-tool yield/timeout ≥ 1200000
# ms) instead of the default ~10s yield — 2026-06-11 telemetry showed ~25
# empty polling turns per cycle, each re-sending the whole conversation
# context, keeping the cycle at ~3.9M tokens despite the consolidation. The
# required post-gateway `generate-intents --reuse-market-artifacts` step has a
# 900s bounded timeout because it must finish repricing current order_intents
# before coverage, acceptance, and support can be trusted. One long wait
# removes both empty-poll token spend and partial-stale sidecar reads.
#
# `--daily-risk-pct 10` is forwarded to every daily-target-state step as the
# current live risk-budget argument. Do not read it as the base operating
# profit target: the session target engine treats +5% from UTC day-start NAV
# as the base target and +10% only as a favorable-market extension gate.
#
# News has a cycle-local freshness floor: `news-snapshot` refreshes public RSS
# artifacts before `mine-market-stories` / `news-health`. The richer curated
# `qr-news-digest` routine may still overwrite or augment these artifacts, but
# a stale external digest must not leave the trader blind to current news.
#
# Failed required steps abort the remaining refresh and exit 2; the digest
# lists `steps_failed` with stderr tails. Optional-step failures (e.g.
# news-health --strict during a stale-news window) appear in `steps_failed`
# but do not stop evidence generation — treat them as named blockers in the
# decision receipt exactly as before.
# `execution-timing-audit` is optional and runs the month-scale TP-progress
# replay (`--lookback-hours 744 --post-close-hours 6 --max-events 80`) before
# generate-intents, self-improvement, and profitability-acceptance. It must not
# be shortened back to the module-default 168h window, and it must stay before
# intent generation because TP_HARVEST_REPAIR exceptions read residual replay
# groups before exposing a repair lane.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10

# The refresh branch is not an end state: it must produce one current receipt
# and then proceed through verification + gateway. Ending the cycle right
# after `cycle-refresh` leaves fresh evidence unused and is treated as
# incomplete. The digest's `route` field is the re-route result; only run
# `trader-prompt-route` again if you changed an artifact after the digest.
# `memory-health` and `profitability-acceptance` must audit the same
# `capture_economics` packet used by `generate-intents`. If capture-economics
# is refreshed after order_intents, or memory-health does not record the current
# capture timestamp, route back to refresh before entry/verify work.
#
# Position close sidecars inside the digest are read-only prediction/thesis
# evidence. Read `protection_sidecars.position_close_recommendations[]`
# before deciding: `blocks_non_close_actions=true` means close-first work;
# `blocks_non_close_actions=false` means soft advisory only and must not
# produce a CLOSE receipt from the entry branch. Fresh thesis_evolution
# BROKEN / RECOMMEND_CLOSE is hard only when its rationale contains price
# invalidation plus technical confirmation; structural authority comes from
# the timestamped H4 / position-management paths, not sidecar prose.
# structural position_management REVIEW_EXIT and position_thesis
# invalidation-hit or structural-break evidence with multi-TF confirmation are
# also hard standing loss-cut authorization only when they do not
# conflict with fresh same-direction HOLD/EXTEND sidecars. If thesis_evolution
# / position_thesis / forecast_persistence still support the open side, treat
# the issue as HOLD/reprice/TP rebalance unless explicit Gate B is present.
# Adverse-entry-buffer-only position_thesis evidence is soft and still needs
# explicit Gate B before CLOSE; without Gate B, soft sidecars are advisory for
# non-CLOSE actions and must not block separate current LIVE_READY entries on
# other pairs or horizons.

# 3. Prepare the deterministic envelope, author the AI market read, then merge
# `trader-draft-decision` first composes one current deterministic baseline from
# the same broker/market/news packet that the verifier will audit. It also
# writes a SHA-256-bound evidence packet. The command is read-only except for
# the baseline/packet/report files: it does not call model APIs, send orders,
# cancel orders, close positions, or change launchd state. It selects current
# LIVE_READY lanes from order_intents / ai_attack_advice when clean. When
# LIVE_READY=0, it still reads
# active_trader_contract / active_opportunity_board /
# non_eurusd_live_grade_frontier / range_rail_geometry_repair and uses the
# current active top lane as the market-read/evidence-acquisition target instead
# of falling back to the first stale order intent; that active lane is context,
# not live permission. A fresh close sidecar emits a deterministic CLOSE
# baseline only when it binds one exact current trader-owned trade, is marked
# `blocks_non_close_actions=true`, and has hard standing or explicit Gate B.
# Multiple close targets, manual positions, soft/no-token recommendations, and
# same-direction HOLD conflicts remain non-CLOSE. Other named blockers such as
# news-health, projection, exposure, active-path blockers, or self-improvement
# gates emit WAIT / REQUEST_EVIDENCE. Before any normal new-entry routing, it also
# writes `data/guardian_receipt_consumption.json` /
# `docs/guardian_receipt_consumption_report.md` from watchdog guardian receipt
# issues and reads `data/guardian_receipt_operator_review.json` /
# `docs/guardian_receipt_operator_review_report.md`; unresolved or stale
# `NEEDS_OPERATOR_REVIEW` classifications block ordinary TRADE / ADD.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-draft-decision \
  --snapshot data/broker_snapshot.json \
  --output data/trader_decision_baseline.json \
  --market-read-evidence-packet data/market_read_evidence_packet.json
#
# Now act as the named discretionary author. Read the complete evidence packet,
# current source artifacts, and latest genuinely resolved market-read feedback.
# Write exactly `data/codex_market_read_overlay.json` with schema_version=2,
# author_kind=CODEX_MARKET_READ, model=gpt-5.5, reasoning_effort=high, the
# packet/baseline SHA values, authored_at_utc, baseline_disposition, a complete
# numeric MARKET_READ_FIRST, prior-error review, strongest counterargument,
# change summary, veto reason (empty only for ACCEPT_BASELINE), and the exact
# capital_allocation object bound to capital_allocation_board_sha256. Accepted
# TRADE uses only ALLOCATE + the baseline lane + 0.5/0.75/1.0 + integer-floor
# selected_units; veto/non-entry/CLOSE uses NO_TRADE + null lane + zero units.
# The overlay contains no action, alternate lane, cancel/close ids, stops,
# targets, risk caps, or permission fields.
#
# Merge only through the fail-closed tool. It re-hashes every material evidence
# state, recomputes the stored allocation-board/body digests, reconstructs the
# packet, and proves the final action transition, bounded sizing authorization,
# and execution envelope, then
# atomically publishes the verifier input. The frequently rewritten watchdog
# uses `QR_TRADER_WATCHDOG_SAFETY_STATE_V1`: observation clocks, age counters,
# and log/report timestamps may advance, but health/issues, automation/weekend
# state, material guardian receipts, and the no-write boundary remain bound.
# Any rejection means refresh/re-author; never copy fields by hand around the
# failed merge.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-apply-market-read \
  --baseline data/trader_decision_baseline.json \
  --packet data/market_read_evidence_packet.json \
  --overlay data/codex_market_read_overlay.json \
  --output data/codex_trader_decision_response.json
#
# If broker refresh made the baseline/packet stale, regenerate both and author a
# new overlay. Never reuse the old AI output against new broker truth.
# For TRADE / WAIT / REQUEST_EVIDENCE, include `twenty_minute_plan`.
# The field name is retained for verifier/backward compatibility, but the
# scheduled full-trader cadence is now 60 minutes. Set `horizon_minutes=60`
# and state the primary path, failure path, trigger, invalidation/cancel
# trigger, strongest counterargument, next-cycle check, and packet evidence
# refs. This is a receipt-depth requirement so the next cycle can audit the
# scenario tree; it is not a new market-risk threshold or permission to invent
# blockers.
# A TRADE must cite current chart evidence plus `news:health` and `news:items`
# or `news:current`. If news-health is missing, ERROR/BLOCK, or carries BLOCK
# issues, write a non-TRADE blocker receipt; campaign pressure must not bypass
# stale or unsynced news.
# A deterministic baseline TRADE is expected to report
# DRAFT_REQUIRES_OPERATOR_REVIEW / AI_MARKET_READ_REQUIRED. That status is the
# handoff boundary, not a blocker to writing the overlay. Other draft verifier
# issues remain real blockers. If overlay application or final verification
# rejects, continue only to the gateway maintenance cycle so existing-position
# protection still runs; never invent a deterministic entry workaround.
# If current trader-owned pending entries consume portfolio capacity, either keep
# that pending basket explicitly or name verified trader pending ids in
# cancel_order_ids when replacing them with current MARKET participation.
# If an accepted TRADE receipt later fails the deterministic prefilter match,
# the gateway still cancels verified cancel_order_ids before returning
# GPT_DECISION_NOT_PREFILTERED; it must not send a fresh entry on that receipt.
# If the action is CANCEL_PENDING, list only current trader-owned pending entry
# ids in cancel_order_ids; the gateway cycle cancels verified ids and sends no
# fresh entry in that same cycle.
#
# CLOSE discipline (AGENT_CONTRACT §10, feedback_no_unilateral_close.md):
# Do not emit CLOSE from loss size, margin pressure, fear, or stale prose. A
# CLOSE receipt is required when current machine-checkable evidence satisfies
# Gate A plus the applicable Gate B path:
#   - Gate A: market evidence — pair_charts shows BOS/CHOCH against the
#     position side on M15 or H4, OR `invalidation_price` + `invalidation_tf`
#     in the receipt with broker truth confirming the level has traded, OR a
#     fresh position sidecar generated after the current broker snapshot marks
#     the same trade REVIEW_CLOSE / RECOMMEND_CLOSE, OR a bounded carry-forward
#     `position_management.json` REVIEW_EXIT for the same still-open trade.
#   - Gate B: hard loss-cut standing authorization OR explicit operator
#     authorization. Hard standing authorization is an exact structured H4
#     close-confirmed BOS/CHOCH against side whose timestamp postdates the
#     matching broker-open / entry-thesis anchor, buffered invalidation_price hit with technical
#     confirmation, fresh thesis_evolution BROKEN / RECOMMEND_CLOSE with that
#     invalidation/technical proof in its rationale,
#     structural position_management REVIEW_EXIT, or position_thesis
#     invalidation-hit / structural-break evidence with multi-TF confirmation.
#     M15 close-confirmed BOS/CHOCH is Gate A evidence, but it is not unattended
#     hard Gate B unless post-entry timestamped H4 structure, recorded
#     invalidation, or a hard sidecar also confirms. Pre-entry, missing-time,
#     mismatched, future-dated, or unanchored H4 is also soft. Softer sidecars (adverse-entry-buffer-only or score-only
#     position_thesis REVIEW_CLOSE, non-structural position_management
#     REVIEW_EXIT, forecast/expiry-only thesis_evolution RECOMMEND_CLOSE,
#     forecast_persistence RECOMMEND_CLOSE, or M15-only structure)
#     require `QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell, OR a fresh
#     `data/.operator_close_token` file. The receipt field
#     `operator_close_authorized: true` is advisory audit text only.
# `gpt-trader-decision` writes `close_gate_evidence[]` for every CLOSE trade id,
# accepted or rejected. Read it before sending/staging any loss-side CLOSE:
# it must show the Gate A reason, standing-vs-explicit Gate B state, P0/timing
# audit citations, and any same-direction support conflict. A missing durable
# close-gate evidence row is an audit defect, not proof that the close is safe.
# A TRADE receipt must not list close_trade_ids. When the exact single-trade
# hard/authorized conditions above are present, `trader-draft-decision` must
# already emit the CLOSE baseline with the exact `close_trade_ids` and current
# sidecar/P0/timing evidence refs. The GPT market-read overlay may only ACCEPT
# that baseline; it contains no action or close-id fields and cannot create,
# remove, or change the target. The final verifier rebuilds and byte-compares
# the same handoff for CLOSE, and rejects multiple/duplicate trade ids, entry
# lanes, or cancel ids. If the deterministic baseline is not CLOSE, refresh the
# packet or fail closed instead of hand-editing a CLOSE. After an accepted
# CLOSE, end the autotrade cycle as close-only. Refresh broker truth / intents on the next
# scheduled cycle, and only re-enter on a fresh LIVE_READY lane with a separate
# verified TRADE receipt. The automation must not re-enter in the same outer
# cycle after the close is sent, staged, or already satisfied.
# If the same-direction market stack still supports the open position, this is
# not a CLOSE+re-entry case. Treat it as geometry management: TP rebalance,
# HOLD, profit-side partial, or a separately risk-bounded ADD lane. Loss-side
# CLOSE is for broken thesis evidence, not for refreshing a valid ticket.
# After an accepted CLOSE receipt, the gateway still re-checks the latest
# broker-snapshot quote before sending. Missing close quotes, missing normal
# spread baselines, or spread wider than `NORMAL_SPREAD_PIPS ×
# RiskPolicy.max_spread_multiple` block market CLOSE unless the operator
# explicitly exports `QR_POSITION_CLOSE_SPREAD_OVERRIDE=1`.
# The verifier also rejects CLOSE when the decision packet's flow snapshot
# already shows the close pair above that cap; citing stressed flow is not
# permission to pay it.
# Hard sidecar Gate A or explicit Gate B close evidence is priority work: do
# not choose TRADE, WAIT, REQUEST_EVIDENCE, PROTECT, or TIGHTEN_SL to sidestep
# it. If only soft Gate A exists and explicit Gate B is missing, or if a hard-
# looking sidecar was downgraded by same-direction HOLD/EXTEND support, the
# sidecar is advisory for non-CLOSE actions; keep TP/profit management active,
# and still evaluate short / medium / long horizon LIVE_READY entries. Do not
# write CLOSE merely to let the verifier reject it; in the entry branch, write
# a TRADE/CANCEL/WAIT/REQUEST_EVIDENCE receipt from the current packet. If
# choosing CLOSE from soft evidence, the verifier must surface
# `CLOSE_OPERATOR_AUTH_REQUIRED`.
# If hard Gate A or explicit Gate B is present, it must require a CLOSE receipt
# first. The default stance when no user instruction is present is HOLD / WAIT
# only when no fresh hard/authorized Gate A close sidecar and no current
# LIVE_READY lane exists — do not write a CLOSE receipt to "reduce risk" from a
# still-valid thesis.

# 4. Verify the receipt
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json

# 5. Run one gateway cycle immediately after every completed verifier result.
# Do not insert refresh, analysis, TP rebalance, projection, or thesis
# sidecar commands between verifier completion and this gateway handoff:
# the decision receipt is tied to the current broker snapshot + order intents,
# and extra work in between can make a tradeable receipt stale.
# This is mandatory even when `gpt-trader-decision` reports REJECTED, WAIT,
# REQUEST_EVIDENCE, or PROTECT. A rejected fresh-entry receipt blocks new risk,
# but it must not skip existing-position maintenance: `autotrade-cycle` and the
# post-cycle sidecars hand open positions to PositionManager and
# PositionProtectionGateway before considering fresh entry risk. Skipping the
# wrapper leaves profit-side partial closes, profitable hedge TPs, profit-lock
# stops, and other dependent-order protection stale.
# `run-autotrade-live.sh` may still refresh a missing or deterministic fallback
# receipt before this gateway step. Schema-v2 TRADE/CLOSE/WAIT/REQUEST_EVIDENCE
# receipts must exactly rebuild from the current baseline/packet/overlay; WAIT
# and REQUEST_EVIDENCE carry veto authority over campaign pressure. The wrapper
# must never overwrite a receipt whose decision_provenance.author_kind is
# CODEX_MARKET_READ, even when broker truth
# is newer or the receipt was rejected/consumed. A stale Codex receipt blocks
# new risk; the next hourly AI pass must regenerate baseline/packet/overlay.
# Verifier rejection still proceeds to gateway maintenance and must not become
# a deterministic send.
# This does not enable target-path live by itself. A target-path send still
# needs QR_TARGET_PATH_LIVE_ENABLED=1 and LiveOrderGateway target-path proof.
QR_LIVE_ENABLED=1 ./scripts/run-autotrade-live.sh \
  --reuse-market-artifacts \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send

# 6. Protection sidecars — automatically run by `run-autotrade-live.sh` after
# a zero-exit wrapper cycle while the live lock is still held. This closes
# the stale-state window where `autotrade-cycle` refreshes broker truth after
# verifier completion but position sidecars / memory-health / self-improvement
# still point at the pre-gateway snapshot. When `autotrade-cycle` exits
# non-zero after refreshing broker truth, the wrapper does NOT run the full
# broker/order sidecar list; it calls the canonical
# `post-autotrade-failure-sidecars` command. That command first refreshes
# `broker-snapshot` and `daily-target-state`, then runs the
# projection/position/audit repair subset, including `position-management`
# followed by `position-execution` when management succeeds, then refreshes
# `broker-snapshot` and `daily-target-state` again, syncs the execution ledger,
# refreshes `capture-economics`, `qr-trader-run-watchdog`, and
# `guardian-receipt-consumption`, and only then reprices `order_intents`
# with `generate-intents --reuse-market-artifacts`. It then
# regenerates `optimize-coverage` and `ai-attack-advice` from that final intent
# packet, and reruns read-only position evidence sidecars against the final
# broker/intent packet before `guardian-trigger-contract` → `guardian-event-router` →
# `qr-trader-run-watchdog` → `guardian-receipt-consumption` →
# `profit-capture-bot` → `memory-health` →
# `self-improvement-audit` →
# `profitability-acceptance` → `trader-support-bot` →
# `trader-repair-orchestrator` → `trader-goal-loop-orchestrator` →
# `active-trader-contract` → `active-opportunity-board` →
# `non-eurusd-proof-lane-mapper` → `non-eurusd-live-grade-frontier` →
# `active-trader-contract` → `entry-frequency-recovery` →
# `active-trader-contract` → `forecast-pattern-refresh` →
# `active-trader-contract` → `range-rail-geometry-repair` →
# `active-trader-contract` → `guardian-trigger-contract` →
# `guardian-event-router` → `operator-review-report` →
# `trader-support-bot` → `trader-repair-orchestrator` →
# `trader-goal-loop-orchestrator`. It preserves the original
# wrapper exit code and avoids carrying a stale P0 into the next route.
# Do not run a second routine `cycle-sidecars` after the wrapper unless the
# wrapper was intentionally called with `QR_RUN_POST_GATEWAY_SIDECARS=0` for
# diagnostics.
#
# `cycle-sidecars` runs (canonical list: `cli._cycle_sidecar_steps`):
#   broker-snapshot → tp-rebalance → execution-ledger-sync → predictive-scout-proof → broker-snapshot
#   → daily-target-state → profit-partial-close → verify-projections
#   → position-thesis-check → thesis-evolution-check → forecast-persistence-check
#   → position-management → position-execution → guardian-trigger-contract
#   → guardian-event-router
#   → execution-ledger-sync → predictive-scout-proof → broker-snapshot → daily-target-state
#   → capture-economics
#   → qr-trader-run-watchdog
#   → guardian-receipt-consumption
#   → generate-intents --reuse-market-artifacts
#   → optimize-coverage → ai-attack-advice
#   → position-thesis-check → thesis-evolution-check
#   → forecast-persistence-check → position-management → guardian-trigger-contract
#   → guardian-event-router
#   → qr-trader-run-watchdog
#   → guardian-receipt-consumption
#   → profit-capture-bot → memory-health
#   → self-improvement-audit → profitability-acceptance → trader-support-bot
#   → as-live-ready-evidence-loop → as-4x-proof-path
#   → trader-repair-orchestrator → trader-goal-loop-orchestrator
#   → active-trader-contract → active-opportunity-board
#   → non-eurusd-proof-lane-mapper → non-eurusd-live-grade-frontier
#   → active-trader-contract → entry-frequency-recovery
#   → active-trader-contract → forecast-pattern-refresh
#   → active-trader-contract → range-rail-geometry-repair
#   → active-trader-contract → guardian-trigger-contract
#   → guardian-event-router → operator-review-report
#   → trader-support-bot → trader-repair-orchestrator
#   → trader-goal-loop-orchestrator
# and prints one compact digest.
#
# Semantics preserved from the per-step skeleton:
# - TP rebalance is TP-only management; SL-free positions keep
#   stop_loss=None untouched. Missing broker TP stays a no-broker-TP runner
#   unless QR_ENABLE_MISSING_TP_REPAIR=1 or profitable-runner insurance
#   conditions apply. Hysteresis + entry-side + safety-margin invariants
#   prevent accidental fires.
# - profit-partial-close is profit-side only and sends only when
#   QR_LIVE_ENABLED=1 (the consolidated runner adds --send --confirm-live
#   under that env, same triple gate as before). It never realizes a loss
#   and never writes a stop-loss to manual/tagless positions.
# - verify-projections resolves PENDING → HIT/MISS/TIMEOUT in
#   data/projection_ledger.jsonl for the self-calibrating projection loop.
# - position-thesis-check / thesis-evolution-check (2026-05-15 user
#   directive 「どの視点でエントリーしたのか…市況が変わってないか」) /
#   forecast-persistence-check emit the per-position EXTEND / HOLD /
#   REVIEW_CLOSE, STILL_VALID / WEAKENED / BROKEN, and N-cycle persistence
#   verdicts. A fresh BROKEN / RECOMMEND_CLOSE from thesis evolution carries
#   hard Gate A standing loss-cut authorization only when its rationale records
#   the canonical buffered price-invalidation hit plus technical confirmation
#   against the same recorded position side. Structural prose in this report
#   is soft; use the timestamped H4 / position-management path instead.
#   Forecast flip, adverse drift,
#   confidence/regime decay, THESIS_EXPIRED, score-only, and
#   adverse-entry-buffer-only reviews still need explicit env/token Gate B.
#   thesis-evolution-check must first backfill active trader-owned entry
#   theses from execution-ledger broker truth when possible, then evaluate.
#   A backfillable ledger gap must not survive as an UNVERIFIABLE blocker
#   until a later memory-health pass; otherwise the route can freeze on a
#   missing-thesis state that the same cycle already knows how to repair.
# - position-management is regenerated against the post-gateway broker
#   snapshot, then position-execution consumes any profit-only
#   TAKE_PROFIT_MARKET / TP-update decision through PositionProtectionGateway.
#   This is the full-cycle fallback for fast TP-progress wins when the separate
#   launchd position guardian is inactive, stale, or skipped under the live
#   lock. Live sends still require QR_LIVE_ENABLED=1 plus --send --confirm-live;
#   target-path sends additionally require QR_TARGET_PATH_LIVE_ENABLED=1 and
#   their target-path receipt proofs. Fresh entries require both a loaded
#   guardian and a recent guardian heartbeat unless the operator uses an
#   explicit override.
# - memory-health BLOCK does not grant/deny a trade by itself;
#   trader-prompt-route reads it for the next cycle's routing.
# - self-improvement-audit is recalculated after the post-gateway snapshot and
#   memory-health pass so the next route sees current P0/P1/P2 gates instead
#   of the pre-gateway refresh audit.
# - profitability-acceptance is the single red/green profit invariant gate:
#   it aggregates self-improvement P0s, negative capture expectancy, TP-proven
#   market-close leakage, unverified loss-side broker-close reconciliation,
#   projection headline-vs-economic precision gaps, and rank-only contrarian
#   replay edges. A broker `TRADE_CLOSE` that was only reconciled after the
#   fact from a trader entry lane is not proved close discipline; loss-side
#   closes need durable `GATEWAY_GPT_CLOSE_ACCEPTED` and/or
#   `GATEWAY_TRADE_CLOSE_SENT` provenance plus the decision packet's
#   `close_gate_evidence[]` before the system can count them as verified
#   structural exits. A missing, stale, or unreadable acceptance file routes
#   back to refresh; P0 findings route to learning/repair and keep high-turn
#   scaling blocked until the named evidence clears. The only entry exception
#   is an attached-TP HARVEST repair lane that explicitly carries the
#   self-improvement P0 repair metadata.
# - profit-capture-bot is read-only and runs after position-management. It
#   recalculates the TP-progress TAKE_PROFIT_MARKET gates for each open
#   trader-owned position and names the current state as BANKABLE_NOW, watch
#   only, or blocked by missing quote/chart/ATR/TP inputs. It never sends a
#   close or changes TP/SL; execution still belongs to PositionProtectionGateway.
# - trader-support-bot is read-only and runs after the acceptance gate so the
#   compact cycle digest names the operational support state: guardian active /
#   heartbeat freshness, qr-trader scheduled-run watchdog status, unresolved
#   guardian receipt issues, latest guardian receipt consumption/operator-review
#   status, whether normal routing is allowed, current profit-capture gate state, TP-progress
#   profit-capture misses, runtime disk pressure / recent ENOSPC artifact-write
#   failures, recovered-after-ENOSPC evidence, fresh-entry send allowed flag, repair-frontier lanes, explicit
#   operator actions, and the latest active contract/board/frontier
#   lane so support visibility cannot fall back to a legacy EUR/USD diagnostic
#   when the terminal active path selected a non-EUR lane. Its next-action text
#   must prefer terminal active_trader_contract.next_trade_enabling_action /
#   next_prompt over older embedded board-lane next_action so already-consumed
#   entry-frequency or forecast-refresh work does not reappear after
#   range-rail geometry repair has advanced the contract. Runtime disk repair
#   requests may run QuantRabbit disk maintenance/read-only checks, but if later
#   critical artifacts were written successfully and free space is above the
#   operating floor, preserve the evidence as ENOSPC_RECOVERED instead of
#   blocking for a fixed time window. Never delete unrelated user files without
#   operator-owned cleanup. It never loads launchd,
#   sends orders, closes positions, cancels entries, or wakes the trader.
# - as-live-ready-evidence-loop and as-4x-proof-path are read-only artifact
#   builders that run after trader-support-bot. They refresh
#   `data/rolling_30d_4x_firepower_board.json`, `data/as_proof_pack_queue.json`,
#   `data/as_lane_candidate_board.json`, `data/portfolio_4x_path_planner.json`,
#   and `data/harvest_live_grade_path.json` from the final broker/target/order/
#   profitability/support packet so A/S proof queue, HARVEST live-grade path,
#   and 4x math cannot stay stale after `cycle-refresh`.
# - trader-repair-orchestrator is read-only and runs after those A/S proof
#   artifacts.
#   It converts `repair_requests` into a Codex repair queue with suggested
#   files, test commands, verification commands, commit/live-sync requirement,
#   and a top-level `codex_work_order` that an external Codex automation can
#   consume directly. It also writes `loop_engineering_prompt`, the
#   continuously updated 5% campaign repair prompt: current operational
#   reachability, A/S proof queue count, live-permission candidate count,
#   rejected proof candidates, latest gateway status, selected/waiting blocker,
#   next loop, self-review questions, anti-loop rules, and verification
#   commands. The prompt is guidance for Codex repair/evidence work, not live
#   permission. The work order repeats the
#   hard boundary that orders, cancels, closes, and launchd load/reload require
#   explicit approval or an existing gateway path. If a blocked support artifact is older and lacks
#   top-level `repair_requests`, it rebuilds the queue from embedded
#   acceptance/guardian/frontier evidence instead of returning
#   `NO_REPAIR_REQUESTS`. It grants no live permission and does not call model
#   APIs from QuantRabbit code.
# - trader-goal-loop-orchestrator is read-only and runs after
#   trader-repair-orchestrator. It reads the current repair/active-contract/
#   payoff/HARVEST/SCOUT/proof queue/lane board/portfolio/live-order/broker artifacts, then writes
#   `data/trader_goal_loop_orchestrator.json` and
#   `docs/trader_goal_loop_orchestrator_report.md` with the next Codex work type
#   and a complete Japanese prompt. The terminal board/contract sync reruns it
#   after terminal `trader-support-bot` refreshes current active-path support
#   visibility from `operator-review-report` and `active_trader_contract`, so
#   the final prompt consumes the latest lane-specific work order instead of
#   stale generic payoff work. It prioritizes evidence growth and expectancy improvement
#   toward rolling 30d funding-adjusted equity 4x:
#   operator-review SCOUT judgement material first when
#   `SCOUT_BLOCKED_OPERATOR_REVIEW` is active, otherwise live-grade HARVEST proof
#   path, SCOUT evidence, NO_TRADE exclusion, or read-only
#   `EDGE_IMPROVEMENT_EXPERIMENT` design, or
#   `ACTIVE_TRADER_CONTRACT_EVIDENCE` when the terminal active contract exposes
#   a current read-only lane-specific `next_prompt`. It never grants live
#   permission,
#   treats `proof_queue_count=0` as a blocker, and does not call model APIs from
#   QuantRabbit code.
# - active-trader-contract is read-only and runs after
#   trader-goal-loop-orchestrator. It writes `data/active_trader_contract.json`
#   and `docs/active_trader_contract.md`, selects exactly one active path
#   (`HARVEST_READY_CHECK`, `SCOUT_READY_CHECK`, `EVIDENCE_ACQUISITION`,
#   `EDGE_IMPROVEMENT_EXPERIMENT`, `OPERATOR_REVIEW_REPORT`,
#   `LIVE_PERMISSION_READY_CHECK`, or `NO_TRADE_WITH_CAUSE`), and emits the
#   machine-readable NO_ACTION contract. It keeps exact LIMIT S5 bid/ask replay,
#   proof queue, negative expectancy, month-scale replay, guardian, and gateway
#   blockers visible with `live_permission_allowed=false` and
#   `live_side_effects=[]`. When a previous `active_opportunity_board` exists,
#   it consumes the board-ranked multi-pair/multi-vehicle facts as loop-break
#   context only: a failed STOP exact replay remains not-SCOUT-ready and must
#   not be repeated as the next active step unless new independent evidence
#   appears. If the latest board reranks every lane as `NO_TRADE_WITH_CAUSE`,
#   the contract must select `NO_TRADE_WITH_CAUSE` instead of falling back to
#   stale single-lane EUR_USD replay work; stale guardian receipt blockers
#   cleared by the board's current guardian artifacts must not remain in
#   `remaining_blockers`. If the selected path comes from the board top lane,
#   `next_prompt` must name that current lane shape rather than the legacy fixed
#   EUR_USD target. It also reads `data/guardian_events.json`: when a matching
#   board-selected range-rail lane has a fired `CONTRACT_ADD_TRIGGER` /
#   `range_rail_recheck`, consume that fired event and advance to fresh broker
#   truth repricing / active-board refresh / exact TP-proof collection instead
#   of repeating `WAIT_FOR_RANGE_RAIL_RECHECK`. This fired event is wake
#   evidence only, not live permission. The board/contract sync then reruns
#   trader-goal-loop-orchestrator so this terminal `next_prompt` becomes the
#   visible next Codex work order in the same cycle.
#   If the board lane is EVIDENCE_ACQUISITION because positive zero-loss local
#   TAKE_PROFIT_ORDER proof exists below the collection floor, preserve that
#   status and blocker instead of normalizing it back to NO_TRADE_WITH_CAUSE;
#   the next work remains exact TP-proof collection with live permission false.
# - active-opportunity-board is read-only and runs after active-trader-contract.
#   It writes `data/active_opportunity_board.json` and
#   `docs/active_opportunity_board.md`, compares all visible pairs, directions,
#   strategy families, and LIMIT/STOP/MARKET vehicles, and ranks the next active
#   4x path without letting EUR_USD|SHORT|BREAKOUT_FAILURE become the only loop.
#   Each lane is classified as LIVE_READY, HARVEST_READY, SCOUT_READY,
#   EVIDENCE_ACQUISITION, OPERATOR_REVIEW_REQUIRED, or NO_TRADE_WITH_CAUSE.
#   It reads guardian receipt consumption/operator-review artifacts. Current
#   `order_intents` guardian receipt blockers remain hard blockers, but when
#   both current guardian artifacts have `normal_routing_allowed=true`, stale
#   `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED` inherited only from older
#   planner/proof/replay artifacts must move to `stale_source_blockers`.
#   If HARVEST live-grade output is older than current `order_intents`, its
#   promotion blockers are stale diagnostics for newer current-intent lanes
#   until refreshed; keep them in `stale_source_blockers` rather than hard
#   blockers. Within the same status, especially all-lane
#   `NO_TRADE_WITH_CAUSE`, rank current `order_intents` executable candidates
#   with fewer current hard blockers before high-score diagnostic lanes.
#   Failed exact replay and current guardian receipt operator review outrank
#   evidence-acquisition; negative-expectancy and replay-negative blockers
#   outrank manual/operator-overlap review. Manual overlap blockers such as
#   `OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED` are live blockers but are not
#   `OPERATOR_REVIEW_REQUIRED` unless an explicit guardian/operator-review code
#   is also present. Exception: if a current `BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE`
#   blocker is backed only by stale packaged evidence or a missing audit report,
#   keep the negative blocker visible, add `BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED`,
#   and route the lane to `EVIDENCE_ACQUISITION` for a fresh exact S5 bid/ask
#   replay/rule-package refresh rather than permanent no-trade. Pair-filtered
#   bid/ask replay refresh is allowed for the board-ranked frontier, but package
#   refresh must preserve existing runtime rule rows for pairs outside
#   `source_pair_filter`; a targeted refresh must not erase unrelated pair
#   blockers/support while updating or clearing the refreshed pairs. `NO_TRADE_WITH_CAUSE` must carry a concrete
#   machine-readable blocker, not an empty cause set.
#   Entry-drought recovery evidence must be materially profitable; microscopic
#   positive closed P/L must not promote a lane over a current live-grade
#   frontier. Positive zero-loss local TAKE_PROFIT_ORDER proof below the floor
#   is read-only EVIDENCE_ACQUISITION for exact TP-proof collection, while zero
#   local TP proof remains NO_TRADE_WITH_CAUSE. If exact local TAKE_PROFIT_ORDER
#   proof already meets the broker-TP floor with zero losses and positive
#   expectancy, do not ask for more TP-proof collection; tag the lane as a
#   read-only TP-proven HARVEST blocker-repair / edge-improvement candidate,
#   preserve guardian, bid/ask, forecast, risk, market-close, and profitability
#   blockers, and rank it ahead of below-floor or zero-proof lanes within the
#   same active-path class. Under the same live-grade / evidence proximity with
#   negative expectancy, prefer LIMIT or STOP proof paths before MARKET without
#   mixing proof between vehicles.
#   If the selected board lane is OPERATOR_REVIEW_REQUIRED and also carries a
#   TP-proven edge-improvement candidate, active-trader-contract must keep the
#   post-review read-only EDGE_IMPROVEMENT_EXPERIMENT follow-up in
#   next_trade_enabling_action while live_permission_allowed remains false.
#   It never grants live order, SCOUT, gateway, cancel/close, launchd, gate
#   relaxation, lot-backsolve, secret-disclosure, or inferred operator approval.
# - predictive-scout-proof is read-only and runs after execution-ledger-sync.
#   The S5 contrarian replay is a forecast-failure hypothesis, not proof of the
#   passive LIMIT retest. Live evidence collection is canonical-digest bound,
#   LIMIT-only, current-NAV risk-sized positive integer units (including
#   1-999u), GTD no later than forecast horizon (max 90m), actual attached
#   TP+intent SL, canonical TP/SL distances, at most two active SCOUTs and 2%
#   NAV aggregate tier risk, failure_trader/BREAKOUT_FAILURE identity, no
#   basket/reprice/chase, and thirty atomic pre-POST reservations/day maximum.
#   TraderBrain/AutoTradeCycle fix the size multiplier at 1.0; SQLite atomically
#   claims one vehicle+forecast-cycle signal, so retrying the same intent cannot
#   duplicate exposure or proof. Filled SCOUT TP/SL is frozen across position
#   manager, position gateway, TP rebalancer, trailing-SL, and partial-close
#   paths. The claim/check/proof DB is always the canonical execution_ledger.db
#   beside order_intents.json; live SCOUT sends reject redirected custom DB paths.
#   One resolved net loss starts a six-hour exact-vehicle cooldown; three
#   resolved losses with cumulative negative net quarantine the stable selector
#   + side + LIMIT + TP/SL vehicle. Evidence-stat refreshes cannot reset it.
#   Generic post-invalidation auto-reversal is forbidden. Promotion remains
#   false; cycle digest/operator-review surface eligibility only after every fill resolves,
#   n>=30 across >=5 days, PF>=1.2, positive-day rate>=2/3, and the one-sided
#   95% all-exit net-mean lower bound is above zero.
# - non-eurusd-proof-lane-mapper and non-eurusd-live-grade-frontier are
#   read-only and run after active-opportunity-board. The mapper prevents
#   historical non-EUR/USD profit evidence from being promoted without exact
#   current lane/vehicle mapping. The frontier ranks the current EUR/USD and
#   non-EUR/USD order-intent lanes by live-grade proximity, surfaces
#   USD_CAD LONG BREAKOUT_FAILURE blockers, and keeps proof-floor, bid/ask,
#   spread, forecast, and loss-budget gaps explicit. `NON_EURUSD_FRONTIER_FOUND`
#   means a non-EUR/USD frontier lane exists, even if EUR/USD remains closer;
#   `required_checks.non_eurusd_closer_than_eurusd` carries the relative
#   closeness verdict. `ONLY_EURUSD_FRONTIER_FOUND` requires no non-EUR/USD
#   frontier lane at all. The terminal
#   active-trader-contract may consume frontier.next_evidence_lane as a
#   read-only EVIDENCE_ACQUISITION target when the board is otherwise all
#   NO_TRADE_WITH_CAUSE, or as a same-shape supplement to a board lane. In
#   either case, `next_prompt` and `next_trade_enabling_action` must name the
#   same frontier lane; stale board-local prompts such as range-rail
#   wait/recheck must not hide or override the frontier action. This still
#   never grants live order, cancel/close,
#   launchd, gate relaxation, lot-backsolve, market-close-loss-as-TP-proof,
#   secret-disclosure, or inferred approval.
# - entry-frequency-recovery is read-only and runs after the terminal
#   active-trader-contract selects an entry-drought evidence lane, then
#   active-trader-contract runs again to consume it. It writes
#   `data/entry_frequency_recovery.json` and
#   `docs/entry_frequency_recovery_report.md`, diagnoses profitable historical
#   lanes with zero recent accepted/filled entries from forecast_history,
#   projection_ledger, strategy_profile, order_intents, active board/frontier,
#   and execution ledger, and emits concrete forecast/pattern/profile/TP-proof
#   tuning actions. It must not force MARKET entries under a RANGE forecast;
#   route that work to RANGE_ROTATION geometry or trigger/TP proof while
#   keeping every blocker visible. It never grants live order, cancel/close,
#   launchd, gate relaxation, lot-backsolve, market-close-loss-as-TP-proof,
#   secret-disclosure, model API calls, or inferred approval.
# - forecast-pattern-refresh is read-only and runs after entry-frequency
#   recovery plus a contract consumption pass, then active-trader-contract runs
#   again to consume it. It writes `data/forecast_pattern_refresh.json` and
#   `docs/forecast_pattern_refresh_report.md`, audits the latest RANGE box
#   location, maps the matching RANGE_ROTATION LIMIT/MARKET counterpart,
#   verifies expired/pending trigger projections, preserves spread/bid-ask/
#   negative-expectancy blockers, and emits lane-local actions such as
#   RANGE_RAIL_GEOMETRY_REPAIR, VERIFY_TRIGGER_PROJECTIONS, or
#   EXACT_TP_PROOF_COLLECTION. It must not force non-range MARKET recovery
#   under a RANGE forecast, chase the current range midpoint/opposite rail, or
#   treat refresh output as live permission.
# - range-rail-geometry-repair is read-only and runs after forecast-pattern
#   refresh plus a contract consumption pass, then active-trader-contract runs
#   again to consume it, then guardian-trigger-contract / guardian-event-router
#   refresh once more so WAIT_FOR_RANGE_RAIL_RECHECK becomes a bot-monitored
#   wake condition in the same cycle. It writes `data/range_rail_geometry_repair.json` and
#   `docs/range_rail_geometry_repair_report.md`, converts
#   RANGE_RAIL_GEOMETRY_REPAIR into an executable rail success condition,
#   checks the RANGE_ROTATION LIMIT counterpart's entry/TP/SL geometry against
#   the range box, and emits WAIT_FOR_RANGE_RAIL_RECHECK,
#   REPRICE_RANGE_ROTATION_COUNTERPART, or
#   RANGE_ROTATION_GEOMETRY_READY_PROOF_BLOCKED while preserving spread,
#   bid-ask, negative-expectancy, and range-location blockers. After
#   guardian-event-router fires the watch-only `CONTRACT_ADD_TRIGGER`, the next
#   active-trader-contract pass must treat the rail as reached and move to
#   repricing/proof work rather than another wait-only prompt. It never grants
#   live order, cancel/close, launchd, gate relaxation, lot-backsolve,
#   market-close-loss-as-TP-proof, secret-disclosure, model API calls, or
#   inferred approval.
# - operator-review-report is read-only and runs after the terminal
#   active-trader-contract. It writes `data/operator_review_report.json` and
#   `docs/operator_review_report.md` from the current contract, active board,
#   non-EUR frontier, guardian receipt consumption/operator-review, watchdog,
#   and broker snapshot. It packages judgement material for the current top
#   lane only; it never writes `data/guardian_receipt_operator_review.json`,
#   infers approval, grants live permission, sends/cancels/closes, changes
#   launchd, relaxes gates, hides negative expectancy, or back-solves lot size.
# Manual recovery only:
# QR_RUN_POST_GATEWAY_SIDECARS=0 QR_LIVE_ENABLED=1 ./scripts/run-autotrade-live.sh ...
# QR_LIVE_ENABLED=1 PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli cycle-sidecars

# 4c. adverse-partial-close is DISABLED 2026-05-14:
# The module closed 50% of 471020 AUD/JPY SHORT for -2,516 JPY based
# on adverse-pips threshold, violating feedback_market_over_risk_budget.md
# (「SL-free 下で CLOSE 判断は構造/MTF/thesis のみ。含み損%/JPY は
# 遅行指標で判断材料にしない」). Locking in losses on adverse-pip
# threshold IS the anti-pattern SL-free was designed to avoid.
# Module file kept in src/quant_rabbit/strategy/adverse_partial_close.py.
# The CLI is dry-run by default and live execution is triple-gated by
# --send --confirm-live plus QR_LIVE_ENABLED=1, but the scheduled trader
# still does not invoke it.
# PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli adverse-partial-close --dry-run
```

## End Report

- Filled `5% PACE BOARD` with a concrete A/S Path A / HERO route, +10% extension setup gate candidate, or exact blocker.
- Filled `ATTACK STACK`, and every non-empty path-board slot maps to NOW, RELOAD, or SECOND SHOT.
- Filled `10% EXTENSION GATE`; if YES, cite each gate condition, otherwise report NO.
- Filled `USER ALPHA CONTINUATION`; if active, state thesis-alive / RELOAD / SECOND SHOT / exact blocker / next trigger.
- Dry-run sizing result from `tools/position_sizing.py` or `tools/place_trader_order.py` for any fresh target-path order.
- Active opportunity board checked for the next read-only active path across pairs, directions, strategy families, and vehicles.
- Final action: `TRADE`, `WAIT`, `REQUEST_EVIDENCE`, `PROTECT`, `TIGHTEN_SL`, `CLOSE`, or `CANCEL_PENDING`.
- Sent flag: `true`, `false`, or dry-run.
- Selected lane id(s), if any.
- Daily target progress from `data/daily_target_state.json`.
- `gpt-trader-decision` result and issue codes.
- Gateway result and report paths under `docs/*_report.md`.
- Target-path live receipt fields, if any, plus the next `daily-review` LIVE-LEARNING classification.
- Execution ledger DB/report: `data/execution_ledger.db`, `docs/execution_ledger_report.md`.
- Verification ledger JSON/SQL/report: `data/verification_ledger.json`, `data/execution_ledger.db`, `docs/verification_ledger_report.md`.
- Profit capture bot JSON/report: `data/profit_capture_bot.json`, `docs/profit_capture_bot_report.md`.
- qr-trader run watchdog JSON/report: `data/qr_trader_run_watchdog.json`, `docs/qr_trader_run_watchdog_report.md`.
- Guardian receipt consumption JSON/report: `data/guardian_receipt_consumption.json`, `docs/guardian_receipt_consumption_report.md`.
- Guardian receipt operator review JSON/report: `data/guardian_receipt_operator_review.json`, `docs/guardian_receipt_operator_review_report.md`.
- Operator review material JSON/report: `data/operator_review_report.json`, `docs/operator_review_report.md`.
- Entry-frequency recovery JSON/report: `data/entry_frequency_recovery.json`, `docs/entry_frequency_recovery_report.md`.
- Forecast-pattern refresh JSON/report: `data/forecast_pattern_refresh.json`, `docs/forecast_pattern_refresh_report.md`.
- Range rail geometry repair JSON/report: `data/range_rail_geometry_repair.json`, `docs/range_rail_geometry_repair_report.md`.
- Trader support JSON/report: `data/trader_support_bot.json`, `docs/trader_support_bot_report.md`.

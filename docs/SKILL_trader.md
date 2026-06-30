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
- `guardian-event-router` is read-only: it writes event/wake artifacts for GPT-5.5 and never sends, cancels, or closes broker orders.
- Main trader runtime policy: `gpt-5.5`, `reasoning_effort=high`, every 60 minutes.
- Do not rely on the hourly full-trader cadence for risk monitoring. `guardian-event-router` / probe paths remain deterministic, non-LLM, and frequent.
- The `com.quantrabbit.guardian-wake-dispatcher` LaunchAgent may wake GPT-5.5 with read-only `codex exec`; its live default must keep `QR_GUARDIAN_WAKE_GATEWAY_HANDOFF=0` and `QR_GUARDIAN_ACTION_EXECUTE=0`, so wake output is review/receipt only unless a separate explicit gateway path is enabled.
- Read `data/guardian_escalation.json`, `data/guardian_events.json`, `data/guardian_action_receipt.json`, `data/guardian_action_cycle_result.json`, and `docs/guardian_action_review.md` every cycle before normal new-entry routing.
- Resolve queued guardian wake actions before ordinary new entries: `queued_for_active_trader=true` means the dispatcher yielded to the active trader, so the trader must review the event/report/receipt first and either consume the receipt through the normal verifier/gateway path, recognize that `guardian-action-cycle` already executed/rejected it, or write the exact reason it is stale/rejected.
- Target-path entry sends require `QR_TARGET_PATH_LIVE_ENABLED=1` in addition to `QR_LIVE_ENABLED=1`; default is dry-run/stage/LIVE-LEARNING receipt only.
- OANDA position changes go only through `PositionProtectionGateway`.
- Direct `OandaExecutionClient.close_trade()` is blocked; live market closes must use the provenance-aware gateway/partial-close paths and leave a position-execution receipt.
- Do not print secrets.
- Do not call `QR_OPENAI_API_KEY`, `OPENAI_API_KEY`, or any model API path from QuantRabbit code.
- Do not invent JPY caps, pip distances, reward/risk multipliers, stale defaults, or extra risk gates.
- Missing required evidence is a blocker, not a value to guess.
- Rolling target accounting uses `ROLLING_30D_4X` as the top KPI: 30 calendar days to 4x. +5% is a pace marker / review trigger / protection milestone, and +10% is extension-only after an explicit favorable-market gate.
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
# Legacy fixed-units fallback used only when QR_TRADER_POSITION_NAV_PCT
# is unset. Do NOT remove — backstops smoke scripts that pin units. The
# NAV-pct path above takes precedence whenever set.
export QR_TRADER_BASE_UNITS="${QR_TRADER_BASE_UNITS:-3000}"
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
# rolling_30d_start_equity, current_equity, current_30d_multiplier,
# remaining_to_4x, required_calendar_daily_return, required_active_day_return,
# and pace_state every cycle. While remaining_to_5pct is above zero, fill a
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
- A blocked but correct read is discovery success / execution miss. A wrong read that passes filters is market-read failure.
- Final `TRADE` / `WAIT` text must reference the next 30m or next 2h prediction from `MARKET READ FIRST`.
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
# daily-target-state → capture-economics → generate-intents --reuse-market-artifacts →
# optimize-coverage → ai-attack-advice →
# learning/execution-timing/manual-market-context/operator-precedent/verification audits →
# generate-predictive-limits → position sidecars → guardian-event-router →
# profit-capture-bot → memory-health → self-improvement-audit → profitability-acceptance →
# trader-support-bot → trader-repair-orchestrator) in one
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
# `memory-health` must audit the same `capture_economics` packet used by
# `generate-intents`. If capture-economics is refreshed after order_intents, or
# memory-health does not record the current capture timestamp, route back to
# refresh before entry/verify work.
#
# Position close sidecars inside the digest are read-only prediction/thesis
# evidence. Read `protection_sidecars.position_close_recommendations[]`
# before deciding: `blocks_non_close_actions=true` means close-first work;
# `blocks_non_close_actions=false` means soft advisory only and must not
# produce a CLOSE receipt from the entry branch. Fresh thesis_evolution
# BROKEN / RECOMMEND_CLOSE, structural position_management REVIEW_EXIT, and
# position_thesis invalidation-hit or structural-break evidence with multi-TF
# confirmation are hard standing loss-cut authorization only when they do not
# conflict with fresh same-direction HOLD/EXTEND sidecars. If thesis_evolution
# / position_thesis / forecast_persistence still support the open side, treat
# the issue as HOLD/reprice/TP rebalance unless explicit Gate B is present.
# Adverse-entry-buffer-only position_thesis evidence is soft and still needs
# explicit Gate B before CLOSE; without Gate B, soft sidecars are advisory for
# non-CLOSE actions and must not block separate current LIVE_READY entries on
# other pairs or horizons.

# 3. Write data/codex_trader_decision_response.json from the active decision branch
# The scheduled trader should first let `trader-draft-decision` compose one
# current receipt from the same broker/market/news packet that the verifier will
# audit. The command is read-only except for the receipt/report files: it does
# not call model APIs, send orders, cancel orders, close positions, or change
# launchd state. It selects current LIVE_READY lanes from order_intents /
# ai_attack_advice when clean, and emits WAIT / REQUEST_EVIDENCE when named
# blockers such as news-health, projection, exposure, close-first, or
# self-improvement gates win.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-draft-decision \
  --snapshot data/broker_snapshot.json \
  --output data/codex_trader_decision_response.json
#
# If broker refresh made an older receipt stale, overwrite it with one current receipt.
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
# If the draft reports DRAFT_REQUIRES_OPERATOR_REVIEW, do not invent a
# deterministic workaround. Continue to `gpt-trader-decision` and the gateway
# maintenance cycle so existing-position protection still runs, then repair the
# named blocker or receipt bug after the handoff.
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
#     authorization. Hard standing authorization is H4 close-confirmed
#     BOS/CHOCH against side, buffered invalidation_price hit with technical
#     confirmation, fresh thesis_evolution BROKEN / RECOMMEND_CLOSE,
#     structural position_management REVIEW_EXIT, or position_thesis
#     invalidation-hit / structural-break evidence with multi-TF confirmation.
#     M15 close-confirmed BOS/CHOCH is Gate A evidence, but it is not unattended
#     hard Gate B unless H4 structure, recorded invalidation, or a hard sidecar
#     also confirms. Softer sidecars (adverse-entry-buffer-only or score-only
#     position_thesis REVIEW_CLOSE, non-structural position_management
#     REVIEW_EXIT, forecast_persistence RECOMMEND_CLOSE, or M15-only structure)
#     require `QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell, OR a fresh
#     `data/.operator_close_token` file. The receipt field
#     `operator_close_authorized: true` is advisory audit text only.
# `gpt-trader-decision` writes `close_gate_evidence[]` for every CLOSE trade id,
# accepted or rejected. Read it before sending/staging any loss-side CLOSE:
# it must show the Gate A reason, standing-vs-explicit Gate B state, P0/timing
# audit citations, and any same-direction support conflict. A missing durable
# close-gate evidence row is an audit defect, not proof that the close is safe.
# A TRADE receipt must not list close_trade_ids. If hard Gate A or explicit
# Gate B close evidence is present, write one current CLOSE receipt first and
# end that autotrade cycle as close-only. Refresh broker truth / intents on the
# next scheduled cycle, and only re-enter on a fresh LIVE_READY lane with a
# separate verified TRADE receipt. The automation must not re-enter in the same
# outer cycle after the close is sent, staged, or already satisfied.
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
# `broker-snapshot` and `daily-target-state` again before repricing
# `order_intents` with `generate-intents --reuse-market-artifacts`. It then
# regenerates `optimize-coverage` and `ai-attack-advice` from that final intent
# packet, and reruns read-only position evidence sidecars against the final
# broker/intent packet before `guardian-event-router` →
# `profit-capture-bot` → `memory-health` →
# `self-improvement-audit` →
# `profitability-acceptance` → `trader-support-bot` →
# `trader-repair-orchestrator`. It preserves the original
# wrapper exit code and avoids carrying a stale P0 into the next route.
# Do not run a second routine `cycle-sidecars` after the wrapper unless the
# wrapper was intentionally called with `QR_RUN_POST_GATEWAY_SIDECARS=0` for
# diagnostics.
#
# `cycle-sidecars` runs (canonical list: `cli._cycle_sidecar_steps`):
#   broker-snapshot → tp-rebalance → execution-ledger-sync → broker-snapshot
#   → daily-target-state → profit-partial-close → verify-projections
#   → position-thesis-check → thesis-evolution-check → forecast-persistence-check
#   → position-management → position-execution → guardian-event-router
#   → broker-snapshot → daily-target-state
#   → generate-intents --reuse-market-artifacts
#   → optimize-coverage → ai-attack-advice
#   → position-thesis-check → thesis-evolution-check
#   → forecast-persistence-check → position-management → guardian-event-router
#   → profit-capture-bot → memory-health
#   → self-improvement-audit → profitability-acceptance → trader-support-bot
#   → trader-repair-orchestrator
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
#   verdicts. A fresh BROKEN / RECOMMEND_CLOSE from thesis evolution is hard
#   Gate A standing loss-cut authorization; score-only or
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
#   heartbeat freshness, current profit-capture gate state, TP-progress
#   profit-capture misses, fresh-entry send allowed flag, repair-frontier lanes,
#   and explicit operator actions. It
#   never loads launchd, sends orders, closes positions, or cancels entries.
# - trader-repair-orchestrator is read-only and runs after trader-support-bot.
#   It converts `repair_requests` into a Codex repair queue with suggested
#   files, test commands, verification commands, commit/live-sync requirement,
#   and a top-level `codex_work_order` that an external Codex automation can
#   consume directly. It also writes `loop_engineering_prompt`, the
#   continuously updated 5% campaign repair prompt: current operational
#   reachability, selected/waiting blocker, next loop, self-review questions,
#   anti-loop rules, and verification commands. The prompt is guidance for
#   Codex repair/evidence work, not live permission. The work order repeats the
#   hard boundary that orders, cancels, closes, and launchd load/reload require
#   explicit approval or an existing gateway path. If a blocked support artifact is older and lacks
#   top-level `repair_requests`, it rebuilds the queue from embedded
#   acceptance/guardian/frontier evidence instead of returning
#   `NO_REPAIR_REQUESTS`. It grants no live permission and does not call model
#   APIs from QuantRabbit code.
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
- Trader support JSON/report: `data/trader_support_bot.json`, `docs/trader_support_bot_report.md`.

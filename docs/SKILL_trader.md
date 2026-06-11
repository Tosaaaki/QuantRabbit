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
- OANDA position changes go only through `PositionProtectionGateway`.
- Do not print secrets.
- Do not call `QR_OPENAI_API_KEY`, `OPENAI_API_KEY`, or any model API path from QuantRabbit code.
- Do not invent JPY caps, pip distances, reward/risk multipliers, stale defaults, or extra risk gates.
- Missing required evidence is a blocker, not a value to guess.
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
# survive a flash move / intervention inside the 20-minute blind window.
export QR_DISASTER_SL="${QR_DISASTER_SL:-1}"
export QR_DISASTER_SL_H4_ATR_MULT="${QR_DISASTER_SL_H4_ATR_MULT:-2.5}"
# Fresh entries need both executable forecast context and auditable telemetry.
# If forecast_history, projection_ledger, or execution_ledger sync is missing,
# generate-intents may diagnose the lane but must not emit LIVE_READY.
export QR_REQUIRE_FORECAST_FOR_LIVE="${QR_REQUIRE_FORECAST_FOR_LIVE:-1}"
export QR_REQUIRE_TELEMETRY_FOR_LIVE="${QR_REQUIRE_TELEMETRY_FOR_LIVE:-1}"

# 1. Route to the right prompt branch
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-prompt-route

# 2. Refresh evidence when routed there — ONE consolidated command.
#
# `cycle-refresh` runs the full refresh step list (broker-snapshot →
# daily-target-state → execution-ledger-sync → import-legacy → mine-strategy →
# pair-charts → cross-asset/context/flow/strength/levels/calendar/COT/skew →
# market-context-matrix → mine-market-stories → news-health --strict →
# daily-review → tp-rebalance → verify-projections → generate-intents →
# optimize-coverage → ai-attack-advice → learning/capture/verification audits →
# generate-predictive-limits → position sidecars → memory-health →
# self-improvement-audit) in one
# process, in the same order and with the same arguments the per-step
# skeleton used (`cli._cycle_refresh_steps` is the canonical list), then
# prints ONE compact digest including the re-routed prompt branch.
#
# Token discipline (2026-06-10): the per-step skeleton burned ~3M tokens per
# 20-minute cycle (one shell turn per command × full-context resend) and
# exhausted the scheduler's credits on 2026-06-09, stopping live trading.
# Read the digest, then drill into `data/order_intents.json`,
# `data/pair_charts.json`, `data/market_context_matrix.json` etc. with
# TARGETED queries (jq / python -c) only where the digest flags something.
# Never cat a multi-megabyte artifact into the conversation.
#
# Long-running commands (2026-06-11): `cycle-refresh`, the live wrapper, and
# `cycle-sidecars` take minutes. Invoke them with ONE long wait (shell-tool
# yield/timeout ≥ 300000 ms) instead of the default ~10s yield — 2026-06-11
# telemetry showed ~25 empty polling turns per cycle, each re-sending the
# whole conversation context, keeping the cycle at ~3.9M tokens despite the
# consolidation. One long wait removes that entire class of spend.
#
# `--daily-risk-pct 10` is forwarded to every daily-target-state step: the
# day's risk budget is % of starting NAV so the per-trade cap auto-scales
# with equity (feedback_use_nav_percent.md), and 10% matches the campaign
# `target_return_pct`.
#
# News stays consumer-only: the digest's `news_health` comes from the
# `qr-news-digest` routine artifacts. Do NOT run `news-snapshot` here — that
# would replace the curated digest with raw RSS (reference_news_pipeline.md).
#
# Failed required steps abort the remaining refresh and exit 2; the digest
# lists `steps_failed` with stderr tails. Optional-step failures (e.g.
# news-health --strict during a stale-news window) appear in `steps_failed`
# but do not stop evidence generation — treat them as named blockers in the
# decision receipt exactly as before.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10

# The refresh branch is not an end state: it must produce one current receipt
# and then proceed through verification + gateway. Ending the cycle right
# after `cycle-refresh` leaves fresh evidence unused and is treated as
# incomplete. The digest's `route` field is the re-route result; only run
# `trader-prompt-route` again if you changed an artifact after the digest.
#
# Position close sidecars inside the digest are read-only prediction/thesis
# evidence. Fresh thesis_evolution BROKEN / RECOMMEND_CLOSE, structural
# position_management REVIEW_EXIT, and position_thesis invalidation-hit or
# structural-break evidence with multi-TF confirmation are hard standing
# loss-cut authorization. Adverse-entry-buffer-only position_thesis evidence
# is soft and still needs explicit Gate B before CLOSE; without Gate B, soft
# sidecars are advisory for non-CLOSE actions and must not block separate
# current LIVE_READY entries on other pairs or horizons.

# 3. Write data/codex_trader_decision_response.json from the active decision branch
# If broker refresh made an older receipt stale, overwrite it with one current receipt.
# For TRADE / WAIT / REQUEST_EVIDENCE, include `twenty_minute_plan`.
# The live cadence is about 20 minutes; the plan must state the primary
# path, failure path, trigger, invalidation/cancel trigger, strongest
# counterargument, next-cycle check, and packet evidence refs. This is a
# receipt-depth requirement so the next cycle can audit the scenario tree;
# it is not a new market-risk threshold or permission to invent blockers.
# If current trader-owned pending entries consume portfolio capacity, either keep
# that pending basket explicitly or name verified trader pending ids in
# cancel_order_ids when replacing them with current MARKET participation.
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
#     authorization. Hard Gate A is M15/H4 close-confirmed BOS/CHOCH against
#     side, buffered invalidation_price hit with technical confirmation,
#     fresh thesis_evolution BROKEN / RECOMMEND_CLOSE, structural
#     position_management REVIEW_EXIT, or position_thesis invalidation-hit /
#     structural-break evidence with multi-TF confirmation. Softer sidecars
#     (adverse-entry-buffer-only or score-only position_thesis REVIEW_CLOSE,
#     non-structural position_management REVIEW_EXIT, or forecast_persistence RECOMMEND_CLOSE)
#     require `QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell, OR a fresh
#     `data/.operator_close_token` file. The receipt field
#     `operator_close_authorized: true` is advisory audit text only.
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
# Hard sidecar Gate A or explicit Gate B close evidence is priority work: do
# not choose TRADE, WAIT, REQUEST_EVIDENCE, PROTECT, or TIGHTEN_SL to sidestep
# it. If only soft Gate A exists and explicit Gate B is missing, the sidecar is
# advisory for non-CLOSE actions; keep TP/profit management active, and still
# evaluate short / medium / long horizon LIVE_READY entries. If choosing CLOSE
# from soft evidence, the verifier must surface `CLOSE_OPERATOR_AUTH_REQUIRED`.
# If hard Gate A or explicit Gate B is present, it must require a CLOSE receipt
# first. The default stance when no user instruction is present is HOLD / WAIT
# only when no fresh hard/authorized Gate A close sidecar and no current
# LIVE_READY lane exists — do not write a CLOSE receipt to "reduce risk" from a
# still-valid thesis.

# 4. Verify the receipt
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json

# 5. Run one gateway cycle immediately after every ACCEPTED receipt.
# Do not insert refresh, analysis, TP rebalance, projection, or thesis
# sidecar commands between verifier acceptance and this gateway handoff:
# the decision receipt is tied to the current broker snapshot + order intents,
# and extra work in between can make a tradeable receipt stale.
# The gateway cycle syncs data/execution_ledger.db before and after broker work
# and records live_order / position_execution receipts.
# This is mandatory even when the accepted action is WAIT /
# REQUEST_EVIDENCE / PROTECT: `autotrade-cycle` hands open positions to
# PositionManager and PositionProtectionGateway before considering fresh
# entry risk. Skipping the wrapper on WAIT leaves profitable hedge TPs,
# profit-lock stops, and other dependent-order protection stale.
QR_LIVE_ENABLED=1 ./scripts/run-autotrade-live.sh \
  --reuse-market-artifacts \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send

# 6. Protection sidecars — ONE consolidated command. Run after the single
# gateway cycle above has either sent, blocked, or recorded no action. It
# must not delay the handoff from verifier acceptance to LiveOrderGateway /
# PositionProtectionGateway, but it is part of every completed cycle with
# open positions.
#
# `cycle-sidecars` runs (canonical list: `cli._cycle_sidecar_steps`):
#   broker-snapshot → tp-rebalance → execution-ledger-sync → broker-snapshot
#   → profit-partial-close → verify-projections → position-thesis-check
#   → thesis-evolution-check → forecast-persistence-check → memory-health
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
# - memory-health BLOCK does not grant/deny a trade by itself;
#   trader-prompt-route reads it for the next cycle's routing.
QR_LIVE_ENABLED=1 PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli cycle-sidecars

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

- Final action: `TRADE`, `WAIT`, `REQUEST_EVIDENCE`, `PROTECT`, `TIGHTEN_SL`, `CLOSE`, or `CANCEL_PENDING`.
- Sent flag: `true`, `false`, or dry-run.
- Selected lane id(s), if any.
- Daily target progress from `data/daily_target_state.json`.
- `gpt-trader-decision` result and issue codes.
- Gateway result and report paths under `docs/*_report.md`.
- Execution ledger DB/report: `data/execution_ledger.db`, `docs/execution_ledger_report.md`.
- Verification ledger JSON/SQL/report: `data/verification_ledger.json`, `data/execution_ledger.db`, `docs/verification_ledger_report.md`.

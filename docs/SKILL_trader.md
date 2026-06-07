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
# Fresh entries need both executable forecast context and auditable telemetry.
# If forecast_history, projection_ledger, or execution_ledger sync is missing,
# generate-intents may diagnose the lane but must not emit LIVE_READY.
export QR_REQUIRE_FORECAST_FOR_LIVE="${QR_REQUIRE_FORECAST_FOR_LIVE:-1}"
export QR_REQUIRE_TELEMETRY_FOR_LIVE="${QR_REQUIRE_TELEMETRY_FOR_LIVE:-1}"

# 1. Route to the right prompt branch
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-prompt-route

# 2. Refresh evidence when routed there
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
# `--daily-risk-pct` sets the day's risk budget as % of starting NAV so the
# per-trade cap auto-scales with equity (feedback_use_nav_percent.md). 10%
# matches `target_return_pct` (campaign daily goal) and gives the basket
# validator enough room to hold the existing exposure plus add fresh lanes
# under SL-free (user directive 2026-05-11「市況読めばいいだけ」: the
# synthetic worst-case loss inside the validator is advisory; exits are
# market-derived, not loss-cap-derived).
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli import-legacy
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli mine-strategy
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli pair-charts --timeframes M1,M5,M15,M30,H1,H4,D --output data/pair_charts.json
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli cross-asset-snapshot
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli context-asset-charts
# Default flow snapshot reads spread only. OANDA orderBook/positionBook is
# opt-in via `--include-books` after book entitlement is confirmed.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli flow-snapshot
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli currency-strength
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli levels-snapshot
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli economic-calendar
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli cot-snapshot
# Provider-unconfigured option skew is written as a disabled optional artifact,
# not as repeated missing evidence.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli option-skew
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli market-context-matrix
# News is produced by a separate dedicated routine (`qr-news-digest`,
# Codex Desktop, hourly). That routine runs in the live runtime worktree
# at `/Users/tossaki/App/QuantRabbit-live/` and writes WebSearch-curated
# trader-perspective content to ignored runtime artifacts:
# `data/news_items.json`, `logs/news_digest.md`, and
# `logs/news_flow_log.md`. The trader cycle must NOT run
# `news-snapshot` here — that would replace the curated digest with raw
# RSS output. During the open FX week, `news-health --strict` below fails
# loud when the digest is stale, raw-RSS-only, missing required sections, or
# not reflected into `data/market_story_profile.json`. During the weekend
# guard window it accepts the paused scheduler when the weekend snapshot says
# `mode=paused`.
#
# Reflect the curated news into the live decision artifact before any intent
# pricing. `logs/news_digest.md` alone is not enough: trader_brain and
# gpt_trader read `data/market_story_profile.json`, so that derived
# profile must be newer than the news files. Write the side report under
# `data/` to avoid tracked `docs/*_report.md` drift during precheck
# refresh.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli mine-market-stories \
  --news-dir logs \
  --profile data/market_story_profile.json \
  --report data/market_story_report.md
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli news-health --strict
#
# daily-review: refresh `data/trader_overrides.json` from the last 24h
# of realized P&L plus structural pair/side underperformance, so
# trader_brain's Module C reads a current snapshot.
# Idempotent and fast (single SQLite read, no network), so safe to run
# every cycle. Expiry is JST midnight, so this also keeps the file
# rolling without manual intervention. `trader-prompt-route` treats a
# missing or expired file as a refresh requirement before target-open
# entry/verify routing, while existing-position management still keeps
# priority.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli daily-review
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli execution-ledger-sync
# TP rebalance is protection, not an entry decision. Run it once the
# current broker truth + pair_charts packet is fresh, before intent pricing
# and before a WAIT receipt can end the cycle. If it writes a dependent TP
# order, refresh broker truth immediately so generate-intents / GPT evidence
# sees the new broker order price, and sync the execution ledger so direct
# TP sidecar writes do not disappear from local audit reports. This is
# mandatory even when the final decision becomes WAIT; otherwise profitable
# existing TPs can sit stale.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli tp-rebalance
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
# Resolve expired forward-projection telemetry before intent pricing. Fresh
# entries cannot become LIVE_READY while old PENDING predictions are past
# their resolution window because the next trade would not be auditable.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli verify-projections
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli optimize-coverage
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli ai-attack-advice
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli learning-audit
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli verification-ledger-audit
# Predictive LIMIT timing evidence is generated before the decision receipt so
# the trader can compare market/pending participation against liquidity-sweep
# traps. Default is dry-run evidence; live placement still requires a separate
# explicit send path and gateway validation.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli generate-predictive-limits

# Position close sidecars are read-only prediction/thesis evidence. Refresh
# them before routing so a trapped position whose plus-recovery edge is gone
# reaches the position-management prompt when close authorization is hard or
# explicit. `position_management.json` from the previous gateway cycle is also
# read as bounded carry-forward Gate A evidence when it still marks the same
# open trader-owned trade REVIEW_EXIT. Fresh thesis_evolution BROKEN /
# RECOMMEND_CLOSE, structural position_management REVIEW_EXIT, and
# position_thesis invalidation-hit or structural-break evidence with multi-TF
# confirmation are hard standing loss-cut authorization. Adverse-entry-buffer
# only position_thesis evidence is soft and still needs explicit Gate B before
# CLOSE; without Gate B, soft sidecars are advisory for non-CLOSE actions and
# must not block separate current LIVE_READY entries on other pairs or horizons.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli position-thesis-check
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli thesis-evolution-check
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli forecast-persistence-check
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli memory-health

# Re-route after refresh. The refresh branch is not an end state: it must
# produce one current receipt and then proceed through verification + gateway.
# Ending the cycle after `generate-predictive-limits` leaves fresh evidence
# unused and is treated as incomplete.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-prompt-route

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
# Never autonomously emit CLOSE — the trader cannot decide on its own to close
# trader-owned positions. A CLOSE receipt requires Gate A plus the applicable
# Gate B path:
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
# A TRADE receipt must not list close_trade_ids. If the recovery edge is gone,
# close first and end that autotrade cycle as close-only. Refresh broker truth /
# intents on the next scheduled cycle, and only re-enter on a fresh LIVE_READY
# lane with a separate verified TRADE receipt. The automation must not re-enter
# in the same outer cycle after the close is sent, staged, or already satisfied.
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

# 6. Protection sidecars. Run after the single gateway cycle above has either
# sent, blocked, or recorded no action. These commands must not delay the
# handoff from verifier acceptance to LiveOrderGateway / PositionProtectionGateway,
# but they are part of every completed cycle with open positions.
#
# 6a. Dynamic TP rebalance: expand/contract TP on open positions as
# market regime shifts. Trader-owned and manual/tagless positions with
# an existing broker TP are eligible for TP-only management. Missing
# broker TP is preserved as a no-broker-TP runner unless
# QR_ENABLE_MISSING_TP_REPAIR=1 is explicitly set, or unless an
# already-profitable runner needs insurance because forecast horizon /
# confidence, next-session timing, or multiple technical exhaustion
# readings say MFE capture is at risk. SL-free positions keep
# stop_loss=None untouched; only takeProfit orders are written.
# Hysteresis + entry-side + safety-margin invariants prevent accidental fires.
# Refresh broker truth first so newly filled gateway trades are visible to the
# TP pass; refresh again after the TP pass so later sidecars see the updated
# dependent order state.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli tp-rebalance
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json

# 6b. Profit partial close: when trader-owned or manual/tagless
# exposure is already in profit and has crossed the next ATR-derived
# milestone, close a market-derived fraction and keep the remaining
# units as a runner. This is profit-side only; it never realizes a loss
# from adverse P/L and never writes a stop-loss to manual/tagless
# positions.
# Same trade/milestone is persisted in
# data/profit_partial_close_state.json after a successful send to avoid
# repeat partial closes on the same band.
QR_LIVE_ENABLED=1 PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli profit-partial-close --send --confirm-live

# 6c. Verify pending forward-projection predictions against OANDA
# prices. Resolves PENDING → HIT/MISS/TIMEOUT in
# data/projection_ledger.jsonl. The next trader cycle reads the
# rolling hit-rate from this ledger and calibrates the projection
# layer's confidence weights — self-improving feedback loop.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli verify-projections

# 6d. Position thesis check — apply the full prediction stack to each
# open position. Emits data/position_thesis_report.json with per-position
# EXTEND / HOLD / REVIEW_CLOSE verdicts and score breakdowns. The GPT
# trader reads a fresh REVIEW_CLOSE as Gate A evidence for CLOSE decisions.
# It is hard standing authorization only when the report records adverse
# technical loss / invalidation-hit evidence plus multi-TF confirmation;
# score-only reviews still require explicit env/token Gate B unless a separate
# hard Gate A path also exists.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli position-thesis-check

# 6e. Thesis evolution check (2026-05-15, user directive: 「どの視点で
# エントリーしたのか、時間がたって今のポジ状況はエントリーしたときと
# 市況が変わってないか」). Reads the entry-time thesis recorded at
# fill time from data/entry_thesis_ledger.jsonl, refreshes the current
# pair-level forecast from broker_snapshot + pair_charts when needed,
# and compares against data/forecast_history.jsonl.
# Emits data/thesis_evolution_report.json with per-position
# STILL_VALID / WEAKENED / BROKEN + HOLD / EXTEND / RECOMMEND_CLOSE.
# A fresh BROKEN / RECOMMEND_CLOSE is hard Gate A and satisfies standing
# structural loss-cut authorization. Pairs with TP-なし・SL-なし mode:
# trader holds runners on STILL_VALID, considers close on BROKEN.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli thesis-evolution-check

# 6f. Forecast persistence check — N-cycle consistency rule. Reads
# data/forecast_history.jsonl (refreshed from broker_snapshot +
# pair_charts even when no fresh-entry lane is scored) and asks: are
# the last N forecasts for this pair pointing AGAINST
# the position (≥ QR_FORECAST_FLIP_PERSISTENCE=3 cycles) or have they
# all gone RANGE/UNCLEAR (≥ QR_FORECAST_RANGE_PERSISTENCE=5 cycles)?
# Either pattern → RECOMMEND_CLOSE. Aligned ≥3 cycles → EXTEND.
# Single-cycle noise does NOT trigger close. Output goes to
# data/forecast_persistence_report.json. Pairs with the
# thesis-evolution-check verdict; a fresh RECOMMEND_CLOSE is Gate A
# evidence only and still needs explicit Gate B operator authorization.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli forecast-persistence-check

# 6g. Memory health check — aggregate short, medium, long, and position
# memory into data/memory_health.json. BLOCK does not grant/deny a trade by
# itself; trader-prompt-route reads it and sends target-open entry/verify work
# back to refresh when forecast_history, projection_ledger, execution_ledger,
# learning_audit, strategy_profile, or entry_thesis_ledger has a hole.
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli memory-health

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

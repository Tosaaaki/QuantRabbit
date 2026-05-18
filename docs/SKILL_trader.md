# QuantRabbit Trader Runtime

This is the trader entry prompt. Keep it small. The branch prompts under
`docs/trader_prompts/` carry the task-specific instructions.

## Load Order

1. Read `docs/AGENT_CONTRACT.md`.
2. Read `docs/trader_prompts/00_router.md`.
3. Ask Python which branch is active:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli trader-prompt-route
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
# Deterministic REVIEW_EXIT is advisory by default in SL-free live mode.
# Full closes must pass the gpt_trader close discipline and operator token.
export QR_DISABLE_AUTO_CLOSE="${QR_DISABLE_AUTO_CLOSE:-1}"
# Broker-side SL/trailing are opt-in only. The live default keeps NEW
# entries SL-free because widened broker SLs were still harvested by
# thin-session noise on 2026-05-13.
export QR_NEW_ENTRY_INITIAL_SL="${QR_NEW_ENTRY_INITIAL_SL:-0}"
export QR_DISABLE_TRAILING_SL="${QR_DISABLE_TRAILING_SL:-1}"

# 1. Route to the right prompt branch
PYTHONPATH=src python3 -m quant_rabbit.cli trader-prompt-route

# 2. Refresh evidence when routed there
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
# `--daily-risk-pct` sets the day's risk budget as % of starting NAV so the
# per-trade cap auto-scales with equity (feedback_use_nav_percent.md). 10%
# matches `target_return_pct` (campaign daily goal) and gives the basket
# validator enough room to hold the existing exposure plus add fresh lanes
# under SL-free (user directive 2026-05-11「市況読めばいいだけ」: the
# synthetic worst-case loss inside the validator is advisory; exits are
# market-derived, not loss-cap-derived).
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src python3 -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src python3 -m quant_rabbit.cli pair-charts --timeframes M1,M5,M15,M30,H1,H4,D --output data/pair_charts.json
PYTHONPATH=src python3 -m quant_rabbit.cli cross-asset-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli flow-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli currency-strength
PYTHONPATH=src python3 -m quant_rabbit.cli levels-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli economic-calendar
PYTHONPATH=src python3 -m quant_rabbit.cli cot-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli option-skew
# News is produced by a separate dedicated routine (`qr-news-digest`,
# Claude Desktop, hourly at :23 JST). That routine runs in the dev
# worktree at `/Users/tossaki/App/QuantRabbit/` and writes
# WebSearch-curated trader-perspective content to its own
# `logs/news_digest.md` + `logs/news_flow_log.md`. The live worktree's
# `logs/news_digest.md` is symlinked to the dev path so this cycle sees
# the dedicated routine's curated digest. The trader cycle must NOT run
# `news-snapshot` here — that would write raw RSS output through the
# symlink and clobber the curated digest. If the digest goes stale (the
# routine fails or is paused), `strategy/market_story.py` will surface
# missing-evidence rationale on lanes but will not crash.
#
# daily-review: refresh `data/trader_overrides.json` from the last 24h
# of realized P&L so trader_brain's Module C reads a current snapshot.
# Idempotent and fast (single SQLite read, no network), so safe to run
# every cycle. Expiry is JST midnight, so this also keeps the file
# rolling without manual intervention.
PYTHONPATH=src python3 -m quant_rabbit.cli daily-review
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src python3 -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage
PYTHONPATH=src python3 -m quant_rabbit.cli ai-attack-advice
# Predictive LIMIT timing evidence is generated before the decision receipt so
# the trader can compare market/pending participation against liquidity-sweep
# traps. Default is dry-run evidence; live placement still requires a separate
# explicit send path and gateway validation.
PYTHONPATH=src python3 -m quant_rabbit.cli generate-predictive-limits

# Re-route after refresh. The refresh branch is not an end state: it must
# produce one current receipt and then proceed through verification + gateway.
# Ending the cycle after `generate-predictive-limits` leaves fresh evidence
# unused and is treated as incomplete.
PYTHONPATH=src python3 -m quant_rabbit.cli trader-prompt-route

# 3. Write data/codex_trader_decision_response.json from the active decision branch
# If broker refresh made an older receipt stale, overwrite it with one current receipt.
# If current trader-owned pending entries consume portfolio capacity, either keep
# that pending basket explicitly or name verified trader pending ids in
# cancel_order_ids when replacing them with current MARKET participation.
# If the action is CANCEL_PENDING, list only current trader-owned pending entry
# ids in cancel_order_ids; the gateway cycle cancels verified ids and sends no
# fresh entry in that same cycle.
#
# CLOSE discipline (AGENT_CONTRACT §10, feedback_no_unilateral_close.md):
# Never autonomously emit CLOSE — the trader cannot decide on its own to close
# trader-owned positions. A CLOSE receipt (or a TRADE receipt that lists
# close_trade_ids) requires BOTH:
#   - Gate A: market evidence — pair_charts shows BOS/CHOCH against the
#     position side on M15 or H4, OR `invalidation_price` + `invalidation_tf`
#     in the receipt with broker truth confirming the level has traded.
#   - Gate B: operator authorization — require either
#     `QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell, OR a fresh
#     `data/.operator_close_token` file. The receipt field
#     `operator_close_authorized: true` is advisory audit text only.
# If either gate fails, `gpt-trader-decision` REJECTs the receipt with
# `CLOSE_THESIS_STILL_VALID` or `CLOSE_OPERATOR_AUTH_REQUIRED`. The default
# stance when no user instruction is present is HOLD / WAIT — do not write a
# CLOSE receipt to "reduce risk" from a still-valid thesis.

# 4. Verify the receipt
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json

# 4b. Dynamic TP rebalance: expand/contract TP on open positions as
# market regime shifts. Trader-owned and manual/tagless positions with
# an existing broker TP are eligible for TP-only management. Missing
# broker TP is preserved as a no-broker-TP runner unless
# QR_ENABLE_MISSING_TP_REPAIR=1 is explicitly set, or unless an
# already-profitable runner needs insurance because forecast horizon /
# confidence, next-session timing, or multiple technical exhaustion
# readings say MFE capture is at risk. SL-free positions keep
# stop_loss=None untouched; only takeProfit orders are written.
# Hysteresis + entry-side + safety-margin invariants prevent accidental fires.
PYTHONPATH=src python3 -m quant_rabbit.cli tp-rebalance

# 4b2. Profit partial close: when trader-owned or manual/tagless
# exposure is already in profit and has crossed the next ATR-derived
# milestone, close a market-derived fraction and keep the remaining
# units as a runner. This is profit-side only; it never realizes a loss
# from adverse P/L and never writes a stop-loss to manual/tagless
# positions.
# Same trade/milestone is persisted in
# data/profit_partial_close_state.json after a successful send to avoid
# repeat partial closes on the same band.
PYTHONPATH=src python3 -m quant_rabbit.cli profit-partial-close --send --confirm-live

# 4d. Verify pending forward-projection predictions against OANDA
# prices. Resolves PENDING → HIT/MISS/TIMEOUT in
# data/projection_ledger.jsonl. The next trader cycle reads the
# rolling hit-rate from this ledger and calibrates the projection
# layer's confidence weights — self-improving feedback loop.
PYTHONPATH=src python3 -m quant_rabbit.cli verify-projections

# 4e. Position thesis check — apply the full prediction stack to each
# open position. Emits data/position_thesis_report.json with per-position
# EXTEND / HOLD / REVIEW_CLOSE verdicts and score breakdowns. The GPT
# trader reads this as evidence for CLOSE decisions; Gate A/B still
# required to actually close.
PYTHONPATH=src python3 -m quant_rabbit.cli position-thesis-check

# 4e2. Thesis evolution check (2026-05-15, user directive: 「どの視点で
# エントリーしたのか、時間がたって今のポジ状況はエントリーしたときと
# 市況が変わってないか」). Reads the entry-time thesis recorded at
# fill time from data/entry_thesis_ledger.jsonl and compares against
# the latest per-cycle forecast in data/forecast_history.jsonl.
# Emits data/thesis_evolution_report.json with per-position
# STILL_VALID / WEAKENED / BROKEN + HOLD / EXTEND / RECOMMEND_CLOSE.
# INFORMATION ONLY — Gate A/B still required for actual close. Pairs
# with TP-なし・SL-なし mode: trader holds runners on STILL_VALID,
# considers manual close on BROKEN.
PYTHONPATH=src python3 -m quant_rabbit.cli thesis-evolution-check

# 4e3. Forecast persistence check — N-cycle consistency rule. Reads
# data/forecast_history.jsonl (written every cycle by trader_brain)
# and asks: are the last N forecasts for this pair pointing AGAINST
# the position (≥ QR_FORECAST_FLIP_PERSISTENCE=3 cycles) or have they
# all gone RANGE/UNCLEAR (≥ QR_FORECAST_RANGE_PERSISTENCE=5 cycles)?
# Either pattern → RECOMMEND_CLOSE. Aligned ≥3 cycles → EXTEND.
# Single-cycle noise does NOT trigger close. Output goes to
# data/forecast_persistence_report.json. Pairs with the
# thesis-evolution-check verdict; the trader/operator decides whether
# to close manually + Gate A/B. INFORMATION ONLY.
PYTHONPATH=src python3 -m quant_rabbit.cli forecast-persistence-check

# 4c. adverse-partial-close is DISABLED 2026-05-14:
# The module closed 50% of 471020 AUD/JPY SHORT for -2,516 JPY based
# on adverse-pips threshold, violating feedback_market_over_risk_budget.md
# (「SL-free 下で CLOSE 判断は構造/MTF/thesis のみ。含み損%/JPY は
# 遅行指標で判断材料にしない」). Locking in losses on adverse-pip
# threshold IS the anti-pattern SL-free was designed to avoid.
# Module file kept in src/quant_rabbit/strategy/adverse_partial_close.py
# for reference + future "structural-trigger only" rewrite; CLI line
# below is commented out so no cycle invokes it.
# PYTHONPATH=src python3 -m quant_rabbit.cli adverse-partial-close

# 5. Run one gateway cycle only
# The gateway cycle syncs data/execution_ledger.db before and after broker work
# and records live_order / position_execution receipts.
./scripts/run-autotrade-live.sh \
  --reuse-market-artifacts \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send
```

## End Report

- Final action: `TRADE`, `WAIT`, `REQUEST_EVIDENCE`, `PROTECT`, `TIGHTEN_SL`, `CLOSE`, or `CANCEL_PENDING`.
- Sent flag: `true`, `false`, or dry-run.
- Selected lane id(s), if any.
- Daily target progress from `data/daily_target_state.json`.
- `gpt-trader-decision` result and issue codes.
- Gateway result and report paths under `docs/*_report.md`.
- Execution ledger DB/report: `data/execution_ledger.db`, `docs/execution_ledger_report.md`.

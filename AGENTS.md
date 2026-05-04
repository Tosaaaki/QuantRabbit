# QuantRabbit vNext Agent Instructions

## Package Manager
- None. Use system Python with `PYTHONPATH=src`.
- Test with `PYTHONPATH=src python3 -m unittest discover -s tests -v`.

## Commit Attribution
- AI commits MUST include:
```
Co-Authored-By: Codex <noreply@openai.com>
```

## Prime Directive
- Build from broker truth outward.
- Enforce broker gateway and risk contracts in code before prompt or strategy-prose optimization.
- Express discretionary autonomy as executable receipts, not vague market prose.
- Default to dry-run/read-only behavior unless live gates are explicitly satisfied.

## Product Goal
- Build an autonomous professional FX trader where GPT reasons, compares evidence, and chooses trades through executable system contracts.
- The completion target is a trader that can pursue 10% of that day's starting equity as daily profit without bypassing risk gates.
- Treat the 10% daily target as a mandatory product KPI, campaign metric, backtest/live-simulation benchmark, and risk-bounded execution objective.
- If the system cannot reach the 10% daily target in evidence review, backtests, simulation, or live dry-runs, treat the gap as a product blocker and improve strategy, timing, risk geometry, or execution quality.
- Never describe the 10% daily target as guaranteed return, and never use it as permission to bypass broker truth, protection, or risk validation.
- The trader must behave like a professional: wait when edge is absent, size from risk geometry, protect exposure, and explain decisions in receipts.

## Archive Evolution Requirements
- vNext is an evolution of the archived QuantRabbit trader, not a replacement with a separate bot architecture.
- Preserve the archived winning operating model: Codex automation is the discretionary GPT trader; Python tools calculate, fetch broker truth, validate risk, send through gateways, and write receipts.
- The canonical active trader automation is `QR vNext Trader`, which must read and execute `docs/SKILL_trader.md`; do not create duplicate trader automations for the same workspace.
- Do not move live discretion into an API-key model call inside QuantRabbit. In production, GPT means the Codex automation model itself.
- Keep the old system's useful trader behavior, market-reading cadence, and lessons, but convert unsafe legacy mechanics into vNext tests, risk contracts, receipts, and gateway calls before reuse.
- Prefer the simplest archive-compatible flow: Codex decides, `gpt-trader-decision` verifies the Codex decision receipt, and `autotrade-cycle` sends only through `LiveOrderGateway` or manages exposure through `PositionProtectionGateway`.
- Added structure must reduce known archive failure modes such as stale broker state, unprotected exposure, direct OANDA writes, duplicated orders, stale pending ids, missing audit receipts, and risk math drift.

## Autonomous Pro Trader Requirements
- GPT is the discretionary decision layer, but every decision must be grounded in structured broker state, mined history, current market context, and risk geometry.
- The trader must decide whether to trade, wait, cancel, protect, tighten, close, or request more evidence.
- No-trade is a valid professional decision when the edge, spread, timing, liquidity, narrative, or risk/reward is inadequate.
- Every trade decision must compare historical analogs, current regime, market story, chart structure, campaign role, open exposure, pending orders, and invalidation distance.
- Every decision receipt must state selected lane, rejected alternatives, thesis, method, narrative, chart story, invalidation, TP, SL, units, expected reward, worst-case loss, and reason for action.
- The system must separate observation, reasoning, risk validation, order staging, order sending, position protection, and post-trade learning.
- The trader must learn from wins, losses, missed edges, bad entries, premature exits, and protection repairs without allowing memory to override live risk checks.
- The trader must preserve a professional audit trail: broker snapshots, intents, dry-run receipts, live-stage receipts, execution reports, protection actions, and post-trade reviews.
- The trader must continuously look for root-cause bugs in strategy logic, risk math, stale broker state, duplicated exposure, missing protection, and legacy assumptions.

## Daily 10% Campaign
- Each trading day starts with a recorded starting equity and a target profit equal to 10% of that starting equity.
- Campaign planning must translate the 10% target into candidate lanes, required reward/risk, maximum acceptable loss, and exposure budget.
- The system is not complete while it merely avoids bad trades; it must actively search for valid high-quality opportunities capable of reaching the daily target.
- Campaign pressure must become bounded execution behavior: better timing, better filtering, stronger confluence, clearer invalidation, and disciplined sizing.
- When the 10% target is missed, the system must produce a gap report covering missed setup, blocked trade, market absence, risk rejection, execution cost, timing error, or strategy weakness.
- Reaching the daily target must trigger protection-first behavior; do not keep adding risk just because more trades are possible.

## System Requirements
- OANDA entry orders go only through `LiveOrderGateway`.
- OANDA position changes go only through `PositionProtectionGateway`.
- No independent OANDA write helpers, direct order scripts, scheduler bypasses, or prompt-only workarounds.
- Runtime automation reconciles broker truth and calls gateways; it does not create strategy outside executable contracts.
- Learning memory is advisory; it cannot silently force entries, suppress exits, or resize trades.
- Every live order intent must include entry, TP, SL, worst-case JPY loss, reward/risk, spread, owner, units, and lane id.
- Every executable lane must include thesis, method, narrative, chart story, invalidation, TP, SL, and units.
- Do not reduce entry selection to a single score threshold.
- Daily 10% campaign planning is a target ledger, not permission to force weak trades or exceed risk gates.

## Autotrade Contract
- `autotrade-cycle --send` is the live automation entrypoint.
- Flat-account entry loop: `BrokerSnapshot` -> `IntentGenerator` -> `TraderBrain` -> `LiveOrderGateway`.
- Exposure-management loop: `BrokerSnapshot` -> `TraderBrain` -> `PositionManager` -> `PositionProtectionGateway`.
- `gpt-trader-decision` verifies a Codex-created GPT decision receipt by default; `autotrade-cycle --use-gpt-trader --gpt-decision-response ...` may hand off only an ACCEPTED `TRADE` from a deterministic prefilter lane.
- In automation, GPT means the Codex automation model itself. Do not use API-key GPT paths (`QR_OPENAI_API_KEY` or `OPENAI_API_KEY`) for live operation.
- Unprotected, external/manual, over-budget, or pending-entry exposure blocks fresh entries.
- Protected trader-owned exposure may add only through portfolio risk validation; total worst-case loss must stay inside the active exposure budget.
- Pending orders vetoed by the current `TraderBrain` can be canceled before the next cycle.
- Trader decisions compare mined history, market story, campaign role, narrative risk, current broker state, risk geometry, and live exposure.
- Portfolio Director records which desk wins and why when trader desks disagree.

## Live Trading Gates
- Real entry sends require `QR_LIVE_ENABLED=1`, `--send`, fresh broker truth, explicit lane id, `RiskEngine.validate(..., for_live_send=True)`, and strategy-profile validation.
- `stage-live-order` stages by default; real send additionally requires `--send --confirm-live`.
- Use explicit `QR_OANDA_TOKEN`, `QR_OANDA_ACCOUNT_ID`, and `QR_OANDA_BASE_URL`.
- Local OANDA credentials are recorded in `.env.local` at repo root with those `QR_OANDA_*` keys; do not print their values.
- `OandaReadOnlyClient` loads `.env.local` automatically when process env vars are absent. `QR_OANDA_ENV_FILE=/path/to/file` overrides the lookup for tests or alternate environments.
- Do not read legacy `config/env.toml` into vNext.
- Manual, tagless, or broker-synced exposure blocks new entries until adopted or closed.

## Position Protection
- Missing TP/SL is a repair requirement.
- Profitable protected positions can tighten SL to break-even or better.
- Contradicted trader-owned positions can be closed.
- Existing SL cannot be widened.
- Existing TP is not moved by the protection gateway.

## Strategy Evidence
- Run `PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy` before strategy work.
- Mine the archived trade logs, trade history, journals, strategy memory, market stories, and audit logs before changing strategy behavior.
- Treat `logs/live_trade_log.txt`, `logs/trader_journal.jsonl`, `logs/s_hunt_ledger.jsonl`, `logs/audit_history.jsonl`, structured memory, and daily handoffs as primary evidence.
- Extract repeatable lessons from profitable trades, losing trades, near misses, manual interventions, execution failures, and broker-state mismatches.
- Reuse strong legacy mechanisms only after they are converted into vNext tests, receipts, risk checks, or read-only evidence pipelines.
- `risk-dry-run` reads `data/strategy_profile.json` when present.
- `promote-receipts` feeds passing dry-run receipts back into `data/strategy_profile.json`.
- `RISK_REPAIR_CANDIDATE` can reopen only when the current receipt passes risk geometry.
- `MINE_MISSED_EDGE` can reopen only from a LIMIT or STOP-ENTRY receipt.
- `BLOCK_UNTIL_NEW_EVIDENCE` is never auto-promoted.

## Legacy Evidence
- Archive root: `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z`.
- Treat the archive as read-only evidence and a source of mined trader experience.
- Manifest: `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z/ARCHIVE_MANIFEST_20260430T151527Z.md`.
- See `ARCHIVE_POINTER.md` for archived log locations.
- Do not copy old schedulers, automation prompts, or order helpers into vNext wholesale.

## Current Commands
```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories
PYTHONPATH=src python3 -m quant_rabbit.cli replay-backtest --start-balance 222781
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance 222781
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --start-balance 222781 --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision --snapshot data/broker_snapshot.json --decision-response data/codex_trader_decision_response.json
PYTHONPATH=src python3 -m quant_rabbit.cli risk-dry-run --intent intent.json --snapshot snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli promote-receipts
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage
PYTHONPATH=src python3 -m quant_rabbit.cli stage-live-order --lane-id 'failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE'
PYTHONPATH=src python3 -m quant_rabbit.cli autotrade-cycle --use-gpt-trader --gpt-decision-response data/codex_trader_decision_response.json
PYTHONPATH=src python3 -m quant_rabbit.cli replay-execution --prices data/quote_path.json --target-jpy 22278
PYTHONPATH=src python3 -m quant_rabbit.cli learn-post-trade --outcome outcome.json
PYTHONPATH=src python3 -m quant_rabbit.cli certify-dry-run
QR_LIVE_ENABLED=1 PYTHONPATH=src python3 -m quant_rabbit.cli autotrade-cycle --send
```

## Acceptance Bar
- Add a failing regression test for known legacy failure modes.
- Add a passing test for new behavior.
- Document the live-risk failure mode for accepted rebuild components.
- Preserve dry-run receipt paths for new execution features.
- No live side effect unless explicitly enabled.
- Never promise guaranteed returns; convert campaign pressure into bounded, testable execution behavior.

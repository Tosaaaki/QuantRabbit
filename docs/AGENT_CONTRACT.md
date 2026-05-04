# QuantRabbit vNext — Agent Contract

This file is the **single source of truth** for both Codex and Claude. The repo's `CLAUDE.md` and `AGENTS.md` are 1-line stubs that point here. Edit only this file when changing the contract; never edit the stubs.

If you are an automation reading this for runtime, also read `docs/SKILL_trader.md` for the executable cycle.

---

## 1. Document Map (導線)

| Purpose | File |
|---|---|
| Single source of truth (this file) | `docs/AGENT_CONTRACT.md` |
| Claude Code auto-load stub | `CLAUDE.md` |
| Codex auto-load stub | `AGENTS.md` |
| Trader runtime playbook (one cycle) | `docs/SKILL_trader.md` |
| Live send wrapper | `scripts/run-autotrade-live.sh` |
| Codex scheduled task | `~/.claude/scheduled-tasks/<codex-task>/` (Codex-managed) |
| Claude scheduled task | `~/.claude/scheduled-tasks/trader/` |
| Legacy archive root | `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z` |
| Archive manifest | `…/ARCHIVE_MANIFEST_20260430T151527Z.md` |
| Archive log pointer | `ARCHIVE_POINTER.md` |
| Latest reports | `docs/*_report.md` |

If a sub-doc disagrees with this contract, **this contract wins**. Update the sub-doc.

---

## 2. Operator Model (Codex / Claude switchable)

The discretionary trader role can be filled by **either** the Codex automation or the Claude scheduled task. They share the same playbook (`docs/SKILL_trader.md`), the same gateways, and the same broker account.

Rules:

- **Exactly one operator enabled at a time.** Never both. Two operators on the same OANDA account = duplicate orders, double-counted exposure, contradictory protection actions.
- **Switch by enabling/disabling the scheduled task** on each side. No code changes required to switch.
- **Both operators read the same `docs/SKILL_trader.md`.** Changes there affect both.
- **Codex remains the production default** unless the user explicitly switches to Claude. Claude is a peer operator, not a replacement.
- **Operator identity must appear in every receipt** (`operator: "codex"` or `operator: "claude"`) so the audit trail stays clean across switches.
- **Handoff discipline.** When switching operators, the incoming operator must (1) refresh `broker-snapshot`, (2) read the most recent `data/codex_trader_decision_response.json` or equivalent, (3) inherit any open trader-owned positions and pending orders, (4) not cancel the other operator's pending orders without an explicit reason recorded in the next decision receipt.

In automation, "GPT" means whichever model (Codex or Claude) is currently the enabled operator. **Do not call any API-key model path from QuantRabbit code** (`QR_OPENAI_API_KEY`, `OPENAI_API_KEY`). Live discretion lives in the scheduled-task model, not in a library call.

---

## 3. Prime Directive

- Build from broker truth outward.
- Enforce broker gateway and risk contracts in code before prompt or strategy-prose optimization.
- Express discretionary autonomy as executable receipts, not vague market prose.
- Default to dry-run / read-only behavior unless live gates are explicitly satisfied.

---

## 4. Product Goal

- Build an autonomous professional FX trader where the operator (Codex or Claude) reasons, compares evidence, and chooses trades through executable system contracts.
- The completion target is a trader that can pursue **10 % of that day's starting equity as daily profit** without bypassing risk gates.
- Treat the 10 % daily target as a mandatory product KPI, campaign metric, backtest / live-simulation benchmark, and risk-bounded execution objective.
- If the system cannot reach 10 % in evidence review, backtests, simulation, or live dry-runs, treat the gap as a product blocker and improve strategy, timing, risk geometry, or execution quality.
- **Never describe the 10 % daily target as a guaranteed return**, and never use it as permission to bypass broker truth, protection, or risk validation.
- The trader must behave like a professional: wait when edge is absent, size from risk geometry, protect exposure, and explain decisions in receipts.

---

## 5. Daily 10 % Campaign

- Each trading day starts with a recorded starting equity and a target profit equal to 10 % of that starting equity.
- Starting equity is sourced from the latest broker snapshot via `daily-target-state` (OANDA `/v3/accounts/{id}/summary` → `balance`). Do **not** hardcode JPY figures.
- Campaign planning translates the 10 % target into candidate lanes, required reward / risk, maximum acceptable loss, and exposure budget.
- The system is not complete while it merely avoids bad trades — it must actively search for valid high-quality opportunities capable of reaching the daily target.
- Campaign pressure must become **bounded execution behavior**: better timing, better filtering, stronger confluence, clearer invalidation, disciplined sizing.
- When the 10 % target is missed, produce a gap report (missed setup, blocked trade, market absence, risk rejection, execution cost, timing error, or strategy weakness).
- Reaching the daily target triggers **protection-first** behavior; do not keep adding risk just because more trades are possible.

---

## 6. Autonomous Pro Trader Requirements

- The operator is the discretionary decision layer, but every decision must be grounded in structured broker state, mined history, current market context, and risk geometry.
- The trader must decide whether to TRADE, WAIT, CANCEL, PROTECT, TIGHTEN, CLOSE, or REQUEST_EVIDENCE.
- **No-trade is a valid professional decision** when edge, spread, timing, liquidity, narrative, or risk / reward is inadequate.
- Every trade decision compares historical analogs, current regime, market story, chart structure, campaign role, open exposure, pending orders, and invalidation distance.
- Every decision receipt states: selected lane, rejected alternatives, thesis, method, narrative, chart story, invalidation, TP, SL, units, expected reward, worst-case loss, reason for action, and `operator`.
- The system separates observation, reasoning, risk validation, order staging, order sending, position protection, and post-trade learning.
- The trader learns from wins, losses, missed edges, bad entries, premature exits, and protection repairs **without allowing memory to override live risk checks**.
- Maintain a professional audit trail: broker snapshots, intents, dry-run receipts, live-stage receipts, execution reports, protection actions, post-trade reviews.
- Continuously root-cause bugs in strategy logic, risk math, stale broker state, duplicated exposure, missing protection, and legacy assumptions.

---

## 7. System Requirements

- OANDA entry orders go **only** through `LiveOrderGateway`.
- OANDA position changes go **only** through `PositionProtectionGateway`.
- No independent OANDA write helpers, direct order scripts, scheduler bypasses, or prompt-only workarounds.
- Runtime automation reconciles broker truth and calls gateways; it does not create strategy outside executable contracts.
- Learning memory is **advisory**; it cannot silently force entries, suppress exits, or resize trades.
- Every live order intent includes: entry, TP, SL, worst-case JPY loss, reward / risk, spread, owner, units, lane id.
- Every executable lane includes: thesis, method, narrative, chart story, invalidation, TP, SL, units.
- Do **not** reduce entry selection to a single score threshold.
- Daily 10 % campaign planning is a target ledger, not permission to force weak trades or exceed risk gates.

---

## 8. Autotrade Contract

- `autotrade-cycle --send` is the live automation entrypoint.
- Flat-account entry loop: `BrokerSnapshot` → `IntentGenerator` → `TraderBrain` → `LiveOrderGateway`.
- Exposure-management loop: `BrokerSnapshot` → `TraderBrain` → `PositionManager` → `PositionProtectionGateway`.
- `gpt-trader-decision` verifies the operator's decision receipt by default.
  - `autotrade-cycle --use-gpt-trader --gpt-decision-response …` may hand off **only an ACCEPTED `TRADE`** from a deterministic prefilter lane.
- Unprotected, external / manual, over-budget, or pending-entry exposure **blocks fresh entries**.
- Protected trader-owned exposure may add only through portfolio risk validation; total worst-case loss must stay inside the active exposure budget.
- Pending orders vetoed by the current `TraderBrain` can be canceled before the next cycle.
- Trader decisions compare mined history, market story, campaign role, narrative risk, current broker state, risk geometry, and live exposure.
- Portfolio Director records which desk wins and why when trader desks disagree.

---

## 9. Live Trading Gates

- Real entry sends require **all** of:
  - `QR_LIVE_ENABLED=1`
  - `--send`
  - Fresh broker truth (recent `data/broker_snapshot.json`)
  - Explicit lane id
  - `RiskEngine.validate(..., for_live_send=True)` passes
  - Strategy-profile validation passes
- `stage-live-order` stages by default; real send additionally requires `--send --confirm-live`.
- Use explicit env vars: `QR_OANDA_TOKEN`, `QR_OANDA_ACCOUNT_ID`, `QR_OANDA_BASE_URL`.
- Local OANDA credentials live in `.env.local` at repo root with `QR_OANDA_*` keys. **Never print their values.**
- `OandaReadOnlyClient` loads `.env.local` automatically when process env vars are absent. `QR_OANDA_ENV_FILE=/path/to/file` overrides the lookup for tests or alternate environments.
- Do **not** read legacy `config/env.toml` into vNext.
- Manual, tagless, or broker-synced exposure **blocks new entries** until adopted or closed.

---

## 10. Position Protection

- Missing TP / SL is a **repair requirement** (not optional).
- Profitable protected positions can tighten SL to break-even or better.
- Contradicted trader-owned positions can be closed.
- **Existing SL cannot be widened.**
- **Existing TP is not moved by the protection gateway.**

---

## 11. Strategy Evidence

- Run `PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy` before strategy work.
- Mine archived trade logs, trade history, journals, strategy memory, market stories, and audit logs **before** changing strategy behavior.
- Treat as primary evidence: `logs/live_trade_log.txt`, `logs/trader_journal.jsonl`, `logs/s_hunt_ledger.jsonl`, `logs/audit_history.jsonl`, structured memory, daily handoffs.
- Extract repeatable lessons from profitable trades, losing trades, near misses, manual interventions, execution failures, and broker-state mismatches.
- Reuse strong legacy mechanisms only after they are converted into vNext tests, receipts, risk checks, or read-only evidence pipelines.
- `risk-dry-run` reads `data/strategy_profile.json` when present.
- `promote-receipts` feeds passing dry-run receipts back into `data/strategy_profile.json`.
- `RISK_REPAIR_CANDIDATE` can reopen only when the current receipt passes risk geometry.
- `MINE_MISSED_EDGE` can reopen only from a LIMIT or STOP-ENTRY receipt.
- `BLOCK_UNTIL_NEW_EVIDENCE` is **never auto-promoted**.

---

## 12. Archive Evolution

- vNext is an **evolution** of the archived QuantRabbit trader, not a replacement with a separate bot architecture.
- Preserve the archived winning operating model: the operator (Codex or Claude) is the discretionary trader; Python tools calculate, fetch broker truth, validate risk, send through gateways, and write receipts.
- The canonical active trader automation reads and executes `docs/SKILL_trader.md`. Do not create duplicate trader automations within the same operator side.
- Keep the old system's useful trader behavior, market-reading cadence, and lessons, but convert unsafe legacy mechanics into vNext tests, risk contracts, receipts, and gateway calls before reuse.
- Prefer the simplest archive-compatible flow: operator decides → `gpt-trader-decision` verifies the decision receipt → `autotrade-cycle` sends only through `LiveOrderGateway` or manages exposure through `PositionProtectionGateway`.
- Added structure must reduce known archive failure modes: stale broker state, unprotected exposure, direct OANDA writes, duplicated orders, stale pending ids, missing audit receipts, risk math drift.

### Legacy archive

- Archive root: `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z`
- Manifest: `…/ARCHIVE_MANIFEST_20260430T151527Z.md`
- See `ARCHIVE_POINTER.md` for archived log locations.
- Treat the archive as **read-only evidence**.
- Do not copy old schedulers, automation prompts, or order helpers into vNext wholesale.

---

## 13. Engineering Conventions

### Package manager
- None. Use system Python with `PYTHONPATH=src`.
- Test with `PYTHONPATH=src python3 -m unittest discover -s tests -v`.

### Commit attribution
AI commits MUST include the line for the operator who authored the change:

- Codex commits:
  ```
  Co-Authored-By: Codex <noreply@openai.com>
  ```
- Claude commits:
  ```
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

If both contributed in the same commit, include both lines.

### Acceptance bar
- Add a **failing regression test** for known legacy failure modes.
- Add a **passing test** for new behavior.
- Document the live-risk failure mode for accepted rebuild components.
- Preserve dry-run receipt paths for new execution features.
- **No live side effect unless explicitly enabled.**
- Never promise guaranteed returns; convert campaign pressure into bounded, testable execution behavior.

---

## 14. Current Commands

```bash
# Evidence
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories

# Backtest / planning
PYTHONPATH=src python3 -m quant_rabbit.cli replay-backtest --start-balance 222781
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance 222781

# Broker truth
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --start-balance 222781 --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json

# Decision verification
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
    --snapshot data/broker_snapshot.json \
    --decision-response data/codex_trader_decision_response.json

# Risk / receipts
PYTHONPATH=src python3 -m quant_rabbit.cli risk-dry-run --intent intent.json --snapshot snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli promote-receipts
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage

# Stage / send
PYTHONPATH=src python3 -m quant_rabbit.cli stage-live-order \
    --lane-id 'failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE'
PYTHONPATH=src python3 -m quant_rabbit.cli autotrade-cycle \
    --use-gpt-trader \
    --gpt-decision-response data/codex_trader_decision_response.json

# Replay / learning / certification
PYTHONPATH=src python3 -m quant_rabbit.cli replay-execution --prices data/quote_path.json --target-jpy 22278
PYTHONPATH=src python3 -m quant_rabbit.cli learn-post-trade --outcome outcome.json
PYTHONPATH=src python3 -m quant_rabbit.cli certify-dry-run

# LIVE (gated)
QR_LIVE_ENABLED=1 PYTHONPATH=src python3 -m quant_rabbit.cli autotrade-cycle --send
```

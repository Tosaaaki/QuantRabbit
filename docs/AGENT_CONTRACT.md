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
| Trader prompt branches | `docs/trader_prompts/*.md` |
| Live send wrapper | `scripts/run-autotrade-live.sh` |
| Live runtime sync | `scripts/sync-live-runtime.sh` |
| Append-only execution ledger | `data/execution_ledger.db` |
| Execution ledger latest report | `docs/execution_ledger_report.md` |
| Codex scheduled task | `~/.codex/automations/<automation-id>/automation.toml` (Codex Desktop-managed) |
| Claude scheduled task | `~/.claude/scheduled-tasks/trader/` |
| Legacy archive root | `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z` |
| Archive manifest | `…/ARCHIVE_MANIFEST_20260430T151527Z.md` |
| Archive log pointer | `ARCHIVE_POINTER.md` |
| Latest reports | `docs/*_report.md` |

If a sub-doc disagrees with this contract, **this contract wins**. Update the sub-doc.

---

## 2. Operator Model

The trader is a single role with one playbook (`docs/SKILL_trader.md`). The scheduled task picks which model executes a given cycle; the playbook, the gateways, and the broker account are identical regardless of which model runs. There is no "Codex trader" vs "Claude trader" — there is **the trader**, and a scheduler that dispatches it.

Rules:

- **Exactly one trader scheduled task enabled at a time.** Two tasks against the same OANDA account = duplicate orders, double-counted exposure, contradictory protection actions.
- **Switch by enabling / disabling scheduled tasks.** No code, prompt, or playbook changes are required to swap the executing model.
- **Every cycle reads the same `docs/SKILL_trader.md`.** Changes there affect every subsequent cycle whichever model runs.
- **Runtime/development worktrees are separated.** Active trader scheduled tasks must use `/Users/tossaki/App/QuantRabbit-live`, a clean committed worktree dedicated to broker reads, receipts, and gateway execution. Development and strategy improvement may happen in `/Users/tossaki/App/QuantRabbit` or another worktree, but a dirty development tree must never be the live trader `cwd`.
- **Main is the stable integration branch.** Development commits are promoted by `scripts/sync-live-runtime.sh`: source branch → `main` (fast-forward only) → `codex/live-trader-runtime` (fast-forward only). Do not commit directly in `/Users/tossaki/App/QuantRabbit-live`; that branch mirrors `main` for runtime and creates no live-side commits. The live wrapper runs `scripts/sync-live-runtime.sh --live-only --skip-tests` before each gateway cycle so a promoted `main` cannot be forgotten and runtime `docs/*_report.md` drift is cleared.
- **Promotion is automated and guarded.** The development post-commit hook installed by `scripts/install-live-runtime-hooks.sh` runs the sync script after each commit. Promotion is blocked by source/config/data/decision dirt, non-fast-forward history, failing tests, missing live worktree, or a scheduled trader automation that does not point at `/Users/tossaki/App/QuantRabbit-live`. Report-only `docs/*_report.md` drift may be cleared automatically in the live worktree because the next trader cycle overwrites latest reports.
- **Precheck before report writes.** Concurrency and clean-source checks must run before any command that writes tracked reports under `docs/*_report.md`. Dirty source/config/data/decision files block the cycle. Report-only tracked diffs under `docs/*_report.md` from the immediately previous trader cycle are runtime drift and do not block a scheduled run; the next cycle may overwrite latest reports. A stopped precheck must not create new diffs; otherwise the next scheduled cycle will self-block before it reaches the market read. A recent decision receipt is not a stop condition by itself: the router must distinguish unconsumed receipts from receipts already rejected, already handed to a gateway, or made stale by a newer broker snapshot / repriced intent packet.
- **Handoff discipline.** When the active scheduled task changes, the next cycle must (1) refresh `broker-snapshot`, (2) read the most recent `data/codex_trader_decision_response.json` (the filename is kept for compatibility, not as an attribution), (3) inherit any open trader-owned positions and pending orders, (4) not cancel another cycle's pending orders without an explicit reason recorded in the next decision receipt. When a scheduled trader has already generated `data/broker_snapshot.json` and `data/order_intents.json` before writing the decision receipt, the gateway cycle must use `autotrade-cycle --reuse-market-artifacts` so GPT verification sees the same evidence packet the decision cited; `LiveOrderGateway` still fetches fresh broker truth immediately before staging or sending.
- **Broker truth reconciliation must continue the cycle.** If refreshed broker truth says old local state, journal text, pending-order claims, or a prior receipt are stale, broker truth wins and the trader must re-route, rewrite the decision when needed, and continue through the gateway path. Absence of a locally remembered pending order in the refreshed OANDA snapshot is not a no-trade gate. Only an active live lock, another enabled trader scheduler, dirty source/config/data/decision files, missing required credentials, or actual current OANDA broker-truth exposure/risk gates may stop the cycle before decision work.

In automation, "GPT" means whichever model is currently dispatched by the scheduler. **Do not call any API-key model path from QuantRabbit code** (`QR_OPENAI_API_KEY`, `OPENAI_API_KEY`). Live discretion lives in the scheduled-task model, not in a library call.

---

## 3. Prime Directive

- Build from broker truth outward.
- Enforce broker gateway and risk contracts in code before prompt or strategy-prose optimization.
- Express discretionary autonomy as executable receipts, not vague market prose.
- Default to dry-run / read-only behavior unless live gates are explicitly satisfied.
- **Adapt to live market conditions, not to fixed numbers.** Sizing, stop distance, target distance, spread tolerance, exposure budget, and pacing must all be derived from current ATR, regime, spread, equity, daily progress, and liquidity — not from JPY / pip / multiplier literals baked into code or prompts.

---

## 3.5. No Thoughtless Hardcodes or Fallbacks

This rule applies to every file the trader reads or writes — `src/quant_rabbit/**`, `tools/**`, CLI flags, scheduled-task SKILLs, prompts, and config.

- **Hardcoded JPY / pip / multiplier values are forbidden** unless they are documented, named (`InstrumentSpec.normal_spread_pips`, `RiskPolicy.daily_risk_pct`, etc.), and clearly marked as a *test default* or a *broker-spec constant* — not as a production risk decision.
- **Silent fallbacks to a literal are forbidden.** If a required input is missing, raise (or emit a `*_MISSING` issue with severity `BLOCK`) so the operator sees it. Falling through to `RiskPolicy().max_loss_jpy` or `8 pips` or `1.5 R` quietly is the failure mode this contract was written to prevent.
- **Risk caps must be equity-derived, and per-trade ≠ per-day.** The daily-target ledger (`start_balance × daily_risk_pct`) is the **whole-day** budget. The **per-trade** cap is `daily_risk_budget_jpy / target_trades_per_day` — this is what flows into `intent.metadata['max_loss_jpy']`. The split is mandatory: a single losing trade must not be able to exhaust the day's budget, because the campaign needs many attempts to reach the 10% target. `target_trades_per_day` must come from `ai_test_bot_backtest.firepower.required_trades_per_day_at_observed_expectancy` when available; `RiskPolicy.target_trades_per_day = 10` is only the no-evidence fallback. Override via `--target-trades-per-day` on `daily-target-state` for an explicit operator pace. Never derive either cap from a JPY literal stored in code or in a stale state file.
- **Geometry must be ATR-derived.** SL distance and TP distance must scale with current ATR (per pair, per timeframe, per regime). `spread × N` is a *floor*, not the geometry. If pair-charts data is unavailable, the cycle waits — it does not invent numbers.
- **Reward / risk targets must be regime-derived.** A single fixed `target_reward_risk` for all conditions is the same anti-pattern as a fixed loss cap. Trend regimes deserve wider targets; range regimes deserve faster rotation.
- **Spread tolerance must be liquidity-derived.** A static `2.5 ×` multiplier across all sessions ignores Tokyo open vs London close. Tighten in liquid hours, loosen in thin hours, and BLOCK above session-aware caps.
- **No "default-to-yesterday" stale persistence.** State files (e.g. `data/daily_target_state.json`) must invalidate on a new campaign day or when the snapshot post-dates the file by more than one cycle. Reading a stale value and pretending it is current is a silent fallback in disguise.
- **Every numeric constant in production code requires a docstring or comment** stating: (a) what market reality it represents, (b) why it is constant rather than market-derived, and (c) what should replace it if it ever needs to be changed. If you cannot write that comment, the value should not be a constant.
- **When in doubt, fail loud.** A `BLOCK` issue with a clear message is always preferable to a guess. The operator can act on a blocker; the operator cannot recover from a silent miscalculation.

This rule is enforceable: any reviewer (Codex or Claude) seeing a JPY literal, a `pip` constant, a fixed multiplier, or a silent `or default_value` in trader code MUST flag it before merging.

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
- **Campaign exposure occupancy.** While `daily_target_state.json` is `PURSUE_TARGET` with `remaining_target_jpy > 0`, if no trader-owned position is active and at least one current `LIVE_READY` lane survives deterministic prefiltering, the cycle must not end as flat WAIT / REQUEST_EVIDENCE / discretionary NO_TRADE. Existing trader-owned pending entries must be counted through basket risk/margin/geometry validation; they satisfy occupancy only when basket capacity or duplicate/stale-pending gates block every additional lane. Manual/tagless operator exposure does not satisfy this occupancy requirement and does not block it.
- When the 10 % target is missed, produce a gap report (missed setup, blocked trade, market absence, risk rejection, execution cost, timing error, or strategy weakness).
- Reaching the daily target triggers **protection-first** behavior; do not keep adding risk just because more trades are possible. Trader-owned pending entry orders are canceled rather than left fillable after the target is reached.

---

## 6. Autonomous Pro Trader Requirements

- The operator is the discretionary decision layer, but every decision must be grounded in structured broker state, mined history, current market context, and risk geometry.
- The trader must decide whether to TRADE, WAIT, CANCEL, PROTECT, TIGHTEN, CLOSE, or REQUEST_EVIDENCE.
- **No-trade is a valid professional decision** when edge, spread, timing, liquidity, narrative, or risk / reward is inadequate — but it must be defensible against the campaign, not against an invented threshold.
- **WAIT discipline.** A WAIT decision is only valid when it cites a specific gate that fired, by name from this contract or from `data/`. Generic prose ("Golden Week thin liquidity", "EVENT_RISK", "UNCLEAR regime") is not sufficient. When `daily_target_state.json` shows `progress_pct < 50` AND `data/order_intents.json` lists ≥ 3 `LIVE_READY` lanes, WAIT additionally requires (a) one chart-story sentence per LIVE_READY lane stating why **that lane's specific invalidation** is hit right now, citing M5 numbers from `pair_charts.json`, and (b) explicit citation of the contract gate (e.g. §9 spread cap, §11 strategy block) or the `*_BLOCK` issue from `risk-dry-run`.
- **No invented thresholds.** The trader must not stack additional risk gates beyond those listed in §3.5, §9, §10, §11. Examples of forbidden invented gates: "ATR×2 safety floor for thin markets", "need 2× normal spread before entry", "skip all trades during a holiday week", "regime must be CLEAR before entry". When a condition feels risky, **size it down** — `per_trade_risk_budget_jpy` already shrinks the per-shot exposure — do not block the lane in prose.
- **Precedent must be rescaled.** Citing a past loss (e.g. "Apr 3 -984 JPY") to justify a current decision requires rescaling the precedent to **today's** per-trade cap: `precedent_loss × (current_per_trade_cap / cap_at_time_of_precedent)`. A precedent recorded at per-trade=4000 has 1/10 the weight under per-trade=400. Cite the rescaled figure or do not cite the precedent.
- Historical worst loss is a repair warning before current risk geometry is proven, not a second hard veto. Once the current `LIVE_READY` receipt fits `per_trade_risk_budget_jpy`, old loss memory becomes audit context only.
- Past negative evidence must not harm a current `LIVE_READY` lane. Negative `live_net`, missing positive mined evidence, low capture rate, old `WAIT` language, or stale story rejection markers may be printed as advisory context, but they must not create blockers, reduce size, or reduce rank after current broker-truth risk geometry and strategy profile validation have already produced `LIVE_READY`.
- Every trade decision compares historical analogs, current regime, market story, chart structure, campaign role, open exposure, pending orders, and invalidation distance.
- Every decision receipt states: selected lane, rejected alternatives, thesis, method, narrative, chart story, invalidation, TP, SL, units, expected reward, worst-case loss, reason for action, and `operator`.
- The system separates observation, reasoning, risk validation, order staging, order sending, position protection, and post-trade learning.
- The trader learns from wins, losses, missed edges, bad entries, premature exits, and protection repairs **without allowing memory to override live risk checks**.
- Maintain a professional audit trail: broker snapshots, intents, dry-run receipts, live-stage receipts, execution reports, protection actions, post-trade reviews, and the append-only execution ledger.
- `execution-ledger-sync` records OANDA transaction truth into `data/execution_ledger.db`. `stage-live-order` and `autotrade-cycle` sync it before and after gateway work and record gateway receipt files, so entries, exits, protections, cancels, and broker-side transaction ids have one durable DB trail.
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
- `autotrade-cycle` syncs the execution ledger before and after the broker gateway phase; do not bypass it with direct OANDA writes or out-of-band receipt edits.
- Flat-account entry loop: `BrokerSnapshot` → `IntentGenerator` → `TraderBrain` → `LiveOrderGateway`.
- Exposure-management loop: `BrokerSnapshot` → `TraderBrain` → `PositionManager` → `PositionProtectionGateway`.
- `gpt-trader-decision` verifies the operator's decision receipt by default.
  - `autotrade-cycle --use-gpt-trader --gpt-decision-response …` may hand off **only an ACCEPTED `TRADE`** from deterministic prefilter lane(s). A receipt may use `selected_lane_ids` to authorize a risk-bounded basket in the same gateway cycle. When the daily target is open, an accepted GPT trade authorizes deterministic basket expansion across every prefiltered `LIVE_READY` lane the gateway can fit; `LiveOrderGateway` remains responsible for portfolio position count, margin, cumulative risk, duplicate geometry, and broker-truth validation.
  - The verifier must include every current `LIVE_READY` lane from `data/order_intents.json`; `--max-lanes` may cap blocked/diagnostic lanes, but must not clip executable lanes visible to the operator.
  - If the campaign exposure occupancy rule is active, GPT WAIT / REQUEST_EVIDENCE / rejected / non-prefiltered output is not allowed to keep the trader flat when deterministic prefiltered lanes exist. The cycle may recover to the deterministic prefilter lane and must still stage/send only through `LiveOrderGateway`, which remains the final broker-truth and risk authority.
- Unprotected trader-owned exposure, external exposure, or over-budget exposure **blocks fresh entries**. Trader-owned pending entries do not blanket-block the campaign; they are either canceled when the current `TraderBrain` vetoes them, or counted as existing basket occupancy/risk/margin before any additional order is staged.
- Operator-managed manual/tagless exposure (`owner=manual` or `owner=unknown`) is observed in broker truth but is not modified, protected, closed, or counted as a fresh-entry blocker. The operator owns that risk. The trader may run in parallel and must still size its own entries from equity, ATR, spread, and trader-owned portfolio risk.
- Protected trader-owned exposure and trader-owned pending entries may add only through basket/portfolio risk validation; total worst-case loss and estimated margin must stay inside the active exposure budget.
- Pending orders vetoed by the current `TraderBrain` can be canceled before the next cycle. An accepted `TRADE` receipt may also list current trader-owned pending entry ids in `cancel_order_ids` when the decision explicitly replaces stale or lower-priority pending exposure with a current basket; verifier must reject unknown/manual/non-pending ids, and the gateway cancels verified ids before basket capacity validation. An accepted `CANCEL_PENDING` receipt must list current trader-owned pending entry ids in `cancel_order_ids`; the automation cancels those verified ids and stops that cycle without sending a fresh entry.
- Trader decisions compare mined history, market story, campaign role, narrative risk, current broker state, risk geometry, and live exposure.
- Portfolio Director records which desk wins and why when trader desks disagree.
- Parallel strategy review is allowed only as read-only reasoning over the same broker/market packet. Trend, range, and breakout-failure reviewers may produce advisory `strategy_reviews`, but only the final trader receipt may select a lane or `selected_lane_ids` basket, and execution still flows only through the verified `gpt-trader-decision` → gateway path.
- Parallel specialist observation is allowed only as processed read-only input to the single trader. Macro/news, indicator, flow/levels, risk-audit, strategy, and portfolio-context specialists may produce advisory `specialist_reviews` over the same broker/market packet, but they must declare `read_only=true`, `live_permission=false`, cite only packet evidence refs, and must not select lanes, cancel orders, change risk budgets, stage orders, or create a second trader loop.
- `ai-attack-advice` is also read-only. It may rank current `LIVE_READY` lanes and expose advisory parameter surfaces to Codex automation, but it cannot grant live permission, raise risk budgets, stage orders, or create a second trader loop.
- When `ai_attack_advice.recommended_now_lane_ids` intersects current tradeable `LIVE_READY` lanes and the daily target remains open, WAIT / REQUEST_EVIDENCE is invalid unless a named deterministic exposure, risk, strategy, spread, event, or broker-truth gate blocks the lane. Protected trader-owned exposure is not by itself a no-trade gate; additional entries are still decided by `gpt-trader-decision` and validated by `LiveOrderGateway` basket / portfolio checks. A TRADE using advised lanes must cite both `attack:advice` and `attack:lane:<lane_id>`. The first tradeable lane in `recommended_now_lane_ids` is the dynamic missed-edge repair priority for the cycle; the selected basket must include it before lower-ranked advice may be used. If that first lane is `EUR_USD` `SHORT`, it is the primary repair target unless a named deterministic gate blocks it after the advice packet was built.
- Basket breadth is not a GPT JSON formatting veto. If the accepted decision includes the first advised lane but omits other advised pairs, `gpt-trader-decision` records a warning and the gateway cycle expands to the deterministic prefilter basket; pairs that cannot fit are rejected by executable risk, margin, exposure, strategy, spread, event, duplicate-geometry, or broker-truth gates.
- Deterministic prefilter overlays `ai_attack_advice`. The trader_brain prefilter that produces `basket_lane_ids` must surface the same primary advised lanes the verifier expects: a `LIVE_READY` lane appearing in `ai_attack_advice.recommended_now_lane_ids[:K]` (where K = `ATTACK_ADVICE_PROMOTION_RANK_CEILING`, mirroring `gpt_trader.PRIMARY_ATTACK_RANK_CEILING = 4`) receives a documented score bonus + rationale entry, so it does not silently fall out of `SEND_ENTRY` because `pair_charts` long/short bias points the other way. The overlay never overrides §11 hard blocks (`BLOCK_UNTIL_NEW_EVIDENCE`, missing strategy profile, missing receipt) or §9 exposure blocks; it only nudges scoring among already-LIVE_READY candidates. The K cap is the same conviction gate the verifier uses for `BASKET_PAIR_COVERAGE_INCOMPLETE` — both must move together.
- **Directional gating + margin-aware basket (C-1 / C-2 / C-4, 2026-05-12).** The trader_brain `_apply_directional_gating` pass, run between `_score_lane` and basket construction, reshapes the NEW-entry lane set when bias and attack_advice agree on direction. It never touches `PositionManager` / `PositionProtectionGateway` / existing-position state — only the LaneScore tuple feeding `_basket_lane_plan`.
  - **C-1 directional gating.** For every pair appearing in scores, if `pair_charts[pair].confluence.score_balance` is `LONG_LEAN` or `SHORT_LEAN` AND `|score_gap| >= DIRECTIONAL_GATING_STRONG_GAP` (= `2 ×` the documented `chart_reader` TIED noise floor `0.05`, encoded as `DIRECTIONAL_GATING_STRONG_GAP_MULTIPLIER`) AND the `ai_attack_advice` top-K majority direction for the same pair agrees with the bias, every `LIVE_READY` lane in the OPPOSITE direction is demoted from `SEND_ENTRY` to `NO_TRADE` with a `directional_gating_demoted` blocker.
  - **C-2 attack_advice directional veto.** Independently of pair_charts bias, when the top-K advice for a pair has a strict majority in one direction, lanes in the opposite direction lose `ATTACK_ADVICE_VETO_PENALTY = 25.0` score points and gain an `attack_advice_veto` rationale entry. The magnitude matches the existing `-25` penalty grid for direction-conflict downgrades, so the veto remains internally consistent.
  - **C-4 margin-aware basket truncation.** `_basket_lane_plan` and `_expanded_gpt_basket_plan` track cumulative `LaneScore.estimated_margin_jpy` (sourced from `intent_generator.risk_metrics.estimated_margin_jpy`) and stop adding lanes once cumulative margin would exceed `snapshot.account.margin_available_jpy × MARGIN_AWARE_BASKET_BUFFER` (= 0.9, documented engineering buffer for intra-cycle drift). This prevents the basket from over-claiming margin and letting the gateway reject every candidate at staging with `BASKET_MARGIN_UTILIZATION_CAP_EXCEEDED`. The buffer is engineering tolerance, not a market-derived threshold — comment in code states the rationale.
  - **Existing-position invariant.** None of C-1 / C-2 / C-4 reads `snapshot.positions`, `snapshot.orders`, or any TP/SL field. Position management remains the exclusive domain of `PositionManager` + `PositionProtectionGateway`, which run against a separate `managed_snapshot` and a separate `position_decision` flow. The directional gate cannot — by construction — alter SL/TP/CLOSE behavior on open trades.
- Strategy-review identity is `lane_id` plus `method`, not a loose desk alias. A review for `TREND_CONTINUATION` cannot authorize a `RANGE_ROTATION` or `BREAKOUT_FAILURE` lane.

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
- Manual/tagless operator exposure does **not** block new trader entries and must not be modified by the trader. External/broker-synced exposure that is not explicitly operator-managed still blocks new entries until adopted or closed.

---

## 10. Position Protection

- Missing TP / SL is a **repair requirement** (not optional).
- This repair requirement applies to trader-owned positions. Operator-managed manual/tagless positions are not repaired or closed by the trader.
- Profitable protected positions can tighten SL to break-even or better.
- Contradicted trader-owned positions can be closed.
- **Existing SL cannot be widened.**
- **Existing TP is not moved by the protection gateway.**
- **CLOSE discipline (two-gate, 2026-05-12, `feedback_no_unilateral_close.md`).** A `CLOSE` receipt — and a `TRADE` receipt that lists `close_trade_ids` — must satisfy **both** of the following or be rejected by `gpt-trader-decision`:
  - **Gate A — market evidence of thesis invalidation.** For each named trade id, either (i) `pair_charts` reports a `BOS_<DIR>` or `CHOCH_<DIR>` against the position side on `M15` or `H4` (parsed from `chart_story`; same structure-event lens trader_brain already uses), or (ii) the receipt populates `invalidation_price` + `invalidation_tf` AND broker truth shows the level traded through (LONG needs `bid <= invalidation_price`; SHORT needs `ask >= invalidation_price`). Prose `invalidation` text alone is not enough — the gate requires a machine-checkable signal so the verifier can reproduce the operator's reasoning. Code: `_close_thesis_invalidated` in `gpt_trader.py`.
  - **Gate B — explicit operator authorization.** The receipt must carry `operator_close_authorized: true`, or the operator shell must export `QR_OPERATOR_CLOSE_OVERRIDE=1`. The trader cannot autonomously decide to close a trader-owned position; the operator must opt in. The override env is for documented emergencies (broker disconnect, regulatory order); production cycles set the receipt field.
- The two-gate model is a §3.5-compliant repair for the 2026-05-11 18:17 UTC regression where the GPT trader autonomously closed four valid SHORT positions for **-3,291 JPY** (12 % of the day's `daily_risk_budget_jpy`) before their wide TPs could fire. The structural gate is anchored on `chart_reader.structure_events`, not on JPY/pip literals.

---

## 11. Strategy Evidence

- Run `PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy` before strategy work.
- Mine archived trade logs, trade history, journals, strategy memory, market stories, and audit logs **before** changing strategy behavior.
- Treat as primary evidence: `logs/live_trade_log.txt`, `logs/trader_journal.jsonl`, `logs/s_hunt_ledger.jsonl`, `logs/audit_history.jsonl`, structured memory, daily handoffs.
- Extract repeatable lessons from profitable trades, losing trades, near misses, manual interventions, execution failures, and broker-state mismatches.
- Reuse strong legacy mechanisms only after they are converted into vNext tests, receipts, risk checks, or read-only evidence pipelines.
- `risk-dry-run` reads `data/strategy_profile.json` when present.
- `promote-receipts` feeds passing dry-run receipts back into `data/strategy_profile.json`.
- Strategy profile history is at least pair/direction scoped, and may become method-scoped when a receipt proves a specific method. When method-scoped evidence exists for a pair/direction, another method must not reuse that evidence to become live-ready.
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

### Always keep commits current
- The working tree must be **committed before ending a session**, before switching operator (Codex ↔ Claude), and before any scheduled-task cycle that may run live (`autotrade-cycle --send`).
- Commit logical units as they finish — do not let unrelated changes pile up across sessions. Untracked or uncommitted work invalidates the audit trail and risks the next operator inheriting an unknown state.
- Receipts and reports written into `data/` and `docs/*_report.md` are runtime artifacts; commit them only when they are part of a deliberate snapshot, not on every cycle (they are usually `.gitignore`-d under `data/`).
- Before starting work, run `git status` / `git log -1` to confirm the tree is clean and the previous operator's commit is the latest. If it is not, either commit the leftover changes (with the correct operator attribution) or ask the user before discarding.
- Never `--amend` a published commit, never force-push, and never skip hooks (`--no-verify`). Stack a new commit instead.

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
# Prompt routing
PYTHONPATH=src python3 -m quant_rabbit.cli trader-prompt-route

# Evidence
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories

# Backtest / planning
PYTHONPATH=src python3 -m quant_rabbit.cli replay-backtest --start-balance 222781
PYTHONPATH=src python3 -m quant_rabbit.cli ai-test-bot-backtest --start-balance 222781
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance 222781

# Broker truth
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src python3 -m quant_rabbit.cli execution-ledger-sync

# Per-pair indicator stack (Phase A+B+C extended)
PYTHONPATH=src python3 -m quant_rabbit.cli pair-charts --timeframes M1,M5,M15,M30,H1,H4,D --output data/pair_charts.json

# Market context layers (must be refreshed before every trader cycle)
PYTHONPATH=src python3 -m quant_rabbit.cli cross-asset-snapshot   # DXY synthetic, US bond CFDs, SPX/Gold/Oil/BTC, FX correlations
PYTHONPATH=src python3 -m quant_rabbit.cli flow-snapshot          # OANDA OrderBook/PositionBook + spread time series
PYTHONPATH=src python3 -m quant_rabbit.cli currency-strength      # G8 strength meter
PYTHONPATH=src python3 -m quant_rabbit.cli levels-snapshot        # Pivots, PDH/PDL/PDC, sessions, round numbers
PYTHONPATH=src python3 -m quant_rabbit.cli economic-calendar      # ForexFactory High/Medium events + per-pair window
PYTHONPATH=src python3 -m quant_rabbit.cli cot-snapshot           # CFTC TFF leveraged-funds positioning
PYTHONPATH=src python3 -m quant_rabbit.cli option-skew            # IV/RR adapter (currently MISSING_OPTION_SKEW_FEED)

# Intent pricing uses the broker snapshot freshness gate. Refresh broker truth
# again after market-context fetches, otherwise a slow cycle can turn every lane
# into STALE_QUOTE before risk validation.
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage
PYTHONPATH=src python3 -m quant_rabbit.cli ai-attack-advice

# Decision verification
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
    --snapshot data/broker_snapshot.json \
    --decision-response data/codex_trader_decision_response.json

# Risk / receipts
PYTHONPATH=src python3 -m quant_rabbit.cli risk-dry-run --intent intent.json --snapshot snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli promote-receipts

# Stage / send
PYTHONPATH=src python3 -m quant_rabbit.cli stage-live-order \
    --lane-id 'failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE'
PYTHONPATH=src python3 -m quant_rabbit.cli autotrade-cycle \
    --reuse-market-artifacts \
    --use-gpt-trader \
    --gpt-decision-response data/codex_trader_decision_response.json

# Replay / learning / certification
PYTHONPATH=src python3 -m quant_rabbit.cli replay-execution --prices data/quote_path.json --target-jpy 22278
PYTHONPATH=src python3 -m quant_rabbit.cli learn-post-trade --outcome outcome.json
PYTHONPATH=src python3 -m quant_rabbit.cli certify-dry-run

# LIVE (gated)
scripts/sync-live-runtime.sh
./scripts/run-autotrade-live.sh \
    --reuse-market-artifacts \
    --use-gpt-trader \
    --gpt-decision-response data/codex_trader_decision_response.json \
    --send
```

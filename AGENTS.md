# QuantRabbit — Codex Discretionary FX Trading System

## Core Philosophy: Codex IS a Human Elite Pro Trader

**Codex is not a bot. Codex acts as a human elite pro trader — nothing less.**

### Tools are extensions of your arm. Sharpen, adjust, and evolve them to match market conditions.

OANDA is a tool. Trailing stops, BE moves — those are OANDA features. Whether and how to use them is the trader's call.
The registry, scripts — same thing. All of them are **Codex's tools**.

The relationship between a pro trader and their tools:
- **Choose**: Read the market, then decide which tools to use and how today
- **Adjust**: Change parameters. Tune settings per position
- **Improve**: Rewrite the code itself. Change the calculation logic
- **Build**: If a tool doesn't exist, build it
- **Discard**: If a tool is useless, throw it out

**Keep sharpening your tools. But keep your mind the mind of a pro trader.**

### Codex thinking like a bot is NOT OK.

- "Score is above 3, so go long" → NOT OK. Can you explain why long in terms of market conditions?
- "Checklist conditions are met, so enter" → NOT OK. Is this the result of reading the market?

**How an elite pro trader thinks:**
- Read the market → Form a hypothesis → Verify with tools → Adjust tools → Make the call → Evolve the tools

### Prompt Design Principle: Think at the Point of Output

**All prompts (SKILL.md, rules, strategy_memory) must work equally well on Opus and Sonnet.** This is a hard constraint. If a prompt only works on one model, it's a bad prompt.

How to achieve this:

| Approach | Opus | Sonnet | Verdict |
|---|---|---|---|
| **Rules / checklists** ("if X then BLOCKED") | Follows but feels constrained. May override with better judgment | Follows mechanically. Becomes a smarter bot | ❌ Model-dependent |
| **Self-questioning** ("ask yourself why") | Thinks deeply | Reads, agrees, then ignores when writing output | ❌ Model-dependent |
| **Output format forces thinking** ("write 'I would enter if...' for each pair") | Thinks deeply while writing | Must think to fill in the format. Can't write "Skip" | ✅ Model-independent |

**The principle**: Don't tell Codex what to think. Shape the format of what Codex writes so that thinking is required to produce the output. A bot can follow a rule. A bot can skip a self-question. But a bot cannot fill in "I would enter if..." without forming a trade plan.

**When editing prompts**:
- Never add a rule that says "don't do X." Instead, change the output format so X is impossible to write
- Never add a preamble section ("before you begin, consider..."). Instead, embed the consideration into the field the model fills in
- Test: "Could a model produce this output by copy-pasting from last session?" If yes, the format is too loose

---

## Architecture (v8.5)

### Scheduled tasks driving everything

**Codex automations** (runtime in `~/.codex/automations/`; canonical prompts in repo `docs/SKILL_*.md`; Codex wrappers in `docs/codex_automations/*.md`):

| Task | Runtime | Interval | Session Length | Role |
|------|---------|----------|----------------|------|
| trader | Cron | Every 20 min | Shared runtime: 15-min session window (10-min minimum), 15-min stale-lock threshold, 17-min watchdog | Pro trader. Does analysis, news, and trading all itself |
| inventory-director | Cron (paused) | Daily 00:00 JST | ~2-3 min | Disabled backup task retained for recovery workflows; not part of the live Codex recurring path |
| daily-review | Cron | Daily 15:00 JST | ~5 min | Daily retrospective. Reviews the most recent completed UTC trading day, emits a memory-promotion gate plus lesson-state review queue, then adds Bayesian evidence + AAR prompts before evolving strategy_memory.md |
| daily-performance-report | Cron (paused) | Daily 10:30 JST | ~2 min | Aggregate realized P&L from OANDA → post to #qr-daily |
| daily-slack-summary | Cron | Daily 07:00 JST | ~2 min | Auto-post daily trade summary to Slack #qr-daily |
| intraday-pl-update | Cron (paused) | Every 3h (weekdays 9-24 JST) | ~1 min | Post today's realized P&L to #qr-daily |
| quality-audit | Cron (paused) | Every 30 min | ~3-4 min | Independent market analyst prompt retained in Codex, currently paused here. Claude compatibility still runs its own quality-audit cadence |

**Cowork tasks** (runs on Cowork platform, not in scheduled-tasks/):

| Task | Model | Interval | Role |
|------|-------|----------|------|
| qr-news-digest | Cowork | Hourly | News collection + trader-perspective summary via WebSearch |
| qr-news-flow-append | Cowork | Hourly (:15) | Append compact snapshot from news_digest.md → logs/news_flow_log.md |

**Claude compatibility note**: Claude's scheduled-task host is not identical to Codex. The canonical trader cadence is 20 minutes on every host when enabled, while Claude currently remains the quality-audit host and its trader compatibility task may be disabled locally. Disabled prompts live under `~/.claude/scheduled-tasks/*.DISABLED`. Shared prompts and `tools/task_runtime.py` must therefore stay host-neutral.

**Local bot layer**: Removed from live operation on 2026-04-17. Legacy bot prompts, scripts, policy files, and disabled task links are retained for rollback or historical analysis, but routine recurring trader sessions do not steer worker policy or run any local bot cycle.

**Method**: Current Codex recurring path is `qr-trader` every 20 minutes. Shared runtime rules live in `tools/task_runtime.py` and `tools/session_end.py`: 15-minute session window (10-minute minimum gate, 15-minute stale-lock threshold), plus a 17-minute watchdog. `session_end.py` now validates `collab_trade/state.md` before lock release, auto-derives carry-forward `Hot Updates` from the final receipts, snapshots that day's `state.md` into `collab_trade/daily/<UTC-date>/state.md`, and syncs formal `seat_outcomes` before memory ingest. `session_data.py` refreshes stale or missing chart PNGs as a fallback when audit artifacts are old, now including the M1 execution set (`M1/M5/H1`), so the trader's visual read is not blocked by missing images.

**Tag taxonomy**:
- `trader`, `trader_scalp`, `trader_rotation`, `trader_swing` = live discretionary inventory owned directly by the trader task
- `range_bot`, `range_bot_market`, `trend_bot_market` = legacy historical tags only; no longer produced by routine recurring trader sessions

- Memory handoff: `collab_trade/state.md` (external memory across sessions)
- Long-term learning memory: `collab_trade/strategy_memory.md` (distilled daily by daily-review)
- Vector memory: `collab_trade/memory/memory.db` (SQLite + sqlite-vec. Ruri v3 embeddings). Memory chunks are direction-tagged, historical re-ingest must use only day-specific state snapshots, never today's live `state.md`, and `strategy_memory.md` is re-ingested as section-level lesson chunks instead of one coarse blob. Re-ingest also auto-syncs lesson state markers in the markdown from the registry review queue before rebuilding chunks
- Lesson state registry: `collab_trade/memory/lesson_registry.json` (generated from `strategy_memory.md`; tracks lesson state/type/trust so recall can prefer confirmed recurring-trader knowledge over loose prose)
- Feedback DB: `pretrade_outcomes` table (pretrade_check predictions vs. actual P&L, plus learning-gate / session-bucket / regime / execution-style metadata). Identical unmatched pretrade probes inside the same short window are deduped so review learns from distinct decisions, not spam
- Seat-outcome DB: `seat_outcomes` table (formal `discovered / orderable / deployed / captured / missed` chain synced from `logs/s_hunt_ledger.jsonl`, scored from the written reference price to the best favorable excursion by the review cutoff so missed seats are not understated by a flat close)
- **News cache**: `logs/news_digest.md` (Cowork updates hourly) + `logs/news_cache.json` (API parser structured data + session_data.py fallback)
- **News flow log**: `logs/news_flow_log.md` (48h of hourly snapshots — HOT/THEME/WATCH per hour. Cowork appends at :15 via tools/news_flow_append.py. daily-review reads this to track narrative evolution)
- **S Hunt ledger**: `logs/s_hunt_ledger.jsonl` (append-only short/medium/long horizon receipts captured at session end; daily-review scores discovery vs deployment vs capture from it)
- Prompt sync: `docs/SKILL_*.md` are the single source of truth. Codex automation wrappers (`docs/codex_automations/*.md`) and Claude compatibility links (`~/.claude/scheduled-tasks/*/SKILL.md`) must reference those files and must not fork prompt content. Active Claude task directories must not keep `SKILL_ja.md` alternates, and their `schedule.json` descriptions should match the canonical prompt frontmatter.

### News Pipeline (Cowork → Codex)
```
Every 1 hour: Cowork qr-news-digest (:00)
  ├── WebSearch × 3 (breaking news · central banks · economic calendar)
  ├── python3 tools/news_fetcher.py (API structured data: Finnhub+AV+FF)
  └── WRITES: logs/news_digest.md (trader-perspective summary) + logs/news_cache.json

Every 1 hour: Cowork qr-news-flow-append (:15, runs after qr-news-digest)
  └── python3 tools/news_flow_append.py → APPENDS to logs/news_flow_log.md (HOT/THEME/WATCH snapshot)

Every 20 min: trader session
  ├── session_data.py reads logs/news_digest.md (macro context in 10 seconds)
  └── Incorporates news into thesis construction (the "why is it moving" evidence)
```

### Self-Improvement Loop
```
Every 20 min: trader session
  ├── reads: strategy_memory.md + state.md + quality_audit.md (regime map + visual read + range opportunities)
  ├── session_data.py refreshes stale `logs/charts/*.png` if needed before the trader reads them (M1/M5/H1)
  ├── profit_check.py --all + protection_check.py  ← every session, first thing
  ├── reads regime + visual chart observations from quality_audit.md (auditor's eyes)
  ├── if quality_audit.md has issues → address them (re-evaluate missed S-candidates, fix sizing)
  ├── per entry: pretrade_check.py → records to pretrade_outcomes
  ├── trades → trades.md + live_trade_log.txt + Slack
  └── SESSION_END (15-min session window, 10-min minimum enforced in code; stale-lock recovery at 15 min): validate_trader_state.py + auto_hot_updates.py + record_s_hunt_ledger.py + seat_outcomes.py + trade_performance.py + ingest.py → memory.db

When quality-audit runs (Codex paused here; Claude compatibility currently every 45 min):
  ├── runs: quality_audit.py + profit_check + fib_wave + chart_snapshot.py (14 PNGs)
  ├── READS: chart PNGs visually (candle patterns, BB position, momentum character)
  ├── WRITES: logs/quality_audit.md (facts + Regime Map + Visual Read + Range Opportunities)
  ├── WRITES: logs/quality_audit.json (machine) + logs/audit_history.jsonl (scanner facts + final narrative picks)
  ├── Sonnet-auditor exercises JUDGMENT on each finding (REPORT / NOISE)
  └── if REPORT items → posts summary to #qr-daily via Slack

Daily 06:00 UTC: daily-review session
  ├── runs: daily_review.py for the most recent completed UTC trading day (fact collection + recurring-trader vs other execution split with `trades.md` fallback ownership + signal-window audit scoring from each signal window's best favorable excursion + memory-promotion gate + lesson-state suggestions + pretrade result correlation + S Hunt capture review)
  ├── THINKS: What worked? Why?
  ├── WRITES: strategy_memory.md (pattern promotion/addition/refutation)
  └── runs: ingest.py --force (enriched re-ingestion, including fresh section-level `strategy_memory.md` chunks plus automatic lesson-state marker sync)

Next day's trader → reads updated strategy_memory.md → behavior changes
```

### Memory System (memory.db) — 3 Layers + Feedback
- **SQL layer**: trades / user_calls / market_events / **pretrade_outcomes** → quantitative analysis + prediction accuracy tracking
- **Seat-outcome layer**: `seat_outcomes` → promoted `S Hunt` horizons plus near-S `S Excavation` podium seats, with discovery / orderability / deployment / capture / miss scoring for missed-opportunity learning
- **Vector layer**: Ruri v3-30m (256-dim) QA chunks → narrative search for "similar situations". Includes day-level trades/state plus section-level distilled strategy-memory lessons
- **Registry layer**: `lesson_registry.json` → structured lesson state machine (`candidate/watch/confirmed/deprecated`) plus trust scores for runtime prioritization
- **Distillation layer**: strategy_memory.md → experiential knowledge updated daily by daily-review

**Usage**: See `.Codex/skills/` — `/pretrade-check`, `/memory-save`, `/memory-recall`

## Absolute Rules

→ Details auto-loaded from `.Codex/rules/`. Summary:
- **Codex IS the pro trader**: Not the one building the system — the one doing the trading
- **Don't think like a bot**: Every decision is market-condition-based
- **Build tools freely**: Persistent bot processes are prohibited
- **Direct OANDA orders**: Hit the REST API directly via urllib
- Every order must be logged to `logs/live_trade_log.txt`

## .Codex/ Structure

```
.Codex/
├── settings.json          ← Shared permissions (committed to repo)
├── settings.local.json    ← Personal permissions (gitignored)
├── rules/                 ← Auto-loaded rules (always present in every session)
│   ├── trading-philosophy.md   ← Pro trader philosophy & prohibitions
│   ├── recording.md            ← Recording rules (4-point set)
│   ├── risk-management.md      ← Stop-loss · take-profit · failure patterns
│   ├── technical-analysis.md   ← MTF hierarchy · cross-pair scan · indicator selection
│   ├── oanda-api.md            ← API connection · data fetch tools
│   └── change-protocol.md     ← Required protocol on changes
├── skills/                ← Slash commands (37 skills total)
│   ├── secretary.md       ← /secretary status report + command hub
│   ├── collab-trade.md    ← /collab-trade launch collaborative trading
│   ├── pretrade-check.md  ← /pretrade-check pre-entry 3-layer risk check
│   ├── market-order.md    ← /market-order market order
│   └── ... (see .Codex/skills/ for full list)
└── projects/              ← Memory
```

## Required Rules on Changes

→ Details auto-loaded from `.Codex/rules/change-protocol.md`
1. **Update AGENTS.md**: Update this file on any architecture changes
2. **Update memory**: Update the relevant memory files
3. **Append to changelog**: Add an entry to `docs/CHANGELOG.md`
4. **Merge to main**: Always merge to main when editing in a worktree
5. **Deploy immediately**: Once changed, reflect it right away. Don't ask.
6. **English only**: Active prompt, task, and guide files are English-only. Do not keep Japanese alternates in the repo or the live Claude task directories
7. **Smoke test**: Run the script, verify actual output. Both `python3` and `.venv/bin/python`. "Syntax OK" ≠ "works"

## Document Map

### Must-Read

| File | Contents |
|------|----------|
| `AGENTS.md` (this file) | Full architecture overview, absolute rules |
| `docs/SKILL_trader.md` | Canonical recurring trader task definition shared by Claude/Codex |

### Operational Documents (reference as needed)

| File | Contents |
|------|----------|
| `docs/CHANGELOG.md` | Chronological log of all changes |
| `docs/SKILL_daily-review.md` | Canonical daily-review task definition shared by Claude/Codex |
| `docs/SKILL_quality-audit.md` | Canonical quality-audit task definition shared by Claude/Codex |
| `docs/codex_automations/` | Thin Codex automation wrappers that reference canonical prompts |
| `docs/TRADER_PROMPT.md` | Trader mental model only. Not a runtime or scheduler contract |
| `collab_trade/CLAUDE.md` | Manual collaborative-trading guide. Only used when scheduled tasks are stopped |
| `docs/TRADER_LESSONS.md` | Historical failure patterns (daily-review reference, not loaded in trader sessions) |

### Runtime Files

| File | Contents |
|------|----------|
| `collab_trade/state.md` | State handoff across sessions (positions · story · lessons). Includes `Hot Updates` for intraday carry-forward plus the required `S Excavation Matrix` (all 7 pairs: blocker / upgrade / dead condition) before `S Hunt` promotion. A promoted `S Hunt` horizon must name how that blocker cleared; otherwise it closes as `dead thesis because no seat cleared promotion gate` |
| `collab_trade/strategy_memory.md` | Long-term learning memory (per-pair tendencies, pattern validity, lessons) |
| `collab_trade/memory/lesson_registry.json` | Structured lesson registry generated from `strategy_memory.md` (state/type/trust snapshot for recall and review) |
| `collab_trade/summary.md` | All-day performance · trend summary (updated at session end) |
| `logs/live_trade_log.txt` | Trade execution log (chronological) |
| `logs/news_digest.md` | News summary updated by Cowork hourly |
| `logs/news_cache.json` | Structured news data from API parser |
| `logs/bot_inventory_policy.md` | Legacy local-bot policy file retained for rollback/history. Not part of the routine recurring trader loop |
| `logs/bot_inventory_policy.json` | Legacy machine policy file retained for rollback/history. Not part of the routine recurring trader loop |
| `logs/technicals_*.json` | Technical cache across M1/M5/M15/H1/H4 |
| `logs/quality_audit.md` | Quality audit facts (updated every 45 min when the audit runs. Trader reads at session start and ignores stale prose if session_data flags it) |
| `logs/quality_audit.json` | Quality audit machine-readable output (for daily-review parsing) |
| `logs/audit_history.jsonl` | Append-only audit opportunity tracking (scanner fires + final narrative A/S picks. daily-review uses it for recipe and judgment accuracy) |
| `logs/s_hunt_ledger.jsonl` | Append-only trader-receipt log from each session (`S Hunt` horizons + `S Excavation Matrix` podium + reference prices). daily-review uses it to score both promoted and near-S missed seats |

### Scripts

| File | Contents |
|------|----------|
| `tools/session_data.py` | Full data fetch at trader session start (technicals M1/M5/M15/H1/H4 + OANDA + macro + **Currency Pulse** (cross-currency triangulation) + **H4 Position** (lifecycle) + **Tokyo-open breadth board** when the trader is inside the Tokyo complex, so multi-lane mornings are visible instead of inferred late + Fib M5+H1 + Slack + **direction-aware actionable memory recall for held/pending/scanner pairs plus carry-forward watch targets parsed from the prior `state.md`** with a wider scanner intake + **learning-weighted edge board / fresh-risk tournament** that converts lesson history into pair-direction size caps, then tightens them further with the current session bucket and live M5 regime proxy + **S-hunt deployment cues** that pre-close the best fresh seats as `MARKET / LIMIT / STOP-ENTRY / PASS`, now including market-scout lanes for live B seats so valid S-hunt ideas do not remain prose-only and allowing live `headwind` trend seats to demote into scout/trigger participation instead of auto-flatness when spread is still normal + **S Excavation seeds** that prefill the near-S podium from the live fresh-risk board plus fresh audit strongest-unheld / narrative A-S calls when available + **multi-vehicle deployment lanes** so broad sessions can keep up to `PRIMARY / BACKUP / THIRD CURRENCY` alive as separate expressions + **S Excavation Matrix template** so every pair names blocker / upgrade / dead condition before `S Hunt` + **intraday OODA / Decision Journal / Micro AAR prompts** + **Hot Updates from the prior state handoff**), all at once. Refreshes trader chart artifacts as needed, now including M1 execution charts |
| `tools/mid_session_check.py` | Lightweight mid-session check (~1s): Slack + prices + trades + margin only |
| `tools/profit_check.py` | **Run at every session start** — 6-axis TP evaluation (ATR ratio, M5 momentum, H1 structure, correlation, S/R, peak) |
| `tools/protection_check.py` | **Run at every session start** — TP/SL/Trailing status check per ATR. NO PROTECTION = immediate action. Detects rollover window |
| `tools/rollover_guard.py` | Rollover SL guard — remove/restore SL/Trailing around daily OANDA maintenance (5 PM ET) |
| `tools/preclose_check.py` | **Run before every close** — re-confirms thesis before exit |
| `tools/close_trade.py` | Position close (PUT /trades/{id}/close. Prevents hedge account mistakes). Historical worker-tag safeguards remain for recovery workflows, but routine recurring trader sessions manage discretionary `trader` inventory |
| `tools/fib_wave.py` | N-wave structure + Fibonacci levels. Run at session start for all pairs |
| `tools/refresh_factor_cache.py` | OANDA candle refresh + technical cache builder (M1/M5/M15/H1/H4). Re-execs under `.venv` automatically if `python3` lacks pandas |
| `tools/chart_snapshot.py` | **Visual charts + regime detection** — generates candlestick PNG (BB/EMA/KC overlay) + detects TREND/RANGE/SQUEEZE. **Run primarily by quality-audit**; `session_data.py` refreshes the chart set as a fallback when PNGs are stale or missing. Auditor reads PNGs visually, writes Regime Map + Range Opportunities to quality_audit.md. `--all` = 7 pairs × M5+H1, `--all --with-m1` adds the trader's M1 execution set |
| `tools/oanda_performance.py` | **OANDA API-based performance analysis** — ground truth P&L, win rate, R:R, best streak, per-pair breakdown. USE THIS for any performance analysis, not grep on log files |
| `tools/trade_performance.py` | Performance aggregation (legacy — parses log file). Still used by the recurring trader loop for strategy-feedback compatibility; use `oanda_performance.py` for ground-truth analysis |
| `collab_trade/memory/ingest.py` | Session memory ingestion. Splits `trades.md`/`notes.md`/historical `state.md` snapshots into direction-tagged chunks, auto-syncs `strategy_memory.md` lesson state markers from the registry review queue, re-ingests the markdown as section-level lesson chunks, refreshes `lesson_registry.json`, embeds them, and writes structured trade/user/event data to `memory.db`. Historical re-ingest must not reuse the live `collab_trade/state.md` for past dates |
| `tools/slack_trade_notify.py` | Slack notifications |
| `tools/news_fetcher.py` | News fetch (Finnhub+AlphaVantage+FF. Called from Cowork task) |
| `tools/slack_daily_summary.py` | Daily summary |
| `tools/quality_audit.py` | Quality audit — cross-checks trader decisions against rules and S-conviction data |
| `tools/record_audit_narrative.py` | Parses the final Auditor's View from `logs/quality_audit.md` and appends the auditor's unheld A/S narrative picks to `logs/audit_history.jsonl` |
| `tools/auto_hot_updates.py` | Auto-derives compressed carry-forward `Hot Updates` from the final `S Hunt` / `Capital Deployment` receipts at session end. Safety net for the next session; does not replace writing the update live when you learn it |
| `tools/record_s_hunt_ledger.py` | Parses `collab_trade/state.md` at session end and appends the trader's short / medium / long S-hunt receipts plus `S Excavation Matrix` podium seats to `logs/s_hunt_ledger.jsonl` for daily-review scoring |
| `tools/seat_outcomes.py` | Syncs `logs/s_hunt_ledger.jsonl` into `memory.db` `seat_outcomes`, now for both promoted `S Hunt` horizons and near-S `S Excavation` podium seats. Scores discovery / orderability / deployment / capture / miss from the written reference price to the best favorable excursion by the review cutoff, and exposes the review scoreboard. The scoreboard collapses a continuing same-pair seat into one chain so repeated state snapshots do not inflate deployment counts |
| `tools/state_hot_update.py` | Fast helper to prepend one compressed intraday learning bullet into `collab_trade/state.md` `## Hot Updates`, keeping only the latest carry-forward corrections for the next session |
| `tools/validate_trader_state.py` | Validates that `state.md` closes each S-hunt horizon as a real order receipt or an explicit `no seat cleared promotion gate` dead-thesis close, and (for 2026-04-18+ handoffs) blocks SESSION_END if the `S Excavation Matrix` is missing pair coverage or podium lines |
| `tools/task_runtime.py` | Host-neutral trader/audit runtime helper for shared prompts (Claude/Codex) |
| `tools/check_task_sync.py` | Verifies canonical prompts, Claude symlinks, and Codex wrappers stay aligned |
| `tools/s_conviction_scan.py` | S-conviction pattern scanner — auto-detects TF × indicator combinations |
| `tools/range_scalp_scanner.py` | **Range scalp scanner** — detects RANGE across 7 pairs, outputs ready-to-trade plans with BB levels, signal strength, sizing, R:R. Run at session start when ranges detected in regime map. `--json` for programmatic use |
| `tools/trend_bot.py` | Legacy disabled local-bot tool retained for rollback and historical analysis. Not part of the live recurring trader path |
| `tools/range_bot.py` | Legacy disabled local-bot tool retained for rollback and historical analysis. Not part of the live recurring trader path |
| `tools/bot_policy_guard.py` | Legacy disabled local-bot helper retained for rollback and historical analysis. Not part of the live recurring trader path |
| `tools/bot_trade_manager.py` | Legacy disabled local-bot helper retained for rollback and historical analysis. Not part of the live recurring trader path |
| `tools/bot_inventory_snapshot.py` | Legacy disabled inspection helper retained for rollback and historical analysis. Not part of the live recurring trader path |
| `tools/render_bot_inventory_policy.py` | Legacy disabled helper retained for rollback and historical analysis. Not part of the live recurring trader path |
| `tools/cancel_order.py` | Pending-order cancel helper. Still useful for manual cleanup and recovery workflows |
| `tools/local_bot_cycle.sh` | Legacy disabled local-bot supervisor retained for rollback only |
| `scripts/install_local_bot_launchd.sh` | Legacy rollback helper for the removed local-bot layer |

## Key Directories

- `collab_trade/` — state.md, strategy_memory.md, summary.md, daily/, memory/, indicators/
- `docs/` — prompts, changelog, SKILL_*.md reference copies
- `tools/` — all analysis, notification, and execution scripts
- `indicators/` — low-level technical indicator engine (calc_core, divergence, factor_cache)
- `collab_trade/indicators/` — collab session quick calculation (quick_calc.py)
- `logs/` — trade log, technical cache, news cache, lock files
- `config/env.toml` — OANDA API keys etc. (gitignored)

### Archive (legacy, no need to reference)
- `archive/` — Historical artifacts from v1-v7 only (old bot workers, systemd, GCP/VM infra, old prompts, VM core DB, etc.). Current QuantRabbit does not run on VM/GCP.

## User Commands

- "秘書" → triggers `/secretary` skill — live OANDA status + full command hub
- "共同トレード" → triggers `/collab-trade` skill — reads `collab_trade/CLAUDE.md`, stops scheduled tasks, starts collaborative session
- "トレード開始" → **trader is a scheduled task** (Codex currently every 20 min). This phrase in conversation launches a manual collaborative session same as "共同トレード"

## Context Management

### Session management during collaborative trading
- Suggest switching to a new session after 1-2 hours or at major decision breakpoints
- Before suggesting, make sure `collab_trade/state.md` is up to date

### Recovery procedure on context overflow / new session
1. Read `collab_trade/state.md` immediately
2. Read `collab_trade/CLAUDE.md`
3. Check current account state via OANDA API
4. Go look at the market immediately

## Operational Principles

- **Write down what you notice or promise to do — immediately**
- **Don't just say you'll do a TODO — actually complete it**
- **Use external memory**: Context overflows. If you write state to md files, you can recover

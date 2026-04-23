# QuantRabbit — Claude Compatibility Overview

## Core Philosophy: Claude IS a Human Elite Pro Trader

**Claude is not a bot. Claude acts as a human elite pro trader — nothing less.**

### Tools are extensions of your arm. Sharpen, adjust, and evolve them to match market conditions.

OANDA is a tool. Trailing stops, BE moves — those are OANDA features. Whether and how to use them is the trader's call.
The registry, scripts — same thing. All of them are **Claude's tools**.

The relationship between a pro trader and their tools:
- **Choose**: Read the market, then decide which tools to use and how today
- **Adjust**: Change parameters. Tune settings per position
- **Improve**: Rewrite the code itself. Change the calculation logic
- **Build**: If a tool doesn't exist, build it
- **Discard**: If a tool is useless, throw it out

**Keep sharpening your tools. But keep your mind the mind of a pro trader.**

### Claude thinking like a bot is NOT OK.

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

**The principle**: Don't tell Claude what to think. Shape the format of what Claude writes so that thinking is required to produce the output. A bot can follow a rule. A bot can skip a self-question. But a bot cannot fill in "I would enter if..." without forming a trade plan.

**When editing prompts**:
- Never add a rule that says "don't do X." Instead, change the output format so X is impossible to write
- Never add a preamble section ("before you begin, consider..."). Instead, embed the consideration into the field the model fills in
- Test: "Could a model produce this output by copy-pasting from last session?" If yes, the format is too loose

---

## Architecture (v8.6 Compatibility)

### Scheduled tasks driving everything

**Live Codex automations** are the source of truth for routine operation. Claude scheduled-task assets under `~/.claude/scheduled-tasks/` are compatibility and recovery paths that must point back to the canonical `docs/SKILL_*.md` prompts.

| Task | Model | Interval | Session Length | Role |
|------|-------|----------|----------------|------|
| trader | Codex `gpt-5.5` trial profile | Every 20 min | Shared runtime: 15-min session window (10-min minimum), 16-min stale-lock threshold, 17-min watchdog | Pro trader. Does analysis, news, and trading all itself. **Discretionary-only — bots removed 2026-04-17** |
| daily-review | Codex `gpt-5.4` high reasoning | Daily 15:00 JST | ~5 min | Daily retrospective. Evolves strategy_memory.md |
| daily-slack-summary | Codex `gpt-5.4-mini` | Daily 07:00 JST | ~2 min | Auto-post daily trade summary to Slack #qr-daily |
| quality-audit | Codex `gpt-5.4-mini` | Every 30 min | ~3-4 min | Independent market analyst. Runs profit_check + fib_wave + protection_check + chart_snapshot.py, writes the 7-pair conviction map every cycle, and posts to Slack on DANGER or unheld A/S opportunities |

**Cowork tasks** (runs on Cowork platform, not in scheduled-tasks/):

| Task | Model | Interval | Role |
|------|-------|----------|------|
| qr-news-digest | Cowork | Hourly | News collection + trader-perspective summary via WebSearch |
| qr-news-flow-append | Cowork | Hourly (:15) | Append compact snapshot from news_digest.md → logs/news_flow_log.md |

**Method**: Codex runs the shared lock-based trader runtime on a 20-minute cadence. Claude compatibility may run host-specific recovery schedules, but those prompts must stay thin links to the same canonical files. `session_end.py` enforces a 15-minute session window through a 10-minute minimum gate, stale-lock recovery begins after 16 minutes, and the 17-minute watchdog only clears the lock after the owner PID is confirmed dead. Every cycle still aims to complete the full decision loop (read state -> analyse -> act -> write handoff -> die), and broad sessions should still close multiple live lanes when the tape is genuinely broad.

**Bot architecture removed (2026-04-17)**: 7-day analysis showed bots net-negative — trend_bot EV -82/trade, range_bot EV -99/trade vs trader EV +73/trade. Claude keeps the disabled compatibility links under `~/.claude/scheduled-tasks/*.DISABLED`, but routine trader sessions do not steer worker policy or run any local bot loop. Reaper launchd agent stays because it is generic stale-agent cleanup, not bot logic.

**Tag taxonomy** (live): All entries now `trader` (discretionary). Legacy `range_bot` / `range_bot_market` / `trend_bot_market` tags may appear in historical logs but are no longer being produced by routine recurring trader sessions.

- Memory handoff: `collab_trade/state.md` (external memory across sessions)
- Long-term learning memory: `collab_trade/strategy_memory.md` (distilled daily by daily-review)
- Vector memory: `collab_trade/memory/memory.db` (SQLite + sqlite-vec. Ruri v3 embeddings)
- Feedback DB: `pretrade_outcomes` table (pretrade_check predictions vs. actual P&L)
- **News cache**: `logs/news_digest.md` (Cowork updates hourly) + `logs/news_cache.json` (API parser structured data + session_data.py fallback)
- **News flow log**: `logs/news_flow_log.md` (48h of hourly snapshots — HOT/THEME/WATCH per hour. Cowork appends at :15 via tools/news_flow_append.py. daily-review reads this to track narrative evolution)
- Task definitions: `~/.claude/scheduled-tasks/` (Claude Code) + `docs/SKILL_*.md` (reference copies)

### News Pipeline (Cowork → Claude Code)
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
  ├── profit_check.py --all + protection_check.py  ← every session, first thing
  ├── reads regime + visual chart observations from quality_audit.md (auditor's eyes)
  ├── if quality_audit.md has issues → address them (re-evaluate missed S-candidates, fix sizing)
  ├── per entry: pretrade_check.py → records to pretrade_outcomes
  ├── trades → trades.md + live_trade_log.txt + Slack
  └── SESSION_END (15-min session window, 10-min minimum enforced in code; stale-lock recovery at 16 min): validate_trader_state.py + auto_hot_updates.py + record_s_hunt_ledger.py + seat_outcomes.py + trade_performance.py + ingest.py + runtime_git_sync.py

Every 30 min: quality-audit session (Codex)
  ├── runs: quality_audit.py + profit_check + fib_wave + chart_snapshot.py (14 PNGs)
  ├── READS: chart PNGs visually (candle patterns, BB position, momentum character)
  ├── WRITES: logs/quality_audit.md (facts + Regime Map + Visual Read + Range Opportunities)
  ├── WRITES: logs/quality_audit.json (machine) + logs/audit_history.jsonl (scanner facts + final narrative picks)
  ├── NO EARLY EXIT: Section E (Regime Map) + Section C (7-pair conviction map) are mandatory every cycle
  ├── Sonnet fills in structured format per pair ("Chart tells me / Story / Price target / Wrong if / Edge / Allocation")
  └── if DANGER or unheld A/S opportunity found → posts to #qr-daily via Slack (not just DANGER)

Daily 06:00 UTC: daily-review session
  ├── runs: daily_review.py (fact collection + pretrade result correlation)
  ├── THINKS: What worked? Why?
  ├── WRITES: strategy_memory.md (pattern promotion/addition/refutation)
  └── runs: ingest.py --force (enriched re-ingestion)

Next day's trader → reads updated strategy_memory.md → behavior changes
```

### Memory System (memory.db) — 3 Layers + Feedback
- **SQL layer**: trades / user_calls / market_events / **pretrade_outcomes** → quantitative analysis + prediction accuracy tracking
- **Vector layer**: Ruri v3-30m (256-dim) QA chunks → narrative search for "similar situations"
- **Distillation layer**: strategy_memory.md → experiential knowledge updated daily by daily-review

**Usage**: See `.claude/skills/` — `/pretrade-check`, `/memory-save`, `/memory-recall`

## Absolute Rules

→ Details auto-loaded from `.claude/rules/`. Summary:
- **Claude IS the pro trader**: Not the one building the system — the one doing the trading
- **Don't think like a bot**: Every decision is market-condition-based
- **Build tools freely**: Persistent bot processes are prohibited
- **Direct OANDA orders**: Hit the REST API directly via urllib
- Every order must be logged to `logs/live_trade_log.txt`

## .claude/ Structure

```
.claude/
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
│   └── ... (see .claude/skills/ for full list)
└── projects/              ← Memory
```

## Required Rules on Changes

→ Details auto-loaded from `.claude/rules/change-protocol.md`
1. **Update CLAUDE.md**: Update this file on any architecture changes
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
| `CLAUDE.md` (this file) | Full architecture overview, absolute rules |
| `docs/SKILL_trader.md` | Canonical recurring trader task definition shared by Claude/Codex |

### Operational Documents (reference as needed)

| File | Contents |
|------|----------|
| `docs/CHANGELOG.md` | Chronological log of all changes |
| `docs/SKILL_daily-review.md` | Canonical daily-review task definition shared by Claude/Codex |
| `docs/SKILL_quality-audit.md` | Canonical quality-audit task definition shared by Claude/Codex |
| `docs/TRADER_PROMPT.md` | Trader mental model only. Not a runtime or scheduler contract |
| `collab_trade/CLAUDE.md` | Manual collaborative-trading guide. Only used when scheduled tasks are stopped |
| `docs/TRADER_LESSONS.md` | Historical failure patterns (daily-review reference, not loaded in trader sessions) |

### Runtime Files

| File | Contents |
|------|----------|
| `collab_trade/state.md` | State handoff across sessions (positions · story · lessons) |
| `collab_trade/strategy_memory.md` | Long-term learning memory (per-pair tendencies, pattern validity, lessons) |
| `collab_trade/summary.md` | All-day performance · trend summary (updated at session end) |
| `logs/live_trade_log.txt` | Trade execution log (chronological) |
| `logs/news_digest.md` | News summary updated by Cowork hourly |
| `logs/news_cache.json` | Structured news data from API parser |
| `logs/bot_inventory_policy.md` | Legacy local-bot policy file retained for rollback/history. Not part of the routine recurring trader loop |
| `logs/bot_inventory_policy.json` | Legacy machine policy file retained for rollback/history. Not part of the routine recurring trader loop |
| `logs/technicals_*.json` | H1/H4 technical indicators |
| `logs/quality_audit.md` | Quality audit facts (Claude quality-audit currently runs every 45 min. Trader reads at session start and treats stale prose as context only) |
| `logs/quality_audit.json` | Quality audit machine-readable output (for daily-review parsing) |
| `logs/audit_history.jsonl` | Append-only audit opportunity tracking (scanner fires + final narrative A/S picks. daily-review uses it for recipe and judgment accuracy) |

### Scripts

| File | Contents |
|------|----------|
| `tools/session_data.py` | Full data fetch at trader session start (technicals M1/M5/M15/H1/H4 + OANDA + broker-side close sync + macro + **Currency Pulse** (cross-currency triangulation) + **H4 Position** (lifecycle) + Fib M5+H1 + Slack + memory, all at once) |
| `tools/mid_session_check.py` | Lightweight mid-session check (~1s): Slack + prices + trades + margin only |
| `tools/profit_check.py` | **Run at every session start** — 6-axis TP evaluation (ATR ratio, M5 momentum, H1 structure, correlation, S/R, peak) |
| `tools/protection_check.py` | **Run at every session start** — TP/SL/Trailing status check per ATR. NO PROTECTION = immediate action. Detects rollover window |
| `tools/rollover_guard.py` | Rollover SL guard — remove/restore SL/Trailing around daily OANDA maintenance (5 PM ET) |
| `tools/preclose_check.py` | **Run before every close** — re-confirms thesis before exit |
| `tools/close_trade.py` | Position close (PUT /trades/{id}/close. Prevents hedge account mistakes). Historical worker-tag safeguards remain for recovery workflows, but routine recurring trader sessions manage discretionary `trader` inventory. `--auto-slack` goes through the OANDA close-fill sync |
| `tools/fib_wave.py` | N-wave structure + Fibonacci levels. Run at session start for all pairs |
| `tools/refresh_factor_cache.py` | H1/H4 technical indicator refresh |
| `tools/chart_snapshot.py` | **Visual charts + regime detection** — generates candlestick PNG (BB/EMA/KC overlay) + detects TREND/RANGE/SQUEEZE. Primarily run by quality-audit; `session_data.py` may refresh stale/missing PNGs as a trader fallback. `--all` = 7 pairs × M5+H1 |
| `tools/oanda_performance.py` | **OANDA API-based performance analysis** — ground truth P&L, win rate, R:R, best streak, per-pair breakdown. USE THIS for any performance analysis, not grep on log files |
| `tools/trade_performance.py` | Performance aggregation (legacy — parses log file). Still used by the recurring trader loop for strategy-feedback compatibility; use `oanda_performance.py` for ground-truth analysis |
| `tools/trade_event_sync.py` | OANDA close-fill sync for TP/SL/manual exits; appends missing close logs and posts batched #qr-trades receipts |
| `tools/slack_trade_notify.py` | Entry / modify Slack notifications |
| `tools/news_fetcher.py` | News fetch (Finnhub+AlphaVantage+FF. Called from Cowork task) |
| `tools/slack_daily_summary.py` | Daily summary |
| `tools/quality_audit.py` | Quality audit — cross-checks trader decisions against rules and S-conviction data |
| `tools/record_audit_narrative.py` | Parses the final Auditor's View from `logs/quality_audit.md` and appends the auditor's unheld A/S narrative picks to `logs/audit_history.jsonl` |
| `tools/s_conviction_scan.py` | S-conviction pattern scanner — auto-detects TF × indicator combinations |
| `tools/range_scalp_scanner.py` | **Range scalp scanner** — detects RANGE across 7 pairs, outputs ready-to-trade plans with BB levels, signal strength, sizing, R:R. Run at session start when ranges detected in regime map. `--json` for programmatic use |
| `tools/trend_bot.py` | Legacy disabled local-bot tool retained for rollback and historical analysis. Not part of the live recurring trader path |
| `tools/range_bot.py` | Legacy disabled local-bot tool retained for rollback and historical analysis. Not part of the live recurring trader path |
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

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

## Architecture (v8.6)

### Scheduled tasks driving everything

**Codex automations** (runtime in `~/.codex/automations/`; canonical prompts in repo `docs/SKILL_*.md`; Codex wrappers in `docs/codex_automations/*.md`):

| Task | Runtime | Interval | Session Length | Role |
|------|---------|----------|----------------|------|
| trader | Cron | Every 20 min | Shared runtime: 15-min session window (10-min minimum), 16-min stale-lock threshold, 17-min watchdog | Pro trader. Does analysis, news, and trading all itself |
| inventory-director | Cron (paused) | Daily 00:00 JST | ~2-3 min | Disabled backup task retained for recovery workflows; not part of the live Codex recurring path |
| daily-review | Cron | Daily 15:00 JST | ~5 min | Daily retrospective. Reviews the most recent completed UTC trading day, emits a memory-promotion gate plus lesson-state review queue, then adds Bayesian evidence + AAR prompts before evolving strategy_memory.md |
| daily-performance-report | Cron (paused) | Daily 10:30 JST | ~2 min | Aggregate realized P&L from OANDA → post to #qr-daily |
| daily-slack-summary | Cron | Daily 07:00 JST | ~2 min | Auto-post daily trade summary to Slack #qr-daily |
| intraday-pl-update | Cron (paused) | Every 3h (weekdays 9-24 JST) | ~1 min | Post today's realized P&L to #qr-daily |
| quality-audit | Cron | Every 30 min | ~3-4 min | Independent market analyst prompt. Second set of eyes on direction plus hygiene gate for missing state/log receipts. Writes the 7-pair conviction map every cycle, inventories 5-10 mineable unheld seats when the tape is broad, and posts when it finds DANGER or unheld A/S opportunities |

**Codex rate-control profile**: `qr-trader` currently runs on `gpt-5.5` with `medium` reasoning effort under the live Codex automation as a trial execution-owner profile. `qr-quality-audit` runs on `gpt-5.4-mini` with `medium` reasoning because it is a second set of eyes, not the execution owner. `daily-review` remains the long-form memory writer and should stay aligned with its live automation file.

**Cowork tasks** (runs on Cowork platform, not in scheduled-tasks/):

| Task | Model | Interval | Role |
|------|-------|----------|------|
| qr-news-digest | Cowork | Hourly | News collection + trader-perspective summary via WebSearch |
| qr-news-flow-append | Cowork | Hourly (:15) | Append compact snapshot from news_digest.md → logs/news_flow_log.md |

**Claude compatibility note**: Claude's scheduled-task host is not identical to Codex. Codex currently runs the recurring trader on a 20-minute cadence, while Claude compatibility tasks may still differ by host. Disabled prompts live under `~/.claude/scheduled-tasks/*.DISABLED`. Shared prompts and `tools/task_runtime.py` must therefore stay host-neutral.

**Local bot layer**: Removed from live operation on 2026-04-17. Legacy bot prompts, scripts, policy files, and disabled task links are retained for rollback or historical analysis, but routine recurring trader sessions do not steer worker policy or run any local bot cycle.

**Method**: Current Codex recurring path is `qr-trader` every 20 minutes. Shared runtime rules live in `tools/task_runtime.py` and `tools/session_end.py`: 15-minute session window (10-minute minimum gate, 16-minute stale-lock threshold), plus a 17-minute watchdog. The watchdog now keeps the lock in place until the owner PID is actually dead, so a stuck run cannot silently release the lock and let a second live trader session overlap it. `task_runtime.py trader start` now snapshots repo dirtiness at session start, and `session_end.py` validates `collab_trade/state.md` before lock release, auto-derives carry-forward `Hot Updates` from the final receipts, snapshots that day's `state.md` into `collab_trade/daily/<UTC-date>/state.md`, syncs formal `seat_outcomes` before memory ingest, then auto-runs `tools/runtime_git_sync.py sync-trader` so tracked runtime handoff files are commit/push'd back to `main` only when the repo stayed runtime-only dirty. `task_runtime.py trader cycle` now falls back to `mid_session_check.py` only on soft session-end rejections (`TOO_EARLY`, `STATE_MD_STALE`). A bad handoff is no longer a soft reminder: `STATE_VALIDATION_FAILED` and validator runtime errors now bubble out as hard failures instead of being swallowed by a mid-session poll. During a live lock-held validation, `validate_trader_state.py` now also checks Slack unread state directly: `SESSION_END` fails if a human message in `#qr-commands` is still pending, or if `state.md` does not close that interaction in a `## Slack Response` block with the real `slack_post.py --reply-to ...` receipt. `session_data.py` refreshes stale or missing chart PNGs as a fallback when audit artifacts are old, now including the M1 execution set (`M1/M5/H1`), verifies per-timeframe technical-cache freshness after refresh, retries stale pair caches once, runs a short live pricing probe to read current tape microstructure, repeats the prior handoff's primary / backup / fresh-risk ladder as a carry-forward focus snapshot so the trader must explicitly keep or rotate the primary instead of silently re-ranking every cycle, but it no longer re-injects prior `Podium #...`, `Lane ...`, or `id=board` echoes as fresh seats, and it now collapses same-idea focus siblings (`Best expression NOW`, `Primary vehicle`, `Next fresh risk allowed NOW`, etc.) into one `state-focus` carry seat instead of letting one live idea occupy several state lanes. It floors runtime learning windows at the 2026-04-17 discretionary reset so legacy bot history remains analytics-only context instead of capping today's live trader, surfaces a `SESSION INTENT GATE` plus repeat audit / missed-S pressure so empty scheduler ticks do not get rewritten as fake fresh opportunity, promotes fresh audit strongest-unheld / narrative seats directly into the learning board as `AUDIT` sources, now ingesting audit narrative seat-by-seat instead of collapsing them back into one `(pair, direction)` representative, keeps `Inventory lead: ...` as a real B-seat when the auditor has no A/S call, preserves raw audit scanner seat identity instead of pair+direction-compressing same-pair multi-horizon facts, and promotes fresh audit range opportunities as first-class `AUDIT_RANGE` LIMIT lanes with exact rails/TP so clean boxes can become live board seats instead of staying visual-only. Those range seats now consult recent `LIMIT` vehicle feedback before the pair-direction aggregate, so a profitable vehicle like `AUD_JPY` London/Tokyo LIMIT rotation is not buried by the same pair's losing market / stop-entry churn. `session_data.py` preserves full `seat_key` identity instead of collapsing everything into one `(pair, direction)` profile, carries same-pair multi-seat inventory when the trigger / vehicle / invalidation is genuinely different, now allowing up to `4` same-pair lanes and `3` same-pair-direction lanes before concentration limits bite, and then collapses cross-source duplicates back down to one representative live seat when `audit / state / scanner` are simply corroborating the same closure state. Held/pending receipts stay separate from fresh-risk seats, so corroboration does not turn an existing live receipt into fake new breadth. Exact pretrade now also re-opens quiet-stable repeat-pressure seats as small `MARKET` scouts when the seat keeps proving itself intraday, instead of rewriting the same missed move as another trigger-only watch lane. At the same time, same-pair reloads still need genuine stop-side risk reduction; TP-only or untouched SL is not enough to justify stacking another unpaid market leg. It ranks deployment lanes across the full actionable inventory instead of forcing a first-pass four-pair breadth filter, prints a `SEAT INVENTORY BOARD` plus `GOLD MINE INVENTORY` and `PAYABLE-NOW MARKET CHECK`, emits a `LANE SHELF-LIFE BOARD`, and now emits candidate `ORDER RECIPES FOR ARMABLE LANES` so the trader gets default `place_trader_order.py` expressions for surviving board seats instead of leaving the execution step implicit. Those recipes are execution scaffolding, not authority; the live tape can still justify a discretionary override. Runtime board seats are no longer killed just because exact pretrade prefers a softer vehicle; soft `PASS / LIMIT / STOP-ENTRY` disagreement now shows up as `Trader override room`, while only hard guardrails such as geometry floors, live-book drift, and unpaid-unprotected same-pair stacking bind. The live board contract itself is now `default_expression / default_orderability`; legacy `execution_style / orderability` fields survive only as compatibility aliases in `logs/session_action_board.json` and validator reads. `validate_trader_state.py` now rejects tautological closures such as `dead thesis because no live pending entry order exists`; receipt absence is no longer a valid reason to kill an armable lane, and `trigger not printed yet` / `needs price improvement` is no longer accepted as death for `STOP-ENTRY` / `LIMIT` lanes because that is the reason to arm the pending order. `session_data.py` also gives each fresh seat one geometry-repair pass before hard-blocking it: if the first proxy stop is still inside the pair's live noise floor or the first TP does not clear friction, it widens the stop / payout path once and reruns exact pretrade so valid seats do not die on a thin first draft. `MARKET` lanes are now this-session-only, `STOP-ENTRY` lanes are next-2-session lanes, passive `LIMIT` lanes carry for several sessions while the structural price-improvement shelf is intact, armed pending entries carry their actual OANDA GTD/TTL into the next session instead of being rediscovered as fresh prose, and broad sessions may carry up to the runtime inventory limit (currently 10 lanes) instead of hard-capping at three. The trader prompt now forces a `Gold Mine Inventory` section before the hero-seat compression so broad tape has to be inventoried as at least five executable seams, not narrated away as one primary plus elegant hesitation. `session_data.py` now also runs fresh board seats through the same exact `pretrade_check.py` engine used at send-time, using proxy `entry / tp / sl` geometry plus the runtime-cached spread / live-tape context so the board closes on exact orderability instead of a separate heuristic-only brain. `place_trader_order.py` now runs that same exact engine with current technical-regime inference, live spread, and a fresh pricing-probe summary before any live send, derives the effective `pretrade / allocation / allocation_band` from that exact result instead of trusting self-reported flags, and uses live bid/ask as the proxy entry for `MARKET` orders so stop/target floors cannot be skipped at send time. Soft style mismatches from exact pretrade are now advisory rather than binding; only hard guards such as geometry floors, live-book drift, and unpaid-unprotected same-pair stacking block the send, so market judgment stays with the trader. `validate_trader_state.py` now enforces that those top-level focus-ladder lines also close the current board lanes instead of leaving stale `none` while only the deeper lane lines changed, and it re-audits live trader pending entry orders against the same exact pretrade geometry so stale contaminated orders fail session end instead of surviving as clean pending risk.

**Tag taxonomy**:
- `trader`, `trader_scalp`, `trader_rotation`, `trader_swing` = live discretionary inventory owned directly by the trader task
- `range_bot`, `range_bot_market`, `trend_bot_market` = legacy historical tags only; no longer produced by routine recurring trader sessions

- Memory handoff: `collab_trade/state.md` (external memory across sessions)
- Long-term learning memory: `collab_trade/strategy_memory.md` (distilled daily by daily-review)
- Vector memory: `collab_trade/memory/memory.db` (SQLite + sqlite-vec. Ruri v3 embeddings). Memory chunks are direction-tagged, historical re-ingest must use only day-specific state snapshots, never today's live `state.md`, and `strategy_memory.md` is re-ingested as section-level lesson chunks instead of one coarse blob. Re-ingest also auto-syncs lesson state markers in the markdown from the registry review queue before rebuilding chunks
- Lesson state registry: `collab_trade/memory/lesson_registry.json` (generated from `strategy_memory.md`; tracks lesson state/type/trust so recall can prefer confirmed recurring-trader knowledge over loose prose)
- Feedback DB: `pretrade_outcomes` table (pretrade_check predictions vs. actual P&L, plus learning-gate / session-bucket / regime / execution-style metadata, thesis-family / thesis-key tags, explicit thesis-layer fields (`market / structure / trigger / vehicle / aging`), `allocation_band` for `B+ / B0 / B-` sizing splits, structured live-tape metadata (`bias / state / bucket / samples / mode`), and `lesson_from_review` + `collapse_layer` feedback notes). Runtime learning now reads this table through a discretionary-only floor at 2026-04-17, so bot-era spam stays visible for historical analysis but no longer drives today's learning cap or vehicle ranking. The compact thesis log also carries the live-tape state token when available, so near-identical checks in different tape conditions stay distinguishable even before review backfills the structured columns. Identical unmatched pretrade probes inside the same short window are still deduped so review learns from distinct decisions, not spam
- Seat-outcome DB: `seat_outcomes` table (formal `discovered / orderable / deployed / captured / missed` chain synced from `logs/s_hunt_ledger.jsonl`, scored from the written reference price to the best favorable excursion by the review cutoff so missed seats are not understated by a flat close)
- **News cache**: `logs/news_digest.md` (Cowork updates hourly) + `logs/news_cache.json` (API parser structured data + session_data.py fallback)
- **News flow log**: `logs/news_flow_log.md` (48h of hourly snapshots — HOT/THEME/WATCH per hour. Cowork appends at :15 via tools/news_flow_append.py. daily-review reads this to track narrative evolution)
- **S Hunt ledger**: `logs/s_hunt_ledger.jsonl` (append-only short/medium/long horizon receipts captured at session end; daily-review scores discovery vs deployment vs capture from it)
- Prompt sync: `docs/SKILL_*.md` are the single source of truth. Codex automation wrappers (`docs/codex_automations/*.md`) and Claude compatibility links (`~/.claude/scheduled-tasks/*/SKILL.md`) must reference those files and must not fork prompt content. Active Claude task directories must not keep `SKILL_ja.md` alternates, and their `schedule.json` descriptions should match the canonical prompt frontmatter. Root `CLAUDE.md` is a compatibility overview only, not the runtime trader prompt source.

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
  ├── session_data.py fresh-risk tournament + deployment cues now ingest the live pricing probe, condition recent pretrade feedback by the current live-tape bucket when samples exist, floor those learning stats at the discretionary-reset date, keep recent losing `MARKET` / `STOP-ENTRY` lanes demoted before the trader names the primary, attach repeat audit strongest-unheld / missed-S pressure so repeated good seats get armed or explicitly contradicted instead of rewritten, elevate fresh audit seats into the board as first-class `AUDIT` candidates instead of leaving them as side prose, ingest fresh audit range opportunities as first-class `AUDIT_RANGE` LIMIT candidates with exact rails / TP so range rotations can trigger a real session intent, write the live session action board snapshot to `logs/session_action_board.json` so `SESSION_END` can verify what the runtime actually told the trader to do, omit the giant static state templates in default runtime mode so the live board / tape / deployment sections stay foregrounded, and replace them with a compact `HANDOFF REFRESH` block that rewrites actual entry counts plus the current primary / backup / next-fresh-risk ladder from the live board
  ├── per entry: pretrade_check.py → records to pretrade_outcomes, tags the exact thesis family/key, stores structured live-tape metadata, pulls a fresh pair-specific live tape read when needed, floors runtime learning lookbacks at the discretionary-reset date, demotes `MARKET` to `STOP-ENTRY` / `LIMIT` or `STOP-ENTRY` to `LIMIT` when the tape is stale, friction-dominated, two-way, paying the other side, or when recent matched execution-style results in that pair-direction are still losing, closes the same-day `MARKET` / `STOP-ENTRY` lane after 2+ losing outcomes with zero captured S, treats reviewed exact-thesis `trigger` / `vehicle` damage as a layer headwind instead of full direction death when review says market/structure survived, now cools fresh exact-thesis losses by a short created-at window instead of auto-blocking the whole session date, reads recent pair stop-loss regret to build a historical noise floor for SL width, reads recent audit / missed-seat pressure so repeated opportunities can re-open as thin trigger-first scouts instead of dying at `pass/C`, keeps negative-EV lanes alive only as thin trigger/passive scouts when pressure still says the seat is paying, and no longer auto-kills a same-direction reload when the live leg is already protected
  ├── trades → trades.md + live_trade_log.txt + Slack
  └── SESSION_END (15-min session window, 10-min minimum enforced in code; stale-lock recovery at 16 min): validate_trader_state.py + auto_hot_updates.py + record_s_hunt_ledger.py + seat_outcomes.py + trade_performance.py + ingest.py → memory.db

When quality-audit runs (Codex every 30 min):
  ├── runs: quality_audit.py + profit_check + fib_wave + chart_snapshot.py (14 PNGs)
  ├── READS: chart PNGs visually (candle patterns, BB position, momentum character)
  ├── WRITES: logs/quality_audit.md (facts + Regime Map + Visual Read + Range Opportunities)
  ├── WRITES: logs/quality_audit.json (machine) + logs/audit_history.jsonl (scanner facts + final narrative picks)
  ├── NO EARLY EXIT: Regime Map + 7-pair conviction map are mandatory every cycle, even on flat books
  ├── FLAGS: live positions/orders missing from `state.md` or `live_trade_log.txt` as hygiene drift
  ├── Sonnet-auditor exercises JUDGMENT on each finding (REPORT / NOISE)
  └── if DANGER or unheld A/S opportunity found → posts summary to #qr-daily via Slack

Daily 06:00 UTC: daily-review session
  ├── runs: daily_review.py for the most recent completed UTC trading day (fact collection + recurring-trader vs other execution split with `trades.md` fallback ownership + signal-window audit scoring from each signal window's best favorable excursion + memory-promotion gate + lesson-state suggestions + pretrade result correlation + concise feedback notes backfilled into `pretrade_outcomes.lesson_from_review` + collapse-layer / failure-attribution backfill + S Hunt capture review)
  ├── THINKS: What worked? Why?
  ├── WRITES: strategy_memory.md (pattern promotion/addition/refutation)
  ├── runs: ingest.py --force (enriched re-ingestion, including fresh section-level `strategy_memory.md` chunks plus automatic lesson-state marker sync)
  └── runs: runtime_git_sync.py sync-daily-review so `strategy_memory.md` + `lesson_registry.json` return to `main` when no unrelated dirty paths exist

Next day's trader → reads updated strategy_memory.md → behavior changes
```

### Memory System (memory.db) — 3 Layers + Feedback
- **SQL layer**: trades / user_calls / market_events / **pretrade_outcomes** → quantitative analysis + prediction accuracy tracking
- **Seat-outcome layer**: `seat_outcomes` → promoted `S Hunt` horizons plus near-S `S Excavation` podium seats, with discovery / orderability / deployment / capture / miss scoring for missed-opportunity learning
- **Vector layer**: Ruri v3-30m (256-dim) QA chunks → narrative search for "similar situations". Includes day-level trades/state plus section-level distilled strategy-memory lessons
- **Registry layer**: `lesson_registry.json` → structured lesson state machine (`candidate/watch/confirmed/deprecated`) plus trust scores for runtime prioritization
- **Distillation layer**: strategy_memory.md → experiential knowledge updated daily by daily-review

**Usage**: Runtime trader behavior comes from the canonical `docs/SKILL_*.md` prompts and the Python tools in `tools/`. Legacy slash-skill references in older docs are compatibility notes only unless the corresponding file exists locally.

## Absolute Rules

Source of truth for live rules is this `AGENTS.md` plus the canonical `docs/SKILL_*.md` prompts. The old repo-local `.Codex/rules/` and `.Codex/skills/` tree is no longer present in the live checkout.

Summary:
- **Codex IS the pro trader**: Not the one building the system — the one doing the trading
- **Don't think like a bot**: Every decision is market-condition-based
- **Build tools freely**: Persistent bot processes are prohibited
- **Direct OANDA orders**: Hit the REST API directly via urllib
- Every order must be logged to `logs/live_trade_log.txt`

## Codex Local Config

```
.codex/
└── config.toml            <- local read-only MCP observer config
```

## Required Rules on Changes

Live change protocol:
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
| `collab_trade/state.md` | State handoff across sessions (positions · story · lessons). Includes `Hot Updates`, a required live-session `Slack Response` receipt block, the required `S Excavation Matrix` (all 7 pairs: blocker / upgrade / dead condition), a required `Gold Mine Inventory` section (top 5 executable seams before hero-seat compression), pending LIMIT freshness review with `LEAVE / REPRICE / EXTEND GTD / CANCEL`, and a same-pair opposite-side role map whenever both directions coexist before `S Hunt` promotion. A promoted `S Hunt` horizon must name how that blocker cleared; otherwise it closes as `dead thesis because no seat cleared promotion gate` |
| `collab_trade/strategy_memory.md` | Long-term learning memory (per-pair tendencies, pattern validity, lessons) |
| `collab_trade/memory/lesson_registry.json` | Structured lesson registry generated from `strategy_memory.md` (state/type/trust snapshot for recall and review) |
| `collab_trade/summary.md` | All-day performance · trend summary (updated at session end) |
| `logs/live_trade_log.txt` | Trade execution log (chronological) |
| `logs/trade_event_sync_state.json` | OANDA close-fill sync cursor for broker-side TP/SL/manual close notifications. Prevents duplicate #qr-trades posts while letting missed closes be recovered from transactions |
| `logs/news_digest.md` | News summary updated by Cowork hourly |
| `logs/news_cache.json` | Structured news data from API parser |
| `logs/bot_inventory_policy.md` | Legacy local-bot policy file retained for rollback/history. Not part of the routine recurring trader loop |
| `logs/bot_inventory_policy.json` | Legacy machine policy file retained for rollback/history. Not part of the routine recurring trader loop |
| `logs/technicals_*.json` | Technical cache across M1/M5/M15/H1/H4 |
| `logs/quality_audit.md` | Quality audit facts (updated every 30 min when the audit runs. Trader reads at session start and ignores stale prose if session_data flags it) |
| `logs/quality_audit.json` | Quality audit machine-readable output (for daily-review parsing) |
| `logs/audit_history.jsonl` | Append-only audit opportunity tracking (scanner fires + final narrative A/S picks. daily-review uses it for recipe and judgment accuracy) |
| `logs/s_hunt_ledger.jsonl` | Append-only trader-receipt log from each session (`S Hunt` horizons + `S Excavation Matrix` podium + reference prices). daily-review uses it to score both promoted and near-S missed seats |

### Scripts

| File | Contents |
|------|----------|
| `tools/session_data.py` | Full data fetch at trader session start (technicals M1/M5/M15/H1/H4 + OANDA + macro + **broker-side close sync** via `trade_event_sync.py --notify-slack` so TP/SL/manual fills that happened between sessions are logged and batched to #qr-trades before the trader reads the book + **Currency Pulse** (cross-currency triangulation) + **H4 Position** (lifecycle) + **Tokyo-open breadth board** when the trader is inside the Tokyo complex, so multi-lane mornings are visible instead of inferred late + Fib M5+H1 + Slack + **direction-aware actionable memory recall for held/pending/scanner pairs plus carry-forward watch targets parsed from the prior `state.md`, fresh audit strongest-unheld / narrative seats, and fresh audit range opportunities** with a wider scanner intake, while explicitly ignoring old `Podium #...`, `Lane ...`, and `id=board` echoes so stale board prose does not re-enter fresh-risk inventory + **carry-forward focus snapshot** that repeats the last `Best expression / Primary / Backup / Next fresh risk` ladder and forces a `KEEP / ROTATE / DEAD` primary continuity verdict before fresh-risk re-ranking + **SESSION INTENT GATE** so flat, pressure-free ticks stay watch-only instead of pretending to be fresh opportunity, unless a fresh audit range box is honest enough to justify a live LIMIT lane + **learning-weighted edge board / fresh-risk tournament** that now prioritizes recent matched `pretrade_outcomes` feedback before falling back to discretionary-only closed-trade history (floored at the 2026-04-17 reset), then conditions that feedback by the current live-tape bucket when samples exist, attaches repeat audit / missed-S pressure, promotes audit seats into the board as first-class `AUDIT` sources and audit range rails into the board as `AUDIT_RANGE` LIMIT sources with exact entry/TP, preserves full seat identity instead of collapsing everything into one `(pair, direction)` profile, keeps same-pair multi-seat inventory alive when trigger/vehicle differs, writes the latest payable / armable board snapshot to `logs/session_action_board.json`, computes a **LANE SHELF-LIFE BOARD** (`MARKET≈1 session`, `STOP-ENTRY≈2 sessions`, passive `LIMIT≈4-6 sessions`, armed pending=`actual GTD`), prints a **SEAT INVENTORY BOARD** plus **GOLD MINE INVENTORY** plus **PAYABLE-NOW MARKET CHECK**, and now omits the giant static state templates in normal runtime mode unless `--emit-templates` is requested while still printing a compact `HANDOFF REFRESH` block that rewrites actual fills / entry orders / rejects plus the current `Primary / Backup / Next fresh risk` ladder, their shelf-life labels, and the current `20-minute backup trigger armed NOW` line from the real pending-entry book (`id=...` or `none because ...`), before tightening pair-direction size caps with the current session bucket, live M5 regime proxy, and live pricing-probe tape quality so friction-dominated / opposite-pressure seats lose board rank before raw score + **B-band sizing split** so `B` seats surface as `B+ / B0 / B-` instead of one blunt scout bucket + **exact-pretrade proxy closure** so fresh board seats now run through the same `pretrade_check.py` orderability engine as send-time, using runtime-cached spread / live tape and proxy `entry / tp / sl` geometry before the board names `MARKET / LIMIT / STOP-ENTRY / PASS` + **S-hunt deployment cues** that pre-close the best fresh seats as `MARKET / LIMIT / STOP-ENTRY / PASS`, now choosing each bucket's board seat by actual orderability before raw score so the trader sees the best armable expression first, printing the live tape reason next to the closure state, and emitting a **PAYABLE-NOW MARKET CHECK** when no honest market lane exists + **S Excavation seeds** that prefill the near-S podium from the live fresh-risk board plus fresh audit strongest-unheld / narrative A-S calls and fresh audit range boxes when available, now retaining high-quality `one print away` PASS seats instead of dropping them from the podium entirely, attaching repeat pressure to the seed, and preferring passive `LIMIT` lanes over losing proof-chase `STOP-ENTRY` lanes when vehicle stats disagree + **multi-vehicle deployment lanes** so broad sessions can keep up to the runtime lane limit as separate expressions instead of hard-capping at three + **S Excavation Matrix template** so every pair names blocker / upgrade / dead condition before `S Hunt` + **intraday OODA / Decision Journal / Micro AAR prompts** + **Hot Updates from the prior state handoff**), all at once. Refreshes trader chart artifacts as needed, now including M1 execution charts, prints per-timeframe technical-cache ages, retries stale caches once, runs a short pricing probe for all 7 pairs, and filters the calendar section to real events in the next 4 hours |
| `tools/pricing_probe.py` | Short-burst OANDA pricing microstructure reader. Opens the bounded OANDA pricing stream for a few seconds when possible, falls back to pricing snapshots if the stream is unavailable, summarizes per-pair microstructure (`buyers pressing / sellers pressing / two-way`, spread stability, range vs friction), now separates `quiet / stable` tape from truly `friction-dominated` tape so calm but orderly microstructure is not treated as automatic market poison, writes `logs/pricing_probe.json`, and provides the live tape lane used by `session_data.py`, `mid_session_check.py`, and market-order gating |
| `tools/mid_session_check.py` | Lightweight mid-session check (~1s): Slack + prices + short live tape probe + trades + margin only |
| `tools/profit_check.py` | **Run at every session start** — 6-axis TP evaluation (ATR ratio, M5 momentum, H1 structure, correlation, S/R, peak) |
| `tools/protection_check.py` | **Run at every session start** — TP/SL/Trailing status check per ATR. NO PROTECTION = immediate action. Detects rollover window |
| `tools/rollover_guard.py` | Rollover SL guard — remove/restore SL/Trailing around daily OANDA maintenance (5 PM ET) |
| `tools/preclose_check.py` | **Run before every close** — re-confirms thesis before exit, now with post-close-regret evidence plus a required `dead layer` test (`market / structure / trigger / vehicle / aging`) so first wobble is not treated as full thesis death |
| `tools/close_trade.py` | Position close (PUT /trades/{id}/close. Prevents hedge account mistakes). Historical worker-tag safeguards remain for recovery workflows, but routine recurring trader sessions manage discretionary `trader` inventory. Now also enforces thesis-layer close discipline for trader-owned losing seats: `trigger / vehicle` damage on soft `LIMIT / STOP-ENTRY` seats defaults to `HALF / HOLD`, same-`thesis_market` siblings are treated as one inventory family, and only `market / structure / aging` death plus rotten-seat cases (`stale`, bad fill, dirty counter-reversal) get clean full-close permission without `--force-full-close`. `--auto-slack` now calls the OANDA close-fill sync so manual closes and broker-side TP/SL exits share one deduped Slack stream |
| `tools/close_discipline.py` | Shared close-discipline helper — classifies dead layer (`market / structure / trigger / vehicle / aging`), detects same-market sibling inventory, recognizes rotten-seat exceptions (stale / bad fill / dirty counter-reversal), and feeds both `close_trade.py` and runtime close checks |
| `tools/fib_wave.py` | N-wave structure + Fibonacci levels. Run at session start for all pairs |
| `tools/refresh_factor_cache.py` | OANDA candle refresh + technical cache builder (M1/M5/M15/H1/H4). Re-execs under `.venv` automatically if `python3` lacks pandas |
| `tools/chart_snapshot.py` | **Visual charts + regime detection** — generates candlestick PNG (BB/EMA/KC overlay) + detects TREND/RANGE/SQUEEZE. **Run primarily by quality-audit**; `session_data.py` refreshes the chart set as a fallback when PNGs are stale or missing. Auditor reads PNGs visually, writes Regime Map + Range Opportunities to quality_audit.md. `--all` = 7 pairs × M5+H1, `--all --with-m1` adds the trader's M1 execution set |
| `tools/oanda_performance.py` | **OANDA API-based performance analysis** — ground truth P&L, win rate, R:R, best streak, per-pair breakdown. USE THIS for any performance analysis, not grep on log files |
| `tools/post_close_regret.py` | **Post-close regret analysis** — quantifies how often losing closes later traded back to breakeven or better, using OANDA close timestamps plus post-close M1 favorable excursion. Also tags each loss by thesis lens (`archetype / wave / session / regime`), collapse component (`trigger_timing / structure / stale thesis / vehicle friction`), and live-tape bucket so review can separate market-state mistakes from vehicle/tape mistakes, and now exposes a reusable regret map for daily-review backfills |
| `tools/trade_performance.py` | Performance aggregation (legacy — parses log file). Still used by the recurring trader loop for strategy-feedback compatibility; use `oanda_performance.py` for ground-truth analysis |
| `collab_trade/memory/ingest.py` | Session memory ingestion. Splits `trades.md`/`notes.md`/historical `state.md` snapshots into direction-tagged chunks, auto-syncs `strategy_memory.md` lesson state markers from the registry review queue, re-ingests the markdown as section-level lesson chunks, refreshes `lesson_registry.json`, embeds them, and writes structured trade/user/event data to `memory.db`. Historical re-ingest must not reuse the live `collab_trade/state.md` for past dates |
| `tools/trade_event_sync.py` | OANDA transaction sync for close fills. Appends any missing `CLOSE` / `PARTIAL_CLOSE` lines to `logs/live_trade_log.txt`, batches unsynced closes into one #qr-trades `CLOSE SYNC` post, and advances `logs/trade_event_sync_state.json` to prevent duplicate noise |
| `tools/slack_trade_notify.py` | Entry / modify Slack helper retained for explicit manual posts; routine close notifications should go through `trade_event_sync.py` |
| `tools/news_fetcher.py` | News fetch (Finnhub+AlphaVantage+FF. Called from Cowork task). Normalizes event timestamps to UTC internally and renders summaries in JST so the displayed event day / countdown stay aligned with the trader's locale |
| `tools/slack_daily_summary.py` | Daily summary |
| `tools/quality_audit.py` | Quality audit — cross-checks trader decisions against rules and S-conviction data, preserves same-pair multi-horizon S-scan facts instead of collapsing them to one pair-direction representative, recognizes bare logged fill receipts and broker `STOP_FILL` receipts as real trader trades, and parses prose `## Positions (Current)` live-book lines without mistaking pending LIMIT rows for held positions |
| `tools/record_audit_narrative.py` | Parses the final Auditor's View from `logs/quality_audit.md` and appends the auditor's unheld A/S narrative picks plus structured range opportunities to `logs/audit_history.jsonl` |
| `tools/auto_hot_updates.py` | Auto-derives compressed carry-forward `Hot Updates` from the final `S Hunt` / `Capital Deployment` receipts at session end. Safety net for the next session; does not replace writing the update live when you learn it, and it must restate the actual tape/trigger blocker instead of recycling receipt-absence prose like `no live pending entry order exists` |
| `tools/record_s_hunt_ledger.py` | Parses `collab_trade/state.md` at session end and appends the trader's short / medium / long S-hunt receipts plus `S Excavation Matrix` podium seats to `logs/s_hunt_ledger.jsonl` for daily-review scoring |
| `tools/seat_outcomes.py` | Syncs `logs/s_hunt_ledger.jsonl` into `memory.db` `seat_outcomes`, now for both promoted `S Hunt` horizons and near-S `S Excavation` podium seats. Scores discovery / orderability / deployment / capture / miss from the written reference price to the best favorable excursion by the review cutoff, and exposes the review scoreboard. The scoreboard collapses a continuing same-pair seat into one chain so repeated state snapshots do not inflate deployment counts |
| `tools/state_hot_update.py` | Fast helper to prepend one compressed intraday learning bullet into `collab_trade/state.md` `## Hot Updates`, keeping only the latest carry-forward corrections for the next session |
| `tools/validate_trader_state.py` | Validates that `state.md` closes each S-hunt horizon as a real order receipt or an explicit `no seat cleared promotion gate` dead-thesis close, blocks SESSION_END if the `A/S Excavation Mandate` says `Order now` / `Arm now as` without a real `id=...` receipt or if the `S Excavation Matrix` is missing pair coverage or podium lines, cross-checks live trade / pending-order ids against the current OANDA book for the live `state.md`, fails when the live book contains trades / entry orders missing from `state.md` or `live_trade_log.txt`, re-audits live trader pending entry orders against the same exact `pretrade_check.py` geometry used at send time, and now also fails a live lock-held session when a human Slack message in `#qr-commands` is still unread or when `state.md` omits the required `## Slack Response` receipt block for that interaction. It fails underdeployed broad sessions that leave lane 2 / 3 as `trigger-only watch lane` or `none` instead of a real armed receipt or explicit dead thesis, and compares the latest `logs/session_action_board.json` snapshot against `state.md` only for session-end accountability so payable / armable lanes cannot disappear into a flat handoff without a real receipt or explicit dead-thesis contradiction. The same gate now applies to the top-level `Backup vehicle`, `Next fresh risk allowed NOW`, and required `Gold Mine Inventory` lines: stale `none` or a mismatched Gold #1-#5 is invalid when the live board still has a lane. It also hard-fails duplicate critical headings, impossible armed-count claims (`Pending orders: none` with `1 armed receipts`), multi-seat lane lines, focus lines that keep naming a seat after its lane already closed as `dead thesis because ...`, tautological dead closures inside `Best A/S ...` headers, `Gold Mine Inventory`, `S Hunt` / `Capital Deployment` deployment results, and `Hot Updates` lines that explain a seat only by missing execution. |
| `tools/task_runtime.py` | Host-neutral trader/audit runtime helper for shared prompts (Claude/Codex). `trader cycle` only auto-falls through to `mid_session_check.py` on soft `session_end.py` rejections (`TOO_EARLY`, `STATE_MD_STALE`); state-validation failures now stay loud. |
| `tools/runtime_git_sync.py` | Strict runtime git helper. Snapshots trader-session baseline dirtiness, then auto-commit/pushes only approved tracked runtime files (`state.md`, day snapshots, `lesson_registry.json`, and daily-review `strategy_memory.md`) back to `main` when no unrelated dirty paths exist |
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
| `tools/place_trader_order.py` | Discretionary trader entry helper. Places real `MARKET` / `LIMIT` / `STOP` orders directly in OANDA, auto-logs the receipt to `logs/live_trade_log.txt`, prints the exact `STATE_RECEIPT` string for `state.md`, refuses fresh risk when live-book hygiene drift exists, auto-refreshes the pair's technical cache before order send and blocks if M1/M5/M15/H1 data is still stale, runs the exact `pretrade_check.py` geometry on the real `entry / tp / sl` before any send, derives the effective `pretrade / allocation / allocation_band` from that exact result instead of trusting self-reported flags, uses live bid/ask as the entry proxy for `MARKET` orders so stop/target floors still apply, runs a short live pricing probe before `MARKET` orders and blocks friction-dominated / spread-unstable tape, enforces lot honesty against the exact allocation band, blocks oversized `counter / reversal` seats, and emergency-closes pathological STOP / MARKET fills when the actual fill spread blows out versus the planned path |
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

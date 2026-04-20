# Changelog

## 2026-04-20 — trader runtime: stop lingering watchdog overlap

- Updated `tools/task_runtime.py`
  - Added a `.trader_watchdog` pid file so the detached watchdog can be cleaned up explicitly on the next preflight or session end
  - Bound each watchdog to the session's original `.trader_start` timestamp, so an old watchdog exits instead of mistaking the next 20-minute trader session for its own
  - Stopped any stale watchdog before starting a new trader session, which prevents duplicate watchdog processes from stacking across clean runs
- Updated `tools/session_end.py`
  - Cleanly terminates the detached trader watchdog before releasing the lock, so a normal 10-13 minute session does not leave a Python watchdog process sleeping until the 17-minute timeout

## 2026-04-20 — session_data: survive missing APSW under system python

- Updated `tools/session_data.py`
  - Added a read-only `sqlite3` fallback for trade-context stats when system `python3` cannot import `apsw`
  - Restored the canonical `python3 tools/session_data.py` trader playbook path without requiring a manual `.venv` rerun
  - Verified by rerunning `python3 tools/session_data.py` successfully during the 2026-04-20 02:21 UTC trader session


## 2026-04-20 — trader gating: reduce prose-only opportunity loss

- Updated `tools/session_data.py`
  - `headwind` no longer auto-flattens every live `trending / squeeze / transition` seat; when spread is still normal and the tape is already paying, the seat can degrade into `MARKET` scout or `STOP-ENTRY`
  - Loosened live `MARKET` thresholds slightly for B seats and changed weak live trend fallbacks from passive `LIMIT` to `STOP-ENTRY`
  - Added `STATE CARRY-FORWARD WATCHLIST` parsing so prior backup / podium / rotation seats survive into the next 20-minute session instead of being rediscovered late
  - Added `TOKYO-OPEN BREADTH` so the trader sees when the morning is broad enough for more than one live lane
- Updated `collab_trade/memory/pretrade_check.py`
  - Matched the same scout/trigger logic at entry time so pretrade execution style no longer teaches a stricter flat bias than the session board
- Updated `tools/daily_review.py`
  - Audit/scanner signal review now scores each signal from the best favorable excursion inside its own signal window instead of only the current/flat close
- Updated `docs/SKILL_trader.md`
  - Clarified that `headwind` is a size cap first, not an automatic no-trade command
  - Added a required `20-minute backup trigger armed NOW` line so valid backup lanes do not remain visual-only across the next cadence
- Updated `docs/SKILL_daily-review.md`
  - Clarified that scanner/audit signal review must use the same best-favorable standard as `seat_outcomes`
- Updated `AGENTS.md`
  - Documented the new live scout/trigger downgrade for `headwind` trend seats, the Tokyo-open breadth board, and carry-forward watch targets from `state.md`
- Updated `collab_trade/strategy_memory.md`
  - Added the lessons that auto-passing live `headwind` trend seats is opportunity-loss behavior and that 20-minute cadence requires armed backup lanes, not visual-only watchlists

## 2026-04-20 — seat-outcomes: score missed seats by best favorable excursion

- Updated `tools/seat_outcomes.py`
  - Missed / not-captured seat scoring now measures `pip_move` from the written reference price to the best favorable M5 excursion by the review cutoff, instead of only the cutoff close/current price
  - Added a 1.0 pip minimum-favorable threshold so tiny flickers do not count as meaningful `directionally_correct` misses
  - Updated seat review text so reports say `best favorable excursion` instead of implying the flat close was the whole move
- Updated `docs/SKILL_daily-review.md`
  - Daily-review guidance now tells the reviewer to compare against the best favorable excursion window, not just the review-time price
- Updated `AGENTS.md`
  - Documented that `seat_outcomes` miss scoring now uses best favorable excursion by the review cutoff
- Updated `collab_trade/strategy_memory.md`
  - Added the lesson that flat-close scoring understates real missed edge

## 2026-04-20 — trader prompt: stop calling blocked seats `S Hunt`

- Updated `docs/SKILL_trader.md`
  - Added a formal `Promotion proof` line to every `S Hunt` horizon so promoted S must state which `S Excavation Matrix` blocker cleared on tape
  - Tightened `S Hunt` wording so horizons whose blocker is still alive must close as `dead thesis because no seat cleared promotion gate: ...`, not as a fake blocked S
  - Updated the mandatory output summary so `S Hunt` / `Capital Deployment` no-promotion closes stay explicit
- Updated `tools/validate_trader_state.py`
  - Validation now requires `Promotion proof:` on every `S Hunt` horizon
  - `STILL PASS` is now valid only when both `Promotion proof` and `Deployment result` explicitly say `no seat cleared promotion gate`
  - Live/orderable promoted S now must say `blocker was ... -> cleared by ...` before SESSION_END can pass
- Updated `docs/SKILL_daily-review.md`
  - `S Hunt Capture Review` now distinguishes real promoted S from honest no-promotion closes
- Updated `AGENTS.md`
  - Documented that promoted `S Hunt` horizons must show how the excavation blocker cleared, otherwise the horizon closes as `no seat cleared promotion gate`
- Updated `collab_trade/strategy_memory.md`
  - Added the lesson that a live blocker means the seat still belongs in excavation, not on the promoted S board

## 2026-04-18 — excavation review: formalize podium outcomes and open deployment

- Updated `tools/seat_outcomes.py`
  - `seat_outcomes` sync now records `S Excavation Matrix` podium seats as `source=s_excavation` in addition to promoted `S Hunt` horizons
  - Added formal excavation review rendering from `memory.db`, keeping `S Hunt` and podium reviews separate
  - Later podium deployment now counts both closed trades and still-open trades that were entered after the podium was written
- Updated `tools/daily_review.py`
  - `S Excavation Review` now reads from the formal `seat_outcomes` sync instead of ad-hoc JSONL parsing
- Updated `tools/validate_trader_state.py`
  - `S Excavation Matrix` validation now requires `Best expression` on every pair row and concrete `Closest-to-S because` / `Still blocked by` text on every podium line
- Updated `docs/SKILL_daily-review.md`
  - Documented that excavation review is now DB-backed and that open later deployments count as real deployment, not as fake misses
- Updated `AGENTS.md`
  - Documented that `seat_outcomes` now covers both `S Hunt` horizons and excavation podium seats
- Updated `collab_trade/strategy_memory.md`
  - Added the lesson that a later-open podium seat is deployed evidence, not a bookkeeping miss

## 2026-04-18 — session_data: seed the near-S podium from the live board

- Updated `tools/session_data.py`
  - Added `S EXCAVATION SEEDS` output that auto-builds default `Podium #1-#3` candidates from the live fresh-risk tournament
  - When `logs/quality_audit.md` is still fresh, the podium seed also incorporates the auditor's strongest-unheld / narrative A-S picks
  - The mandatory `S Excavation Matrix` template now pre-fills podium lines from those seeds instead of starting as a blank brainstorm
- Updated `docs/SKILL_trader.md`
  - Documented that the near-S podium should default to `S EXCAVATION SEEDS` unless the live chart provides a concrete contradiction
- Updated `AGENTS.md`
  - Documented the new session-data podium seeding behavior
- Updated `collab_trade/strategy_memory.md`
  - Added the lesson that a blank podium after the live board already ranked seats is avoidance, not openness

## 2026-04-18 — daily-review metrics: stop counting the wrong things

- Updated `tools/daily_review.py`
  - `Execution Split` no longer depends only on `tag=trader*` / `qr-trader`; it now falls back to the reviewed UTC day's `collab_trade/daily/<date>/trades.md` when a clean recurring-trader execution forgot to persist a tag/comment
  - Added ownership-source reporting so the review can separate explicit recurring evidence, daily-record fallback, explicit non-recurring tags, and unresolved trades
  - Reworked audit opportunity scoring to keep timestamped signals separate instead of collapsing same pair/direction calls into one bucket
  - Audit outcome analysis now links entries inside each signal window and can label losing-but-later-right trades as `PREMATURE`
- Updated `tools/seat_outcomes.py`
  - `Horizon Scoreboard` now collapses a continuing same-pair seat into one chain, so repeated state snapshots do not inflate `discovered / orderable / deployed`
- Updated `docs/SKILL_daily-review.md`
  - Documented the daily-trade-record ownership fallback, per-timestamp audit scoring rule, and chain-based horizon scoreboard
- Updated `AGENTS.md`
  - Documented the revised `daily_review.py` ownership/audit logic and the chain-based `seat_outcomes.py` scoreboard

## 2026-04-18 — S excavation: make the near-S board reviewable

- Updated `tools/validate_trader_state.py`
  - `S Excavation Matrix` now fails validation if it still contains `___` placeholders
  - `Podium #1-#3` must now name a concrete `PAIR LONG/SHORT` seat and the upgrade action (`MARKET / LIMIT / STOP-ENTRY`)
- Updated `tools/record_s_hunt_ledger.py`
  - `logs/s_hunt_ledger.jsonl` now stores `S Excavation Matrix` pair rows and podium seats in addition to `S Hunt` horizons
- Updated `tools/daily_review.py`
  - Added `S Excavation Review` so daily-review scores whether the near-S podium seats were later deployed, right without deployment, or correctly blocked noise
- Updated `docs/SKILL_trader.md`
  - Tightened the podium format to require a real pair and direction so the next day's review can score it
- Updated `docs/SKILL_daily-review.md`
  - Added explicit `S Excavation Review` instructions so near-S misses become memory evidence instead of disappearing after the session
- Updated `AGENTS.md`
  - Documented that `s_hunt_ledger.jsonl` now carries both promoted S horizons and near-S podium seats

## 2026-04-18 — prompt + session_data: restore multi-currency deployment lanes

- Updated `tools/session_data.py`
  - Added `MULTI-VEHICLE DEPLOYMENT LANES` so the session output can now name up to three concurrent fresh-risk lanes instead of collapsing everything into a single live seat
  - Lane selection prefers separate pairs, separate base currencies, and separate vehicle buckets when possible, while still skipping `PASS` seats
  - Extended the output template with `Multi-Vehicle Deployment` and a `PRIMARY / BACKUP / THIRD CURRENCY` lane field
- Updated `docs/SKILL_trader.md`
  - Documented that broad sessions may legitimately carry 2-3 currencies if they are separate expressions and worst-case margin stays below 90%
  - Expanded `Capital Deployment` and the state template so multi-lane deployment is an explicit receipt, not an afterthought
- Updated `AGENTS.md`
  - Documented that `session_data.py` now emits multi-vehicle deployment lanes in addition to the best-seat cues
- Updated `collab_trade/strategy_memory.md`
  - Added the lesson that broad sessions should not be compressed into one pair when multiple currencies are genuinely alive

## 2026-04-18 — session_data + pretrade_check: widen live market lane for real opportunities

- Updated `tools/session_data.py`
  - Broadened scanner-derived memory targets from the top 3 seats to the top 6, with a larger overall fresh-seat intake, so live opportunities are not over-pruned before deployment cues run
  - Reworked deployment-cue logic so `MARKET` is no longer reserved for only the narrowest trending A/S lane
  - Added pair-normal spread bands plus a `B conviction + live tape` market-scout lane, and now allow live `MARKET` participation for strong `squeeze` / `transition` seats when the tape is already leaning one way
- Updated `collab_trade/memory/pretrade_check.py`
  - Matched the execution-style gate to the new deployment-cue logic so the pre-entry tool no longer re-blocks the same valid live market seats
  - Added the same pair-normal spread treatment and market-scout note for B-conviction live entries
- Updated `docs/SKILL_trader.md`
  - Clarified that `MARKET` on a B seat means scout first plus reload `LIMIT`, not a full-size chase
- Updated `AGENTS.md`
  - Documented the wider scanner intake and market-scout deployment lane in `session_data.py`
- Updated `collab_trade/strategy_memory.md`
  - Added the lesson that under-deployment can come from making the `MARKET` lane too narrow even when the live tape is already leaning

## 2026-04-18 — trader prompt: force S excavation before S Hunt

- Updated `docs/SKILL_trader.md`
  - Added a required `S Excavation Matrix` between the 7-pair scan and `S Hunt`
  - Each pair must now name the best expression, the exact blocker that keeps it below S, the exact print that upgrades it into S, and the death condition that kills the idea
  - Added a `Podium #1-#3` so the nearest undiscovered S seats stay visible even if they are not promoted into the short / medium / long horizon board
  - Updated the state template and time-allocation flow so `S Excavation Matrix` is part of the mandatory handoff
- Updated `tools/session_data.py`
  - Added a matching `S Excavation Matrix` template to the mandatory output block
- Updated `tools/validate_trader_state.py`
  - For 2026-04-18+ state handoffs, `SESSION_END` now fails if the `S Excavation Matrix` is missing pair coverage or podium lines
- Updated `AGENTS.md`
  - Documented `S Excavation Matrix` in the live state handoff, trader runtime, and validation flow
- Updated `collab_trade/strategy_memory.md`
  - Recorded the process lesson that S is often undiscovered, not absent, and that each pair now needs blocker / upgrade / death wording before the session can call the market "no S"

## 2026-04-18 — session_data: deployment cues close every fresh S as an order state

- Updated `tools/session_data.py`
  - Added execution-style cues on top of the learning edge board so each fresh-risk seat now closes by default as `MARKET`, `LIMIT`, `STOP-ENTRY`, or `PASS`
  - Added `S-HUNT DEPLOYMENT CUES` for the best direct-USD / cross / USD_JPY seats so valid S ideas stop dying as prose-only waits
  - Extended the mandatory new-entry template with an `Execution cue from session_data` line and an explicit `exact structural level / exact trigger` field when the closure is not `MARKET`
- Updated `docs/SKILL_trader.md`
  - Documented the new deployment-cue section and made it the default closure source for `S Hunt`
- Updated `AGENTS.md`
  - Documented that `session_data.py` now emits S-hunt deployment cues, not only learning caps

## 2026-04-18 — pretrade_check: learning gate + execution style now persist into DB

- Updated `collab_trade/memory/pretrade_check.py`
  - Added a learning-aware pretrade profile using `lesson_registry.json`, pair-direction history, current UTC session bucket, and the current regime proxy
  - Caps allocation with the learning verdict instead of leaving the learning board as read-only context
  - Emits a concrete execution-style recommendation per seat: `MARKET`, `LIMIT`, `STOP-ENTRY`, or `PASS`
  - Persists those decisions into `pretrade_outcomes` so later review can measure not only whether the seat won, but whether the gate and order style were right
- Updated `collab_trade/memory/schema.py`
  - Added `pretrade_outcomes` columns for `learning_score`, `learning_verdict`, `learning_cap`, `session_bucket`, `regime_snapshot`, `execution_style`, and `execution_note`
  - `init_db()` now supports `quiet=True` so runtime tools can apply migrations without noisy output
- Updated `docs/SKILL_trader.md`
  - Added a required `Execution style from pretrade_check` field in the pre-entry block
  - Documented that `LIMIT` / `STOP-ENTRY` is the path to higher entry frequency without degrading entry quality into bad market orders
- Updated `AGENTS.md`
  - Documented that `pretrade_outcomes` now stores the learning and execution metadata used at the moment of entry

## 2026-04-18 — state hot updates: intraday learning carried into the next session

- Updated `tools/session_data.py`
  - Reads `## Hot Updates` from `collab_trade/state.md` and surfaces those bullets at the top of the next session as the first carry-forward learning read
  - Added a `Hot Updates` template in the mandatory state handoff block so `Micro AAR` can be compressed into one-line next-seat corrections
- Added `tools/state_hot_update.py`
  - Fast helper to prepend one carry-forward learning line into `collab_trade/state.md` without manually re-editing the whole file
- Updated `collab_trade/memory/ingest.py`
  - State snapshots now ingest `Hot Updates` bullets as `lesson` chunks so intraday corrections survive into memory snapshots and future recall
- Updated `docs/SKILL_trader.md`
  - Added the `Hot Updates` operating rule and state template section so intraday PDCA is explicitly carried forward
- Updated `AGENTS.md`
  - Documented `Hot Updates` as part of the state handoff and trader runtime

## 2026-04-18 — hot updates auto-sync + formal seat-outcome DB

- Added `tools/auto_hot_updates.py`
  - Derives carry-forward `Hot Updates` automatically from the final `S Hunt` / `Capital Deployment` receipts at session end
  - Replaces prior hot-update bullets seat-by-seat so the next session sees the latest live-seat correction instead of stale prose
- Updated `tools/session_end.py`
  - Runs `auto_hot_updates.py` before the state snapshot and memory ingest
  - Syncs formal `seat_outcomes` after the session-end ledger append so the discovery / deployment / capture chain is persisted immediately
- Added `tools/seat_outcomes.py`
  - Syncs `logs/s_hunt_ledger.jsonl` into `memory.db` `seat_outcomes`
  - Scores each horizon as `discovered / orderable / deployed / captured / missed`
  - Exposes a review-style scoreboard CLI for verification and daily-review reuse
- Updated `collab_trade/memory/schema.py`
  - Added the `seat_outcomes` table and indexes to `memory.db`
- Updated `tools/daily_review.py`
  - `S Hunt Capture Review` now syncs the formal `seat_outcomes` table first, then reads the scoreboard from SQL instead of ad-hoc JSONL parsing
- Updated `tools/state_hot_update.py`
  - Added `--state` and `--replace-prefix` support so auto-generated and manual hot updates can safely replace seat-specific carry-forward bullets
- Updated `docs/SKILL_trader.md`, `docs/SKILL_daily-review.md`, and `AGENTS.md`
  - Documented the session-end hot-update safety net and the formal seat-outcome review chain

## 2026-04-18 — intraday learning loop: OODA / Decision Journal / AAR / Bayesian update

- Updated `tools/session_data.py`
  - Added `INTRADAY LEARNING LOOP (OODA + DECISION JOURNAL)` for the top fresh-risk seats so the trader writes Observe / Orient / Decide / Act before the order
  - Added `Micro AAR` prompt so each entry / exit / miss can update the next seat immediately instead of waiting for daily-review
- Updated `tools/daily_review.py`
  - Added `Bayesian Evidence Update` so daily-review treats today's outcome as evidence for or against an existing prior instead of rewriting memory from one trade
  - Added `After Action Review Queue` for the biggest win, biggest loss, and repetition candidate
- Updated `docs/SKILL_trader.md`
  - Added the intraday learning-loop workflow and an `OODA / Decision Journal` section to the state handoff template
- Updated `docs/SKILL_daily-review.md`
  - Daily review now explicitly reads the Bayesian evidence and AAR sections before promoting or demoting lessons
- Updated `AGENTS.md`
  - Documented that trader runtime now includes intraday OODA / Micro AAR prompts and daily-review emits Bayesian / AAR review aids

## 2026-04-18 — trader runtime: learning-weighted edge board + fresh-risk tournament

- Updated `tools/session_data.py`
  - Added a `LEARNING EDGE BOARD` that turns pair-direction lessons plus trade stats from `lesson_registry.json` into a live verdict and allocation cap for each held / pending / scanner seat
  - Added a `FRESH-RISK TOURNAMENT` that ranks current non-held seats and names the best direct-USD / cross / USD_JPY learning-weighted vehicles
  - Extended the trader's prefilled entry template with `Learning verdict`, `Learning cap`, and `Tournament rank` so pair memory directly constrains size, not just post-hoc reflection
- Updated `docs/SKILL_trader.md`
  - Documented how to use the edge board as a size gate on top of the market read
  - Made `Learning cap` and `Tournament rank` explicit fields in the entry contract
- Updated `AGENTS.md`
  - Documented that `session_data.py` now converts lesson history into pair-direction size caps for live vehicle selection

## 2026-04-18 — lesson registry: state/type/trust for strategy memory

- Added `collab_trade/memory/lesson_registry.py`
  - Extracts a structured registry from `strategy_memory.md`
  - Assigns each lesson a `state` (`candidate/watch/confirmed/deprecated`), `lesson_type`, and `trust_score`
  - Writes `collab_trade/memory/lesson_registry.json` for runtime/review use
- Updated `collab_trade/memory/ingest.py`
  - Strategy-memory refresh now also rebuilds `lesson_registry.json`
  - Re-ingest now auto-syncs explicit lesson state markers back into `strategy_memory.md` before chunking, so markdown, registry, and vector memory stay aligned after daily-review
  - Strategy lesson chunks now strip `[WATCH]`/`[CONFIRMED]` markers from embedded text and keep the state only as metadata tags, avoiding marker noise in semantic recall
- Updated `tools/session_data.py`
  - `ACTIONABLE MEMORY` now merges concrete trade memory with trusted registry lessons so recall includes not only what happened before, but which lesson is currently trusted
- Updated `tools/daily_review.py`
  - Added `Lesson State Suggestions` so daily-review gets a registry-backed queue of promotion/demotion candidates instead of editing `strategy_memory.md` from prose intuition alone
- Updated `docs/SKILL_daily-review.md`
  - Daily review now reads `lesson_registry.json` and treats lesson promotion as a state-machine update instead of prose-only editing
  - Documented that the post-review `ingest.py --force` step auto-enforces lesson-state markers before rebuilding memory
- Updated `AGENTS.md`
  - Documented the new lesson-registry layer in the memory stack and the ingest-time marker sync

## 2026-04-18 — strategy-memory section ingest + daily-review promotion gate

- Updated `collab_trade/memory/ingest.py`
  - `strategy_memory.md` is now re-ingested on every memory refresh as multiple section-level lesson chunks instead of being absent from vector memory
  - Pair/direction metadata is attached where possible so recall can surface distilled pair learnings without mixing opposite-side trade chunks
- Updated `tools/daily_review.py`
  - Added `Memory Promotion Gate` output so recurring-trader evidence and quarantined non-recurring execution are separated before `strategy_memory.md` is edited
- Updated `docs/SKILL_daily-review.md`
  - Tightened the writing contract so new pair/direction lessons must be justified by the promotion-gate evidence set and quarantine evidence cannot be promoted directly
- Updated `AGENTS.md`
  - Documented section-level `strategy_memory.md` ingest and the new promotion-gate review flow

## 2026-04-18 — memory precision cleanup: direction-tagged recall + historical state hygiene

- Updated `collab_trade/memory/schema.py`
  - Added `chunks.direction` plus a `(pair, direction)` index so recall can filter by actual trade side instead of `LIKE '%LONG%'`
- Updated `collab_trade/memory/ingest.py`
  - Trade chunks now derive from `parse_trades()` instead of loose keyword heuristics, cutting memory chunk noise sharply
  - Chunks now store explicit direction metadata
  - Historical re-ingest now uses `daily/<date>/state.md` when present and otherwise only ingests the live `collab_trade/state.md` for today, preventing current-state leakage into past dates
  - Re-ingested the full memory corpus after the schema/chunking change
- Updated `collab_trade/memory/recall.py`
  - Direction filtering now uses structured metadata and excludes directionless `trade` chunks from same-side recall, reducing opposite-side contamination
- Updated `collab_trade/memory/pretrade_check.py`
  - Suppresses identical unmatched pretrade logs within a short window so `pretrade_outcomes` reflects distinct decisions instead of repeated probes
- Updated `tools/daily_review.py`
  - Added duplicate unmatched pretrade-probe cleanup before outcome matching so legacy spam does not keep polluting review statistics
- Updated `tools/session_end.py`
  - `SESSION_END` now archives `collab_trade/state.md` into `collab_trade/daily/<UTC-date>/state.md` before ingest so future historical re-ingest has a real day-specific thesis snapshot
- Updated `tools/session_data.py`
  - `ACTIONABLE MEMORY` ranking now prefers recent, lesson-rich trade memory and de-prioritizes stale/cancel/position noise
  - Fixed an old recency bug where memory compression favored older dates instead of newer ones
- Verification
  - Full `memory.db` re-ingest reduced chunk count from `797` to `459`
  - `trade` chunks with explicit direction now cover `356/356` rows
  - `ACTIONABLE MEMORY` no longer surfaces current `state.md` under historical dates after re-ingest
  - First `daily_review.py` run after the change pruned unmatched `pretrade_outcomes` from `668` to `204`

## 2026-04-18 — daily-review date boundary + recurring-trader split

- Updated `tools/daily_review.py`
  - The default review target is now the most recent completed UTC trading day instead of the in-progress current day
  - OANDA fetch windows are now labeled explicitly in UTC and JST so review output cannot silently mix date conventions
  - Added `Execution Split` output that separates recurring trader (`tag=trader*` / `qr-trader`) from other or unknown-tag execution before lessons are written
  - Enriched closed-trade parsing with tag/comment recovery from OANDA clientExtensions and `logs/live_trade_log.txt`
- Updated `docs/SKILL_daily-review.md`
  - The playbook now computes `REVIEW_DAY` as the most recent completed UTC trading day and reuses it for `daily_review.py`, `trades.md`, and `ingest.py`
  - Reflection now starts from the recurring-trader vs other-execution split so strategy memory does not learn from the wrong flow
- Updated `docs/SKILL_trader.md`
  - Tightened `S Hunt` / `Deepening Pass` wording so a valid trigger beyond live price must close as `STOP-ENTRY`, `LIMIT`, or `dead thesis because ...`, never prose-only `STILL PASS`
- Updated `AGENTS.md`
  - Documented that `daily-review` now reviews the most recent completed UTC trading day and splits recurring-trader vs other execution before distillation

## 2026-04-18 — trader runtime: block prose-only flat books at SESSION_END

- Added `tools/validate_trader_state.py`
  - Validates `collab_trade/state.md` before `SESSION_END`
  - Blocks prose-only closures such as `armed mentally only`, `retest only`, `breakout only`, and flat books with zero real `id=` receipts across `S Hunt` / `Capital Deployment`
- Updated `tools/session_end.py`
  - Runs `validate_trader_state.py` before lock release so invalid flat-book receipts fail fast and force a deeper deployment decision
- Updated `tools/record_s_hunt_ledger.py`
  - Hardened the parser for live `state.md` phrasing (`Why S:`, `Exact trigger:`, `Invalidation:`)
  - `build_entry()` now accepts a custom state path so validation and future tooling can reuse the same parser
- Updated `docs/SKILL_trader.md`
  - Tightened `S Hunt` and `Capital Deployment` so each horizon must close as `id=...` or `dead thesis because ...`
  - Added explicit rejection of prose-only closures and documented the new `STATE_VALIDATION_FAILED` loop
- Updated `AGENTS.md`
  - Documented the new `validate_trader_state.py` runtime gate in the recurring trader path
- Updated `collab_trade/strategy_memory.md`
  - Recorded the lesson that a flat handoff with zero `id=` receipts is fake participation, not deployment

## 2026-04-18 — trader runtime: add M1 execution artifacts + S-hunt capture ledger

- Updated `tools/chart_snapshot.py`
  - Added optional `--with-m1` support so the trader runtime can generate an M1 execution set alongside the default M5/H1 chart pack
  - Switched matplotlib to lazy import so lightweight paths such as `--regime-only` stay usable even when host `python3` lacks the plotting stack
- Updated `tools/session_data.py`
  - Trader fallback chart refresh now requests `.venv/bin/python tools/chart_snapshot.py --all --with-m1`
  - Chart status output now explicitly reports the M1/M5/H1 execution set
- Added `tools/record_s_hunt_ledger.py`
  - Parses `collab_trade/state.md` and appends short / medium / long horizon receipts to `logs/s_hunt_ledger.jsonl`
  - Captures deployment result, trigger/invalidation, and cached technical reference prices so daily-review can score misses vs captures
- Updated `tools/session_end.py` and `tools/task_runtime.py`
  - Session close and stale-lock recovery now best-effort persist the S-hunt ledger before ingest
- Updated `tools/daily_review.py`
  - Added `S Hunt Capture Review` sourced from `logs/s_hunt_ledger.jsonl`
  - Daily review now scores horizon discovery vs deployment vs actual capture using closed trades plus current-price direction checks
- Updated `docs/SKILL_trader.md`
  - Trader must now open M1 PNGs for the primary / backup / best direct-USD / S-hunt pairs
  - Each S-hunt horizon now requires a `Deployment result` receipt (entered / armed / dead thesis)
  - Capital Deployment now explicitly mirrors the horizon receipt so discovery and execution cannot drift apart
- Updated `docs/SKILL_daily-review.md`
  - Daily review now reads `logs/s_hunt_ledger.jsonl` and writes at least one S-hunt capture observation into `strategy_memory.md`
- Updated `AGENTS.md`
  - Documented the M1 trader chart set and the new `logs/s_hunt_ledger.jsonl` / `tools/record_s_hunt_ledger.py` runtime artifacts
- Updated `collab_trade/strategy_memory.md`
  - Recorded the lesson that M1 trigger quality requires real M1 artifacts and that each S-hunt horizon must close as entered, armed, or dead thesis

## 2026-04-18 — trader prompt: kill trigger drift and prose-only clean seats

- Updated `docs/SKILL_trader.md`
  - Added required `Last session trigger audit` and `Best direct-USD seat NOW` lines to the `Market Narrative` block
  - Extended the `Capital Deployment Check` receipt with trigger-audit resolution and explicit direct-USD deployment status
  - Tightened flat-book accountability so already-traded triggers must resolve as `ENTER NOW`, `LEAVE STOP-ENTRY`, or `MISSED — wait first retest` instead of being silently rewritten at a worse price
  - Added `STOP-ENTRY + reload LIMIT` guidance for reclaim / breakout expressions that can fire between 20-minute trader sessions
  - Added a concrete OANDA `STOP` order example so the new `STOP-ENTRY` path is executable, not just described
  - Added required `MTF chain` and `6-category evidence` fields to Tier 1, no-trade best-seat review, and the pre-entry conviction block so `none` cannot be justified from only M5 shape + one trigger
  - Added `Deepening Pass (TOO_EARLY only)` and rewired the runtime buffer so the extra window must compare best direct-USD vs best cross with full MTF/6-category evidence instead of narrating the timer
  - Clarified that a `TOO_EARLY` session is still incomplete until `state.md` contains the `Deepening Pass` receipt, so a timer-only wait no longer counts as a clean handoff
  - Added a mandatory `S Hunt` block so every session must name the best short-term, medium-term, and long-term S candidate before capital deployment is considered complete
- Updated `collab_trade/strategy_memory.md`
  - Recorded the distilled lesson that trigger drift is avoidance and that cleaner direct-USD seats must be deployed or explicitly rejected on tape grounds
  - Recorded the new lesson that under-reading across timeframes/categories makes S disappear artificially
  - Recorded the new lesson that the TOO_EARLY buffer must produce a second-pass market read, not compliance theater

## 2026-04-17 — English-only prompt refresh + host metadata sync

- Refreshed active agent-facing docs into English-only current-state versions
  - Rewrote `README.md` around the live v8.5 architecture instead of the old multi-agent Claude stack
  - Rewrote `docs/TRADER_PROMPT.md` as a judgment-only mental-model document and removed stale runtime/cadence guidance
  - Rewrote `collab_trade/CLAUDE.md` as a concise English collaborative-trading guide
  - Updated `AGENTS.md` / `CLAUDE.md` document maps so the canonical recurring contract is `docs/SKILL_trader.md`, while `TRADER_PROMPT.md` is explicitly non-runtime
- Removed deprecated Japanese prompt copies from live QuantRabbit Claude task directories
  - Deleted `SKILL_ja.md` from `trader`, `daily-review`, `daily-performance-report`, `daily-slack-summary`, and `intraday-pl-update`
  - Updated the active `schedule.json` descriptions in `~/.claude/scheduled-tasks/*` to match canonical prompt frontmatter exactly
- Hardened sync verification
  - `tools/check_task_sync.py` now also fails on deprecated `SKILL_ja.md` files and stale Claude `schedule.json` descriptions
  - Updated stale helper wording in `tools/mid_session_check.py` and `tools/news_fetcher.py` so comments match the current recurring runtime

## 2026-04-17 — trader consistency pass (preserve working runtime, remove split-brain)

- Updated `docs/SKILL_trader.md`
  - Reframed the trader prompt around the shared lock-based runtime instead of one hardcoded host cadence
  - Removed the dead bot-inventory / worker-policy workflow from the live trader path
  - Aligned the timing section with the actual runtime gates: 10-minute minimum session, 15-minute stale-lock threshold
- Updated `AGENTS.md` and `CLAUDE.md`
  - Separated live recurring trader behavior from disabled legacy bot tooling
  - Corrected host-specific scheduler notes (`Codex` 20-minute trader automation, `Claude` 10-minute trader schedule)
  - Corrected stale references such as `1-min cron`, bot-live script tables, and quality-audit freshness wording
- Updated `tools/task_runtime.py`
  - Stale-lock ingest now uses the previous session timestamp to derive the UTC session date
  - `trader cycle` now falls back to `mid_session_check.py` only on soft `session_end.py` rejections, not on hard runtime errors
- Updated `tools/session_data.py`
  - Prints the real `state.md` modification timestamp/age instead of echoing the current session time as if it were a file timestamp
- Updated `tools/check_task_sync.py`
  - Treats `~/.claude/scheduled-tasks/*.DISABLED/SKILL.md` compatibility links as valid sync points so intentionally disabled prompts no longer produce false failures

## 2026-04-17 — Shared OANDA config loader + auth smoke test

- Added `tools/config_loader.py` to centralize `config/env.toml` parsing and normalize accepted OANDA key aliases (`oanda_token`, `OANDA_TOKEN`, `OANDA_API_KEY`, `api_key`)
- Added `tools/oanda_auth_check.py` for a read-only OANDA auth summary check instead of ad-hoc snippets
- Updated `tools/session_data.py`, `tools/profit_check.py`, `tools/quality_audit.py`, and `tools/market_monitor.py` to use the shared loader and honor `oanda_practice`

## 2026-04-17 — Trader prompt: expose benchmark-pressure chasing + remove stale Codex trader automation

- Updated `docs/SKILL_trader.md`
  - Added explicit `Benchmark pressure`, `Tempting P&L-driven trade`, and `Next fresh risk allowed NOW` lines to the required `Market Narrative` block
  - Added matching fields to the `Capital Deployment Check` receipt so each session must separate the tempting recovery trade from the actually valid next seat
  - Clarified that the benchmark changes the quality bar, not the pair selection
- Updated `collab_trade/strategy_memory.md`
  - Added the distilled lesson that benchmark deficit is not a setup and may only tighten the quality bar
- Removed the stale paused Codex trader automation `qr-hourly-trader-2` so `qr-trader` remains the only Codex trader automation path

## 2026-04-17 — ALL BOTS REMOVED — discretionary-only architecture

**Decision by user after 7-day P&L analysis**: bots were net-negative. Per-tag breakdown over last 7 days:

| tag | trades | WR | P&L | EV/trade |
|-----|--------|-----|-----|----------|
| **trader (discretionary)** | 128 | 51.6% | **+9,325** | **+73** |
| trend_bot_market | 30 | 17% | -2,458 | -82 |
| range_bot | 5 | 20% | -494 | -99 |
| range_bot_market | 1 | 0% | -88 | -88 |

Trader's +9,325 was being partially eaten by bot's -3,040 losses. Removing bots is projected to improve daily EV by ~420 JPY without sacrificing trader-side wins. Speed ("爆速で稼ぐ") is now interpreted as **profit-accumulation speed**, not trade frequency — bots were adding trades while subtracting profit.

**Removed**:
- **launchd**: `com.quantrabbit.local-bot-cycle` (60-sec cycle running trend_bot, range_bot, bot_trade_manager, inventory_brake, regime_switch, bot_policy_guard, stranded_drain). Uninstalled via `scripts/uninstall_local_bot_launchd.sh`.
- **Scheduled tasks** (renamed to `*.DISABLED` in `~/.claude/scheduled-tasks/`): `range-bot/`, `bot-trade-manager/`, `inventory-director/`. Can be restored by renaming back if needed.
- **Not removed**: `com.quantrabbit.reaper` launchd (generic stale-agent cleanup, not bot-specific). Tools in `tools/` are retained but uninvoked (`trend_bot.py`, `range_bot.py`, `bot_trade_manager.py`, `inventory_brake.py`, `regime_switch.py`, `stranded_drain.py`, `bot_policy_guard.py`, `range_scalp_scanner.py`, `bot_inventory_snapshot.py`, `render_bot_inventory_policy.py`).

**Trader optimized for discretionary speed**:
- Cron `*/20 * * * *` → `*/10 * * * *` (20-min → 10-min).
- Session budget description: 15 min → 8 min in `schedule.json` + SKILL_trader banner. Actual session_end enforcement governed by `task_runtime.py` (unchanged for this commit).
- `docs/SKILL_trader.md`: added top-of-file override banner instructing future trader sessions to skip bot-inventory workflow, tag all entries as `trader`, and ignore any bot / worker references below.
- `CLAUDE.md`: scheduled-task table rewritten. Bot architecture section replaced with removal note. Cycle references 1-min / 15-min → 10-min. Tag taxonomy simplified (all live entries = `trader`; historical bot tags read-only).
- `collab_trade/state.md`: prominent architecture-change notice so next trader session sees it immediately.

**OANDA state at removal**: 0 bot-tagged open trades, 0 bot-tagged pending orders. Clean handoff.

**Reversibility**: To restore bots, rename `*.DISABLED` dirs back in scheduled-tasks, reinstall launchd plist (plist template still at `scripts/install_local_bot_launchd.sh`), restore the pre-banner SKILL_trader content via git. All tooling and state files retained.

## 2026-04-17 — bot_trade_manager: stop reaping range_bot LIMITs (the *actual* "nothing fills" cause)

**Forensic dig after user "深掘りして"**: reviewed `logs/live_trade_log.txt` — turns out range_bot WAS firing after the earlier fixes (18 LIMITs placed between 02:39-03:00 UTC: EUR_CHF×15, EUR_GBP×1, NZD_USD×2). Zero filled. All cancelled within 25 minutes by `bot_trade_manager`:

```
reason=projected margin 93.3% > 90%   ← 2 LIMITs
reason=policy PAUSE pending=CANCEL    ← many EUR_CHF
reason=trader_worker_policy           ← early EUR_CHF
```

**Why this is wrong for range_bot**:
- range_bot LIMITs sit at BB extremes (far from live price). Low simultaneous-fill probability.
- Structural SLs 3-5pip wide. Actually-filled risk per LIMIT is small.
- `projected_pct` treats every pending LIMIT as "will fill immediately" → inflates margin estimate → kills range_bot first.
- PAUSE cancellation for range_bot contradicts the range_bot.py placement bypass shipped in the earlier commit (which allows CLEAN_RANGE to bypass PAUSE for LIMIT-only). bot_trade_manager was canceling what range_bot had just placed.

**Fix**: `tools/bot_trade_manager.py` pending cleanup:
- Added `is_range_bot_limit = order_tag in {"range_bot", "range_bot_market"}`.
- `projected_pct > panic_cap` cancellation: exempt range_bot LIMITs (risk is bounded by their own tight SL).
- PAUSE mode cancellation: exempt range_bot LIMITs (they're mean-reversion, directional mode doesn't apply). Explicit `pending=CANCEL` director kill still cancels (respects explicit intent).
- Rollover, staleness, pair-already-filled paths unchanged — those safety properties still apply.

**Expected effect**: range_bot LIMITs now survive long enough to be tested. If the strategy has edge, trades fill and reach TP (now allowed by the earlier timeout fix). If not, SL-based losses will prove the strategy is broken — first real signal possible.

**Files changed**: `tools/bot_trade_manager.py`

## 2026-04-17 — range_bot: stop force-closing before TP (the real reason ranges weren't earning)

**Diagnosis after user challenge "レンジで稼げないの？"**: pulled 7-day tag-level P&L. range_bot: 5 trades, WR 20%, **avgW +6 JPY, avgL -137 JPY, R:R 0.04** — wins existed but were microscopic. Hold-time forensics on all 5: 1.4min / 1.5min / 7.2min / 12.6min / 14.9min. None reached TP (BB mid on H1 = typically 60-120 min to fill). 3 of 5 were `MARKET_ORDER_TRADE_CLOSE` (forced close by bot_trade_manager), 2 were SL hits.

**Root cause**: `bot_trade_manager.scalp_timeout_profile` used generic PASSIVE profile for range_bot (FAST: 12/19 min, BALANCED: 20/32 min). Designed for scalp-worker lanes that must turn over fast. range_bot's TP horizon is an order of magnitude longer. **Every range_bot trade was force-closed before its TP became reachable.**

**Fix**: `tools/bot_trade_manager.py` — new range_bot-specific timeout profile. `scalp_timeout_profile()` checks tag first; if tag in {range_bot, range_bot_market}, returns `{stale_min: 75, full_close_min: 150, min_progress: 0.35}`. Matches H1 BB-mid fill horizon. Scalp profiles unchanged for trend_bot.

**Expected effect**: range_bot trades now allowed to reach TP. If the strategy's edge is real (BB extremes mean-revert to BB mid), it should show up in R:R. If it's not, this will prove the strategy is broken (a faster diagnosis than the current noise).

**Files changed**: `tools/bot_trade_manager.py`

## 2026-04-17 — 3-layer fix for bot-churn spiral (SQUEEZE lockout + safe-harvest + extended S-only)

**Incident**: On 2026-04-17, trend_bot stacked positions in SQUEEZE regime (all 7+ pairs compressed). Extended universe (NZD/AUD_NZD/EUR_CHF/EUR_GBP/NZD_JPY) posted 0% WR. Margin reached 92.8% → `inventory_brake` panic-closed all worker positions. Director set global `REDUCE_ONLY` — which also locked out `range_bot`, the one strategy that belongs in SQUEEZE. Net 2026-04-17: -3,462 JPY, WR 17.1% across 41 trades.

**Root cause**: trend_bot and range_bot share the same safety gates. trend_bot's self-immolation (entering SQUEEZE false-breakouts it can't win) triggers defenses that disable range_bot. The two strategies fight each other.

**Fix**: 3 surgical layers.

**Layer 1 — trend_bot SQUEEZE lockout** (root-cause block on the bleed source):
- `tools/trend_bot.py`: new `_is_tf_squeeze()` helper (BB width < KC width × 0.9). `scan_trends()` skips pairs where both M5 AND H1 are in SQUEEZE. These are range_bot's territory — momentum continuation has no signal there.
- Logged as "Squeeze lockout: ... → range_bot territory" in output.

**Layer 2 — range_bot REDUCE_ONLY safe-harvest** (unlock paralysis after trend_bot blow-ups):
- `tools/range_bot.py`: new `safe_harvest_mode` flag = (global REDUCE_ONLY AND margin < 60%). When active, `is_mean_reversion` opportunities pass the reduce-only gate. MARKET orders force-downgrade to LIMIT (no chasing under defense).
- `is_mean_reversion` widened to accept FORMING_RANGE/SEMI_RANGE with A/S conviction + ADX<27 (prior was CLEAN_RANGE only). Captures more chop-harvest opportunities while excluding B-confidence noise.

**Layer 3 — trend_bot extended universe S-only** (stop the bleed at source):
- `tools/trend_bot.py`: new `STANDARD_7_PAIRS` constant (USD_JPY, EUR_USD, GBP_USD, AUD_USD, EUR_JPY, GBP_JPY, AUD_JPY). For pairs outside this set, require `conviction == "S"` to proceed. B and A rejected. Directly blocks the extended-universe churn (CAD_JPY, NZD_JPY, AUD_NZD, EUR_CHF, EUR_GBP, NZD_USD, USD_CHF) unless signal is overwhelming.

**Smoke tests**:
- `python3 tools/range_bot.py --dry-run`: 1 EUR_CHF S-LONG LIMIT 4500u @ BB lower placed (first real post-crash placement — previously 100% blocked).
- `python3 tools/trend_bot.py --dry-run`: "Squeeze lockout: AUD_USD, NZD_USD → range_bot territory". Trends filtered 7→3 after extended S-only gate.
- Both `python3` and `.venv/bin/python` verified.

**Files changed**: `tools/range_bot.py`, `tools/trend_bot.py`, `tools/brake_gate.py` (earlier commit, rolled into context).

## 2026-04-17 — range_bot: harvest SQUEEZE via directional-policy + CAUTION-brake bypass

**Problem**: SQUEEZE regime (all 7 pairs compressed) = range_bot's ideal hunting ground, but 6 CLEAN_RANGE opportunities found by `range_scalp_scanner` were being 100% gated off. Three bleeds stacked:
1. `brake_global_halt` halted all new entries at margin CAUTION (70%), even tight-SL range scalps
2. `mode_allows_direction` LONG_ONLY / SHORT_ONLY (H4 macro direction guard for trend trades) blocked BB-upper SELLs in H4-bullish pairs, even when range was perfectly oscillating
3. `mode PAUSE` (director's "LIMIT-only posture" during SQUEEZE) blocked the very strategy that matches SQUEEZE

Result: trend_bot was firing in chop (avg_W +60 JPY, WR 19% today) while the one strategy that belongs in SQUEEZE sat idle.

**Fix**: Surgical bypass on range_bot for confirmed mean-reversion setups. Directional policies were designed for trend trades — they don't apply when the trade itself is symmetric chop harvest.

- `tools/brake_gate.py`: `check(pair, direction, lane="default")` — new `lane` param. `lane="range_scalp"` skips `brake_global_halt` at CAUTION (only PANIC stage halts). Pair-level block and TREND regime block still apply.
- `tools/range_bot.py`:
  - Entry loop passes `lane="range_scalp"` to `brake_gate.check`
  - Added `is_mean_reversion` check: `range_type == CLEAN_RANGE` + `ADX < 25` + `high_hits >= 5` + `low_hits >= 5`. When True, bypasses `LONG_ONLY`/`SHORT_ONLY`/`PAUSE` — but only ever places LIMIT orders, preserving "LIMIT-only posture"
  - Orphan cancel loop: `range_bot`-tagged LIMITs survive LONG_ONLY/SHORT_ONLY/PAUSE (consistent with entry bypass). Explicit `pending=CANCEL` and PANIC still clear them

**Safety**: PAUSE override is limited to LIMIT orders at BB extremes with structural SL. Explicit director kill (`pending=CANCEL`) still works. TREND regime block still works. Small size, tight SL — loss per shot is bounded.

**Files changed**: `tools/brake_gate.py`, `tools/range_bot.py`

**Test**: `python3 tools/range_bot.py --dry-run` — AUD_USD (previously blocked by `brake_global_halt`) now passes through. EUR_USD S-LONG (previously blocked by PAUSE) now reaches budget check. Both `python3` and `.venv/bin/python` confirmed working.

## 2026-04-16 — bot_policy_guard: respect unexpired trader policies, skip Level3 override

**Bug**: `force_aggression_override()` in `bot_policy_guard.py` was overriding ALL conservative policies (PAUSE/LONG_ONLY/SHORT_ONLY) every 60 seconds whenever margin was NORMAL — including freshly written trader policies that had not yet expired. This caused repeated Level 3 aggression (all 7 pairs BOTH/FAST/EARLY/MaxPending=2) even during worst-hour cold streaks.

**Fix**: Added an expiry check at the start of `force_aggression_override()`. If the policy's `expires_at` is in the future, skip the aggression override entirely. The override now only applies to stale (expired) policies, which was the original design intent.

**Files changed**: `tools/bot_policy_guard.py` (added `datetime` import + expiry check in `force_aggression_override`)

**Test**: `python3 tools/bot_policy_guard.py --dry-run` → "No repair: coverage target already satisfied." Policy stays at trader-intended values.

## 2026-04-17 — Level 3 aggression: kill all entry filters, let brake handle risk

**Problem after the 4-layer brake landed**: bots were placing 0 entries — the OLD layered filters (poison hour, late-session LIMIT-only, M15 ADX>=22, MTF strict alignment, R:R>=1.0, M1 must be aligned) assumed there was no safety net so they were paranoid.

**Now there is one** (inventory_brake + stranded_drain + brake_gate). So:

- `tools/range_bot.py`:
  - `LATE_SESSION_LIMIT_ONLY_HOURS_UTC = set()` (was 19-23)
  - `POISON_HOURS_UTC = set()` (was 19-23)
  - `MIN_MARKET_RR 1.0 -> 0.40`, `FAST_MIN_MARKET_RR 0.85 -> 0.35`, `MICRO_MIN_MARKET_RR 1.0 -> 0.40`
  - LIMIT R:R floor `1.0 -> 0.35`
  - reload-cap R:R floor `1.0 -> 0.3`
- `tools/trend_bot.py`:
  - `TREND_M15_ADX_MIN 22 -> 13` (was the biggest single blocker — no pair was passing 22)
  - `TREND_H1_ADX_MIN 18 -> 13`, `TREND_M5_ADX_MIN 17 -> 13`
  - `TREND_*_DI_GAP` halved across the board
  - `TREND_M5_BANDWALK_ADX 24 -> 18`, `_DI_GAP 6 -> 4`
  - `TREND_PULSE_ALIGN_SCORE 4 -> 2`, `TREND_PULSE_BLOCK_SCORE 4 -> 6`
  - `TREND_M1_READY_SCORE 4 -> 2`
  - `TREND_FAST/MICRO_ALLOWED_M1_STATES` add `"mixed"`
  - `TREND_MAX_TAG_TRADES 1 -> 3`, `TREND_MAX_PAIR_TRADES 2 -> 4`
  - `TREND_MIN_RR 1.25 -> 0.85`, `FAST 1.0 -> 0.65`, `MICRO 1.05 -> 0.70`
  - `assess_tf_trend` score floor `>=6 / +2 -> >=3 / +1` (most pairs were stuck at score 4-5 with old floor)
  - MTF conflict (H1 vs M15) no longer blocks: pick the higher-score side instead of skipping
  - `detect_m5_continuation` band_walk `bb_pos>=0.78 -> 0.65`; follow_through DI gap requirement removed; momentum AND chain became OR chain
  - `conviction_from_scores` floors lowered, fallback returns "B" without `m1.market_ready` requirement
- `tools/inventory_brake.py`:
  - Single fresh trade (units < 6000 OR upl > -150) no longer flagged as "stranded" — was blocking trend bot from adding to its own first entry
- `tools/stranded_drain.py`:
  - `MIN_ATR_PIPS 3.0 -> 1.0` (was skipping every drain on tight EUR/USD-style pairs)
- `logs/bot_inventory_policy.md`:
  - All 7 pairs `BOTH SHARED_MARKET FAST EARLY MaxPending=2`
  - `target_active_worker_pairs 1 -> 4`

**Result**: within 3 cycles after the change, EUR_USD trend_bot LONG x2 entered (+2050u each), AUD_USD range_bot LIMIT placed, USD_JPY range entry correctly blocked by regime_gate (TREND SHORT). brake + drain are the safety net, not the entry filters.

## 2026-04-17 — Stranded inventory brake: 4-layer safety to prevent closeout death

**Problem**: The bot harvest model (no SL, TP-only) wins fast on the trending side but accumulates a counter-side bag. Past pattern: +15% in hours → counter-side fills margin → OANDA forced closeout at the worst price → all profit returned (or worse). The bot has no exit plan for the half that doesn't TP.

**Change** — four new safety mechanisms wired into `tools/local_bot_cycle.sh`:

1. **`tools/inventory_brake.py`** (runs first):
   - Per-pair LONG/SHORT units imbalance check. Block adds on heavy stranded side at >= 3:1 ratio.
   - Account margin staged brake: NORMAL (<60%), CAUTION (60-75%, halt new), EMERGENCY (75-85%, drain mode forced), PANIC (>=85%, force-close 50% of largest stranded bag at market).
   - Writes `logs/bot_brake_state.json`.

2. **`tools/regime_switch.py`**: Per-pair regime detection (TREND/RANGE/MIXED) from M5+M15+H1 ADX/DI/BBW. Writes `logs/bot_regime_state.json`. When regime=TREND, only entries IN trend direction are allowed.

3. **`tools/stranded_drain.py`** (runs after entry bots): For each pair flagged drain_mode, computes weighted-avg entry of stranded side and sets a TAKE_PROFIT order on each trade at avg_entry +/- 0.5 ATR (close to BE). Gives the counter-side bag a dignified exit on retracement before margin fills.

4. **`tools/brake_gate.py`**: Helper consumed by `range_bot.py` and `trend_bot.py` at entry-decision time. Returns `(blocked, reason)` based on brake_state + regime_state. State files older than 5 min are ignored (fail-safe permissive).

**Wiring**: `local_bot_cycle.sh` runs `inventory_brake` + `regime_switch` BEFORE entry bots; `stranded_drain` AFTER. Existing `bot_policy_guard` and `bot_trade_manager` still run as before.

**Why this is the kill switch for the give-it-all-back pattern**: previous bot loop had no concept of "stranded inventory" or "margin staging". Now (a) you can't add to the heavy side past 3:1, (b) you can't run hot past 60% margin, (c) accumulated counter-side automatically gets a BE exit ladder, and (d) PANIC margin triggers a controlled half-close before OANDA does an uncontrolled full-close.

## 2026-04-17 — Bot SL reform: MICRO/FAST go SL-free, rely on timeout mechanism

**Problem**: Broker SLs on tight levels were getting hunted by noise before the timeout protection could kick in. For ultra-short scalp bots where the exit lives in TP1 + timeout, a broker SL on a structural level is redundant and counterproductive.

**Change**:
- `tools/range_bot.py` and `tools/trend_bot.py`:
  - MICRO/FAST tempo: `broker_sl = None` — no `stopLossOnFill` placed. Primary protection is `bot_trade_manager` timeout (MICRO: 4/7 min, FAST: 8/13 min).
  - BALANCED tempo: unchanged — keeps wide disaster backstop.
- `place_limit`, `place_market`, `place_trend_market`, `log_entry`, `slack_notify` all updated to accept `sl: float | None`.
- Log shows `SL=NO_SL` for no-SL entries; display shows `No-SL (timeout mode)`.

## 2026-04-17 — High-speed bot tuning: FAST/MICRO thresholds loosened

**Goal**: Enable FAST and MICRO tempos to fire more readily so the worker layer produces more round-trips per session.

**Changes**:
- `tools/trend_bot.py`
  - `TREND_FAST_ALLOWED_M1_STATES` now `{"aligned", "reload"}` (was `"aligned"` only). FAST entries were blocked whenever M1 was in "reload" state — doubled the blocked window.
  - `TREND_MICRO_MAX_SPREAD_MULTIPLE` 1.20 → 1.25
  - `TREND_MICRO_TP_M5_ATR` 1.35 → 1.20 and `TREND_MICRO_TP_H1_ATR` 0.45 → 0.40 (tighter TP = faster MICRO exits)
- `tools/range_bot.py`
  - `MICRO_MARKET_MAX_SPREAD_MULTIPLE` 1.20 → 1.25 (consistent with trend_bot)
  - `MICRO_RANGE_MIN_SPREAD_MULTIPLE` 3.0 → 2.5 (more MICRO range market shots)
  - `REENTRY_COOLDOWN_BY_TEMPO["MICRO"]` 2 → 1 min
- `tools/bot_trade_manager.py`
  - MICRO MARKET: stale 6→4 min, full-close 10→7 min
  - FAST MARKET: stale 10→8 min, full-close 16→13 min
  - MICRO PASSIVE: stale 10→8 min, full-close 16→13 min
  - FAST PASSIVE: stale 14→12 min, full-close 22→19 min
- `docs/SKILL_trader.md`
  - FAST tempo: BALANCED is no longer the default for active lanes. FAST is now the explicit default for trend/range harvest seats.
  - FAST M1 requirement updated to match code: "aligned or reload" (was "clearly aligned")

## 2026-04-17 — Rebalance trader prompts away from forced-action churn

**Problem**: The trader prompt stack had over-corrected against passivity. In practice it was rewarding "do something" behavior even when the live tape was late, dirty, pre-event, or already paid.

- `docs/SKILL_trader.md` still told the trader to hunt harder when behind, treated drought as worse than a bad trade, blocked session end on an empty book, and framed the hero pair as a day-long concentration quota.
- `docs/TRADER_PROMPT.md` still pushed the same bias in Japanese with "PASSで終わるな", margin-max language, and a pass format that did not require a concrete next trigger.
- `collab_trade/strategy_memory.md` still front-loaded "don't wait, just enter" without the newer lesson that forced anti-drought action can destroy expectancy.

**Fix**:
- `docs/SKILL_trader.md`
  - Reframed the +10% / +5% numbers as benchmarks, not permission to lower the quality bar.
  - Replaced `Hero pair` with `Primary vehicle` / `Backup vehicle` so concentration follows the live tape instead of becoming fixation.
  - Replaced forced drought-entry logic with an explicit no-trade accountability block that requires `best untraded seat + trigger + invalidation`.
  - Changed flat-book handling from action-forced to output-forced: empty book is allowed only when the trigger map is concrete.
  - Softened Tier 2 and Momentum-S language so the prompt exposes avoidance without mechanically forcing an order.
- `docs/TRADER_PROMPT.md`
  - Aligned the Japanese operator prompt with the same principle: pass is allowed only with a concrete next trigger and invalidation.
  - Removed language that equated being behind with mandatory action or high margin usage.
- `collab_trade/strategy_memory.md`
  - Added the new memory that opportunity cost is real, but forced anti-drought action is also a repeatable loss pattern.

**Why**: The repo philosophy is correct: shape the output so the model has to think. "Enter something" is a rule. "If you stay flat, write the exact trigger and invalidation" is a thinking format. The latter should produce better trader behavior on both stronger and weaker models.

## 2026-04-17 — Recast the worker layer as small-wave harvest + inventory cleanup

**Problem**: The local worker layer still behaved too much like a cautious single-shot assistant.

- `FAST` / `MICRO` entries were documented as quick bites, but the runtime still treated `MICRO` like `BALANCED` on cleanup timing.
- Range `FAST` market bites could still be blocked by a final hard `R:R >= 1.0` gate even though the earlier planner already allowed the lower-RR fast-bite posture.
- Worker stops were still acting like thesis stops instead of disaster backstops, which kept recreating the "tight SL gets clipped before the small wave pays" pattern.
- Prompting still left too much room for trader/bot role drift: the trader could think in one-seat hero-pair terms while the user wanted a joint structure of trader thesis + bot harvest + inventory cleanup.

**Fix**:
- `tools/range_bot.py`
  - Switched worker sizing to a small-lot ladder by `order_type × tempo × conviction` so `FAST` / `MICRO` can take more shots with less size per shot.
  - Raised the worker margin budget ceiling so smaller per-shot sizing can keep more seats alive.
  - Added tempo-aware re-entry cooldowns (`BALANCED > FAST > MICRO`) instead of a single 10-minute blanket cooldown.
  - Separated `thesis line` from broker `disaster SL`, and now places the wider disaster stop on OANDA while keeping the tighter thesis line for entry-quality judgment.
  - Fixed the final RR gate so `FAST` market bites can actually use their intended lower RR floor instead of being re-blocked at submit time.
- `tools/trend_bot.py`
  - Same thesis-line vs disaster-stop split for continuation entries.
  - Prints both lines explicitly in the dry-run output.
  - Uses the tempo-aware worker re-entry cooldown path.
- `tools/bot_trade_manager.py`
  - Added real `MICRO` timeout profiles instead of treating `MICRO` as `BALANCED`.
  - Tightened worker timeout cleanup so stranded small-wave seats are handled as inventory issues sooner.
- `docs/SKILL_trader.md`, `docs/SKILL_range-bot.md`, `docs/SKILL_trend-bot.md`, `docs/SKILL_inventory-director.md`, `AGENTS.md`
  - Reframed the worker contract around `small-lot harvest + disaster stop + fast cleanup`.
  - Added explicit trader/bot split output requirements so the trader must name the structural seat, harvest lanes, and inventory owner every session.
- `collab_trade/strategy_memory.md`
  - Updated sizing memory so the old 3k floor remains true for trader-owned seats, while worker `FAST` / `MICRO` lanes are explicitly allowed to go smaller.

**Why**: The user's requested posture is not "bot goes reckless." It is "bot takes many small waves, trader keeps the structural thesis, and inventory management handles the misses." This change moves that posture into both the runtime and the prompts.

## 2026-04-17 — Cap unrealistic passive LIMIT distance so the worker stops parking wish orders

**Problem**: The range worker could still leave passive LIMIT orders far from the live price simply because the Bollinger edge itself was structurally valid. In practice that produced low-fill orders that sat far away, then got canceled later for "unlikely to fill" / "too far below" style reasons. The issue was not the pair; it was the market state: price was still too far from the edge for that passive order to be a real deployment path.

**Fix**:
- `tools/range_bot.py`
  - Added a passive-limit distance cap based on current spread and M5 ATR.
  - If the desired passive entry is too far from live price, the bot now either:
    - caps the reload closer to live price while preserving acceptable R:R, or
    - skips the order entirely if a closer reload would no longer make economic sense.
  - Existing worker pending LIMITs are now re-checked against that same live-gap rule, so stale far-away wish orders get canceled earlier instead of aging into later cleanup churn.
- `docs/SKILL_trader.md`, `docs/SKILL_range-bot.md`
  - Documented the same principle for discretionary and worker behavior: structural is necessary, but not enough. New passive orders also need realistic fill distance.

**Why**: A structural level that is still one full pullback away is analysis, not deployment. The worker should either leave a realistic reload or stay ready for the live-now / second-shot path instead of clogging the book with distant orders.

## 2026-04-17 — Fix trader/worker collaboration leaks and late-session overblocking

**Problem**: The local layer had three live-profit killers.

1. `range_bot.py` hard-skipped the entire `19:00-23:59 UTC` band. That stopped not only bad market chasing, but also clean passive trap placement in late-NY / pre-Tokyo tape. Result: flat book even when the market still offered passive range seats.
2. `range_bot_market` could inherit stops that were only a few pips wide after upgrading a limit idea to a live market fill. Those stops were often narrower than current spread + micro noise, which is exactly the "SL hunted" behavior the user complained about.
3. `close_trade.py` still allowed routine trader-side worker exits after the first 10 minutes. That meant trader/worker collaboration could collapse back into manual churn under vague reasons like `trader_worker_inventory`.

**Fix**:
- `tools/range_bot.py`
  - Replaced the blanket `19:00-23:59 UTC` full stop with `LIMIT-only` behavior in that window. Passive worker entries remain allowed; only market chasing is suppressed.
  - Added a live-entry stop floor for `MARKET` upgrades based on current spread and M5 ATR. If the inherited stop is too tight, it is widened to a minimum safe distance; if that makes the live fill no longer worth it, the market upgrade is skipped.
- `tools/close_trade.py`
  - Worker-tagged trades are now worker-owned for their entire lifecycle, not just the first 10 minutes.
  - Routine trader closes on worker inventory are blocked outright. Only bot-managed reasons (`bot_*`) or explicit emergency `--force-worker-close` reasons are permitted.
- `docs/SKILL_range-bot.md`
  - Updated the operating note so the documented late-session behavior matches runtime: passive-only, not full skip.

**Why**: The trader and worker are supposed to collaborate by role. The worker should keep harvesting passive range seats and manage its own scalp lifecycle; the trader should steer policy and only override in real emergencies. Flat-book starvation, 2-3 pip live-entry stops, and discretionary worker flattening are all violations of that contract.

## 2026-04-17 — Make fresh worker-close protection use OANDA transaction truth

**Problem**: The anti-churn contract around worker trades still had two holes in live trading.

1. `close_trade.py` checked the current trade payload for the worker tag, but OANDA can omit that tag on the trade object even when the opening `ORDER_FILL` transaction still has `tradeOpened.clientExtensions.tag`. In practice that meant fresh `range_bot_market` / `trend_bot_market` trades could still be flattened with routine trader reasons because the guard failed to recognize them as worker inventory.
2. Even when the tag was present, the first-10-minute guard only blocked a narrow set of routine reasons. Any other discretionary reason string could still bypass the intended worker-owned window and turn a fresh bot fill into spread churn.

**Fix**:
- `tools/oanda_trade_tags.py`
  - New helper module that recovers worker tags/comments from the opening OANDA transaction and can enrich open-trade payloads when the trade endpoint omits extensions.
- `tools/range_bot.py`
  - `fetch_open_trades()` now enriches open trade payloads from OANDA transaction truth before any worker/trader conflict logic runs.
- `tools/bot_trade_manager.py`
  - `fetch_open_trades()` now uses the same enrichment path so worker inventory is identified consistently in the manager.
  - Worker-owned close reasons are now normalized to `bot_*` for normal lifecycle actions.
  - Margin/deadlock relief closes now use explicit emergency reasons with `--force-worker-close` so they stay aligned with the runtime contract.
- `tools/close_trade.py`
  - `fetch_trade()` now falls back to the opening transaction to recover worker tags before enforcing the worker guard.
  - The first 10 minutes of a worker trade are now treated as worker-owned by default: only bot-managed reasons (`bot_*`) or explicit emergency `--force-worker-close` reasons may close it.
- `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`
  - Updated the worker-close language to match the stricter runtime behavior.

**Why**: Repeated small closes on fresh worker fills are not risk management; they are spread payments. If the worker seat is supposed to own the first scalp window, that ownership must be enforced from OANDA transaction truth, not assumed from a best-effort tag on the current trade object.

## 2026-04-17 — Force regime-first thinking before pair selection

**Problem**: Recent losses were being analyzed too easily as "bad pair choice" when the deeper failure was upstream: the trader could still jump from pair scan to entry without first writing what the market regime was paying, which vehicle expressed it cleanly, and which tempting expressions were actually traps. That left room for stale H4-story dip-buys and dirty cross expressions to slip through.

**Fix**:
- `docs/SKILL_trader.md`
  - Expanded `Market Narrative` to require `Execution Regime`, `Best Expression NOW`, `Second-best expression`, `Expressions to avoid`, and an explicit `H4-memory trap check`
  - Added `Expression fit` / `Cleaner or dirtier alternative` / `Memory trap check` to Tier 1 and promoted Tier 1 scan blocks
  - Added an `EXPRESSION` line to Tier 2 so every pair is judged as a vehicle for the regime, not just as a standalone setup
  - Expanded the pre-entry conviction block so every trade must explain why this pair is the best expression of the regime and why the obvious alternative is worse
  - Updated the `state.md` handoff template so the next session inherits regime/expression context, not just pair notes
- `collab_trade/strategy_memory.md`
  - Recorded the new lesson that regime and expression must be named before pair selection

**Why**: "GBP_USD long" is a pair idea. "Corrective USD bid with dirty JPY crosses; direct USD pair is the clean expression" is a market read. The prompt now forces the second kind of thinking before the first kind of action.

## 2026-04-17 — Surface realized expectancy so payoff quality stops hiding behind nominal R:R

**Problem**: The system already knew many losing days came from "wins too small / losses too large" rather than from raw direction accuracy, but the runtime summaries still foregrounded win rate and nominal `R:R`. That made it too easy to miss the actual question: "with this win rate and these realized payouts, is the trade distribution making money at all?"

**Fix**:
- `collab_trade/memory/pretrade_check.py`
  - Added realized payoff metrics to historical pair stats: `avg_win`, `avg_loss`, `rr_ratio`, `expectancy`, `profit_factor`, `break_even_win_rate`
  - Added backward-looking warnings when pair history is negative-EV or when actual WR sits meaningfully below the break-even WR implied by realized payout shape
  - Expanded the CLI output so pretrade history now shows `EV/trade`, `avg win/loss`, `R:R`, and break-even WR instead of only WR + avg P&L
- `tools/oanda_performance.py`
  - Added realized expectancy, profit factor, and break-even WR to overall, daily, and by-pair performance summaries
  - Human-readable output now surfaces payoff quality explicitly instead of making `R:R` the only payout lens
- `tools/daily_review.py`
  - Daily report top-line now includes expectancy and break-even WR
  - Pair/day summaries and LOW-entry review now expose `EV` and `R:R`
- `tools/trade_guard.py`
  - Replaced the narrow `RR < 0.7` warning with a realized payoff-quality alert centered on `WR vs break-even WR` and `EV/trade`
- `docs/SKILL_daily-review.md`
  - Tightened the review instruction so the journal must state whether expectancy was actually positive, not just whether win rate looked good

**Why**: Planned `R:R` is only a hypothesis. In live FX trading, spread, execution price, partial exits, and wrong-timeframe management can all destroy the payout shape even when the original diagram looked acceptable. The system needs to keep asking the harder question: does the realized distribution have positive expectancy after repetition?

## 2026-04-17 — Add deterministic bot-policy guard so flat-book coverage cannot be starved by narrow policy

**Problem**: The local worker loop was healthy and scanning every minute, but it could still stay empty for long stretches when the human/LLM policy map came back too narrow. In practice the worker would find live `A/B` seats and then skip them with lines like `policy PAUSE blocks SELL` or `policy market disabled`, even though `Target Active Worker Pairs` still said the flat book should keep one live seat.

**Fix**:
- `tools/bot_policy.py`
  - Added canonical policy markdown rendering and full policy normalization helpers so runtime tools can rewrite the human policy into one stable contract.
- `tools/bot_policy_guard.py`
  - New deterministic guard that runs before the worker bots.
  - Recovers policy from markdown if JSON is missing/corrupt.
  - Detects `coverage target > live worker coverage` plus `all live current lanes blocked only by policy`.
  - Reopens exactly one minimal repair lane on a short TTL.
    Range repair first as passive `LIMIT`; trend continuation second if no usable range seat exists.
  - Rewrites `logs/bot_inventory_policy.md/json` into canonical form so the human-readable and machine policy stay aligned.
- `tools/local_bot_cycle.sh`
  - Now runs `bot_policy_guard.py` immediately after factor-cache refresh and before `bot_trade_manager.py` / `trend_bot.py` / `range_bot.py`.
- `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`, `AGENTS.md`
  - Documented the deterministic safety net so trader and backup repair flows understand the new ownership boundary.
- `collab_trade/strategy_memory.md`
  - Recorded the lesson that coverage-seat intent needs runtime enforcement, not prompt obedience alone.

**Why**: A coverage target is an execution contract, not a suggestion. If the policy says the worker book should not stay empty, the deterministic local layer must stop stale/narrow policy from starving every current seat while still keeping the repair minimal and time-bounded.

## 2026-04-17 — Re-check policy immediately before worker submit

**Problem**: The new local bot layer could still act in lanes that the trader intended to reserve or pause.

- `range_bot.py` and `trend_bot.py` only evaluated `ACTIVE` at startup, so a mid-cycle policy flip could still leave a small race window before submit.
- `range_bot.py` could keep stale worker pending orders even after the lane had been flipped to `PAUSE` / `CANCEL`.

**Fix**:
- `tools/bot_policy.py`
  - Added explicit helpers for `ACTIVE` gating and lane-level block reasons (`mode`, `allow_market`, `pending`, `max_pending`).
- `tools/range_bot.py`
  - Uses the centralized lane gate before placing any passive or market order.
  - Re-loads policy immediately before submit so a fresh `PAUSE_ALL` / `REDUCE_ONLY` / ownership change aborts the order.
  - Cancels stale worker pending orders when the lane has already been moved to `PAUSE` / `CANCEL`.
  - Counts only truly lane-eligible worker seats when sizing the bot budget.
- `tools/trend_bot.py`
  - Uses the same centralized lane gate.
  - Re-loads policy immediately before submit so a fresh policy change aborts the market order.

**Why**: Worker/trader collaboration is intentional, but `PAUSE_ALL`, `REDUCE_ONLY`, and lane-level `PAUSE` / `CANCEL` still need to bite immediately. The local worker layer must respect the latest live policy instead of sneaking an order through a one-cycle race.

## 2026-04-17 — Tighten MICRO so it stops paying spread and praying for TP

**Problem**: The worker layer had good MTF structure, but the newest `MICRO` lane was still too tolerant of noise. In practice that meant:
- trend `MICRO` could accept sub-1R style bites,
- range `MICRO` could accept weak `M1` triggers and average spreads,
- and the result was a design that sometimes looked like "pay spread and hope for immediate follow-through."

**Fix**:
- `tools/trend_bot.py`
  - Tightened `MICRO` payout floors and RR floor to roughly `1.05R`
  - Added a stricter spread gate for `MICRO`
- `tools/range_bot.py`
  - Raised the `MICRO` market RR floor to `1.0`
  - Raised `MICRO` TP floors (spread multiple + ATR fraction)
  - Added a stronger `M1` score requirement for `MICRO`
  - Added a stricter spread gate for `MICRO`
- `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`
  - Updated the prompt contract so `MICRO` explicitly means clean spread + stronger trigger + roughly 1R after costs

**Why**: MTF structure should stay. What needed to change was the payoff quality. A fine-wave lane is useful only if it still has real expectancy after spread and slippage, not if it relies on immediate luck.

## 2026-04-17 — Add flat-book repair lanes and H1 range promotion for the worker layer

**Problem**: Missed participation was still coming from structure, not just judgment.

1. `range_bot.py` dropped any higher-timeframe range seat that started at scanner conviction `C`, even when the `H1` box was clean, signal strength was high, and `M1` was already offering a usable live trigger.
2. Worker policy had no explicit way to say "if the whole book is flat, this pair may be the first live seat." The choice was either the normal quality bar or a full pause, which left the book empty while audits were still pointing at live A/B lanes.

**Fix**:
- `tools/bot_policy.py`
  - Added `target_active_worker_pairs`
  - Added per-pair `entry_bias` (`PASSIVE` / `BALANCED` / `EARLY`)
  - Extended active-policy grace from 60 to 120 minutes so one missed trader cycle does not immediately starve the worker map
- `tools/render_bot_inventory_policy.py`
  - Added backward-compatible parsing/rendering for `Target Active Worker Pairs` and `EntryBias`
- `tools/range_bot.py`
  - Promotes strong `H1` range seats from base `C` to executable `B` when structure plus live trigger support it
  - Uses `target_active_worker_pairs` and `entry_bias` to allow flat-book repair scouts only when the book is under-covered and the pair is explicitly allowed to act early
- `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`, `AGENTS.md`
  - Updated the worker-policy contract and operating language to include coverage target + entry bias
- `collab_trade/strategy_memory.md`
  - Recorded the flat-book repair lesson for later review

**Why**: The goal is not "force a position at all times." The goal is "do not stay empty when the market is already offering a clean first seat." This change moves that behavior into the policy contract and deterministic worker code.

## 2026-04-17 — Add target-race partials so worker trades can pay TP1 and still keep TP2 alive

**Problem**: The worker layer still treated every trade as a single static `TP/SL` outcome. That meant conflicting multi-timeframe reads could not be expressed correctly. A pair could have a valid short-term counter move and a valid higher-timeframe continuation at the same time, but the local bots had no way to ask "which target is likely first?" or "where can the remainder stay alive after the fast target pays?"

**Fix**:
- `tools/range_bot.py`
  - Range entries now build a target-race plan with `TP1`, `TP2`, and `hold_boundary`
  - The worker still uses `TP1` to judge entry quality, but broker `takeProfitOnFill` is now set to `TP2` so the manager can partial at `TP1` instead of letting OANDA flatten the full seat there
- `tools/trend_bot.py`
  - Continuation entries now build the same target-race plan
  - Balanced seats scale at a nearer `TP1` and keep the old continuation target as `TP2`; `FAST` / `MICRO` seats now extend into a controlled second target instead of dying at the first bite
- `tools/worker_target_race.py`
  - Added shared plan encoding/decoding plus persistent execution state for worker runners
- `tools/bot_trade_manager.py`
  - Added `TP1 -> half-close -> runner` promotion
  - After TP1, the manager moves the remaining half to `TP2` with a new `hold_boundary` stop instead of applying normal stale-scalar cleanup immediately
- `tools/close_trade.py`
  - Partial closes now notify Slack as `modify` events instead of incorrectly posting `FULL CLOSE`
- `docs/SKILL_trader.md`, `AGENTS.md`, `collab_trade/strategy_memory.md`
  - Updated the operating contract and recorded the lesson so the worker layer is explicitly allowed to express "fast target first, higher-timeframe target later"

**Why**: The right question is not "long or short?" but "which objective is likely first, and where can the rest survive?" This change moves that logic out of human-only judgment and into the worker stack itself.

## 2026-04-17 — Stop routine trader closes from overriding fresh worker trades

**Problem**: Even after adding the first worker-close guard, `qr-trader` could still flatten fresh worker trades by passing `--force-worker-close` with the routine reason `trader_worker_inventory`. In live operation that meant a narrow trader session could still crush a just-opened bot trade within minutes, even though the intended contract was "policy steering first, emergency override only."

**Fix**:
- `tools/close_trade.py`
  - Added explicit emergency-only force reasons: `worker_emergency_override`, `panic_margin`, `deadlock_relief`, `worker_policy_breach`, `rollover_emergency`
  - Fresh worker trades can no longer be closed in the first 10 minutes with routine reasons like `trader_worker_inventory` or `inventory_director_backup`, even if `--force-worker-close` is passed
- `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`
  - Updated the close command examples and worker-close guard language to match the stricter contract

**Why**: Broad collaboration cannot rely on prompt obedience alone. The runtime must also prevent routine trader taste from killing the fast worker layer before the scalp window has had a chance to play out.

## 2026-04-17 — Force trader worker policy to use a broad 7-pair view

**Problem**: The worker layer had already become capable of `MICRO` / `FAST` / `BALANCED` scalp lanes, but the `trader` policy-writing prompt still made it too easy to collapse the live worker map into a single hero-pair view. In practice that left valid lanes like live `GBP_USD` continuation setups paused simply because another pair or another discretionary thesis was receiving attention.

**Fix**:
- `docs/SKILL_trader.md`
  - Added an explicit breadth contract: worker policy must represent the 7-pair opportunity map, not just the discretionary hero pair
  - Added a required `## Worker Breadth` reasoning block before each worker policy rewrite so trader must name the best trend lane, second trend lane, best range lane, and the real blocker to broader deployment
- `docs/SKILL_inventory-director.md`
  - Added the same repair principle for backup policy rewrites so stale one-pair bias is not preserved

**Why**: The hero pair is a discretionary concentration concept, not a valid reason to starve the local worker layer. Worker policy should be narrow only when the market is narrow, not when the operator's attention is narrow.

## 2026-04-17 — Add MICRO tempo so the local bot layer can harvest finer waves

**Problem**: The local worker layer still behaved like a two-speed system. `BALANCED` was too slow for tiny scalp windows, and `FAST` was intentionally tightened to `M1 aligned` only after the recent stop-loss churn. That fixed late-chase entries, but it also left no explicit lane for the smaller waves that still appear inside a live trend or clean range extreme.

**Fix**:
- `tools/bot_policy.py`
  - Added `MICRO` as a first-class tempo in the worker policy contract
- `tools/trend_bot.py`
  - Added a `MICRO` execution profile with shorter TP / lower RR floor
  - `MICRO` is allowed only when `M1` is `aligned` or `reload`
  - Candidate scanning no longer filters only through the `BALANCED` plan, so micro-only seats can survive discovery
- `tools/range_bot.py`
  - Added a `MICRO` range TP profile and smaller market sizing
  - `MICRO` can take strong `B` range edges with MARKET when the live micro trigger is there, instead of forcing them all to stay passive
- `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`, `AGENTS.md`, `CLAUDE.md`, `collab_trade/state.md`
  - Updated the policy language so trader / backup LLM can deliberately choose `MICRO`

**Why**: Speed should come from a distinct fine-wave lane, not from diluting the safety rules on `FAST`. `MICRO` keeps the fast layer fast without turning every noisy continuation into another stop-loss chase.

## 2026-04-17 — Let range-bot trade the best M5/H1 range lens instead of M5 only

**Problem**: `tools/range_scalp_scanner.py` could already see real range opportunities on both `M5` and `H1`, but `tools/range_bot.py` only built live worker candidates from `M5`. In practice that meant clean `H1` range seats such as `AUD_USD` and `USD_JPY` could show up in the scanner while the local worker reported `Ranges found: 0` and never even evaluated them as executable fade views.

**Fix**:
- `tools/range_bot.py`
  - Scans both `M5` and `H1` range opportunities per pair
  - Scores the surviving views, keeps the best primary lens, and carries the non-selected views as `alternate_views` for dry-run visibility
  - Prints the chosen setup timeframe in trade plans so it is obvious whether the worker is acting on an `M5` or `H1` range
- `AGENTS.md`, `CLAUDE.md`
  - Updated the operating description so the worker contract reflects the real multi-timeframe range selection logic
- `collab_trade/strategy_memory.md`
  - Recorded the lesson that collapsing range reads to `M5` only hides valid higher-timeframe seats

**Why**: The market rarely offers only one valid lens per pair. If the scanner can already see the `H1` box, the worker should not erase that edge just because the `M5` tape is noisy or mid-zone.

## 2026-04-17 — Make the local bot layer scalp-only and push horizon ownership back to trader

**Problem**: The prior collaboration work solved churn, but the horizon contract was still muddy.

1. The local worker layer could still linger long enough to behave like a pseudo-swing book because `bot_trade_manager.py` only intervened on panic / deadlock paths.
2. `Tempo` described exit speed, but not the deeper contract that the worker is a scalp layer while the trader owns the full discretionary ladder from scalp through swing.
3. The trader prompt still logged only `scalp/swing`, which was too coarse to distinguish tactical scalp, rotation, and real swing ownership.

**Fix**:
- `tools/range_bot.py`
  - Replaced the old fixed worker budget with a dynamic scalp budget profile that can expand roughly 18-34% of NAV when the live tape and conviction justify it, while still leaving the longer seat to the trader layer
- `tools/bot_trade_manager.py`
  - Added tag + tempo-aware scalp timeout profiles
  - Worker trades now get flattened when they overstay their scalp window even if panic margin is not yet elevated
  - Preserved the existing margin / deadlock emergency-relief path on top of the new timeout cleanup
- `tools/bot_inventory_snapshot.py`
  - Snapshot output now surfaces worker tag, tempo, scalp state, and timeout window so the trader can see when a bot seat is aging out of scope
- `docs/TRADER_PROMPT.md`
  - Upgraded discretionary tags from generic `scalp/swing` to `trader_scalp`, `trader_rotation`, `trader_swing`
  - Added explicit horizon fields to the registry/log contract: expected hold, first confirmation, promotion trigger, and must-flat boundary
- `docs/SKILL_trader.md`, `docs/SKILL_bot-trade-manager.md`, `docs/SKILL_range-bot.md`, `docs/SKILL_trend-bot.md`, `docs/SKILL_inventory-director.md`, `AGENTS.md`, `collab_trade/state.md`
  - Updated the operating contract so the worker layer is scalp-only and the trader owns all discretionary horizon changes

**Why**: The right split is not "bot trades first, trader promotes later." The right split is "bot harvests speed, trader owns time." That keeps the fast layer fast, preserves swing capital for the trader, and makes the book easier to reason about.

## 2026-04-17 — Require aligned M1 for FAST trend-bot entries

**Problem**: The first `FAST` collaboration pass let trend-bot enter when the micro tape was only `mixed` or `reload`. In live trading that produced late continuation shorts like `GBP_USD 468314`, where higher timeframes were bearish but the immediate `M1` tape had already turned counter and the worker still fired a short-TP market entry.

**Fix**:
- `tools/trend_bot.py`
  - Added an explicit `FAST` gate: the pair policy may say `Tempo=FAST`, but the bot now refuses the trade unless `M1` state is `aligned`
- `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`, `AGENTS.md`, `CLAUDE.md`
  - Updated the policy contract so `FAST` is documented as valid only when the live `M1` micro state is aligned

**Why**: `FAST` is for clean immediate participation, not for fading a micro pullback late in the move. If the micro tape is mixed, the worker should either wait or fall back to the balanced profile instead of paying spread for a weak bite.

## 2026-04-17 — Make worker re-entry cooldown see real OANDA stop-loss fills

**Problem**: The first anti-churn pass only read `logs/live_trade_log.txt`. That caught trader-forced worker closes, but it missed OANDA-native stop-loss fills because those exits were never written to the local log. Result: a worker trade could stop out and the next local cycle would still think no recent close had happened, then immediately re-enter the same pair/direction.

**Fix**:
- `tools/range_bot.py`
  - Added `fetch_recent_worker_close_events()` which inspects recent OANDA transactions via `transactions/idrange`
  - Maps worker `tradeOpened.tradeID` records to later `ORDER_FILL` close events, so `STOP_LOSS_ORDER` and `MARKET_ORDER_TRADE_CLOSE` are recognized as worker exits even when the local log has no matching line
  - `recent_close_cooldown()` now combines local close-log evidence with recent OANDA transaction evidence
- `tools/trend_bot.py`
  - Reuses the same transaction-backed recent-close cooldown

**Why**: The cooldown must follow the broker’s truth, not only the local text log. Otherwise fast workers will keep re-entering right after real stop-losses and turn one bad read into repeated spread burn.

## 2026-04-17 — Refresh stale trader charts automatically and widen trend-bot continuation exits

**Problem**: Two accuracy failures were live at the same time.

1. Trader sessions were reading `logs/charts/*.png` as if they were fresh, but the local chart set could go stale for hours when `quality-audit` missed a cycle. That meant the trader could be looking at old tape while believing it was current.
2. `tools/chart_snapshot.py` printed a wrong bearish plan string (`TREND-BEAR` still said "Buy dips"), and `tools/trend_bot.py` was sizing continuation stops like a micro scalp. A live EUR_USD `trend_bot_market` short printed `TP 6.6pip / SL 3.9pip`, and `protection_check.py` immediately flagged the stop as noise-tight.

**Fix**:
- `tools/session_data.py`
  - Added chart-freshness checks for the full 14-PNG set
  - Auto-runs `.venv/bin/python tools/chart_snapshot.py --all` when the oldest chart is older than 40 minutes or files are missing
  - Added explicit `CHART SNAPSHOTS` and `QUALITY AUDIT STATUS` sections so the trader can see whether the local visual view is fresh and whether `quality_audit.md` is stale
- `tools/chart_snapshot.py`
  - Fixed bearish regime guidance text so `TREND-BEAR` / `MILD-BEAR` no longer tell the trader to "Buy dips"
- `tools/trend_bot.py`
  - Raised continuation stop floors to clear spread, M5 ATR, and H1 ATR noise
  - Raised TP floors to match the wider stop and force continuation trades to target real follow-through instead of 5-7pip scalp scraps
- `docs/SKILL_trader.md`, `AGENTS.md`
  - Updated the operating contract: quality-audit remains the primary chart writer on a 45-minute cadence, but trader startup now refreshes stale/missing PNGs automatically and treats stale audit prose as historical context only
- `collab_trade/strategy_memory.md`
  - Recorded the continuation-stop lesson so daily-review does not regress back to sub-H1-ATR noise stops

**Why**: Visual chart reading is only useful if the pictures are current, and continuation entries are only worth taking if the stop can survive normal H1 noise. This change closes both gaps in code instead of relying on the prompt to remember them.

## 2026-04-17 — Add explicit collaboration tempo for trader + worker co-trading

**Problem**: The anti-churn fix stopped the trader from flattening fresh worker fills, but the system was still too binary.

1. Same-pair cooperation was incomplete because any same-direction discretionary pending entry still blocked the worker, even when the trader actually wanted a shared seat.
2. Worker exits were tuned for balanced swings only. That made the system safer, but slower than the user's intended "trader keeps the thesis, bot takes quick bites" style.

**Fix**:
- `tools/bot_policy.py`
  - Added pair-level `tempo` with two explicit modes: `BALANCED` and `FAST`
- `tools/render_bot_inventory_policy.py`
  - Extended the markdown contract to include a `Tempo` column while staying backward compatible with older 7-column policy rows
- `tools/range_bot.py`
  - Added ownership-aware pending deconfliction so same-direction trader pending orders no longer block the worker when policy explicitly allows sharing
  - Added `FAST` tempo handling that shortens the range bot TP into a quicker bite profile
  - Lowered the required market RR floor for `FAST` tempo so quick worker scalps can execute intentionally instead of being silently filtered out
- `tools/trend_bot.py`
  - Reused the same ownership-aware pending deconfliction
  - Added `FAST` tempo handling for shorter continuation TP targets so the worker can scalp band-walk follow-through while the trader keeps the broader seat
- `tools/bot_inventory_snapshot.py`
  - Now surfaces `tempo` in the human-readable bot snapshot
- `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`, `AGENTS.md`, `CLAUDE.md`, `collab_trade/state.md`
  - Updated the operating contract from simple ownership-only steering to `Ownership + Tempo`

**Why**: The user does not want "trader or bot"; they want a deliberate stack. The trader owns the bigger thesis and failure path, while the worker can harvest shorter intraday bites on the same pair when policy explicitly says that coexistence is allowed and fast exits are desired.

## 2026-04-17 — Stop trader-vs-worker churn on fresh bot fills

**Problem**: The worker layer was not just losing to tight stops. A more expensive failure mode had emerged in live trading: `qr-trader` could close a fresh bot trade with `reason=trader_worker_inventory`, and then the 60-second worker immediately re-entered the same pair/direction. That turned one weak idea into repeated spread payments and made the trader layer look like it was "crushing" the bot instead of steering it.

**Fix**:
- `tools/range_bot.py`
  - Added a recent-close cooldown that reads `logs/live_trade_log.txt` and blocks same-pair same-direction re-entry for 10 minutes after fresh `trader_worker_inventory` or stop-loss exits
- `tools/trend_bot.py`
  - Reused the same recent-close cooldown, so trend MARKET continuation cannot immediately refill a seat the trader or stop just killed
- `tools/close_trade.py`
  - Added a worker-trade guard: routine closes with `reason=trader_worker_inventory` or `inventory_director_backup` are rejected during the first 10 minutes unless `--force-worker-close` is explicitly passed
- `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`
  - Changed the ownership contract from "trader routine owner" to "worker-owned lifecycle, trader policy steering"
  - Restricted worker close commands to emergency overrides and documented the new `--force-worker-close` requirement
- `AGENTS.md`, `CLAUDE.md`, `collab_trade/state.md`
  - Updated runtime descriptions and handoff memory so the new anti-churn boundary is explicit

**Why**: The trader layer should steer worker permissions, not repeatedly flatten fresh worker inventory and invite the bot to re-enter the same idea one minute later. By moving this boundary into code, the system stays robust even if a future prompt regresses.

## 2026-04-17 — Add explicit worker ownership contract and enforce MaxPending in code

**Problem**: The first deconfliction pass fixed trader pending-entry collisions, but the worker layer still had two design holes.

1. Same-pair coexistence with live discretionary trades was still implicit and inconsistent. `range_bot.py` blocked any live position on the pair, while `trend_bot.py` could still add in some same-direction cases. The system had no explicit contract saying whether a trader-owned seat could be shared with passive worker reloads, market worker adds, or neither.
2. The policy schema exposed `MaxPending`, but the local bot code did not actually enforce that cap. Prompt and runtime could drift apart, and duplicate pending orders could still persist until manually cleaned.

**Fix**:
- `tools/bot_policy.py`
  - Added pair-level `ownership` normalization with three explicit modes: `TRADER_ONLY`, `SHARED_PASSIVE`, `SHARED_MARKET`
  - Added `ownership_allows_worker()` so worker code can make the same seat-sharing decision deterministically
- `tools/render_bot_inventory_policy.py`
  - Extended the markdown contract to include an `Ownership` column
  - Kept backward compatibility with the older 6-column table so old policy files still render safely during migration
- `tools/range_bot.py`
  - Added shared helpers for open-trade conflict detection and direction parsing
  - Range bot now distinguishes discretionary open trades from worker trades instead of blanket-skipping any open pair
  - Enforces `Ownership` by order type: passive LIMITs may coexist only when policy says so; market-style adds require `SHARED_MARKET`
  - Enforces `MaxPending` while keeping, replacing, placing, and cleaning pending range orders
- `tools/trend_bot.py`
  - Reused the same ownership-aware open-trade deconfliction helpers
  - Trend bot now only shares a pair with discretionary live inventory when policy explicitly says `SHARED_MARKET`
- `tools/bot_inventory_snapshot.py` and `tools/bot_trade_manager.py`
  - Count only entry pending orders for snapshot / projected-margin views
  - Surface `ownership` + `max_pending` in the trader-facing bot snapshot
- `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`, `AGENTS.md`, `CLAUDE.md`
  - Updated the canonical policy format and runtime docs to include `Ownership`

**Why**: `PAUSE` should express market judgment, not be abused as duplicate-execution bookkeeping. With an explicit ownership contract, trader and worker can intentionally share a pair when desired, and stay out of each other’s way when not. With `MaxPending` enforced in code, the policy schema finally means what it says.

## 2026-04-17 — Repair audit handoff persistence and pretrade outcome matching

**Problem**: Two feedback loops were still decaying in production.

1. `tools/quality_audit.py` rewrote `logs/quality_audit.md` from scratch every cycle. On a `CLEAN` run it replaced the whole file with a 3-line stub, so the next audit lost the previous `## Auditor's View`, and trader sessions saw either nothing or only the stub.
2. `tools/daily_review.py` had regressed to same-day-only `pretrade_outcomes` matching and could reuse the same `trade_id` for multiple pretrade rows. Cross-day closes were missed, duplicate labels accumulated, and audit opportunity scoring also misread closed-trade direction because it inferred side from absolute units.
3. `collab_trade/memory/pretrade_check.py` still left `pretrade_outcomes.thesis` blank, so downstream review had no compact record of what setup was actually being evaluated.

**Fix**:
- `tools/quality_audit.py`
  - Added `extract_auditor_view()` + `merge_previous_auditor_view()`
  - Script facts now preserve the previous `## Auditor's View` block until the current cycle writes a new one
  - Always writes `logs/quality_audit.json`, even on clean cycles, instead of deleting it
- `docs/SKILL_quality-audit.md`
  - Switched Bash D to `.venv/bin/python tools/chart_snapshot.py --all` so the visual chart pass keeps working when host `python3` lacks `matplotlib`
- `tools/session_data.py`
  - Always shows the recent `quality_audit.md` block instead of suppressing it on `CLEAN`
  - Expanded the displayed snippet so the preserved auditor narrative is visible to the trader
- `tools/daily_review.py`
  - Restored recent-unmatched matching across the last 3 days
  - Matches the nearest pretrade check before the trade entry/fill time instead of pop-by-key
  - Repairs duplicate `trade_id` links by keeping only the newest linked pretrade row
  - Carries `entry_time` / `close_time` through closed-trade parsing
  - Uses explicit `direction` for audit opportunity outcome analysis instead of guessing from absolute units
  - Daily review output now includes carry-over pretrade matches when a prior-day setup closes today
- `collab_trade/memory/pretrade_check.py`
  - Added compact thesis logging (`edge / allocation / wave / top details`) into `pretrade_outcomes.thesis`

**Verification**:
- `python3 tools/quality_audit.py` → `CLEAN (0.5s)`
- `.venv/bin/python tools/quality_audit.py` → `CLEAN (0.6s)`
- `python3 tools/chart_snapshot.py --all` → fails on this host with `ModuleNotFoundError: matplotlib`
- `.venv/bin/python tools/chart_snapshot.py --all` → generated 14 charts successfully
- `python3 tools/daily_review.py --date 2026-04-16` → report generated with deduped pretrade links
- `.venv/bin/python tools/daily_review.py --date 2026-04-16` → report generated with deduped pretrade links
- `python3 tools/session_data.py` → recent `QUALITY AUDIT` section rendered from the persisted file
- `.venv/bin/python tools/session_data.py` → recent `QUALITY AUDIT` section rendered from the persisted file
- `python3 collab_trade/memory/pretrade_check.py GBP_USD SHORT`
- `.venv/bin/python collab_trade/memory/pretrade_check.py GBP_USD SHORT`
- `sqlite3 collab_trade/memory/memory.db "SELECT ... thesis ..."` → latest rows include thesis text

## 2026-04-17 — Worker deconfliction now reads discretionary pending entries

**Problem**: The worker layer already skipped same-pair live positions, but it still could not see discretionary pending entry orders. That forced the trader to `PAUSE` pairs in policy just to avoid duplicate limits such as the AUD_JPY floor-defense thesis. The result was the wrong abstraction: policy was doing pair-ownership bookkeeping that the code itself should have handled.

**Fix**:
- `tools/range_bot.py`
  - Added generic pending-order helpers that distinguish entry orders from protective orders
  - Range bot now skips same-pair, same-direction discretionary pending entries in code
- `tools/trend_bot.py`
  - Reused the same pending-entry deconfliction helpers
  - Trend bot now also skips same-pair, same-direction discretionary pending entries in code
- `AGENTS.md`
  - Documented that worker deconfliction now includes trader pending entries, not just live positions

**Why**: Market permission and ownership are different. Policy should say whether a market is worth trading. Code should prevent duplicate execution against the trader's existing live or pending book. Moving duplicate-limit avoidance into code keeps the worker fast without using `PAUSE` as a blunt workaround.

## 2026-04-17 — Fix live-vs-state drift visibility for quality-audit and trader sessions

**Problem**: `tools/quality_audit.py` self-check still parsed only the legacy `### EUR_USD LONG ...` state format. The live trader handoff now writes open positions under `## Positions (Current)` as plain lines like `GBP_USD SHORT 3000u id=...`, so audit runs were raising false `state.md parsed []` failures even when the handoff was correct. Even when a real mismatch existed, the output only said `FAILED` without telling the trader whether the problem was a live orphan, a missing state update, or both.

**Fix**:
- `tools/quality_audit.py`
  - Added a dedicated `parse_state_positions()` helper
  - Parse the current `## Positions (Current)` block first, matching plain `PAIR LONG/SHORT 3000u` lines
  - Keep the old `### ...` header parser as a fallback for older state snapshots
  - Continue excluding `LIMIT` lines so pending orders are not misread as held inventory
  - Self-check now reports concrete `OANDA only` vs `state.md only` positions, including trade IDs and manual/no-log context for live orphan candidates
- `tools/session_data.py`
  - Added a `STATE POSITION SYNC` block at session start
  - Warns when live OANDA positions and `## Positions (Current)` disagree, without auto-editing `state.md`
  - Prints both sides of the mismatch so the trader can repair the handoff before analysis

**Verification**:
- `python3 tools/quality_audit.py` → `EXIT:0`, `CLEAN`
- `.venv/bin/python tools/quality_audit.py` → `EXIT:0`, `CLEAN`
- `python3 tools/session_data.py` → `STATE POSITION SYNC` block rendered with actual sync status
- `.venv/bin/python tools/session_data.py` → `STATE POSITION SYNC` block rendered with actual sync status

## 2026-04-16 — Trader now owns routine worker policy; inventory-director demoted to daily backup

**Problem**: The system had three different cadences touching the worker book: a 60-second deterministic emergency brake, a 5-minute LLM inventory task, and the 20-minute discretionary trader. That was the wrong split. The fast risk was already covered by `bot_trade_manager.py`, while the 5-minute LLM layer duplicated judgment, created ownership blur, and still let stale policy files become a separate failure mode. Routine worker inventory decisions belong with the trader who already reads the whole market every 20 minutes.

**Fix**:
- `docs/SKILL_trader.md`
  - Promoted `qr-trader` from worker-book awareness to routine worker-book ownership
  - Added mandatory worker-policy rewrite + JSON render every session
  - Added direct worker cleanup commands and updated the `## Bot Layer` handoff block
- `docs/SKILL_inventory-director.md`
  - Reframed `inventory-director` as a daily backup repair task
  - Limited it to stale/missing/corrupt policy repair and low-frequency stranded-book cleanup
- `AGENTS.md`
  - Updated cadence, ownership, and runtime descriptions so the repo agrees that trader owns intraday worker policy
- `tools/bot_policy.py`, `tools/render_bot_inventory_policy.py`
  - Updated helper text so the live policy is no longer described as inventory-director-owned by default

**Why**: One fast deterministic brake plus one trader brain is coherent. A separate 5-minute LLM manager was neither fast enough to be the brake nor authoritative enough to be the trader. Making `qr-trader` the routine worker owner removes duplicate judgment and leaves `inventory-director` as the recovery layer it should be.

## 2026-04-16 — Split edge from allocation and persist audit narrative picks

**Problem**: The trading loop was collapsing too much into one `S/A/B/C` letter. Strong execution-time reads could be demoted just because allocation needed to stay smaller, counter-trades were structurally capped at `B`, and `audit_history.jsonl` only preserved scanner fires, not the auditor's actual A/S market view. That made `S` look artificially scarce and taught daily-review the wrong lesson.

**Fix**:
- `collab_trade/memory/pretrade_check.py`
  - Split setup output into `edge_grade` and `allocation_grade`
  - Normal setups default to `edge = allocation`
  - Counter-trades can now reach `Edge S/A`; allocation is still smaller when fighting upper-TF flow, but no longer hard-capped at `B`
  - CLI output now prints `Edge` and `Allocation` separately
- `tools/session_data.py`
  - Updated the new-entry template so theme confidence changes allocation, not edge
- `tools/record_audit_narrative.py`
  - New helper that parses the final auditor-written `logs/quality_audit.md`
  - Appends `narrative_picks` and `strongest_unheld` into `logs/audit_history.jsonl`
- `tools/quality_audit.py`, `tools/daily_review.py`
  - Scanner history now tags itself with `source=s_scan`
  - Daily review now analyzes both scanner detections and final narrative A/S picks
- Docs / prompts
  - Updated trader + quality-audit + daily-review prompts, plus `AGENTS.md` / `CLAUDE.md`, to use the two-axis language and to persist explicit unheld narrative opportunities

**Why**: A setup can be locally dominant without deserving max size. Edge and allocation answer different questions. Once they were separated, the system could stop pretending that only "global all-clear" setups count as `S`, and the audit learning loop could finally remember what the auditor actually saw.

## 2026-04-16 — Bot inventory ownership split: trader aware, inventory-director owns worker book

**Problem**: The prompts still blurred ownership between discretionary trading and worker inventory. `qr-trader` was being told to actively manage bot-tagged positions, while `inventory-director` already existed as the dedicated worker-book owner. That overlap invites double management, conflicting closes, and accountability drift. The inverse risk is just as bad: an inventory task should never touch discretionary `trader` inventory.

**Fix**:
- `docs/SKILL_trader.md`
  - Changed bot inventory handling from direct management to awareness-first
  - Trader now requests bot changes via `state.md` and only uses direct close/cancel on bot tags for emergency override
  - Updated the `## Bot Layer` handoff block so ownership stays with `inventory-director`
- `docs/SKILL_inventory-director.md`
  - Explicitly states that `inventory-director` owns all worker tags
  - Explicitly forbids it from managing discretionary `trader` inventory
- `AGENTS.md`, `CLAUDE.md`
  - Updated tag-ownership language so trader awareness and inventory-director ownership are aligned repo-wide

**Why**: Fast worker inventory and discretionary trader inventory are different books. If both tasks can manage both books by default, the system will eventually fight itself. Awareness should be shared; ownership should not.

## 2026-04-16 — Trend bot added; stale policy no longer kills fast entry after one missed heartbeat

**Problem**: The local layer could now correctly refuse range fades during band-walks, but that still left a major hole: trend continuation itself was not being traded by the worker at all. At the same time, one missed `inventory-director` update could still drop the whole fast layer into `REDUCE_ONLY`, which turned scheduler drift into pure opportunity loss.

**Fix**:
- `tools/trend_bot.py`
  - Added a new deterministic trend-continuation worker
  - Requires aligned `H1/M15` trend, `M5` band-walk or follow-through continuation, supportive `M1` momentum, and 7-pair currency-pulse confirmation
  - Fires MARKET-only continuation orders tagged `trend_bot_market`
  - Can add one worker trend position alongside an existing same-direction discretionary trade, while still refusing opposite-side conflict
- `tools/local_bot_cycle.sh`
  - Now runs `bot_trade_manager.py` -> `trend_bot.py` -> `range_bot.py` every minute
- `tools/bot_trade_manager.py`, `tools/bot_inventory_snapshot.py`
  - Added `trend_bot_market` to the worker tag set / legend so the manager and inventory director see the trend book as first-class inventory
- `tools/bot_policy.py`
  - Extended ACTIVE stale-policy grace from 15 minutes to 60 minutes so one missed inventory-director heartbeat does not instantly shut off fast entry
- Docs
  - Updated `AGENTS.md`, `CLAUDE.md`, `docs/SKILL_trader.md`, `docs/SKILL_inventory-director.md`, `docs/SKILL_trend-bot.md`, and `collab_trade/state.md`

**Why**: If the worker is fast enough to detect band-walks, it also needs a way to trade them. Range veto without trend follow just replaces one form of opportunity loss with another. And if the local layer dies after one stale policy file, the system is still too brittle for a 60-second engine.

## 2026-04-16 — Fix local automation lock path for quality-audit

**Problem**: `quality-audit` automation could start, but its preflight failed before lock acquisition because `tools/task_lock.py` resolved the repo root one directory too high. It tried to create lock files under `/Users/tossaki/App/logs/locks/...` instead of `/Users/tossaki/App/QuantRabbit/logs/locks/...`, which broke local automation runs under sandboxed writable roots.

**Fix**:
- `tools/task_lock.py`
  - Fixed repo-root resolution from `parents[2]` to the actual repo root (`parent.parent`)
  - Lock files now write under the project `logs/locks/` directory as intended
- `tools/market_monitor.py`
  - Fixed the same repo-root resolution bug so its logs/config paths also resolve inside the QuantRabbit workspace

**Why**: Local automations only work if every runtime helper resolves paths inside the real workspace. A one-level path error is enough to make the task look alive while the actual playbook never starts.

## 2026-04-16 — Range bot stops fading band-walks and reads cross-currency pulse

**Problem**: The local range worker had become faster and deeper, but it could still misread a band-walk as a fadeable Bollinger extreme. It was also still too pair-local: even with a 60-second cadence, it was not yet using `M15` or the 7-pair currency map to tell "healthy range edge" from "real directional expansion," which meant both false fades and opportunity loss.

**Fix**:
- `tools/range_bot.py`
  - Added explicit `M5` band-walk veto so the bot does not fade a wall-hugging move just because BB position is extreme
  - Added `M15` context checks for fast breakout / band-walk / squeeze pressure
  - Added 7-pair currency-pulse scoring so each setup is filtered by base-vs-quote pressure across the whole basket
  - Conviction / signal strength / MARKET readiness now incorporate `M15`, `H1`, `M1`, and cross-currency alignment instead of a single-pair `M5` read
- `tools/local_bot_cycle.sh`
  - Expanded local refresh from `M1/M5/H1` to `M1/M5/M15/H1`
- Docs
  - Updated `AGENTS.md`, `CLAUDE.md`, `docs/SKILL_range-bot.md`, and `collab_trade/state.md` to reflect the new band-walk veto and currency-pulse logic

**Why**: A range bot should fade exhaustion, not momentum. The extra minute is best spent reading one more timeframe and the whole currency board, so the worker can take more real range chances while refusing the moves that are actually breaking out.

## 2026-04-16 — Trader prompt aware of live bot layer + tag ownership

**Problem**: `qr-trader` could still behave as if the local worker book were invisible. That is a serious ownership bug: worker orders/trades are real inventory, and without explicit tag semantics the trader can duplicate, conflict with, or ignore them.

**Fix**:
- `docs/SKILL_trader.md`
  - Added mandatory `bot_inventory_snapshot.py` read every session
  - Added explicit tag ownership for `trader`, `range_bot`, and `range_bot_market`
  - Added required `## Bot Layer` state block whenever bot-tagged inventory exists
- `tools/bot_inventory_snapshot.py`
  - Added a tag legend in the human-readable snapshot output
- `AGENTS.md`, `CLAUDE.md`, `docs/SKILL_inventory-director.md`
  - Added shared tag taxonomy and responsibility language so trader/inventory-director agree on who owns what

**Why**: Tags are not bookkeeping trivia. They are the boundary between discretionary inventory and worker inventory. If the trader does not read that boundary every session, the system will eventually manage the same exposure twice or not at all.

## 2026-04-16 — Range bot MTF depth: H1 regime + M1 trigger, still 1-minute fast

**Problem**: The local worker had been fixed for freshness, but the actual entry logic was still too shallow: `range_bot.py` scanned `M5` only, so it could fade into an `H1` breakout or use MARKET without a real `M1` trigger. That is fast, but not deep enough for stranded-book risk.

**Fix**:
- `tools/range_bot.py`
  - Added `H1` context checks for breakout-risk / squeeze-risk before any range fade is accepted
  - Added `M1` micro-timing checks (StochRSI / CCI / wick pressure / divergence / short momentum state)
  - MARKET fallback now requires aligned `M1` timing; otherwise S/A setups stay passive as LIMITs
  - Conviction is now adjusted by MTF context instead of `M5` alone
- `tools/local_bot_cycle.sh`
  - Expanded local refresh from `M5/H1` to `M1/M5/H1` before protection and entry
- `tools/refresh_factor_cache.py`
  - Added automatic re-exec under `.venv/bin/python` when `python3` lacks `pandas`, so technical refresh works from both invocation paths
- `tools/bot_policy.py`
  - Added a short ACTIVE grace window after policy expiry so the local bot does not fall into instant `REDUCE_ONLY` just because one inventory-director heartbeat is late
- Docs
  - Updated `AGENTS.md`, `docs/SKILL_range-bot.md`, and `collab_trade/state.md` to reflect the new MTF worker logic

**Why**: Fast is only useful if the bot is reading the right layers of the market. `M5` finds the range edge, `H1` says whether that edge still belongs to a range, and `M1` says whether the entry is actually ready now.

## 2026-04-16 — Trader prompt: kill one-limit-and-done deployment

**Problem**: The trader prompt made under-deployment look acceptable. Tier 1 had a single `My trade` line, Tier 2 only required one `LONG if / SHORT if` pair then often defaulted to a single pullback LIMIT, and the flat-book blocker could be bypassed by leaving one orphan order. Net effect: the trader often saw a live move, posted one passive LIMIT, and finished the session flat.

**Fix**:
- `docs/SKILL_trader.md`
  - Replaced single-trade Tier 1 / promotion blocks with a required execution ladder:
    - `NOW`
    - `RELOAD`
    - `SECOND SHOT / OTHER SIDE`
  - Reworked Tier 2 from `LONG if / SHORT if` into `NOW / RELOAD / OTHER SIDE` so each pair must name:
    - immediate live execution
    - pullback reload level
    - failure / opposite-side contingency
  - Updated examples so trend candidates can no longer be expressed as one pullback-only LIMIT
  - Tightened the flat-book blocker to reject fake deployment states:
    - no live action
    - only one same-side pullback LIMIT
    - no fallback path for active setups
  - Changed market-vs-limit guidance so live S/A moves require `market now + reload LIMIT`, and B setups require two executable paths instead of `LIMIT always`
  - Raised pending LIMIT cap from 2 to 3 so the trader can leave reload + second-shot coverage without spraying orders

**Why**: The problem was not lack of analysis; it was a prompt format that allowed "one LIMIT and done." The new format forces explicit participation in the live move, the pullback, and the miss/failure case. If the trader stays flat now, the missing deployment path is visible.

## 2026-04-16 — Local bot layer + LLM inventory director

**Problem**: The fast edge belongs to deterministic local entry, not to an LLM loop. But the old local bot failed because no one owned the inventory book as a book: when range fades got stranded on both sides, pending worst-case fills and deadlocked exposure drifted into closeout. Running `range-bot` / `bot-trade-manager` as Codex automations also created interference risk with the real trader.

**Fix**:
- Local deterministic bot layer
  - `tools/range_bot.py` now reads `logs/bot_inventory_policy.json`
  - New `tools/local_bot_cycle.sh` runs `bot_trade_manager.py` then `range_bot.py` under one short lock-protected local cycle
  - `tools/local_bot_cycle.sh` now refreshes OANDA-backed `M5/H1` technical cache before protection/entry, so the worker no longer waits on trader sessions for candle freshness
  - New `scripts/install_local_bot_launchd.sh` / `scripts/uninstall_local_bot_launchd.sh` install/remove `com.quantrabbit.local-bot-cycle`
- LLM inventory control layer
  - New `docs/SKILL_inventory-director.md` + `docs/codex_automations/inventory-director.md`
  - New `tools/bot_inventory_snapshot.py` gives a stable view of bot pending/trades/projected margin/deadlock
  - New `tools/render_bot_inventory_policy.py` converts the human policy markdown into deterministic JSON
  - New `tools/cancel_order.py` lets the inventory director cancel pending bot orders with proper logging
- Emergency-only local guard
  - `tools/bot_trade_manager.py` is narrowed to a deterministic emergency brake: policy pause, panic margin, rollover, or deadlock pressure
  - Normal stranded inventory judgment moves to `inventory-director`
- Runtime / docs
  - `tools/task_runtime.py` now supports `inventory-director` preflight/release
  - `AGENTS.md`, `CLAUDE.md`, and range-bot docs now describe the system as:
    - local bot = fast entry + emergency brake
    - inventory-director = LLM inventory owner
    - trader = upper discretionary manager

**Why**: Speed should stay local. Inventory judgment should stay with the LLM. Closeout prevention still needs a deterministic brake because closeout happens faster than a deliberative loop.

## 2026-04-16 — Fast Bot Layer: 1-minute entry + 1-minute protection manager

**Problem**: The old fast bot edge was real, but the failure mode was always the same: fast entry accumulated exposure much faster than the 20-minute trader cadence could unwind it. Archive history and current strategy memory both point to the same tail risk: pending orders stacked into trapped long/short books, then `MARKET_ORDER_MARGIN_CLOSEOUT` or forced close cleaned up the damage. The previous `range_bot` also was not safe to run every minute because it cancelled/repriced too aggressively.

**Fix**:
- `tools/range_bot.py`
  - Removed blanket cancel-and-replace behavior
  - Keeps fresh pending bot LIMITs when direction and edge are still valid
  - Reprices only when age/drift exceeds tolerance or a live setup upgrades to MARKET
  - Becomes safe to run on a 1-minute heartbeat as the fast entry layer
- `tools/bot_trade_manager.py`
  - New 1-minute protection manager for `range_bot` / `range_bot_market`
  - Cancels stale or dangerous pending bot orders
  - Detects projected-margin stress and deadlocked bot books
  - Partially/full closes trapped bot trades before closeout pressure escalates
- Prompt / task sync
  - Added canonical prompt + Codex wrapper for `bot-trade-manager`
  - Updated task registry and architecture docs to define the split clearly:
    - `range-bot` = fast entry
    - `bot-trade-manager` = fast protection
    - `trader` = discretionary override and broader account management

**Why**: Fast entry without fast relief is not a trading edge, it's a delayed closeout. The profitable part of the old bot came from speed; the fatal part came from stranded inventory. This change keeps the speed while giving the fast book its own risk owner.

## 2026-04-16 — Memory DB: actionable recall + current-format ingest fix

**Problem**: `memory.db` was being written, but the trader was not using it as a real decision aid. Three concrete issues were blocking it:
- `session_data.py` only showed memory recall for held pairs, so pending orders and fresh scanner candidates got no DB context
- `recall.py` keyword search used the whole query as one LIKE string, making retrieval mostly vector-only and noisy
- `ingest.py` / `parse_structured.py` still assumed many `trades.md` sections started with `###`, while current logs use `##` heavily, so fresh trade narrative chunks were being missed or weakly parsed

**Fix**:
- `tools/session_data.py`
  - Replaced held-only memory recall with **ACTIONABLE MEMORY**
  - Pulls memory for held positions, pending orders, and top scanner candidates
  - Uses pair-filtered hybrid search and prints concise, decision-ready hits instead of raw dumps
  - De-prioritizes `state.md` self-echoes so current-day theses do not masquerade as historical memory
- `collab_trade/memory/recall.py`
  - Switched keyword search from whole-string LIKE to token-based scoring
  - Orders hits by match score, recency, and pair filter so pair/direction/regime terms actually matter
- `collab_trade/memory/pretrade_check.py`
  - Pair-filters similar-memory lookup, enriches the query with direction/wave/lesson context, and prefers real trade/lesson chunks over `state.md` theses
- `collab_trade/memory/ingest.py`
  - Added `##` + `###` markdown section splitting for current trade log format
  - Better detects trade-table sections so fresh narrative chunks enter `memory.db`
- `collab_trade/memory/parse_structured.py`
  - Added support for current English-table trade records (`Pair`, `Direction`, `Reason`, `Lesson`, `P/L`, `id`)
  - Accepts `JPY` P/L notation and newer `##` trade headers

**Why**: A memory system that only archives is not a brain. These changes make the DB visible at the moment of decision and keep the latest trade narrative flowing into retrieval, so Codex can actually use past outcomes and similar setups when managing or initiating trades.

## 2026-04-16 — Range Bot: avoid OANDA immediate cancel on crossed passive limits

**Problem**: Order `468224` (`USD_JPY` short LIMIT by `range_bot`) was cancelled by OANDA immediately with transaction reason `STOP_LOSS_ON_FILL_LOSS`. The order was effectively marketable at placement, while the stop-loss trigger side was already through the SL.

**Fix**:
- `tools/range_bot.py`
  - Added guard for marketable LIMIT detection at live quotes
  - Added guard for "stop trigger already through SL" before placement
  - Passive `B` setups now `SKIP` instead of sending crossed LIMITs that OANDA will auto-cancel
  - Wide-spread or over-progressed crossed limits also `SKIP` instead of chasing

**Why**: If a passive LIMIT is already crossed, it is no longer passive. Sending it just hands validation to OANDA and creates fake "placed" logs. The bot now rejects that state locally.

## 2026-04-16 — Range Bot: S/A market fallback for live edge

**Problem**: LIMIT-only range entries were too slow when a clean range touched the BB edge and immediately bounced. The trader playbook already allows MARKET on S/A extremes under normal liquidity, but `range_bot.py` could only wait passively.

**Fix**:
- `tools/range_bot.py`
  - Added live LIMIT vs MARKET routing
  - S/A conviction only: use MARKET when price is still at the BB edge or has only bounced a small distance off it
  - Keeps LIMIT as the default for B setups, wide spreads, or moves that already traveled too far toward TP
  - Uses reduced size for MARKET fills and separate tag `range_bot_market`
- `docs/SKILL_range-bot.md`, `AGENTS.md`, `CLAUDE.md`
  - Updated docs to reflect the new live-entry behavior
- `docs/codex_automations/range-bot.md`, `tools/check_task_sync.py`
  - Added the missing Codex wrapper and sync guard for `range-bot`

**Why**: A range edge that is live now should be hit now. A range that already ran halfway to target should still wait on LIMIT. This keeps the bot aggressive only where the edge is still present.

## 2026-04-16 — Trader automation moved back to 20-minute heartbeat

**Problem**: The Codex app UI showed `QR Trader` as `毎週 0:00` / `Next run: Tomorrow at 0:00` even though the automation TOML used a dense weekly RRULE with `BYMINUTE=0,20,40`. That made the 20-minute trader cadence untrustworthy in practice.

**Fix**:
- Created a dedicated heartbeat automation `qr-trader-20m` with `FREQ=MINUTELY;INTERVAL=20`
- Paused the fallback cron automation `qr-trader` to prevent duplicate runs
- Updated `AGENTS.md` to reflect the real trader runtime again: heartbeat every 20 minutes

**Why**: Minute-based cadence is what the trader session actually needs. The heartbeat form matches the app's supported 20-minute scheduling model directly, instead of relying on a weekly RRULE that the UI/scheduler may flatten incorrectly.

## 2026-04-16 — Codex migration scaffolding: canonical prompts + shared runtime + automation wrappers

**Problem**: The live trading playbooks were effectively tied to Claude scheduled-task plumbing. `docs/SKILL_trader.md` embedded Claude-specific process matching (`pgrep -f "bypassPermissions"`), Codex had no repo-tracked wrapper prompts, and there was no guard to keep Claude compatibility links and Codex prompts pointing at the same canonical playbooks. Migrating to Codex risked prompt drift and unsafe overlap.

**What changed**:
- Added `tools/task_runtime.py` — host-neutral runtime helper for shared prompts
  - `trader preflight/start/cycle/watchdog` replaces host-specific shell glue while preserving `.trader_lock`, `.trader_start`, stale handoff ingest, and `session_end.py` gatekeeping
  - `quality-audit preflight/release` wraps `tools/task_lock.py` so audit runs can skip/release cleanly under either Claude or Codex
- Added `tools/check_task_sync.py` — verifies the canonical prompt files, Claude compatibility symlinks, and Codex wrapper prompts stay aligned
- Added `docs/codex_automations/*.md` — thin Codex-only wrapper prompts that reference the canonical `docs/SKILL_*.md` files instead of duplicating task logic

**Canonical prompt changes**:
- `docs/SKILL_trader.md`
  - Replaced Claude-specific Bash scaffolding with `python3 tools/task_runtime.py trader preflight/start/cycle`
  - Preserved the same trading logic, state/logging rules, and `session_end.py`-only release discipline
- `docs/SKILL_quality-audit.md`
  - Added Step 0 audit lock acquisition via `task_runtime.py`
  - Added Step 8 audit lock release via `task_runtime.py`

**Architecture docs**:
- `AGENTS.md`
  - Updated scheduled-task architecture to Codex automations
  - Declared `docs/SKILL_*.md` as the single source of truth, with Claude symlinks and Codex wrappers as compatibility layers
  - Updated cadence references (trader 20 min, quality-audit 45 min) and documented the new runtime helper/checker

**Safety outcome**: Claude/Codex can now share the same canonical playbooks without embedding host-specific process assumptions in the prompt itself. Codex automation prompts are intentionally thin wrappers, so future trading logic edits only happen once.

## 2026-04-16 — Range Bot: Automated range scalp entry bot

**Problem**: Trader task drought — 18 LIMITs cancelled vs 24 entries on 4/15-16. 12 layers of rules collectively block all entries despite 0% margin. 5 S-scan signals missed.

**Solution**: `tools/range_bot.py` — separate scheduled task (20-min cron) that detects ranges via range_scalp_scanner and places LIMIT orders at BB extremes automatically. Entries only — exits handled by trader task's profit_check.py.

**Key design**:
- LIMIT orders at BB lower (BUY) / BB upper (SELL) with TP at BB mid + SL outside range
- 10 safety layers: market state, poison hours (19-23 UTC), margin cap (30% NAV), min 3k/max 5k units, conviction B+ filter, pair deconfliction, GTD 4h auto-expiry, mandatory SL, tag isolation (clientExtensions "range_bot")
- Cancel-then-replace: every cycle cancels stale bot LIMITs and places fresh ones at current BB levels
- Zero changes to trader task: discovers bot positions via profit_check.py --all (already runs every session)

**Files**: `tools/range_bot.py` (new), `docs/SKILL_range-bot.md` (new), CLAUDE.md (updated architecture table + scripts table)

## 2026-04-16 — Acceleration Engine: Hero Pair + Mandatory Rotation + Momentum-S

**Problem**: 28 days of trading = -755 JPY total. System has zero growth. But 5 good days averaged +5,385 JPY each. The good days had a clear recipe that wasn't being replicated.

**Root cause analysis** (500-trade deep dive):
- Rotation chains (TP → re-enter <60min) = +72,774 JPY. Everything else = -73,529 JPY
- Good days: 1 hero pair produces 36-143% of P&L via rotation. WR 71%
- Bad days: capital spread across 5+ pairs equally. WR 32%. No rotation
- Momentum-S scanner: 96% accuracy, 55+ fires, ZERO entries. System's biggest leak

**SKILL_trader.md changes**:
- **Hero Pair system**: Replaces "Top 2 pairs" with Hero (50%+ capital, all rotations) + Sidekick (20-30%). Other pairs get B-size max. Forces concentration that made every good day work
- **Mandatory Rotation Engine**: After EVERY TP, must place re-entry LIMIT within 30 seconds. Includes size escalation table (3k→4k→6k→8k as theme confirms). Rotation is no longer optional — it's the only thing that makes money
- **Momentum-S forced entry**: When ≥3 pairs fire same direction, enter hero pair at A-size market order. No LIMIT, no "consider." 96% accuracy can't be optional
- All "Top 2" references → "Hero pair" throughout

**strategy_memory.md changes**:
- Added "Good Day Recipe" at top of winning patterns (4 shared traits of 5 best days)
- Added Momentum-S as proven winning pattern with forced entry rule
- Updated reference day from 3/31 (+4,591) to 4/7 (+11,014) as the target blueprint
- Added rotation as mandatory system component (+72,774 data)

## 2026-04-16 — Data-Driven Sizing: min 3k, sweet spot 30-120min, circuit breaker, time filter

**Problem**: 500-trade deep analysis reveals 3 structural profit killers:
1. **Size**: 330 trades at 1-2k units = -23,098 JPY. Same trader at 3-5k = +19,226 JPY. S-size (9k+) = 4/4 wins, +5,921 JPY
2. **Hold time**: <30min trades = -5,893 JPY. 30-120min = +27,539 JPY. 2h+ = -21,593 JPY
3. **Time of day**: 22:00-23:00 UTC = -12,191 JPY (37 trades). 08:00 UTC = +7,344 JPY (37 trades)
4. **No circuit breaker**: 4/15 had 16 consecutive losses, WR 23.8%, -5,226 JPY day

**SKILL_trader.md changes**:
- **Minimum 3,000u per entry** (was 2,000u). Below 3k is proven net-negative. All B-size references updated
- **S-size minimum raised to 8,000u** (was ~6,000u effective). S-size has 100% WR (4/4, +5,921)
- **30-120min sweet spot awareness**: Added hold time data to Self-check. <5min and 2h+ are explicitly flagged
- **3-loss circuit breaker**: 3 consecutive losses in same direction → STOP that direction this session
- **Time-of-day filter**: 19:00-23:00 UTC entries require "late session penalty" block — LIMIT only, B-size max

**strategy_memory.md changes**:
- Added SIZE/HOLD TIME/TIME OF DAY evidence blocks at top of winning patterns (with exact numbers)
- Added circuit breaker to loss patterns section
- Updated minimum size reference from 2,000u to 3,000u

## 2026-04-16 — Remove deprecated Japanese prompt files

Deleted `CLAUDE_ja.md` and `.claude/rules-ja/` (6 files). English versions are the single source of truth since v8. Japanese copies were never updated and just added confusion.

## 2026-04-16 — Anti-Drought: Entry-First format + drought detector

**Problem**: Trader task produces beautiful analysis (Currency Pulse, Regime Map, 7-pair scan) but exits sessions with 0 entries, 0% margin, 0 JPY. On 4/15-16: 18 LIMITs cancelled vs 24 entries. 7/7 pairs rated "C → no action" despite 0% margin. 5 S-scan signals missed. Root cause: 12 layers of rules (event wait, spread check, theme late, B-max, etc.) each individually valid but collectively creating a situation where NO entry can pass all filters.

**SKILL_trader.md changes**:
- **Tier 2 "ENTRY FIRST" format**: Old format allowed "C → pass" in 3 seconds. New format requires writing "LONG if: [price+condition]" and "SHORT if: [price+condition]" for EVERY pair BEFORE allowing conviction/skip. Makes skipping expensive and entering cheap. Next session checks if the "if" conditions triggered
- **Drought detector in Self-check**: Counts entries vs sessions elapsed. 0 entries after 5+ sessions = DROUGHT → forced B-size market entry. 0 entries + 0% margin = CRITICAL DROUGHT. Includes LIMIT trap detection (cancelled > filled for 3+ sessions = forced market order)
- **0% margin blocker escalation**: In drought, the blocker now says "pick the closest Tier 2 'if' condition and enter market order NOW" instead of asking 3 philosophical questions

**strategy_memory.md changes**:
- **Added anti-drought header**: "The #1 profit killer is NOT entering" with asymmetry math (bad trade -354 JPY vs drought day -12,569 JPY = 36:1)
- **Added fear/courage rebalancing**: Warning before loss patterns section that 30+ warnings create paralysis. Reminder after winning patterns to "go find one and ENTER"
- **New top principle**: "入らないことが最大のリスク" (not entering is the biggest risk) added as first mental principle, above all other rules

## 2026-04-16 — Quality Audit v3: Deep analysis rewrite + model correction

**Problem**: Quality audit was completing only 3% of cycles with full analysis (7-pair predictions). 41% of cycles were abandoned (script ran but Sonnet wrote nothing). 48% were partial (position challenges only, no independent market view). Root cause: SKILL design allowed early exit, didn't force structured output, and lacked the "output format forces thinking" principle that makes the trader SKILL effective.

**SKILL_quality-audit.md rewrite**:
- **Removed early exit** (Step 3 old → Step 3 new "minimum output gate"). Section E (Regime Map) + Section C (7-pair conviction map) are now mandatory every cycle, even with 0 positions
- **Structured fill-in format for predictions**: Each pair gets "Chart tells me / Story / Price target / Wrong if / Conviction" — same design principle as trader SKILL's Tier 1/Tier 2 blocks. Sonnet can't skip without leaving visible blanks
- **Completion gate** (Step 5): Checklist that must be satisfied before saving. Same pattern as trader's SESSION_END blocker
- **Removed maxTurns: 30** — trader has no maxTurns limit; audit shouldn't either
- **S-conviction Slack notification**: Now posts to Slack when S-conviction found on unheld pair (not just DANGER). Ensures trader sees opportunities even if they don't read quality_audit.md thoroughly
- **Chart PNG fallback**: Explicit instructions for when PNGs are unavailable
- **5 filled-in examples** for M5 Visual (TREND/RANGE/SQUEEZE/EXHAUSTION/REVERSAL) — Sonnet mimics examples

**Schedule change**: 30-min cron → 45-min cron (`3,48 * * * 1-6`). Deeper analysis per cycle, fewer shallow cycles. Each audit has more time to read all 7 chart PNGs and write meaningful predictions.

**CLAUDE.md corrections**:
- trader model: Opus → Sonnet (default) — was always Sonnet (no model field in schedule.json), CLAUDE.md was wrong
- daily-performance-report, daily-slack-summary, intraday-pl-update: Opus → Sonnet (default) — same reason
- quality-audit interval: 30 min → 45 min, description updated to reflect mandatory output + S-conviction Slack

## 2026-04-15 — Fix: All summary tasks using wrong day boundary (UTC→JST)

**Problem**: All 4 summary/reporting scripts grouped trades by UTC date (00:00-23:59 UTC) instead of JST date (00:00-23:59 JST). Trades between 00:00-08:59 JST were attributed to the previous day. April 14 showed +1,580 JPY (UTC) instead of the actual +6,745 JPY (JST) — a 5,165 JPY discrepancy.

**Additional bug**: `slack_daily_summary.py` calculated daily return % using current balance (which already includes subsequent days' P&L) instead of the actual day-start balance from OANDA transaction history.

**Fixed scripts**:
- `tools/slack_daily_summary.py` — JST day boundary + accurate % from `accountBalance` field
- `tools/daily_performance_report.py` — JST day grouping in aggregate + report
- `tools/intraday_pl_update.py` — JST day boundary for "today" query
- `tools/oanda_performance.py` — JST day grouping in daily P&L analysis

**Also noted**: `daily-performance-report` and `intraday-pl-update` tasks are disabled (`enabled: false`).

## 2026-04-15 — v8.4: Comprehensive Market Analysis Upgrade

**Problem**: Trader task was missing critical analysis dimensions — M15 timeframe entirely absent from pipeline, no cross-currency triangulation, no M1 synchrony detection, no H4 lifecycle positioning, no momentum quality analysis, no event positioning analysis. Result: suboptimal vehicle selection (EUR_USD instead of GBP_USD), blind to M15 corrections, unable to detect currency-specific flows.

**Session time**: 10 min → 15 min (20-min cron) to accommodate deeper analysis.

**Data pipeline changes (session_data.py ecosystem)**:
- `refresh_factor_cache.py`: Added M15 to TF_MAP (200 candles). Now M1/M5/M15/H1/H4
- `adaptive_technicals.py`: M15 row displayed. Momentum quality tags: [FRESH]/[MATURE]/[EXHAUSTING]/[REVERSING] on TREND situations
- `session_data.py`: 3 new sections:
  - **CURRENCY PULSE**: Cross-currency triangulation at H4/M15/M1. Per-currency BID/OFFERED/neutral. MTF conflict detection. M1 synchrony (all crosses same direction). Correlation break detection. Best vehicle recommendation
  - **H4 POSITION**: Lifecycle label per pair (EARLY/MID/LATE/EXHAUSTING + BULL/BEAR). StRSI zone, EMA slope acceleration, VWAP deviation
  - **H1 FIB WAVE**: Multi-TF Fib confluence (M5 + H1 levels)
- `chart_snapshot.py`: Regime transition tracking via `logs/regime_history.json`. Shows `[was RANGE]` when regime changes between runs
- `macro_view.py`: (correlation break logic co-located in session_data Currency Pulse)

**SKILL prompt changes (SKILL_trader.md)**:
- **New: Currency Pulse block** — forces cross-currency synthesis BEFORE pair scan. 5 currencies × 3 TFs. MTF conflict, M1 synchrony, correlation breaks, best vehicle, position match check
- **New: Self-check block** — 30-second bias detection. Entry count, pair fixation, cold streak, holding bias
- **Enhanced Market Narrative** — event positioning (asymmetry analysis) + macro chain (per-currency effects)
- **Enhanced Position Management** — C block requires 4 items (was 3): added M15 momentum, M1 currency pulse, H4 position/room
- **Enhanced Conviction Block** — added H4 position (StRSI lifecycle), cross-currency M15 check, event asymmetry
- **Session timing** — 15-min sessions, 10-min minimum (was 8), 13-min normal end (was 9), 17-min hard kill (was 12), 900s stale lock (was 600)

**Files changed**: refresh_factor_cache.py, adaptive_technicals.py, session_data.py, chart_snapshot.py, session_end.py, SKILL_trader.md, CHANGELOG.md, CLAUDE.md

## 2026-04-15 — v8.3c: Range Scalp Scanner — range markets as profit engine

**Problem**: System was optimized for trend/momentum. Range markets (5/7 pairs often in SQUEEZE/RANGE) treated as "wait for breakout" instead of profit opportunities. 10% daily target unreachable without range scalping.

**Key Insight**: Range = bounded risk = size up. BB lower→mid = 70%+ probability. Rotation (BUY low→TP mid→SELL high→TP mid) generates base income. Trend plays become bonus.

**New tool: `tools/range_scalp_scanner.py`**:
- Scans all 7 pairs on M5 + H1 for RANGE regime
- Classifies: CLEAN_RANGE / SEMI_RANGE / FORMING_RANGE / SQUEEZE
- Detects entry signals: BB position × StochRSI × CCI × RSI × wick patterns
- Outputs ready-to-trade plans: entry/TP/SL levels, R:R, sizing, rotation plan
- Spread coverage ratio (range÷spread, must be >5× to trade)
- Range health: touch symmetry, BB width trend, DI balance, squeeze proximity
- MTF confirmation: M5+H1 both range = higher confidence
- Signal strength scoring: BUY/SELL NOW vs WATCH vs MID_ZONE

**Enhanced `chart_snapshot.py`**:
- RANGE regime now includes: BB position (zone), touch symmetry, scalp rotation plan
- New field: `bb_width_trend_pct` (negative = narrowing = squeeze forming)
- New field: `range_scalp` dict with BB levels and zone when RANGE detected

**Updated `docs/TRADER_PROMPT.md`**:
- Added full range scalping section with execution flow, conviction framework, sizing table
- Range scalp = explicit trade type alongside Scalp/Momentum/Swing
- Math: how range rotation reaches 10% daily target

## 2026-04-15 — v8.3b: Quality audit aligned with v8.2/8.3 trader changes

**Problem**: Trader SKILL was overhauled (v8.2: Close/Hold flip, zombie detection; v8.3: theme confidence, candle filter, concentration, rotation) but quality-audit SKILL still referenced old formats. Audit couldn't catch v8.3 violations (sizing without theme confirmation, indicator-only theses, regime mismatches).

**Changes to SKILL_quality-audit.md:**
1. **Position Challenge updated**: Now checks regime at entry vs now (regime mismatch = first-class finding), zombie time, entry type + held time ratio, and candle filter compliance from chart PNGs
2. **v8.3 Compliance Check added**: 5 items checked every cycle:
   - Theme confidence: is sizing consistent with proving/confirmed/late?
   - Top 2 concentration: is margin concentrated (>60%) or diluted?
   - Candle filter: do entries have "Last 5 candles" or indicator-only?
   - Rotation: after TP, was re-entry plan written?
   - Regime consistency: entry regime matches current regime?
3. **Market read section**: Now validates trader's Theme confidence and Top 2 pair selection, proposes alternatives when disagreeing
4. **DANGER criteria updated**: Regime changed + data contradicts + profit_check recommends TP = DANGER

## 2026-04-15 — v8.3: 10%+ daily system — concentration, rotation, candle filter

**Goal**: Consistent 10%+ daily returns. 4/7 did +14,186 (13%) — reverse-engineer and systematize.

**What makes 10%+ days (data-driven)**:
- Top 1 pair = 41-47% of P&L (concentration, not diversification)
- 4/7 EUR_USD: 7 rotations, 500u→5,000u progressive sizing = +5,880 from ONE pair
- Good days: avg size 5,327u, avg hold 140m, WR 84%, worst trade +1,048
- Bad days: avg size 2,993u, avg hold 201m, WR 39%, worst trade -2,273
- Losers cut <30m = avg -354. Losers held >2h = avg -818 (2.3× worse, 75% of ALL losses)

**Changes**:

1. **Theme confidence tracker**: "proving / confirmed / late" in Market Narrative
   - proving (untested thesis) → B-size 2,000u only
   - confirmed (at least 1 TP today) → A/S-size 4,000-6,000u
   - late (6h+ into theme) → reduce, protect gains
   → Forces the 4/7 progressive sizing pattern: start small, prove, scale up

2. **Top 2 pairs concentration**: Market Narrative must name top 2 pairs → 80% of margin. Others = B-size scouting max. Prevents dilution across 4-5 pairs

3. **Candle filter on entries**: Thesis must be a STORY ("sellers made staircase, buyers absorbing at 215.35") not indicator list ("StRSI=0.0 ADX=61"). "Last 5 candles → Buyers defending? YES/NO" — if NO, pass. Same StRSI=0.0 can be a real bounce or a trap; only candle shapes distinguish them

4. **15-minute first confirmation**: "If no movement in my direction by entry+15m → close." Based on data: quick-cut losers cost -354 avg vs slow-cut -818. Pre-commits the exit clock at entry

5. **Rotation mandate**: After every TP, write re-entry plan. Theme confidence upgrades from "proving" → "confirmed" on first TP, unlocking S-sizing. This IS the compound engine that turned +400 into +5,880 on 4/7

6. **profit_check time penalties tightened**: 2h+ losing position → warning (+1 take_signal if UPL<0). 4h+ → +2. 8h+ ZOMBIE → +3

7. **Max loss per trade: 500 JPY**: units × SL_distance capped. Good days have worst trade ~-350. Bad days have -2,000+. The cap prevents bad days from forming

## 2026-04-15 — v8.2b: Kill LIMIT churn (156 placed, 14 filled, 75 cancelled)

**Problem**: April data: 156 LIMITs placed, 14 filled (9% fill rate), 75 cancelled (48%), 128 modified. 53 cancel→re-place cycles on same pair. Each cycle wastes analysis time, log entries, state.md updates. EUR_USD alone: 18 cancel→re-place cycles.

**Root cause**: LIMIT treated as "my current best guess" → re-evaluated and moved every 15-minute session. Non-structural levels (M5 price, StochRSI) change every candle, triggering constant re-placement. SKILL said "A wrong LIMIT costs nothing — cancel it next session" which encouraged over-placement and churn.

**Fixes**:
1. **LIMIT must cite structural level** ("H1 BB lower" / "Fib 38.2%" / "cluster 215.52") — structural levels don't move in 15 minutes
2. **Session review = thesis check only**: "thesis alive? YES → leave / NO → cancel: [reason]". No price re-evaluation
3. **Invalid cancel reasons codified**: "M5 moved", "StochRSI changed", "slightly better level" = NOT reasons to cancel
4. **GTD minimum 4-8 hours** (was 2-4). Short GTDs caused premature expiry → re-placement churn
5. **Max 2 pending LIMITs** — forces quality over quantity. Pick best 2 structural levels, not scatter 5 hoping one fills
6. **Removed "A wrong LIMIT costs nothing"** — it does cost something: analysis time, log pollution, decision fatigue

## 2026-04-15 — v8.2: Kill the always-C hold bias (structural overhaul)

**Problem**: 30-day data shows trader chooses C (hold) on virtually every position, every session. Result: 14 trades held 8h+ = -14,246 JPY (14% WR). Losers held 296m avg vs winners 156m avg (disposition effect). The system generates +53k from 136 normal trades but 24 big losers eat -32.5k.

**Root causes identified (3 structural, not behavioral)**:

1. **profit_check.py was structurally biased toward HOLD**: H1 ADX>25 alignment gave hold_signals += 2. One correlated pair gave +1 more. Total hold_signals = 3 → HALF_TP condition (`take >= 2 AND hold < 3`) was physically unreachable for any trending pair with correlation. Tool always output "All positions HOLD", giving the trader false confidence to write "→ C".

2. **Close/Hold format asked "why hold?" (always answerable)**: Format required "I'm not closing because: ___" — trivially satisfiable for any trending pair ("H1 ADX=46 intact"). The "Default = Take Profit" principle existed as text but the output structure contradicted it.

3. **Closing cost 6× more time than holding**: HOLD = 30 seconds (write 3 lines). CLOSE = 3-4 minutes (preclose_check + close_trade.py + manual log + manual Slack + state.md update). In a 10-minute session with 3 positions, closing one consumed 30-40% of available time. Rational time optimization, wrong outcome.

**Fixes**:

1. **profit_check.py overhaul**:
   - H1 alignment → context display only (no hold_signals). H1 doesn't change for hours — it's background, not a hold reason
   - Cross-pair correlation → context display only (no hold_signals)
   - HALF_TP gate removed: `take >= 2` triggers HALF_TP regardless of hold count
   - New: M5 active momentum (MACD positive + StRSI mid + slope positive) = only real hold signal
   - New: Time-held penalty: 4h+ → take_signals += 1, 8h+ → take_signals += 2 (ZOMBIE warning)

2. **Close/Hold format flipped (Default = Close)**:
   - A (close) shown FIRST with JPY amount — see the number before justifying hold
   - C (hold) requires ALL 3: (1) new info since last session, (2) entry TF chart description, (3) "would I enter NOW?"
   - "nothing changed" → forced to A. "H1 thesis intact" not valid for momentum trades
   - Zombie ratio (held/expected) displayed — ratio > 2.0 triggers warning

3. **close_trade.py upgraded**: `--auto-log --auto-slack` flags handle log entry + Slack notification automatically. Single command reduces close cost from 3-4 min to ~30 sec. Time parity with HOLD removes the structural incentive to avoid closing.

4. **Entry format additions**:
   - "Expected hold → Zombie at: HH:MMZ" — forces writing when the trade dies
   - "Session: NY_PM" triggers explicit penalty block (April WR=25%, avg_size=4,625u — sizing UP during worst session)

**Counterfactual**: Removing 8h+ holds + NY PM long entries + AUD_USD from April → 120 trades, +31,058 JPY (72% WR), vs actual +20,563 JPY. Improvement: +10,495 JPY.

## 2026-04-15 — Fix "B only. Pass" passivity: conviction → action in Tier 2

**Problem**: Trader sees setups, writes entry levels and TPs in Tier 2 scan, then adds "B only. Pass." and doesn't place the LIMIT. 52% margin idle, 0 closes today, while GBP_JPY moved 27.9pip and AUD_JPY 21.2pip during Tokyo. The conviction framework says B = LIMIT at B-size (1,667u), but the format allowed Pass as a B-action.

**Root causes**: (1) Tier 2 format had no action field → B was a dead end. (2) B treated as disqualification when it's a sizing instruction. (3) Lessons written as commands ("Wait for reset") blocked future entries like rules.

**Fixes**:
1. Tier 2 format: `[S/A/B/C] — reason` → `[S/A/B/C] → [action] — reason`. B → LIMIT posted. Only C can pass
2. B→A upgrade path: "check one more lens" on promising Bs. Different Lens that supports = 3× size upgrade
3. Lesson format: commands ("Wait for X") → observations ("X happened, Y was the result"). Next session decides

## 2026-04-15 — Fix daily-performance-report: use script instead of in-session API calls

**Problem**: daily-performance-report SKILL had Claude fetch and aggregate OANDA transactions API in-session. With 888+ trades (thousands of transactions), pagination + 2-min session limit caused data loss. Report showed only 3/18 data (-1,197 JPY) instead of actual cumulative (-4,074 JPY).

**Fix**:
1. Created `tools/daily_performance_report.py` — dedicated script that handles full pagination, aggregates by today/week/all-time, and posts to Slack
2. Updated SKILL.md to just run the script instead of doing API calls in-session

**Principle**: Heavy data processing belongs in scripts, not in Claude session prompts.

## 2026-04-15 — Remove Tokyo "thin liquidity" passivity bias

**Problem**: Trader sits idle during Tokyo saying "wait for London" while 20+ pip moves happen on GBP_JPY and AUD_JPY. Root cause: session_data.py labels ALL of 00-06 UTC as "Tokyo late (thin liquidity)" regardless of actual volatility. SKILL format has "Tokyo thin" as the only option. Trader writes "Session: Tokyo thin" → anchors on "thin" → waits.

**Fix**:
1. **session_data.py**: Removed "(thin liquidity)" from Tokyo label. Now shows "Tokyo" or "Tokyo (pre-London positioning)". Late NY shows "Late NY (rollover zone)" (the actually dangerous session)
2. **SKILL narrative format**: "Tokyo thin / London / NY" → "Tokyo / London / NY / Late NY"
3. **SKILL SL/trail rules**: Replaced time-based triggers ("Tokyo session") with condition-based ("Spread > 1.5× normal"). Judge by measurement, not label
4. **SKILL scan discipline**: Replaced generic "thin market ≠ no entries" with specific Tokyo edge data (+4,997 JPY, +347/trade for Tokyo→London positioning)

**Principle**: Liquidity is a measurement (spread, M5 body size), not a session label. If M5 bodies are 3-5pip, the market is moving — trade it.

## 2026-04-15 — S-Conviction v2: prediction format + follow-up loop (anti-bot refinement)

**Problem with v1**: The 7-pair conviction assessment used FOR/Different lens/AGAINST format = category-checking. "FOR: Direction + Timing + Momentum" can be copy-pasted every cycle without thinking. Also: audit-trader loop was one-way (audit writes → trader reads → no follow-up). No accountability for predictions.

**Fix**:
1. **Audit Section C**: FOR/AGAINST category-checking → prediction format. Each pair now requires: specific chart observation → specific price target + timeframe → "wrong if" condition → conviction. "Price will reach 1.1835 in 1h because band walk + ECB hawkish" forces chart-specific thinking. "Direction + Timing" doesn't.
2. **Audit Follow-up**: New sub-section at top of Section C. Audit checks its own previous predictions vs actual price movement. "Predicted 186.00, actual 185.87, 4/7 correct direction." Anti-bot: facts change every cycle. Pattern awareness: "I keep getting JPY crosses wrong."
3. **Audit Response in state.md**: Trader writes structured responses to audit S/A predictions. Audit reads these next cycle and evaluates whether trader was right. Closes the feedback loop.
4. **Tier 2 examples**: Category names ("H1+M5+CS aligned") → chart-specific ("band walk + GBP strongest CS, missing: Fib 38.2% untested"). Models mimic examples.

## 2026-04-15 — "If nothing by" conviction block field

**Problem**: 41 NY entries held overnight without exit plan, dumped in Tokyo thin liquidity = -14,094 JPY total drag. "If I'm wrong" covered loss scenario but not the "move never comes" orphan scenario.

**Fix**: Added `If nothing by: ___` to conviction block (SKILL_trader.md + risk-management.md). Format forces trader to write exit timeline at entry, naturally embedding session awareness. Not a rule — a thinking prompt.

## 2026-04-15 — Session Dynamics: Tokyo positioning edge discovered

**Analysis**: 500-trade OANDA history analyzed by ENTRY time (not close time). Key findings:
- Tokyo entries are net +4,997 JPY (119t, 56%WR). Previous belief that "Tokyo loses" was caused by measuring CLOSE time, which includes NY overnight losers being dumped during Tokyo morning
- Tokyo entry → London close = avg +347/trade (29t). 7× the system average. Asian range → London breakout positioning
- Momentum trades (30m-2h) are the system's edge across ALL sessions. Scalps (<30m) lose in Tokyo AND London
- Late NY (21-00 UTC) is the system's worst session: -11,898 JPY, 36%WR. GBP_USD alone -9,601 from 8 entries
- NY is highest volume (171 entries) for zero return (-103 JPY)

**Changes**: Added to strategy_memory.md:
- Session dynamics in Confirmed Patterns (entry-time P&L, momentum vs scalp, Tokyo→London positioning)
- NY overnight orphan pattern in Active Observations
- Per-pair: AUD_JPY Tokyo natural home, EUR_USD Tokyo→London play, GBP_USD Late NY death zone

## 2026-04-15 — S-Conviction Discovery Overhaul: narrative assessment replaces scanner gating

**Problem**: S-conviction trades are the system's biggest profit driver, but neither trader nor audit finds them. Root cause: S-conviction discovery was bottlenecked through `s_conviction_scan.py` — a pattern matcher with 6 fixed recipes and binary thresholds (StochRSI ≤0.05 / ≥0.95). Real S-conviction comes from story coherence ("everything points the same way"), not hitting exact indicator values. A strong trend pullback at StochRSI=0.15 IS S-conviction but the scanner doesn't fire. Result: ~0-1 S-setups found per day when 3-5 exist at any given time.

**Fix** (applied "think at the point of output" principle — format forces thinking, not rules):

1. **Quality Audit SKILL** (`docs/SKILL_quality-audit.md`): Replaced Section C "Missed Opportunities" (scanner relay) with "My Best Trades Right Now" — 7-pair conviction assessment. Auditor now writes trade plan + FOR/Different lens/AGAINST + conviction for ALL 7 pairs based on chart reading + data. S-conviction surfaces naturally from story coherence. Scanner becomes supplementary data.

2. **Trader SKILL** (`docs/SKILL_trader.md`):
   - Tier 2 format now includes `| [S/A/B/C] — [reason]` suffix → conviction assessment for every pair
   - Added "Tier 2 → Tier 1 promotion" section: any S/A in Tier 2 MUST get full Tier 1 analysis
   - Audit response changed from "S-scan NOT_HELD" to "Audit Conviction Map" — trader must respond to auditor's S/A ratings with agree/disagree + specific reasoning
   - S-Conviction Recipes section restructured: narrative assessment is PRIMARY path, scanner is SUPPLEMENT with accuracy tiers

3. **s_conviction_scan.py** (`tools/s_conviction_scan.py`):
   - Disabled Squeeze-S recipe (0/4 accuracy — all signals wrong direction)
   - Added accuracy tiers to output: `[proven 3/3]`, `[proven 4/5]`, `[noisy 3/12]`, `[tracking]`
   - Updated dedup regex to handle new format

4. **CLAUDE.md**: Updated quality-audit description to mention 7-pair conviction assessment

## 2026-04-15 — Trade type awareness: time_held + entry thesis recall in evaluation

**Problem**: GBP_USD 8000u S-Momentum entry (PPI miss) held 5h40m, cut at -2,583 JPY. Momentum thesis died within 30min, but trader checked H1 (ADX=55 BULL = "thesis intact") instead of M5 (entry timeframe). Same root cause: trail 8pip on Momentum = too tight (profit side), hold 6h on Momentum = too long (loss side). Both from managing trades on the wrong timeframe.

**Changes**:
- `profit_check.py`: Added `time_held` to output (calculated from OANDA openTime). Every position now shows "held: Xh Ym" — pure data, no rules.
- `SKILL_trader.md`: Added `Entry type`, `Entry thesis was`, `Held vs expected` to the per-position evaluation block. Added `Is my entry thesis still why I'm here?` question. Forces trader to confront thesis drift (Momentum held as Swing) and timeframe mismatch.
- `risk-management.md`: Loss management step 1 changed from "Has the H1 structure changed?" to "Has the structure changed on the timeframe you entered on?" with 4/14 lesson. Removes H1 as hardcoded evaluation timeframe.

**Design**: No rules, no time limits, no automatic actions. Format forces the trader to write their own assessment against their own entry plan. "Held: 5h40m vs expected 30min-2h" is data that makes thesis drift self-evident.

**Files**: `tools/profit_check.py`, `docs/SKILL_trader.md`, `.claude/rules/risk-management.md`.

## 2026-04-15 — Rollover window: ban ALL actions including manual closes

**Problem**: 4/14 AUD_JPY 8000u LONG closed manually at Sp=10.8pip during rollover, citing "thesis invalidation." Price returned to entry (113.168) within hours. -856 JPY self-inflicted. Rollover guard removed SLs correctly, but trader overrode with manual close — defeating the entire purpose of the guard.

**Fix**:
- `risk-management.md`: Added "Rollover Window = NO MANUAL CLOSES" section. From `rollover_guard.py remove` until `restore`, only permitted actions are: wait, cancel pending LIMITs. No closes, no entries, no SL modifications.
- `protection_check.py`: Added ⛔ NO MANUAL CLOSES / NO NEW ENTRIES / WAIT warnings to ROLLOVER WINDOW output
- `risk-management.md`: Added failure pattern entry for "Manual close during rollover"

## 2026-04-15 — Fix state.md UTC date bug causing quality-audit false alarms

**Problem**: Trader Opus session was writing JST date with UTC label in state.md (e.g., "2026-04-15 16:32 UTC" when actual UTC was 2026-04-14 16:32). During UTC 15:00-23:59 (JST 0:00-8:59), the date is 1 day ahead. Quality-audit Sonnet reads this as a future timestamp → "15 hours stale" false alarm.

**Fix**:
- `session_data.py`: Added copy-paste ready `state.md timestamp:` line with correct UTC date
- `SKILL_trader.md`: Added CRITICAL note to copy UTC timestamp from session_data.py output, not compute manually

## 2026-04-15 — De-botify: Pullback Quality verdict → data panel + forced thinking format

**Problem**: 4/14 Pullback Quality Check added NOISE/SQUEEZE/DISTRIBUTION scoring with rule tables ("if NOISE → ATR×1.5"). This is the same bot pattern as profit_check's verdict-following: tool classifies → rule table maps → trader follows. No thinking.

**Broader pattern identified**: profit_check TAKE_PROFIT → action, trail width table → ATR×N lookup, regime+chart → TP target. All "label → rule → action" with no trader judgment.

**Change**:
- `profit_check.py`: Removed scoring/verdict from `assess_pullback_quality()`. Now outputs raw data panel (6 panels: H1 trend health, M5 vol/volatility, candle character, structure, ROC, cross-pair). No NOISE/SQUEEZE/DISTRIBUTION labels. No scores.
- `SKILL_trader.md`: Replaced rule table with indicator knowledge guide (what each indicator MEANS for buyers/sellers) + required output format: "I see / This tells me / So I'm doing". Trader must write own interpretation — can't copy a verdict.
- `risk-management.md`: Removed STEP 3b (was "S+NOISE → hold even if TAKE_PROFIT"). No more verdict-based override rules.

**Design principle applied**: "Don't tell Claude what to think. Shape the format so thinking is required to produce the output." A bot can follow "NOISE → ATR×1.5". A bot cannot fill in "I see [observations] / This tells me [interpretation]" without reading the data.

**Estimated impact**: +1,500-2,000 JPY/day on S-conviction trades (based on 4/14 GBP_USD case: trail 8pip captured 13.8pip; data-reading approach would have held for 46+ pip).

**Files**: `tools/profit_check.py`, `docs/SKILL_trader.md`, `.claude/rules/risk-management.md`.

## 2026-04-14 — Pullback Quality Check in profit_check.py (superseded by 4/15 de-botify)

**Root cause**: S-conviction trades captured 12-14pip (trail ATR×0.6) vs 4/7 best day 25-30pip (trail ATR×1.5). Same conviction, half the profit. Trader used StRSI alone to judge pullbacks, ignoring 12 other relevant indicators in the cache.

**Change**: Added `assess_pullback_quality()` to `profit_check.py`. Originally output NOISE/SQUEEZE/DISTRIBUTION verdict with scoring. Superseded same day — verdict removed, converted to raw data panel (see 4/15 entry above).

**Files**: `tools/profit_check.py`, `docs/SKILL_trader.md`, `.claude/rules/risk-management.md`, `collab_trade/strategy_memory.md`.

## 2026-04-14 — Daily summary: dedup guard + show P&L as % of balance

`slack_daily_summary.py`: Added dedup guard — writes `logs/daily_summary_last.txt` with the posted date, skips if already posted for that date. Prevents duplicate posts when task is re-triggered. `--date` manual runs bypass the guard.

## 2026-04-14 — Daily summary: show P&L as % of balance

`slack_daily_summary.py`: Added percentage change (realized P&L / previous day balance) to the daily P&L line. Example: `+5,871JPY (+4.89%)`.

## 2026-04-14 — BUGFIX: rollover guard restoring SLs while spreads still wide

**Problem**: `protection_check.py` determined rollover end purely by time (15 min after 5 PM ET). In reality, spreads can stay 2-3x+ wider for 30+ minutes after rollover. This caused:
1. `protection_check.py` declaring "rollover passed" and suggesting SL restore
2. `rollover_guard.py restore` restoring SLs into still-wide spreads
3. Restored SLs getting immediately hunted by spread spikes → unnecessary losses

**Fix — spread-aware rollover detection**:
- `protection_check.py`: Added `_check_spreads_wide()` — fetches live pricing from OANDA, checks if any pair's spread exceeds 2x normal. After the initial 20-min pre-rollover window, the post-rollover window now checks actual spreads: if spreads > 2x normal, `is_rollover` stays True regardless of time elapsed (up to 60 min). Time-based window extended from 15 → 30 min as baseline.
- `rollover_guard.py restore`: Now checks live spreads before restoring. If any pair's spread is still wide, restore is BLOCKED with a clear message. Added `--force` flag to bypass the check when needed.

**Result**: SLs stay removed until spreads actually normalize, not just until the clock says so.

## 2026-04-13 — BUGFIX: market_state.py type hint Python 3.9 incompatibility

**Problem**: `tools/market_state.py` used Python 3.10+ union type syntax (`datetime | None`) in function signatures. System runs `python3` = Python 3.9.6, causing `TypeError: unsupported operand type(s) for |` on import. This silently broke `profit_check.py` and `protection_check.py` at session start.

**Fix**: Replaced `datetime | None` with bare `= None` (untyped default). Two functions patched: `get_market_state()` and `is_tradeable()`. Both Python 3.9 and 3.10 environments now work.

## 2026-04-11 — NEW: market_state.py — prevent panic trades during market close/maintenance

**Problem**: profit_check.py and quality_audit.py had no awareness of market state (weekend, daily OANDA maintenance). During these periods, spreads widen 10-20x but positions are fine — the wide spread is illiquidity, not danger. Tools would recommend TAKE_PROFIT based on distorted bid/ask, potentially causing the trader to panic-close and eat massive spread costs (10-19 pip loss from spread alone).

**Design principle**: Detection is TIME-BASED ONLY, never spread-based. Wide spreads from news events or intervention are trading opportunities, not illiquidity. Blocking on spread would miss the best trades.

**New module**: `tools/market_state.py` — shared market tradeability detection:
- `CLOSED` — Weekend (Fri 5 PM ET → Sun 5 PM ET). No orders.
- `ROLLOVER` — Daily OANDA maintenance (5 PM ET ±20 min). No orders.
- `OPEN` — All other times, including volatile/wide-spread periods.

**Changes to profit_check.py**: When CLOSED/ROLLOVER, all TP recommendations suppressed. Positions listed for reference only with `HOLD(MARKET CLOSED)` tag. Prevents panic market orders during untradeable conditions.

**Changes to quality_audit.py**: When CLOSED/ROLLOVER:
- Report header shows `⛔ MARKET CLOSED — All findings are INFORMATIONAL ONLY`
- Exit code forced to 0 (no Slack alert triggered)
- Findings still recorded for reference but won't cause panic actions

**What stays the same**: protection_check.py already handles rollover well (time-based SL removal). No changes needed there. Rollover guard (rollover_guard.py remove/restore) continues to work as before.

## 2026-04-11 — FIX: slack_daily_summary.py day boundary alignment

**Problem**: `slack_daily_summary.py` used local time `datetime.now() - timedelta(days=1)` and parsed `live_trade_log.txt` by date string match. Trader task and `intraday_pl_update.py` use UTC 00:00 as day boundary and OANDA transactions API. The mismatch caused daily summary to show +0 JPY when trades existed.

**Fix**: Rewrote `slack_daily_summary.py` to use OANDA transactions API with UTC day boundary, matching trader and intraday_pl_update. Removed log file parsing entirely. Entry/close counts now come from OANDA ORDER_FILL transactions.

## 2026-04-10 — NEW: verify_user_calls.py + daily-review integration

User market calls ("反発始まる", "あがるよ" etc.) were recorded but never verified. outcome stayed NULL forever, making pretrade_check accuracy stats unreliable.

**New script**: `tools/verify_user_calls.py` — fetches OANDA price at call time and 4h later, compares with predicted direction, marks correct/incorrect/neutral in DB. Also backfills price_at_call, price_after_30m, price_after_1h.

**Integration**: Added as Bash② in daily-review Step 1. Runs automatically every daily-review cycle.

**Initial backfill result**: 6/7 calls verified — 83% accuracy (5 correct, 1 incorrect). The "反発始まる" call was actually correct (+44.4pip in 4h), but stale (14 days old) and should not influence current decisions.

## 2026-04-10 — FIX: pretrade_check user call ghost data poisoning decisions

**Problem**: User call "反発始まる" (3/27, 14 days ago) was blocking USD_JPY SHORT entries. The call was never verified (outcome=NULL, price_at_call=NULL) but pretrade_check showed "user accuracy: 100%" (actually from a different call, n=1). Trader tried SHORT 9+ times today — all blocked by ghost data. Meanwhile all TFs showed DI- dominant.

**Root cause**: 3 structural flaws in pretrade_check.py:
1. `latest_user_call()` had no time limit — 2-week-old calls used as "latest"
2. Unverified calls (outcome=NULL) displayed alongside verified accuracy stats
3. Risk score +2 applied even for unverified stale calls

**Fix**:
1. `latest_user_call()` now takes `max_age_days=3` — calls older than 3 days are ignored (market conditions change)
2. Verified calls: show `(verified 75%, n=4)` — sample size visible
3. Unverified calls: show `(unverified — info only, no score impact)` — no risk_score added

## 2026-04-10 — FIX: slack_post.py guards against garbage replies and wrong channel

**Problem**: Trader session replied "dummy" to user's "状況は？" in #qr-commands. Second message "状況教えて" reply was sent to #qr-trades with `--reply-to` flag, marking it as replied in dedup. User got no proper response for 43 minutes.

**Fix**: Two guards added to `tools/slack_post.py`:
1. `--reply-to` forces channel to #qr-commands (C0APAELAQDN) — user replies can never go to wrong channel
2. Trivially short/garbage replies ("dummy", "test", etc.) are blocked with exit code 1

**Files changed**: `tools/slack_post.py`

## 2026-04-10 — FIX: Phantom margin from pending LIMITs blocking market orders

**Problem**: Trader calculated "worst case margin = 82%" by including all pending LIMITs as if they were filled positions. OANDA pending LIMITs use ZERO margin until fill. The trader was blocking new market orders based on phantom margin from orders that weren't even close to filling (20-50pip away). Evidence from 4/9 log: "margin=82.1%" with 4 unfilled LIMITs, "margin_freed=17080JPY" by cancelling an unfilled LIMIT (freeing 0 from 0).

**Fix**: All margin gates across 3 files now explicitly state: "Pending LIMITs use 0 margin. Check ACTUAL margin (open positions only)." New sequence: if market opportunity appears → cancel competing LIMITs → market order → re-place gap coverage if margin allows.

**Files changed**: `docs/SKILL_trader.md` (margin gate + conviction blocks + execution receipt), `.claude/rules/risk-management.md` (pre-entry check + failure pattern), `collab_trade/strategy_memory.md` (負けパターン table)

## 2026-04-10 — MAJOR: Kill LIMIT carousel, market order as primary weapon

**Problem**: Trader placed 67 LIMIT/cancel/modify actions over 4/9-4/10 but only 15 actual entries. Entry ratio collapsed from 0.69 (4/7, best day) to 0.22. LIMITs placed at structural levels 20-50pip from market → never fill → cancel → replace → repeat. Analysis was deep but produced plans, not trades. SKILL.md's "0% margin blocker" + "LIMIT costs nothing" framing incentivized distant LIMIT placement as a proxy for trading.

**Data**: 4/7 (+11,014 JPY, 10.5%): 21 market orders, 3 limit fills. 4/10 (-1,027 JPY): 3 market orders, 67 LIMIT churn.

**Root cause**: 5 conflicting SKILL.md incentives created LIMIT carousel: (1) "0% margin = bad" → forces LIMIT placement, (2) "LIMIT costs nothing" → enables distant LIMITs, (3) "structural wick-touch levels" → levels far from market, (4) market order restricted to "M5 at extreme NOW", (5) 10+ field conviction block → heavy per-entry overhead.

**Changes (SKILL_trader.md)**:
1. Market order is now the default for TREND regime. LIMIT for RANGE/SQUEEZE/events
2. Added "Anti-LIMIT-carousel rule" — 2 sessions unfilled = market order or abandon
3. Added "LIMIT fillability check" — will price reach this in GTD window?
4. "0% margin blocker" → "0 market orders + 0 positions blocker" — distant LIMITs don't count
5. Added "Quick conviction" format (3 fields) for follow-up entries — breaks the 60sec/entry bottleneck
6. 7-pair scan: "Analyze AND ACT" per pair, not analyze-all-then-act
7. Time allocation: execute during analysis (2-6 min), not after (5-8 min)
8. Held position block: 7 fields → 3 lines
9. Capital Deployment → "Execution Receipt" with market order count
10. Idle margin section: "Market orders FIRST, LIMITs for gap coverage"

**Changes (strategy_memory.md)**:
1. Added "MONEY MAKERS" table at top — 6 highest-edge patterns with conditions, size, expected P&L
2. 負けパターン: paragraph format → compact table. Added warning: "these teach HOW to enter, not NOT to enter"
3. Added "LIMITカルーセル" as new 負けパターン

**Files changed**: `docs/SKILL_trader.md`, `collab_trade/strategy_memory.md`, `docs/CHANGELOG.md`

## 2026-04-10 — Format redesign: examples over rules, embed TP into chart line

**Problem**: Previous format had 9 required lines per Tier 1 pair. Model wrote 3, skipped 6. The critical lines (TP, If ranging, Supports, Warns) were the ones skipped. Band walk TP extension (ATR×2.0+) never happened despite the rule existing. Range both-sides LIMIT never placed despite the instruction existing. R:R today = 0.40.

**Root cause**: TP and range were separate lines that could be skipped. Rules say "do X" but the model writes its own condensed format. The prompt had 9 lines but the model's habit was 3 lines.

**Design change**: Examples over rules. Embed critical behaviors INTO lines the model already writes.
1. **Tier 1**: 9 lines → 4 lines. Structure in header (TREND/RANGE/SQUEEZE). "Chart tells me → [band walk → TP at ATR×2.0]" in one thought. RANGE format inherently has both sides. Three filled-in examples the model mimics.
2. **Tier 2**: Structure-specific one-liners with examples. RANGE line format has BUY+SELL in the same line.
3. **Market Narrative**: Removed redundant "Each pair's story" (7 pairs already covered in scan) and "My best setup" (ranking emerges from Tier 1 selection).
4. **Capital Deployment**: 9-field form → 4-line receipt. Lists what was ACTUALLY placed with order IDs, not what was planned.

**Net result**: -13 lines. Prompt got shorter while embedding all behavioral changes.

**Files changed**: `docs/SKILL_trader.md`

## 2026-04-10 — Range LIMIT both sides + TP line + Tier 2 range format

**Gap fixes after verifying state.md adoption**:
1. **Tier 1 TP line**: Added mandatory TP line tied to structure — band walk→ATR×2.0-3.0, deceleration→ATR×1.0-1.5, range→opposite band, squeeze→first structural level. Forces TP decision at scan time, not exit time.
2. **Tier 2 range mandatory 2nd line**: When Regime=RANGE, second line with BUY @___ + SELL @___ is required. One side only = directional bet, not range trade.
3. **LIMIT section range guidance**: Explicit "RANGE = LIMIT LONG at lower band + SHORT at upper band, always both" with AUD_JPY example. OANDA hedge = zero extra margin.

**Files changed**: `docs/SKILL_trader.md`

## 2026-04-10 — Chart+indicators+narrative integration for 10% daily target

**Problem**: R:R=0.57. Winners average +302 JPY (cut too early at ATR×1.0), losers average -534 JPY (held too long or catastrophic). Best day (+11,014) held winners through ATR×1.0 because chart showed band walk. System has charts, indicators, and narrative but they operate independently.

**Root cause**: profit_check triggers TP at ATR×1.0 without seeing the chart. Chart shows "bodies expanding, band walk, no counter-wicks" = hold signal, but profit_check says "TAKE_PROFIT." The chart was right on 4/7's best trades (+3,366, +2,200, +1,876).

**Changes to SKILL_trader.md**:
1. **Regime-based TP**: TREND+band walk = hold to ATR×2.0-3.0. TREND+deceleration = half TP. RANGE = opposite band. TRANSITION = full TP immediately. Chart determines exit, not ATR formula alone.
2. **Loss cap**: Max 2% of NAV per trade (~2,270 JPY). Prevents -3,500 single-trade disasters (3/30).
3. **Chart-informed hold decisions**: Close-or-Hold block now requires chart PNG description, not just indicator values. "Bodies expanding, hugging BB upper" is valid. "ADX=45" is not.
4. **Pair edge priority**: EUR_USD (+8,812) and GBP_USD (+1,880) get S-size first. AUD_USD/EUR_JPY (negative edge) need exceptional chart confirmation.
5. **S-Type TP table**: Added "Chart says hold" / "Chart says exit" columns. Chart overrides ATR formula when continuation is visible.

**Files changed**: `docs/SKILL_trader.md`

## 2026-04-10 — Trader reads chart PNGs + daily 10% NAV target

**Chart reading**: Trader now reads the 14 chart PNGs (7 pairs × M5 + H1 for held pairs) that quality-audit generates every 30 min. No regeneration — just Read the existing files. Two independent visual reads of the same market (trader's eyes + auditor's text summary). Added to Bash② session start flow as parallel Read batches.

**Daily target**: Changed from "+25% per week (~5%/day)" to "+10% of day-start NAV per day (minimum 5%)". Day starts at 0:00 UTC (9:00 JST). Day-start NAV captured in state.md Action Tracking section (first session after 0:00 UTC). Every session tracks progress vs target: behind → hunt harder, exceeded → protect gains.

**Files changed**: `docs/SKILL_trader.md`

## 2026-04-10 — Structure-first trading: fix range weakness through output format redesign

**Problem**: 21-day data: -8,958 JPY, 441 trades, R:R=0.57. This week: 85 trades, only 5 SHORT. Trader holds trend positions through regime transitions (TREND→RANGE), giving back profits. S-scan has 6 recipes — ALL require trending conditions (ADX≥20-30). When range is detected, no recipe fires, no entry is generated, but existing LONGs are held. Result: 2h+ holds = -1,949 JPY (57% WR but large losses).

**Root cause**: Not missing recipes — missing structural narrative. The execution flow was S-scan→action (bot-like), not chart structure→action (pro-like). Claude could see ranges but had no output format to ACT on them. "My best RANGE trade" was written and forgotten.

**Fix — 6 output format changes in SKILL_trader.md** (no new rules, no new recipes):
1. **Market Narrative**: "My best TREND/RANGE/SQUEEZE" → "Each pair's story" (structure-first 7-pair description) + "My best setup" (regime-agnostic)
2. **Regime table**: RANGE sizing from "Half (B)" → "Conviction-based (clear box 3+ bounces = A)"
3. **Tier 1 scan**: "LONG case / SHORT case" → "Structure → If I had no position → Supports/Warns" (removes anchoring + direction bias)
4. **Close-or-Hold block**: Added "Regime at entry → Regime now" line (makes trend→range transition visible, forces honest hold justification)
5. **Capital Deployment**: "#1 LONG / #1 SHORT" → "#1 setup / #2 setup + Ranging pairs (LIMIT both sides)" (structure determines format, not direction)
6. **Decision flow STEP 1**: Added regime transition check as first evaluation step

**Design principle**: Don't add rules ("if RANGE then do X"). Change the format so range thinking is required to produce the output. A bot can follow a rule. A bot cannot fill in "Structure: RANGE 1.1680-1.1720" without forming a range trade plan.

**Files changed**: `docs/SKILL_trader.md`

## 2026-04-10 — Move chart reading from trader to quality-audit (auditor = trader's eyes)

**Problem**: chart_snapshot.py generates 14 PNGs + regime detection, but running it inside the trader's 10-minute session wastes time (15s generation + 14 Read tool calls + massive image token cost). The trader has limited context budget. Meanwhile, the quality-audit (Sonnet, 30-min intervals) already runs profit_check + fib_wave + protection_check and writes persistent analysis to quality_audit.md.

**Change**: Auditor now generates charts, reads them visually (multimodal), and writes Regime Map + Visual Chart Read + Range Opportunities to quality_audit.md. Trader reads this as text (cheap) instead of generating/reading images (expensive).

**Files changed**:
- `docs/SKILL_quality-audit.md`: Added Bash D (chart_snapshot.py --all), Step 1b (visual chart reading with Read tool), Section E (Regime Map table + Range Opportunities with actionable buy/sell levels)
- `docs/SKILL_trader.md`: Removed Bash②c (chart_snapshot.py). Regime data now comes from quality_audit.md. Kept regime strategy table for reference
- `CLAUDE.md`: Updated quality-audit role description, Self-Improvement Loop diagram, chart_snapshot.py script table entry

## 2026-04-10 — chart_snapshot.py: Visual charts + regime detection (Trend/Range/Squeeze)

**Problem**: The trader has never actually seen a chart. It processes indicator numbers (ADX=43, StochRSI=0.0) and infers chart shape from math — but a pro trader reads visual patterns. This blindness causes: (1) can't detect ranges → enters LONG at range top, (2) can't see momentum exhaustion visually, (3) can't distinguish squeeze from range from trend visually.

**Key insight from performance analysis**: The system is TREND-only. When ADX>35 and DI+ dominates (like 4/7: +14,348 JPY in 14h), it wins. When the market is ranging or transitioning, it forces directional trades and loses. 7 pairs × 2 regimes = 14 potential opportunity types. Currently only ~7 (trend on each pair) are traded.

**Changes**:
- `tools/chart_snapshot.py`: **New script**. Fetches OANDA candle data → generates candlestick PNG with BB, EMA12/20, Keltner Channel overlay + position entry lines. Detects regime: TREND-BULL/BEAR, RANGE, SQUEEZE, MILD. Outputs trade approach per regime. Supports `--all` (7 pairs × M5+H1 = 14 charts) and `--regime-only`. Claude reads PNG via Read tool for actual visual chart perception.
- `tools/oanda_performance.py`: **New script** (see below).
- `logs/charts/`: New directory for chart PNG output.

## 2026-04-10 — oanda_performance.py: OANDA API-based performance analysis (replaces log-grep)

**Problem**: Performance analysis using `grep` on `live_trade_log.txt` produces wildly inaccurate numbers. The log file contains 6-second monitoring loops (UPL= lines), inconsistent formats across dates, and non-trade entries that match P/L regex patterns. An agent analysis reported "+632 JPY breakeven" when the actual OANDA-verified total was -15,550 JPY.

**Root cause**: `trade_performance.py` parses `live_trade_log.txt` with regex. The log was never designed for machine parsing — it's a human-readable chronological record. Any regex approach is fragile against format changes and monitoring line contamination.

**Changes**:
- `tools/oanda_performance.py`: **New script**. Queries OANDA Transaction API directly for ORDER_FILL events. Computes daily P&L, win rate, avg win/loss, R:R ratio, best N-hour windows (streak detection), per-pair breakdown, best/worst trades. Supports `--days N`, `--date YYYY-MM-DD`, `--streak N`, `--json`. Smoke-tested in both `python3` and `.venv/bin/python`.
- **Rule**: Any performance analysis MUST use `oanda_performance.py` (API source of truth), NOT grep on live_trade_log.txt.

## 2026-04-10 — Rollover Guard: auto-remove SL before daily OANDA maintenance

**Problem**: OANDA daily rollover at 5 PM ET (21:00 UTC summer / 22:00 UTC winter) causes spread spikes every day. Any SL/Trailing set at normal levels gets hunted during this 10-15 min window. Same structure as the 4/3 Good Friday -984 JPY loss, but happening daily.

**Changes**:
- `tools/protection_check.py`: `detect_thin_market()` now detects rollover approach (20 min before through 15 min after). Includes US DST calculation. Returns rollover-specific flag. Output shows `ROLLOVER WINDOW` warning with actionable command
- `tools/rollover_guard.py`: **New script**. `remove` strips all SL/Trailing from open trades and saves state to `logs/rollover_guard_state.json`. `restore` re-applies saved SL/Trailing. `status` shows current guard state
- `.claude/rules/risk-management.md`: Added "Daily Rollover SL Guard" section with the remove→wait→restore flow
- `CLAUDE.md`: Added rollover_guard.py to scripts table

## 2026-04-10 — Quality Audit v3: Sonnet becomes independent market analyst

**Trigger**: User observed (1) audit results weren't being used by trader, (2) audit accuracy questionable, (3) Sonnet acting as classification bot (REPORT/NOISE) not a thinking analyst.

**Root causes**:
1. `session_data.py:488` bug: `"### "` check never matched `"## "` headers → audit invisible to trader (fixed in earlier commit)
2. Momentum-S recipe too loose: CS gap 0.5 fired 5-6 pairs simultaneously (fixed in earlier commit)
3. Sonnet had no independent data: never ran profit_check/fib_wave/protection_check, never read state.md or strategy_memory.md, reasoning was ephemeral (never saved)

**Changes**:
- **SKILL.md complete rewrite**: Sonnet now runs 3 parallel tool calls (quality_audit.py + profit_check+protection_check + fib_wave), reads 5 context files (quality_audit.md, state.md, strategy_memory.md, news_digest.md, audit_history.jsonl), then writes structured analysis
- **Output format forces thinking**: "Trader says: ___" requires quoting state.md. "Against this trade NOW: 3 data points" requires citing tools. "If wrong → specific price" requires scenario construction. Cannot copy-paste from prior sessions
- **Persistent Auditor's View**: Analysis written to quality_audit.md (appended below script facts). Trader reads it via session_data.py next session
- **Pattern Alert section**: Cross-references current trader behavior against strategy_memory.md failure patterns
- **Slack only on DANGER**: No more REPORT/NOISE noise. Slack fires only when data actively contradicts a position or failure pattern matched
- **maxTurns 15→25**: More headroom for deeper analysis (~3-4 min sessions)
- **CLAUDE.md**: Updated quality-audit row in scheduled tasks table
- **docs/SKILL_quality-audit.md**: Reference copy synced

## 2026-04-10 — Audit→Trader feedback loop: 3 fixes

**Trigger**: User observed audit results weren't being used by trader, and audit accuracy was questionable.

**Root causes found**:
1. **session_data.py line 488**: Checked `"### "` to detect audit findings, but quality_audit.md uses `"## "` headers. Condition **never matched** → audit findings were invisible to trader in session_data output
2. **Momentum-S recipe (s_conviction_scan.py)**: CS gap threshold of 0.5 was too low. During macro themes, 5-6 pairs fired Momentum-S simultaneously — describing the regime, not identifying opportunities
3. **No outcome tracking**: audit_history.jsonl recorded detection prices but never checked if entering would have been profitable

**Changes**:
- **`session_data.py`**: Fixed `has_issues = "### "` → `has_issues = "## " in text and "CLEAN" not in text`. Audit findings now visible to trader
- **`s_conviction_scan.py` Recipe 4 (Momentum-S)**: Tightened: CS gap 0.5→0.8, added H1 ADX≥20 requirement, added M5 StochRSI momentum zone filter. Before: 5-6 simultaneous triggers. After: fires only on genuine momentum setups
- **`daily_review.py`**: New `analyze_s_scan_outcomes()` function. Reads audit_history.jsonl, correlates with OANDA closed trades, checks direction accuracy via current prices. Outputs per-recipe accuracy summary (e.g., "Momentum-S: 83%, Structural-S: 57%")

## 2026-04-10 — quality_audit.py: detect manual (user-entered) positions

**Trigger**: User entered USD_JPY SHORT via OANDA directly. Trader session adopted it as its own in state.md. Quality audit showed it as "ALREADY_HELD" but never flagged that it had no trade log entry, no pretrade_check, no Slack notification. Invisible to the entire audit pipeline.

**Root cause**: quality_audit.py only checked OANDA openTrades. Never cross-referenced with live_trade_log.txt to determine origin.

**Changes**:
- **`load_logged_trade_ids()`**: New function that extracts all trade IDs from ENTRY/LIMIT_FILL lines in live_trade_log.txt
- **`gather_position_facts()`**: Now cross-references each OANDA trade ID against logged IDs. Unmatched = `is_manual: true`
- **Markdown report**: New `⚠ Manual Positions` section at top. `[MANUAL]` tag on position lines
- **JSON report**: `trade_id` and `is_manual` fields added to each position
- **FINDINGS summary**: `manual:N` appears first in stdout when manual positions detected
- **has_findings**: Manual positions now trigger FINDINGS exit code (auditor will report them)

## 2026-04-10 — session_end.py: structural enforcement of session duration

**Trigger**: Session claimed "18:21–18:36 UTC" (15 min) but file timestamps proved 18:21–18:28 (7 min). End time fabricated. Root cause: model bypassed the Next Cycle Bash time check by running `ingest.py` + `rm logs/.trader_lock` directly, then writing "SESSION_END. LOCK_RELEASED." as text. Bash-only time guards cannot prevent this — the model can see and copy the cleanup commands.

**Root cause**: Time enforcement was in a bash one-liner that the model could bypass. Lock cleanup commands (`rm`, `ingest.py`) were visible in SKILL.md, allowing the model to run them directly. Also: cron was `*` (every minute) instead of `*/15`.

**Changes**:
- **New `tools/session_end.py`**: Python script that encapsulates ALL session-end logic (time check, trade_performance, ingest, lock release). Model cannot run ingest or remove lock without going through this script. Time check is enforced in code, not in a bash condition the model can skip
  - `< 480s (8 min)` → prints TOO_EARLY, exits 1
  - `>= 480s` → runs trade_performance + ingest + lock release, prints real timestamps
- **Next Cycle Bash simplified**: `session_end.py || mid_session_check.py`. If session_end.py rejects (exit 1), falls through to mid_session_check. No cleanup commands visible in the bash
- **SKILL.md CRITICAL rule added**: "session_end.py is the ONLY way to release lock and run ingest. Do NOT write SESSION_END or LOCK_RELEASED as text — those words must come from session_end.py output"
- **Cron expression**: `* * * * 1-6` → `*/15 * * * 1-6` (fixed every-minute bug)

## 2026-04-10 — Session timing overhaul: 8min/2min-cron → 10min/15min-cron

**Trigger**: Data analysis of 3 weeks of trades (3/20-4/8) showed:
- Most profitable bucket = 1-4h hold (65% WR, +200 JPY avg). <5min scalps = negative avg P&L
- Winners held 127-334min avg vs losers 13-131min. Patience = profit
- S-candidates missed due to shallow analysis (audit finding), not cron frequency
- ~5 of 7 sessions/hour were "profit_check → HOLD → nothing changed" (wasted Opus time)

**Changes**:
- **schedule.json**: `*/2 * * * *` → `*/15 * * * *` (15-min cron)
- **SKILL.md session length**: 8min → 10min (+2min for deeper 7-pair scan, fib_wave --all, Different lens)
- **Zombie reaper**: kill threshold 10min → 14min (session + buffer)
- **Lock staleness**: 480s → 600s
- **Hard kill timeout**: sleep 900 → sleep 720 (12min)
- **SESSION_END trigger**: 420s (7min) → 540s (9min)
- **Time allocation**: 7+1 → 9+1 (deeper scan window: 2-5min instead of 2-4min)

**Impact**: Opus usage 56min/hr → 24min/hr (57% cost reduction). Worst-case reaction 10min → 25min (covered by TP/SL/trailing protection orders). Structurally eliminates <5min negative-EV scalps.

## 2026-04-10 — Force multi-angle market reading: chart shape + narrative + cross-pair into output format

**Trigger**: Audit showed trader reads NUMBERS not CHARTS. 96% of entry reasons cite indicators, 2% cite news. M5 price action data generated but never referenced. Narrative evolution (news_flow_log) never cited. Cross-pair validation absent.

**Root cause**: Output format ALLOWED filling with numbers. "Price action: [NOT indicators]" was written as "M5: neutral (RSI=48, ADX=24)". "3 questions (plain words)" section was separate from state.md → skipped entirely.

**Changes (SKILL_trader.md)**:
- **Market Narrative**: Added "vs last session: ___ changed" (forces reading news evolution), "M5 verdict: buyers/sellers × accel/exhaust" (chart reading embedded), "My best LONG: ___ / My best SHORT: ___" (both directions before analysis)
- **Tier 1 block**: "Price action" → "Chart: Last 5 M5 candles — bodies ___. Wicks ___. Momentum ___" + "Why moving: [cite news] — currency-wide or pair-specific? [checked: ___ pair]" (forces narrative + cross-pair)
- **Tier 2 block**: Added "M5 candles=[shape] momentum=[accel/exhaust/revers]" — chart shape not indicators
- **state.md template**: Added Market Narrative as first section (was missing)
- **Removed**: Old "3 questions" section (merged into Market Narrative M5 verdict line)

**Principle**: "Think at the Point of Output." Can't fill "bodies shrinking, lower wicks expanding" with RSI=48.

## 2026-04-10 — Fix structural SHORT blindness: pretrade_check wave classification + SKILL output format

**Trigger**: 4/8-4/10: 13+ consecutive LONG entries, 0 SHORTs. USD_JPY SHORT signal identified and analyzed correctly in Slack but never traded — price fell 100+pip.

**Root cause (4 layers)**:
1. **Wave classification**: H4+M5 aligned (H1 transitioning) classified as "small wave" → score capped low
2. **Mid-wave scoring**: No H4 bonus when H4 supports direction → +2 instead of +3
3. **WR hard cap**: All-time WR=33% (biased bullish-period sample) → grade hard-capped at B. Contradicts recording.md ("you make the call") and 4/9 feedback ("stats are regime-dependent")
4. **SKILL output format**: Tier 2 future conditions never followed up. Capital Deployment one direction only

**Changes**:
- `pretrade_check.py`: Added `h4+m5 aligned → wave="mid"` (was falling through to "small")
- `pretrade_check.py`: Mid-wave M5-aligned branch +3 when H4 supports (was always +2)
- `pretrade_check.py`: WR < 40% changed from hard grade cap → warning only. Grade preserved
- `SKILL_trader.md` Tier 2: `LONG if / SHORT if` → `Best NOW: {LONG/SHORT @price}`
- `SKILL_trader.md` Capital Deployment: `#1 best setup` → `#1 LONG / #1 SHORT` both directions
- `SKILL_trader.md` Directional mix: Must write trade plan BEFORE deciding to pass
- `strategy_memory.md`: Added USD_JPY 4/10 + "H4-supported SHORT ≠ counter-trade"

**Result**: USD_JPY SHORT same chart: C(~2, small) → **A(6, mid)**. LONGs unaffected.

## 2026-04-09 — strategy_memory: remove SHORT-biased rules, add sample-period context

**Trigger**: User feedback — SHORT win rate stats are market-regime-dependent, not permanent pair properties. Treating them as rules blocks profit in range/bearish markets.

**Changes to strategy_memory.md**:
- All per-pair SHORT stats now annotated with "(sample: 3/17-4/9, predominantly bullish period)"
- Removed "Avoid" / "money pit" / "size down" directives on SHORT side
- "LONG-only bias" lessons reframed: problem was "not reading chart for both directions," not "SHORTs are bad"
- Pretrade HIGH-SHORT failures reframed: regime-dependent, applies equally to LONGs in bear market
- USD_JPY flow rules softened: chart-first, not direction-first

**Principle**: Statistics from a trending sample don't generalize to all market conditions. Read the chart, not the win-rate table.

## 2026-04-09 — Self-audit: 13 bugs found and fixed across 4 files

**Found by**: Recursive self-questioning ("穴がないか自問熟考繰り返して")

### CRITICAL bugs (silently failing in production):
1. **session_data.py**: `by_pair` from strategy_feedback.json is a dict, code iterated as list → **pair edge inline display was always empty** (dead feature since deployment). Fixed dict iteration + field name `total_pl_jpy`.
2. **session_data.py**: Calendar key `"economic_calendar"` → should be `"calendar"`. Field names wrong: `title`→`event`, `currencies`→`country`. Economic calendar was silently showing nothing.
3. **quality_audit.py**: `self_check()` regex counted LIMIT orders as held positions → false SELF-CHECK mismatches (AUD_USD LIMIT appearing as "held"). Added LIMIT exclusion.
4. **quality_audit.py**: `@price` tag from s_conviction_scan output was ignored — `append_audit_history()` re-loaded from stale technicals cache instead. Now parses `@price` from scan output directly.

### HIGH fixes:
5. **quality_audit.py**: BE SL detection gate `upl > 100` too high → lowered to `upl > 0`. Any profit position with BE SL is now flagged.
6. **quality_audit.py**: `audit_history.jsonl` grew unbounded. Added rotation (keep last 5000 lines, ~6 months).
7. **session_data.py**: Churn detection only scanned last 50 lines of live_trade_log → now scans all lines for today's date.

### Prompt design fixes:
8. **trader SKILL.md**: Close/Hold "freed margin" line allowed "nothing better available" escape → now requires naming a specific pair ("scanned all 7 pairs, best was [PAIR] but [why not]").
9. **trader SKILL.md**: Capital Deployment Check was conditional (margin < 60% only) → now required EVERY session.
10. **trader SKILL.md**: Pair edge line referenced vague "strategy_memory / session_data" → now says "copied from session_data TRADES line" with exact format reference.
11. **daily-review SKILL.md**: audit_history.jsonl format was undocumented → added JSON schema, field descriptions, recipe attribution instructions.
12. **daily-review SKILL.md**: Recipe scorecard added — running tally per recipe for promotion/deprecation after 10+ data points.

**Files**: `tools/session_data.py`, `tools/quality_audit.py`, `~/.claude/scheduled-tasks/trader/SKILL.md`, `~/.claude/scheduled-tasks/daily-review/SKILL.md`, `docs/CHANGELOG.md`

---

## 2026-04-09 — Trader Performance: Market Narrative + Knowledge-Action Gap Fix

**Problem**: Trader (Sonnet) knows what to do but doesn't do it. strategy_memory has 260 lines of wisdom that's read at session start and forgotten by output time. Rotation SHORTs identified but never executed (4/8-4/9: 13 entries all LONG, 0 SHORTs). S-conviction undersized 6/7 times. pretrade_check scored EUR_JPY LOW(1) despite 69% WR + 6/6 wins. session_data shows "what candles look like" but not "why the market is moving."

**Changes**:
1. **trader SKILL.md — Market Narrative**: New required block BEFORE indicators: "Driving force / Theme / My best edge / Session." Forces WHY before WHAT. Can't copy-paste (market changes).
2. **trader SKILL.md — Conviction block**: Added "Pair edge: ___% WR, avg ___JPY" and "Margin after: ___%". Forces Sonnet to look up pair history BEFORE committing conviction. AUD_USD LONG (50% WR) can't be rated S when the number is visible.
3. **trader SKILL.md — Rotation force**: ALL_LONG/SHORT → must name "Best rotation candidate" with M5 indicators OR write specific trigger. "No setup" escape hatch replaced with commitment.
4. **trader SKILL.md — Close or Hold**: Added "If I closed, I would use freed margin for: ___". Makes opportunity cost visible.
5. **session_data.py**: Added session time marker (Tokyo/London/NY), per-pair edge stats inline with TRADES, economic calendar events, today's entry count per pair with churn warning.
6. **pretrade_check.py**: Pair WR <40% caps conviction at B (prevents AUD_USD MEDIUM). WR >60% + ADX>35 + macro aligned → +2 trending bonus (fixes EUR_JPY LOW). Added macro regime conflict warning at CS gap >0.3.

**Design principle**: Don't add rules — embed checks INTO the output format at the point of action. Sonnet can't write the conviction block without first looking up pair history. That's the mechanism, not "remember to check pair history."

## 2026-04-09 — Quality Audit System Overhaul: fact-based + discretionary + exit quality

**Problem**: Quality audit was fundamentally broken and philosophically misaligned:
1. **Broken regex** (line 88: `\(id=` vs actual `id=`): `held_pairs` always empty → ALL S-candidates flagged as "NOT ENTERED" including pairs already held. 100% false positive rate. Trader noticed ("audit stale or mismatched") and started ignoring all audit output.
2. **Bot-making machine**: Audit told Sonnet-trader "S-CANDIDATE MISSED → fix it" = mechanical rule-following. Contradicts "conditions met, so enter → NOT OK" philosophy.
3. **Blind to biggest losses**: Exit failures (3/27 HOLD trap -4,796 JPY, 4/8 BE SL -1,160 JPY) completely unmonitored. Audit only checked entries.
4. **No self-verification**: S-scan accuracy never measured. No feedback loop. No "audit of the audit."
5. **S-scan 3x per run**: Redundant subprocess calls.
6. **Recipe overlap**: Trend-Dip + Structural fire simultaneously on same M5 StRSI extreme, inflating candidate count.

**Changes**:
1. **quality_audit.py rewrite**: OANDA API as ground truth (not state.md regex). Script presents FACTS, not judgments. Added: exit quality checks (peak drawdown, BE SL detection, ATR×1.0 stall), self-check (OANDA vs state.md verification), audit_history.jsonl (outcome tracking). S-scan runs once, result cached. Output: quality_audit.md (human) + quality_audit.json (machine) + audit_history.jsonl (append-only).
2. **s_conviction_scan.py**: Added deduplication (same pair+direction → strongest recipe only). Added current price to output for outcome tracking.
3. **quality-audit SKILL.md**: Rewritten for "Think at the Point of Output". Auditor MUST write judgment for each finding (REPORT/NOISE with reasoning). Self-questioning step added. No more copy-paste relay.
4. **trader SKILL.md**: "Read and fix" → "Read and respond". Audit is DATA, not instructions. Trader writes "If I would enter: ___ / If I would not: ___" for each S-scan finding.
5. **daily-review SKILL.md**: Added Step 2.5 (Audit Accuracy Review). Reads audit_history.jsonl, correlates S-scan signals with actual price movement, writes recipe accuracy to strategy_memory.md. Enables recipe promotion/deprecation.
6. **CLAUDE.md**: Updated architecture table and self-improvement loop description.

**Design principle**: Separate fact-gathering (script) from judgment (Sonnet-auditor). Force thinking at every node: script presents data → auditor judges → trader responds. Every assertion has a verification mechanism.

## 2026-04-09 — Fix LONG-only bias: both-direction scan + rotation trading

**Problem**: 4/8-4/9: 13 entries, 0 SHORTs. Trader used M5 bearish signals (StRSI=1.0, bear div, sellers dominant) defensively only (tighten TP, add SL) — never as SHORT entry signals. USD_JPY in clear H4+H1 downtrend, tried LONG 3x, all lost. Root cause: shallow indicator scan (ADX+StRSI+CS = 3 indicators) locks into one direction. Quality audit flagged "全ポジションLONG" repeatedly but escape hatch ("no H4 extreme") was too easy.

**Changes**:
1. **SKILL_trader.md Tier 1 format**: Replaced single-direction "I would enter if" with both-direction indicator analysis. Now requires LONG case + SHORT case with 3+ indicator categories each, and explicit comparison to choose direction
2. **SKILL_trader.md Tier 2 format**: Added "SHORT if" alongside "LONG if" — can't skip opposite direction
3. **SKILL_trader.md Directional mix check**: Replaced "no H4 extreme" escape hatch with requirement to check M5 depth across all 7 pairs for opposite-direction setups. Writing "no setup" now requires listing what was checked
4. **Added rotation trade concept**: Rotation SHORT within LONG thesis (2000-3000u, M5 pullback, 15-30min) is distinct from counter-trade (swing size against trend). Clarified in both SKILL and strategy_memory
5. **strategy_memory.md**: Added 3 new 負けパターン (defensive-only M5 use, macro overriding chart, shallow scan bias), 1 new 勝ちパターン (rotation trading), clarified counter-trade warning, added observations

**Design principle**: Format forces thinking — trader must fill in indicators for BOTH directions. Can't write "LONG" without also evaluating SHORT and explaining why LONG is stronger.

## 2026-04-08 — Fix reaper killing active sessions (root cause of exit code 143)

**Problem**: Trader sessions dying mid-execution with exit code 143 (SIGTERM). Investigation revealed the LaunchAgent reaper (`reap_stale_agents.sh`) was the killer. ORPHAN_AGE=300s threshold treated non-lock-owner `bypassPermissions` processes as "orphans" and killed them at 5 minutes. But Claude Code's `per_task_limit (active=1, limit=1)` means only one session runs at a time — ALL bypassPermissions processes belong to the current session. The reaper was killing the active session's own processes.

**Changes**:
1. **Single threshold**: Replaced orphan/owner split (300s/600s) with single `KILL_AGE=660s` (11 min). Session self-destruct timer is 540s, so only truly stuck processes (survived past self-destruct) get killed.
2. **Removed LOCK_PID distinction**: No more owner vs orphan logic. Every bypassPermissions process gets the same generous threshold.

**Impact**: Sessions no longer killed by reaper during normal 8-minute execution. Only genuinely stuck processes (>11 min) get reaped.

## 2026-04-08 — Zombie process prevention (6-layer fix)

**Problem**: Trader cron (every 1 min) spawned a new Claude process each invocation. 87.5% hit ALREADY_RUNNING but the process never terminated — creating 7+ zombies per 8-min session. Root causes: (1) "write no text" instruction left harness waiting, (2) lock PID was bash shell `$$` not Claude process `$PPID`, (3) existing reaper had wrong grep pattern (`disallowedTools` didn't match trader processes), (4) reaper had octal parsing bug (08/09 caused bash errors).

**Changes**:
1. **Layer 1 — Zombie reaper in Bash①**: Every session start kills ALL `bypassPermissions` processes older than 10 min.
2. **Layer 2 — PID fix**: `$$` → `$PPID` in lock file writes (Bash②, Next Cycle Bash). Stale lock cleanup now kills Claude, not bash shell.
3. **Layer 3 — Cron `*/2`**: 1-min → 2-min interval. Halves zombie creation rate and API cost.
4. **Layer 4 — ALREADY_RUNNING output**: "write no text" → "output SKIP". Gives harness a clear completion signal.
5. **Layer 5 — Reaper → Supervisor upgrade** (`reap_stale_agents.sh`):
   - Fixed grep: `disallowedTools|scheduled-tasks` → `bypassPermissions` (was matching ZERO trader processes)
   - Fixed octal bug: `10#$var` prefix prevents bash treating 08/09 as octal
   - Added Phase 3: detect trader dead (state.md age >10min) → Slack alert with dedup
   - Graceful shutdown: SIGTERM → 2s → SIGKILL (was: immediate SIGKILL)
6. **Layer 6 — Self-destruct timer**: Bash② spawns background `(sleep 540; kill $PPID)` — hard kill guarantee even if SESSION_END never reached. PID verified against lock file to prevent misfire on PID reuse.
7. **maxTurns 200 → 50**: Prevents runaway sessions.

**Impact**: Zombie accumulation eliminated. Stuck sessions killed within 60s (reaper) or 540s (self-destruct). Slack alert if trader dead >10min. API cost ~50% reduction.

## 2026-04-08 — Mid-session lightweight check (Next Cycle Bash: 27s → 1s)

**Problem**: Next Cycle Bash re-ran full `session_data.py` (27s) on every mid-session cycle. In an 8-min session with 2-3 cycles, this consumed 54-81s on redundant data fetches (technicals, news, macro, S-scan, memory don't change within 8 minutes). Sessions consistently cut off before state.md update.

**Changes**:
1. **tools/mid_session_check.py**: New lightweight script. Fetches only what changes mid-session: Slack messages, OANDA prices/spreads, open trades with P&L, account margin. Runs in ~1s.
2. **SKILL_trader.md**: Next Cycle Bash now calls `mid_session_check.py` instead of `session_data.py` when ELAPSED < 420s. Full `session_data.py` runs once at session start (Bash②).

**Impact**: Each mid-session cycle saves ~26s. Sessions now have ~50s more for analysis, execution, and state.md cleanup.

## 2026-04-08 — Parallelize session_data.py (43s → 27s, -37%)

**Problem**: session_data.py took 43-50s, consuming half of the 8-minute session. Two bottlenecks: refresh_factor_cache (10.6s, sequential 28 API calls) and memory recall (9.4s, model load per pair).

**Changes**:
1. **refresh_factor_cache.py**: `for pair: await` → `asyncio.gather` + `run_in_executor` for true thread parallelism. 10.6s → 2.8s (-74%).
2. **session_data.py**: Heavy I/O tasks (tech refresh, M5 candles, memory recall) run concurrently via ThreadPoolExecutor. OANDA trades fetched early to provide held_pairs for memory recall. 43s → 27s (-37%).

## 2026-04-08 — Trader session 5min → 8min (S-candidate放置対策)

**Problem**: Quality audit flagged 10 S-candidates with 41% margin idle. Trader couldn't evaluate S-candidates AND manage existing positions in 5 minutes. The extra 3 minutes are dedicated to 7-pair scan, S-candidate evaluation, and LIMIT placement — the exact steps being skipped.

**Changes**:
1. **SKILL_trader.md**: Lock timeout 300s→480s, SESSION_END trigger 240s→420s. Time allocation restructured: 0-1 data, 1-3 positions, 3-5 scan+S-candidates+LIMITs, 5-7 execute, 7 cleanup.
2. **schedule.json**: Description updated.
3. **CLAUDE.md**: Architecture table and method description updated.

## 2026-04-08 — Fix: "Default is Take Profit" was gated behind ATR×1.0

**Problem**: "Default is Take Profit" principle existed at the top of risk-management.md, but the execution format only triggered at ATR×1.0. Profits in the ATR×0.5-0.8 range (the most common profit level) were invisible to the trader. Data: 28 winning trades averaged 71% peak capture. 14 losing trades were once in profit — 6,110 JPY wasted. Total left on table: 11,902 JPY.

**Root cause**: The 3-option format ("A/B/C — Hold as-is") could be filled in without reading the market. "C — Hold as-is. H1 thesis intact." is copy-pasteable. The format didn't force thinking.

**Changes**:
1. **risk-management.md**: "Default is Take Profit" now applies at ALL profit levels, not just ATR×1.0. ATR×1.0 still triggers profit_check for data, but the principle is unconditional.
2. **SKILL_trader.md**: Replaced 3-option table with "Close or Hold" block that must be written every session for every position. Format: `Close now: +Xpip = +Y JPY / Peak: +Zpip / I'm not closing because: ___ / This reason disappears if: ___`. Can't be filled without reading M5 price action.
3. **state.md template**: Removed separate "3-Option Management" section — Close-or-Hold block is now part of each position block.

## 2026-04-08 — Fix: Margin pre-check + limit order discipline

**Problem**: Trader stacked EUR_JPY + EUR_USD + GBP_JPY without margin calculation → 97% margin → forced EUR_JPY close at -319 JPY. Also used market orders on Easter Monday thin liquidity.

**Changes**:
1. **SKILL.md**: Added mandatory "Margin gate" step BEFORE conviction block. Must calculate current + new + pending LIMIT margin. Blocked above 85% (90% with S-conviction only). Output format forces the calculation.
2. **SKILL.md**: Changed "S/A = market order" rule → market conditions determine order type. Thin market/holiday = LIMIT even for S-conviction. M5 mid-range = LIMIT at structural level.
3. **risk-management.md**: Added pre-entry margin check section with calculation template. Added two failure patterns (margin overflow forced close, market order in thin liquidity).

## 2026-04-08 — Fix: Slack user messages consumed without reply

**Problem**: `session_data.py` called `slack_read.py` which updated `last_read_ts` on read. If the trader session didn't reply, the message was lost — next session wouldn't see it.

**Fix**: `slack_read.py` now accepts `--no-update-ts` (used by session_data.py). `last_read_ts` is only advanced by `slack_post.py --reply-to` after a successful reply. Unread messages keep appearing until replied to.

## 2026-04-08 — New: quality-audit scheduled task (Sonnet, every 30 min)

**Purpose**: Cross-check trader decisions against rules in near-real-time. Catches issues that previously required manual review (missed S-candidates, undersizing, rule misapplication).

**Components**:
1. `tools/quality_audit.py` — audit script (6 checks: S-candidates missed, sizing discipline, margin utilization, rule misapplication, pass reason quality, directional bias)
2. `~/.claude/scheduled-tasks/quality-audit/` — task definition (Sonnet, */30 cron)
3. `docs/SKILL_quality-audit.md` — reference copy

**Integration (導線)**:
- `tools/session_data.py` → shows `logs/quality_audit.md` in session output if recent (<1h)
- `SKILL_trader.md` → tells trader to read and act on audit issues
- `CLAUDE.md` → task table, runtime files, scripts, self-improvement loop diagram all updated
- Slack `#qr-daily` → CRITICAL/WARNING issues posted automatically

**Files changed**: `tools/quality_audit.py` (new), `tools/session_data.py`, `CLAUDE.md`, `docs/SKILL_trader.md`, `docs/SKILL_quality-audit.md` (new), `docs/CHANGELOG.md`

## 2026-04-08 — Fix remaining PASS excuses: circuit breaker direction + spread/S-Type mismatch

**Problem**: Despite previous fixes, trader still blocking entries:
1. AUD_JPY LONG Momentum-S + Squeeze-S (double S!) blocked by "SHORT circuit breaker" — rule says direction-only but trader applying to both
2. GBP_JPY Momentum-S blocked by "spread 2.8pip too wide for scalp" — but Momentum-S is NOT a scalp. TP=10-15pip, spread=19-28% = fine

**Fix**:
1. **SKILL_trader.md**: Added explicit "Circuit breaker is DIRECTION-ONLY" section with example
2. **SKILL_trader.md**: Added "Match S-Type to spread" — Momentum-S recipe = Momentum hold time/TP, not scalp

**Files changed**: `docs/SKILL_trader.md`, `docs/CHANGELOG.md`

## 2026-04-08 — Fix false PASS excuses: spread normalization + thin market ≠ no entry

**Problem**: Trader passed on GBP_JPY Squeeze-S (H1 ADX=33 + M5 squeeze + M1 confirmed) because "spread 2.8pip too wide." But 2.8pip IS GBP_JPY's normal spread. Also passed on AUD_JPY LONG because "4-SHORT-loss circuit breaker" — but S-scan detected LONG, not SHORT. Also wrote "Easter Monday thin liquidity" as reason for zero LIMITs while simultaneously holding a market-ordered EUR_JPY LONG.

**Fix (3 changes)**:
1. **SKILL_trader.md**: Added normal spread reference table. "Wide" means above normal range, not the normal range itself. S-candidates can't be passed on spread within normal range
2. **SKILL_trader.md**: Added "Thin market ≠ no entries" — thin market affects SL design, not entry decisions
3. **strategy_memory.md**: Added "circuit breaker is same-direction only" to Confirmed Patterns

**Files changed**: `docs/SKILL_trader.md`, `collab_trade/strategy_memory.md`, `docs/CHANGELOG.md`

## 2026-04-08 — S-Conviction Scanner: auto-detect TF × indicator patterns

**Problem**: Trader sees individual indicators (H4 StRSI=1.0, H1 CCI=200, M5 StRSI=0.0) as separate data points and rates B+. But as a CROSS-TF PATTERN, this is textbook S-conviction counter. EUR_JPY had 6 extreme markers and was entered at 700u (0.3% NAV).

**Root cause**: No tool maps TF × indicator combinations to conviction levels. The trader must mentally assemble patterns from raw data every session — and under time pressure, defaults to B.

**Fix**: New `tools/s_conviction_scan.py` with 6 proven recipes:
1. Multi-TF Extreme Counter (H4+H1 extreme + M5 opposite)
2. Trend Dip (H1 ADX≥25 + M5 extreme, Confirmed Pattern)
3. Multi-TF Divergence (H4+H1 div + extreme)
4. Currency Strength Momentum (CS gap≥0.5 + MTF aligned)
5. Structural Confluence (M5 BB edge + extreme + H1 trend)
6. Squeeze Breakout (M5 squeeze + H1 strong + M1 confirmed)

**Integration**: Added to session_data.py as `S-CONVICTION CANDIDATES` section (runs after ADAPTIVE TECHNICALS). When 🎯 fires, trader must enter at S-size or explain which part of the recipe fails.

**Current scan result**: 8 S-candidates found (EUR_USD LONG, EUR_JPY SHORT counter, GBP_JPY LONG dip, AUD_JPY LONG momentum, etc.) while trader had 0 positions and 700u LIMIT.

**Files changed**: `tools/s_conviction_scan.py` (new), `tools/session_data.py`, `docs/SKILL_trader.md`

## 2026-04-08 — Fix sizing discipline + anti-churn + margin deployment (entry speed postmortem)

**Problem**: 4/1-4/8 performance: 40% WR, -2,765 JPY net, avg size 2,927u. Compare 3/31: 65% WR, +4,591 JPY, avg 4,737u. Three root causes identified:

1. **Double-discounting**: S-conviction trades averaged 3,273u (target: 10,000u). Trader rated S in conviction block, then saw pretrade WR=37% and panicked to B-size. Historical WR is already in the pretrade score — counting it twice
2. **Junk-size entries**: 500u/700u/1000u entries that can't cover spread cost. 4/7: EUR_USD 500u won +32 JPY (meaningless)
3. **Churn**: 4/7 AUD_JPY closed and re-entered 3× in succession = 9.6pip spread burned for -778 JPY total
4. **0% margin as default**: 4/7 ended with 0 open positions, 2 pending LIMITs, +40 JPY. Capital sat idle
5. **strategy_memory.md fear bias**: 18 warnings vs 4 success patterns. Trader reads a minefield map before every session

**Fix (5 changes)**:
1. **SKILL_trader.md**: Added "Sizing discipline — the 3 rules" (no double-discount, min 2000u, S/A=market order)
2. **SKILL_trader.md**: Added "0% margin = SESSION_END blocker" with 3 required questions
3. **SKILL_trader.md**: Added "Anti-churn rule" requiring better price + new reason for same-pair re-entry
4. **strategy_memory.md**: Rebalanced — added 7 success patterns to Confirmed Patterns. Split mental rules into "攻め" (read first) and "守り" sections
5. **pretrade-check.md**: Added "二重割引禁止" section — pretrade output changes conviction judgment, NOT size calculation

**Files changed**: `docs/SKILL_trader.md`, `collab_trade/strategy_memory.md`, `.claude/skills/pretrade-check.md`

## 2026-04-08 — BE SL ban at ATR×1.0+ / TP spread buffer (AUD_JPY +1,200→+40 postmortem)

**Problem**: AUD_JPY LONG 5000u peaked at +1,200 JPY (bid 111.096). Trader moved SL to breakeven (entry+1pip=110.860) instead of taking profit. Price reversed, BE SL hit, closed at +40 JPY. Two root causes:
1. **BE SL bypassed profit_check** — ATR×1.0 reached but profit_check was never run. SL→BE was used as a "safe" alternative to profit evaluation, identical pattern to the 3/27 Default HOLD trap
2. **TP missed by 0.4pip due to spread** — TP=111.100, bid peaked 111.096. Spread=2.4pip. TP didn't account for spread buffer

**Fix (3 changes)**:
1. **BE SL banned at ATR×1.0+**: Only 3 actions allowed — HALF TP (default) / FULL TP / HOLD+trailing(≥50% profit). Moving SL to entry price gives back 100% of profit — that's not risk management. If trader writes "SL moved to BE", must first state how much profit is being forfeited and why it's better than HALF TP
2. **profit_check mandatory before SL modification**: When ATR×1.0 reached, profit_check must run FIRST. SL changes without prior profit_check = rule violation
3. **TP spread buffer**: `TP = structural_level - spread` for LONGs, `+ spread` for SHORTs. Prevents fills missed by fraction of a pip

**Files changed**: `.claude/rules/risk-management.md`, `docs/SKILL_trader.md`

## 2026-04-07 — pretrade_check.py --counter mode: counter-trades no longer structurally blocked

**Problem**: `assess_setup_quality()` scores MTF alignment 0-4 based on DI+/DI- direction agreement across TFs. Counter-trades are by definition against the upper TF → always score 0 on MTF alignment → always grade C → trader never enters counter-trades even when H4 StRSI=1.00 extreme.

**Fix**: New `assess_counter_trade()` function with inverted evaluation axes:
1. **H4 Extreme (0-3)**: The more extreme the upper TF (StRSI near 0/1, CCI ±200, RSI <30/>70), the HIGHER the score — opposite of normal mode
2. **H1 Divergence/Fatigue (0-2)**: Divergence + CCI extreme confirms reversal
3. **M5 Reversal Signal (0-2)**: StRSI + MACD hist timing trigger
4. **Spread penalty (0 to -1)**: 8pip reference target for counter-trades

Grades capped at B+ max (counter-trades never get S/A sizing — 2000-3000u max). CLI: `pretrade_check.py PAIR DIR --counter`. Format output clearly labeled `🔄 COUNTER-TRADE` with inverted axis explanation.

Also fixed: "pass recommended" → "data suggests caution — you decide" (tool output is data, not orders).

**Files changed**: `collab_trade/memory/pretrade_check.py`, `docs/SKILL_trader.md`, `.claude/skills/pretrade-check.md`

## 2026-04-07 — Counter-trade execution + directional mix + LIMIT deployment

**Problem**: Trader identifies MTF counter-trades in scan ("H4 overbought, M5 SHORT scalp") but never executes them. All positions are same direction (LONG only). Idle margin (34%) sits with no LIMIT orders deployed. Result: missing pullback profits, concentrated directional risk.

**Fix (3 changes)**:
1. **Directional mix check (output format)**: Required block in state.md — `N LONG / N SHORT | one-sided ⚠️ | Counter-trade candidate: ___`. Can't write "all LONG because thesis is bullish" — must identify a specific counter-trade or explain with numbers why none exists
2. **MTF counter-trade → Action mandatory**: Tier 1 scan now requires `→ Action: [LIMIT placed / not placing because ___]` after each counter-trade identification. Identifying without acting = analyst, not trader
3. **Idle margin → LIMIT orders**: New section in Capital Deployment. When margin > 30% idle, deploy LIMITs at structural levels with TP+SL on fill. Event risk ≠ "do nothing" — event risk = "place LIMITs for BOTH outcomes"
4. **Counter type added**: Conviction block Type field now includes "Counter" (M5 against H1/H4, B-max size, ATR×0.3-0.7 target, tight SL)

## 2026-04-07 — Trader prompt overhaul: 5 structural improvements

**Problem**: SKILL_trader.md was 837 lines. 30+ dated failure patterns embedded inline created "don't do X" cognitive overload. Trader spent tokens reading rules instead of reading the market. Output formats didn't force depth — "Checked" step had no output field, 7-pair scan was uniformly shallow, wave position was never explicit, and indicators were output before price action.

**Fix (5 changes)**:
1. **Prompt halved (837→405 lines)**: All dated lesson/history moved to `docs/TRADER_LESSONS.md`. SKILL retains only flow, formats, and principles. Lessons live in strategy_memory.md (distilled by daily-review)
2. **"Checked" line in Capital Deployment**: Format now requires `→ Checked: [what I looked at] → Result: [value] → [supports/contradicts]`. Cannot complete the block without actually checking the indicator
3. **session_data.py outputs M5 PRICE ACTION first**: New section fetches 20 M5 candles per pair, outputs candle shape analysis (buyers/sellers, momentum phase, wick pressure, high/low updates) BEFORE indicator data. Model reads chart shape before forming indicator-based opinions
4. **7-pair scan Tier 1/Tier 2**: Held positions + best candidates get deep analysis (price action + wave position + entry condition + MTF counter-trade). Remaining pairs get 1-line quick scan. Depth where it matters, coverage everywhere
5. **Wave position mandatory**: Tier 1 scan requires `Wave position: [Fib X%] / [BB position] / [structural level] [N]pip away`. Prevents "StRSI=1.0 → skip" without knowing the structural context (e.g., "H1 BB upper 3pip away")

**Files changed**: `docs/SKILL_trader.md` (rewrite), `docs/TRADER_LESSONS.md` (new), `tools/session_data.py` (M5 PRICE ACTION section added)

## 2026-04-07 — "I would enter at price X" → must place LIMIT ORDER

**Problem**: Trader writes "LONG if pulls back to 1.1535" in scan but never places a limit order. Next session, conditions change, writes new "if..." plan. Endless waiting loop. Margin stays idle.

**Fix**: In 7-pair scan column 2, if the entry trigger names a price → it's a limit order. Place it now. "Writing a price without placing a limit = leaving money on the table." Added ❌ example of "wish without limit" and ✅ example of "limit placed with id."

## 2026-04-07 — Fix stale state.md: freshness check + mandatory update enforcement

**Problem**: state.md was stuck on 4/4 data while trader actively traded on 4/7 (17+ trades, add-ons, SL modifications). Next sessions read 3-day-old positions/thesis/scan = blind trading. Root cause: "update state.md" was a rule (ignorable), not enforced in output or tooling.

**Fix**:
1. SESSION_END Bash now checks state.md age — emits `⚠️ STATE.MD STALE` warning if >1 hour old
2. Added explicit "state.md update is NOT optional" block with minimum required content
3. Framed as consequence ("next session starts blind") not rule ("you must update")

## 2026-04-07 — Capital Deployment Check + cautionary bias antidote

Refined margin < 60% output block: from "best 2 setups, why not entered" (pushes quantity) to "#1 best setup, current conviction, what would upgrade to S, P&L at S-size" (pushes quality + sizing). Goal: fewer trades, bigger size. Added antidote to strategy_memory cautionary bias (30 warnings vs 12 positive patterns → trader becomes too cautious → undersizes).

## 2026-04-07 — SL recommendation: ATR×1.2 formula → structural level menu

**Problem**: protection_check.py recommended SL at `ATR×1.2` with copy-paste PUT commands since 3/31. TP was migrated to structural levels on 3/31(6), but SL was never migrated. Despite SKILL.md and risk-management.md repeatedly saying "SL must be structural, not ATR×N," the script output `SL recommendation: 184.380 (ATR x1.2 = 12.1pip)` — and the trader copied it verbatim. This is the root cause of repeated tight-SL hunting losses (4/3 -984 JPY, and continued pattern on 4/7).

**Fix (protection_check.py)**:
1. **New `find_structural_sl_levels()`**: Collects invalidation-side structural levels (H4/H1/M5 swing, cluster, BB, Ichimoku cloud) sorted by distance from entry. Same approach as the existing `find_structural_levels()` for TP
2. **SL section rewritten**: No more `recommended_sl_pips = atr_pips * 1.2`. Instead shows `📍 Structural SL candidates` menu with price, label, and ATR ratio for context
3. **Removed auto-generated SL fix commands**: No more copy-paste PUT commands for SL. The trader must choose a structural level and articulate why
4. **ATR shown as "size reference only"**: Still displayed for context but explicitly labeled as not-for-placement
5. **Too tight / too wide warnings**: Still fire (ATR x0.7 / x2.5 thresholds) but recommend structural levels instead of ATR×1.2

**What changed for the trader**: Instead of seeing `SL recommendation: 184.380 (ATR x1.2)` and copying it, the trader now sees a menu like:
```
📍 Structural SL candidates (if you want SL):
  1. 184.366 = M5 BB lower (ATR x1.1)
  2. 184.353 = H1 BB mid (ATR x1.2)
  3. 184.300 = M5 cluster (ATR x1.6)
  ATR=12.3pip (size reference only, not placement)
```
This forces choosing based on market structure, not formula.

**Files changed**: `tools/protection_check.py`, `docs/CHANGELOG.md`

## 2026-04-07 — Margin Deployment Check: forced output when margin < 60%

Added required output block to SKILL_trader.md 7-Pair Scan section. When margin < 60%, trader must write: best 2 setups, why not entered, and worst-case if entered both. Forces confrontation with idle capital instead of defaulting to "nothing here." 60% is the minimum, 70-85% is healthy and aggressive.

## 2026-04-07 — Weekly +25% NAV performance target added to trader prompt

Added performance target to SKILL_trader.md: +25% of NAV per week (~5%/day). Placed in the prompt (not state.md) so it persists across sessions and isn't overwritten. Framed as a self-question ("did I look hard enough?") rather than a rule, per prompt design principles.

## 2026-04-07 — PDCA high-speed loop: instant learning + memory.db integration

**Problem**: Self-improvement loop was too slow (24h feedback delay). Trader noticed mistakes and wrote them to state.md Lessons, but they never reached strategy_memory.md until daily-review ran (once/day, and it was broken). Memory.db had 281 chunks of past trade lessons but recall was only triggered for held pairs, missing recently-lost pairs.

**Fix (3 changes)**:
1. `docs/SKILL_trader.md`: Added "Learning record" section — when trader notices a pattern/mistake, write to BOTH state.md Lessons AND strategy_memory.md Active Observations immediately. 5-min PDCA instead of 24h. Daily-review distills and promotes, no longer sole writer.
2. `tools/session_data.py`: MEMORY RECALL now triggers for held pairs AND today's loss pairs. Adds "(HELD)" / "(RECENT LOSS)" tags. Lost on GBP_USD? Past GBP_USD failure lessons auto-surface.
3. `docs/SKILL_trader.md`: Added "How to use MEMORY RECALL" guidance — read recalled lessons BEFORE making decisions on held positions.

**Design**: strategy_memory.md is a living document that the trader writes to during trading (fast lane) and daily-review distills nightly (cleanup lane). Two writers, one document. daily-review owns promotion (Active→Confirmed) and pruning (300 line limit).

**Files changed**: `docs/SKILL_trader.md`, `tools/session_data.py`, `tools/daily_review.py`, `~/.claude/scheduled-tasks/daily-review/SKILL.md`, `docs/CHANGELOG.md`

## 2026-04-07 — Self-improvement loop fix: daily-review + pretrade matching

**Problem**: PDCA loop was broken. strategy_memory.md hadn't been updated since 4/6. pretrade_outcomes had only 10% match rate (24/240). lesson_from_review was always NULL. The trader kept making the same SL mistakes because lessons weren't persisted across days.

**Root causes**:
1. `daily_review.py` matched pretrade_outcomes only for `session_date = today` — trades entered on day N but closed on day N+1 were never matched
2. daily-review SKILL had 4 bash steps + 5 file reads before writing strategy_memory.md — too much work, session timed out before reaching the write step
3. No feedback path from review back to pretrade_outcomes.lesson_from_review

**Fix (3 changes)**:
1. `tools/daily_review.py` `match_pretrade_outcomes()`: now matches ALL unmatched outcomes (not just today's) and looks back 3 days for closed trades. Match rate: 10% → 17%
2. `~/.claude/scheduled-tasks/daily-review/SKILL.md`: simplified from 4 bash steps to 2. Bash① collects ALL data in one command. LLM focuses on thinking and writing. Bash② verifies + ingests + posts
3. Added explicit "2 bash calls maximum" rule to prevent the session from spending all its time on data collection instead of reflection

**Files changed**: `tools/daily_review.py`, `~/.claude/scheduled-tasks/daily-review/SKILL.md`, `docs/CHANGELOG.md`

## 2026-04-07 — 7-Pair Scan: MTF counter-trade column added

**Problem**: All 7 pairs had LONG-only plans on 4/7 while H4 data showed AUD_JPY StRSI=1.0 + MACD div=-1.0, GBP_JPY MACD div=-1.0. Short-term SHORT scalps were available but invisible because macro direction (USD weak, JPY weakest) biased all analysis toward LONG. The existing "directional bias check" rule was ignored — adding more rules doesn't help.

**Fix**: Added 4th column to 7-Pair Scan table: `MTF counter-trade`. Format: `___TF overextended → ___ if ___`. Forces the model to check H4 StRSI/div for every pair and write the number. When H4 is overextended, the model must articulate the short-term reversal trade. When not overextended, writing "N/A" requires the H4 StRSI number as proof of checking.

**Design principle**: Not a rule ("check for shorts"). An output format that makes bias visible during the act of writing. The model can't fill the column without looking at the higher TF — if H4 StRSI=1.0 is staring at it while writing "LONG if...", the contradiction becomes self-evident.

**Files changed**: `docs/SKILL_trader.md`, `docs/CHANGELOG.md`

## 2026-04-07 — P&L reporting fix: OANDA API as single source of truth

**Problem**: state.md "Today confirmed P&L" was manually tallied by the trader LLM, causing:
1. Date boundary errors: 4/6 trades mixed into 4/7 totals (state.md claimed +1,851 JPY, OANDA actual was -612 JPY — 2,463 JPY discrepancy)
2. `slack_daily_summary.py` had path bug (`../..` instead of `..`) — P&L and trade counts always returned 0
3. `live_trade_log.txt` had recording gaps (log showed +32 JPY for 4/7, OANDA showed -612 JPY — 10 closes missing from log)

**Fix (3 changes)**:
1. `tools/slack_daily_summary.py` lines 58, 71, 107: fixed `../..` → `..` (path was resolving to `/Users/tossaki/App/` instead of `/Users/tossaki/App/quantrabbit/`)
2. `tools/session_data.py`: replaced `trade_performance.py --days 1` (log parsing) with `intraday_pl_update.py --dry-run` (OANDA API). Added "NOTE: This is the AUTHORITATIVE P&L" label
3. `docs/SKILL_trader.md`: added P&L reporting rule — "Use OANDA number from session_data, not manual tallies. Past Closed table is TODAY only (JST). Clear at day boundary."

**Root cause**: The trader LLM was summing P&L from its own trade log in state.md, which accumulated across days and missed trades not recorded in live_trade_log.txt. OANDA transactions API is the only authoritative source.

**Files changed**: `tools/slack_daily_summary.py`, `tools/session_data.py`, `docs/SKILL_trader.md`, `docs/CHANGELOG.md`

## 2026-04-06 — Reverted 10-min → 5-min + mandatory SESSION_END + duplicate instance cleanup

**Problem**: Trader sessions were running ~5 min and completing "healthy" per Claude Desktop, but SESSION_END (performance + ingest + lock release) was not reliably firing. The LLM would self-terminate before reaching the 240s ELAPSED threshold, skipping cleanup. Additionally, trader was registered in 2 Claude Desktop instances (a98d068e + 14227c4c), causing resource waste and potential conflicts. Slack responses were delayed or missing because sessions ran but didn't post.

**Root cause analysis**:
- Session JSONs showed 0-3s duration — this was misleading. Actual durations from `CCD CycleHealth` logs were 263-401s (all "healthy")
- `global_limit=3` and `per_task_limit=1` in Claude Desktop prevented concurrent sessions (expected behavior)
- The LLM completed analysis in 2-3 cycles and exited without running the final Next Cycle Bash that would trigger SESSION_END
- `codex_trade_supervisor.out` (6.4MB of `/tmp/codex_trade_supervisor.sh: No such file or directory`) was a dead legacy artifact — deleted

**10-min attempt failed**: 2 consecutive sessions hit Claude Desktop's ~600s inactivity timeout (API response stalls when context grows large over multiple cycles). Both ended as "unhealthy" at 1099s. Same failure mode as the previous 10-min attempt (see below). Reverted to 5-min.

**Fix (4 changes)**:
1. SESSION_END threshold: kept at `ELAPSED >= 240` (5 min sessions)
2. Stale lock threshold: kept at `AGE -lt 300`
3. **Mandatory SESSION_END rule added to SKILL**: "NEVER end a session without LOCK_RELEASED. Every response MUST end with Next Cycle Bash." — this is the key fix that ensures cleanup runs
4. Disabled all 6 tasks in 14227c4c instance (trader + jam-deploy + daily-review + daily-performance-report + intraday-pl-update + daily-slack-summary). Single instance (a98d068e) only.

**Files changed**: `docs/SKILL_trader.md`, `CLAUDE.md`, `docs/CHANGELOG.md`, deleted `logs/codex_trade_supervisor.out`, disabled tasks in `claude-code-sessions/14227c4c/.../scheduled-tasks.json`

## 2026-04-06 — Session extended to 15 minutes + STALE_LOCK auto-ingest

**Problem**: Sessions dying without reaching SESSION_END. ingest.py never runs → memory.db stale. Root cause: session_data.py output is massive (7 pairs × M5 20 candles + full technicals + news), model spends all 10 minutes analyzing without emitting Next Cycle Bash.

**Fix (3 changes)**:
1. Lock timeout: 600s → 900s (15 min hard limit before cron kills session)
2. SESSION_END threshold: 600s (10 min — gives 5 min buffer before kill)
3. STALE_LOCK detection: now runs `ingest.py` automatically before starting new session (guaranteed cleanup even if previous session died)

**Effect**: SESSION_END triggers at 10 min, cron kills at 15 min. 5-min buffer for ingest to complete. If session still dies, next session's STALE_LOCK path runs ingest as insurance.

## 2026-04-06 — Session extended to 10 minutes (lock threshold fix)

**Problem**: Earlier 10-min attempt failed because Bash① lock check (`AGE -lt 300`) and Next Cycle Bash (`ELAPSED -ge 300`) were out of sync — one was changed but the other wasn't. New cron killed running sessions at 5 min (STALE_LOCK), causing 30-second zombie sessions (PID 3292 incident).

**Fix**: Both thresholds changed to 600 (10 min) simultaneously:
- Bash① lock check: `AGE -lt 300` → `AGE -lt 600`
- Next Cycle Bash: `ELAPSED -ge 300` → `ELAPSED -ge 600`
- Updated: SKILL_trader.md, schedule.json description, CLAUDE.md

**Rationale**: Average hold time is long enough that 11-min max monitoring gap is acceptable. 10 min gives time for proper chart reading, Different lens, cross-pair analysis, and Fib — all of which were being skipped under 5-min pressure.

## 2026-04-06 — Trader: chart-first time allocation + strategy_memory lessons

**Problem**: Trader pattern-matched indicators (H1 StRSI=1.0 → "overbought → SHORT") instead of reading chart shape. Skipped pretrade_check, conviction block, and Different lens. AUD_JPY SHORT -203 JPY — H4 was BULL (N-wave q=0.65), pullback bodies shrinking (4.9→2.7→1.7→0.5), limit filled into rising market.

**Attempted 10-min fix → reverted**: Extended session to 10 min, but Claude Code kills processes at ~5 min. Relay mechanism added complexity without adding thinking time. Reverted to 5-min sessions.

**Actual fix**: Restructured 5-minute time allocation to prioritize chart reading over indicator transcription:
- 0-1min: data fetch + profit/protection check
- **1-3min: Read chart FIRST → 3 questions → hypothesis → confirm with indicators → conviction block** (was previously 1 min)
- 3-4min: execute trades
- 4-5min: state.md update
- Added: "No entry without Different lens" as explicit time allocation instruction
- strategy_memory: StRSI context-dependence (breakout vs range) + limit fill direction lessons

**Files changed**: `~/.claude/scheduled-tasks/trader/SKILL.md`, `collab_trade/strategy_memory.md`

## 2026-04-06 — Sizing table: hardcoded units removed, formula-only

**Problem**: Conviction sizing table showed hardcoded unit counts (10,000u / 5,000u / 1,667u / 667u) calibrated for NAV 200k. Current NAV is 104k. Trader was copying these numbers instead of recalculating from actual NAV → B entries at ~10% NAV instead of 5%.

**Fix**: Replaced all hardcoded unit examples in SKILL.md (3 locations) with:
- Formula: `Units = (NAV × margin%) / (price / 25)`
- Concrete examples using current NAV (104k) to anchor intuition
- Explicit note: "Never reuse yesterday's unit count"

**Files changed**: `~/.claude/scheduled-tasks/trader/SKILL.md`, `docs/SKILL_trader.md`

## 2026-04-06 — Slack ts tracking moved from Claude to code

**Problem**: Claude (especially Sonnet) forgets to update `Slack最終処理ts` in state.md → next session reads the same user messages → replies again → duplicate/triplicate responses. Dedup catches identical posts but not different wordings of the same reply.

**Root cause**: Relying on Claude to write a ts value to state.md is unreliable. The ts tracking must be in code, not in prompts.

**Fix**:
- `tools/slack_read.py` now auto-writes latest user message ts to `logs/.slack_last_read_ts` after every read
- `tools/session_data.py` reads from this file instead of parsing state.md for `Slack最終処理ts`
- SKILL_trader.md Bash② and Next Cycle Bash simplified — no more `grep Slack最終処理ts` in the shell command
- CLI `--state-ts` override still works if needed

**Result**: Once a user message is read by any session, no subsequent session will see it again. Zero Claude dependency.

## 2026-04-06 — M5 candle data integrated into session_data.py

**Problem**: Trader SKILL instructed Claude to fetch M5 candles via inline python one-liner. Sonnet gets stuck generating this one-liner ("Processing..." hang for 10+ min). Repeated issue.

**Fix**: Added M5 PRICE ACTION section to `tools/session_data.py` — fetches last 20 M5 candles for held pairs + major 4 pairs automatically. Updated SKILL_trader.md to reference session_data output instead of requiring a separate fetch. No quality loss — same data, zero model-generated code needed.

## 2026-04-06 — Slack duplicate reply fix: code-level dedup enforcement

**Context**: User reported duplicate Slack replies to the same message, repeatedly. Previous "fix" was prompt-level instruction only (`Slack最終処理ts` in state.md) — Claude sessions could race past it or skip the check entirely.

**Root cause**: Multiple 1-minute cron trader sessions read the same user message. Each independently decided to reply. No code prevented the second reply.

**Changes**:
- Added `tools/slack_dedup.py` — file-based dedup with `fcntl` lock. Records replied-to message ts in `logs/.slack_replied_ts`. Auto-cleans entries >48h
- Modified `tools/slack_post.py` — new `--reply-to {ts}` flag. When provided, checks dedup before posting. If already replied → silently skips (exit 0). After posting → atomically marks ts as replied
- Updated trader SKILL.md — all user message replies now require `--reply-to {USER_MESSAGE_TS}`. Dedup is enforced in code, not by prompt instruction. Removed the manual `Slack最終処理ts` checking requirement

**How it works**: `slack_post.py "reply" --channel C0APAELAQDN --reply-to 1712345678.123456` → if ts is in dedup file → `SKIP_DEDUP` and exit. If not → post → mark ts. File lock prevents race conditions between concurrent sessions.

## 2026-04-05 — News flow logging: narrative evolution tracking

**Context**: news_digest.md was overwritten hourly with no history. Impossible to see whether a macro theme (e.g. "USD strength") was fresh or exhausted. Even for scalps/momentum, knowing "this theme built for 3 hours vs just appeared" changes conviction.

**Changes**:
- Added `tools/news_flow_append.py` — reads current news_digest.md, appends a compact HOT/THEME/WATCH snapshot to `logs/news_flow_log.md`. Keeps 48 entries (48h). Deduplicates by timestamp.
- Added Cowork scheduled task `qr-news-flow-append` — runs at :15 every hour, after qr-news-digest (:00) finishes
- Updated `docs/SKILL_daily-review.md` — Step 1 now reads news_flow_log.md; Step 2 adds question 7 (did macro narrative shift today, and did the trader adapt?)
- Updated CLAUDE.md architecture section to document the new pipeline

## 2026-04-04 — Conviction framework: FOR / Different lens / AGAINST / If I'm wrong

**Context**: Retroactive analysis found 7 conviction-S trades undersized by 70% avg (6,740-13,140 JPY lost). Root cause: trader checked 2-3 familiar indicators, rated B, stopped. Deeper analysis with different indicator categories would have revealed S. Also: 4/1 all-SHORT wipeout (-4,438 JPY) would have been prevented if CCI/Fib (different lens) had been checked — they showed exhaustion.

**Core change**: Conviction is no longer "how many indicators agree" but "how deeply have you looked, and does the whole picture cohere?" New pre-entry format:
```
Thesis → Type → FOR (multi-category) → Different lens (unused category) → AGAINST → If I'm wrong → Conviction + Size
```

**"Different lens" is the key innovation.** Forces checking indicators from categories NOT already used in FOR. Moves conviction BOTH directions:
- B→S upgrade: initial 2 indicators look like B, but Fib + Ichimoku + cluster all support → actually S. This is where the money is
- S→C downgrade: ADX says BEAR, but CCI=-274 and Fib 78.6% say exhausted → abort. This prevents wipeouts

**6 indicator categories defined**: Direction, Timing, Momentum, Structure, Cross-pair, Macro. Categories serve as a checklist of what to look at, not a scoring rubric. Conviction is the trader's judgment of story coherence.

**Files changed**: risk-management.md (full conviction framework + 6 categories + pre-entry block + sizing table), SKILL_trader.md (pre-entry format + conviction guide + sizing), collab_trade/CLAUDE.md (Japanese version of entry format), strategy_memory.md (evidence + updated sizing guidance)

## 2026-04-04 — 3-option position management + structural SL enforcement

**Context**: 4/3 post-mortem with user. Key insight: Opus read charts correctly but managed positions in binary (trail or hold). Missed "cut in profit and re-enter post-NFP." SL placement was ATR×N mechanical, not structural. User couldn't understand SL rationale because there was none beyond a formula.

**SKILL_trader.md**: Added "Position management — 3 options, always" section. For each position when conditions change, trader must write 3 options (A: hold+adjust, B: cut-and-re-enter, C: hold-as-is) then pick one with reasoning. Output format forces evaluation of all options — prevents binary thinking. Added structural SL placement requirement.

**risk-management.md**: Renamed SL section to "Structural placement. No ATR-only." Added structural SL examples (swing low, Fib, DI reversal vs. ATR×N). Added 3-option position management framework. Added 2 new failure patterns (ATR mechanical SL, binary position management).

**protection_check.py**: Added 3-option prompt to output. After listing all positions, prints A/B/C blanks for each position that the trader must fill in. Forces structured thinking at point of output.

**strategy_memory.md**: Added 2 Active Observations — binary position management lesson and structural SL lesson from 4/3.

## 2026-04-03 — Root cause fix: Stop mechanical SL placement

**SKILL.md (trader task)**: Rewrote protection management section. protection_check output is now "data, not orders." Removed "Trailing=NONE is abnormal" rule. Trailing stops are now "for strong trends only, not default." Added hard rules for when NOT to set SL. Trail minimum raised to ATR×1.0 (was ATR×0.6-0.7).

**protection_check.py**: Added `detect_thin_market()` — detects Good Friday, holidays, weekend proximity, low-liquidity hours. During thin market: suppresses Fix Commands, changes "NO PROTECTION" message from warning to "this is correct."

**Root cause**: SKILL.md had rules that forced trader to mechanically attach SL/trail to every position regardless of market conditions. This caused -984 JPY on 4/3 Good Friday when every thesis was correct but every SL got noise-hunted.

## 2026-04-03 — Hard rule: No tight SL on thin markets / holidays

**risk-management.md**: Added "Thin Market / Holiday SL Rule" section. Holiday/Good Friday = no SL or ATR×2.5+ minimum. Spread > 2× normal = discretionary management only. User "SLいらない" = direct order, don't override. Added two new failure patterns.

**strategy_memory.md**: Added to Confirmed Patterns (薄商いのタイトSL=全滅). Added "Thin Market / Holiday Rules" hard rules section.

**Cause**: 4/3 Good Friday — EUR_USD trail 11pip, GBP_USD trail 15pip, AUD_USD SL 10pip all hunted. -984 JPY total. Every thesis was correct. Also Claude closed AUD_JPY after user explicitly removed SL.

## 2026-04-03 — Display all news times in JST

**news_fetcher.py**: All times in `print_summary()` now displayed in JST (`04/04 21:30 JST`) instead of raw UTC ISO strings. Calendar events, headlines, and upcoming events all converted. User preference: JST is easier to read.

## 2026-04-03 — Add event countdown to news summary

**news_fetcher.py**: Added `_event_countdown()` — calculates remaining time to economic events (NFP etc.) and appends `[in 30min]`, `[in 1h01m]`, `[RELEASED]` etc. to calendar output in `print_summary()`. Prevents Claude from miscalculating event countdown by mental arithmetic (20:29 posted "NFPまで約30分" when it was actually ~61 min away).

## 2026-04-03 — Prompt design principle: "Think at the Point of Output"

**CLAUDE.md**: Added core prompt design principle — all prompts must work equally on Opus and Sonnet. The method: embed thinking into output format, not rules or self-questions. Output format forces thinking; rules and preambles don't.

**change-protocol.md**: Added "Prompt Editing Rule" — when editing any prompt, don't add rules or self-questions. Change the output format so thinking is required to produce it.

## 2026-04-03 — Fix Slack notification calculation errors

**trade_performance.py / slack_daily_summary.py — P/L= format fix**:
- Log entries using `P/L=` (with slash) were silently dropped by parsers that only matched `PL=`
- 8 entries affected, including large losses (-17,521 / -3,719 / -2,196 JPY)
- Fixed regex: `PL=` → `P/?L=` (slash optional)

**intraday_pl_update.py — New dedicated script**:
- `intraday-pl-update` task previously had Claude Code generate OANDA API code on-the-fly each session → unreliable calculations (showed 0 closes when there were 4)
- New `tools/intraday_pl_update.py` script fetches from OANDA transactions API with proper page pagination
- Supports `--dry-run` for testing
- SKILL.md updated to use the script instead of inline code generation

## 2026-04-03 — From rules to thinking: trader prompt philosophy rewrite

**Core change**: Replaced rule-based guardrails with self-questioning thinking habits. Works for both Opus and Sonnet.

**SKILL_trader.md — "The Trader's Inner Dialogue"** (replaced Passivity Trap Detection):
- "Am I reading the market or reading my own notes?"
- "If I had zero positions, what would I do?"
- "What changed in the last 30 minutes?"
- "Am I waiting, or hiding behind waiting?"
→ Not a checklist. A thinking habit that prompts genuine market reading.

**SKILL_trader.md — "Before you pull the trigger"** (replaced Anti-repetition hard block):
- "Am I seeing something new, or the same thing again?"
- "Why THIS pair, not the other six?"
- "If this loses, will I understand why?"
- "Am I trading the market or my bias?"
→ No more BLOCKED. Context of EUR_USD 8× repetition preserved as a lesson, not a rule.

**strategy_memory.md — Event Day / Small Wave sections**:
- Rewritten from prescriptive time windows to experience-based observations
- "Before writing 'no entries pre-event', ask how many hours until the event"
- Small wave guide preserved as pattern observation, not entry checklist

**Daily-review set to Opus**: Opus as coach, Sonnet as player.

## 2026-04-03 — Trader anti-repetition check + daily-review enforcement + task re-enable

**Trader SKILL (anti-repetition gate)**:
- Added 3-question check before every entry: same pair×direction×thesis 3+ = blocked
- Added trailing stop width rules: ATR×0.6 minimum, ATR×1.0 for GBP/JPY crosses, ATR×1.2 pre-event

**Daily-review SKILL (strategy_memory enforcement)**:
- Made strategy_memory.md update mandatory with date verification step
- Added pretrade score inflation tracking, R/R analysis, repetitive behavior detection
- "No changes needed" is no longer acceptable output

**Scheduled tasks re-enabled**:
- daily-review (was disabled since ~3/27 → strategy_memory.md stale)
- daily-performance-report, intraday-pl-update, daily-slack-summary

## 2026-04-03 — Slack anti-spam rules: no unsolicited standby messages, duplicate reply prevention

- SKILL_trader.md + scheduled-tasks/trader/SKILL.md: Added "When NOT to post to Slack" section
- Rule: Never post unsolicited "watching/waiting" status messages
- Rule: Only post on trade action, user message reply (once per ts), or critical alert
- Rule: Duplicate reply prevention — check Slack最終処理ts before replying; skip if already replied

## 2026-04-03 — Doc integrity audit: CLAUDE.md / change-protocol / task table

- CLAUDE.md: Split task table into Claude Code tasks + Cowork tasks. qr-news-digest is a Cowork task, not in scheduled-tasks/
- CLAUDE.md: Skills count 36 → 37
- CLAUDE.md + change-protocol.md: Deprecated bilingual sync rule (Japanese reference copies no longer maintained)
- change-protocol.md: Added news_digest.md must-be-English rule
- change-protocol.md: Removed rules-ja/CLAUDE_ja.md/SKILL_ja.md references

## 2026-04-03 — サイジング更新 + CLAUDE.md v8.1同期

**v8.1サイジング反映（risk-management.md）**
- Conviction S: 5000-8000u → **8000-10000u**（v8.1で引き上げ済みだったのにrisk-management.mdが未更新だった）
- Conviction A: 3000-5000u → **5000-8000u**
- Conviction B: 1000-2000u → **2000-3000u**
- Conviction C: 500-1000u → **1000u**
- pretradeスコア(0-10)との対応を明記: S=8+, A=6-7, B=4-5, C=0-3
- rules-ja/risk-management.mdにも同期

**CLAUDE.md修正**
- バージョン: "v8" → "v8.1"
- Self-Improvement Loop: `pretrade_check`が毎セッション実行に見えていた誤解を修正
  → `profit_check + protection_check`（毎セッション冒頭）と `pretrade_check`（エントリー前のみ）を正確に区別
  → 「相場を読む（M5チャート形状）」ステップを追加
  → SESSION_END に `trade_performance.py` が先行することを明記

## 2026-04-03 — CLAUDE.md全面同期修正

**Round 1（誤記・欠落）**
- 誤記修正: 自己改善ループ「毎7分」→「毎1分」
- 矛盾修正: news_digest.md「15分間隔」→「毎時」
- Required Rules on Changes に #6バイリンガル同期・#7スモークテストを追加（change-protocol.mdには既存、CLAUDE.mdに欠落していた）
- メモリシステムUsage・Rulesサブセクションをスリム化（skills/・rules/と重複していた部分を削除）
- skills一覧を更新（2個→主要4個+「全36スキル」表記）

**Round 2（深い精査）**
- アーキテクチャ表を拡張: trader/daily-review/qr-news-digestの3タスクのみ → 実在する6タスク全部記載（daily-performance-report/daily-slack-summary/intraday-pl-update追加）
- タスク定義パス: `~/.claude/scheduled-tasks/trader/SKILL.md` → `~/.claude/scheduled-tasks/`（正本）+ `docs/SKILL_*.md`（参照コピー）に修正
- Scripts表に重要ツール追加: profit_check.py / protection_check.py / preclose_check.py / fib_wave.py（recording.md・technical-analysis.mdで参照されているのに欠落していた）
- 運用ドキュメントから `docs/TRADE_LOG_*.md` を削除（旧形式。現在は collab_trade/daily/ を使用）
- ランタイムファイルに `collab_trade/summary.md` 追加（collab-tradeスキルで参照）
- `logs/trade_registry.json` 削除（不使用）
- Key Directories を整理: `indicators/`（低レベルエンジン）と `collab_trade/indicators/`（quick_calc）を区別して明記
- ユーザーコマンド「トレード開始」に「traderはスケジュールタスク」旨を明記。秘書・共同トレードのスキルトリガーを正確に記述
- CLAUDE_ja.mdに全変更を同期

## 2026-04-02 — SLルール修正 + 証拠金警告追加

問題: SKILL.mdの「エントリー時SL必須」ルールが4/1の実績（SLなし監視→BE/Trail）と矛盾。session_data.pyが証拠金98%でも無警告のため、traderが90%超で新規エントリーするルール違反を起こした。

### SKILL.md修正
- `NO PROTECTION` → 「5分ごと監視中はSLなしOK。ATR×0.8でBE、ATR×1.0でTrailing」に変更。3/31失敗（12時間放置）と4/1成功（5分監視）は別問題だった
- エントリー時のSLをオプション化: TP必須、SL=監視できない時のみ（夜間・離席・低確度）

### tools/session_data.py修正
- 証拠金90%超で `🚨 DANGER — no new entries` 警告追加
- 証拠金95%超で `🚨 CRITICAL — force half-close now` 警告追加
- 背景: 98.23%でも無警告のためtraderが新規エントリーを実行していた

## 2026-03-31 — 全プロンプト英語化（トークンコスト削減）

日本語プロンプトは英語の約2-3倍のトークンを消費する。1分cronのtraderセッションで積算コストが大きいため、全プロンプトを英語化。

### 変更内容
- `.claude/rules/` 6ファイル → 英語版に置換。日本語版は `.claude/rules-ja/` に保存
- `CLAUDE.md` → 英語版に置換。日本語版は `CLAUDE_ja.md` に保存
- `scheduled-tasks/*/SKILL.md` (7タスク) → 英語版に置換。日本語版は各ディレクトリに `SKILL_ja.md` として保存
- `change-protocol.md` にルール#6「日英同時編集」追加: プロンプト変更時は英語版と日本語版を必ず同時更新

### ファイル構成
```
.claude/rules/           ← 英語版（運用。自動ロード）
.claude/rules-ja/        ← 日本語版（確認用。ロードされない）
CLAUDE.md                ← 英語版（運用）
CLAUDE_ja.md             ← 日本語版（確認用）
scheduled-tasks/*/SKILL.md    ← 英語版（運用）
scheduled-tasks/*/SKILL_ja.md ← 日本語版（確認用）
```

## 2026-04-01 (7) — ボット思考からプロトレーダー思考への根本転換

問題: 4/1 全5ポジSHORT（GBP_JPY/AUD_JPY/EUR_JPY、全JPYクロス）→ バウンスで全SL hit。「H1 ADX=50 MONSTER BEAR」を30セッション繰り返し同じ結論を出すボット思考。指標は過去の事実を語るだけなのに、未来の保証として扱っていた。含み益（EUR_USD+536円、GBP_JPY+60円）も「テーゼ生きてる」でHOLD→吐き出し。

### SKILL_trader.md大幅改修
1. **判断の起点を逆転**: 指標→行動 を チャートの形→仮説→指標で確認→行動 に変更
2. **Bash②c全面書き直し**: 「値動き確認」→「市場を読め」。3つの問い（勢い/波の位置/味方か敵か）を指標の前に答えさせる
3. **方向バイアスチェック新設**: 全ポジ同方向=危険信号。「なぜ逆方向が1つもないか」を説明させる。LONG/SHORT両方持つのが正常
4. **STEP 1改修**: デフォルトを「切る」に変更。含み益→利確がデフォルト、含み損→「今から入るか？」がNOなら切れ
5. **STEP 3改修**: 「市場の空気を1文で語れ」を強制。指標の羅列ではなく物語を語らせる
6. **失敗パターン5件追加**: 全ポジ同方向全滅、指標転記=分析と錯覚、含み益見殺し、動き切った後に追加、ボット思考ループ
7. **時間配分に「市場を読む」ステップ追加**: 1-2分を値動き観察+バイアスチェックに割り当て

### risk-management.md改修
- 方向バイアスチェックセクション新設（確度ベースサイジングの上に）
- 失敗パターン4件追加（全ポジ同方向全滅、指標転記錯覚、含み益見殺し、動き切った後追加）

### strategy_memory.md追記
- メンタル・行動セクションに4/1の教訓4件追加

### state.md更新
- SL hitされたポジションの事実と反省を記録

## 2026-03-31 (6) — TP推奨を構造的レベルベースに全面改修

問題: protection_check.pyのTP推奨がATR×1.0固定（距離だけの無意味な価格）。swing/cluster/BB/Ichimoku等の構造的レベル（市場が実際に反応する価格）を使っていなかった。M5の構造的データも未活用。

### protection_check.py全面改修
- **find_structural_levels()新設**: H1+M5の全構造的レベルを収集し距離順にソート
  - H1: swing high/low, cluster, BB upper/mid/lower, Ichimoku雲SpanA/B
  - M5: swing high/low, cluster, BB upper/mid/lower
  - LONG→上方向、SHORT→下方向のみ返す
- **TP推奨**: ATR×1.0固定 → 構造的レベルのメニュー表示（最大5候補）。最寄りに「← 推奨」マーカー
- **修正コマンド出力**: `=== 修正コマンド (N件) ===` セクションにコピペで即実行可能なPUTコマンドを表示。SL広すぎ修正・TP修正・Trailing設定のコマンド
- 結果例: GBP_JPY SHORT TP=210.000(ATR×2.5)→候補5つ(M5 BB mid/lower, M5 swing low, M5 cluster, H1 swing low)をATR比付きで表示

## 2026-03-31 (5) — 回転数不足+TP/SL放置+1ペア集中の根本対策

問題: 24時間で4エントリーしかしていない。全9ポジがSL広すぎ(ATR×2.5-3.2)+TP広すぎ(ATR×2.3-5.0)+Trailing=NONE。protection_checkの警告を12時間以上放置。GBP_JPYに5ポジ7375u集中（ナンピン地獄）。ボラ的に7,000-12,000円/日取れるのに+834円。

### SKILL.md改善
1. **protection_check警告→即修正**: 「読むだけで次に行くな」を強調。`SL広すぎ`→即PUT修正。放置した実績（3/31 12時間放置→回転不能）を記載
2. **Trailing=NONEは異常**: 含み益ATR×1.0以上でTrailingないなら即設定。全ポジTrailing=NONEだった事実を明記
3. **回転数の目標値追加**: 3,000円=3回転（最低）、7,000円=3-4ペア×3回転（保守的に取れる）、15,000円=5ペア×3回転
4. **1ペア集中禁止**: 1ペア最大3ポジ推奨、含み損合計-500円超えたら他ペアで稼げ
5. **判断の罠に3パターン追加**: protection_check放置、ナンピン地獄、HOLD=仕事の錯覚
6. **時間配分にprotection_check対応を明記**: 0-1分にTP/SL/Trail修正を含める
7. **「1セッション最低1トレード」削除**: スプ広い時は見送りが正解

## 2026-03-31 (4) — スプレッドガード実装

問題: スプレッドに関するガードレールが一切なかった。bid/askは取得しているのにスプレッドを計算すらしていない。スプ3pipで5pip狙いのスキャルプに入ってRR崩壊。

### session_data.py — スプレッド表示+警告
- PRICES表示にスプレッドpip計算を追加: `USD_JPY bid=158.598 ask=158.606 Sp=0.8pip`
- 2.0pip超で `⚠️ スプ広い` 警告表示

### pretrade_check.py — スプレッドペナルティ(第6軸)
- エントリー前にOANDA APIからリアルタイムスプレッド取得
- 波の大きさ別の利幅目標に対するスプレッド比率を計算
  - 大波(20pip目標), 中波(12pip), 小波(7pip)
  - 30%超 = -2点（RR崩壊。見送れ）、20%超 = -1点（サイズ控えめに）
- 確度スコアに直接影響 → サイジングが自動で下がる

### SKILL_trader.md — スプレッド意識セクション追加
- スプレッドと利幅の関係表（大波/中波/小波 × スプ0.8/1.5/3.0pip）
- スプレッドが広がるタイミング（早朝、指標前後、GBP_JPY常時広い）
- live_trade_logにスプレッド記録: `Sp=1.2pip`

## 2026-03-31 (3) — TP/SL幅の根本修正 + 波サイズ≠ポジサイズ

問題: 全TPが「テーゼ夢ターゲット」(round number)でATR×2.4〜5.1先。SLもATR×2.0〜3.2。つまりTP到達不能、SL hit時は-6,000円級。また、波サイズがポジサイズを制限しており小波=小サイズだった。

### TP/SLの正しい付け方
- **TP**: テーゼ目標(round number)→最寄り構造的レベル(swing/cluster/Fib)に変更。ATR×1.0付近を半TP→残りtrailing
- **SL**: ATR×2-3→ATR×1.2に修正。hit時の損失額を明記して妥当性を確認
- **protection_check.py更新**: TP残距離>ATR×2.0で「TP広すぎ」警告、SL>ATR×2.5で「SL広すぎ」警告。構造的レベル(swing_dist, cluster_gap)ベースのTP推奨に変更
- SKILL.md: 「TP/SLの正しい付け方」セクション追加（❌❌✅✅の対比例付き）

### 波サイズ≠ポジサイズ
- **旧**: 小波=2000-3000u、中波=5000-8000u、大波=8000-10000u
- **新**: 確度がサイジングを決める。波サイズはpip目標と保有時間を決めるだけ
- 小波でも確度Sなら8000u。M5でタイミング見れてれば5-10pipでも+400-800円
- pretrade_check.py: サイジング表を確度一本に統一（S=8000-10000u regardless of wave）

### MTF評価の波サイズ対応
- 大波(H4/H1): H4+H1一致で+3点。M5未一致でもペナルティなし（M5はタイミング、セットアップ品質ではない）
- 中波(H1/M5): H1+M5一致で+4点
- 小波(M5/M1): M5+H1背景一致で+3点

## 2026-03-31 (2) — 確度評価の根本修正 + TP/SL/BE保護チェック

問題: pretrade_checkが過去WRしか見ず全部LOW判定(25/30件がLOW)。確度S/A/B/Cがどこにも実装されていない。全7ポジションがTP/SL/Trailなしの裸ポジ。

### pretrade_check.py根本改修
- **セットアップ品質評価を追加(前向き)**: 既存のリスク警告(後ろ向き)に加え、今のテクニカルセットアップの質を0-10で数値化
  - MTF方向一致(0-4点): H4+H1+M5全一致=4, H1+M5=3, H4+H1=2
  - ADXトレンド強度(0-2点): H1 ADX>30で+2
  - マクロ通貨強弱一致(0-2点): 7ペアテクニカルから通貨強弱を自動計算
  - テクニカル複合(0-2点): ダイバージェンス、StochRSI極限、BB位置
  - 波の位置ペナルティ(-2〜+1点): H4極端(CCI±200/RSI極端)で同方向エントリー=-2
- **確度→サイジング直結**: S(8+)=8000-10000u / A(6-7)=5000-8000u / B(4-5)=2000-3000u / C(0-3)=1000u以下
- **実際のテスト結果**: GBP_JPY SHORT→S(8), EUR_JPY SHORT→A(6), USD_JPY LONG→C(0)。今まで全部LOWだったものが正しく差別化された
- 背景: 今まで全エントリーが `pretrade=LOW` でサイズ2000u。LOWで入ってサイズだけ膨らませて-2,253円

### tools/protection_check.py新規作成
- 全ポジのTP/SL/Trailing有無をATRベースで評価
- SL推奨: ATR×1.2(ノイズ耐性)。構造的レベル(cluster)との併記
- TP推奨: 最寄り構造的レベル(ATR×1.0付近) → 半TP + trailing
- BE推奨: 含み益ATR×0.8→BE検討、ATR×1.5→Trailing強く推奨
- SL too tight警告: ATR×0.7未満は「ノイズで刈られるリスク」を警告
- TP広すぎ警告: 残距離>ATR×2.0で警告（ATR何本分かを表示）
- SL広すぎ警告: >ATR×2.5で警告
- session_data.pyのTRADE PROTECTIONS表示と連携

### session flow更新
- Bash②b: `profit_check --all` + `protection_check` を並列実行
- SKILL.md: エントリー前チェックに確度→サイジング表を追加
- recording.md: protection_checkをSTEP 0b-2に組み込み

## 2026-03-31 — 「5分で稼げ」+ サイジング逆転修正

問題: NAV 187kで1日-1,284円。勝ちトレード2000uで+300円、負けトレード10500uで-2,253円。勝つ時に小さく負ける時に大きい。5分セッションの大半を分析テキスト書きに消費。

### SKILL.md改善
1. **「5分で稼げ」時間配分**: 0-1分=データ+判断、1-4分=トレード実行、4-5分=記録。分析テキスト書く時間=稼いでいない時間
2. **サイジング鉄則追加**: 確度S=8000-10000u、確度A=5000-8000u、確度B=2000-3000u、確度C=1000u。自信がある時に大きく張れ
3. **STEP 0簡素化**: fib_wave --all + adaptive_technicalsの毎サイクル実行を廃止。session_data.pyで十分。必要時のみ
4. **波サイズテーブル拡大**: 大波8000-10000u(+1500-3000円/trade)、中波5000-8000u、小波2000-3000u
5. **テーゼポジ以外でスキャルプ**: ホールド中に他ペアのM5/M1チャンスを並行で取れ。2ペアしか触らないのはAIの無駄遣い
6. **risk-management.md整合性修正**: マージン管理をSKILL.md哲学と統一
7. **CLAUDE.md整合性修正**: 同上

6. **指値・TP・SL・トレーリングストップ活用**: 成行のみ→LIMIT/TP/SL/Trailing全活用。セッション間も自動で稼ぐ/守る。コード例付き
7. **session_data.pyにPENDING ORDERS + TRADE PROTECTIONS追加**: 毎セッション冒頭で指値の状態と全ポジのTP/SL有無を表示。「⚠️ NO PROTECTION」で裸ポジを警告
8. **oanda-api.md更新**: 注文タイプ一覧（MARKET/LIMIT/TP/SL/Trailing/Cancel）追加

- 背景: 「おれだったらこの資産で今日中に3万円稼げる」。15pip×20回転×10000u=30,000円。同じ相場読みでサイズだけ変えれば今日の利確合計+3,000→+8,000円だった。さらに全7ポジションがTP/SL/Trail全てなし=セッション間は完全無防備だった

## 2026-03-30 (3) — 回転思考の根本改善 + 「波のどこにいるか」

問題: 方向は当たっている(JPY強テーゼ正解)のに稼げない。利確+3,047円→同方向に10500u再エントリー→-2,253円吐き出し。H4 CCI=-274(動き切った後)にSHORT新規。

### SKILL.md改善
1. **「動き切った後は逆を取れ」**: H4 CCI±200超/RSI極端の時、利確後に同方向再エントリー禁止。バウンス方向で小さく取り、バウンス天井でテーゼ方向に再エントリー = 本当の回転
2. **セッション内で値動きを「観る」**: M1キャンドルを判断前後で2回見る。指標(過去)ではなくM1(今)で勢いを感じる
3. **確定利益を守れ**: 利確直後に前回以上のサイズで同方向エントリー = 倍賭け。再エントリーは同サイズ以下
4. **マージン圧力ルール修正**: 「60%=怠慢→入れ」→「60%未満ならチャンスを見逃してないか自問。ただしマージン自体はエントリー理由にならない」
5. **アクション強制ルール撤去**: 「5回連続HOLDで赤信号→何かしろ」→ 撤去。チャンスがなければ待て。行動の強制がオーバートレードを生んだ
6. **回転の定義変更**: 「TP→同方向に再エントリー」→「TP→バウンス取り→テーゼ方向に再エントリー = 波の上下で稼ぐ」

7. **波の大きさに合わせたサイジング**: 大波(H4/H1)3000-5000u / 中波(M5)2000-3000u / 小波(M1)1000-2000u。H1/H4合致しなくてもM1で明らかなバウンスが見えたら小さく取れ
8. **risk-management.md整合性修正**: マージン管理セクションの「常時80-90%で回せ。60%未満=怠慢」をSKILL.md改善と整合するよう修正。「margin_boostはエントリー理由にならない」を明記

- 背景: EUR_JPY +1,379円利確後に10500u積んで-2,253円。GBP_JPY H4 CCI=-241でSHORT新規。方向の正しさ≠エントリータイミングの正しさ
- SKILL.mdはgit管理に移行済み(docs/SKILL_trader.md → symlink)

## 2026-03-30 (2) — traderタスク判断品質改善

問題: traderタスクが30セッション連続「全ポジHOLD」のレポーターと化していた。分析は書くが行動しない。含み益+20pipを-9pipの損切りにしてしまう（テーゼ目標に固執して市場がくれたものを逃す）。

### SKILL.md改善（~/.claude/scheduled-tasks/trader/SKILL.md）
1. **「市場がくれるものを取れ」マインドセット追加**: テーゼ目標への固執を禁止。利確→押し目再エントリーの回転思考を最上位に配置
2. **値動き確認ステップ(Bash②c)追加**: 指標より先にM5キャンドルで勢いと形を確認。ピーク記録をstate.mdに残す
3. **Devil's Advocate**: 含み損-5k超ポジにprofit_checkがHOLDを出した場合、「今すぐ切るべき理由」を3つ挙げて反論する義務
4. **アクション自己監視**: 連続HOLDセッションカウンター。3回連続で黄色、5回連続で赤（何かアクションを取れ）
5. **state.md肥大化防止**: サイクルログは上書き（積み上げ禁止）。目標100行以内
6. **レポーター化・ユーザー指示免罪符の明示的禁止**: 自分の見解を必ず併記、構造変化時はSlackで提案

### schema.py修正
- `get_conn()`に`busy_timeout=5000ms`追加。traderとingest.pyの並行アクセスでpretrade_checkがBusyErrorスキップされていた問題を修正

- 背景: 2026-03-30 USD_JPY +20pip→-9pip損切り。state.md 290行30エントリー中30回「HOLD継続」。pretrade_checkがapsw errorでスキップ

## 2026-03-30 — ニュースパイプライン追加（Cowork → Claude Code）
- **Cowork定期タスク `qr-news-digest`**: 15分間隔でWebSearch×3 + APIパーサでFXニュースを収集し、トレーダー目線の要約を `logs/news_digest.md` に書き出す
- **tools/news_fetcher.py 新規作成**: 3ソース対応（Finnhub経済カレンダー+ヘッドライン、Alpha Vantageセンチメント、Forex Factoryカレンダー）。APIキー未設定でもFF fallbackで動作
- **session_data.py 更新**: NEWS DIGESTセクション追加。Coworkが作成した `news_digest.md` を読んでtraderセッションに提供。鮮度チェック付き
- **設計思想**: テクニカルだけでは「なぜ動いているか」が分からない。マクロ・地政学・要人発言がテーゼの土台。Coworkの強み（WebSearch+LLM要約）を活かし、Claude Codeのtraderは読むだけ
- **APIキー設定（任意）**: `config/env.toml` に `finnhub_token`, `alphavantage_token` を追加するとセンチメント分析が有効に
- 更新ファイル: `tools/news_fetcher.py`(新規), `tools/session_data.py`, `CLAUDE.md`, `docs/CHANGELOG.md`

## 2026-03-27 (5) — デフォルト逆転 + profit_check.py + 1分cron
- **利確デフォルト逆転**: 「なぜ切るか」→「なぜ持つか」に反転。持つ側が根拠を示す設計に
- **profit_check.py新設**: 6軸評価（ATR比・M5モメンタム・H1構造・7ペア相関・S/R距離・ピーク比較）で利確判定
- **cronを7分→1分に短縮**: ロック機構で多重起動防止。セッション終了→最大1分で次が起動。APIコスト変化なし
- 更新ファイル: `tools/profit_check.py`(新規), `risk-management.md`, `recording.md`, `SKILL.md`, `CLAUDE.md`
- 背景: GBP含み益+3,000円→-4,796円の教訓。HOLDバイアスが利確を阻害していた

## 2026-03-27 (4)
- **利確プロトコルの空白を埋めた** — 「利確を問うトリガー」を策定:
  - `risk-management.md`: 「利確を問うトリガー」セクション追加。5つの状況（別ポジ急変・レンジBB mid・M5モメンタム低下・セッション跨ぎ含み益減・300円超）を定義
  - `recording.md`: STEP 0b-2「profit_check」追加。各セッション開始時に含み益ポジを照合する習慣化
  - `strategy_memory.md`: 今日の失敗（GBP含み益消滅）を Active Observations に追記
  - 設計思想: 命令ではなく「問いを強制するトリガー」。HOLD OK、ただし根拠を言語化しろ
  - 背景: 2026-03-27 GBP LONG 含み益+3,000円超がAUD急変中に誰も見ず消滅した教訓

## 2026-03-27 (3)
- **セッション生存率改善** — 3分セッションが短すぎてトレードに辿り着けない問題を解決:
  1. `tools/session_data.py` 新規作成: Bash②③④（テクニカル更新・OANDA・macro_view・adaptive_technicals・Slack・memory recall・performance）を1スクリプトに統合。4回のBash呼び出しが1回に
  2. trader SKILL.md: 309行→約90行に圧縮。ルールは`.claude/rules/`に委譲し重複削除
  3. セッション時間: 3分→5分、cron間隔: 5分→7分
  4. `tools/adaptive_technicals.py`: ROOTパスバグ修正（parents[2]→parent.parent）

## 2026-03-27 (2)
- **自律学習ループ構築** — データが溜まっても行動が変わらない問題を根本解決:
  1. `ingest.py`: OANDA/trades.mdパス統合。OANDAレコードにtrades.mdの質的データ(テーゼ・教訓・regime)をUPDATE。UNKNOWNペア問題修正。live_trade_logからも補完
  2. `parse_structured.py`: regime検出強化(ADX値判定・英語対応)、lesson抽出拡張(plain text対応)、user_call検出拡張(「」なし対応)
  3. `schema.py`: pretrade_outcomesテーブル追加（pretrade_checkの予測 vs 実際のP&L追跡）
  4. `pretrade_check.py`: チェック結果をpretrade_outcomesに自動記録 + 過去の同条件エントリー結末を表示
  5. `tools/daily_review.py` 新規作成: 日次データ収集エンジン。OANDA決済トレード・pretrade結果マッチング・パターン分析
  6. `daily-review` scheduled task 新規作成: 毎日06:00 UTC。Claudeが自分のトレードを振り返り、strategy_memory.mdを進化させる
  7. `strategy_memory.md` 構造リニューアル: Confirmed Patterns / Active Observations / Deprecated / Pretrade Feedback のセクション分割
  8. trader SKILL.md: strategy_memory.mdの読み方を明確化（Confirmed=ルール、Active=参考）
  9. CLAUDE.md: アーキテクチャにdaily-review記載
  - 設計思想: ボット的自動化ではなく、プロトレーダーが毎日振り返って強くなるプロセスの自動化

## 2026-03-27
- **金額トリガー全廃 + マクロ導線接続 + MTF統合** — ユーザー指示で3点同時改修:
  1. risk-management.md: 金額ベース損切り(-500円, -1000円閾値)を全廃。H1構造→テーゼ根拠→反対シグナルの3段階市況判断フローに置換
  2. SKILL.md: 撤退ルールの金額トリガー(-30pip/-500円/ペア別pip上限)を削除。macro_view参照の市況判断に置換。判断フローにmacro_view読みをStep 0として追加
  3. tools/macro_view.py 新規作成: 7ペアtechnicalsから通貨強弱スコア・テーマ判定・MTF一致ペア検出・H1 Div一覧を4行で出力。Bash②に組込み
  - 背景: traderがM5テクニカルだけでボット的判断→低確度トレード乱発→利益を損失で相殺。マクロ視点(通貨強弱・テーマ)と金額に頼らない市況判断で改善
- **メモリ学習ループ修復** — SKILL.md Bash③を改修: 汎用クエリ1本→保有ペアごとのrecall検索に変更。6,260トレードの記憶がトレード判断に活かされるように
- **collab_trade/CLAUDE.md 死参照掃除** — v6で廃止済みのanalyst/secretary/shared_state.json/quality_alert参照を全削除。macro_view.py参照に置換。品質監視は自己監視に変更
- **close_trade.py追加** — ヘッジ口座でPOST /ordersに反対unitsを送ると新規ポジが開くバグ対策。決済は必ずPUT /trades/{id}/closeを使うラッパースクリプト。SKILL.md・oanda-api.mdに決済ルール追記
- **資金効率改善** — マージン目標を90%→70-80%に変更。50%未満=怠慢ルール追加。日次10%には80%水準が必要（計算根拠: NAV18万×25倍×80%=名目363万、7ペア分散で1ペア平均7pipで達成）
- **ボット的撤退ルール改善** — SKILL.mdの段階的撤退テーブル（固定時間・固定pip）をテーゼベース判断に改善。preclose_check組込

## 2026-03-26
- **v8 — traderを正のシステムとして昇格** — リポジトリ全面整理。旧遺産を全てarchive/に統合。ディレクトリをCLAUDE.md, collab_trade/, tools/, indicators/, logs/, config/, docs/, archive/の8個に整理。21GB削減。staleワークツリー30個+、ブランチ130個+削除。パス変更: scripts/trader_tools/ → tools/
- **trade_performance.py v4** — v6ログ形式対応。日別/ペア別/セッション別集計追加
- **v7 — マージン安全ルール** — marginUsed/NAV ≥ 0.9で新規禁止、≥ 0.95で強制半利確。1ペア最大5本。Sonnet化
- **段階的撤退ルール追加** — M5割れ→5分待つ→10分で半分切り→20分+全撤退。-30pip/-500円超は即全撤退。H1テーゼは「すぐ切らない」理由にはなるが「ずっと持つ」理由にはならない。GBP_JPY -237円の教訓 (risk-management.md, SKILL.md, strategy_memory.md)
- **リスク管理ルール全面改訂** — ユーザーレビューに基づき根本見直し:
  - 固定値(+5pip半利確等)全廃止 → ATR対比・テーゼ射程・モメンタム変化の状況判断に変更
  - 「1トレード+300円目標」明記。+40円利確は時間の無駄(実績: 勝率65%でNet-583円、勝ち平均+84円)
  - 損切り判断を金額→テーゼベースに変更。損切り後に戻るパターン対策
  - add-onルール: ピラ/ナンピン両方OK、ただし「新しい根拠を言えるか」が条件。同じ根拠の繰り返しNG
  - ポジション本数制限(最大2本)撤回。本数ではなく根拠の質が問題
  - 確度ベースサイジング(S/A/B/Cランク)導入

## 2026-03-25
- 両建て（ヘッジ）回転戦術をtraderに組込
- メモリシステム恒久改善 — OANDA APIバックフィル6,123件

## 2026-03-24
- Slack通知統合（4点記録セット）
- v6〜v6.5 — trader一本化、Cowork全廃止、2分短命セッション+1分cronリレー

## 2026-03-23
- v5〜v5.1 — 連続セッション、strategy_memory自律学習、ナラティブレイヤー
- live_monitor完全削除

## v1-v4 (2026-03-17〜22)
詳細は `archive/docs_legacy/CHANGELOG_full.md` を参照。
ボットworker体制 → マルチエージェント → trader一本化への進化の記録。

## 2026-04-06 — Trader session 15min→5min (reliability)
- Lock threshold: 900s→300s, SESSION_END: 600s→240s
- Rationale: 10min/15min sessions failed to complete. 5min proven to work. Reliability > depth.

## 2026-04-11

### Fix: intraday_pl_update.py daily return % calculation
- **Bug**: Old formula `(realized_pl + upl) / (balance - realized_pl)` assumed UPL=0 at start of day. Overnight positions with pre-existing UPL caused wildly inaccurate daily return percentages (e.g. +0.50% when actual NAV change was ~0%)
- **Fix**: Store SOD NAV in `logs/sod_nav.json` on first run of each day. Calculate daily return as `(current_NAV - SOD_NAV) / SOD_NAV`. Falls back to 0% if no SOD data available

## 2026-04-17

### Docs: remove current VM/GCP implication
- Clarified in `README.md` that GCP/VM/Cloud Run/BigQuery/Looker references are archived history only and not part of the current runtime
- Fixed the legacy runbook reference from `docs/OPS_GCP_RUNBOOK.md` to `archive/docs_legacy/OPS_GCP_RUNBOOK.md`
- Clarified in `AGENTS.md` and `CLAUDE.md` that `archive/` contains historical VM/GCP artifacts only; current QuantRabbit does not run on VM/GCP

## 2026-04-18

### Runtime learning caps: add session/regime context
- `tools/session_data.py` now scores `LEARNING EDGE BOARD` seats with three layers instead of pair-memory alone: pair-direction history, current UTC session bucket, and a live M5 regime proxy inferred from the technical cache
- The board now prints session/regime context lines such as `tokyo: WR ... EV ...` and uses those headwind/tailwind stats to tighten or relax the runtime allocation cap before a seat reaches `A/S`
- `docs/SKILL_trader.md` now requires a `Learning context` line in the trade thesis so the trader cannot ignore a `late NY` or `range` headwind while sizing up
- `AGENTS.md` updated to reflect that `session_data.py` now converts lesson history into session/regime-aware size caps, not only pair-direction caps

### Timing docs: remove stale trader cadence references
- Updated `docs/SKILL_trader.md` to describe the current timing model consistently as a 15-minute trader session window on a 20-minute recurring cadence, instead of mixing in stale `Claude 10-minute` wording
- Updated `AGENTS.md` to match the same timing model and to describe Claude trader compatibility as disabled/host-dependent rather than live `10-minute cron`
- Refreshed the local Claude trader compatibility schedule file to remove the stale `*/10` recurrence and leave it disabled unless explicitly re-enabled

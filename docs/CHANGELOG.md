# Changelog

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

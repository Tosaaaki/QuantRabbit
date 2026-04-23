---
name: daily-review
description: Daily review — reflect on trades and evolve strategy_memory.md
---

You are the same pro trader who was trading during the day. Now it's time to reflect.
Review today's trades, find patterns, and **write concrete updates to strategy_memory.md**.

**This task exists for ONE reason: to make the trader session smarter tomorrow.** If strategy_memory.md doesn't change, this task failed.

## Step 1: Collect Data

Set the review-day variable once. **This task reviews the most recent completed UTC trading day**, not the in-progress local calendar day at 15:00 JST.

```bash
REVIEW_DAY=$(python3 - <<'PY'
from datetime import datetime, timedelta, timezone
print((datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat())
PY
)
```

Bash①: Gather the review day's data with daily_review.py

cd /Users/tossaki/App/QuantRabbit && python3 tools/daily_review.py --date "$REVIEW_DAY"

Bash②: Verify user market calls (compare predicted direction vs actual price movement)

cd /Users/tossaki/App/QuantRabbit && python3 tools/verify_user_calls.py

This verifies all unverified user calls (outcome=NULL, >2h old) by fetching actual prices from OANDA.
Results feed directly into pretrade_check accuracy stats. Include any notable findings in Step 2.

Read: collab_trade/strategy_memory.md (current knowledge)
Read: collab_trade/memory/lesson_registry.json (current lesson states / trust snapshot)
Read: collab_trade/daily/$REVIEW_DAY/trades.md (that UTC day's records, if any)
Read: logs/live_trade_log.txt (last 100 lines — for details daily_review.py might miss)
Read: logs/s_hunt_ledger.jsonl (that UTC day's raw receipts — `daily_review.py` now syncs these into `memory.db` `seat_outcomes` first)

## Step 2: Reflect (Think with your own head)

Read the output of daily_review.py and think through the following:

1. **Execution ownership split first**: Read the `Execution Split` section before anything else. It uses explicit tags/comments first, then the reviewed UTC day's `trades.md` as the fallback ownership proof when a clean recurring-trader trade forgot to persist a tag.
2. **Memory promotion gate second**: Read the `Memory Promotion Gate` section and treat it as the evidence filter for today's edits. Pair/direction lessons in `strategy_memory.md` must come from the allowed recurring-trader evidence set, not from quarantined execution.
3. **Lesson state suggestions third**: Read the `Lesson State Suggestions` section. Use it as a registry-backed queue for what should be promoted to watch/confirmed or cut back to watch/deprecated.
4. **Bayesian evidence fourth**: Read the `Bayesian Evidence Update` section. Treat today as evidence against or in favor of an existing prior, not as an excuse to rewrite memory from one print.
5. **Pretrade feedback fifth**: Read the `Pretrade Feedback Notes` section. `daily_review.py` now writes concise machine notes back into `pretrade_outcomes.lesson_from_review`; if a note correctly identifies a repeat-loss thesis or a vehicle failure, turn that into cleaner `strategy_memory.md` language.
6. **AAR queue sixth**: Read the `After Action Review Queue`. For the biggest win, biggest loss, and repetition candidate, fill `Planned / Actual / Gap / Next hypothesis`.
7. **Today's winning patterns**: Why did you win? Is it reproducible?
8. **Today's losing patterns**: Why did you lose? Could it have been avoided?
9. **Pretrade score accuracy**: HIGH/S-score entries that lost = score is inflated. LOW entries that won = score is too conservative. **Track this.**
10. **Repetitive behavior**: Did the trader enter the same pair × same direction × same thesis more than 3 times? That's bot behavior, not trading
11. **Trailing stop effectiveness**: How many positions were closed by trailing stops? Were the trail widths appropriate vs ATR?
12. **R/R ratio**: Compare average win size vs average loss size. If losses > wins, the sizing or hold duration is wrong

## Step 2.5: Audit Accuracy Review

Read: `logs/audit_history.jsonl` (today's entries — append-only audit opportunity history: scanner facts + final narrative picks)

**audit_history.jsonl format** (append-only JSON lines):
```json
{"timestamp": "2026-04-09T01:33Z", "source": "s_scan", "s_scan": [
  {"pair": "EUR_USD", "direction": "SHORT", "recipe": "Squeeze-S", "status": "NOT_HELD", "price_at_detection": 1.16518}
], "positions": 1, "margin_pct": 38.0}
{"timestamp": "2026-04-09 01:35 UTC", "source": "narrative", "narrative_picks": [
  {"pair": "GBP_JPY", "direction": "SHORT", "edge": "A", "allocation": "B", "entry_price": 215.44, "tp_price": 215.22, "held_status": "NOT_HELD"}
], "strongest_unheld": {"pair": "GBP_JPY", "direction": "SHORT", "edge": "A", "allocation": "B"}}
```

Key fields:
- `source="s_scan"` → raw scanner facts. `recipe` identifies which S-scan recipe fired. `price_at_detection` is the M5 close when the signal fired. `status` is ALREADY_HELD / HELD_OPPOSITE / NOT_HELD.
- `source="narrative"` → auditor's final market view. `narrative_picks` are unheld actionables with separate `edge` and `allocation`.

**For each audit opportunity that fired today:**
1. Get current price for that pair from OANDA: `GET /v3/accounts/{acct}/pricing?instruments={PAIR}`
2. Calculate pip movement from detection/entry price to current price
3. Cross-reference with `live_trade_log.txt` — was the pair entered after this signal? P&L?
4. Verdict: CORRECT (moved +5pip in predicted direction) / PREMATURE (moved against, then correct) / WRONG (moved against)
5. Do **not** merge two same-pair / same-direction calls if their timestamps differ. Score each timestamped signal window separately.

**Write findings in strategy_memory.md → Active Observations, grouped by source:**
- `[date] Trend-Dip: fired 3×. Entered 2. Results: +180, -50. Accuracy: 2/3.`
- `[date] Squeeze-S: EUR_USD SHORT @1.16518. NOT entered. Price→1.16380 (+13.8pip in 4h). CORRECT.`
- `[date] Counter: USD_JPY SHORT @158.856. Entered. Lost -200 JPY. H4 extreme persisted. PREMATURE.`
- `[date] Narrative strongest unheld: GBP_JPY SHORT Edge A / Allocation B @215.44. NOT entered. Price→215.12 (+32pip). CORRECT.`

**Recipe scorecard** — keep a running tally in strategy_memory.md:
```
| Recipe | Fires | Entered | Correct | Accuracy | Status |
|--------|-------|---------|---------|----------|--------|
| Trend-Dip | 12 | 8 | 9/12 | 75% | Confirmed |
| Counter | 6 | 4 | 2/6 | 33% | Watch |
```
After 10+ data points: >60% accuracy → Confirmed (size up). <40% accuracy → Deprecated (remove from scan).

For narrative picks, track the same way in prose: strongest-unheld A/S calls, whether the trader acted, and whether the direction was right. The point is to learn whether the auditor's judgment is useful, not just whether the scanner recipe fired.

## Step 2.6: S Hunt Capture Review

Read: `logs/s_hunt_ledger.jsonl` (today's raw rows — append-only horizon receipts from each trader session)

`s_hunt_ledger.jsonl` captures what the trader claimed was the best short-term / medium-term / long-term promoted S, plus whether that horizon ended as entered, armed, or `dead thesis because no seat cleared promotion gate`. `daily_review.py` now syncs those receipts into `memory.db` `seat_outcomes`, so the review output includes a formal `discovered / orderable / deployed / captured / missed` chain instead of ad-hoc JSONL parsing.

For each horizon that appeared today:
1. First classify whether the horizon was a true promoted S (`entered id=...`, `armed STOP`, `armed LIMIT`) or an honest no-promotion close (`dead thesis because no seat cleared promotion gate: ...`).
2. Cross-reference with OANDA closed trades and `live_trade_log.txt` — did the seat turn into real P&L?
3. Compare the reference price versus the best favorable excursion between the state timestamp and the review cutoff. If the tape moved enough in the intended direction but the book stayed flat, count it as a capture miss or a missed promotion, not as "no chance".
4. Apply the same best-favorable standard to scanner / audit signals inside each signal window. Current price or flat close alone understates opportunity loss when the tape moved first and faded later.
4. Write one distilled observation about the failure mode:
   - discovery was wrong
   - vehicle was right but deployment was absent
   - closest seat was right but never cleared the promotion gate
   - deployment existed but sizing was too small
   - long-term thesis existed but was killed too early

Write findings in strategy_memory.md → Active Observations with concrete language:
- `[4/18] Short-term S found the right EUR_USD SHORT, but it closed as no seat cleared promotion gate and price still moved +11 pip. Vehicle right, promotion absent.`
- `[4/18] Medium-term S armed the correct EUR_JPY SHORT as a STOP order and captured +420 JPY. Horizon stacking worked when the receipt contained a real order id.`
- `[4/18] Long-term S kept closing as no-promotable-seat even when the medium-term continuation was already proving. Long-term blindness, not no-opportunity.`
- Use the `Horizon Scoreboard` counts in `daily_review.py` as the official chain totals for today: `discovered`, `orderable`, `deployed`, `captured`, `missed`. Continuing same-pair seats count once, so repeated state snapshots do not create fake extra deployments.

## Step 2.7: S Excavation Review

Read: `daily_review.py` `## S Excavation Review` section

`daily_review.py` now reads `S Excavation Review` from the formal `memory.db` `seat_outcomes` sync, not from ad-hoc JSONL parsing. That means near-S podium seats are reviewed with the same persistent receipt model as `S Hunt`, and a seat that later opened and was still live at review time still counts as `LATER DEPLOYED / OPEN`, not as a fake non-deployment.

`S Excavation Matrix` is the near-S board: seats that were close enough to matter but did not get promoted into the short / medium / long `S Hunt` board. The review goal is to learn whether those near-S seats were real edge that went undiscovered, or just noise that was correctly blocked.

For each podium seat:
1. Check whether it later became a real trade the same day, including seats that later opened and were still live at review time.
2. If it never deployed, compare its reference price versus the UTC day close.
3. Write one distilled observation:
   - blocker was valid, seat stayed noise
   - blocker was valid, but the seat later upgraded and was deployed
   - blocker was too conservative; the seat was right without deployment
   - podium named the wrong vehicle even though the horizon board later found the right one

Write findings in `strategy_memory.md` → `Active Observations` with concrete language:
- `[4/18] Podium #1 GBP_USD SHORT was right without deployment. The blocker was trigger quality, not direction quality. Next session should arm the stop-entry sooner when the reclaim shelf is the only missing piece.`
- `[4/18] Podium #2 EUR_USD LONG stayed wrong into the UTC close. The blocker was real; this was not hidden S, just a floor idea that never reclaimed.`

## Step 3: Update strategy_memory.md (MANDATORY — every section must be touched)

**You MUST edit strategy_memory.md. "No changes needed" is not acceptable.** At minimum:
- Update the `最終更新` date at the top
- Add at least 1 new Active Observation from today
- Update Per-Pair Learnings for any pair that was traded today
- Add pretrade feedback entries for HIGH-score losses and LOW-score wins
- Add at least 1 Active Observation from `S Hunt Capture Review` when the ledger shows a miss or a successful horizon stack

### Promotion gate rules

- If a pair/direction appears only under `Quarantine this evidence`, do **not** promote it as a recurring-trader pair lesson today.
- If quarantine execution reveals something real, convert it into a process / tooling / memory-hygiene lesson, not a pair-edge lesson.
- Every new pair/direction lesson should be defensible with the `Promote into recurring trader memory only from this clean evidence` list.
- Use `lesson_registry.json` to keep the lesson state machine coherent: `candidate -> watch -> confirmed -> deprecated`. Do not promote by prose mood alone.
- Use `Bayesian Evidence Update` to describe whether today supported or contradicted an existing prior. One trade can change the next day's caution level, but it should not instantly rewrite a confirmed lesson into a new worldview.

### What to write in each section

**Confirmed Patterns** (promote from Active Observations when verified 3+ times):
- Pattern + concrete examples with dates + why it works

**Active Observations** (new insights — include date and verification count):
- `[4/2] EUR_USD LONG entered 8 times on same "USD weak" thesis → WR 43%. Repetition ≠ edge. Verified: 1x`
- `[4/2] Trail 9-11pip on EUR_USD clipped 3 positions that later went to TP. Trail too tight vs ATR(16pip). Verified: 1x`

**Per-Pair Learnings** (pair-specific behavioral patterns):
- Update with today's data. Add new observations

**Pretrade Feedback** (track score accuracy):
- `[4/2] USD_JPY SHORT pretrade=HIGH(10) → -1,929 JPY. H4 was already extreme oversold. Score inflation.`
- `[4/2] EUR_USD LONG pretrade=HIGH(8) → +416 JPY. Score was accurate.`

**Indicator Combination Learnings** (what worked / didn't work):
- Any new combination tested today

### Writing rules
- **Use concrete examples**: "4/2 EUR_USD LONG 1500u pretrade=HIGH → -98 JPY. Trail 9pip hit = ATR×0.56, too tight"
- **Don't stop at statistics**: Write a hypothesis for why it happened
- **Promotion to Confirmed Patterns**: Only patterns consistently confirmed 3+ times
- **Moving to Deprecated**: Move disproven patterns with reasoning
- **Keep under 300 lines**: Distill. Bloat is laziness

## Step 4: Verify the update was written

Bash②: Confirm strategy_memory.md was actually modified

cd /Users/tossaki/App/QuantRabbit && head -3 collab_trade/strategy_memory.md && echo "---" && wc -l collab_trade/strategy_memory.md && echo "---last_modified:" && stat -f "%Sm" collab_trade/strategy_memory.md

**If the 最終更新 date is not today, you failed. Go back to Step 3.**

## Step 5: Re-ingest

Bash③: Re-ingest that reviewed UTC day with enriched ingest

cd /Users/tossaki/App/QuantRabbit/collab_trade/memory && python3 ingest.py "$REVIEW_DAY" --force 2>/dev/null; echo "ingest done"

This re-ingest now also enforces the lesson-state markers in `strategy_memory.md` from the registry review queue, then rebuilds `lesson_registry.json` from the updated markdown before embedding the fresh lesson chunks.

## Step 6: Slack Report

Bash④: Post review results to Slack (in Japanese)

cd /Users/tossaki/App/QuantRabbit && python3 tools/slack_post.py "📖 Daily Review完了。strategy_memory.md更新済み。" --channel C0APAELAQDN 2>/dev/null || echo "slack skip"

## Step 7: Keep main clean

Bash⑤: Auto-commit/push the reviewed memory files back to `main`

cd /Users/tossaki/App/QuantRabbit && python3 tools/runtime_git_sync.py sync-daily-review --date "$REVIEW_DAY"

This helper is intentionally strict. It stages only `collab_trade/strategy_memory.md` and `collab_trade/memory/lesson_registry.json`, and it skips entirely if any unrelated dirty path exists. If it prints `RUNTIME_GIT_SYNC_SKIP`, leave the repo alone and investigate what else is dirty instead of force-adding files.

## Absolute Rules
- **strategy_memory.md MUST be modified every run.** If no trades happened, still update the date and note "no trades"
- Don't write a stats report. Write a pro trader's journal
- Don't stop at "win rate 65%". Write whether expectancy was actually positive, why, and what must change
- If strategy_memory.md exceeds 300 lines, distill old and duplicate content to shorten it
- Don't casually delete existing Confirmed Patterns. If disproven, move to Deprecated

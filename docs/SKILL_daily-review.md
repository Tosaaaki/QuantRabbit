---
name: daily-review
description: Daily review — reflect on trades and evolve strategy_memory.md
model: claude-opus-4-6
---

You are the same pro trader who was trading during the day. Now it's time to reflect.
Review today's trades, find patterns, and **write concrete updates to strategy_memory.md**.

**This task exists for ONE reason: to make the trader session smarter tomorrow.** If strategy_memory.md doesn't change, this task failed.

## Step 1: Collect Data

Bash①: Gather today's data with daily_review.py

cd /Users/tossaki/App/QuantRabbit && python3 tools/daily_review.py --date $(date -u +%Y-%m-%d)

Read: collab_trade/strategy_memory.md (current knowledge)
Read: collab_trade/daily/$(date -u +%Y-%m-%d)/trades.md (today's records, if any)
Read: logs/live_trade_log.txt (last 100 lines — for details daily_review.py might miss)

## Step 2: Reflect (Think with your own head)

Read the output of daily_review.py and think through the following:

1. **Today's winning patterns**: Why did you win? Is it reproducible?
2. **Today's losing patterns**: Why did you lose? Could it have been avoided?
3. **Pretrade score accuracy**: HIGH/S-score entries that lost = score is inflated. LOW entries that won = score is too conservative. **Track this.**
4. **Repetitive behavior**: Did the trader enter the same pair × same direction × same thesis more than 3 times? That's bot behavior, not trading
5. **Trailing stop effectiveness**: How many positions were closed by trailing stops? Were the trail widths appropriate vs ATR?
6. **R/R ratio**: Compare average win size vs average loss size. If losses > wins, the sizing or hold duration is wrong

## Step 3: Update strategy_memory.md (MANDATORY — every section must be touched)

**You MUST edit strategy_memory.md. "No changes needed" is not acceptable.** At minimum:
- Update the `最終更新` date at the top
- Add at least 1 new Active Observation from today
- Update Per-Pair Learnings for any pair that was traded today
- Add pretrade feedback entries for HIGH-score losses and LOW-score wins

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

Bash③: Re-ingest today's data with enriched ingest

cd /Users/tossaki/App/QuantRabbit/collab_trade/memory && python3 ingest.py $(date -u +%Y-%m-%d) --force 2>/dev/null; echo "ingest done"

## Step 6: Slack Report

Bash④: Post review results to Slack (in Japanese)

cd /Users/tossaki/App/QuantRabbit && python3 tools/slack_post.py "📖 Daily Review完了。strategy_memory.md更新済み。" --channel C0APAELAQDN 2>/dev/null || echo "slack skip"

## Absolute Rules
- **strategy_memory.md MUST be modified every run.** If no trades happened, still update the date and note "no trades"
- Don't write a stats report. Write a pro trader's journal
- Don't stop at "win rate 65%". Write "why 65%, and what to change to hit 70%"
- If strategy_memory.md exceeds 300 lines, distill old and duplicate content to shorten it
- Don't casually delete existing Confirmed Patterns. If disproven, move to Deprecated

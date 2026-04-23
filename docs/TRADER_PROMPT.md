# Trader Mental Model

This document is about judgment only.

- Runtime cadence, required bash steps, and session-end rules live in `docs/SKILL_trader.md`
- Manual collaborative mode lives in `collab_trade/CLAUDE.md`
- Do not use this file as a scheduler or task contract

## Market First

- Read price behavior before you lean on indicator labels
- Write the market story in plain language: who is pressing, who is fading, and whether momentum is expanding or exhausting
- A checklist can confirm a thesis, but it must not create one

## Passes Need Triggers

- `PASS` is valid only if it includes a next trigger and an invalidation level
- “Nothing there” is not analysis
- If all seven pairs are unusable, say which pair was closest and what would make it tradeable

## Benchmark Discipline

- Compare realized P&L with day-start NAV every session
- Being behind raises the quality bar; it does not justify weaker entries, larger size, or late-tape chasing
- If the tape is dirty, patience is a position

## Existing Positions

- Ask every session: “If I were flat, would I enter here now?”
- If the answer is no, reduce or close unless you can name the exact condition that still justifies holding
- A live thesis must describe what changed since the last session and what would prove it wrong now
- Same-pair opposite-side exposure is allowed only with a role map: hero thesis, hedge or range-rotation leg, invalidation for both sides, and the condition that collapses the pair back to one side

## New Entries

- Thesis: what the market is doing now and why this pair is the best vehicle
- FOR: use multiple categories, not one indicator cluster
- Different lens: force a second read from a different category
- AGAINST: name the real contradiction
- Wrong if: write the concrete failure path and level

## Edge And Allocation

- Keep edge quality and capital allocation separate
- Strong edge with poor timing is not full size
- Strong timing with weak structure is not full size
- Size up only when structure, timing, macro context, and invalidation are all coherent

## Exit Discipline

- Chart first, metric second
- Band-walk trend: hold or scale, do not auto-cut at the first profit threshold
- Deceleration: reduce
- Reversal evidence: take the trade off
- Break-even is not “free” if it gives back most of the unrealized edge

## Pending Orders

- A pending LIMIT is not a timer. Ask whether you still want that exact price now
- If the thesis is alive but the old price has become a wish, reprice only when the tighter entry still leaves real spread-adjusted payout and a valid stop
- Do not reprice for a few pips of noise or because waiting feels uncomfortable; that is churn, not discretion
- Every pending review ends as LEAVE, REPRICE, EXTEND, or CANCEL with the market reason written plainly

## Audit Usage

- `quality-audit` is a second set of eyes for market direction, not a permission system for direction
- But unresolved live-book hygiene drift is a hard block for fresh risk: if a live trade/order is missing from `state.md` or `live_trade_log.txt`, fix that before any new entry
- If you disagree with the audit, name the exact chart or market fact you disagree with
- If you cannot name the contradiction, you do not really disagree

## Handoff Standard

- Update `collab_trade/state.md` immediately after any trade, thesis change, or important non-action
- Record the next trigger map, not just what already happened
- The next session should be able to resume without guessing

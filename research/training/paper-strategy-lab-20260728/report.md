# Paper strategy-lab and shared-feed challenger — 2026-07-28

## What changed

The Paper loop now converts a sufficiently large, high-confidence loss cause
into one reviewed declarative sibling strategy. It does not generate arbitrary
Python. New-data hash, candidate hash, minimum impact/sample, 24-hour cooldown,
one-candidate-per-day budget, two-active-candidate cap, independent account
identities and Paper-only authority are fail-closed gates.

The first cause is counter-trend SHORT concentration: -JPY 852.88 over 49
BASE/STRESS settlements in the available partial-seven-day evidence. The cost
arms overlap; the count is gate evidence, not an independent-sample claim.
The proposed sibling is `pullback_limit`: wait for a pullback, then enter only
with the direction of 1,441 completed M1 closes.

## Forward Paper experiment

Experiment `paper-pullback-sibling-20260728-v1` has six isolated JPY 50,000
lanes:

- range-fade champion, BASE/STRESS;
- 24h-direction-limited range inventory, BASE/STRESS;
- trend-pullback challenger, BASE/STRESS.

The executor fetches one read-only quote batch and fans the exact same bytes to
all active lanes. Each lane has a separate VirtualBroker, positions, orders,
balance, hash-chained ledger and DD state. Challenger DD at 5% cancels its
virtual orders, closes its virtual positions and disables that lane. There is
no real broker order client, live promotion, or DOJO dependency.

Admission:

- candidate hash:
  `44322c7db119a4b4411cee59ca1f2eb64cd92f1e84bb89ecca1b88fdc3c8e332`
- evidence hash:
  `2327df2b0c447cb279422ad504bde09bcbffae31aedbf24788a7f9b122bdd92d`
- idempotency hash:
  `039fd79e58b79bc73f15d1343da1d174a521e019d056dbcff8c5e4bbbe1e6d14`
- status: `ADMIT_PAPER_SHADOW`

## Decision boundary

This is an admitted experiment, not an adopted strategy. Continue only after
at least 30 settlements, cost-adjusted PF > 1, positive expectancy, DD no worse
than champion, at least two profitable regimes, BASE/STRESS agreement and an
exact shared-feed-chain match. Low fills are insufficient evidence, not
success. No claim of future profit is made.

The recurring AI model supervisor and fresh strategy-lab task scheduler are
not yet enabled. The first selection is a deterministic reviewed-template
decision derived from the measured loss cause.

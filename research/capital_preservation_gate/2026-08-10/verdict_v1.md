# CAPITAL_PRESERVATION_GATE_V1 verdict

Status: **ADOPT AS RESEARCH CAPITAL FLOOR; PROFIT EDGE NOT PROVEN; LIVE WIRING FORBIDDEN**

## Outcome

The new gate closes the unsafe path where unavailable cost, margin, unwind, or
loss-bound evidence could be interpreted as harmless. On the frozen 251
decision-time rows it produced `WAIT=251`, `TRADE=0`. Therefore this dry-run
policy took no new exposure and had zero realized loss and zero drawdown.

That is capital preservation, not a profitable strategy. It must not be reported
as proof that the market can be traded without losses. The legacy episode labels
sum to `-18,039.7866 JPY`, but that number remains diagnostic because DAILY_FINANCING
and partial-close allocation defects are known. Skipping those historical rows
does not establish that future skipped rows would lose.

## Why every row waited

- causal fee/slippage/financing completeness: missing on 251/251;
- decision-time margin/exposure completeness: missing on 251/251;
- executable unwind completeness: missing on 251/251;
- decision-time equity, peak equity, non-refillable daily loss spend, and a
  candidate worst-case loss bound were not present in this frozen ledger;
- a TRAIN-fixed positive after-cost LCB was not supplied to the gate.

Pricing was additionally missing on 97 rows and fillability on 98 rows. Missing
is not zero and does not grant permission.

## What is implemented

- equity-derived per-trade risk cap: `equity × 0.25%`;
- non-refillable daily gross-loss budget: `equity × 1%`;
- drawdown lock: `5%` from the supplied campaign peak;
- positive after-cost LCB requirement;
- complete pricing/order/fill/cost/margin/unwind evidence requirement;
- deterministic receipt hash, reason codes, and explicit reopening conditions;
- `MANAGE` path for an already-open position and no live permission in any dry-run receipt.

The percentages are preregistered policy fractions; JPY caps are calculated from
the decision-time equity input rather than hardcoded amounts.

## Verification

- unit/regression tests: 9/9 PASS;
- independent replay oracle: 9/9 PASS;
- frozen cohort: 251 unique decisions and 251 receipts;
- realized outcome fields used by decision: 0;
- live permission granted: 0;
- holdout, live, Paper, broker mutation, orders, and deploy: unused.

The regression test blocks the legacy failure shape—missing costs, margin, and
unwind. A separate positive fixture proves a fully evidenced, positive-LCB,
bounded-risk decision can still return `TRADE`; the guard is not an unconditional
ban.

## Reopening condition

Forward acquisition must persist, before every decision, broker equity and peak,
non-refillable gross-loss spend, side-correct bid/ask, fillability, complete cost
schedules, available/used margin and exposure, executable exit/unwind evidence,
and a causal worst-case loss bound. A TRAIN-fixed edge estimate then determines
whether the capital floor permits a bounded trade. Until those fields exist,
`WAIT` is the correct safety result.

# DECISION_TIME_EXECUTION_EVIDENCE_LEDGER_V1 verdict

Status: **HOLD — PIPELINE INSUFFICIENT; MODELED EDGE INSUFFICIENT; FUSION NOT YET EVALUABLE**

## What was established

The frozen 251 decision IDs were joined to the append-only OANDA execution
ledger by exact event/order/trade identity and timestamp. Candidate order intent
is actual for all 251. Historical OANDA S5 bid/ask provides a fresh causal quote
and spread baseline for 154; transaction-bound causal depth provides market-order
fillability evidence for 153. All 251 later fills and closes are retained only in
the observed-execution boundary and never flow back into a decision input.

Strict completeness remains zero because three required evidence families were
not historically archived at decision time:

- fee and financing schedule: 0/251 complete;
- margin available, used, and rate: 0/251 complete;
- executable unwind validity: 0/251 complete.

Consequently, `ALL_TRADES`, frozen single-statistical, and frozen fusion have no
common strict-eligible rows in 16/32/64-day validation. Their Net, PF, DD,
paired LCB, and margin coverage are `null / NOT_EVALUABLE`; zero must not be read
as an economic result.

## Edge versus pipeline

These are two independent findings.

1. **Modeled edge deficiency.** In 64-day validation, the frozen weighted model
   emitted 22 two-family predictions and every TRAIN-fixed lower confidence bound
   was `<= 0`. No candidate reached the execution gate. This rejects edge only
   for those 22 modeled rows, not for all possible family combinations.
2. **Evidence-pipeline deficiency.** No episode has complete decision-time cost,
   margin, and executable-unwind evidence. This prevents a strict paired test of
   whether an edge-positive fused decision would remain executable and profitable.

The fusion architecture is therefore neither accepted nor rejected. It is
unassessed under the preregistered executable evidence contract.

## FUSED_DECISION_V1 rerun

- All 251: `WAIT=229`, `SKIP=22`, `TRADE=0`.
- 64-day validation: `WAIT=79`, `SKIP=22`, `TRADE=0`.
- Evidence-driven action changes versus the frozen renderer: 0.
- A separate renderer defect was corrected: 33 non-validation rows with no
  validation prediction had previously been labeled `SKIP`; they now correctly
  remain `WAIT`. This correction is not profitability evidence.

For context only, the independent financial oracle reproduced the 101-trade
64-day all-trades validation baseline: Net `+15,144.4802 JPY`, PF
`1.5852461375`, max DD `6,794.7768 JPY`. Those figures are not the strict fusion
comparison because strict eligibility is empty.

## Validation

- Unit/invariant tests: 14/14 PASS.
- Independent evidence oracle: 12/12 PASS.
- Independent financial oracle: 6/6 PASS.
- Deterministic regeneration: identical ledger/report/decision hashes.
- No future quote watermark, mid fill, missing-margin default, financing
  backfill, or single-leg-to-dual-unwind inference was admitted.
- Holdout unread; live/Paper/broker/order/deploy untouched.

## Forward-only reopening condition

Use `forward_acquisition_contract_v1.json` on a future research cohort. Before
each candidate order, persist bid/ask/depth and spread watermark, versioned
slippage/fee/financing schedules, account margin available/used/rate, current
exposures/concurrency, and explicit exit/unwind ordering. Post-decision fills and
exits must remain evaluation-only. Do not retrofit these fields into the frozen
251 episodes.

## Capacity checkpoint

Run-owned output is about 1.6 MiB. An unrelated filesystem drop triggered the
hard stop; a fresh storage audit then showed free space stable at about 49.25 GiB
with no deletion, move, or compression. The active paper-shadow DB/WAL under the
separate `d316` worktree was not touched or stopped.

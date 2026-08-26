# FX paper research checkpoint V3

## Outcome

No FX strategy is admitted and profit is not proven. Future profit cannot be guaranteed from a replay. The evidence target remains a full comparable month with both normal and adverse execution producing at least 2.0x, followed by an untouched future holdout. The 3.0x result is a stretch target, not a tuning instruction.

## Hard boundary

- Paper replay only.
- No broker, account, credential, trading API, order endpoint, deployment, commit, push, or external order.
- RAW signals are never removed because of cost.
- BID/ASK, slippage, financing, latency assumptions, currency exposure, and terminal liquidation remain in the final result.
- Lookahead, opened-period retuning, hidden terminal inventory, and leverage fitting are automatic failures.

## Current local source

- 28 OANDA FX pairs.
- 729,056 completed M5 BID/ASK bars.
- 2026-03-11 through 2026-07-15.
- No later untouched local FX source was found on 2026-08-25.

## What V3 built and tested

1. A 15-feature H1/H4 causal indicator factory with semantic quantile deduplication and currency-time exposure diagnostics.
2. Fourteen original composite indicators covering seismic energy, semivariance, auction geometry, failed-auction response, cross-currency propagation, dynamic shape energy, spread relaxation, and sweep reversal pressure.
3. A leave-one-pair-out 8-currency graph family and a 28-ticket to 7-major internal-netting replay.
4. A frozen V250 M15/H8 ridge partial-holdout readback using the exact saved model bytes.
5. A causal H4/H1/M15 hierarchy with identical RAW_SIGNAL, EXECUTABLE_BASE, and ADVERSE_STRESS signal identifiers.
6. A weekly matured-outcome polarity controller whose only states are CONTINUE, INVERT, and FREEZE.

## Integrity correction

The first H1/H4 factory version allowed `rejection_curvature`, which includes the next completed bar, into an immediate-fill worker. That was lookahead. The field is now forbidden for immediate entry and covered by a test. All results produced before that correction are superseded.

## Key evidence

- Corrected H4 factory: 23,868 family hypotheses, 16,336 evaluated, 0 admitted.
- Corrected H1 factory: 24,120 family hypotheses, 22,984 evaluated, 0 admitted.
- Composite factory: 780 H4 plus 780 H1 hypotheses, 0 admitted.
- Frozen V250 partial holdout: 16 decisions, normal total 1.0034456489, adverse total 1.0011702549, full May adverse 0.9948188709, zero 2.0x months.
- Frozen V250 family: 54 registered, 42 observed, 0 family-corrected positive lower bounds.
- Graph family: 798,319 RAW signals, 32 candidates, 0 admitted. Small positive gross means were below execution cost.
- Graph inventory/netting: nine candidates, 0 admitted. The best compressed target had negative gross performance.
- MTF H8: 68,458 RAW signals, 0 admitted. Best walk-forward gross means were roughly 0.33 to 0.38 bp while executable cost left roughly -2.2 to -2.5 bp.
- MTF H16: 69,290 RAW signals, 0 admitted. Longer holding did not bridge the cost gap.
- Online polarity H8/H16: every source proposal remained in the ledger, but no worker developed a sign-stable 90% daily-cluster interval; expected orders were therefore zero under the pre-registered FREEZE rule.

## Diagnosis

The main failure is not a lack of tickets. It is a lack of stable, independently clustered gross edge large enough to survive FX execution cost. The graph result also shows that many pair tickets can be one correlated currency bet. Adding leverage or thresholds would conceal this rather than solve it.

## Frozen next family

`FX_COUNTERPARTY_RESPONSE_SURFACE_V4` measures what happens after a boundary excursion. It combines excursion area, opposing-wick absorption, reclaim velocity and curvature, spread recovery, path torsion, and leave-one-pair-out currency propagation. Continuation, failed-auction reversal, and unresolved states are mutually exclusive. Its six hypotheses were frozen before implementation in `COUNTERPARTY_RESPONSE_PREREGISTRATION_V4.json`. The 15-variable causal feature calculation is now implemented in `counterparty_response_v4.py`; model fitting and the same-signal replay remain next.

## Completion state

- Research-integrity and paper-only boundary: implemented and tested.
- Original technical indicator factory: implemented; current families rejected.
- Counterparty response V4 feature mathematics: implemented and causality-tested; replay not yet complete.
- Resident future-data evidence: unavailable because no canonical matching future FX feed/source has been bound here.
- Profit evidence and monthly 2.0x acceptance: not achieved.

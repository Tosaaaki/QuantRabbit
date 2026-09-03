# Fast-bot method-aware causal entry edge V4

## Decision and requirements

The V3 forward target halted after 10 fills, 0 wins, -67.0 pips, and profit factor 0.0. Its M1-trigger check did not remove entries that began while fast and operating structure still opposed the trade or while the move was already exhausted.

V4 must:

- change entry selection only and keep the all-GO baseline, emitted price geometry, S5 truth, TTL, and hold horizon unchanged;
- decide at signal emission from the already sealed regime row, without outcomes or future candles;
- use method-specific evidence instead of one generic trigger;
- fail closed on missing or tampered evidence;
- preserve `execution_authority=NONE`, GET-only broker access, and zero live permission;
- halt automatically at the preregistered 10-fill dual-metric futility boundary.

## Data flow

```text
sealed chart packets
        |
        v
deterministic regime row ----> unchanged all-GO baseline signal
        |
        v
method-aware entry-edge builder
  trend | range | breakout-failure
        |
        v
hash-sealed emission snapshot
        |
        v
CAUSAL_ENTRY_EDGE_ONLY veto/pass
        |
        v
same exact OANDA S5 GET-only truth
        |
        v
paired baseline/target scorecard ---> futility halt or final evidence review
```

## Contracts and storage

- Signal field: `entry_edge_snapshot`
- Snapshot contract: `QR_FAST_BOT_ENTRY_EDGE_SNAPSHOT_V1`
- Policy: `METHOD_AWARE_CAUSAL_ENTRY_EDGE_V1`
- Challenger config: `QR_FAST_BOT_CORRECTIVE_CHALLENGER_CONFIG_V4`
- Target arm: `CAUSAL_ENTRY_EDGE_ONLY`
- Frozen eligibility cutoff: `2026-09-03T15:54:00Z`
- Storage remains the append-only raw signal, exact-S5 outcome, challenger, and knowledge ledgers. V1-V3 artifacts are historical and are never rewritten or reclassified.

## Reliability and safety

The signal is emitted independently of V4 acceptance so the baseline remains observable. A missing, malformed, or hash-mismatched snapshot vetoes only the target. The resident source bundle binds the new module and V4 policy bytes. Runtime status exposes the active policy contract, cutoff, target arm, collection state, order attempts, orders created, and execution authority. Any source/ledger integrity failure or any nonzero order attempt stops the research lane.

## Tradeoffs and review boundary

The stricter filter may legitimately produce few or zero fills; that is preferable to manufacturing trades. Its thresholds are fixed research hypotheses rather than proof of predictive edge. V4 is rejected early if target net pips and profit factor both fail to exceed baseline after 10 target fills. A profitability claim requires at least 100 target fills across 10 active days, PF at least 1.25, positive net and one-sided daily lower-bound deltas, and no worse loss streak. Revisit the architecture only with a new versioned, future-only cohort; never tune V4 using its own outcomes.

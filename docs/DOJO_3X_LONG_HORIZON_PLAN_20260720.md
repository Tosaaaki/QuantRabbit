# DOJO 3X long-horizon research plan

Status: `ACTIVE_RESEARCH / 3X_NOT_REACHABLE / NO_LIVE_PERMISSION`

This plan turns the monthly 3x objective into a fixed, falsifiable research
denominator.  It does not promise returns and must not grant broker or live
authority.

## Fixed historical denominator

- Broad discovery: M5 bid/ask, 28 pairs, 78 complete calendar months from
  2020-01 through 2026-06.
- Precision rectangle A: M1 bid/ask, AUD_USD, EUR_USD, GBP_USD, NZD_USD and
  USD_JPY, the same 78 months.
- Precision rectangle B: M1 bid/ask, all 28 pairs, 18 complete months from
  2025-01 through 2026-06.
- S5 is reserved for final execution-order checks because the current
  receipt-bound 28-pair corpus has only one complete common calendar month.
- Every historical window is `WORN_HISTORICAL_TRAIN_ONLY`.  It is useful for
  falsification and robustness ranking, not as an untouched holdout.

Every candidate is evaluated both as independent JPY 200,000 monthly resets
and as a continuous walk-forward account.  OHLC and OLHC paths, BASE and
STRESS costs, missing months, aborted runs and rejected trades remain in the
fixed denominator.

## Portfolio hypothesis

The portfolio must earn from genuinely different causal sleeves rather than
counting correlated currency expressions as independent bets.

1. Shock reversion: spike-fade, 25-pip tail guard, 60/240-minute release and
   FIXED/BREAKEVEN/ATR_TRAILING exits.
2. Range rail: range-edge and round-number fades admitted only in a sealed
   range regime.
3. Session breakout: compression and session-open range breaks admitted only
   after independent post-cost edge.
4. Pullback: daily/session pullbacks used only if their own STRESS edge is
   positive.
5. Gap/event recovery: weekend or timestamp-sealed event dislocations.
6. AI discretion: regime veto, exit review and candidate proposal only; it
   does not receive broker authority or expand the risk envelope.

A deterministic allocator sees every simultaneous intent, applies exits
first, then ranks entries without input-order dependence.  It limits pair,
family and currency-factor concentration and records rejected opportunities
and capital-lock hours.

## Monthly 3x gate

The headline gate is deliberately separate from the tuning objective.  A
candidate may be useful below 3x, but it may not be reported as monthly 3x
unless all of these are true:

- pessimistic path with STRESS costs reaches a final monthly multiple of at
  least 3.0 in every sealed historical diagnostic month;
- no loss month, margin closeout or omitted/aborted denominator cell;
- normal continuous MTM drawdown is at most 10% and STRESS drawdown at most
  15%;
- peak margin usage is at most 45% and margin rejection rate at most 10%;
- all required pair, family and currency leave-one-out replays remain
  profitable;
- no pair supplies more than 50% of positive P/L and effective independent
  sleeve count is at least three;
- three non-overlapping prospective 30-day paper blocks also reach 3.0 after
  the complete candidate, prompt, allocator, scorer and risk envelope have
  been frozen;
- a pre-registered block bootstrap reports 12-month ruin probability below
  the fixed gate.

Until then the authoritative label remains `3X_NOT_REACHABLE`.  The optimizer
must not infer leverage from the target gap.

## Bounded AI trainer loop

The first lineage permits at most three attempts and fourteen generated
proposals.  Every model invocation, invalid proposal, duplicate and aborted
attempt consumes its pre-registered budget.  The AI receives only a complete,
lineage-bound prior result.  It never sees a partial cell, future month,
holdout result or broker credential.

The next study is sealed and appended to the lineage before its runner starts.
Crash recovery reuses the same content-addressed phase and output path; it
cannot mint a free retry.

## Storage and analysis

Google Drive folder `QuantRabbit DOJO Archives` is the durable evidence store.

- `completed-runs`: cell/month `.tar.zst` evidence chunks
- `manifests`: source, corpus, code, plan, chunk and lineage hashes
- `reports`: compact monthly, candidate, pair, family and risk summaries read
  by the AI trainer

The raw corpus remains the price source of truth.  Long-horizon evidence stores
state-changing events and bounded checkpoints instead of duplicating every
price and full account state.  An independent reducer streams the source
corpus to reconstruct continuous MTM, financing, margin and P/L.

An archive is remotely verified before local cache eviction.  Source evidence
is never deleted by the archiver.

## Execution order

1. Finish and bind the current 96-replay one-month diagnostic.
2. Archive its evidence to Drive and recover the local disk floor.
3. Differentially prove compact evidence metrics against the existing full
   ledger on fixed fixtures and the completed run.
4. Run broad M5 discovery on the sealed portfolio candidates.
5. Promote only surviving candidates to both M1 precision rectangles.
6. Apply full pair/family/currency LOPO and rolling outer folds.
7. Freeze the final version and begin prospective paper blocks.

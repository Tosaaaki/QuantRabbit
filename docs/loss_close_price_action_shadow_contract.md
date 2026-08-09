# Loss-close price-action inventory shadow v1

This is a read-only research contract. It cannot place Paper/live orders,
change broker state, deploy code, unlock holdout data, or support an “always
profitable” claim.

## Claim under test

The user’s observation is converted into a falsifiable claim:

> Multi-bar price action and chart structure add after-cost information for
> inventory toxicity and unwind decisions beyond inventory-only rules and
> one/two-candle shape features.

The claim is rejected for this use case unless the multi-bar arm is strictly
better on both TRAIN and VALIDATION without worse drawdown, ruin-floor,
margin-closeout, fill-order, or unwind outcomes.

## Two experiment families

The existing hedge experiment keeps its names:

- Hypothesis A: close the original at the frozen SL, then open an opposite
  STOP scenario at scale `0.25` or `0.35`.
- Hypothesis B: open an equal opposite leg at initial entry or the SL trigger,
  retain the original, and explicitly unwind both legs.

Price-action inventory evaluation uses separate labels to avoid redefining A
or B:

- `INVENTORY_ONLY`: deterministic quantity/age/skew/unwind control.
- `CANDLE_1_2`: the same event and costs, adding only the last one or two
  candles’ body, wick, direction, and engulfing shape.
- `PRICE_ACTION_MULTI_BAR`: the same event and costs, adding causal M1/M5
  structure, compression/expansion, directional efficiency, rolling-range
  location, breakout acceptance/failure, retest, and repeated rail attacks.
- `AI_SUPERVISOR`: not evaluated until the multi-bar arm survives both
  pre-holdout splits. AI is an inventory supervisor candidate, not a source of
  new execution permission.

The pattern path is hierarchical rather than a bag of thresholds:

1. establish the higher-frame direction and important horizontal rail;
2. observe lower-frame compression and repeated attacks;
3. identify a double-top/bottom or ascending/descending-triangle candidate;
4. require a close-confirmed break, then acceptance or retest;
5. freeze structural invalidation and unwind before scoring the outcome.

The first raw break is not an entry by itself. A wick through the rail without
close acceptance is a failed-break observation. If the hierarchy is absent,
the correct inventory action in this shadow is `SKIP`, not a lower-confidence
trade. EMA location may be tested later as a separate confirmation layer; it
must not be allowed to hide whether multi-bar price structure itself added the
increment.

## Sweep discipline

Do not launch one full Cartesian search. Use a hierarchy:

1. Freeze event identity, executable S5 bid/ask path, costs, financing,
   margin model, fill-order treatment, and unwind rule.
2. On TRAIN only, sweep frame geometry: M1/M5, structure length, breakout
   rail length, acceptance length, and compression regime length.
3. Keep broad parameter plateaus, not the single best cell.
4. Compare the surviving plateaus once on VALIDATION with an embargo at least
   as long as the maximum holding/unwind horizon.
5. Stop if `PRICE_ACTION_MULTI_BAR` does not beat both controls after costs or
   makes tail risk worse. Holdout remains untouched.

The executable selector is `loss_close_multidimensional_sweep_v1` in
`src/quant_rabbit/loss_close_multidimensional_sweep.py`. Stage 1 contains 27
coupled geometry cells rather than the much larger Cartesian product. Stage 2
changes one structure axis at a time around TRAIN plateaus. Stage 3 changes
only rail tolerance around surviving TRAIN centres.

A plateau is a connected region of at least three adjacent cells with a centre
having at least two neighbours. Eligibility requires the lower of the two
increments versus the same-frame `INVENTORY_ONLY` and `CANDLE_1_2` controls to
be at least 80% of the best TRAIN increment. Component selection is TRAIN-only.
The frozen component must still contain a connected three-cell positive region
on VALIDATION. A single best cell is always rejected, even when it wins on both
splits. The embargo must be at least the maximum unwind horizon.

Sizing is evaluated after signal and unwind quality. An equity-step rule such
as increasing size after a gain or reducing it after a drawdown changes the
distribution of an existing edge; it cannot be counted as evidence that the
entry pattern has positive expectancy.

All price-action bars must be complete and end before the frozen decision
timestamp. Changing any later candle must leave the decision context byte-for-
byte unchanged.

OANDA may omit an S5 row when there was no price update. The feature-context
layer reports those gaps and may aggregate the observed OHLC inside a time
bucket after that bucket has ended. This exception is never inherited by the
hedge/fill-order scorer: a gap remains fatal wherever exact S5 fill order is
being asserted.

## Economic and risk fields

Every paired arm must report after-cost net JPY, spread via executable bid/ask,
fee, slippage, financing, longest-leg margin, gross exposure, fill-order
resolution, trend continuation, maximum drawdown, deterministic ruin-floor
and margin-closeout breaches, and explicit unwind completion. Ruin probability
is not inferred from a single path.

The implementation contracts are `loss_close_price_action_context_v1` and
`loss_close_price_action_ablation_v1`.

# Loss-close hedge paired shadow v1

This contract is a read-only diagnostic extension of
`loss_close_paired_shadow_v1`. It does not authorize Paper, live, broker,
order, deployment, or holdout access.

## Frozen arms

- Control: the existing entry and current TP/SL, resolved from complete S5
  executable bid/ask candles. The experiment is blocked unless the first
  protection touch is an unambiguous SL.
- A: the original closes at SL and an opposite STOP scenario opens at that SL
  candle with scale exactly `0.25` or `0.35`; the opposite leg exits at a
  precommitted fixed complete S5 candle.
- B: an equal-size opposite leg opens either at the initial entry or the SL
  candle. The original remains open, so both legs require an explicit common
  fixed-candle unwind.

Every arm uses the same entry, S5 path, quote-to-JPY conversion, and explicit
non-spread cost model. Spread is intrinsic in executable bid/ask prices. Fee,
slippage, and financing stress are charged separately once.

## Fail-closed and falsifiability rules

- S5 input must be complete, contiguous, aligned, pair-matched, and have a
  strictly positive ask-over-bid spread.
- TP/SL dual touch, a non-SL first touch, a missing or post-hoc unwind, a price
  outside its executable-side candle, non-integral hedge units, and holdout
  use block scoring.
- S5 cannot establish the order of an SL close and hedge open, or dual-leg
  entry/unwind, inside one candle. For B at SL, the original explicitly
  remains open and the unresolved event is the SL trigger versus hedge open,
  not an original close. The result records these separately and remains
  proof-ineligible.
- Margin is a same-pair longest-leg proxy. Strategy authorization is separate
  and always false.
- Ruin output is a deterministic equity-floor and margin-closeout proxy, not a
  ruin probability.
- One path may refute a hypothesis. It cannot support an “always profitable”
  or statistical claim.

The fixed output contract is `loss_close_hedge_paired_shadow_v1`.

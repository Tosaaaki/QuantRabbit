# bitbank Paper Stop-Loss Forward Comparison

Updated: 2026-07-29 JST

## Conclusion

The existing bitbank Strategy Lab already used fixed price stops.  The
unproved part was whether that fixed distance was appropriate for current
micro-volatility and round-trip costs.  This experiment therefore keeps the
current fourteen Paper lanes running and adds six isolated forward-only lanes:

- `ORDER_BOOK_FADE_SL_FIXED_CONTROL` Spot / Margin
- `ORDER_BOOK_FADE_SL_VOLATILITY` Spot / Margin
- `ORDER_BOOK_FADE_SL_TIME` Spot / Margin

The fixed arm retains the current 2.5 bps price stop.  The volatility arm
derives its price stop from the larger of the causal sixteen-event price
range, expected round-trip cost, and the fixed control floor, with a
predeclared Paper-only cap.  The time arm disables the price stop but keeps
the existing twelve-second maximum hold and signal invalidation exits.

No arm may place, cancel, or close a real bitbank order.  Authority remains
`NONE`.

## Forward Evidence Contract

- Start only after the policy was committed.
- Use public-stream observations available at decision time.
- Never read future or terminal outcome data when deciding.
- Keep each mode and stop policy in an independent ledger, state, outbox,
  process, and initial 10,000 JPY Paper account.
- Include maker/taker fees, spread, adverse-selection assumptions, and
  Margin interest in after-cost results.
- Report after-cost PnL, profit factor, expectancy, maximum drawdown,
  stop-out rate, and aggregate opportunity loss versus the fixed control.
- Do not adopt before at least 30 completed trades per lane and three
  distinct unused forward windows.
- Adoption additionally requires profit factor above 1, positive
  expectancy, and drawdown no worse than the control.

The first opportunity-loss metric is explicitly a same-window aggregate, not
a trade-paired counterfactual.  It cannot be used as exact causal proof.

## Root-Cause Priority

1. `ORDER_BOOK_FADE` entry edge and direction are the dominant loss
   contributor.  A better stop cannot repair a negative entry edge.
2. Forced taker exits and turnover can make gross edge uneconomic after
   fees.
3. Predicted gross edge is often below executable spread, adverse selection,
   and fee costs.
4. Fade entries can concentrate in one contrarian side or regime.
5. Stop policy may either crystallize ordinary book noise too early or allow
   a losing inventory to remain until the time exit.

The stop comparison addresses only item 5.  Items 1-4 remain separate
strategy and execution-cost obligations.

## Safety and Rollback

- `NO_EXECUTE=true`
- `CRYPTO_LIVE_READY=false`
- `WITHDRAWAL_ENABLED=false`
- `CRYPTO_ORDER_AUTHORITY=NONE`

Rollback is limited to booting out the six
`com.quantrabbit.crypto-strategy-order-book-fade-sl-*` LaunchAgents.  Existing
fourteen Paper lanes and their ledgers must not be stopped or deleted.

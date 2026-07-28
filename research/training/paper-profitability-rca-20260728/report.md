# Paper profitability RCA — 2026-07-28 07:44 UTC

Scope: local virtual-broker Paper rooms only. The market feed is read-only
OANDA data, but all orders, positions, balances and P&L below are virtual.
`live_permission=false`, `broker_mutation_allowed=false`, and
`order_authority=NONE`.

## Operating normally is not the same as profitable

The four established rooms have run for 4.93 days (70.5% of a requested seven
day window) and have near-complete 24-hour coverage. They are operational:
processes are alive, ledger chains continue, quotes arrive, no orders were
rejected, and only 4–7 quote timeouts were observed per room.

Profitability is mixed and is not good on the longer available window.

| room | 24h settlements | 24h net / PF / expectancy | partial 7d settlements | partial 7d net / PF / expectancy | window DD |
| --- | ---: | --- | ---: | --- | ---: |
| prev-day BASE | 15 | +¥402.01 / 2.633 / +¥26.80 | 24 | -¥320.03 / 0.750 / -¥13.33 | ¥807.16 |
| prev-day STRESS | 15 | +¥188.73 / 1.426 / +¥12.58 | 25 | -¥515.40 / 0.660 / -¥20.62 | ¥785.41 |
| round-number BASE | 0 | unavailable | 1 | +¥139.37 / unavailable / +¥139.37 | ¥0 |
| round-number STRESS | 0 | unavailable | 1 | +¥141.81 / unavailable / +¥141.81 | ¥0 |

The round-number result is not a profitability success: one settlement in
almost five days is insufficient to estimate PF or expectancy. Its issue is
coverage/entry scarcity, not demonstrated edge.

## P&L and cost decomposition

For `prev_day_extreme_fade` over the partial seven-day window:

- BASE gross profit ¥960.05, gross loss ¥1,280.08, net -¥320.03.
- STRESS gross profit ¥1,002.17, gross loss ¥1,517.57, net -¥515.40.
- SHORT contributes -¥553.84 BASE and -¥687.85 STRESS. LONG contributes
  +¥233.81 and +¥172.45.
- Counter-trend trades contribute -¥412.52 BASE and -¥440.36 STRESS.
  Trend-aligned trades contribute +¥92.49 BASE and -¥75.04 STRESS.
- The STRESS minus BASE net difference is -¥195.37 over the available window.
  This is a directional cost sensitivity estimate, not a perfectly
  fill-matched causal cost estimate.
- Average observed entry spread is 0.804 pips BASE and 0.816 pips STRESS.
  Fill-record latency averages 854 ms BASE and 725 ms STRESS. There were no
  rejects. The runtime configured an additional 0.3-pip slippage and
  0.8-pip/day financing for STRESS.

For the latest 24 hours, BASE/STRESS are both profitable, but STRESS trails
BASE by ¥213.28 with the same 15 settlements. This short-window recovery does
not erase the negative partial-seven-day PF and expectancy.

## Current virtual inventory

Established rooms are flat with one resting SHORT order each:

- prev-day BASE balance ¥199,679.98; prev-day STRESS ¥199,484.61;
- round-number BASE ¥200,139.37; round-number STRESS ¥200,141.81.

The new direction-pair pilot has only 26 minutes of runtime and no settlements.
Both Bot-only lanes hold one virtual USD/JPY SHORT:

- BASE Bot-only: equity ¥199,936.98, unrealized -¥63.02, margin usage 17.21%;
- STRESS Bot-only: equity ¥199,947.04, unrealized -¥52.96, margin usage 17.21%.

Both direction-gate lanes are flat at ¥200,000. All four lanes have one resting
LONG order. This is too early for an economic conclusion and the rooms poll
independently, so they do not yet satisfy the proposed shared-feed
champion/challenger contract.

## Root causes ranked by impact × confidence

1. **Counter-trend SHORT concentration — high impact, high confidence.**
   Combined directional contribution is -¥852.88 across BASE/STRESS; combined
   SHORT contribution is -¥1,241.69. The two sums overlap and must not be added.
   The repeated pattern is selling while the 24h direction is LONG.
2. **Round-number strategy starvation — high operational impact, high
   confidence.** One fill/settlement per cost lane in 4.93 days and none in the
   latest 24h means the strategy cannot contribute enough evidence or P&L.
   Whether this is threshold excess or current-regime mismatch remains
   unresolved.
3. **Execution/holding-cost sensitivity — medium impact, medium confidence.**
   STRESS underperforms BASE by ¥195.37 on the available longer window and by
   ¥213.28 in 24h. No rejects are involved; spread, configured slippage and
   financing are the plausible cost channels. Fills are not perfectly matched,
   so attribution is directional.

The first one-category pilot is therefore direction alignment, with Bot-only
kept unchanged and a direction gate run beside it. It is **not adopted**:
there are no settlements and the same-feed requirement is not yet met.

## Unmeasured and prohibited inferences

- TP gross retention, MFE-to-exit giveback and early-cut winning profit are not
  present in these ledgers.
- WAIT/missed-opportunity counterfactuals are not logged.
- Forecast packet duplicate/resolution rates are outside these Paper-room
  ledgers.
- Intratrade equity DD cannot be reconstructed from close-only ledgers; the DD
  table is close-sequence window DD.
- Slippage in JPY cannot be safely isolated from realized P&L with the available
  entry/exit records.
- No claim of positive future returns, no live promotion, and no strategy
  adoption is supported.

Next safe step: keep collecting the existing rooms, build a new shared-feed
Paper executor with isolated lane ledgers, and admit no challenger until the
policy's sample, cost, DD, regime and dedupe gates pass. DOJO is not on this
path.

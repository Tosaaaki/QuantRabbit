# Owner integration: execution path, exit design, and target arithmetic

Status: **ACTIVE SCHEDULE REPAIRED / EXIT ARMS HOLD**

This checkpoint keeps three questions separate:

1. whether the current replay arithmetic is trustworthy;
2. whether a different exit policy is worth a paired replay;
3. whether the stated capital target is arithmetically and operationally reachable.

It does not read holdout and does not authorize live, Paper, broker order, or
deployment changes.

## Frozen evidence

- `loss_close_price_action_sweep/preholdout_stage1_report.json`
  SHA-256 `4567a3205a092a509bd4ae8a247c624c5e3d1d02c00537703cd4c3794186c64d`
- `loss_close_paired_robustness/robustness_report_v2.json`
  SHA-256 `ce44d29152725f6e6361f28ef5b41224459cbec11be4daffa71377f47cfe7069`
- `decision_time_execution_evidence/2026-08-10/coverage_report_v1.json`
  SHA-256 `ee84f61dd89a6069155c4e2bdc4e8200460bb9be2d0a361857e1e10eb0da25b1`
- `continuous_thesis_monitor/2026-08-10/inventory_overlap_report_v2.json`
  SHA-256 `6d6c1d6b2664150becd8778b38df98e98851a1394d006e2450a6140bb35b5822`
- `active_protection_schedule/2026-08-10/coverage_report_v1.json`
  SHA-256 `c3d46833ef16c9477a2b2f8ea4b7ecd45c382654381079a9ec9ba8d756d0db17`
- `active_protection_schedule/2026-08-10/oanda_extension_manifest_v1.json`
  SHA-256 `e5c0273a3e5d02d502a2aa35d666f3cb89e3e785f9f15654a62d0316c964c19b`

## A. Execution-path arithmetic audit

The implemented **static** protection first-touch is executable-side and uses
the candle extremes, including wicks:

- LONG TP/SL: bid high / bid low;
- SHORT TP/SL: ask low / ask high;
- both touched in one S5 candle: `AMBIGUOUS`, never guessed;
- gaps are not interpolated and M1 is not substituted for S5 ordering.

This is the correct price-side contract for a resting TP, SL, BE stop, or
already-active trailing stop, but it is not yet a faithful protection replay.
The admission/replay query takes TP and SL only from the entry
`ORDER_ACCEPTED` row. In the frozen 251 actual-after-cost episodes that leaves
74 with both TP and SL, 159 with TP only, and 18 with neither. For those 251
trades the broker ledger contains 810 `PROTECTION_CREATED` rows: every trade has
a created TP, 99 have a created SL, 133 have replacements, and three have
multiple SL creations. Exact order-key reconstruction finds 664 cancellations:
456 `CLIENT_REQUEST_REPLACED`, 18 direct client cancels, and 190 linked terminal
cancels. The older count of 612 was incomplete. The original replay still does
not consume that changing schedule, but `ACTIVE_PROTECTION_SCHEDULE_V1` now
reconstructs it for all 251 episodes. All 456 replacement links pass, and all
146 broker TP/SL terminal orders match the protection active immediately before
close. The normalized ledger stored the old order id on all 456 replacement
create rows, so the checkpoint uses raw OANDA transaction identity.

A close-confirmed technical exit is a different contract: its signal is
computed only on the completed bar and the resulting order can fill only on a
subsequent executable quote. The bar's earlier wick cannot trigger a stop that
did not exist yet. The existing static touch scan also includes the fill and
close boundary S5 candles; an independent three-pair scan found 6 of 16 detected
first touches on the close boundary candle. Their intrabar order relative to the
close is unknown and must be treated as ambiguous unless raw ticks resolve it.

Current evidence is not sufficient for a full path audit. The 64-day STOP
cohort has 14 events: 7 are emitted as diagnostic calculations and 7 are
blocked, but all seven diagnostic events still have a path gap (2 to 2,304 S5
bars) or unresolved alternative-leg fill ordering. They are not seven fully
proved paths. The price-action
Stage-1 cohort has 7 context-ready events but zero strict economic-score-ready
events because every context event has at least one entry-to-unwind S5 gap.
The decision-time execution ledger also has 0/251 complete fee/financing
schedule evidence, 0/251 decision-time margin evidence, and 0/251 executable
unwind evidence.

The dedicated loss-close paired path contains no per-trade MFE/MAE fields and
does not replay a time-varying BE/trail schedule.
Other repository diagnostics compute MFE/MAE for different cohorts, but those
cannot be joined as if they were the same 251-episode paired truth. Therefore
BE/trailing-path claims cannot yet be inferred from the existing loss-close
report. In particular, the existing execution-timing MFE diagnostic covers only
48 unique admitted trades at M1 resolution and uses a favorable extreme from
the same candle for a counterfactual exit amount; it does not prove executable
ordering, slippage, financing, or partial fills.

Independent readback reproduced 30/30 arm/split arithmetic checks. The 251
actual labels sum to -18,039.7866 JPY across the full acquisition period, while
the frozen 64-day VALIDATION slice is +15,144.4802 JPY. These are different
time/split populations, not a contradiction. Relevant unit/invariant checks
also pass. Fill/close prices match the correct executable side on 251/251 rows,
and episode net equals broker realized P/L plus financing on 251/251. The 64-day
gain/loss totals independently reproduce as 41,021.5928 / 25,877.1126 JPY.
Spread is already embedded in those executable bid/ask prices, so the displayed
11,966.0881 JPY fill-plus-close half-spread sum must not be subtracted again.
The defect is not a discovered summation or executable-side error; it is
protection-schedule, boundary-ordering, path, and decision-time evidence.

## B. Capital-target arithmetic

For starting capital `C = 200,000 JPY`, `N = 200` trades, and a simple terminal
return target `R`, the required fixed-JPY expectancy is `C * R / N`:

| Terminal target over 200 trades | Required expectancy | Starting-capital rate per trade |
|---|---:|---:|
| +10% | 100 JPY | 0.05% |
| +30% | 300 JPY | 0.15% |

With equal proportional compounding, the required per-trade rate is
`(1 + R)^(1/200) - 1`: 0.0476664% for +10% and 0.1312682% for +30%.

The frozen 64-day VALIDATION contains 101 trades, not 200 trades per day:

| Contract | Net | Expectancy | PF | Max DD | 200-trade fixed-JPY projection |
|---|---:|---:|---:|---:|---:|
| ALL_TRADES | +15,144.4802 JPY | +149.9453 JPY/trade | 1.5852 | 6,794.7768 JPY | +29,989.0697 JPY (+14.9945%) |
| Pair-support inventory V2 | +16,422.9542 JPY | +162.6035 JPY/trade | 1.7224 | 5,849.6756 JPY | +32,520.7013 JPY (+16.2604%) |

The point expectancy is above the 100-JPY hurdle for a **200-trade cumulative
+10% target**. It does not prove **daily +10%**, because the observed frequency
is only 101 trades over 64 days (1.578/day). At the observed cadence the simple
average is about +236.63 JPY/day for ALL_TRADES and +256.61 JPY/day for the
inventory candidate, or about 0.1183% and 0.1283% of 200,000 JPY per day.

At the current point expectancy, +30% over 200 trades still needs about 2.0007x
the ALL_TRADES expectancy or 1.8450x the inventory-candidate expectancy. The
inventory candidate is not admitted: only 14 validation decisions change,
paired LCB is -22.2296 JPY, and decision-time margin coverage is zero.

Any 200-trade projection is arithmetic only. It does not prove that 200
independent, fillable opportunities exist in one day, that spread and slippage
remain stable at the higher turnover, or that correlation, concurrency, margin,
and loss tails permit the required sizing.

## C. Paired exit-policy contract

The next exit study should not tune against observed VALIDATION. The three-pair
OANDA S5 archive has now been extended to the sealed anchor, but raw-tick
ordering and internal no-bar intervals remain unresolved. The current three-pair
VALIDATION population is only 3 / 11 / 23 episodes in 16d / 32d / 64d, before
the stricter cost, margin, and unwind gates. These are coverage counts, not
results. Freeze the following arms in TRAIN and evaluate each separately on the same entries,
position sizes, bid/ask source, costs, financing, and 16/32/64-day chronological
splits with the existing embargo:

1. **Fixed BE:** after attached-TP progress reaches 60% and MFE exceeds
   `max(entry-frozen M5 ATR14, spread)`, move the stop to the actual fill price;
   it becomes effective on the next S5 quote.
2. **Cost-buffered BE:** after the same activation rule, move the stop far
   enough to cover entry/exit spread, slippage bound, fee, and financing accrued
   at that decision time. If a causal fee or financing schedule is missing, the
   strict arm is ineligible rather than assuming zero.
3. **Partial TP + BE:** at the first +1.0 entry-frozen ATR, close 50% rounded
   down to 100-unit increments, requiring original units >= 2,000 and runner
   units >= 1,000. Only after that executable fill may the runner stop become
   BE; it is active from the next S5.
4. **ATR trail:** start after +1.0 entry-frozen ATR, calculate a stop 1.5x the
   latest ATR behind the favorable extreme on each completed M1 bar, apply it
   from the next S5, and enforce one-way ratcheting. Keep the original TP.
5. **SMA-angle trail:** after +1 ATR, require side-aligned SMA20 slope < 0 for
   three completed M1 bars, then ratchet a stop 1.5 ATR behind the completed-bar
   close. It must not trigger a same-bar market exit or reduction.
6. **Dynamic sizing:** preserve entry/exit geometry and apply a TRAIN-fixed size
   cap from decision-time risk/inventory evidence. Keep this separate from exit
   improvement so size and path effects are not confounded.

For every arm, record executable-side MFE/MAE, activation time, active stop at
each bar, first-touch reason, ambiguous same-S5 touches, partial-fill ordering,
all leg costs, financing, margin peak, DD, turnover, and terminal reason. Missing
paths remain missing rather than zero or baseline pass-through.

Broker protection uses wick touch: LONG MFE/TP and MAE/SL use bid high/low;
SHORT uses ask low/high. Signal confirmation may separately compare preregistered
`TOUCH_S5` and `CLOSE_M1` scenarios, but the better VALIDATION scenario cannot be
selected afterward. For a same-S5 TP/SL touch, the primary conservative bound is
STOP-first and TP-first is an upper-bound diagnostic. The first incomplete S5
containing the fill is unresolved; completed-bar actions never apply
retroactively. Gap-through stops fill at the next executable open when worse;
favorable TP gaps are capped at the TP level.

Acceptance requires at least 30 TRAIN, 20 VALIDATION, and 10 changed decisions;
on both 32-day and 64-day VALIDATION it also requires after-cost incremental
Net > 0, paired LCB > 0, PF > 1, DD not worse, margin not increased, and complete
fill/unwind validity. The 16-day result is always reported. Negative controls
include forbidden outcome-best exit, mid fill, no-cost, same-bar retroactive
stop, TP-first same-S5 touch, side inversion, and shuffled action time. A
positive point estimate alone is insufficient.

## Integrated disposition

- **Audit defect:** the missing active protection schedule is now reconstructed
  for 251/251 episodes, but existing replay consumers are not yet wired to it.
  Path coverage, MFE/MAE lineage for the paired cohort, fee and financing
  schedule, decision-time margin, and executable unwind remain incomplete.
  Same-S5 ordering cannot be recovered from OHLC.
- **Exit improvement:** plausible and testable, especially cost-buffered BE and
  partial TP + BE, but not yet economically evaluated on a sufficiently complete
  paired cohort. Prior continuous hard technical exits were adverse and should
  not be retuned.
- **Target feasibility:** the system is already positive on the frozen 64-day
  ALL_TRADES cohort and clears the simple 100-JPY-per-trade hurdle. The evidence
  does not support the much stronger claim of daily 10% or 30%; +30% per 200
  trades also misses the expectancy hurdle.

The final seven hours of three-pair OANDA S5 bid/ask candles were acquired twice
with identical expanded hashes and row counts. They are valid candle truth, but
their missing timestamps remain unresolved without raw ticks. Three-pair
VALIDATION counts are only 3 / 11 / 23 for 16d / 32d / 64d; the first two are
below the preregistered minimum 20.

The correct next gate is source-separated raw-tick and remaining-pair path
acquisition, causal fee/financing and margin evidence, and executable
partial-fill/unwind evidence, followed by the frozen paired replay of the six
arms. It is not increasing leverage or declaring that a positive result must
exist.

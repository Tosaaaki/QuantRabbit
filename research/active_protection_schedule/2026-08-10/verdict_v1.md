# Active protection schedule checkpoint — 2026-08-10

## Decision

`ACTIVE_PROTECTION_SCHEDULE_V1 = PASS` for the frozen 251 executed episodes.

This repairs the previously identified static-entry TP/SL audit defect. It does
**not** make BE, partial TP, ATR trail, or SMA-angle trail accepted or rejected.
Those exit arms remain `HOLD_INSUFFICIENT_EXECUTION_EVIDENCE`.

## Direct readback

- Episodes reconstructed: **251 / 251** strict eligible.
- Protection creates joined: **810**.
  - TP: 233 `ON_FILL` + 22 `CLIENT_ORDER` + 453 replacements = 708.
  - SL: 74 `ON_FILL` + 25 `CLIENT_ORDER` + 3 replacements = 102.
  - Every one of the 251 trades has at least one TP create; 99 have at least
    one SL create.
- Cancellations joined to those trades: **664**.
  - `CLIENT_REQUEST_REPLACED`: 456.
  - `CLIENT_REQUEST`: 18.
  - `LINKED_TRADE_CLOSED`: 190.
- Trades with at least one replacement: **133**.
- Exact replacement links: **456 / 456** pass; bad link count **0**.
- Normalized protection `order_id` mismatches broker raw id: **456**. Every one
  is a replacement-create row; the raw broker identity chain is used instead.
- Broker TP/SL terminal closes: **146**. Active protection immediately before
  close matches the broker terminal order **146 / 146**.
- Other terminal paths (market close/margin or non-protection order): **105**;
  they are not relabelled as TP/SL.

The earlier owner checkpoint's 612 cancellation count was incomplete. The exact
trade/order-keyed reconstruction finds 664 because it retains all 190 linked
terminal cancels and 18 direct client cancels in addition to 456 replacements.
No economic conclusion depended on the old 612 count.

## Fixed-window coverage

The schedule itself is complete in every preregistered split:

| Window | TRAIN | VALIDATION | Strict TRAIN | Strict VALIDATION |
|---|---:|---:|---:|---:|
| 16d | 13 | 12 | 13 | 12 |
| 32d | 43 | 31 | 43 | 31 |
| 64d | 145 | 101 | 145 | 101 |

For the currently archived feature/execution three-pair boundary
(`AUD_JPY/EUR_JPY/EUR_USD`), VALIDATION contains only **3 / 11 / 23** episodes
for 16d / 32d / 64d. Thus 16d and 32d cannot meet the preregistered minimum 20
VALIDATION observations without acquiring additional pair truth or changing the
contract. The contract is not relaxed after seeing this result.

## Bounded S5 extension

The missing final seven hours to the sealed anchor were fetched twice from the
read-only OANDA instrument-candles endpoint.

| Pair | Rows | Expanded SHA-256 | Repeat match | Internal gaps >5s |
|---|---:|---|---|---:|
| AUD_JPY | 4,733 | `3841f01730824c3b8783aafeda1b2aedbfc27c90939a79d15ea4e7188890932a` | yes | 275 |
| EUR_JPY | 4,864 | `5a49ef2be7904b19f7c1f5ff7ef487dbb3af1c3daafd866f27bcf194b2a02d44` | yes | 204 |
| EUR_USD | 4,225 | `109725193fbc85b6f79cba27bb2dac7228b2e1ed3b934bf7c6dc86a3f15044ad` | yes | 591 |

All rows are complete, monotonic, duplicate-free, on the S5 grid, and satisfy
bid/ask plus OHLC invariants. Internal missing timestamps remain
`CANDLE_ENDPOINT_NO_BAR_INTERVAL_UNRESOLVED_WITHOUT_RAW_TICK_TRUTH`; they are not
filled or promoted to no-trade intervals.

## Verification

- Unit tests: **12 / 12 PASS** (8 schedule + 4 S5 validator).
- Independent SQLite oracle: **13 / 13 PASS**.
- Repeated build: all five schedule/oracle output hashes identical.
- Double historical fetch: all three expanded hashes and row counts identical.
- Disk use for the primary fetch was about 200 KiB compressed; free space stayed
  about 43 GiB. No active DB/WAL or runtime writer was touched.

## Economic boundary and next prerequisite

The verified 64d ALL_TRADES baseline remains +15,144.4802 JPY for 101
VALIDATION trades. This phase changes none of that arithmetic.

Exit-arm profitability is still not admissible because:

1. raw-tick ordering is absent for boundary candles and unresolved S5 gaps;
2. 16d/32d three-pair VALIDATION counts remain below 20;
3. decision-time fee/financing schedule, margin, partial-fill depth, and
   executable unwind coverage remain zero.

The next safe phase is either (a) acquire exact source-separated raw tick and
the remaining pair coverage needed by the frozen windows, or (b) record a
forward acquisition contract for evidence that cannot be reconstructed. It is
not valid to tune an exit arm, impute margin, use mid fills, or select the better
body/wick rule after observing outcomes.

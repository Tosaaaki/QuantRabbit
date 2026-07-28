# DOJO Paper profitability RCA and direction-alignment pilot

## Scope and authority

- Project: QR (FX trading)
- Worktree: `/Users/tossaki/App/QuantRabbit-worktrees/dojo-dual-eval`
- Evaluation tier: `LINEAGE_UNSEEN_DIAGNOSTIC_NOT_OOS_PROOF`
- Authority: PAPER/replay only, `live_permission=false`,
  `broker_mutation_allowed=false`, `order_authority=NONE`
- Existing accepted decisions, ready packet, ledgers, positions, and orders were
  read only. No accepted cell was re-decided.

## Latest Paper measurement

Measured at 2026-07-28 14:38 JST from the four active USD/JPY PAPER rooms.

- Runtime health: `HEALTHY`; handoff state `WAITING_FOR_FRESH_TASK`;
  accepted decisions 19.
- Profitability: `LOSS_OR_FLAT`; balance/NAV JPY 799,445.77 from JPY 800,000;
  realized/net P/L JPY -554.25; unrealized P/L JPY 0.
- Inventory: 0 positions, 4 resting LIMIT orders, margin used JPY 0, margin
  usage 0%.
- Resting order direction: two `prev_day_extreme_fade` SHORT orders at
  163.794 oppose `trend_24h=LONG`; two `round_number_fade` LONG orders at
  163.500 align with `trend_24h=LONG`.
- Last AI action: `HOLD`, reason
  `FOUR_RESTING_LIMITS_BALANCED_LONG_SHORT_NO_POSITIONS_NO_ACTIVE_RISK_SIGNAL`.
  The decision is shadow only and was not applied to positions.

### Loss decomposition

| Slice | Trades | Gross profit | Gross loss | Net JPY |
|---|---:|---:|---:|---:|
| `prev_day_extreme_fade` | 49 | 1,962.22 | 2,797.65 | -835.43 |
| `round_number_fade` | 2 | 281.18 | 0.00 | +281.18 |
| `SHORT` vs `trend_24h=LONG` | 11 | 364.31 | 1,623.45 | -1,259.14 |
| `LONG` vs `trend_24h=SHORT` | 23 | 1,078.35 | 672.09 | +406.26 |
| `SHORT` vs `trend_24h=SHORT` | 15 | 519.56 | 502.11 | +17.45 |
| `LONG` vs `trend_24h=LONG` | 2 | 281.18 | 0.00 | +281.18 |

Exit attribution:

- TP exits: JPY +2,099.71 across 26 exits.
- Ceiling/normal `CLOSE`: JPY -1,430.48 net; JPY 1,574.17 loss and
  JPY 143.69 profit across 23 exits.
- Stop-loss exits: JPY -1,223.48 across 2 exits.
- Forced liquidation/margin closeout: 0.
- Independent-room historical max drawdown: JPY 807.16 BASE and JPY 785.41
  STRESS for `prev_day_extreme_fade`; zero for the two round-number rooms.
- Largest negative entry-hour slices in JST were 20:00 (JPY -886.21),
  15:00 (JPY -283.69), 19:00 (JPY -244.37), and 22:00 (JPY -159.23).

The declared cost contracts are BASE (0 slippage, 0 financing) and STRESS
(0.3 pips per fill, 0.8 pips/day). Recorded P/L is evaluated under those
contracts, but the ledger does not separately expose complete execution and
financing yen amounts. Per-trade MFE/high-water data is also absent, so
giveback and missed-profit amounts remain unmeasured rather than estimated.

## Why the earlier AI policy reduced losses

The worn r13 paired study reduced large losses mainly by pausing new entries
and therefore suppressing loss-making turnover and its execution cost. It was
not evidence of superior exit timing. Its blanket behavior also damaged
profitable families: `prev_day_extreme_fade` lost JPY 3,053.97 BASE and
JPY 2,860.34 STRESS relative to Bot-only, while `round_number_fade` lost
JPY 675.39 BASE and JPY 680.99 STRESS. This ruled out adopting the blanket
pause/reduction policy.

## Sealed paired pilot

The pre-registered rule is
`SKIP_ENTRY_WHEN_SIDE_OPPOSES_TREND_24H`. It uses only the immutable
`entry_context` attached to `FILL_LIMIT`; exits and terminal outcomes are used
only for scoring. Existing positions are never closed or modified.

- Plan SHA-256:
  `d66473a1871e67dae06970d6aec94b762dd1dc55af0647b67fd5e7da4ad844d6`
- Result SHA-256:
  `626d3edd043e090770fe23676b1194b753a41a0e494e37959f2270c4dc965efa`

| Family / cost | Bot-only net | Shadow net | Delta | Bot DD | Shadow DD | Winner profit retained | Peak margin proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| range fade BASE | -629.00 | +948.37 | +1,577.37 | 1,261.68 | 0.00 | 100% | 103,187.19 → 68,746.83 |
| range fade STRESS | -609.38 | +932.02 | +1,541.40 | 1,233.66 | 0.00 | 100% | 103,149.39 → 68,745.10 |
| spike fade BASE | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | N/A | 0.00 → 0.00 |
| spike fade STRESS | +166.29 | +166.29 | 0.00 | 0.00 | 0.00 | 100% | 34,399.79 → 34,399.79 |

Ruin events remained 0 in all arms. Early-cut winner damage was JPY 0 because
the rule is an admission filter. Skipped winner profit was also JPY 0 in both
range-fade rooms.

## Decision

- `RANGE_FADE_LIMIT`: `PASS_DIAGNOSTIC_SHADOW_ONLY`. Both BASE and STRESS
  improve in the same direction, drawdown/ruin/margin do not worsen, and no
  winner profit is lost.
- `SPIKE_FADE`: `REJECT_NO_PAPER_APPLICATION`. Net improvement is zero.
- Promotion or automatic Paper mutation: none. The source is not globally
  untouched OOS, prior aggregate outcome exposure exists, and margin history
  is represented by a conservative entry-notional proxy.

The hourly Paper report now renders runtime health and profitability as
separate statuses and counts current positions, resting orders, and
counter-trend resting orders. This is visibility only; it does not mutate the
bot, broker, accepted decision chain, or any order.

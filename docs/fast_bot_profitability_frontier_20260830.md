# Fast-bot profitability frontier — 2026-08-30

## Outcome

The implementation now separates capital preservation from profitability. A sealed, zero-authority profitability evidence gate rejects negative expectancy and keeps positive-but-thin evidence out of the primary trading candidate set. It cannot grant live permission or invoke `LiveOrderGateway`.

No tested strategy is currently profitable enough to trade. The improvement is that the system no longer treats the least-bad negative cell or a concentrated positive sample as a deployable profit source.

## Chronological EUR/USD tests

Both searches used the same local OANDA EUR/USD M1 bid/ask truth from 2020-01 through 2026-07. Candidate selection used 2020-2023 training and 2024-2025 validation. The 2026 holdout remained unopened because no pre-holdout cell had positive PF and net result in both train and validation.

| Family | Best validation cell | Trades | Net pips | PF | Risk-scaled net | Risk-scaled PF | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Shock continuation | H1+H4 confirmation, 0.25R | 207 | -346.938839 | 0.531544 | -41.646269 | 0.509733 | Reject |
| Nonshock hourly | H1+H4 trend, TP15/SL10 | 350 | -178.500000 | 0.864926 | -57.120000 | 0.864926 | Reject |

Rejecting those two best negative validation cells avoids 525.438839 replay pips relative to executing them. This is loss avoidance, not earned profit.

## Exact AUD/JPY limit evidence

The existing `AUD_JPY/SHORT/BREAKOUT_FAILURE/LIMIT` exact-shape replay is evaluated without changing its historical truth.

| Evidence | Samples / days | Net pips | PF | Pessimistic expectancy | Positive-day rate | Max one-day share | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Exact shape | 135 / 6 | +116.3 | 1.278563 | -0.364079 | 33.33% | 91.11% | Reject negative tail expectancy |
| Rank-only precision subset | 40 / 2 | +113.5 | 2.318235 | +0.714694 | 50.00% | 95.00% | Collect more independent days |

The rank-only subset is the only remaining profit-source lead, but 38 of 40 samples occur on one day. It is retained only for prospective zero-authority shadow collection. The current gate requires at least 100 samples, 10 active days, PF 1.25, positive pessimistic expectancy, positive-day rate at least two-thirds, and maximum one-day share at most 70% before it can become a primary shadow observation candidate. Even that status remains separate from live admission.

## Authority boundary

- `execution_authority=NONE`
- `live_permission=false`
- `broker_mutation_allowed=false`
- `LiveOrderGateway` invocation count 0
- external order attempts 0
- external orders 0
- manual/tagless positions `NO_TOUCH`

No LaunchAgent, shadow runtime, approval receipt, release receipt, active policy, or live authority is changed by this work.

## Reproduction

1. Run `tools/analyze_fast_bot_shock_profitability.py` on the seven existing EUR/USD M1 BA files.
2. Run `tools/analyze_fast_bot_nonshock_profitability.py` on the same files.
3. Run `tools/build_fast_bot_profitability_frontier.py` with those two JSON results and `data/audjpy_limit_fresh_s5_bidask_replay.json`.

The frontier builder validates zero authority on both walk-forward inputs, seals the AUD/JPY evidence, and fails closed on tampering or nonzero order authority.

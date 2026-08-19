# L029/L030 — the last survivor, closed on its own pre-registered test

`docs/RESEARCH_LOG.md` L026 recorded the only hypothesis in this project that
ever passed three independent stages (L024 M1/2025-26 formation → L025 S5 real
bid/ask, pricing-independent → L026 2020-2023, time-independent):

> `mom_break@2880` × EFF20 bottom-67% quiet-side gate — +2.93 pips/day, one-sided
> 95% lower bound +0.20, positive in all four unseen years, 100th percentile vs
> the random null.

It also recorded, in its own words, exactly what had to happen next and what was
forbidden:

> **次に必要なのは2つ**: 1. 1日1行に集約した正しい下限（ペア×日の延べを使わない）
> 2. 前向きの未経過期間での確認
> **やってはいけないこと**: ここで分位や指標を増やして「もっと良い設定」を探すこと

Both were executed. The setting stayed frozen at EFF20 bottom-67% with the 2024
threshold; nothing was re-searched.

## Step 1 — the correct lower bound (L029)

L026's 1,686 "days" were pair×day rows. Several pairs trade the same calendar
day, so they are not independent. Aggregating to one row per day — the
portfolio's daily P&L, which is what the contract actually grades:

| aggregation | n | mean/day | median | one-sided 95% LB | win |
|---|---|---|---|---|---|
| pair×day (L026's) | 1,686 | +2.93 | −22.93 | **+0.20** | 41% |
| **one row per day** | **889** | **+5.56** | −18.15 | **+0.0044** | 43% |

The bound survives to the fourth decimal place. Supporting cuts:

- **bootstrap 95% CI on the daily series: [−0.91, +12.21]** — includes zero
- **every individual year has a negative lower bound**: 2020 −2.04, 2021 −2.03,
  2022 −2.73, 2023 −14.50 — and 2023's *mean* is −4.25
- **concentration**: the top 5 days out of 889 carry **53% of all profit**.
  Remove them and the bound is **−2.52**

The daily series has sd 100.7 pips against a mean of 5.56. For the lower bound
to clear zero you need n > (1.645 × 100.7 / 5.56)² ≈ **888 days**. The sample is
889. **It passed by one day.** That is not a result; it is the sample size
landing on the threshold.

## Step 2 — the forward window (L030)

The M1 corpus ends 2026-07-09. Fetched 2026-07-10 → 2026-08-19 for all 11 pairs
(read-only, written to a separate directory so no historical result changes) and
joined it to the corpus for warm-up, evaluating only forward dates.

| window | days | mean/day | median | LB | win days |
|---|---|---|---|---|---|
| W1 07-10 → 08-05 (out of corpus, already elapsed at lock) | 20 | **−29.97** | −35.32 | −63.22 | 25% |
| W2 08-06 → now (genuinely unelapsed at lock) | 8 | +9.02 | −9.20 | −66.22 | 50% |
| combined | 28 | **−18.83** | −30.90 | −51.33 | 32% |

**This is not by itself a rejection, and must not be reported as one.** Against
the historical distribution, a contiguous 28-day window at or below −18.83
occurs **9.2%** of the time — 1.28 sd from the historical mean. With sd 100.7,
28 days cannot separate anything. W2's n=8 is meaningless in either direction.

## Verdict

The gate is not rejected as an *improvement* — L026's four-year sign test still
stands, and the forward window is within noise. What fails is the question the
contract actually asks: **is the gated series itself profitable?** Once the
pair×day pooling error is corrected, the answer is a lower bound of +0.004 pips
resting on 5 days out of 889, with no single year positive at the bound.

So the last survivor was never a survivor. It was a pooling artifact sitting one
day above the minimum sample size.

## The number that closes the line of research

To confirm this edge at 95% takes ~888 daily observations. Forward, that is
**about three and a half years** of trading to learn whether +5.56 pips/day is
real — during which the position itself is the experiment, at a size that
produces 932 JPY/day on the full account (0.47%/day).

An edge that requires 3.5 years to distinguish from zero cannot be scaled into,
because you cannot know it is there until after you have risked the capital.
That, and not any single failed test, is what closes the in-hand price-rule
search: **the measurement cost exceeds what the edge can pay for.**

## What is untouched

Everything outside price rules, which the 2026-08-06 log already listed and
which none of this tested: order-book depth (no local data), event calendars
(FOMC/BOJ/payrolls), COT positioning (CFTC, free), and news. And the operator's
own decisions, for which the label count is still zero.

## Reproduce

```
research/paper/quiet_gate_daily_bound.py   # step 1, uses quiet_gate_cache.json
research/paper/fetch_forward_m1.py         # read-only OANDA fetch → forward_m1/
research/paper/quiet_gate_forward.py       # step 2
```

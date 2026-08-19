# The stated entry rule does not describe the actual entries

Asked what the scalp entry rule was, the operator answered:

> なんとなく、チャートの形と勢い。短いトレンド意識してるかも。ここ抜けたら走るみたいな。

That is a breakout description. Measured against all 18 historical scalp-intent
entries, using definitions fixed before looking (distance from the recent range
extreme in the trade's direction, ATR-normalised, from S5 bars):

| window | entries **beyond** the range edge | median position |
|---|---|---|
| 1 min | 39% | −0.49 ATR (inside) |
| 3 min | 28% | −2.10 |
| 5 min | 28% | −3.41 |
| 10 min | 22% | −4.34 |
| 20 min | 17% | −6.20 |
| 60 min (M1) | **0%** | −3.71 |

At every timescale the median entry sits **inside** the range, and the fraction
breaking out is at or below chance. Not one of the 18 broke the 60-minute range.
Directional momentum in the trade's favour runs 44–72% across windows — noise at
n=18.

The behaviour is **pullback entry inside a mild longer trend**, close to the
opposite of what was described. 60-minute momentum was with the trade 61% of the
time while the entry was taken several ATR back from the short-term extreme.

## Why this matters more than the finding itself

A bot built from the verbal description would have traded breakouts — a
different strategy from the one that produced +2.32 pips/trade. This is very
likely how the three live bot lanes acquired their invented entry rules, and
they lost 55,500 JPY over 540 closes.

Introspection is not a reliable source for this. So `tools/scalp_paper.py` now
captures the shape automatically at every decision — ENTER and SKIP alike —
recording ATR plus `break_Nm`, `mom_Nm` and `range_Nm_pips` at 1/3/5/10/20
minutes from live S5 bars. Cost is ~0.6 s per log, and it means the labels
contain what the market actually looked like rather than what it was later
remembered to look like.

## Limits

n = 18, one operator, mostly USD_JPY, five trading days. The direction of the
result is strong (0/18 at 60 min; median inside at every window) but the sample
cannot support a claim about *which* pullback conditions matter. That is what
the 300–500 labels of phase 2 are for.

The feature definitions are one reasonable reading of "抜ける". A different
reading — a trendline, a session high, a figure level, a level the operator drew
by hand — is not tested here and would not appear in these numbers. Recording
the shape at decision time is what lets that be settled later instead of argued.

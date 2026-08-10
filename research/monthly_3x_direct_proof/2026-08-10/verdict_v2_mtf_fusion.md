# Monthly 3x independent hypotheses checkpoint

Target remains frozen at 200,000 JPY to 600,000 JPY in 30 days after execution costs. No validation threshold was changed after seeing outcomes, and the sealed holdout was not read.

## V3 — exact X multi-timeframe contract

`Q-XFX-MTF-001` was executed on its frozen USDJPY S5 archive. Acquisition completeness failed before profitability could be judged: 162,697 S5 rows formed 3,093 observed M5 buckets, but only 563 were exact 60/60 buckets and only four H1 bars were exact 12/12 aggregates. No TRAIN or VALIDATION episode had the required three contiguous completed H1 bars. The result is `REJECT_OR_INSUFFICIENT_EVIDENCE`, caused by evidence coverage rather than demonstrated lack of edge.

## V4 — same MTF rule on the long OANDA cohort

The same completed-H1 structural-opposition rule was applied to AUDJPY, EURJPY, and EURUSD across the five frozen price-action families, three lookbacks, three sessions, and seven exits. It produced 544,548 eligible split/config trade observations, but no positive connected TRAIN plateau and no candidate stable in both 32-day and 64-day VALIDATION. Monthly 3x passes: zero.

## V5 — family-normalized fusion

To prevent three correlated lookbacks from receiving three votes, each strategy family was normalized to one opinion. One executable action was emitted only when at least two independent families agreed and none dissented. The side-correct OANDA bid/ask cost model and 0.5-spread adverse entry/exit slippage were retained. The engine emitted 8,351 decisions, but again produced no positive TRAIN plateau and no stable 32/64-day candidate. Monthly 3x passes: zero.

## Conclusion

The combined search now covers 8,100 V1/V2 condition rows plus the exact X MTF run, the long-window MTF run, and the decision-connected fusion engine. It still does **not** prove monthly 3x. Claiming otherwise would require selecting validation winners after outcomes, treating missing bars as evidence, removing costs, or exceeding the frozen margin/drawdown constraints.

This checkpoint does establish that the failure is not merely “the libraries were post-hoc only”: V4 changed admission decisions and V5 generated one fused executable decision per time. Neither produced a robust after-cost edge. The next independent source must add genuinely different predictive information or a different traded product; retuning these technical thresholds is closed.

# Operator-alpha fast rotation verdict

Status: **RESEARCH CONTRACT COMPLETE / LIVE ADOPTION BLOCKED**

The four consecutive manual wins are broker-confirmed at **+5,052.0833 JPY**
(**1.9874%** of
254209.0185 JPY).  They form a reproducible behavior shape:
pair-specific short direction, executable entry, fast after-cost harvest, close
confirmation, and fresh-evidence rotation.  This is evidence that the operator
performed the behavior, not proof of future expectancy.

| Arm | Selected | Net JPY | PF | Expectancy/decision | DD JPY | Mean hold sec | Turnover units |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASELINE_ACTUAL | 6 | -71147.9167 | 0.0663 | -11857.9861 | 76200.0000 | 43917.7 | 218007 |
| OPERATOR_ALPHA | 6 | -1306.7822 | 0.5954 | -217.7970 | 3229.4860 | 223.3 | 218007 |
| X_STRUCTURE | 2 | -76200.0000 | 0.0000 | -12700.0000 | 76200.0000 | 130352.3 | 85000 |
| X_OPERATOR_INTERACTION | 2 | -1020.0000 | 0.3462 | -170.0000 | 1560.0000 | 70.8 | 85000 |

`OPERATOR_ALPHA` derives its profit floor (532.0000 JPY)
and maximum holding time (1740 seconds) from the four
wins, then applies an equity-derived 0.25% loss budget and side-correct S5
bid/ask plus adverse half-spread exit stress to all six entries.  The result is
an in-sample diagnostic, not a validation result.  `X_STRUCTURE` adds only the
post's completed-H4/reasons/skip checklist; it does not import the post's
monthly-income or ten-minutes-per-day claims.

The two margin closeouts are broker-confirmed at -45720.0000 and
-30480.0000 JPY.  Together they equal
15.0829 observed four-win batches.
They are classified as contract failures: a fast scalp became a long,
margin-controlled hold.  Margin closeout is never an acceptable exit or a
reason to increase leverage.

## Adoption boundary

The fusion table returns `WAIT_EVIDENCE_INCOMPLETE` for live use because the
frozen packet lacks decision-time margin available/used, full inventory,
forecast lineage, and an executable unwind packet.  The currently open manual
or unknown position at entry fill 473207 remains `NO_TOUCH`.

The 10% and 50% figures in `target_arithmetic_v1.json` are arithmetic scenarios,
not guarantees.  They report required P/L, trade density, and break conditions.
No live, Paper, order, broker mutation, deployment, or holdout action occurred.

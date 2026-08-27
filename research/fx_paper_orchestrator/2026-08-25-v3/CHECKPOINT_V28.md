# V28 official paper checkpoint

V28 was preregistered and remotely read back at commit
`80e990fbd831bba2afaff0b740ed70bd120a8b6c` before official execution.
The restart-safe coordinator then performed official execution ordinal 1 once.
No diagnostic result was reused and no rerun occurred.

## Decision

- system acceptance: **passed**
- strategy profit gate: **failed / unproven**
- automatic disposition: **rejected**
- reason code: `BASKET_HOLD_RAW_EDGE_ABSENT`
- holdout: `FUTURE_FX_HOLDOUT_AFTER_2026_07_15` remains `UNOPENED`
- adoption authorized: `false`
- external orders: `0`

The one preregistered rule processed all 500 frozen V25 RAW signals with the
same ids, decision timestamps, fill timestamps, raw exits, directions, and
fixed 1/7 sleeves. RAW, BASE, and ADVERSE used the same execution-state
transitions. Same-pair/same-direction signals never added units or extended
expiry. Every period ended with zero inventory and realized terminal MTM.

## Walk-forward metrics

The walk-forward interval remained 2026-05-01 through 2026-07-01, with 230
RAW signals, 37 effective days, and 175 realized inventory episodes. The state
machine recorded 55 same-direction hold aggregations, 50 reversals, 37
max-age close/reopens, and 88 flat opens.

| Metric | RAW_SIGNAL | EXECUTABLE_BASE | ADVERSE_STRESS |
|---|---:|---:|---:|
| Equity multiple | 0.9826974228 | 0.9705364475 | 0.9616665442 |
| Gross edge, bps/episode | -6.9751614471 | -6.9751614471 | -6.9751614471 |
| Realized cost, bps/episode | 0.0000000000 | 5.0056553615 | 8.6841644861 |
| Net edge, bps/episode | -6.9751614471 | -11.9808168087 | -15.6593259332 |
| Break-even cost, bps/episode | -6.9751614471 | -6.9751614471 | -6.9751614471 |
| Direction accuracy | 0.4114285714 | 0.4114285714 | 0.4114285714 |
| Turnover, NAV | 50.0000000000 | 50.0000000000 | 50.0000000000 |
| Maximum drawdown | -0.0263657243 | -0.0356797785 | -0.0424782853 |
| Maximum gross exposure, NAV | 1.0000000000 | 1.0000000000 | 1.0000000000 |
| Conservative 1x margin notional, JPY | 200000 | 200000 | 200000 |
| Maximum inventory age, seconds | 313200 | 313200 | 313200 |
| Terminal inventory MTM | 0 | 0 | 0 |
| N_eff days / episodes | 37 / 175 | 37 / 175 | 37 / 175 |

Full comparable-month multiples were also below the immutable 2.0x gate:

| Month | EXECUTABLE_BASE | ADVERSE_STRESS |
|---|---:|---:|
| 2026-05 | 0.9817889676 | 0.9771581577 |
| 2026-06 | 0.9884543827 | 0.9840486562 |

The turnover reduction therefore did not uncover gross edge. No leverage,
evaluation-period, threshold, direction, or holdout change was made after the
result.

Relative to the frozen predecessors, V28 reduced walk-forward turnover from
V25's 65.7143 NAV to 50.0000 NAV, but it was higher than V27's selected-subset
10.5714 NAV. More importantly, RAW gross edge deteriorated from V25's
+1.4525 bps and V27's -0.8968 bps to V28's -6.9752 bps. BASE net edge moved
from V25 -0.9124 bps and V27 -2.4727 bps to V28 -11.9808 bps; ADVERSE net edge
moved from V25 -2.8127 bps and V27 -3.9007 bps to V28 -15.6593 bps. N_eff
days stayed fixed at 37. This is evidence against the preregistered hold rule,
not evidence for parameter adjustment after the result.

## Immutable evidence

- result file SHA-256: `be6914d6bef4268d39022cb134bbf9ab4fd72206f5b8fe980c05c64c919c343f`
- embedded result SHA-256: `d99fe491b28edbd465dc21d578441f027e97efd590cbb7ca1072af762d989dc7`
- proposal ledger SHA-256: `ce386c8fc9fc1a99fca82cd180f967fcfc26ea75fb170abd157edfa9f1c09ade`
- signal-id-set SHA-256: `4100dd95a74526fddee1a495a8a1bbe0d7568a6a5f5147cb048509a989f23f8e`
- official seal file SHA-256: `2a0004a6daea3cdb328b1dfddb762644901d87ae56ea97ec8fa23fb421a7ee93`
- embedded official seal SHA-256: `07247acd19060100d9b176cd35530c20c1cd4bbabe9597165b08a5f3e3db57ef`
- V29 work-order SHA-256: `46eafc1452f02080d6a7d3264febaa11a31b51ebd15b321edf49ec3781efb360`

## Verification and next work

The V28 rule and coordinator acceptance suites passed 35/35 before execution.
The dedicated full discovery ran 145 tests: 138 passed and the same seven
documented legacy import errors remained; none is on the V28 execution path.
Post-run coordinator audit returned `SEALED_SYSTEM_PASS_PROFIT_UNPROVEN`.

The generated V29 proposal is not registered or executable. It has
`reason_code=BASKET_HOLD_RAW_EDGE_ABSENT` and permits exactly one next changed
variable:
`one_preregistered_causal_basket_consensus_release_rule_preserving_all_v25_raw_signals_and_fixed_sleeves`.

Authority remained paper-only throughout: no live, broker-account, credential,
order-endpoint, deploy, or external-configuration authority.

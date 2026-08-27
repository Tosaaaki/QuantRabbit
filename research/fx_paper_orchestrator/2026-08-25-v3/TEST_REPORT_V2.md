# v2 verification report

All commands were paper/local and used `PYTHONDONTWRITEBYTECODE=1`. No live,
broker-account, credential, order, deploy, or external-configuration path was
used.

## Targeted

- frozen V25 detector: 6/6 passed
- v2 acceptance: 15/15 passed

## Coordinator

The combined frozen V25 + v1 coordinator + v2 coordinator run passed 26/26.
The v2 preflight then read and validated all seven registered source files:
completed BID/ASK rows only, strictly increasing timestamps, and exact hashes.

## Official V25 replay

The registry was committed and remotely read back before execution. The v2
coordinator performed official execution ordinal 1 once, returned code 0, and
sealed the result without diagnostic-result reuse.

- official result file SHA-256: `eb646f6933d1078e86a025fa28b941fa2ddfd20f1a885a61ced7bbc23bfdef45`
- result embedded SHA-256: `85b0e9836652993850d317ba09ebfa29e2c0c30d54faf4eafa8c556c186cff11`
- proposal ledger SHA-256: `4657f39f444529f8b267d2295485cf8ea66c8506fefdf50bdf1456a1acafc6db`
- signal-id-set SHA-256: `4100dd95a74526fddee1a495a8a1bbe0d7568a6a5f5147cb048509a989f23f8e`
- official embedded seal SHA-256: `07867edb4fd250f68c3c6ebac434afa4f88a6ddf003f004076773637329bb45b`
- external orders: `0`
- terminal inventory: `0` in every period and arm

Post-run `audit` returned `SEALED_SYSTEM_PASS_PROFIT_UNPROVEN`.

## Full research suite

The inherited sibling research dependencies were assembled in a disposable
`/tmp` mirror so the protected old source worktree remained unchanged. The
full v3 suite ran 107 tests: 106 passed and the one documented legacy V4
nanosecond fixture errored in
`test_counterparty_response_pipeline_v4.test_raw_signal_does_not_consult_cost`.
`datetime.fromisoformat` rejected
`2026-01-02T05:00:00.000000000+00:00`. Per migration policy, no V4 script,
fixture, result, or seal was changed.

The repository-wide prescribed suite also ran: 832 tests, with 11 failures,
152 import/setup errors, and 1 skip. The dominant pre-existing blocker was
`SpreadCalibrationError: spread calibration is expired`; two other observed
baseline assertions were `CLOSE_RECEIPT_REQUIRED` vs
`CLOSE_AUTHORIZATION_REQUIRED` and `MEMORY_HEALTH_BLOCKED` vs expected pass.
These failures are outside the dedicated research path and were not repaired or
included in this task's commit.

## Acceptance

- v2 system acceptance: passed
- restart-safe result sealing: passed
- strategy profit gate: failed / unproven
- unopened holdout reproduction: not performed
- strategy adoption: not authorized

## V28 preregistration checkpoint

V28 added one deterministic, training-only basket-hold rule while preserving
the frozen V25 500-signal RAW ledger and the rejected V27 evidence. Before any
official V28 replay, the new rule tests and coordinator acceptance tests passed
35/35, and coordinator `audit` reported V28 as
`REGISTERED_PREFLIGHT_PASS_PENDING`.

The dedicated full discovery ran 145 tests: 138 passed and exactly seven
legacy import errors remained. Five depend on the intentionally absent sibling
`research/llm_paper_experiment/2026-08-24-v250/run_expectancy_regression_v250.py`
fixture (`counterparty_response_pipeline_v4`, `mtf_tension_v3`,
`online_polarity_v3`, `v250_family_partial_holdout_v3`, and
`v250_partial_holdout_v3`). Two graph tests require the legacy
`aggregate_bars` export not present in the sealed paper replay compatibility
module (`graph_inventory_netting_v3` and `graph_residual_v3`). These are
migration evidence, not V28 strategy or coordinator failures; no legacy result,
fixture, compatibility seal, or V4-V24 evidence was changed.

## V28 official replay

After the preregistration commit and remote SHA readback, coordinator official
execution ordinal 1 completed once and sealed V28. Post-run targeted tests
remained 35/35 and coordinator audit returned
`SEALED_SYSTEM_PASS_PROFIT_UNPROVEN`. The strategy was automatically rejected
with `BASKET_HOLD_RAW_EDGE_ABSENT`; walk-forward multiples were RAW
0.9826974228, BASE 0.9705364475, and ADVERSE 0.9616665442. Terminal inventory
and terminal MTM were zero, the holdout remained unopened, and external orders
remained zero. The generated V29 work order is proposal-only and not executable.

## V29 preregistration checkpoint

V29 freezes one training-only, completed-data-only, cost-independent basket
consensus release formula. The structural selection inspected no prices,
returns, costs, direction accuracy, drawdown, monthly multiple, walk-forward
outcome, or holdout data. In the training window it preserves 202 RAW signals,
produces 11 deterministic releases, 153 episodes, and a maximum observed
inventory age of 313,200 seconds. All 500 V25 RAW identities and fixed 1/7
sleeves remain in the global ledger; V28 evidence remains frozen and is not
rerun.

Before official execution, the V29 targeted rule tests passed 8/8 and the
coordinator acceptance suite passed 27/27. The official replay remains blocked
until this preregistration checkpoint is committed, pushed, and read back by
remote SHA.

## V29 official replay

After remote readback of preregistration commit
`449e053312ba5721152eb3d810a365097440b6a6`, coordinator official execution
ordinal 1 ran exactly once and sealed V29 with system acceptance passed and the
strategy profit gate failed. The unchanged ledger contains all 500 V25 RAW
signals, all cost arms share one action/state transition stream, holdout remains
`UNOPENED`, terminal inventory and terminal MTM are zero, and external orders
are zero.

Walk-forward results were RAW `0.9824806484227545`, EXECUTABLE_BASE
`0.9703351079466831`, and ADVERSE_STRESS `0.9614944774663403`. Gross edge was
`-7.064810981455403` bps; realized costs were `0.0`, `5.000320971852313`, and
`8.667380289743601` bps respectively. Turnover was `50.0` NAV, direction
accuracy was `0.4114285714285714`, maximum drawdown ranged from
`-0.026580074375925133` to `-0.04264907079833424`, maximum currency exposure
was `0.9999999999999998` NAV, maximum inventory age was `313200.0` seconds,
and N_eff was 37 days / 175 episodes. The two full-month BASE multiples were
`0.9817889675954863` and `0.9882459699396912`; ADVERSE multiples were
`0.9771581577253762` and `0.9838692639246738`.

V29 was automatically rejected without alteration under reason code
`BASKET_CONSENSUS_RELEASE_RAW_EDGE_ABSENT`. The generated V30 proposal changes
one variable only:
`one_preregistered_causal_consensus_release_scope_rule_preserving_all_v25_raw_signals_and_fixed_sleeves`.

Post-run V29 plus coordinator tests passed 35/35 and the mechanical result
verification passed. Dedicated full discovery ran 154 tests: 147 passed and
the same seven documented legacy import errors remained; no V29 test failed.

## V30 preregistration checkpoint

V30 freezes exactly one training-only, completed-data-only, cost-independent
change to V29: peer membership is restricted to simultaneous signals attached
to already-open inventory with the same signed USD exposure as the target.
Direction, unanimity, the two-peer minimum, exact timestamp requirement,
self-pair exclusion, own-signal handling, fixed 1/7 sleeves, 48-hour nominal
hold, 96-hour hard max-age, terminal liquidation, all 500 V25 RAW identities,
cost arms, evaluation periods, leverage, and unopened holdout remain frozen.

Plain USD-node adjacency was rejected structurally because all seven pairs are
USD crosses and would reproduce V29's market-wide peer set. The selected
inventory-induced subgraph inspected no price, return, cost, accuracy,
drawdown, monthly multiple, walk-forward outcome, or holdout data. In the
training window it preserves 202 signals and 11 releases while removing
market-wide peer members at 6 of those 11 release timestamps, proving that the
scope is not a naming-only change.

Before official execution, V30 targeted rule tests passed 9/9 and the
coordinator acceptance suite passed 28/28. Coordinator `audit` reported V30 as
`REGISTERED_PREFLIGHT_PASS_PENDING`. The official replay remains blocked until
this preregistration checkpoint is committed, pushed, and read back by remote
SHA.

Dedicated full discovery ran 164 tests: 157 passed and the same seven
documented legacy import errors remained. Five require the absent sibling V250
fixture and two require the legacy `aggregate_bars` compatibility export. No
V30 or coordinator test failed, and no frozen legacy artifact was changed.

## V30 official replay

After remote readback of preregistration commit
`2e397b18783f1c84e144901c49ac794782a1b945`, coordinator official execution
ordinal 1 ran exactly once. Mechanical system acceptance passed: all 500 V25
RAW identities, directions, and decision timestamps remain unchanged; all cost
arms share the same action/state-transition stream; actual completed BID/ASK,
slippage, latency, financing, currency inventory, finite max-age, terminal
liquidation, walk-forward boundaries, and rerun prohibition were validated.
Terminal inventory and terminal MTM are zero, holdout remains `UNOPENED`, and
external orders remain zero.

The strategy profit gate failed. Walk-forward multiples were RAW
`0.9824806484227545`, EXECUTABLE_BASE `0.9703351079466831`, and
ADVERSE_STRESS `0.9614944774663403`. Gross edge was
`-7.064810981455403` bps; realized costs were `0.0`, `5.000320971852313`,
and `8.667380289743601` bps; net edges were `-7.064810981455403`,
`-12.065131953307716`, and `-15.732191271199007` bps. Turnover was `50.0`
NAV, break-even cost was `-7.064810981455403` bps, direction accuracy was
`0.4114285714285714`, and maximum drawdown was `-0.026580074375925133`,
`-0.03587931299291747`, and `-0.04264907079833424`. Maximum gross exposure
was `1.0` NAV, maximum currency exposure was `0.9999999999999998` NAV,
maximum 1x margin requirement was `200000.0` JPY, maximum inventory age was
`313200.0` seconds, N_eff was 37 days / 175 episodes, and the scoped release
count was 2. The narrowed membership removed nonlocal peers at both
walk-forward release events but produced the same release/action stream and
the same outcomes as V29.

The full-month BASE multiples were `0.9817889675954863` and
`0.9882459699396912`; ADVERSE multiples were `0.9771581577253762` and
`0.9838692639246738`. V30 was automatically rejected without alteration under
`CONSENSUS_RELEASE_SCOPE_RAW_EDGE_ABSENT`. The sealed V31 proposal changes one
variable only:
`one_preregistered_causal_consensus_release_persistence_confirmation_rule_preserving_all_v25_raw_signals_and_fixed_sleeves`.

Result file SHA-256 is
`27e9b006ba115214d8f2590bf21add290c1df406e0565717016584265a2d0bd8`,
ledger SHA-256 is
`3d7e60bc8322133d78d7f5cdec65defca3204306d5a08500fde76e9e4ce6ad51`,
and the official embedded seal SHA-256 is
`65dcc55d13c204a3d993fac0b3d615f33aae06c56df379355eeea15bfe5238a9`.
Post-run V30 plus coordinator tests passed 37/37. Dedicated full discovery
again ran 164 tests: 157 passed and the same seven legacy import errors
remained; no V30 or coordinator test failed.

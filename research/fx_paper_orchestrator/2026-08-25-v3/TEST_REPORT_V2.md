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

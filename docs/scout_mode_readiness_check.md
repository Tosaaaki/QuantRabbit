# Scout Mode Readiness Check

- Generated: `2026-07-06T13:36:34Z`
- Candidate: `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`

## Classification

- Current readiness: `NOT_APPROVED`
- Contract state: `READY_FOR_OPERATOR_APPROVAL`
- Next required state: `APPROVAL_REQUIRED`
- Execution state: `BLOCKED`

The contract is ready for the operator to review, but it is not approved and it is not executable.

## Why Blocked

- Exact operator approval text is missing.
- `LIVE_READY=0`.
- `PROOF_READY=0`.
- Normal routing remains `BLOCKED`.
- AUD_JPY SHORT BREAKOUT_FAILURE LIMIT remains `EVIDENCE_GAP`.
- Fresh max loss JPY cap has not been recalculated.
- RiskEngine pass is missing.
- LiveOrderGateway pass is missing.
- GPT verifier pass is missing.
- Guardian/operator review pass is missing.
- Profitability blockers remain proof-collection risk, not `LIVE_READY`.

## Reference Quote

The local snapshot shows AUD_JPY bid `112.455`, ask `112.471`, timestamp `2026-07-06T08:54:03.755665+00:00`. This is reference only and stale for execution. Any approved scout run must fetch a fresh quote and recalculate max loss JPY. The diagnostic estimate is about `125 JPY` at `1000u`.

## Expected Outcome Now

- No order.
- No execution flags.
- No `LIVE_READY`.
- Normal routing remains `BLOCKED`.
- EUR_USD `472987` and TP `472996` remain protected.

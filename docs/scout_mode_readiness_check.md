# Scout Mode Readiness Check

- Generated: `2026-07-06T15:07:56Z`
- Candidate: `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`

## Classification

- Current readiness: `STALE_CURRENT_PACKET`
- Contract state: `STALE_NOT_READY_FOR_OPERATOR_APPROVAL`
- Next required state: `SELECT_NEXT_CANDIDATE_OR_COLLECT_NEW_PROOF`
- Execution state: `BLOCKED`

The prior scout contract is stale against the refreshed live packet. It is not ready for operator approval and it is not executable.

## Why Blocked

- Candidate is absent from refreshed live order_intents.
- AUD_JPY DOWN S5 bid/ask direction evidence is `REJECTED_NEGATIVE_EXPECTANCY`.
- AUD_JPY SHORT BREAKOUT_FAILURE LIMIT scout contract is stale and not ready for approval.
- Exact operator approval text is missing as a standalone operator approval directive.
- `LIVE_READY=0`.
- `PROOF_READY=0`.
- Normal routing remains `BLOCKED`.
- AUD_JPY SHORT BREAKOUT_FAILURE LIMIT exact proof remains `EVIDENCE_GAP` and not live-grade.
- Fresh max loss JPY cap has not been recalculated against the `200 JPY` cap.
- RiskEngine pass is missing.
- LiveOrderGateway pass is missing.
- GPT verifier pass is missing.
- Guardian/operator review pass is missing.
- Profitability blockers remain proof-collection risk, not `LIVE_READY`.

## Required Approval Text

`I approve one AUD_JPY SHORT BREAKOUT_FAILURE LIMIT proof-collection scout, max loss 200 JPY, units 1000, this run only.`

The exact text appearing as a required string in the objective is not counted as operator approval.

## Reference Quote

The refreshed read-only snapshot shows AUD_JPY bid `112.644`, ask `112.66`, timestamp `2026-07-06T15:00:47.339013+00:00`. This is reference only because the candidate is stale and absent from current order_intents. Any future approved scout run must regenerate a current candidate, fetch a fresh quote, and recalculate max loss JPY. The old diagnostic estimate of about `125 JPY` at `1000u` is historical only, not executable proof.

## Expected Outcome Now

- No order.
- No execution flags.
- No `LIVE_READY`.
- Normal routing remains `BLOCKED`.
- EUR_USD `472987` and TP `472998` remain protected.

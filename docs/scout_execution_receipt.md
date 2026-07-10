# Scout Execution Receipt

- Generated: `2026-07-06T15:07:56Z`
- Active: `false`; historical evidence only, superseded by schema v2 in `config/predictive_scout_policy.json`.
- Candidate: `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`
- Mode: proof-collection scout
- Execution state: `BLOCKED`
- Order sent: `false`
- Broker transaction id: `none`

## Decision

No order was staged or sent. The old scout contract is stale against refreshed live order_intents, and the exact approval text was present only as the required approval string in the objective, not as a standalone operator approval directive for this run.

Required approval text:

`I approve one AUD_JPY SHORT BREAKOUT_FAILURE LIMIT proof-collection scout, max loss 200 JPY, units 1000, this run only.`

## Gate Result

- Approval detected: `false`
- Fresh broker snapshot: `PASS_READ_ONLY_2026-07-06T15:00:47.452920+00:00`
- Fresh AUD_JPY quote: `REFERENCE_ONLY_2026-07-06T15:00:47.339013+00:00`
- Spread acceptable: `NOT_RUN_STALE_CANDIDATE`
- RiskEngine pass: `false`
- LiveOrderGateway pass: `false`
- GPT verifier pass: `false`
- Guardian/operator-review pass: `false`
- Execution flags scout-only: `false`
- Max loss <= `200 JPY`: `NOT_RECALCULATED_APPROVAL_MISSING`
- Units <= `1000`: `true`
- Order type: `LIMIT`
- Candidate id exact: `false`
- Candidate id status: `ABSENT_FROM_CURRENT_ORDER_INTENTS`

## Local Safety Evidence

- Local broker snapshot: `2026-07-06T15:00:47.452920+00:00`
- Local `last_transaction_id`: `472998`
- AUD_JPY quote in local snapshot: bid `112.644`, ask `112.66`, quote timestamp `2026-07-06T15:00:47.339013+00:00`
- Quote freshness for execution: `false`
- EUR_USD `472987`: `operator_manual` / `OPERATOR_MANUAL` / `KEEP`
- TP `472998`: `TAKE_PROFIT`, `PENDING`, price `1.1361`
- EUR_USD `472987` and TP `472998` touched by this run: `false`

## Safety

- Normal routing created: `false`
- `LIVE_READY` marked: `false`
- Execution flags enabled: `false`
- Market order allowed: `false`
- Retry / averaging / same-theme add: `false`
- Position close / order cancel / SL/TP modification: `false`

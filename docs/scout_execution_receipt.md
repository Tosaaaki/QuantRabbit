# Scout Execution Receipt

- Generated: `2026-07-06T14:39:46Z`
- Candidate: `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`
- Mode: proof-collection scout
- Execution state: `BLOCKED`
- Order sent: `false`
- Broker transaction id: `none`

## Decision

No order was staged or sent. The exact approval text was present only as the required approval string in the objective, not as a standalone operator approval directive for this run.

Required approval text:

`I approve one AUD_JPY SHORT BREAKOUT_FAILURE LIMIT proof-collection scout, max loss 200 JPY, units 1000, this run only.`

## Gate Result

- Approval detected: `false`
- Fresh broker snapshot: `NOT_RUN_APPROVAL_MISSING`
- Fresh AUD_JPY quote: `NOT_RUN_APPROVAL_MISSING`
- Spread acceptable: `NOT_RUN_APPROVAL_MISSING`
- RiskEngine pass: `false`
- LiveOrderGateway pass: `false`
- GPT verifier pass: `false`
- Guardian/operator-review pass: `false`
- Execution flags scout-only: `false`
- Max loss <= `200 JPY`: `NOT_RECALCULATED_APPROVAL_MISSING`
- Units <= `1000`: `true`
- Order type: `LIMIT`
- Candidate id exact: `true`

## Local Safety Evidence

- Local broker snapshot: `2026-07-06T08:53:59.140771+00:00`
- Local `last_transaction_id`: `472996`
- AUD_JPY quote in local snapshot: bid `112.455`, ask `112.471`, quote timestamp `2026-07-06T08:54:03.755665+00:00`
- Quote freshness for execution: `false`
- EUR_USD `472987`: `operator_manual` / `OPERATOR_MANUAL` / `KEEP`
- TP `472996`: `TAKE_PROFIT`, `PENDING`, price `1.13968`
- EUR_USD `472987` and TP `472996` touched by this run: `false`

## Safety

- Normal routing created: `false`
- `LIVE_READY` marked: `false`
- Execution flags enabled: `false`
- Market order allowed: `false`
- Retry / averaging / same-theme add: `false`
- Position close / order cancel / SL/TP modification: `false`

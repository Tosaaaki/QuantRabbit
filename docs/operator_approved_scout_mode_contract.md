# Operator Approved Scout Mode Contract

- Generated: `2026-07-06T15:07:56Z`
- Candidate: `failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT`
- Mode: proof-collection scout
- Status: stale contract only; no execution permission.

## Current State

- `LIVE_READY=0`
- `PROOF_READY=0`
- Normal routing: `BLOCKED`
- Candidate classification: `STALE_REJECTED_CURRENT_PACKET`
- Candidate present in refreshed order_intents: `false`
- Current packet reason: `AUD_JPY SHORT BREAKOUT_FAILURE LIMIT is absent from refreshed live order_intents; broader AUD_JPY DOWN S5 bid/ask evidence is REJECTED_NEGATIVE_EXPECTANCY.`
- No order, cancel, close, SL/TP modification, execution flag, or broker-state mutation is authorized by this contract.

## Scout Contract

- Order type: `LIMIT` only.
- No market chase. `MARKET` and `STOP-ENTRY` are prohibited for this approval.
- One order only.
- Default units: `1000u` unless RiskEngine computes a lower allowed size or no-trade.
- Max loss JPY cap is `200 JPY` for the exact approval text and must still be recalculated from a fresh broker snapshot and fresh AUD_JPY quote before any approved run. The current diagnostic estimate is about `125 JPY` at `1000u`, derived from `375 JPY` at `3000u`; this is reference only.
- No averaging.
- No same-theme add.
- No auto retry.
- No automatic TP/SL modification after placement unless separately approved.

## Required Preflight

Every item must pass after a fresh candidate contract and exact operator approval text are present:

- Fresh broker snapshot. Latest checked snapshot: `2026-07-06T15:00:47.452920+00:00`.
- Fresh AUD_JPY quote.
- Candidate present in current order_intents.
- RiskEngine pass with live-send validation.
- LiveOrderGateway pass.
- GPT verifier pass.
- Guardian/operator-review pass.
- Profitability blockers acknowledged as proof-collection risk, not `LIVE_READY`.
- Manual EUR_USD `472987` protected, with TP `472998` not touched.

## Stop Conditions

Stop without staging or sending if any condition appears:

- Stale quote.
- Spread too wide.
- Forecast mismatch.
- RiskEngine fail.
- Gateway fail.
- GPT verifier fail.
- Guardian/operator review missing.
- Execution flags not explicitly approved.
- Broker `last_transaction_id` unexpected.

## Required Approval Text

`I approve one AUD_JPY SHORT BREAKOUT_FAILURE LIMIT proof-collection scout, max loss 200 JPY, units 1000, this run only.`

The text must match exactly, and this stale contract still cannot execute unless regenerated against a current candidate. Approval for this scout does not approve any retry, same-theme add, averaging, market chase, TP/SL modification, EUR_USD `472987` action, TP `472998` action, or later run.

## Sources

- Local: `data/audjpy_short_breakout_failure_limit_proof_pack.json`
- Local: `data/audjpy_limit_live_ready_decision.json`
- Local: `data/order_intents.json`
- Local: `data/guardian_receipt_consumption.json`
- Local: `data/guardian_receipt_operator_review.json`
- Local: `data/broker_snapshot.json`
- Notion read-only reference: `quant-rabbit-profitability-evidence-repair-2026-07-03` (`392f1c8e-53a7-81af-a467-d6ce4635b5cd`)

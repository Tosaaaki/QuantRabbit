# Dry-Run Certification Report

- Generated at UTC: `2026-05-04T02:53:25.241953+00:00`
- Status: `BLOCKED`
- Checks: `7`
- Blockers: `3`

## Blockers

- coverage optimization still has blockers
- execution replay receipt is missing
- latest GPT trader decision was rejected

## Checks

- `BLOCK` coverage: coverage optimization still has blockers
- `BLOCK` execution_replay: execution replay receipt is missing
- `PASS` post_trade_learning: learning status READY_FOR_REVIEW
- `PASS` order_intents: 10 LIVE_READY intents have required contracts
- `PASS` live_order: no entry send was requested
- `PASS` position_execution: no position write was requested
- `BLOCK` gpt_decision: latest GPT trader decision was rejected

## Certification Contract

- Certification is dry-run only and does not enable live trading.
- Any artifact showing a live send blocks certification.
- Coverage, replay, and learning receipts must exist before live expansion.

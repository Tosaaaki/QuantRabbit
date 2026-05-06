# Position Management

## Use When

- Trader-owned position exists.
- Trader-owned position is missing TP or SL.
- Trader-owned pending entry needs cancel review.
- Target reached and protection-first behavior is active.

## Valid Actions

- `PROTECT`
- `TIGHTEN_SL`
- `CLOSE`
- `CANCEL_PENDING`
- `WAIT`

## Protection Rules

- Missing TP / SL on trader-owned exposure is a repair requirement.
- Operator-managed manual/tagless positions are observed only.
- Existing SL must not be widened.
- Existing TP is not moved by the protection gateway.
- Profitable protected positions may tighten SL to break-even or better.
- Contradicted trader-owned positions may close.
- Fresh entries are blocked only by non-layerable trader-owned or external exposure; protected trader-owned exposure may add only through portfolio validation.

## Pending Orders

- Pending entries are inherited across scheduler handoff.
- Do not cancel another cycle's pending order without an explicit reason in the next decision receipt.
- `CANCEL_PENDING` must list current OANDA order ids in `cancel_order_ids`.
- Manual/tagless pending orders are observed only.

## Commands

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json

./scripts/run-autotrade-live.sh \
  --reuse-market-artifacts \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send
```

## Receipt Notes

- Cite `broker:snapshot`, affected trade/order ids, and the exact repair reason.
- If choosing WAIT with open trader-owned exposure, state why no gateway action is required now.
- If closing, cite the contradiction in current broker/market packet evidence.

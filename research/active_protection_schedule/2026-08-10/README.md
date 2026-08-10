# ACTIVE_PROTECTION_SCHEDULE_V1

This research-only checkpoint reconstructs the broker-causal TP/SL schedule for
the frozen 251 `ACTUAL_AFTER_COST` episodes. It does not change live/Paper/order
behavior and does not evaluate or tune any exit arm.

## Reproduce

```bash
python3 test_schedule.py
python3 build_schedule.py
python3 verify_oracle.py
python3 test_oanda_extension.py
python3 validate_oanda_extension.py
```

`build_schedule.py` opens `data/execution_ledger.db` with SQLite `mode=ro`.
Protection identity comes from the original OANDA transaction fields, because
the normalized `execution_events.order_id` stores the old order id on 456
replacement-create rows.

The bounded OANDA extension was fetched only through the repository's read-only
instrument-candles client:

```bash
PYTHONPATH=src python3 scripts/oanda_history_fetch.py \
  --pairs AUD_JPY,EUR_JPY,EUR_USD --granularities S5 --price BA \
  --from 2026-07-09T00:39:37Z --to 2026-07-09T07:46:03Z \
  --output-dir research/active_protection_schedule/2026-08-10/oanda_extension \
  --compress
```

The same request was repeated under `oanda_extension_repeat`; gzip container
hashes differ because publication metadata differs, while expanded content
hashes and row counts match exactly for all three pairs.

## Truth boundary

- OANDA transaction JSON is the protection/order/terminal identity truth.
- OANDA S5 bid/ask candles are executable candle truth, not raw tick truth.
- Missing S5 timestamps remain unresolved. They are not forward-filled and are
  not called no-trade intervals without raw-tick completeness proof.
- `LINKED_TRADE_CLOSED` cancellations occur at/after terminal and do not erase
  the protection state immediately before the close.
- No realized outcome is used to choose or synthesize an active order.

Official candle schema reference:
<https://developer.oanda.com/rest-live-v20/instrument-df/>

See `verdict_v1.md` for the checkpoint decision and remaining blockers.

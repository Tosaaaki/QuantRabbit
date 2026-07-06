# Manual EUR_USD TP Replacement Provenance

- Generated: `2026-07-06T06:08:36Z`
- Manual trade: `472987`
- Audit order: `472994` classified `PROVENANCE_UNKNOWN_BLOCK_AUTOMATION` with lifecycle `REPLACED`
- Active broker TP: `472996` at `1.13968` classified `PROVENANCE_UNKNOWN_BLOCK_AUTOMATION`
- Snapshot last transaction: `472996` fetched `2026-07-06T05:51:56.023182+00:00`
- Gateway receipt search: `NO_LOCAL_QUANTRABBIT_GATEWAY_RECEIPT_FOUND`
- Can automation use or modify this TP: `False`

## Current Broker Truth

- Active TP order `472996` replaces `472994`.
- No live side effects in this run: `[]`

## Replacement Chain

| order | lifecycle | price | replaces | replaced by | class | gateway receipts |
|---|---|---:|---|---|---|---:|
| `472988` | `REPLACED` | 1.1388 | `None` | `472994` | `PROVENANCE_UNKNOWN_BLOCK_AUTOMATION` | 0 |
| `472994` | `REPLACED` | 1.136 | `472988` | `472996` | `PROVENANCE_UNKNOWN_BLOCK_AUTOMATION` | 0 |
| `472996` | `ACTIVE_BROKER_TRUTH` | 1.13968 | `472994` | `None` | `PROVENANCE_UNKNOWN_BLOCK_AUTOMATION` | 0 |

## Conclusion

Broker truth shows EUR_USD trade 472987 is operator-manual and the active TP is protected from automation. Order 472994 is historical/replaced by 472996, and neither 472994 nor 472996 has a local QuantRabbit gateway receipt. Classify TP provenance as unknown and block automation from using, modifying, or inferring permission from it.

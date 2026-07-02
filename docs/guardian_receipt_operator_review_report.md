# Guardian Receipt Operator Review Report

- Generated at UTC: `2026-07-02T07:34:31+00:00`
- Status: `GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED_CURRENT_P0_BLOCKS_ROUTING`
- Normal routing allowed: `False`
- Current P0/P1 blocks routing: `True`
- no_live_side_effects: `True`
- Live side effects: `0`
- Broker truth checked at UTC: `2026-07-02T07:33:42.324184+00:00`

## Reviewed Receipt

| Event | Action | Lifecycle | Original Issue | Operator Decision | Same-Event Emergency Active | Normal Routing For This Receipt | Expires UTC | Reason |
|---|---|---|---|---|---|---|---|---|
| `832d2908eeb84b2f` | `REDUCE` | `EXPIRED` | `GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW` | `OPERATOR_ACKNOWLEDGED_HISTORICAL` | `False` | `True` | `2026-07-03T07:34:31+00:00` | original USD_JPY unknown/gateway-outside emergency no longer exists in current broker truth or current guardian events; receipt is historical evidence only |

## Broker Truth Check

- Open positions: `AUD_USD` `LONG` `14000` units, owner=`unknown`, trade_id=`472965`, SL=`null`, TP=`0.69038`.
- Pending orders: `TAKE_PROFIT` order_id=`472966` for trade_id=`472965`.
- USD_JPY positions/orders: none in the read-only broker snapshot.
- Original USD_JPY trade_id `472944`: not open.
- Exact USD_JPY REDUCE dedupe `USD_JPY|GATEWAY_OUTSIDE_BROKER_POSITION|UNKNOWN_ORDER|REDUCE`: absent from current guardian events/escalation.
- Active REDUCE / HARVEST / CANCEL_PENDING receipt for event `832d2908eeb84b2f`: none observed.

## Current Routing Blockers

| Severity | Issue | Pair | Event | Dedupe | Routing |
|---|---|---|---|---|---|
| `P0` | `CURRENT_GUARDIAN_P0_UNKNOWN_EXPOSURE` | `AUD_USD` | `aafaf3622a11c9c7` | `AUD_USD\|GATEWAY_OUTSIDE_BROKER_POSITION\|UNKNOWN_ORDER\|REDUCE` | `blocked` |
| `P1` | `CURRENT_GUARDIAN_P1_SPREAD_ANOMALY` | `AUD_USD` | `3ffe24ccc8b74a56` | `AUD_USD\|SPREAD_ANOMALY_SAFETY_TRIGGER\|SPREAD_ANOMALY\|HOLD` | `blocked` |

## Boundary

- This artifact acknowledges only the expired historical USD_JPY REDUCE receipt.
- It does not place orders, cancel orders, close positions, enable execution flags, or modify broker state.
- Ordinary fresh `TRADE`, `ADD`, and `campaign_exposure_recovery` remain blocked while the current AUD_USD P0/P1 issues remain.
- Existing-position protection, reporting, and operator-requested fresh review paths remain available through their own safety paths.

# Guardian Receipt Operator Review Report

- Generated at UTC: `2026-07-02T04:37:47+00:00`
- Status: `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- Normal routing allowed: `False`
- no_live_side_effects: `True`
- Live side effects: `0`

## Classifications

| Event | Action | Lifecycle | Original Issue | Operator Decision | Normal Routing | Generated UTC | Expires UTC | no_live_side_effects | Reason |
|---|---|---|---|---|---|---|---|---|---|
| `832d2908eeb84b2f` | `REDUCE` | `EXPIRED` | `GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW` | `OPERATOR_REQUESTS_KEEP_BLOCKED` | `False` | `2026-07-02T04:37:47+00:00` | `2026-07-03T04:37:47+00:00` | `True` | Expired REDUCE guardian receipt for event 832d2908eeb84b2f is historical but has not been explicitly cleared by a local operator decision file; ordinary fresh-entry routing remains blocked pending review. |

## Boundary

- This artifact records operator review state for expired/historical guardian receipts.
- It does not place orders, cancel orders, close positions, enable execution flags, or modify broker state.
- The current decision intentionally keeps ordinary fresh-entry routing blocked until a valid local operator decision file explicitly clears the reviewed receipt and broker truth has no active emergency condition for the same event.

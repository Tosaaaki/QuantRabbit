## 2026-04-19 23:34 UTC — GBP_JPY SHORT closed

| Field | Value |
|------|-------|
| Pair | GBP_JPY |
| Side | SHORT |
| Trade ID | 469062 |
| Close price | 214.593 |
| P&L | -654 JPY |
| Reason | `zombie_hold` |
| Why | 51h hold, live M1/M5 kept bidding against the stale short, and I would not enter it fresh here |

## 2026-04-19 23:34 UTC — EUR_USD LONG LIMIT armed

| Field | Value |
|------|-------|
| Pair | EUR_USD |
| Side | LONG |
| Order type | LIMIT |
| Order ID | 469072 |
| Units | 3000 |
| Entry | 1.17400 |
| TP | 1.17540 |
| SL | 1.17300 |
| GTD | 2026-04-20 06:00 UTC |
| Thesis | Late-session direct-USD reload. H4 early-bull reset + M5 squeeze floor made EUR_USD cleaner than defending the stale GBP_JPY SHORT or chasing a fresh JPY-cross fade. |

## 2026-04-19 23:42:59 UTC — EUR_USD LONG LIMIT filled

| Field | Value |
|------|-------|
| Pair | EUR_USD |
| Side | LONG |
| Order ID | 469072 |
| Trade ID | 469073 |
| Fill price | 1.17400 |
| TP | 1.17540 |
| SL | 1.17300 |
| Why | The prior session's structural reload filled after SESSION_END. This session synced the exact OANDA fill time into the record instead of leaving the book as a fake flat state. |

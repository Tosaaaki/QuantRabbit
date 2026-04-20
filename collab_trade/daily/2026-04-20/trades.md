## 2026-04-20 00:07:44 UTC — EUR_USD LONG protection widened

| Field | Value |
|------|-------|
| Pair | EUR_USD |
| Side | LONG |
| Trade ID | 469073 |
| Action | Modify TP/SL |
| TP | 1.17540 (replaced as order 469077) |
| SL | 1.17286 (replaced as order 469079) |
| Why | The inherited 1.17300 stop was a noise stop. Hold stayed valid only with a structural H1 swing-low stop after the fill was synced into the live book. |

## 2026-04-20 00:46:42 UTC — EUR_USD LONG Tokyo breakout reclaim

| Field | Value |
|------|-------|
| Pair | EUR_USD |
| Side | LONG |
| Trade ID | 469083 |
| Order ID | 469082 |
| Units | 3000 |
| Entry price | 1.17562 |
| TP | 1.17710 (order 469084) |
| SL | 1.17458 (order 469085) |
| Pretrade | Edge B (4/10) / Allocation B / MARKET |
| Why | The live book was flat, `EUR_USD` was the only direct-USD seat that cleared both the breakout-reclaim chart read and pretrade, and `GBP_USD SHORT` failed the learning gate as a no-edge counterfade. |

## 2026-04-20 03:06:14 UTC — EUR_USD LONG zombie shelf scratch

| Field | Value |
|------|-------|
| Pair | EUR_USD |
| Side | LONG |
| Trade ID | 469083 |
| Action | Close |
| Close price | 1.17569 |
| Realized P&L | +33.2964 JPY |
| Why | The reclaim never expanded beyond a shelf squeeze, the hold had crossed into zombie territory under adverse H1 structure, and the paid edge had already rotated into the `AUD_JPY` box. |

## 2026-04-20 03:10:13 UTC — AUD_JPY SHORT structural stop widen

| Field | Value |
|------|-------|
| Pair | AUD_JPY |
| Side | SHORT |
| Trade ID | 469094 |
| Action | Modify TP/SL |
| TP | 113.580 (order 469095, unchanged) |
| SL | 113.800 (order 469096, widened from 113.760) |
| Why | The upper-edge short was valid, but the inherited 6pip stop was a thin-window noise stop inside the live range. The honest protection was the first structural level at H1 BB-mid / 113.800. |

## 2026-04-20 05:44:01 UTC — AUD_JPY LONG lower-box fill

| Field | Value |
|------|-------|
| Pair | AUD_JPY |
| Side | LONG |
| Trade ID | 469101 |
| Order ID | 469088 |
| Units | 3000 |
| Entry price | 113.600 |
| TP | 113.700 |
| SL | 113.540 |
| Pretrade | Counter-B (2/7) / Allocation B / LIMIT |
| Why | The live Tokyo box finally tagged the structural lower edge, so the pre-armed opposite-side receipt became a real floor probe instead of prose-only backup. |

## 2026-04-20 05:44:35 UTC — AUD_JPY SHORT upper-box TP auto fill

| Field | Value |
|------|-------|
| Pair | AUD_JPY |
| Side | SHORT |
| Trade ID | 469094 |
| Action | Close |
| Close price | 113.580 |
| Realized P&L | +600.0000 JPY |
| Why | The upper-box fade completed the planned 113.70 -> 113.58 rotation and closed automatically at the attached take-profit. |

## 2026-04-20 06:08:21 UTC — USD_JPY LONG lower-box backup armed

| Field | Value |
|------|-------|
| Pair | USD_JPY |
| Side | LONG |
| Order ID | 469106 |
| Action | LIMIT entry |
| Units | 3000 |
| Entry price | 158.900 |
| TP | 159.000 |
| SL | 158.810 |
| GTD | 2026-04-20 14:08 UTC |
| Why | The book was flat after the AUD_JPY floor probe failed, and the only clean cheap-spread receipt left was the quiet USD_JPY lower-box retest. |

## 2026-04-20 06:09:14 UTC — AUD_JPY LONG floor-break invalidation cut

| Field | Value |
|------|-------|
| Pair | AUD_JPY |
| Side | LONG |
| Trade ID | 469101 |
| Action | Close |
| Close price | 113.556 |
| Realized P&L | -132.0000 JPY |
| Why | The 113.56 floor stopped acting like support on bodies, so the lower-box probe was closed instead of defended. |

## 2026-04-20 06:25:53 UTC — EUR_USD SHORT structural retest backup armed

| Field | Value |
|------|-------|
| Pair | EUR_USD |
| Side | SHORT |
| Order ID | 469111 |
| Action | LIMIT entry |
| Units | 3000 |
| Entry price | 1.17570 |
| TP | 1.17470 |
| SL | 1.17610 |
| GTD | 2026-04-20 11:55 UTC |
| Pretrade | Edge B (5/10) / Allocation B / LIMIT |
| Why | Direct-USD downside was still the cleanest backup lane, but only from a broken-shelf retest; arming the limit closed the backup seat as a real receipt instead of prose-only watchlist. |

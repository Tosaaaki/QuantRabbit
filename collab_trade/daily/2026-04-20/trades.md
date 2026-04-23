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

## 2026-04-20 19:21:35 UTC — AUD_JPY LONG continuation backup armed

| Field | Value |
|------|-------|
| Pair | AUD_JPY |
| Side | LONG |
| Order ID | 469216 |
| Action | STOP entry |
| Units | 3000 |
| Entry price | 114.060 |
| TP | 114.200 |
| SL | 113.920 |
| GTD | 2026-04-21 03:30 UTC |
| Pretrade | Edge A (6/10) / Allocation B / STOP-ENTRY |
| Why | The live M5/M1 continuation remained the cleanest cross backup, but the current quote was still a late chase. `pretrade_check` validated the seat only as trigger-honest continuation, so the backup lane was closed as a real STOP order instead of prose-only trend admiration. |

## 2026-04-20 20:06:29 UTC — AUD_JPY LONG continuation stop fill

| Field | Value |
|------|-------|
| Pair | AUD_JPY |
| Side | LONG |
| Trade ID | 469217 |
| Order ID | 469216 |
| Action | Entry (STOP fill) |
| Units | 3000 |
| Entry price | 114.061 |
| TP | 114.200 (order 469218) |
| SL | 113.920 (order 469219) |
| Pretrade | Edge A (6/10) / Allocation B / STOP-ENTRY |
| Why | The proof-price continuation finally traded through 114.060, so the trigger-honest backup lane became real inventory. M1 JPY then flipped offered across the crosses, which kept the fill valid as live risk instead of a stale pending thesis. |

## 2026-04-20 22:08:45 UTC — EUR_USD LONG late squeeze runner banked

| Field | Value |
|------|-------|
| Pair | EUR_USD |
| Side | LONG |
| Trade ID | 469209 |
| Action | Close |
| Close price | 1.17869 |
| Realized P&L | +261.4713 JPY |
| Why | The upper-half squeeze runner had already done its work, only 1-2 pip remained to TP, and I would not re-buy the live quote. The honest action was to bank the stale runner and re-arm the original first-defense instead of holding inertia. |

## 2026-04-20 22:08:56 UTC — AUD_JPY LONG failed breakout cut

| Field | Value |
|------|-------|
| Pair | AUD_JPY |
| Side | LONG |
| Trade ID | 469217 |
| Action | Close |
| Close price | 113.981 |
| Realized P&L | -240.0000 JPY |
| Why | The 114.06 continuation stop had already failed back into the range, M1 never reclaimed the highs, and the current quote was no longer an honest continuation buy. |

## 2026-04-20 22:09:15 UTC — EUR_USD LONG first-defense re-armed

| Field | Value |
|------|-------|
| Pair | EUR_USD |
| Side | LONG |
| Order ID | 469229 |
| Action | LIMIT entry |
| Units | 3000 |
| Entry price | 1.17815 |
| TP | 1.17893 |
| SL | 1.17758 |
| GTD | 2026-04-21 04:30 UTC |
| Pretrade | Edge B (4/10) / Allocation B / LIMIT |
| Why | After flattening both stale live runners, the only honest late-session receipt left was the normal-spread EUR_USD first-defense reload. Arming the exact shelf again kept the book in the core direct-USD lane without paying the late upper-half quote. |

## Reconstructed Runtime Appendix — 2026-04-20 06:43 UTC onward

Recovered verbatim from `logs/live_trade_log.txt` after an in-session journal repair. This preserves the full runtime receipt stream from the point where the committed journal stops.

```text
[2026-04-20 06:43:02 UTC] ENTRY USD_JPY LONG 3000u @158.900 id=469115 TP=159.000 SL=158.810 Sp=0.8pip | LIMIT_FILL from id=469106 | tag=trader | comment=backup_lane_m5_lower_box_retest
[2026-04-20 06:43:02 UTC] CLOSE EUR_USD SHORT 3000u @1.17612 P/L=-200.6094JPY Sp=0.8pip reason=stop_loss_order id=469112 txn=469118
[2026-04-20 06:49:50 UTC] ENTRY EUR_USD SHORT 3000u @1.17615 id=469121 TP=1.17540 SL=1.17635 Sp=0.8pip | LIMIT_FILL from id=469120 | thesis=tokyo_range_upper_rearm | pretrade=B(4/10) allocation=B tag=trader
[2026-04-20 07:07:47 UTC] PLACE_LIMIT AUD_JPY SHORT 5000u @113.700 TP=113.580 SL=113.800 id=469126 tag=trader reason=backup_lane structural=aud_jpy_upper_box_retest
[2026-04-20 07:12:49 UTC] ENTRY AUD_JPY SHORT 5000u @113.700 id=469127 TP=113.580 SL=113.800 Sp=1.6pip | LIMIT_FILL from id=469126 | tag=trader | comment=backup_lane_aud_jpy_upper_box_retest
[2026-04-20 07:13:39 UTC] CLOSE USD_JPY LONG 3000u @158.810 P/L=-270.0000JPY Sp=0.8pip reason=STOP_LOSS_ORDER id=469115 txn=469130
[2026-04-20 08:07:50 UTC] ENTRY AUD_JPY SHORT 3000u @113.630 id=469135 TP=113.550 SL=113.760 Sp=1.6pip | thesis=london_aud_jpy_market_scout | pretrade=B(5/10) allocation=B tag=trader
[2026-04-20 08:07:50 UTC] PLACE_LIMIT AUD_JPY SHORT 3000u @113.700 TP=113.580 SL=113.800 id=469138 tag=trader reason=reload_lane structural=aud_jpy_upper_box_retest
[2026-04-20 08:41:33 UTC] ENTRY AUD_JPY SHORT 3000u @113.700 id=469139 TP=113.580 SL=113.800 Sp=1.6pip | LIMIT_FILL from id=469138 | tag=trader | comment=london_aud_jpy_upper_box_reload
[2026-04-20T09:05:36Z] ENTRY_ORDER EUR_JPY LONG 3000u @186.950 id=469142 TP=187.080 SL=186.880 GTD=2026-04-20T13:05:36Z Sp=2.0pip pretrade=B allocation=B tag=trader thesis=london_eur_jpy_pullback_backup_limit
[2026-04-20T09:06:30Z] LIMIT_FILL EUR_JPY LONG 3000u @186.950 id=469143 (LIMIT id=469142 filled) TP=187.080 SL=186.880 Sp=2.0pip | thesis=london_eur_jpy_pullback_backup_limit | tag=trader
[2026-04-20 09:27:02 UTC] CLOSE EUR_JPY LONG 3000u @186.878 P/L=-216.0000JPY Sp=1.9pip reason=STOP_LOSS_ORDER id=469143 txn=469146
[2026-04-20 09:57:38 UTC] CLOSE AUD_JPY SHORT 3000u @113.761 P/L=-393.0000JPY Sp=1.6pip reason=STOP_LOSS_ORDER id=469135 txn=469148
[2026-04-20 10:06:11 UTC] CLOSE AUD_JPY SHORT 3000u @113.761 P/L=-183.0000JPY Sp=1.6pip reason=lid_acceptance_breakout id=469139
[2026-04-20 10:25:42 UTC] ENTRY_ORDER GBP_USD LONG 3000u @1.35160 id=469154 TP=1.35210 SL=1.35074 GTD=2026-04-20T14:30:42Z Sp=1.3pip pretrade=B allocation=B tag=trader thesis=london_gbp_usd_breakout_stop
[2026-04-20 10:25:42 UTC] PLACE_LIMIT GBP_USD LONG 3000u @1.35105 TP=1.35210 SL=1.35074 id=469155 tag=trader reason=reload_lane structural=old_box_lid_retest
[2026-04-20 10:59:39 UTC] ENTRY GBP_USD LONG 3000u @1.35161 id=469156 via STOP-ENTRY fill | TP=1.35210 SL=1.35074 | pretrade=B(learning-cap) | conviction=A edge / B allocation | thesis=GBP M15/M1 BID, M5 TREND-BULL breakout above 1.3510 lid | Sp=1.3pip
[2026-04-20 11:08Z] LIMIT EUR_USD LONG 3000u @1.17660 TP=1.17780 SL=1.17560 GTD=15:30Z id=469159 Sp=0.8pip pretrade=B(4) thesis=H4_floor_M5_pullback_buy conviction=B
[2026-04-20 11:14:50 UTC] CLOSE GBP_USD LONG 3000u @1.35210 P/L=+233.1217JPY Sp=1.3pip reason=TAKE_PROFIT_ORDER id=469156 txn=469160
[2026-04-20 11:55:53 UTC] ENTRY EUR_USD LONG 3000u @1.17660 id=469162 TP=1.17780 SL=1.17560 Sp=0.8pip | LIMIT_FILL from id=469159 | thesis=H4_floor_M5_pullback_buy | pretrade=B(4) allocation=B tag=trader
[2026-04-20 12:10:47 UTC] ENTRY GBP_USD LONG 3000u @1.35105 id=469165 TP=1.35210 SL=1.35074 Sp=1.3pip | LIMIT_FILL from id=469155 | thesis=old_lid_reload_after_london_tp | pretrade=B(learning-cap) allocation=B tag=trader
[2026-04-20 12:12:05 UTC] CLOSE GBP_USD LONG 3000u @1.35074 P/L=-148.0986JPY Sp=1.3pip reason=STOP_LOSS_ORDER id=469165 txn=469168
[2026-04-20 12:12:37 UTC] CLOSE EUR_USD LONG 3000u @1.17560 P/L=-477.7556JPY Sp=0.8pip reason=STOP_LOSS_ORDER id=469162 txn=469170
[2026-04-20 12:16:19 UTC] PLACE_LIMIT EUR_USD LONG 3000u @1.17480 TP=1.17620 SL=1.17420 id=469172 tag=trader reason=post_stop_lower_floor_reload
[2026-04-20T13:06:49Z] ENTRY_ORDER AUD_JPY SHORT 3000u @113.704 id=469174 TP=113.620 SL=113.780 GTD=2026-04-20T19:06:48Z Sp=1.6pip pretrade=B allocation=B tag=trader thesis=london_aud_jpy_breakdown_stop
[2026-04-20 13:08:38 UTC] ENTRY AUD_JPY SHORT 3000u @113.704 id=469175 via STOP-ENTRY fill | TP=113.620 SL=113.780 | pretrade=B(4/10) allocation=B | thesis=london_aud_jpy_breakdown_stop | Sp=1.6pip
[2026-04-20 13:06:36Z] CANCEL_ORDER EUR_USD LONG 3000u @1.17480 id=469172 tag=trader reason=same_pair_contest_long_dead_cleaner_live_short
[2026-04-20 13:26:26 UTC] CLOSE AUD_JPY SHORT 3000u @113.720 P/L=-48.0000JPY Sp=1.6pip reason=failed_break_reclaim id=469175
[2026-04-20T13:35:56Z] ENTRY_ORDER USD_JPY SHORT 3000u @158.640 id=469182 TP=158.420 SL=158.840 GTD=2026-04-20T18:35:56Z Sp=0.8pip pretrade=B allocation=B tag=trader thesis=usd_jpy_trigger_honest_break
[2026-04-20 13:37:48 UTC] ENTRY USD_JPY SHORT 3000u @158.640 id=469183 via STOP-ENTRY fill | TP=158.420 SL=158.840 | pretrade=B(5/10) allocation=B | thesis=usd_jpy_trigger_honest_break | Sp=0.8pip
[2026-04-20T15:52:01Z] ENTRY_ORDER USD_JPY SHORT 3000u @158.548 id=469188 TP=158.420 SL=158.705 GTD=2026-04-20T20:30:00Z Sp=0.8pip pretrade=B allocation=B tag=trader thesis=usd_jpy_squeeze_rebreak_stop
[2026-04-20 16:20:19 UTC] ENTRY EUR_USD LONG 3000u @1.17878 id=469190 TP=1.17995 SL=1.17795 Sp=0.8pip | thesis=direct_usd_continuation_shelf_hold | pretrade=B(5/10) allocation=B tag=trader
[2026-04-20 16:35:16 UTC] CLOSE EUR_USD LONG 3000u @1.17828 P/L=-238.4570JPY Sp=0.8pip reason=no_confirmation id=469190
[2026-04-20 17:06:11 UTC] LIMIT EUR_USD LONG 3000u @1.17820 TP=1.17905 SL=1.17760 GTD=2026-04-20T23:06:11Z id=469197 Sp=0.8pip pretrade=B(5) thesis=ny_eur_usd_first_retest_limit conviction=B tag=trader
[2026-04-20 17:06:11 UTC] ENTRY_ORDER USD_JPY SHORT 3000u @158.616 id=469198 TP=158.500 SL=158.705 GTD=2026-04-20T23:06:11Z Sp=0.8pip pretrade=B allocation=B tag=trader thesis=usd_jpy_live_shelf_break_stop
[2026-04-20 17:06:20Z] CANCEL_ORDER USD_JPY SHORT 3000u @158.548 id=469188 tag=trader reason=tighten_to_live_shelf_break
[2026-04-20 17:30:33 UTC] ENTRY EUR_USD LONG 3000u @1.17820 id=469200 TP=1.17905 SL=1.17760 Sp=0.8pip | LIMIT_FILL from id=469197 | thesis=ny_eur_usd_first_retest_limit | pretrade=B(5) allocation=B tag=trader
[2026-04-20 18:20:45 UTC] CLOSE EUR_USD LONG 3000u @1.17882 P/L=294.5582JPY Sp=0.8pip reason=upper_range_pre_waller_hold_test_failed id=469200
[2026-04-20 18:49:47Z] CANCEL_ORDER USD_JPY SHORT 3000u @158.616 id=469198 tag=trader reason=wish_distance_cleanup_after_usd_repricing
[2026-04-20 18:50:09 UTC] LIMIT EUR_USD LONG 3000u @1.17815 TP=1.17893 SL=1.17758 GTD=2026-04-21T00:50:00.000000000Z id=469208 Sp=0.8pip pretrade=B(5) thesis=post_waller_eur_usd_first_defense conviction=B tag=trader
[2026-04-20 18:50:14 UTC] ENTRY EUR_USD LONG 3000u @1.17814 id=469209 TP=1.17893 SL=1.17758 Sp=0.8pip | LIMIT_FILL from id=469208 | thesis=post_waller_eur_usd_first_defense | pretrade=B(5) allocation=B tag=trader
[2026-04-20 19:06:56 UTC] MODIFY EUR_USD LONG 3000u trade=469209 TP 1.17893(replaced 469210->469213) SL 1.17657(replaced 469211->469215) reason=structural_widen_h1_bb_mid Sp=0.8pip
```

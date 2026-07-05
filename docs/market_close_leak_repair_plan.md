# Market Close Leak Repair Plan

- Generated: `2026-07-05T17:21:46Z`
- Blocker: `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`
- Fresh entries blocked: `True`

## Evidence

| trade_id | pair | side | strategy | entry | exit | P/L JPY | close reason | attribution | campaign recovery | count/exclude |
|---|---|---|---|---|---|---:|---|---|---|---|
| 470356 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-07T09:43:08.055483437Z @ 1.1776 | 2026-05-07T22:30:31.532281538Z @ 1.17282 | -751.3958 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | NO_CAMPAIGN_EXPOSURE_RECOVERY_EVIDENCE_IN_LEDGER | COUNT_AGAINST_SYSTEM_EDGE |
| 470353 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-07T09:30:02.798746356Z @ 1.1769 | 2026-05-07T22:30:31.824952384Z @ 1.17282 | -1924.0762 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | NO_CAMPAIGN_EXPOSURE_RECOVERY_EVIDENCE_IN_LEDGER | COUNT_AGAINST_SYSTEM_EDGE |
| 470730 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-11T13:51:06.495419415Z @ 1.17857 | 2026-05-12T15:33:20.090528634Z @ 1.17268 | -8378.521 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | NO_CAMPAIGN_EXPOSURE_RECOVERY_EVIDENCE_IN_LEDGER | COUNT_AGAINST_SYSTEM_EDGE |
| 471174 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-14T10:17:34.844024422Z @ 1.17084 | 2026-05-14T13:20:30.859309142Z @ 1.16946 | -1092.1868 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | NO_CAMPAIGN_EXPOSURE_RECOVERY_EVIDENCE_IN_LEDGER | COUNT_AGAINST_SYSTEM_EDGE |
| 471089 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-14T03:39:30.919225941Z @ 1.17146 | 2026-05-14T13:20:31.099893312Z @ 1.16946 | -2216.0032 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | NO_CAMPAIGN_EXPOSURE_RECOVERY_EVIDENCE_IN_LEDGER | COUNT_AGAINST_SYSTEM_EDGE |
| 471255 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-15T02:50:16.273370788Z @ 1.16542 | 2026-05-18T00:52:01.622782140Z @ 1.16102 | -700.6068 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | NO_CAMPAIGN_EXPOSURE_RECOVERY_EVIDENCE_IN_LEDGER | COUNT_AGAINST_SYSTEM_EDGE |
| 472280 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-06-11T22:19:59.142911796Z @ 1.15761 | 2026-06-11T23:47:54.025605457Z @ 1.15758 | -28.8767 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | NO_CAMPAIGN_EXPOSURE_RECOVERY_EVIDENCE_IN_LEDGER | COUNT_AGAINST_SYSTEM_EDGE |

## Count Check

- Capture market-close losses: `7`
- Published blocker IDs: `470730, 471089, 470353, 471174, 470356`
- Ledger reconciled loss IDs: `470356, 470353, 470730, 471174, 471089, 471255, 472280`
- Ledger extra not in published examples: `471255, 472280`

## Repair

- Ban loss-side SYSTEM_GATEWAY MARKET_ORDER_TRADE_CLOSE on TP-proven HARVEST lanes unless a durable close-gate packet proves thesis invalidation and contained risk.
- Do not use operator/manual positions as system close evidence or system P/L repair material.
- Do not replace attached broker TP with discretionary market-close leakage on the same lane.

Required evidence before market close:
- fresh broker quote and spread snapshot at close decision time
- position owner and lane provenance linking the close request to a system-owned trade
- hard close-gate evidence: thesis invalidation, risk containment, and no same-direction support conflict
- execution-timing post-close replay showing market close preserves edge versus attached TP/HARVEST
- ledger receipt tying ORDER_ACCEPTED TRADE_CLOSE to the original system lane

TP-edge protection:
- Prefer attached broker TAKE_PROFIT_ORDER and TP-progress profit-capture paths for proven shapes.
- Reject fresh entries for a lane if its historical market-close loss net exceeds its TP edge.
- Keep market-close loss examples counted against system edge until capture_economics and profitability_acceptance clear.
- Require a fresh replay/capture packet before increasing exposure on EUR_USD LONG BREAKOUT_FAILURE.


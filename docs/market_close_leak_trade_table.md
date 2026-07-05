# Market Close Leak Trade Table

- Generated: `2026-07-05T12:34:08Z`
- MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE present: `True`
- RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK present in current acceptance: `False`
- Fresh entries blocked: `True`

## Contributing Trades

| trade_id | pair | side | strategy | entry | close | P/L JPY | close reason | attribution | campaign recovery | counts against system edge | regression coverage |
|---|---|---|---|---|---|---:|---|---|---|---|---|
| 470356 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-07T09:43:08.055483437Z @ 1.1776 | 2026-05-07T22:30:31.532281538Z @ 1.17282 | -751.3958 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | no | True | behavioral coverage only; this live trade id is not a fixture |
| 470353 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-07T09:30:02.798746356Z @ 1.1769 | 2026-05-07T22:30:31.824952384Z @ 1.17282 | -1924.0762 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | no | True | behavioral coverage only; this live trade id is not a fixture |
| 470730 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-11T13:51:06.495419415Z @ 1.17857 | 2026-05-12T15:33:20.090528634Z @ 1.17268 | -8378.521 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | no | True | behavioral coverage only; this live trade id is not a fixture |
| 471174 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-14T10:17:34.844024422Z @ 1.17084 | 2026-05-14T13:20:30.859309142Z @ 1.16946 | -1092.1868 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | no | True | behavioral coverage only; this live trade id is not a fixture |
| 471089 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-14T03:39:30.919225941Z @ 1.17146 | 2026-05-14T13:20:31.099893312Z @ 1.16946 | -2216.0032 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | no | True | behavioral coverage only; this live trade id is not a fixture |
| 471255 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-05-15T02:50:16.273370788Z @ 1.16542 | 2026-05-18T00:52:01.622782140Z @ 1.16102 | -700.6068 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | no | True | behavioral coverage only; this live trade id is not a fixture |
| 472280 | EUR_USD | LONG | BREAKOUT_FAILURE | 2026-06-11T22:19:59.142911796Z @ 1.15761 | 2026-06-11T23:47:54.025605457Z @ 1.15758 | -28.8767 | MARKET_ORDER_TRADE_CLOSE | SYSTEM_GATEWAY | no | True | behavioral coverage only; this live trade id is not a fixture |

## Recent Gateway Loss Leak

- Current contributing trades: `0`
- Status: `stale_or_runtime_mention_only_in_current_evidence`
- Note: Current profitability_acceptance does not raise this blocker; stale mentions remain non-permission evidence.

## Mitigation Families

| family | fix type | banned path | evidence to allow again | files/modules | tests |
|---|---|---|---|---|---|
| `TP_PROVEN_BREAKOUT_FAILURE_SYSTEM_MARKET_CLOSE` | `CODE_FIX, ROUTING_GATE, REPLAY_FILTER` | loss-side SYSTEM_GATEWAY MARKET_ORDER_TRADE_CLOSE on TP-proven HARVEST/BREAKOUT_FAILURE lanes without durable close-gate evidence | fresh quote/spread packet at close decision time, system-owned trade provenance and lane receipt, thesis invalidation plus contained-risk close-gate proof, 744h execution timing replay showing the market-close path is non-negative versus attached TP/profit capture, profitability_acceptance clears MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE | `src/quant_rabbit/gpt_trader.py, src/quant_rabbit/automation.py, src/quant_rabbit/profitability_acceptance.py, src/quant_rabbit/capture_economics.py, src/quant_rabbit/trader_support_bot.py, src/quant_rabbit/strategy/trader_brain.py` | `tests/test_gpt_trader.py, tests/test_profitability_acceptance_replay_repair.py, tests/test_capture_economics.py, tests/test_trader_brain.py` |
| `RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK` | `EVIDENCE_GAP, ROUTING_GATE` | recent loss-side gateway market-close path if refreshed acceptance re-raises the blocker | recent loss-side gateway market closes are zero, or each loss close has contained-risk timing evidence plus durable close-gate receipt, profitability_acceptance refresh keeps the blocker absent | `src/quant_rabbit/profitability_acceptance.py, src/quant_rabbit/execution_timing_audit.py, src/quant_rabbit/trader_support_bot.py` | `tests/test_profitability_acceptance_replay_repair.py, tests/test_execution_timing_audit.py, tests/test_trader_support_bot.py` |
| `OPERATOR_MANUAL_EXCLUSION` | `MANUAL_EXCLUSION` | using operator/manual EUR_USD 472987 as system P/L, system occupancy, auto-close, SL attach, TP modification, or same-theme add evidence | explicit operator reclassification and new proof packet; current artifact says OPERATOR_MANUAL / KEEP | `src/quant_rabbit/broker_snapshot.py, src/quant_rabbit/trader_support_bot.py, src/quant_rabbit/position_manager.py, src/quant_rabbit/risk_engine.py` | `tests/test_trader_support_bot.py, tests/test_position_manager.py, tests/test_risk_engine.py` |

## Gate Definitions

- `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`: No TP-proven segment remains net-damaged by loss-side system MARKET_ORDER_TRADE_CLOSE, and refreshed acceptance no longer raises the blocker. Can create A/S permission: `False`
- `RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK`: Refresh acceptance keeps the blocker absent, or if re-raised, every recent loss-side gateway market close is removed or proven contained-risk. Can create A/S permission: `False`

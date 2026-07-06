# Broker Mutation Bypass Audit

- Generated: `2026-07-06T09:13:36Z`
- Mode: `read_only_broker_mutation_bypass_audit`
- Permission boundary: This audit does not create live permission. No row may enable orders, cancels, closes, SL/TP changes, execution flags, or broker mutation.
- Dev broker truth: `last_transaction_id=472996`; `472987 EUR_USD SHORT 30000 owner=operator_manual TP=1.13968 SL=None`; pending TP `472996 @ 1.13968`.
- Live broker truth: `last_transaction_id=472996`; refresh status `OK`; pending TP `472996 @ 1.13968`.
- Protected IDs: trade `472987`; TP order lineage `472988/472994/472996`.

## Conclusion

The TP-rebalance manual-position leak is contained and the adjacent PositionManager/PositionProtectionGateway bypass is fixed. `auto_tp_modify_allowed=false` now blocks upstream TP proposals and downstream broker TP replacement. `472987` and active TP `472996` remain untouched, and no transaction advanced past `472996` in the audited broker truth.

Normal routing remains `BLOCKED`; `LIVE_READY=0`; `AUD_JPY SHORT BREAKOUT_FAILURE LIMIT` remains the only A/S proof candidate and is `REPAIR_REQUIRED` / `DRY_RUN_BLOCKED`.

## Direct Wrapper Search

| wrapper | HTTP mutation | status | note |
|---|---|---|---|
| `src/quant_rabbit/broker/oanda.py:OandaExecutionClient.post_order_json` | `POST /orders` | `SAFE` | Policy enforcement is caller-side; static search shows production callers go through LiveOrderGateway. |
| `src/quant_rabbit/broker/oanda.py:OandaExecutionClient.cancel_order` | `PUT /orders/{id}/cancel` | `SAFE` | Production callers are automation cancel paths with send/live and verified trader-owned pending ids. |
| `src/quant_rabbit/broker/oanda.py:OandaExecutionClient.replace_trade_dependent_orders` | `PUT /trades/{id}/orders` | `SAFE` | Production callers are PositionProtectionGateway, tp_rebalancer, trailing_sl; operator manual TP opt-out is now guarded in gateway and rebalancer. |
| `src/quant_rabbit/broker/oanda.py:OandaExecutionClient.close_trade` | `blocked before IO` | `BLOCKED` | Raises RuntimeError; approved paths must use close_trade_with_provenance. |
| `src/quant_rabbit/broker/oanda.py:OandaExecutionClient.close_trade_with_provenance` | `PUT /trades/{id}/close` | `SAFE` | Accepts only approved provenance labels before network IO. |

## Mutation Path Matrix

| path | file/function | mutation | status | gateway receipt use | manual flag checks | dry-run/live guard |
|---|---|---|---|---|---|---|
| `fresh_entry_live_order_gateway` | `src/quant_rabbit/broker/execution.py:LiveOrderGateway.run/run_batch` | `POST broker entry order` | `SAFE` | Required: LiveOrderGateway writes stage/send receipt and execution ledger events; autotrade/guardian handoff must arrive through verified GPT/gateway context. | RiskEngine blocks same-theme adds against 472987 when same_theme_auto_add_allowed=false unless operator_authorized_manual_overlap=true. | No broker post unless send=true, live/confirm gates pass, RiskEngine/StrategyProfile/freshness/self-improvement/GPT/guardian/target-path/SL-lint gates pass. |
| `pending_order_cancel` | `src/quant_rabbit/automation.py:AutoTradeCycle pending_cancel_order_ids/_cancel_gpt_pending_orders` | `PUT broker pending-order cancel` | `SAFE` | Accepted CANCEL_PENDING receipt or accepted TRADE receipt with verified cancel_order_ids; gpt_trader verifies ids against current trader-owned pending entries. | Only visible trader-owned pending ids or forced self-improvement review ids can pass; manual/operator ids fail closed. | send && live_enabled required; current-thesis/live-ready pending preservation can prevent cancel. |
| `position_gateway_dependent_order_replace` | `src/quant_rabbit/broker/position_execution.py:PositionProtectionGateway._plan_action/run` | `PUT broker dependent SL/TP order replace` | `FIXED` | PositionProtectionGateway writes position-execution receipt with sent/blocked action rows before/after broker mutation. | Manual/tagless SL writes and market closes blocked; operator_manual TP replace blocked when auto_tp_modify_allowed=false; TP side/entry/quote validation still runs. | send=true and live_enabled=true required for broker mutation; plan/action issues block before IO. |
| `position_manager_manual_tp_repair` | `src/quant_rabbit/strategy/position_manager.py:PositionManager._manage_manual_take_profit_position` | `Could propose dependent TP repair/replace` | `FIXED` | No broker call; output is position-management JSON/report consumed by PositionProtectionGateway. | auto_tp_modify_allowed=false returns ACTION_HOLD_SL_FREE with no recommended TP/SL; manual path never proposes SL/loss close. | No broker IO in this module; missing TP repair also requires QR_ENABLE_MISSING_TP_REPAIR=1 for non-opt-out manual packets. |
| `tp_rebalancer_dependent_order_replace` | `src/quant_rabbit/strategy/tp_rebalancer.py:compute_all_tp_adjustments/apply_tp_adjustments` | `PUT broker take-profit replace` | `SAFE` | CLI writes dry-run/send adjustment report; no GPT receipt grants permission. | auto_tp_modify_allowed=false skip; SL never touched; external skipped. | dry_run default prevents broker call; send path must explicitly disable dry-run/live gate at CLI level. |
| `trailing_sl_dependent_order_replace` | `src/quant_rabbit/strategy/trailing_sl.py:apply_trailing_sls` | `PUT broker stop-loss tighten` | `SAFE` | Trailing report only; no entry/close receipt can bypass owner and existing-SL checks. | Owner must be trader and an existing broker SL must already exist; SL-free 472987 skipped. | dry_run prevents broker mutation; QR_DISABLE_TRAILING_SL live default disables path. |
| `position_gateway_market_close` | `src/quant_rabbit/broker/position_execution.py:PositionProtectionGateway.run; src/quant_rabbit/automation.py:_close_gpt_trades` | `PUT broker trade close` | `SAFE` | Accepted GPT CLOSE Gate A/B or structural explicit opt-in feeds PositionProtectionGateway; close receipt/provenance is required. | manual/tagless/operator market close forbidden; 472987 cannot be loss-closed by this path. | send=true and live_enabled=true required; raw close_trade fallback raises before IO; spread close gate must pass. |
| `adverse_partial_close` | `src/quant_rabbit/strategy/adverse_partial_close.py:apply_partial_closes` | `PUT broker adverse partial close` | `SAFE` | CLI/report path only; operator-only margin relief path, not scheduled live gateway. | manual/unknown/operator_manual skipped before action generation. | Dry-run by default; live requires --send --confirm-live and QR_LIVE_ENABLED=1; raw close fallback blocked. |
| `profit_partial_close` | `src/quant_rabbit/strategy/profit_partial_close.py:apply_profit_partial_closes` | `PUT broker profitable partial close` | `SAFE` | CLI/report path with milestone state; profit-side only and not a loss-close receipt. | manual/operator allowed only when already profitable and ATR milestone proof exists; loss-side forbidden. | send=true, live_enabled=true, --confirm-live required; raw close fallback blocked. |
| `guardian_action_cycle` | `src/quant_rabbit/guardian_action_cycle.py:run_guardian_action_cycle; tools/guardian_wake_dispatcher.py:_maybe_gateway_handoff` | `Potential entry handoff only through LiveOrderGateway` | `SAFE` | Accepted guardian receipt plus strict receipt validation; TRADE/ADD handoff revalidates RiskEngine and LiveOrderGateway. | manual loss close forbidden without explicit operator_manual_loss_close_authorized; manual overlap entry requires explicit operator_manual_overlap_authorized. | Default off: QR_LIVE_ENABLED=1, QR_GUARDIAN_WAKE_GATEWAY_HANDOFF=1, and QR_GUARDIAN_ACTION_EXECUTE=1 required. |

## Current Evidence

- `data/order_intents.json`: total `84`, status counts `{'DRY_RUN_BLOCKED': 84}`, `LIVE_READY=0`.
- A/S board: generated `2026-07-06T09:04:03Z`, `normal_routing_status=BLOCKED`, `as_live_ready_path_exists=False`.
- AUD_JPY candidate blockers: `RANGE_FORECAST_REQUIRES_RANGE_ROTATION, SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH, BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE, GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`.
- Profitability blocker reconciliation: `OPERATOR_MANUAL_TP_OPT_OUT_BYPASS=FIXED_NEEDS_CLEAN_WINDOW`, still blocks live-ready `True`.

## Regression Coverage

- `tests.test_tp_rebalancer.TpRebalancerTest.test_operator_manual_auto_tp_modify_false_blocks_stale_harvest_reanchor`
- `tests.test_cli.CliTest.test_snapshot_json_lifts_operator_manual_position_packet`
- `tests.test_position_manager.PositionManagerTest.test_operator_confirmed_eurusd_manual_loss_is_kept_without_sl_or_close`
- `tests.test_position_manager.PositionManagerTest.test_operator_manual_auto_tp_modify_false_blocks_missing_tp_repair`
- `tests.test_position_manager.PositionManagerTest.test_operator_manual_auto_tp_modify_false_keeps_existing_tp_unchanged`
- `tests.test_position_protection_gateway.PositionProtectionGatewayTest.test_operator_manual_auto_tp_modify_false_blocks_take_profit_replace`
- `tests.test_position_protection_gateway.PositionProtectionGatewayTest.test_manual_position_blocks_stop_loss_write`
- `tests.test_position_protection_gateway.PositionProtectionGatewayTest.test_manual_position_blocks_market_close`
- `tests.test_risk_engine.RiskEngineTest.test_operator_manual_eurusd_same_theme_system_add_is_blocked`
- `tests.test_guardian_action_cycle.GuardianActionCycleTest.test_manual_exposure_cannot_be_loss_closed_by_guardian_action`

## Current Blockers

- `PROFITABILITY_ACCEPTANCE_BLOCKED`
- `SELF_IMPROVEMENT_P0_PRESENT`
- `NEGATIVE_EXPECTANCY_ACTIVE`
- `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- `NO_LIVE_READY_LANES`

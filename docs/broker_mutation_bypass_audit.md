# Broker Mutation Bypass Audit

- Generated: `2026-07-06T08:54:19Z`
- Mode: `read_only_broker_mutation_bypass_audit`
- Permission boundary: this audit creates no live permission.
- Broker truth: `last_transaction_id=472996`; open trade `472987 EUR_USD SHORT 30000 owner=operator_manual TP=1.13968 SL=None`; only pending broker order is `472996 TAKE_PROFIT tradeID=472987 price=1.13968`.
- Final read-only dev/live snapshots: dev fetched `2026-07-06T08:53:59.140771+00:00`, live fetched `2026-07-06T08:53:59.558238+00:00`; both stayed at `last_transaction_id=472996`.
- Protected IDs: trade `472987`; TP order lineage `472988/472994/472996`.

## Conclusion

The original TP rebalance leak is contained, and the newly found adjacent bypass is fixed:

- `tp_rebalancer` already skips operator-manual packets with `auto_tp_modify_allowed=false`.
- `PositionManager` now refuses to propose missing/existing TP edits for operator-manual packets with `auto_tp_modify_allowed=false`.
- `PositionProtectionGateway` now blocks any take-profit replacement for those packets even if a bad upstream decision supplies one.
- `472987` remains untouched in current broker truth; no broker transaction advanced past `472996` during this audit.

Normal routing remains blocked. `AUD_JPY SHORT BREAKOUT_FAILURE LIMIT` stays only a proof candidate and is still `DRY_RUN_BLOCKED`, not `LIVE_READY`.

## Mutation Paths

| path | broker mutation | status | guard summary |
|---|---|---|---|
| `LiveOrderGateway.run/run_batch` | `post_order_json` entry order | `SAFE_BLOCKED_BY_CURRENT_ARTIFACTS` | `send`, live confirmation, RiskEngine, StrategyProfile, intent freshness, self-improvement, GPT verifier, target-path, guardian, receipt-consumption, and SL-lint gates must all pass. |
| `automation` pending cancel paths | `cancel_order` | `SAFE_FOR_TRADER_PENDING_ONLY` | Requires `send && live_enabled` plus verified trader-owned pending IDs or forced self-improvement review IDs; current-thesis pending preservation can prevent cancel. |
| `PositionProtectionGateway` dependent order replace | `replace_trade_dependent_orders` | `FIXED` | Manual/tagless SL writes blocked; TP replacement now also blocks `auto_tp_modify_allowed=false`. |
| `PositionManager` manual TP repair proposal | upstream proposal only | `FIXED` | `auto_tp_modify_allowed=false` returns `ACTION_HOLD_SL_FREE` with no TP/SL recommendation. |
| `tp_rebalancer.apply_tp_adjustments` | `replace_trade_dependent_orders` TP | `SAFE_FIXED_PREEXISTING` | Dry-run unless explicitly sent; compute phase skips operator-manual TP opt-out packets. |
| `trailing_sl.apply_trailing_sls` | `replace_trade_dependent_orders` SL | `SAFE` | Trader-owned positions only, existing broker SL required, dry-run can block mutation. |
| `PositionProtectionGateway` market close | `close_trade_with_provenance` | `SAFE` | Raw close is blocked; manual/tagless close forbidden; loss-side close needs GPT Gate A/B or structural explicit opt-in and spread gate. |
| `adverse_partial_close` | `close_trade_with_provenance` partial | `SAFE` | Trader-owned only; manual/operator positions skipped; raw close fallback blocked. |
| `profit_partial_close` | `close_trade_with_provenance` partial | `SAFE_BY_CONTRACT_PROFIT_ONLY` | Profit-only ATR milestone close; requires `send`, live enabled, and `--confirm-live`; raw close fallback blocked. |
| `guardian_action_cycle` / dispatcher | gateway-mediated only | `SAFE_DEFAULT_OFF` | No direct OANDA writes; handoff requires explicit guardian env flags and LiveOrderGateway revalidation. |

## Regression Coverage

- `tests.test_tp_rebalancer.TpRebalancerTest.test_operator_manual_auto_tp_modify_false_blocks_stale_harvest_reanchor`
- `tests.test_position_manager.PositionManagerTest.test_operator_manual_auto_tp_modify_false_blocks_missing_tp_repair`
- `tests.test_position_manager.PositionManagerTest.test_operator_manual_auto_tp_modify_false_keeps_existing_tp_unchanged`
- `tests.test_position_protection_gateway.PositionProtectionGatewayTest.test_operator_manual_auto_tp_modify_false_blocks_take_profit_replace`
- Existing RiskEngine tests block same-theme adds against operator-manual `472987` unless `operator_authorized_manual_overlap=true`.
- Existing guardian tests block manual loss-close by guardian action.

## Current Blockers

- `PROFITABILITY_ACCEPTANCE_BLOCKED`
- `SELF_IMPROVEMENT_P0_PRESENT`
- `NEGATIVE_EXPECTANCY_ACTIVE`
- `MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`
- `MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE`
- `GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED`
- `NO_LIVE_READY_LANES`

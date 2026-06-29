# Strategy Memory

## 2026-06-28 UTC Daily Target Engine

- Base operating target is +5% from UTC day-start NAV.
- +10% is extension-only after an explicit favorable-market gate; default mode after +5% is PROTECT.
- Day-start NAV is persisted under `logs/day_start_nav/YYYY-MM-DD.json`.
- Intraday P/L selection uses UTC 00:00 boundaries. Slack/operator displays may show JST, but the trading-day math stays UTC.
- This change is read-only for live ordering: no gateway, send, cancel, or close behavior was changed.

## 2026-06-28 FULL_TRADER 5% Path Discipline

- Every FULL_TRADER session must fill the `5% PATH BOARD` and map it to `ATTACK STACK` before reporting a decision.
- Under +5%, Path A / HERO must be a concrete route to the +5% floor or an exact blocker with next trigger and shelf-life.
- B/C second-shot or no-honest-path churn cannot be presented as the +5% target path, and one distant pending order is not enough.
- This change is read-only for live ordering: it changes session reporting and validation only.

## 2026-06-28 Target-Aware Dry-Run Sizing

- Added `tools/position_sizing.py` and dry-run-only `tools/place_trader_order.py` so fresh target-path orders can be sized against target contribution, explicit risk capacity, optional margin capacity, and conviction grade before any gateway path.
- S/A can be the main +5% path. B+ is scout/reload support only. B0/B-/C are not valid +5% path trades.
- The 10% Extension Gate defaults to NO; YES requires strong progress or protected S/A carry, paying hero thesis, theme/trend confirmation, stable spread, no near whipsaw event, healthy last A/S trade, and a real reload/second-shot level.
- RiskEngine now blocks target-path metadata that violates the grade, extension, recent same-thesis loss, or 5% PATH / ATTACK STACK mapping contract. This is still dry-run/gateway-safe: no live OANDA order helper was added.

## 2026-06-29 Controlled Target-Path Live Learning

- Target-path live sends are still off by default. They require the normal `QR_LIVE_ENABLED=1 --send --confirm-live` path plus `QR_TARGET_PATH_LIVE_ENABLED=1`.
- `tools/place_trader_order.py` remains dry-run only. It can emit a LiveOrderGateway intent with `--gateway-intent-output`, but all OANDA entry posts still happen only inside `LiveOrderGateway`.
- Live target-path sends must carry `LIVE_LEARNING` receipts with daily target mode, remaining-to-5%, path role, attack-stack slot, grade, suggested/final units, risk, target contribution, and gateway receipt id.
- `daily-review` now classifies sent target-path trades as `discovery failure`, `deployment failure`, `sizing failure`, `vehicle failure`, `management failure`, or `good execution` so the next cycle can learn from the exact failure mode instead of treating all losses as generic lane bias.

## 2026-06-29 USER_ALPHA Continuation

- A profitable manual/operator-discovered winner is `USER_ALPHA` / `OPERATOR_ALPHA`, not proof that the system discovered the edge.
- The EUR_USD LONG operator-discovered winner exposed the gap: the broker/system could manage TP, but the trader did not convert the green outcome into RELOAD / SECOND_SHOT / +5% continuation.
- `daily-review` now publishes user-led winners separately under `user_alpha_trades` and `user_alpha_continuation`; GPT trader receipts must cite `user_alpha:continuation` and either continue the same pair/side or name an exact blocker and next trigger.
- Stale pending replacements must carry ignored pending ids through preflight and final send so `BASKET_DUPLICATE_PARENT_LANE` cannot block cancel/replace of the pending order being replaced.

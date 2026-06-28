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

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

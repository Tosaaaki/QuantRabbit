# Strategy Memory

## 2026-06-28 UTC Daily Target Engine

- Base operating target is +5% from UTC day-start NAV.
- +10% is extension-only after an explicit favorable-market gate; default mode after +5% is PROTECT.
- Day-start NAV is persisted under `logs/day_start_nav/YYYY-MM-DD.json`.
- Intraday P/L selection uses UTC 00:00 boundaries. Slack/operator displays may show JST, but the trading-day math stays UTC.
- This change is read-only for live ordering: no gateway, send, cancel, or close behavior was changed.

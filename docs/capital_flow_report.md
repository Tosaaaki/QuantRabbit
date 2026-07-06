# Capital Flow Report

- Generated at UTC: `2026-07-06T01:46:24.561175Z`
- Scope: accounting/reporting only; no orders, cancels, closes, execution flags, or broker-state writes.
- Source basis: operator statement plus local target state when available; no broker transaction fetch was performed for this record.

## Recorded Flows

| timestamp_utc | amount_jpy | type | source | note | included_in_raw_equity | excluded_from_funding_adjusted_return |
| --- | ---: | --- | --- | --- | --- | --- |
| `2026-07-02T08:33:11Z` | `100000` | `DEPOSIT` | `operator` | `100,000 JPY operator capital injection` | `true` | `true` |

## 30D Target Accounting

| field | value |
| --- | --- |
| rolling_30d_start_equity | `171435.0552` |
| current_equity_raw | `273282.7334` |
| capital_flows_30d | `100000.0` |
| funding_adjusted_equity | `173282.7334` |
| rolling_30d_multiplier_raw | `1.594089` |
| rolling_30d_multiplier_funding_adjusted | `1.010778` |
| remaining_to_4x_raw | `412457.4874` |
| remaining_to_4x_funding_adjusted | `512457.4874` |
| required_calendar_daily_return_funding_adjusted | `5.413648` |
| required_active_day_return_funding_adjusted | `7.454094` |
| performance_basis | `funding_adjusted` |
| sizing_basis | `raw_nav` |

## Policy

- Raw NAV includes deposits and withdrawals because broker equity includes funding flows.
- Funding-adjusted equity excludes deposits and withdrawals from trading performance.
- Rolling 30d 4x performance uses `funding_adjusted_equity`, `rolling_30d_multiplier_funding_adjusted`, and `remaining_to_4x_funding_adjusted`.
- Risk, margin, and sizing use raw broker NAV / `current_equity_raw`.
- A raw NAV increase caused by a deposit must not be described as trading P/L or as the authoritative 30d 4x KPI.

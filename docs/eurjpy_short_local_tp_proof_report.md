# EUR_JPY SHORT Local TP Proof Report

- Generated: `2026-07-05T09:14:46Z`
- Classification: `REJECTED_NEGATIVE_EXPECTANCY`
- A/S candidate: `False`
- LIVE_READY allowed: `False`

## Decision

Broad EUR_JPY SHORT S5 and pair-shape evidence is negative; narrow positive confluences are post-hoc, audit-only, and below live-grade sample/Wilson/active-day requirements.

## Candidates

| candidate | bucket | n | days | avg pips | PF | hit | pos-day | adverse | drawdown | class | sufficient |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| S5 forecast-history EUR_JPY SHORT broad DOWN | current forecast_history DOWN; no strategy bucket | 11 | 3 | 0.7272727272727273 | 1.1904761904761905 | 0.5454545454545454 | 0.6667 | {"metric": "avg_mae_pips", "value": 22.29999999999977} | {"metric": "worst_daily_realized_pips", "value": -14.0} | REJECTED_UNDER_SAMPLED | False |
| Packaged/live S5 EUR_JPY DOWN negative rule | all packaged DOWN samples | 1147 | 37 | -2.0 | 0.0 | 0.2947 | 0.0 | {"metric": "avg_mae_pips", "value": 11.8349} | {"metric": "worst_daily_realized_pips", "value": -208.0} | REJECTED_NEGATIVE_EXPECTANCY | False |
| TREND_CONTINUATION SHORT | all pair-shape samples | 273 | 28 | -2.296985 | 0.482251 | 0.249084 | 0.10714285714285714 | {"metric": "not_reported_by_shape_miner", "proxy": {"validation_stop_first_rate": 0.725275}, "value": null} | {"metric": "worst_validation_day_avg_atr", "value": -0.834255} | REJECTED_NEGATIVE_EXPECTANCY | False |
| RANGE_ROTATION SHORT | all pair-shape samples | 363 | 39 | -2.681056 | 0.358404 | 0.242424 | 0.1282051282051282 | {"metric": "not_reported_by_shape_miner", "proxy": {"validation_stop_first_rate": 0.743802}, "value": null} | {"metric": "worst_validation_day_avg_atr", "value": -0.835992} | REJECTED_NEGATIVE_EXPECTANCY | False |
| BREAKOUT_FAILURE SHORT | all pair-shape samples | 41 | 26 | -2.88872 | 0.335963 | 0.219512 | 0.2692307692307692 | {"metric": "not_reported_by_shape_miner", "proxy": {"validation_stop_first_rate": 0.756098}, "value": null} | {"metric": "worst_validation_day_avg_atr", "value": -0.82963} | REJECTED_NEGATIVE_EXPECTANCY | False |
| failed acceptance / major-figure fade proxy | all pair-shape samples | 60 | 27 | -2.888036 | 0.330779 | 0.216667 | 0.18518518518518517 | {"metric": "not_reported_by_shape_miner", "proxy": {"validation_stop_first_rate": 0.766667}, "value": null} | {"metric": "worst_validation_day_avg_atr", "value": -0.842891} | REJECTED_NEGATIVE_EXPECTANCY | False |
| major-figure exhaustion fade proxy | not available in focused miner output | None | None | None | None | None | None | null | null | EVIDENCE_GAP | False |
| session-specific RANGE_ROTATION SHORT | session:london_ny_overlap + spread_regime:mid | 27 | 9 | 2.838228 | 2.03119 | 0.666667 | 0.7777777777777778 | {"metric": "not_reported_by_shape_miner", "proxy": {"validation_stop_first_rate": 0.259259}, "value": null} | {"metric": "worst_validation_day_avg_atr", "value": -1.0} | REJECTED_UNDER_SAMPLED | False |
| session-specific RANGE_ROTATION SHORT | session:london_ny_overlap + spread_regime:mid | 27 | 9 | 1.703208 | 1.507859 | 0.592593 | 0.5555555555555556 | {"metric": "not_reported_by_shape_miner", "proxy": {"validation_stop_first_rate": 0.333333}, "value": null} | {"metric": "worst_validation_day_avg_atr", "value": -1.0} | REJECTED_UNDER_SAMPLED | False |
| live-grade evidence queue best EUR_JPY SHORT | bar_range:wide + session:ny | 10 | 7 | 2.904911 | 2.776219 | 0.7 | 0.7142857142857143 | {"metric": "not_reported_by_shape_miner", "proxy": {"validation_stop_first_rate": null}, "value": null} | {"metric": "worst_validation_day_avg_atr", "value": null} | REJECTED_UNDER_SAMPLED | False |

## Missing Evidence

- non-negative spread-included S5 bid/ask replay for the exact EUR_JPY SHORT lane shape
- sufficient validation samples and active days for the selected session/location bucket
- exact local TP scope for EUR_JPY|SHORT|strategy|TAKE_PROFIT_ORDER
- fresh order_intent emitted with units > 0 and all RiskEngine/LiveOrderGateway blockers cleared
- fresh GPT-5.5 TRADE/ADD receipt after proof and risk gates pass

## Safety

- `broker_state_modified`: `False`
- `orders_placed`: `False`
- `orders_cancelled`: `False`
- `positions_closed`: `False`
- `sl_tp_modified`: `False`
- `execution_flags_enabled`: `False`
- `normal_routing_remains_blocked`: `True`
- `raw_nav_usage`: `sizing_and_risk_only`
- `performance_kpi`: `funding_adjusted_equity`
- `deposit_100000_jpy_is_capital_flow_not_pl`: `True`

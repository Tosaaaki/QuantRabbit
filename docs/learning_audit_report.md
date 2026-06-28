# Learning Audit Report

- Generated at UTC: `2026-06-26T06:40:12.684027+00:00`
- Status: `LEARNING_AUDIT_WARN`
- Checks: `14`
- Blockers: `0`
- Warnings: `3`
- Influenced recommended lanes: `0`
- Total learning score delta: `0.0`

## Effect Window

- Window hours: `168.0`
- Closed trades: `13`
- Net JPY: `276.5`
- Profit factor: `1.059`
- Expectancy JPY: `21.272`

## Exit Reasons

- `STOP_LOSS_ORDER` closed=`4` net=`-3297.7` pf=`0.000` expectancy=`-824.421`
- `MARKET_ORDER_TRADE_CLOSE` closed=`5` net=`-65.8` pf=`0.952` expectancy=`-13.155`
- `TAKE_PROFIT_ORDER` closed=`4` net=`3640.0` pf=`n/a` expectancy=`910.000`

## Blockers

- none

## Warnings

- AI backtest is profitable research only; reduced weighting required
- effect sample is below stability floor
- market-order trade closes are negative in the recent effect window; prefer TP/TP-rebalance/profit-side exits unless CLOSE Gate A/B is hard

## Learning Influence

- none

## Contract

- Learning may rank already-live-ready lanes only.
- Learning cannot override RiskEngine, gateway, entry-thesis, or TP blockers.
- Research-stage positive edges use reduced weight and remain WARN-level audit evidence.
- Every recommended lane influenced by learning must expose `learning_influence_details`.

# Operator Precedent Audit

- Generated at UTC: `2026-06-12T04:31:24.454509+00:00`
- Status: `OPERATOR_PRECEDENT_WARN`
- Best funding-adjusted 30d return: `319.72`% (`457471.1871` JPY)
- Peak funding-adjusted return: `200.28`%
- Winning shape: `USD_JPY LONG`; primary sessions `LONDON_AM, NY_OVERLAP`; median hold `0.48`h
- Failure shape: margin closeout `24` exits, net `-217327.8` JPY, median hold `12.38`h
- Current LIVE_READY lanes: `1`; precedent-aligned: `0`

## Checks

| check | status | message |
|---|---|---|
| `manual_history_readable` | `PASS` | manual history artifact readable: /Users/tossaki/App/QuantRabbit/data/manual_history_2025_mining.json |
| `order_intents_readable` | `PASS` | order intents readable: /Users/tossaki/App/QuantRabbit/data/order_intents.json |
| `target_state_readable` | `PASS` | daily target state readable: /Users/tossaki/App/QuantRabbit/data/daily_target_state.json |
| `funding_adjusted_30d_claim` | `PASS` | best funding-adjusted 30d return 319.72% verifies the operator 200%+ claim |
| `raw_balance_not_used_as_strategy_pnl` | `PASS` | manual history separates account funding from strategy P/L |
| `winning_shape_extracted` | `PASS` | manual history exposes a primary pair/direction/session shape |
| `failure_shape_extracted` | `PASS` | manual history exposes the margin-closeout failure mode |
| `current_live_ready_alignment` | `WARN` | LIVE_READY lanes exist, but none match the manual precedent's pair/direction/session shape |

## Contract

- Advisory only: this audit may rank or explain already-current LIVE_READY lanes.
- It cannot override RiskEngine, LiveOrderGateway, forecast, spread, event, broker-truth, or close Gate A/B checks.

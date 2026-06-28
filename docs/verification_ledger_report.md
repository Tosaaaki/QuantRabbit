# Verification Ledger Report

- Generated at UTC: `2026-06-26T06:41:21.571649+00:00`
- Status: `BLOCKED`
- DB: `/Users/tossaki/App/QuantRabbit/data/execution_ledger.db`
- Observations inserted: `119`
- Measurements inserted: `7`
- Blocking observations: `82`
- Missing observations: `0`

## Effect Window

- Window hours: `168.0`
- Closed trades: `13`
- Net JPY: `276.5`
- Profit factor: `1.059`
- Win rate: `0.615`
- Expectancy JPY: `21.272`
- Sample warning: `INSUFFICIENT_SAMPLE_LT_30`

## Blocking Evidence

- `order_intents` `lane_blockers` lane=range_trader:AUD_CHF:SHORT:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:AUD_CHF:SHORT:RANGE_ROTATION:MARKET: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:AUD_JPY:SHORT:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:AUD_NZD:SHORT:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:AUD_NZD:SHORT:RANGE_ROTATION:MARKET: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:AUD_USD:SHORT:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:AUD_USD:LONG:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:CAD_CHF:LONG:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:CAD_CHF:LONG:RANGE_ROTATION:MARKET: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:CAD_JPY:SHORT:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:CAD_JPY:SHORT:RANGE_ROTATION:MARKET: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:EUR_AUD:LONG:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:EUR_AUD:LONG:RANGE_ROTATION:MARKET: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:EUR_CAD:SHORT:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:EUR_CAD:SHORT:RANGE_ROTATION:MARKET: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:EUR_CHF:LONG:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:EUR_CHF:LONG:RANGE_ROTATION:MARKET: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:EUR_JPY:LONG:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:EUR_NZD:LONG:RANGE_ROTATION: `BLOCK`
- `order_intents` `lane_blockers` lane=range_trader:EUR_NZD:LONG:RANGE_ROTATION:MARKET: `BLOCK`

## Missing Artifacts

- none

## Learning Evidence

- `ai_backtest` `walk_forward_certification` ai_backtest=RESEARCH_PROFITABLE_NOT_CERTIFIED: `WARN` value=`40294.9646`JPY
- `ai_backtest` `read_only_learning` ai_backtest=live_permission: `PASS`
- `ai_backtest` `sample_size` ai_backtest=selected_trades: `PASS` value=`211.0`count
- `outcome_mart` `read_only_learning` outcome_mart=read_only: `PASS`
- `outcome_mart` `condition_walk_forward` outcome_mart=CONDITION_WALK_FORWARD_READY: `PASS` value=`7966.0`count
- `outcome_mart` `outcome_source_coverage` outcome_mart=source_coverage: `PASS` value=`8077.0`count
- `post_trade_learning` `learning_review_status` post_trade_learning=READY_FOR_REVIEW: `PASS` value=`1.0`count
- `post_trade_learning` `profile_update_requires_review` post_trade_learning=profile_update_candidates: `PASS` value=`0.0`count
- `ai_attack_advice` `read_only_learning` ai_attack_advice=read_only: `PASS`
- `ai_attack_advice` `recommended_learning_influence` ai_attack_advice=NO_ATTACK_ADVICE: `PASS` value=`0.0`score_delta
- `learning_audit` `learning_audit_status` learning_audit=LEARNING_AUDIT_WARN: `WARN` value=`0.0`blockers
- `learning_audit` `learning_influence_score_delta` learning_audit=learning_influence: `PASS` value=`0.0`score_delta
- `learning_audit` `learning_effect_window` learning_audit=effect_window: `WARN` value=`13.0`closed_trades

## Measurement Contract

- Verification observations are append-only DB rows; do not overwrite a prior cycle's evidence.
- Effect metrics are computed from `execution_events` broker/gateway truth for the declared window.
- A metric with fewer than 30 closed trades is tracked but not treated as statistically stable.
- Learning artifacts may influence live-ready lane ranking, but every influenced recommended lane is recorded here and cannot override hard risk/gateway gates.

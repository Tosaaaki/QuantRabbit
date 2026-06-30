# Decision Receipt Schema

Write `data/codex_trader_decision_response.json`. In scheduled automation,
`trader-draft-decision` is the default read-only composer; the final authority
is still `gpt-trader-decision` plus `LiveOrderGateway`.

```json
{
  "generated_at_utc": "2026-06-08T12:32:26Z",
  "market_read_first": {
    "naked_read": {
      "currency_bought": "EUR",
      "currency_sold": "USD",
      "cleanest_pair_expression": "EUR_USD",
      "is_cleanest_currency_theme": "YES - EUR_USD is the cleanest current EUR strength / USD weakness expression.",
      "location_24h": "LOWER",
      "h1_h4_alignment": "H1=WITH_H1_TREND; H4=WITH_H4_TREND",
      "tape_state": "TREND",
      "known_winning_trade_shape_match": "MATCH - generalized 2025 operator trade shape.",
      "proposed_building_style_allowed": "YES - SINGLE",
      "thesis_state": "ALIVE",
      "what_price_is_trying_to_do_now": "EUR_USD is trying to lift from the lower 24h shelf before execution filters."
    },
    "next_30m_prediction": {
      "pair": "EUR_USD",
      "direction": "LONG",
      "expected_path": "Hold the shelf and press toward the nearest liquidity pocket.",
      "target_zone": "current target zone from the packet",
      "invalidation": "current invalidation from the packet"
    },
    "next_2h_prediction": {
      "pair": "EUR_USD",
      "direction": "LONG",
      "expected_path": "Extend only if the H1/H4 thesis remains alive.",
      "target_zone": "current 2h target zone from the packet",
      "invalidation": "current 2h invalidation from the packet"
    },
    "best_trade_if_forced": {
      "pair": "EUR_USD",
      "direction": "LONG",
      "vehicle": "STOP",
      "entry": "current broker-refreshed entry only",
      "tp": "current TP from packet",
      "sl": "current invalidation/emergency stop from packet",
      "why_this_pays": "It pays only if the naked 30m/2h read reaches target before invalidation."
    }
  },
  "action": "TRADE",
  "selected_lane_id": "desk:PAIR:SIDE:METHOD",
  "selected_lane_ids": ["desk:PAIR:SIDE:METHOD", "desk:PAIR:SIDE:METHOD:MARKET"],
  "cancel_order_ids": [],
  "confidence": "HIGH",
  "thesis": "...",
  "method": "BREAKOUT_FAILURE",
  "narrative": "...",
  "chart_story": "Cite pair-chart, market-context-matrix, cross-asset, flow, levels, calendar, news, COT, option-skew state with numbers.",
  "invalidation": "...",
  "rejected_alternatives": ["..."],
  "risk_notes": ["bounded by per_trade_risk_budget_jpy from data/daily_target_state.json"],
  "twenty_minute_plan": {
    "horizon_minutes": 20,
    "primary_path": "What should happen before the next trader cycle if this decision is right.",
    "failure_path": "What would make the decision wrong before the next trader cycle.",
    "entry_or_hold_trigger": "Concrete trigger for entry, basket keep, or WAIT continuation.",
    "invalidation_or_cancel_trigger": "Concrete structure/level/packet gate that cancels the idea.",
    "counterargument": "Strongest current evidence against this action and why it does not win.",
    "next_cycle_check": "First check the next trader must perform if no fill or blocker occurs.",
    "evidence_refs": ["intent:<lane_id>", "chart:<pair>:M5", "chart:<pair>:M15"]
  },
  "evidence_refs": [
    "broker:snapshot",
    "target:daily",
    "intent:<lane_id>",
    "campaign:<lane_id>",
    "strategy:<pair>:<side>",
    "story:<pair>",
    "chart:<pair>:M1",
    "chart:<pair>:M5",
    "chart:<pair>:M15",
    "chart:<pair>:M30",
    "chart:<pair>:H1",
    "chart:<pair>:H4",
    "chart:<pair>:D",
    "chart:<pair>:structure",
    "cross:dxy",
    "cross:USB10Y_USD",
    "cross:correlations:<pair>",
    "strength:<pair>",
    "flow:<pair>",
    "levels:<pair>",
    "calendar:<pair>",
    "news:<pair>",
    "news:<currency>",
    "news:items",
    "news:health",
    "cot:<currency>",
    "option:skew:<pair>",
    "timing:audit",
    "timing:canceled_order:<order_id>",
    "timing:loss_close:<trade_id>",
    "timing:market_close:<trade_id>",
    "attack:advice",
    "attack:lane:<lane_id>"
  ],
  "evidence_ref_note": "Use option:skew refs only when data/option_skew_snapshot.json has enabled=true; omit them for disabled optional skew artifacts. Use timing:* refs only when data/execution_timing_audit.json shaped cancel, HOLD, CLOSE, TP-rebalance, or profit-take reasoning. TRADE must cite news:health and news:items or news:current.",
  "strategy_reviews": [
    {
      "lane_id": "desk:PAIR:SIDE:METHOD",
      "method": "TREND_CONTINUATION",
      "verdict": "SUPPORTS",
      "summary": "Review is advisory and method-specific."
    }
  ],
  "specialist_reviews": [
    {
      "role": "indicator",
      "lane_id": "desk:PAIR:SIDE:METHOD",
      "method": "TREND_CONTINUATION",
      "verdict": "SUPPORTS",
      "summary": "Observation only.",
      "cited_evidence_refs": ["chart:<pair>:M5"],
      "hard_gate_codes": [],
      "read_only": true,
      "live_permission": false
    }
  ],
  "operator_summary": "..."
}
```

## Allowed Values

- `action`: `TRADE`, `WAIT`, `REQUEST_EVIDENCE`, `PROTECT`, `TIGHTEN_SL`, `CLOSE`, `CANCEL_PENDING`
- `confidence`: `LOW`, `MEDIUM`, `HIGH`
- `method`: `TREND_CONTINUATION`, `RANGE_ROTATION`, `BREAKOUT_FAILURE`, `EVENT_RISK`, `POSITION_MANAGEMENT`
- Specialist `role`: `macro_news`, `indicator`, `flow_levels`, `risk_audit`, `strategy`, `portfolio_context`
- `strategy_reviews[].verdict` and `specialist_reviews[].verdict`: `SUPPORTS`, `REJECTS`, `BLOCKED`, `WATCH`. Do not encode gate reasons in the verdict string; put gate details such as `CLOSE_OPERATOR_AUTH_REQUIRED` or `NO_LIVE_READY_LANES` in `hard_gate_codes`.
- `market_read_first.naked_read.thesis_state`: `ALIVE`, `WOUNDED`, `INVALIDATED`, `EMERGENCY`, or `UNKNOWN`.
- `twenty_minute_plan.horizon_minutes`: `20` for `TRADE`, `WAIT`, and `REQUEST_EVIDENCE`; this is the scheduled decision cadence, not a market-derived holding target.
- `cancel_order_ids` on `TRADE` or `CANCEL_PENDING` is allowed only for current trader-owned pending entry ids from the broker snapshot. For `TRADE`, the gateway cancels those ids before validating the selected basket, so use it only when replacing stale or lower-priority pending exposure with a current tradeable basket. For `CANCEL_PENDING`, the gateway cycle cancels the verified ids and sends no fresh entry in that same cycle.
- `close_trade_ids` is allowed only with `action=CLOSE`. `TRADE` with `close_trade_ids` is rejected; loss-cut and re-entry must be separated by a fresh broker snapshot / intent cycle.

## Verifier Rejection Triggers

- Unknown evidence refs.
- Missing or unparseable `generated_at_utc`.
- Missing, incomplete, or blocker-first `market_read_first`.
- `market_read_first.naked_read` omits cleanest theme expression, 24h location,
  H1/H4 alignment, known winning trade-shape match, proposed building-style
  allowance, thesis state, or tape state.
- `TRADE` without current `LIVE_READY` selected lane(s).
- `TRADE` with `close_trade_ids`.
- `TRADE` without `news:health` and `news:items` or `news:current`.
- `TRADE` while `data/news_health.json` is missing, ERROR/BLOCK, or carries BLOCK issues (`NEWS_HEALTH_BLOCKS_TRADE`).
- `TRADE` or `CANCEL_PENDING` names unknown, manual, or non-pending order ids in `cancel_order_ids`.
- Selected lane not cited as `intent:<lane_id>`.
- Method mismatch with selected primary lane.
- WAIT / REQUEST_EVIDENCE while rolling pace is open and clean A/S or attack-recommended tradeable lanes exist, unless a named gate blocks them. B/C lanes are not forced solely to hit today's +5% pace marker.
- Ignoring current tradeable `ai_attack_advice` priority.
- Selecting a learning-influenced lane without non-blocked `learning_audit`
  coverage and `learning:audit` / `learning:lane:<lane_id>` evidence refs.
- Specialist review grants live permission or carries execution fields.
- `TRADE`, `WAIT`, or `REQUEST_EVIDENCE` without a complete `twenty_minute_plan`.
- `twenty_minute_plan` cites unknown refs, omits chart refs while current tradeable lanes exist, or omits the selected `intent:<lane_id>` for `TRADE`.

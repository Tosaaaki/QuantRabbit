# Decision Receipt Schema

Write `data/codex_trader_decision_response.json`.

```json
{
  "action": "TRADE",
  "selected_lane_id": "desk:PAIR:SIDE:METHOD",
  "selected_lane_ids": ["desk:PAIR:SIDE:METHOD", "desk:PAIR:SIDE:METHOD:MARKET"],
  "cancel_order_ids": [],
  "confidence": "HIGH",
  "thesis": "...",
  "method": "BREAKOUT_FAILURE",
  "narrative": "...",
  "chart_story": "Cite pair-chart, cross-asset, flow, levels, calendar, COT, option-skew state with numbers.",
  "invalidation": "...",
  "rejected_alternatives": ["..."],
  "risk_notes": ["bounded by per_trade_risk_budget_jpy from data/daily_target_state.json"],
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
    "cot:<currency>",
    "option:skew:<pair>",
    "attack:advice",
    "attack:lane:<lane_id>"
  ],
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
- `cancel_order_ids` on `TRADE` or `CANCEL_PENDING` is allowed only for current trader-owned pending entry ids from the broker snapshot. For `TRADE`, the gateway cancels those ids before validating the selected basket, so use it only when replacing stale or lower-priority pending exposure with a current tradeable basket. For `CANCEL_PENDING`, the gateway cycle cancels the verified ids and sends no fresh entry in that same cycle.

## Verifier Rejection Triggers

- Unknown evidence refs.
- `TRADE` without current `LIVE_READY` selected lane(s).
- `TRADE` or `CANCEL_PENDING` names unknown, manual, or non-pending order ids in `cancel_order_ids`.
- Selected lane not cited as `intent:<lane_id>`.
- Method mismatch with selected primary lane.
- WAIT / REQUEST_EVIDENCE while target is open and clean tradeable lanes exist, unless a named gate blocks them.
- Ignoring current tradeable `ai_attack_advice` priority.
- Specialist review grants live permission or carries execution fields.

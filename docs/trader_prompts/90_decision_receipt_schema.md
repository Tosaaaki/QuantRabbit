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
- `strategy_reviews[].verdict` and `specialist_reviews[].verdict`: `SUPPORTS`, `REJECTS`, `BLOCKED`, `WATCH`. Do not encode gate reasons in the verdict string; put gate details such as `CLOSE_OPERATOR_AUTH_REQUIRED` or `NO_LIVE_READY_LANES` in `hard_gate_codes`.
- `twenty_minute_plan.horizon_minutes`: `20` for `TRADE`, `WAIT`, and `REQUEST_EVIDENCE`; this is the scheduled decision cadence, not a market-derived holding target.
- `cancel_order_ids` on `TRADE` or `CANCEL_PENDING` is allowed only for current trader-owned pending entry ids from the broker snapshot. For `TRADE`, the gateway cancels those ids before validating the selected basket, so use it only when replacing stale or lower-priority pending exposure with a current tradeable basket. For `CANCEL_PENDING`, the gateway cycle cancels the verified ids and sends no fresh entry in that same cycle.
- `close_trade_ids` is allowed only with `action=CLOSE`. `TRADE` with `close_trade_ids` is rejected; loss-cut and re-entry must be separated by a fresh broker snapshot / intent cycle.

## Verifier Rejection Triggers

- Unknown evidence refs.
- `TRADE` without current `LIVE_READY` selected lane(s).
- `TRADE` with `close_trade_ids`.
- `TRADE` or `CANCEL_PENDING` names unknown, manual, or non-pending order ids in `cancel_order_ids`.
- Selected lane not cited as `intent:<lane_id>`.
- Method mismatch with selected primary lane.
- WAIT / REQUEST_EVIDENCE while target is open and clean tradeable lanes exist, unless a named gate blocks them.
- Ignoring current tradeable `ai_attack_advice` priority.
- Selecting a learning-influenced lane without non-blocked `learning_audit`
  coverage and `learning:audit` / `learning:lane:<lane_id>` evidence refs.
- Specialist review grants live permission or carries execution fields.
- `TRADE`, `WAIT`, or `REQUEST_EVIDENCE` without a complete `twenty_minute_plan`.
- `twenty_minute_plan` cites unknown refs, omits chart refs while current tradeable lanes exist, or omits the selected `intent:<lane_id>` for `TRADE`.

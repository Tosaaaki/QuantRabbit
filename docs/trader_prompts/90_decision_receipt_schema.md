# Decision Receipt Schema

Do not hand-write `data/codex_trader_decision_response.json`. In scheduled
automation, `trader-draft-decision` writes `data/trader_decision_baseline.json`
and `data/market_read_evidence_packet.json`; Codex writes only
`data/codex_market_read_overlay.json`; and `trader-apply-market-read` publishes
the merged response. Final authority remains `gpt-trader-decision` plus
`LiveOrderGateway`.

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
      "target_zone": "1.1740 to 1.1750",
      "invalidation": "1.1700"
    },
    "next_2h_prediction": {
      "pair": "EUR_USD",
      "direction": "LONG",
      "expected_path": "Extend only if the H1/H4 thesis remains alive.",
      "target_zone": "1.1760 to 1.1780",
      "invalidation": "1.1685"
    },
    "best_trade_if_forced": {
      "pair": "EUR_USD",
      "direction": "LONG",
      "vehicle": "STOP",
      "entry": "1.1730",
      "tp": "1.1760",
      "sl": "1.1700",
      "why_this_pays": "It pays only if the naked 30m/2h read reaches target before invalidation."
    }
  },
  "decision_provenance": {
    "schema_version": 2,
    "author_kind": "CODEX_MARKET_READ",
    "model": "gpt-5.5",
    "reasoning_effort": "high",
    "authored_at_utc": "2026-06-08T12:31:50Z",
    "applied_at_utc": "2026-06-08T12:32:26Z",
    "baseline_sha256": "<64hex>",
    "evidence_packet_sha256": "<64hex>",
    "overlay_sha256": "<64hex>",
    "market_read_sha256": "<64hex>",
    "execution_envelope_sha256": "<64hex>",
    "baseline_execution_envelope_sha256": "<64hex>",
    "final_execution_envelope_sha256": "<64hex>",
    "baseline_action": "TRADE",
    "final_action": "TRADE",
    "baseline_selected_lane_ids": ["desk:PAIR:SIDE:METHOD"],
    "baseline_disposition": "ACCEPT_BASELINE",
    "action_downgrade_only": false,
    "capital_allocation_sha256": "<64hex>",
    "capital_allocation_board_sha256": "<64hex>",
    "authorized_size_multiple": 0.75,
    "authorized_units": 750,
    "execution_fields_preserved": true,
    "risk_envelope_not_expanded": true,
    "live_permission_granted": false
  },
  "market_read_review": {
    "prior_prediction_ids": ["mr2:<sha256>"],
    "what_failed": "The prior 30m direction was correct but invalidation touched first.",
    "adjustment": "Require the current shelf to hold before accepting the baseline trigger.",
    "no_change_reason": ""
  },
  "market_read_counterargument": "The lift can fail back below 1.1700 before reaching target.",
  "market_read_change_summary": "Moved target and invalidation to current quote-relative numeric levels.",
  "market_read_disposition": "ACCEPT_BASELINE",
  "market_read_veto_reason": "",
  "market_read_vetoed_lane_ids": [],
  "capital_allocation": {
    "decision": "ALLOCATE",
    "lane_id": "desk:PAIR:SIDE:METHOD",
    "size_multiple": 0.75,
    "selected_units": 750,
    "allocation_board_sha256": "<64hex>",
    "rationale": "Direction-specific economic hit rate, timeout rate, exact-vehicle expectancy, reward/risk, and maximum loss support 75% of the already risk-capped 1000-unit baseline."
  },
  "action": "TRADE",
  "selected_lane_id": "desk:PAIR:SIDE:METHOD",
  "selected_lane_ids": ["desk:PAIR:SIDE:METHOD"],
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
    "horizon_minutes": 60,
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
- `market_read_disposition`: `ACCEPT_BASELINE`, `VETO_WAIT`, or `VETO_REQUEST_EVIDENCE`. A veto is valid only when the deterministic baseline action was `TRADE`; it clears final selected lanes while preserving their ids in `market_read_vetoed_lane_ids`.
- `capital_allocation.decision`: `ALLOCATE` only for final `TRADE`; `NO_TRADE` for every veto, non-entry action, and `CLOSE`. `ALLOCATE` must bind the one baseline lane, use only an exact advertised `0.5`, `0.75`, or `1.0`, and set `selected_units=floor(base_units × size_multiple)`; choices that floor to zero units are not advertised. Predictive SCOUT and `HEDGE` allow only `1.0`. Broad/global positive expectancy does not prove a lane. Negative or unknown edge is non-trade unless the board proves exact pair/side/method/vehicle all-exit net evidence with at least 20 financing-adjusted outcomes and positive net/expectancy/Wilson-stressed expectancy, validated bounded SCOUT evidence, `HEDGE` risk reduction, or the narrower attached technical-TP repair exception with exact source scope, at least five trades, zero losses, positive expectancy, and positive average win. `NO_TRADE` requires `lane_id=null`, `size_multiple=0`, and `selected_units=0`.
- For an ordinary schema-v2 `ALLOCATE`, the receipt's numeric forecast fields are live hypotheses only. The gateway accepts MARKET, then recomputes the economic Wilson lower bound, strict EV, quarter-Kelly unit ceiling, fresh NAV/conversions/RiskEngine geometry, and the OANDA S5 no-touch path. It fixes the maximum adverse fill as `priceBound` and repeats the exact proof after reservation; any consumed barrier, adverse drift, incomplete/malformed path, worse NAV/margin, changed transaction id, off-grid broker price, or looser/tampered bound blocks without POST and retains the reservation. The receipt must not pre-author a bound or assert that these final broker checks passed.
- Ordinary `ALLOCATE` provenance must also carry `execution_cost_floor_sha256` from allocation-surface v2. The proof has no static or zero-cost fallback: MARKET entry p95 is joined from gateway RiskMetrics to official OANDA fill truth; TP/SL exit p95 is joined by the close's exact protection order id and trade id to the authoritative OANDA transaction; financing is adverse-only, unit-scaled, and the maximum global/exact Wilson-95 upper stress, with no credit offset. Latest-anchored 90-day cohorts, a latest-age limit of 90 days, applicable 20-sample floors, exact scope, and all digests must pass. Fresh numeric EV uses entry + exit + financing cost; worst-fill EV embeds the adverse entry move once and retains exit + financing cost once, with at least entry-p95 bound allowance and cost-adjusted Kelly/NAV/portfolio/basket/attached-stop caps. The allocation board, merged receipt, early pre-POST proof, reserved `priceBound`, and final pre-POST proof must retain the same cost SHA and exact pair/side/authoritative `market_context.method`/MARKET scope; optional `metadata.method` must agree. A globally calibrated entry/exit/financing row can invalidate this receipt even when it belongs to another exact edge lane. V1 packets/receipts and missing, thin, stale, mismatched, malformed, bool/non-finite, or tampered cost evidence are rejection triggers and require a fresh packet/overlay/receipt.
- `decision_provenance.author_kind=CODEX_MARKET_READ` is mandatory for final `TRADE`. Its schema version, model, age, market-read/allocation/final-envelope digests, transition, and no-permission/no-risk-expansion flags must validate. Schema-v2 `TRADE`, `CLOSE`, `WAIT`, and `REQUEST_EVIDENCE` must each exactly rebuild from the current baseline, stored packet/body, overlay, and material evidence; veto receipts receive no weaker provenance treatment.
- A current market-read receipt binds one primary pair, side, and lane. For `TRADE`, `selected_lane_ids` must contain exactly one item equal to `selected_lane_id`. A different or additional lane requires a fresh broker/evidence snapshot, GPT wake, overlay, verification, and receipt; downstream expansion, substitution, or deterministic recovery is forbidden.
- `twenty_minute_plan.horizon_minutes`: `60` for `TRADE`, `WAIT`, and `REQUEST_EVIDENCE`; the field name is retained for compatibility, and this is the scheduled hourly decision cadence, not a market-derived holding target.
- `cancel_order_ids` on `TRADE` or `CANCEL_PENDING` is allowed only for current trader-owned pending entry ids from the broker snapshot. For `TRADE`, the gateway cancels those ids before validating the exact single selected lane, so use it only when replacing stale or lower-priority pending exposure with that current lane. For `CANCEL_PENDING`, the gateway cycle cancels the verified ids and sends no fresh entry in that same cycle.
- `close_trade_ids` is allowed only with `action=CLOSE`. `TRADE` with `close_trade_ids` is rejected; loss-cut and re-entry must be separated by a fresh broker snapshot / intent cycle.

Measurement output keeps two lineages separate. A schema-v2 prediction row's top-level `originating_decision_receipt_id`, `direct_execution_attribution`, and `direct_realized_outcome` describe that same row's originating `gptd:` receipt and exact `(gptd, mr2)` execution/P&L. Its `reaction_chain.first_subsequent_decision`, reaction execution attribution, and reaction realized outcome describe the next recorded decision after the prior prediction. Neither path may join by pair/time proximity, and reaction evidence must never be presented as the originating prediction's own execution or P/L. A pending LIMIT/STOP order may later extend from its exact gateway order id to an execution-ledger trade id only through exact `ORDER_FILLED.order_id` equality.

## Verifier and Gateway Rejection Triggers

- Unknown evidence refs.
- Missing or unparseable `generated_at_utc`.
- Missing, incomplete, or blocker-first `market_read_first`.
- Deterministic `TRADE` without fresh valid `CODEX_MARKET_READ` provenance (`AI_MARKET_READ_REQUIRED`).
- Codex read with missing/non-numeric/wrong-side target, invalidation, forced entry/TP/SL, or invalid `RANGE` rails.
- Missing latest truly resolved score-eligible prior prediction review, missing counterargument/change summary, stale baseline/evidence SHA, changed execution envelope, or invalid accept/veto transition.
- Missing/stale or body-tampered packet/allocation board, missing/invalid allocation rationale, alternate lane, non-allowlisted/approximate multiplier, non-finite or bool numeric fields, wrong integer-floor units, negative/unknown edge without the exact permitted positive-edge proof, `TRADE+NO_TRADE`, or non-entry/`CLOSE+ALLOCATE`.
- A selected exact-vehicle close/reduction or non-zero audited `DAILY_FINANCING` on a still-open attributed trade that changes the WAL-aware semantic ledger surface after authoring, or failure of the same-basis all-exit/TP edge or execution-cost reproof after final ledger/broker reconciliation. Exact-zero and manual financing components do not become system edge, while account-total zero still preserves non-zero system components. A WAL checkpoint alone is non-material; an unrelated edge-only row is non-material only if it changes neither the selected exact rows nor a global execution-cost cohort/digest. A cost-cohort change stales the receipt even when the row belongs to another exact lane. Missing raw financing transactions, broken gateway-fill identity, unreadable/arithmetic-inconsistent evidence, or any unresolved selected cash is blocking.
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

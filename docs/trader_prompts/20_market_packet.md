# Market Packet Reading

## Core Files

- `data/broker_snapshot.json`
  - Positions, pending orders, owner tags, quote timestamps, spreads, account truth.
  - Manual/tagless exposure is observed only; the trader does not protect, close, or use it as a fresh-entry blocker.
- `data/daily_target_state.json`
  - `status`, `target_jpy`, `minimum_target_jpy`, `progress_jpy`, `progress_pct`, `minimum_progress_pct`, `remaining_minimum_jpy`, `remaining_target_jpy`.
  - `daily_risk_budget_jpy` is the whole-day risk budget.
  - `per_trade_risk_budget_jpy` is the single-shot cap used by intents.
  - Do not conflate the whole-day and per-trade caps.
- `data/order_intents.json`
  - Current lane ids, status, entry, TP, SL, units, ATR-derived geometry, blockers, receipt text.
  - `LIVE_READY` means deterministic risk and strategy checks found no blockers.
- `data/ai_attack_advice.json`
  - Read-only rank of current `LIVE_READY` lanes.
  - It cannot grant live permission, increase risk, or place orders.

## Chart Layer

- `data/pair_charts.json` and `docs/pair_charts_report.md`.
- M1 is execution and recent jump quality.
- M5/M15 are the operating setup.
- M30/H1 are intraday structure.
- H4/D are higher-timeframe contradiction checks.
- `pair_charts[pair].confluence` summarises the higher-TF anchor and
  score balance the trader chronically under-weights — read it before
  picking direction. Fields: `score_balance` (LONG_LEAN / SHORT_LEAN /
  TIED), `higher_tf_regime`, `higher_tf_alignment` (ALIGNED / OPPOSED /
  NEUTRAL / MIXED). OPPOSED requires a declared counter-trend thesis;
  TIED demands smaller size or WAIT.
- Cite actual numbers for ATR, regime reading, family scores, disagreement, jump filters, session tag, and structure.
- State the next forecast first: `UP`, `DOWN`, `RANGE`, or `UNCLEAR`. `RANGE` is actionable only when the lane is `RANGE_ROTATION` and the intent metadata proves executable rails / box geometry.
- `family_scores.disagreement > 0.7` is a stand-aside signal unless one composite dominates and the regime gate supports it.
- New entries inside `last_jump_bars_ago < 5` require explicit justification.
- `chart.session` is mandatory context: current tag, `judas_armed`, `ny_midnight_open_price`, next killzone, and JPY holiday flag.

## Macro And Flow Layer

- `data/cross_asset_snapshot.json`
  - EUR/USD must cite DXY.
  - JPY pairs must cite DXY and `USB10Y_USD`.
  - Correlations are context, not execution permission.
- `data/flow_snapshot.json`
  - `STRESSED` spread blocks new entry.
  - `ELEVATED` spread requires the intent geometry to already reflect wider conditions.
- `data/currency_strength.json`
  - Direction against ranking needs explicit `risk_notes`.
- `data/levels_snapshot.json`
  - TP, SL, and invalidation should reference pivots, PDH/PDL/PDC, session ranges, and round numbers.
- `data/economic_calendar.json`
  - `pair_windows[].in_window=true` means WAIT unless the receipt explicitly records an override reason.
- `data/cot_snapshot.json`
  - Extreme positioning is a warning and must be cited when relevant.
- `data/option_skew_snapshot.json`
  - Missing vendor feed is `option:skew:unknown`, not invented option data.

## Evidence Rule

- A decision must cite packet refs that the verifier accepts.
- Do not cite raw indicator values as conclusions. Explain what the reading layer says those values mean in this regime.
- If a required number cannot be read, choose `REQUEST_EVIDENCE` or refresh the artifact.

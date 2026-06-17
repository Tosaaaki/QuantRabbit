# Market Packet Reading

## Core Files

- `data/broker_snapshot.json`
  - Positions, pending orders, owner tags, quote timestamps, spreads, account truth.
  - Manual/tagless exposure is TP-managed only; the trader does not SL, loss-close, or use it as a fresh-entry blocker.
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
- `data/learning_audit.json`
  - Audits whether learning evidence is allowed to influence lane ranking.
  - If the selected lane was moved by learning, cite `learning:audit` and
    `learning:lane:<lane_id>`; a blocked or stale audit is a no-trade gate for
    that lane.
- `data/verification_ledger.json`
  - Trader-readable summary of the current JSON artifacts, SQLite
    `verification_observations` / `effect_measurements`, and the latest MD
    report path.
  - Cite `verification:ledger` when using this packet. Cite individual
    `verification:*` refs when discussing reproducible blockers, missing
    artifacts, learning evidence, or recent effect metrics.
  - It is read-only evidence. It cannot grant live permission, override risk
    gates, or suppress broker-truth blockers.
- `data/execution_ledger.db` and `docs/verification_ledger_report.md`
  - SQL is the durable source for accumulated observations and measured
    outcomes; the MD report is the operator-readable digest.

## Chart Layer

- `data/pair_charts.json` and `docs/pair_charts_report.md`.
- M1 is execution and recent jump quality.
- M5/M15 are the operating setup.
- M30/H1 are intraday structure.
- H4/D are higher-timeframe contradiction checks.
- Do not pre-arm a range/failure retest LIMIT against a live M5/M15/M30
  impulse. Wait for a close-confirmed rejection first; a resting order
  fills before that confirmation exists.
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
  - USD pairs should read `XAU_USD` as USD-pressure context.
  - CAD pairs should read `WTICO_USD` or `BCO_USD` as commodity/CAD context.
  - Correlations are context, not execution permission.
- `data/flow_snapshot.json`
  - `STRESSED` spread blocks new entry.
  - `ELEVATED` spread requires the intent geometry to already reflect wider conditions.
- `data/currency_strength.json`
  - Direction against ranking needs explicit `risk_notes`.
- `data/levels_snapshot.json`
  - TP, SL, and invalidation should reference pivots, PDH/PDL/PDC, session ranges, and round numbers.
- `data/market_context_matrix.json`
  - Advisory pair/side matrix joining chart, strength, cross-asset, flow, levels, calendar, COT, and option-skew observations.
  - It raises certainty by exposing `supports`, `rejects`, `warnings`, and `missing` evidence; it must not create new live blockers or reduce `LIVE_READY` lane count.
  - TRADE receipts should cite `matrix:<PAIR>:<SIDE>` and carry the strongest reject into the 20-minute counterargument.
- `data/economic_calendar.json`
  - `pair_windows[].in_window=true` means WAIT unless the receipt explicitly records an override reason.
- `data/news_items.json` and `data/news_health.json`
  - `news_health.status` tells whether the current news layer is fresh enough for the active market window.
  - `news_items.items[]` gives current public-news context by pair/currency/topic. Cite it as context for catalysts, macro risk, or missing evidence; it cannot override broker truth, current `LIVE_READY`, spread, event, or risk gates by itself.
- `data/cot_snapshot.json`
  - Extreme positioning is a warning and must be cited when relevant.
- `data/option_skew_snapshot.json`
  - Optional. When no provider is configured, it is a disabled artifact and should not be cited as missing evidence.

## Evidence Rule

- A decision must cite packet refs that the verifier accepts.
- Do not cite raw indicator values as conclusions. Explain what the reading layer says those values mean in this regime.
- If a required number cannot be read, choose `REQUEST_EVIDENCE` or refresh the artifact.

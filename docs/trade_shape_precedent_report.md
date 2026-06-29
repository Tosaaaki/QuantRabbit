# Trade Shape Precedent

- Generated at UTC: `2026-06-29T23:25:37+00:00`
- Status: `TRADE_SHAPE_PRECEDENT_READY`
- Sample pairs: `USD_JPY`
- Constraint: The observed 2025 sample is USD_JPY-heavy, so lessons are expressed as theme/location/session/shape contracts, not as pair-specific permission.

## Lessons

### THEME_READ

- Lesson: A trade shape starts from the current theme, not from a single pair memory.
- Reusable rule: Map the cleanest expression of the bought/sold currency theme, then select pair/vehicle only after current evidence confirms the theme.
- Blocked behavior: Do not replay the 2025 USD_JPY precedent as a USD_JPY-only rule or as permission to ignore other pairs.
- Allowed use: Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.

### LOCATION_24H

- Lesson: 24h location is a first-class edge filter.
- Reusable rule: Prefer pair-agnostic broad-discount LONG and broad-premium SHORT structures when they match current thesis and risk geometry.
- Blocked behavior: Block sell-the-low / buy-the-high churn unless a separate breakout/trend proof exists.
- Allowed use: Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.

### H1_H4_ALIGNMENT

- Lesson: H1/H4 alignment must be evidence, not a slogan.
- Reusable rule: Use H1/H4 as current context buckets and require extra current reason when the current alignment conflicts with the positive bounded precedent.
- Blocked behavior: Do not infer H4 support from H1-only data; missing H4 is a data gap, not permission.
- Allowed use: Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.

### SESSION

- Lesson: Session changes payoff and should shape aggressiveness.
- Reusable rule: Rank session-conditioned shapes separately and prefer the current liquid/session-compatible expression.
- Blocked behavior: Do not force the same vehicle in Tokyo, London, NY overlap, and off-hours without session-specific evidence.
- Allowed use: Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.

### BOUNDED_ADVERSE_ADD

- Lesson: Adverse adds were only useful when bounded and thesis-specific.
- Reusable rule: An adverse add can only be considered when invalidation, max entries, margin, and harvest path are explicit before the add.
- Blocked behavior: Block martingale-style averaging, margin rescue, or adds whose only reason is a red open P/L.
- Allowed use: Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.

### WITH_MOVE_PYRAMID

- Lesson: With-move pyramiding was not automatically positive in bounded replay.
- Reusable rule: Pyramids need independent fresh edge and portfolio room; a move already in profit is not enough.
- Blocked behavior: Block stack-on-green behavior when the new layer lacks its own entry location, TP, invalidation, and margin proof.
- Allowed use: Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.

### SL_FREE_THESIS_INVALIDATION

- Lesson: SL-free does not mean loss-free; it means exits must be thesis invalidation based.
- Reusable rule: When using SL-free or wide-catastrophe-stop logic, store the thesis, invalidation level/timeframe, and review trigger before entry.
- Blocked behavior: Block tight broker SLs inside normal noise/major-figure battle zones and block loss-side closes from red P/L alone.
- Allowed use: Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.

### HOUSEKEEPING_HARVEST

- Lesson: Harvest exits are the reusable profit engine; housekeeping must not become panic closing.
- Reusable rule: Prefer attached TP / harvest / profit capture when current forecast weakens or range rail is reached.
- Blocked behavior: Block market-close leakage where one large give-up close erases multiple average winners.
- Allowed use: Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.

### MARGIN_CLOSEOUT_FAILURE

- Lesson: Margin closeout is a hard failure mode, not a strategy exit.
- Reusable rule: Treat margin capacity as pre-entry and add-layer risk control; once margin pressure appears, do not add risk to rescue the basket.
- Blocked behavior: Block any target policy that requires trading through margin pressure to maintain a daily cadence.
- Allowed use: Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.

### LONG_UNATTENDED_HOLD_FAILURE

- Lesson: Long unattended holds contaminate precedent because they mix thesis edge with carry tail risk.
- Reusable rule: Separate bounded intraday precedent from >=12h carry/tail rows before citing edge.
- Blocked behavior: Block using raw long-hold winners to justify unattended exposure without current thesis-evolution and margin controls.
- Allowed use: Advisory precedent only; current forecast, risk, spread, event, and gateway checks remain authoritative.

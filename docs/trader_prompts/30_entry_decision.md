# Entry Decision

## Use When

- Daily target remains open.
- Account is flat or existing trader-owned exposure is layerable.
- Current `LIVE_READY` lane(s) exist in `data/order_intents.json`.

## Decision Order

1. Read `docs/trader_prompts/20_market_packet.md`.
2. State the pair-level next forecast for each candidate pair: `UP`, `DOWN`, `RANGE`, or `UNCLEAR`. Treat forecast target/invalidation levels inside current M1/M5 ATR/spread noise as non-structural. In an active range, lower-half price carries bounce/retest risk and upper-half price carries fade risk before a breakout is proved.
3. Build the all-horizon opportunity map before deciding: M1/M5 for immediate execution, M15/M30/H1 for operating swing, and H4/D for anchor/bias. A valid trade may come from any horizon when entry, invalidation, TP, spread, and portfolio gates pass. Do not collapse the packet to a single short-term read.
4. Treat soft-only close sidecars without explicit Gate B as advisory for entries: keep the existing TP-managed runner under monitoring, but do not let that advisory block unrelated current `LIVE_READY` lanes on another pair or horizon. If `protection_sidecars.position_close_recommendations[].blocks_non_close_actions=false`, do not write `CLOSE` just to "test" the verifier; `CLOSE` is not a valid action in this branch. Hard close evidence or explicit Gate B still requires close-first discipline.
5. Read `data/market_context_matrix.json` for every current candidate. Use it to raise confidence and expose the strongest counterargument, not to invent a new blocker or reduce trade count.
6. Read `data/news_health.json` and current news refs before selecting `TRADE`. If news-health is missing, ERROR/BLOCK, or carries BLOCK issues, write WAIT / REQUEST_EVIDENCE with that named blocker; do not trade on a stale market story.
7. List every current `LIVE_READY` lane.
8. Intersect `ai_attack_advice.recommended_now_lane_ids` with current tradeable lanes.
9. If an advised lane has `learning_influences`, require `data/learning_audit.json`
   to be non-blocked and to cover that lane before treating the advice as
   executable. Cite `learning:audit` and `learning:lane:<lane_id>` in the receipt.
10. Read `data/operator_precedent_audit.json` and
   `data/manual_market_context_audit.json` if present. Use them only as
   advisory ranking/context among already-current `LIVE_READY` lanes; absence
   of alignment is not a blocker, and alignment cannot override current risk,
   forecast, spread, event, broker-truth, or close Gate A/B checks. If you cite
   the 2025 manual precedent as a reason to attack harder, also state whether
   the current lane matches the manual technical context (`prefer_h1_alignment`
   / `prefer_session_jst`) or cite a separate current deterministic edge.
11. If the target is open and the first advised lane is tradeable, include it in the selected basket unless a named deterministic gate now blocks it.
12. If advice spans multiple distinct pairs, include one lane per advised pair up to portfolio capacity when practical; otherwise the verifier records a warning and the gateway cycle expands the accepted trade to the deterministic prefilter basket so margin, cumulative risk, duplicate geometry, and position-count gates decide what fits.
13. Prefer a `MARKET` variant for immediate participation when it is current `LIVE_READY`; pending entries are basket-counted by the gateway and are not blanket no-trade reasons. Exception: `BREAKOUT_FAILURE` must be at the retest/rejection side of the M5/M15 box. For SHORT, do not market-sell the lower half/support and do not arm a lower-half sell-stop; wait for upper-half resistance rejection/LIMIT or require a separate true trend-continuation breakout lane. For LONG, do not market-buy the upper half/resistance and do not arm an upper-half buy-stop; wait for lower-half support rejection/LIMIT or require a separate true trend-continuation breakout lane.
14. If current trader-owned pending entries already consume portfolio capacity, explicitly decide whether to keep that pending basket or replace it. A `TRADE` that needs capacity for current `LIVE_READY` lanes may include `cancel_order_ids` for current trader-owned pending entry ids that should be cleared before gateway validation; never name manual/unknown orders. When self-improvement flags `PENDING_ENTRY_CANCEL_REVIEW_REQUIRED` and a current replacement lane is `LIVE_READY`, prefer this `TRADE` + `cancel_order_ids` path over a standalone `CANCEL_PENDING`, because the verifier rejects standalone cancel while executable replacement risk exists.
15. Fill `twenty_minute_plan` before choosing the final action. The trader runs roughly one decision every 20 minutes, so a receipt that only says "trend looks good" or "timing unclear" is too shallow. State the next-cycle primary path, the failure path, the exact entry/hold trigger, the invalidation/cancel trigger, the strongest counterargument, and what must be checked on the next cycle.
16. In scheduled automation, use `trader-draft-decision` as the default receipt composer after `cycle-refresh`. If it emits TRADE and `gpt-trader-decision` accepts it, proceed to the gateway. If it emits WAIT / REQUEST_EVIDENCE because a named blocker wins, do not hand-write a TRADE unless a refreshed artifact clears that blocker.
17. Write exactly one `data/codex_trader_decision_response.json`.

## Valid Actions

- `TRADE`
- `CANCEL_PENDING`
- `WAIT`
- `REQUEST_EVIDENCE`

Do not write `CLOSE` from this branch. A soft close advisory that lacks explicit
Gate B is a monitoring/reprice input, not a close-first router decision. If a
later refresh upgrades the sidecar to `blocks_non_close_actions=true`, the
router will move to `position_management`.

## Operator Precedent (`data/operator_precedent_audit.json`, `data/manual_market_context_audit.json`, `docs/manual_trading_2025_evidence.md`)

The 5%/10% daily target reproduces the operator's own 2025 manual record on
this account. Raw balance moved 200k → 1.23M peak in ~6 weeks, but that
includes 634k of additional funding; funding-adjusted trading equity still
peaked at 600.6k (**+400.6k / +200.28%**) and ended at 469.2k (**+269.2k /
+134.60%**). The best funding-adjusted 30d window was **+457.5k / +319.72%**,
after subtracting 634k of net transfers inside that window. USD_JPY only, 411
exit events. The shape of that edge, as advisory evidence for lane selection
— never a substitute for current risk geometry or contract gates:

- Fewer, larger, faster: ~10 exit events/day at meaningful size, payoff 1.30,
  median hold 29 minutes — not 30 micro-trades across 8 pairs.
- Bounded replay is the usable precedent, not the raw long-hold tail. Exclude
  >=12h holds and margin closeouts before copying the shape.
- Technical shape: USD_JPY extreme rotation, not blind trend chase.
  `LONG_LOWER_THIRD_24H` and `SHORT_UPPER_THIRD_24H` were the strongest
  replayable buckets; `SHORT_WITH_H1_TREND` and middle-third shorts were bad.
- Position building: the operator did use nanpin-like averaging into adverse
  same-side USD_JPY exposure, but the replayable part was bounded. In the
  >=12h/margin-closeout-excluded profile, averaging-into-adverse clusters were
  net positive and small (median 3 entries, max 4, average adverse add about
  7 pips), while bounded pyramiding-with-the-move was negative. This is
  evidence for selective, risk-budgeted retest/add logic only; it is not
  permission for unbounded martingale, weak forecasts, or same-pair additions
  outside current basket risk, current ATR-bounded adverse-add validation, and
  `LIVE_READY` validation.
- When an intent is a same-pair same-side add, read its
  `position_building.same_pair_add_type`. Do not describe a
  `PYRAMID_WITH_MOVE` add as nanpin; manual precedent supports only bounded
  adverse retest/add behavior after current gates pass.
- H1 context: bounded `AGAINST_H1_TREND` paid far better than
  `WITH_H1_TREND`. A lane using the 2025 precedent as an aggression reason must
  explain whether current H1/M5 and 24h-location context is comparable.
- Session: historical session buckets are descriptive/ranking evidence only,
  not a hard time-of-day no-trade gate. The AI trader is expected to run across
  all hours and let current spread, ATR, forecast, flow, broker truth, and risk
  geometry decide whether a lane is executable.
- The operator's own blowup mode was holding decayed positions past ~12h
  (margin closeouts −217k) — the thesis-horizon expiry and disaster stop
  exist to bound exactly that; do not fight them.

`manual_market_context_audit` adds the technical replay layer around those
manual entries. It may gate only the *use of the precedent as an aggression
reason*: a lane that conflicts with the mined H1/M5/session context needs its
own current deterministic edge. It is not a no-trade gate by itself. If a
`TRADE` receipt cites `operator:precedent`, the verifier now also requires
`manual:market_context`, at least one selected lane aligned by the current
operator-precedent audit, and no bounded manual H1/M5/session/24h-location
conflict on that selected precedent-aligned lane.

## WAIT Discipline

- WAIT is valid only when a named contract or packet gate fires.
- Generic market prose is not a gate.
- Past losses, low old capture rate, missing positive mined evidence, or stale rejection memory are audit context only after current risk geometry is `LIVE_READY`.
- If `progress_pct < 50` and at least three `LIVE_READY` lanes exist, WAIT must reject each lane with a specific M5 chart-story sentence and cite the exact gate.

### Invalid WAIT reasons (user 2026-05-11「マージン使えてないし、他の通貨も入れるんじゃないの？」)

WAIT is **not** valid when any of these is the only argument:

- "ASIA/quiet session, wait for London/NY" — sessions are an input to TF
  weighting, not a blanket no-trade gate. ASIA range scalps and overnight
  swing carryovers are valid trades when the structure says so.
- "Existing positions cover this pair" — under SL-free + OANDA hedging
  same-pair opposite-side adds zero margin (`feedback_hedging.md`); a SHORT
  on a pair already LONG is a counter-trend pullback trade, not a
  duplicate. Phase 2 mirror lanes (`feature_dynamic_tf_weighting`) exist
  exactly so you can take that trade.
- "Margin headroom = wait for clearer setup" — if `margin_available_jpy`
  > one-position margin AND a `LIVE_READY` lane has combined MTF+PA
  score ≥ +12, take it. The system was sized so 3-4 concurrent positions
  is the design point, not the ceiling.
- "Direction bias of pair_charts long/short_score" — that's the historical
  aggregate. Phase 2 explicitly enables the AI trader to override it when
  the structural lens (PA aggregate + micro override + MTF confluence)
  favors the opposite. A SHORT lane scoring higher combined than its LONG
  twin is a trade, not a contradiction.

When margin is available AND ≥1 lane combined score ≥ +12, prefer TRADE.
WAIT requires citing a specific *new* blocker not present at the time the
margin freed up.

## Confluence Audit (mandatory before TRADE)

Regime label alone (e.g. "M5 TREND_DOWN") is not a direction signal. Before
finalizing direction, write all five lines in `thesis`. If any is empty or
contradicts your chosen direction, downgrade size, switch method, or WAIT.

1. **Highest TF anchor** — cite D/H4 regime explicitly. SHORT against an
   H4/D TREND_UP needs a declared counter-trend thesis (reversal, key
   level rejection, exhaustion bounce) with a bounded scope; otherwise it
   is method-misuse.
2. **Entry TF quality** — for each entry TF (M5/M15/M30) cite the
   `Read=` qualifier (TREND_FRESH / TREND_WEAK / TRANSITION /
   BREAKOUT_PENDING / FAILURE_RISK). TREND_CONTINUATION method requires
   FRESH, not WEAK. WEAK + TRANSITION means the move is dying — pick
   BREAKOUT_FAILURE or wait for the next leg.
3. **Momentum / exhaustion read** — cite RSI, %R, MFI, AroonOsc on entry
   TFs. If multiple oscillators sit at the same extreme as your direction
   (SHORT into RSI floor + %R near −100 + MFI floor), you are entering at
   exhaustion. Either declare reversal thesis or step aside.
4. **Score balance** — cite `long_score` vs `short_score`. When they are
   close, the chart is not voting either way; size down or WAIT.
5. **Strongest counter-argument** — name the single most damaging
   evidence against your direction from the chart_story and explain why
   you still win. Prefer the strongest `rejects[]` message from
   `market_context_matrix` when it exists, then compare it against the
   selected lane's risk geometry. If you can't find one, you haven't read
   the chart.

## 20-Minute Plan (mandatory for TRADE / WAIT / REQUEST_EVIDENCE)

Add `twenty_minute_plan` to the decision receipt:

- `horizon_minutes`: `20`.
- `primary_path`: what price / structure / flow should do before the next cycle if the decision is right.
- `failure_path`: what would make the decision wrong before the next cycle.
- `entry_or_hold_trigger`: the concrete trigger for entry, basket keep, or WAIT continuation.
- `invalidation_or_cancel_trigger`: the concrete level / structure / packet gate that cancels the idea.
- `counterargument`: the strongest current evidence against the chosen action, not a generic risk sentence.
- `next_cycle_check`: what the next trader must verify first if no fill / no close / no new blocker occurs.
- `evidence_refs`: known packet refs supporting this plan, including the selected `intent:<lane_id>` for TRADE and at least one chart ref when current tradeable lanes exist.

This does not create a new market-risk gate. It forces the trader to expose
the 20-minute scenario tree so the verifier and the next cycle can catch
shallow or contradictory reasoning.

## Hedge Timing Rubric

Opening an opposite-side position on a pair you already hold is hedging,
not new participation. Classify and justify in `thesis`:

- (a) **Lock-gain hedge** — existing exposure is profitable, opposite-side
  locks the win while preserving optionality. Usually fine.
- (b) **Reversal hedge** — existing exposure is underwater AND a reversal
  signal has fired (BOS / CHOCH against the existing side on entry TF +
  higher-TF confirmation). Conditional — cite the reversal evidence.
- (c) **Continuation hedge** — existing exposure is underwater AND price
  is still trending against it. This freezes the loss at the worst
  moment. **Default: avoid.** If taken, declare the structural level that
  invalidates the existing side and why catching that move is worth
  blocking the bounce-back recovery.

For any selected hedge, cite `intent.metadata.hedge_timing_class`,
`hedge_review_trigger`, and the planned unwind condition. A hedge without an
unwind plan is not a time-efficient trade; it is passive loss-freezing.

- Selected lane id and basket lane ids.
- Thesis, method, narrative, chart story, invalidation, TP, SL, units, expected reward, worst-case loss.
- Rejected alternatives.
- Risk notes naming `per_trade_risk_budget_jpy`, spread state, calendar state, strength alignment or conflict, and any COT warning.
- Evidence refs for broker, target, intent, campaign, strategy, story, charts, matrix, cross-asset, strength, flow, levels, calendar, news, news-health, COT, enabled option skew, and attack advice when used. A `TRADE` must cite `news:health` and `news:items` or `news:current`.
- `twenty_minute_plan` with packet refs as above.

## Specialist Reviews

- Optional and read-only.
- Must declare `read_only=true` and `live_permission=false`.
- Must not select lanes, set units, stage orders, send orders, cancel orders, or change risk budgets.
- Strategy reviews are keyed by `lane_id` and `method`; a review for one method cannot authorize another.

# Position Management

## Use When

- Trader-owned position exists.
- Trader-owned position is missing TP or SL.
- Manual/tagless position is missing TP or has a profit-side TP/partial-close opportunity.
- Trader-owned pending entry needs cancel review.
- Target reached and protection-first behavior is active.

## Valid Actions

- `PROTECT`
- `BREAK_EVEN_STOP`
- `TIGHTEN_SL`
- `TAKE_PROFIT`
- `TAKE_PROFIT_MARKET`
- `CLOSE`
- `CANCEL_PENDING`
- `WAIT`

## Protection Rules

- Missing TP / SL on trader-owned exposure is a repair requirement, **except** under
  SL-free mode (`QR_TRADER_DISABLE_SL_REPAIR=1`): trader-owned SL=None is
  intentional, and a missing broker TP is a no-broker-TP runner unless
  `QR_ENABLE_MISSING_TP_REPAIR=1` is explicitly set or a profitable insurance-TP
  condition fires. Treat these runners as layerable for fresh-entry routing;
  do NOT propose PROTECT/TIGHTEN_SL just to satisfy a static TP/SL checklist.
- Operator-managed manual/tagless positions are TP-only: TP repair/rebalance and
  profit-side partial close are allowed when already profitable or when explicit
  missing-TP repair is enabled, but underwater manual/tagless runners do not
  force this branch and must not block fresh trader entries. SL writes, adverse
  partial closes, and market CLOSE are forbidden.
- Existing SL must not be widened.
- Existing TP may be moved only by TP-management actions with current-price safety.
- When the router reason says `TP rebalance required`, run the `tp-rebalance`
  sidecar before treating WAIT as complete. Do not replace this with prose:
  the command is the executable TP decision for that branch.
- Profitable protected positions may tighten SL to break-even or better.
- Under SL-free, missing SL is still intentional and must not be repaired while
  underwater or inside ordinary execution noise. A trader-owned position that is
  already profitable may use `BREAK_EVEN_STOP` at entry or better only after
  executable profit pips clear a fresh market-derived micro-noise envelope:
  current spread plus fresh M5 ATR and quick M1/M5 realized range from recent
  candles inside the current M5 management window. Stale pair-chart volatility
  must not be used; if fresh quick volatility is unavailable, defer the stop
  rather than falling back to spread alone. When profit exceeds that envelope,
  place the stop behind the current executable price by the same envelope and
  clamp it so it can never be worse than entry. This is a profit-to-flat or
  profit-lock escape hatch, not initial SL repair or trailing. Manual/tagless
  positions remain TP-only.
- If an already-profitable trader-owned runner has macro reversal against it,
  `TAKE_PROFIT_MARKET` may close it immediately. This is profit harvest, not
  loss-side thesis invalidation; it must be blocked if the latest broker snapshot
  is not profitable or the position is manual/tagless/external.
- BB rail context matters for runners: if a profitable SHORT is bouncing into
  the M1/M5 upper Bollinger rail with overbought StochRSI/Williams/MFI, treat it
  as short-continuation evidence. If the existing TP is already at the opposite
  M5 lower rail, keep that TP and use BE/profit-lock; do not shrink the target
  just because the micro matrix says HARVEST. Mirror this for LONG at the lower
  rail.
- Contradicted trader-owned positions may close, **but only on market-structure
  evidence** (see CLOSE rules below).
- Fresh entries are blocked only by non-layerable trader-owned or external exposure;
  protected trader-owned exposure may add only through portfolio validation.

## CLOSE Decision Rules (SL-free; user 2026-05-08「市況>リスク」, 2026-05-28「妥当な損切りならやっていい」)

CLOSE is for genuine thesis breakdown, **not** for risk-budget overshoot. Under
SL-free the per-trade risk number is advisory; market structure is authoritative.
A justified loss-cut is allowed when hard Gate A proves the thesis is broken,
or when softer Gate A is paired with explicit Gate B; holding a broken thesis is
not the objective.
Do not combine loss-cut and re-entry in one `TRADE` receipt. Close the broken
position first, refresh broker truth / intents, then re-enter only if the next
cycle still produces a fresh `LIVE_READY` lane.
Fresh hard sidecar Gate A close evidence, or soft sidecar evidence paired with
explicit Gate B, also blocks `PROTECT` / `TIGHTEN_SL` as a receipt-level escape
hatch: submit the verified `CLOSE` first. Soft-only evidence without Gate B is
advisory for non-CLOSE actions; keep TP/profit management active and do not use
it as a blanket reason to stop separate current `LIVE_READY` entries on other
pairs or horizons. If choosing `CLOSE` from soft evidence, surface
`CLOSE_OPERATOR_AUTH_REQUIRED` unless Gate B is present.

`TAKE_PROFIT_MARKET` is not this loss-side CLOSE path. Use it only for
currently profitable trader-owned positions when the adaptive TP / macro-micro
flow says the runner has reversed and waiting for TP risks giving back MFE.
Do not use it for underwater positions, manual/tagless positions, or external
positions.

### Valid CLOSE triggers (machine-checkable Gate A)

- **Structure-event invalidation (§10 Gate A — primary)**: M15 OR H4 prints BOS
  or CHOCH against the position side (LONG → counter-direction is DOWN, SHORT →
  UP) AND the event is **close-confirmed** (chart_story does NOT carry the
  `:wick` suffix). This is the path implemented in
  `gpt_trader._close_thesis_invalidated`; a single counter-side BOS/CHOCH on
  **either** M15 or H4 is sufficient when its breaking candle closed beyond
  the broken pivot. Cite the TF + event + price in the receipt (e.g. "H4
  BOS_UP@113.587 prints against SHORT thesis, close-confirmed"). Do **not**
  wait for H4/H1/M30 regime labels to all print aligned TREND — `chart_reader`
  regimes lag structure during reversals (transitions sit at
  UNCLEAR/RANGE/FAILURE_RISK while swing structure has already flipped), and
  waiting for the lag is what locks losing positions underwater while the
  move runs. Also do **not** propose CLOSE on a `struct=...:wick` event — the
  wick suffix marks a stop-hunt sweep where the high/low was tagged but the
  close held inside the prior range (added 2026-05-13 after the AUD_JPY M15
  BOS_UP@114.146 0.4-pip wick incident). This hard Gate A path carries
  standing loss-cut authorization and does not need a 5-minute token.
- **Invalidation-price hit (§10 Gate A — receipt-driven)**: receipt populates
  `invalidation_price` + `invalidation_tf` AND broker bid/ask clears the level
  beyond the anti-wick buffer (default `QR_THESIS_INVALIDATION_BUFFER_PIPS=2.0`)
  with chart/technical confirmation across operating timeframes. Cite the
  level + TF + chart/technical evidence. Do not cut on a naked one-tick wick.
  This hard Gate A path also carries standing loss-cut authorization.
- **Fresh prediction/thesis sidecar invalidation (§10 Gate A — recovery edge
  lost)**: a sidecar generated after the current
  `broker_snapshot.fetched_at_utc` marks the same trade `REVIEW_CLOSE` /
  `RECOMMEND_CLOSE`: `position_thesis_report.json`,
  `thesis_evolution_report.json`, or `forecast_persistence_report.json`.
  `position_thesis` can also flag an underwater trader-owned position whose
  entry thesis has no usable invalidation price when broker truth has moved
  beyond the entry-price anti-wick buffer and M5/M15/M30/H1 technicals confirm
  the move against the side. This no-ledger fallback must degrade to HOLD when
  the current position prediction stack still strongly supports the position
  direction; early loss-cut is for broken recovery edge, not cutting into a
  supported recovery only because the ledger was incomplete. This is the
  machine-checkable "no longer likely to recover to plus" path. Cite
  `position:thesis:<trade_id>`,
  `position:evolution:<trade_id>`, or `position:persistence:<trade_id>` and
  the sidecar reason. Stale sidecars are ignored. `thesis_evolution`
  BROKEN / RECOMMEND_CLOSE is hard Gate A and carries standing authorization.
  `position_thesis` REVIEW_CLOSE is also hard only when the sidecar records
  invalidation-hit or structural-break evidence plus M5/M15/M30/H1 technical
  confirmation. Adverse-entry-buffer-only or score-only `position_thesis`
  REVIEW_CLOSE and `forecast_persistence` RECOMMEND_CLOSE are softer Gate A
  and still require explicit env/token Gate B.

Macro shock, large unrealized loss, or margin pressure can strengthen the
reason to review a thesis, but none of them is a standalone Gate A. Convert the
concern into one of the machine-checkable triggers above, or WAIT. For soft Gate
A, Gate B still requires operator-controlled authorization:
`QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell, or a fresh
`data/.operator_close_token` file. The receipt's `operator_close_authorized`
field is advisory audit text only and is not accepted as authorization.

### NOT valid CLOSE triggers (do NOT propose CLOSE on these alone)

- `unrealized_pl_jpy < -per_trade_risk_budget_jpy` — under SL-free this is **advisory
  only**, not a hard close gate. -1,000 JPY ≈ 0.4% NAV is noise.
- **Margin pressure is not a CLOSE trigger.** Thin `margin_available_jpy`
  blocks new entries, cancels stale pending risk, and forces smaller future
  baskets. It does **not** authorize closing a directionally valid position.
  A professional trader sizes so the broker never decides; once a position
  exists, CLOSE still needs Gate A thesis invalidation plus the applicable Gate
  B path.
- `Loss > NAV 5%` by itself — large drawdown is an incident and requires review,
  but it is not a machine-checkable thesis invalidation unless structure or an
  explicit invalidation price confirms the thesis is wrong.
- Macro/news shock by itself — cite it as context, but still express the thesis
  break through structure or a broker-truth invalidation level.
- `chart_aggregate.long_score > short_score` (or vice versa) on its own — long/short
  scores aggregate per-TF reads but a single noise flip on M5/M1 can move them.
  Look at H4/H1/M30 directly before acting on aggregate.
- M5/M1 short-term flip while H4/H1 still align with position — that's pullback noise
  inside the macro thesis, not a reversal. SL-free design specifically holds through
  this.
- Position underwater for less than 30 minutes — give the structure time to play out.

### When in doubt, choose WAIT

`feedback_no_tight_sl_thin_market.md`「SLいらない」 / `feedback_offense_sizing.md`
「損失出さないで稼ぎまくる」 / `feedback_market_over_risk_budget.md`「市況>リスク」.
Holding through noise is the SL-free design's whole point — premature CLOSE locks in
the noise loss AND forfeits the TP that's still reachable.

## Pending Orders

- Pending entries are inherited across scheduler handoff.
- Do not cancel another cycle's pending order without an explicit reason in the next decision receipt.
- `CANCEL_PENDING` must list current trader-owned OANDA pending entry ids in `cancel_order_ids`; verified ids are canceled by the gateway cycle, and no fresh entry is sent in that same cycle.
- If the daily target is already reached, trader-owned pending entry ids are canceled instead of left fillable.
- Manual/tagless pending orders are observed only.

## Commands

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli tp-rebalance \
  --snapshot data/broker_snapshot.json \
  --pair-charts data/pair_charts.json

PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json

QR_LIVE_ENABLED=1 ./scripts/run-autotrade-live.sh \
  --reuse-market-artifacts \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send
```

## Receipt Notes

- Cite `broker:snapshot`, affected trade/order ids, and the exact repair reason.
- If choosing WAIT with open trader-owned exposure, state why no gateway action is required now.
- If closing, cite the contradiction in current broker/market packet evidence.

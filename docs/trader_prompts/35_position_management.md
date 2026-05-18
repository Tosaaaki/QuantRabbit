# Position Management

## Use When

- Trader-owned position exists.
- Trader-owned position is missing TP or SL.
- Manual/tagless position is missing TP or has a profit-side TP/partial-close opportunity.
- Trader-owned pending entry needs cancel review.
- Target reached and protection-first behavior is active.

## Valid Actions

- `PROTECT`
- `TIGHTEN_SL`
- `TAKE_PROFIT`
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
  profit-side partial close are allowed when already profitable, but SL writes,
  adverse partial closes, and market CLOSE are forbidden.
- Existing SL must not be widened.
- Existing TP may be moved only by TP-management actions with current-price safety.
- Profitable protected positions may tighten SL to break-even or better — **disabled
  under SL-free**: do NOT auto-tighten on profit, the operator harvests via TP only.
- Contradicted trader-owned positions may close, **but only on market-structure
  evidence** (see CLOSE rules below).
- Fresh entries are blocked only by non-layerable trader-owned or external exposure;
  protected trader-owned exposure may add only through portfolio validation.

## CLOSE Decision Rules (SL-free, user 2026-05-08「市況>リスク」)

CLOSE is for genuine thesis breakdown, **not** for risk-budget overshoot. Under
SL-free the per-trade risk number is advisory; market structure is authoritative.

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
  BOS_UP@114.146 0.4-pip wick incident).
- **Invalidation-price hit (§10 Gate A — receipt-driven)**: receipt populates
  `invalidation_price` + `invalidation_tf` AND broker bid/ask trades through
  the level (LONG: bid ≤ level, SHORT: ask ≥ level). Cite the level + TF.

Macro shock, large unrealized loss, or margin pressure can strengthen the
reason to review a thesis, but none of them is a standalone Gate A. Convert the
concern into one of the two machine-checkable triggers above, or WAIT. Gate B
still requires
operator-controlled authorization: `QR_OPERATOR_CLOSE_OVERRIDE=1` in the
operator shell, or a fresh `data/.operator_close_token` file. The receipt's
`operator_close_authorized` field is advisory audit text only and is not
accepted as authorization.

### NOT valid CLOSE triggers (do NOT propose CLOSE on these alone)

- `unrealized_pl_jpy < -per_trade_risk_budget_jpy` — under SL-free this is **advisory
  only**, not a hard close gate. -1,000 JPY ≈ 0.4% NAV is noise.
- **Margin pressure is not a CLOSE trigger.** Thin `margin_available_jpy`
  blocks new entries, cancels stale pending risk, and forces smaller future
  baskets. It does **not** authorize closing a directionally valid position.
  A professional trader sizes so the broker never decides; once a position
  exists, CLOSE still needs Gate A thesis invalidation plus Gate B operator
  authorization.
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
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json

./scripts/run-autotrade-live.sh \
  --reuse-market-artifacts \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send
```

## Receipt Notes

- Cite `broker:snapshot`, affected trade/order ids, and the exact repair reason.
- If choosing WAIT with open trader-owned exposure, state why no gateway action is required now.
- If closing, cite the contradiction in current broker/market packet evidence.

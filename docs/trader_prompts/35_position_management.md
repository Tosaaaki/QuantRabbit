# Position Management

## Use When

- Trader-owned position exists.
- Trader-owned position is missing TP or SL.
- Trader-owned pending entry needs cancel review.
- Target reached and protection-first behavior is active.

## Valid Actions

- `PROTECT`
- `TIGHTEN_SL`
- `CLOSE`
- `CANCEL_PENDING`
- `WAIT`

## Protection Rules

- Missing TP / SL on trader-owned exposure is a repair requirement, **except** under
  SL-free mode (`QR_TRADER_DISABLE_SL_REPAIR=1`): trader-owned SL=None with TP set
  is intentional, treat as protected, do NOT propose PROTECT/TIGHTEN_SL.
- Operator-managed manual/tagless positions are observed only.
- Existing SL must not be widened.
- Existing TP is not moved by the protection gateway.
- Profitable protected positions may tighten SL to break-even or better — **disabled
  under SL-free**: do NOT auto-tighten on profit, the operator harvests via TP only.
- Contradicted trader-owned positions may close, **but only on market-structure
  evidence** (see CLOSE rules below).
- Fresh entries are blocked only by non-layerable trader-owned or external exposure;
  protected trader-owned exposure may add only through portfolio validation.

## CLOSE Decision Rules (SL-free, user 2026-05-08「市況>リスク」)

CLOSE is for genuine thesis breakdown, **not** for risk-budget overshoot. Under
SL-free the per-trade risk number is advisory; market structure is authoritative.

### Valid CLOSE triggers (any one is sufficient)

- **Structure-event invalidation (§10 Gate A — primary)**: M15 OR H4 prints BOS
  or CHOCH against the position side (LONG → counter-direction is DOWN, SHORT →
  UP). This is the path implemented in `gpt_trader._close_thesis_invalidated`;
  a single counter-side BOS/CHOCH on **either** M15 or H4 is sufficient. Cite
  the TF + event + price in the receipt (e.g. "H4 BOS_UP@113.587 prints against
  SHORT thesis"). Do **not** wait for H4/H1/M30 regime labels to all print
  aligned TREND — `chart_reader` regimes lag structure during reversals
  (transitions sit at UNCLEAR/RANGE/FAILURE_RISK while swing structure has
  already flipped), and waiting for the lag is what locks losing positions
  underwater while the move runs.
- **Invalidation-price hit (§10 Gate A — receipt-driven)**: receipt populates
  `invalidation_price` + `invalidation_tf` AND broker bid/ask trades through
  the level (LONG: bid ≤ level, SHORT: ask ≥ level). Cite the level + TF.
- **Macro shock**: news event (intervention, FOMC, NFP gap, geopolitical) invalidates
  the pair thesis. Cite the news_digest line in evidence_refs.
- **Structural margin pressure**: closing this position is the only way to free
  margin so the portfolio can keep capturing better setups. Trigger when
  `margin_available_jpy` is below the `MIN_PRODUCTION_LOT_UNITS` floor for
  every fresh `LIVE_READY` lane (i.e. the basket cannot add a single 1000u
  entry without freeing capital). Quantify the margin trade-off in receipt.
- **Loss > NAV 5%** (≈11k JPY at current equity): structural collapse territory per
  `feedback_independent_judgment.md`. Operator may CLOSE to cap damage.

All five triggers are anchored on broker-truth / chart-reader fields, not on
JPY/pip/multiplier literals (§3.5-compliant). Gate B (`operator_close_authorized:
true` in the receipt, or `QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell)
is still required on top of any Gate A trigger.

### NOT valid CLOSE triggers (do NOT propose CLOSE on these alone)

- `unrealized_pl_jpy < -per_trade_risk_budget_jpy` — under SL-free this is **advisory
  only**, not a hard close gate. -1,000 JPY ≈ 0.4% NAV is noise.
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

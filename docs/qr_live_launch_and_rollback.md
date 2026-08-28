# QuantRabbit live launch and rollback contract

Status: `NOT_ADMITTED`. This runbook is executable only after the risk
candidate is explicitly accepted, a sealed forward proof is present, and the
current repository contract separately admits the exact live version.

## Frozen launch inputs

- Commit and configuration hashes must be recorded before screening.
- The live lane accepts only the admitted commit/config pair. Paper and shadow
  use a different ledger plus `paper-*` campaign IDs.
- Manual or tagless positions and their TP/SL orders are always `NO_TOUCH`.
- Every candidate is screened against the whole account: current MCP, adverse
  25-pip stress MCP, margin buffer, USD/EUR/JPY factors, quote freshness,
  spread, bot inventory state, per-order loss, campaign drawdown, and maximum
  bot positions.

## Launch sequence

1. Read back the broker account, open trades, pending orders, and executable
   EUR/USD and USD/JPY quotes with GET only.
2. Verify the admitted commit/config/proof hashes and durable cooldown.
3. Calculate the bounded units. If the result is below the broker minimum,
   remain `SHADOW_ONLY` with broker mutation zero.
4. Freeze instrument, direction, units, maximum loss, stop drawdown, minimum
   margin buffer, proposal hash, and expiry in one receipt.
5. Submit exactly once through `LiveOrderGateway`.
6. Treat any transport exception or ambiguous response as outcome unknown. Do
   not resend. Reconcile by broker transaction, order, and position readback.
7. Start the isolated paper lane from the same market snapshot without sharing
   order, PnL, inventory, campaign, or strategy ledgers with live.

## Promotion and single-gateway boundary

- `quant_rabbit.fast_bot_promotion.build_fast_bot_promotion` is the only
  fast-bot promotion adapter. It is broker-free and emits no intent unless the
  exact signal, applied structured supervision receipt, deterministic sizing
  receipt, forward-admission proof, accepted risk contract, software hash,
  campaign state, cooldown, and strategy/position caps all bind.
- The promoted identity is a distinct `live-fb-*` campaign and `live-*`
  strategy. The source `paper-fb-*` campaign remains provenance only, so live
  and paper orders, PnL, inventory, and ledgers cannot share ownership IDs.
- The sealed sizing receipt must include the whole account, including observed
  manual/tagless exposure, and pass post-entry margin available, current/stress
  MCP, currency-factor concentration, campaign DD, account/quote age, and
  spread gates. It records zero manual/tagless mutations; the existing Gateway
  repeats its ordinary final fresh-snapshot `RiskEngine` checks before POST.
- LLM supervision may name regime, allowed strategies, a risk-budget cap, and
  a position cap. Any pair, side, units, order type, price, TP, or SL field in
  the LLM receipt blocks promotion.
- `dispatch_promotion_once` reserves the content-addressed promotion before it
  invokes the existing `LiveOrderGateway`. A gateway exception consumes that
  reservation as `UNKNOWN_GATEWAY_RESULT_NO_RETRY`; it cannot create a second
  broker attempt.
- `tools/run_inventory_paper_cycle.py` builds canonical OANDA order requests
  only. Its readback explicitly records `live_order_gateway_invocation_count=0`
  and must not describe those requests as Gateway staging.
- The current risk candidate has no accepted user-decision record and no
  forward-proof hash, so this boundary remains `NOT_ADMITTED` and emits no
  live intent from current artifacts.

## Degradation and rollback

- Gate failure with no bot inventory: `SHADOW_ONLY`.
- Gate failure with bot inventory: `FREEZE_NEW`, cancel only bot-owned pending
  entries, then `DRAINING` using profit/near-entry first, margin relief per
  realized loss plus cost second, and age/no-progress/factor concentration
  third. Reduced lots cannot be re-added.
- Hard deadline requires all remaining bot-owned units to be removed, followed
  by verified `FLAT`, `STOPPED`, and durable cooldown.
- A live anomaly never inherits permission from paper performance.
- Roll back only at a safe checkpoint or verified flat state by restoring the
  previously admitted commit/config hashes. Never reset or overwrite the dirty
  canonical main or live runtime worktrees.

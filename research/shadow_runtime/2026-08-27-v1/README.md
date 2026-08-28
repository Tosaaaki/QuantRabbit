# QuantRabbit OANDA LIVE zero-authority paper trader v4

This runtime reads the OANDA v20 LIVE pricing stream for `EUR_USD` and
`USD_JPY`, but it has no broker-order capability. Every trade is an internal,
append-only virtual fill. The historical R5 checkpoint remains frozen as
accounting-only and is not relabelled as a strategy signal.

Fixed boundaries:

- `strategy_status=RESEARCH_NOT_ADMITTED`
- `shadow_status=OBSERVATION_AUTHORIZED`
- `live_order_authority=false`
- `profit_proven=false`
- HTTP allowlist is OANDA `GET` only
- source code contains no `/orders`, `/trades`, or `/positions` endpoint
- `external_order_attempts=0` and `external_orders=0`

## Paper loop

1. The feed service records timestamped best BID/ASK and heartbeat rows.
2. Before the stream starts, a bounded OANDA LIVE-host `GET` reads completed
   BID/ASK M5 candles into a dedicated append-only warmup ledger. Gaps,
   overlaps, future or incomplete candles fail closed. Warmup rows may seed
   indicators only; they cannot create proposals, fills, or PnL. A complete
   prefix is reused unchanged after restart rather than refetched into the
   forward interval.
3. The bot seals only completed, continuous, attested LIVE M5 bars. A signal
   decision uses the arrival time of the later event that proves the prior bar
   complete; that event itself can never fill the new decision.
4. `M5_EMA_STATE_IMPULSE_INVENTORY_V1` emits a raw direction only when fast
   EMA state, three-bar impulse, and slow-EMA slope agree. Spread/cost is not
   an entry gate.
5. One content-addressed `signal_id` fans out to `RAW_SIGNAL`,
   `EXECUTABLE_BASE`, `ADVERSE_STRESS`, and `ACTUAL_LLM_INVENTORY`.
6. Virtual fill is the first strictly later, tradeable BBO with enough
   selected-side liquidity for all virtual units. The adverse arm consumes one
   content-addressed quote event as latency before adding 0.3 pip slippage per
   side; consumed latency survives restart and cannot be counted twice.
7. There is no individual price SL. TP is local ATR-scaled with an observed
   spread floor, and every position has a finite six-M5-bar maximum age.
8. Inventory, exits, realized PnL, and graceful-stop terminal MTM are written
   to hash-chained ledgers. EUR/USD JPY conversion is recorded only when an
   causal, tradeable, sufficiently liquid USD/JPY side exists within the fixed
   fifteen-second arrival-age bound. The conversion BBO identity, side, rate,
   timestamps, and age are persisted; otherwise JPY PnL remains null.
9. An inventory open/close creates a structured trigger. The actual LLM may
   choose only ADD/FREEZE/UNWIND/RESET and a bounded open-position cap for the
   `ACTUAL_LLM_INVENTORY` arm. The bot retains direction, order, fill, TP,
   accounting, and every hard guard. One `UNWIND` policy receipt closes exactly
   one oldest eligible LLM-arm position, then is durably consumed.

The four LaunchAgents are source-attested as one release. All four must be
unloaded before editing and reloaded together after tests; editing a running
release intentionally causes the watchdog to report `RUNTIME_HASH_MISMATCH`.
This candidate writes only to `runs/oanda_live_launchd_v4`; the prior
`runs/oanda_live_launchd_v1`, `runs/oanda_live_launchd_v2`, and
`runs/oanda_live_launchd_v3` evidence is never migrated or overwritten.
Completed OANDA M5 candles are used only as a causal feature warmup; decisions
and PnL begin with later completed LIVE-stream bars. Startup deterministically
reconciles split-ledger fill/open and close/PnL cut points, then replays durable
BBO rows through exits and fills without creating historical signals.

Focused verification:

```bash
python3 -m unittest -v \
  test_oanda_paper_execution.py \
  test_oanda_live_feed.py \
  test_oanda_live_llm_inventory.py \
  test_oanda_launchd_runtime.py
python3 oanda_launchd_manage.py preinstall
```

The runtime can accumulate forward shadow evidence, but neither a positive
fixture nor a short live sample proves profitability.

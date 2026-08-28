# QuantRabbit OANDA LIVE zero-authority paper trader v2

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
2. The bot seals only completed, continuous, attested M5 bars.
3. `M5_EMA_STATE_IMPULSE_INVENTORY_V1` emits a raw direction only when fast
   EMA state, three-bar impulse, and slow-EMA slope agree. Spread/cost is not
   an entry gate.
4. One content-addressed `signal_id` fans out to `RAW_SIGNAL`,
   `EXECUTABLE_BASE`, `ADVERSE_STRESS`, and `ACTUAL_LLM_INVENTORY`.
5. Virtual fill is the first eligible post-decision BBO; the adverse arm adds
   one quote-event latency and 0.3 pip slippage per side.
6. There is no individual price SL. TP is local ATR-scaled with an observed
   spread floor, and every position has a finite six-M5-bar maximum age.
7. Inventory, exits, realized PnL, and graceful-stop terminal MTM are written
   to hash-chained ledgers. EUR/USD JPY conversion is recorded only when an
   observed USD/JPY quote exists.
8. An inventory open/close creates a structured trigger. The actual LLM may
   choose only ADD/FREEZE/UNWIND/RESET and a bounded open-position cap for the
   `ACTUAL_LLM_INVENTORY` arm. The bot retains direction, order, fill, TP,
   accounting, and every hard guard.

The four LaunchAgents are source-attested as one release. All four must be
unloaded before editing and reloaded together after tests; editing a running
release intentionally causes the watchdog to report `RUNTIME_HASH_MISMATCH`.
This candidate writes only to `runs/oanda_live_launchd_v2`; the prior
`runs/oanda_live_launchd_v1` evidence is never migrated or overwritten.

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

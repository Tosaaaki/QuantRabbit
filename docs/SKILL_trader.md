# QuantRabbit Trader Runtime

This is the trader entry prompt. Keep it small. The branch prompts under
`docs/trader_prompts/` carry the task-specific instructions.

## Load Order

1. Read `docs/AGENT_CONTRACT.md`.
2. Read `docs/trader_prompts/00_router.md`.
3. Ask Python which branch is active:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli trader-prompt-route
```

4. Read every file in the returned `read_order`.
5. Use only the branch prompt that matches the current state.

## Shared Invariants

- Broker truth wins over memory, prose, and prior prompts.
- OANDA entry orders go only through `LiveOrderGateway`.
- OANDA position changes go only through `PositionProtectionGateway`.
- Do not print secrets.
- Do not call `QR_OPENAI_API_KEY`, `OPENAI_API_KEY`, or any model API path from QuantRabbit code.
- Do not invent JPY caps, pip distances, reward/risk multipliers, stale defaults, or extra risk gates.
- Missing required evidence is a blocker, not a value to guess.
- One final decision receipt selects action; specialist and strategy prompts are read-only observation.
- A blocked, rejected, monitor-only, or no-trade cycle must not be followed by a workaround send.
- Do not stop solely because a decision receipt was written recently or stale local state disagrees with refreshed broker truth. Use `trader-prompt-route`: unconsumed receipts go to verify; rejected, consumed, or broker-stale receipts go back to fresh decision work.

## Branches

| Branch | Read |
|---|---|
| Refresh broker truth and market context | `docs/trader_prompts/10_precheck_refresh.md` |
| Read the current market packet | `docs/trader_prompts/20_market_packet.md` |
| Flat / layerable account entry decision | `docs/trader_prompts/30_entry_decision.md` |
| Open exposure, pending order, protection decision | `docs/trader_prompts/35_position_management.md` |
| Receipt verification and gateway execution | `docs/trader_prompts/40_verify_execute.md` |
| Post-trade learning, missed-edge, gap work | `docs/trader_prompts/50_learning_gap.md` |
| Shared decision JSON schema | `docs/trader_prompts/90_decision_receipt_schema.md` |

## Runtime Skeleton

```bash
# 0. Export SL-free strategy env vars so generate-intents and risk validation
#    pick up the wider stop geometry, suppressed SL repair, and expanded
#    portfolio cap. These mirror `scripts/run-autotrade-live.sh` defaults so
#    direct CLI invocations and the wrapper produce identical receipts.
export QR_GEOMETRY_ATR_MULT="${QR_GEOMETRY_ATR_MULT:-5.0}"
export QR_GEOMETRY_SPREAD_FLOOR_MULT="${QR_GEOMETRY_SPREAD_FLOOR_MULT:-12.0}"
export QR_TRADER_DISABLE_SL_REPAIR="${QR_TRADER_DISABLE_SL_REPAIR:-1}"
export QR_MAX_PORTFOLIO_POSITIONS="${QR_MAX_PORTFOLIO_POSITIONS:-10}"
# NAV-pct sizing: each new position locks % of current NAV as margin so
# unit count auto-scales with equity (feedback_use_nav_percent.md). 30%
# per position lands ≈10000u for EUR_USD at NAV 227k — three concurrent
# positions reach ~90% margin utilization, just inside the 92% cap.
# Mirrors scripts/run-autotrade-live.sh so direct CLI invocations and the
# wrapper produce equivalent sizing.
export QR_TRADER_POSITION_NAV_PCT="${QR_TRADER_POSITION_NAV_PCT:-30}"
# Legacy fixed-units fallback used only when QR_TRADER_POSITION_NAV_PCT
# is unset. Do NOT remove — backstops smoke scripts that pin units. The
# NAV-pct path above takes precedence whenever set.
export QR_TRADER_BASE_UNITS="${QR_TRADER_BASE_UNITS:-3000}"

# 1. Route to the right prompt branch
PYTHONPATH=src python3 -m quant_rabbit.cli trader-prompt-route

# 2. Refresh evidence when routed there
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
# `--daily-risk-pct` sets the day's risk budget as % of starting NAV so the
# per-trade cap auto-scales with equity (feedback_use_nav_percent.md). 10%
# matches `target_return_pct` (campaign daily goal) and gives the basket
# validator enough room to hold the existing exposure plus add fresh lanes
# under SL-free (user directive 2026-05-11「市況読めばいいだけ」: the
# synthetic worst-case loss inside the validator is advisory; exits are
# market-derived, not loss-cap-derived).
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src python3 -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src python3 -m quant_rabbit.cli pair-charts --timeframes M1,M5,M15,M30,H1,H4,D --output data/pair_charts.json
PYTHONPATH=src python3 -m quant_rabbit.cli cross-asset-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli flow-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli currency-strength
PYTHONPATH=src python3 -m quant_rabbit.cli levels-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli economic-calendar
PYTHONPATH=src python3 -m quant_rabbit.cli cot-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli option-skew
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src python3 -m quant_rabbit.cli execution-ledger-sync
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage
PYTHONPATH=src python3 -m quant_rabbit.cli ai-attack-advice

# 3. Write data/codex_trader_decision_response.json from the active decision branch
# If broker refresh made an older receipt stale, overwrite it with one current receipt.
# If current trader-owned pending entries consume portfolio capacity, either keep
# that pending basket explicitly or name verified trader pending ids in
# cancel_order_ids when replacing them with current MARKET participation.
# If the action is CANCEL_PENDING, list only current trader-owned pending entry
# ids in cancel_order_ids; the gateway cycle cancels verified ids and sends no
# fresh entry in that same cycle.

# 4. Verify the receipt
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json

# 5. Run one gateway cycle only
# The gateway cycle syncs data/execution_ledger.db before and after broker work
# and records live_order / position_execution receipts.
./scripts/run-autotrade-live.sh \
  --reuse-market-artifacts \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send
```

## End Report

- Final action: `TRADE`, `WAIT`, `REQUEST_EVIDENCE`, `PROTECT`, `TIGHTEN_SL`, `CLOSE`, or `CANCEL_PENDING`.
- Sent flag: `true`, `false`, or dry-run.
- Selected lane id(s), if any.
- Daily target progress from `data/daily_target_state.json`.
- `gpt-trader-decision` result and issue codes.
- Gateway result and report paths under `docs/*_report.md`.
- Execution ledger DB/report: `data/execution_ledger.db`, `docs/execution_ledger_report.md`.

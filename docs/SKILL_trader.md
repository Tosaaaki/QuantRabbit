# QuantRabbit vNext Trader Playbook

You are **the trader**. The scheduled task picks which model executes you on a given cycle (Codex or Claude); the playbook is identical for both. QuantRabbit code is the broker-truth, risk, receipt, and gateway layer. Do not call any API-key model path from QuantRabbit.

## Contract

- Read `docs/AGENT_CONTRACT.md` before acting (single source of truth; `AGENTS.md` and `CLAUDE.md` are stubs to it). Pay particular attention to §3.5 (no thoughtless hardcodes / fallbacks) — every numeric input on the risk path must be market-derived.
- Use OANDA only through the vNext CLI and gateways.
- Do not print secrets.
- Do not use VM/deploy scripts.
- Do not run a second send or workaround after a blocked, monitor-only, rejected, or no-trade cycle.
- The 10% daily target is an operating KPI, not a guaranteed return and not permission to bypass risk gates.

## Runtime

### 1. Refresh broker truth + market context

The trader **must** look at live market conditions before deciding. ATR, regime, spread, equity, and daily progress all enter the decision; none of them are inferred from prose or memory.

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli pair-charts --output data/pair_charts.json
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage
```

If strategy artifacts are missing or stale, refresh evidence first:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance "$(jq -r .start_balance_jpy data/daily_target_state.json)"
```

### 2. Read what the market is doing right now

Before writing any decision, open and actually read:

- `data/daily_target_state.json` — current equity, today's target, **daily_risk_budget_jpy** (per-trade JPY cap, equity-derived; no fixed literal). Use this cap when sizing; it flows into every intent's `metadata.max_loss_jpy`.
- `data/pair_charts.json` (and `docs/pair_charts_report.md`) — per-pair regime + M5/M15/H1 indicators (ATR pips, RSI, ADX, Bollinger, Ichimoku, VWAP, Donchian, MACD, Stoch, ROC, CCI). The high-score pairs are where indicator agreement lines up; the regime tag tells you which methods (TREND_CONTINUATION / RANGE_ROTATION / BREAKOUT_FAILURE) match.
- `data/order_intents.json` (and `docs/order_intents_report.md`) — pre-validated lane intents with current geometry (ATR-derived SL, equity-derived units). `LIVE_READY` lanes have no risk or strategy blockers; `DRY_RUN_BLOCKED` lanes carry their reason.
- `data/market_story_profile.json` — current narrative pressure (intervention risk, event risk, JPY-cross conditions, etc.).
- `data/broker_snapshot.json` — open positions, pending orders, ages, spreads.

The decision must reference these inputs explicitly. Do not invent ATR, regime, or equity numbers from prose.

### 3. Decide

Write `data/codex_trader_decision_response.json` (the filename is kept for compatibility regardless of which model wrote it):

```json
{
  "action": "TRADE",
  "selected_lane_id": "desk:PAIR:SIDE:METHOD",
  "cancel_order_ids": [],
  "confidence": "HIGH",
  "thesis": "...",
  "method": "BREAKOUT_FAILURE",
  "narrative": "...",
  "chart_story": "ATR(M5)=N.Np, regime=TREND_UP, ADX=NN, ...",
  "invalidation": "...",
  "rejected_alternatives": ["..."],
  "risk_notes": ["units bounded by equity cap NNNN JPY", "..."],
  "evidence_refs": ["broker:snapshot", "target:daily", "intent:<lane_id>", "campaign:<lane_id>", "strategy:<pair>:<side>", "story:<pair>", "chart:<pair>:M5"],
  "operator_summary": "..."
}
```

Action values: `TRADE`, `WAIT`, `REQUEST_EVIDENCE`, `PROTECT`, `TIGHTEN_SL`, `CLOSE`, `CANCEL_PENDING`. For `CANCEL_PENDING` put the OANDA order ids in `cancel_order_ids`. For `TRADE` choose only a current `LIVE_READY` lane that can survive deterministic prefiltering.

`chart_story` and `risk_notes` MUST cite numbers from `pair_charts.json` and `daily_target_state.json` — not hand-waving. If you cannot cite the numbers, the decision is `WAIT` or `REQUEST_EVIDENCE`.

### 4. Verify

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json
```

### 5. Run one gateway cycle

```bash
./scripts/run-autotrade-live.sh \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send
```

Live send still requires `QR_LIVE_ENABLED=1` and the gates in `AGENT_CONTRACT.md §9`. Without them the cycle stays dry-run.

## Report at end

- Final status (TRADE / WAIT / PROTECT / TIGHTEN_SL / CLOSE / CANCEL_PENDING / REQUEST_EVIDENCE)
- Sent flag (true / false / dry-run)
- Selected lane id
- Daily target progress (% of target, current vs starting equity)
- gpt-trader-decision verification result
- Blockers (if any) — including any `MISSING_*` issues that surfaced
- Report paths under `docs/*_report.md`

## Anti-patterns the contract forbids

- Inventing JPY caps, pip distances, or reward/risk multipliers from memory.
- Choosing WAIT without citing which input was missing or which gate fired.
- Submitting a `TRADE` without checking `pair_charts.json` regime + ATR for that pair.
- Reusing yesterday's `daily_target_state.json` past a JST campaign-day rollover (the ledger auto-rolls; don't bypass it).
- Sending again after a blocked / rejected / no-trade outcome to "force" a fill.

# QuantRabbit vNext Trader Playbook

Codex automation is the discretionary GPT trader. QuantRabbit code is the broker-truth, risk, receipt, and gateway layer. Do not call any API-key model path from QuantRabbit.

## Contract

- Read `docs/AGENT_CONTRACT.md` before acting (single source of truth; `AGENTS.md` and `CLAUDE.md` are stubs to it).
- Use OANDA only through the vNext CLI and gateways.
- Do not print secrets.
- Do not use VM/deploy scripts.
- Do not run a second send or workaround after a blocked, monitor-only, rejected, or no-trade cycle.
- The 10% daily target is an operating KPI, not a guaranteed return and not permission to bypass risk gates.

## Runtime

1. Refresh broker truth:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage
```

If strategy artifacts are missing or stale, refresh evidence first:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance "$(jq -r .start_balance_jpy data/daily_target_state.json)"
PYTHONPATH=src python3 -m quant_rabbit.cli pair-charts --output data/pair_charts.json
```

Daily start balance is sourced from the latest broker snapshot via `daily-target-state` (OANDA `/v3/accounts/{id}/summary` → `balance`). Do not hardcode a JPY figure here. If `data/daily_target_state.json` is missing or you need to bootstrap, run `broker-snapshot` then `daily-target-state --snapshot data/broker_snapshot.json` (without `--start-balance`) before `plan-campaign` so the value is auto-derived.

`pair-charts` writes per-pair indicator scores (M5/M15/H1: ATR/EMA/RSI/ADX/Bollinger/Ichimoku/VWAP/Donchian/MACD/Stoch/ROC/CCI) and a regime tag (TREND_UP/DOWN, RANGE, IMPULSE_UP/DOWN, FAILURE_RISK, UNCLEAR) to `data/pair_charts.json` and a sortable score table to `docs/pair_charts_report.md`. The trader reads this to pick which pair to attack — high-score pairs are where indicator agreement lines up.

2. Decide as Codex. Write `data/codex_trader_decision_response.json` with:

```json
{
  "action": "TRADE",
  "selected_lane_id": "desk:PAIR:SIDE:METHOD",
  "cancel_order_ids": [],
  "confidence": "HIGH",
  "thesis": "...",
  "method": "BREAKOUT_FAILURE",
  "narrative": "...",
  "chart_story": "...",
  "invalidation": "...",
  "rejected_alternatives": ["..."],
  "risk_notes": ["..."],
  "evidence_refs": ["broker:snapshot", "target:daily", "intent:<lane_id>", "campaign:<lane_id>", "strategy:<pair>:<side>", "story:<pair>"],
  "operator_summary": "..."
}
```

Use `WAIT`, `REQUEST_EVIDENCE`, `PROTECT`, `TIGHTEN_SL`, `CLOSE`, or `CANCEL_PENDING` when the trade is not clean. For `CANCEL_PENDING`, put the current pending-entry OANDA order ids in `cancel_order_ids`. For `TRADE`, choose only a current `LIVE_READY` lane that can survive deterministic prefiltering.

3. Verify the Codex decision:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli gpt-trader-decision \
  --snapshot data/broker_snapshot.json \
  --decision-response data/codex_trader_decision_response.json
```

4. Run one gateway cycle:

```bash
./scripts/run-autotrade-live.sh \
  --use-gpt-trader \
  --gpt-decision-response data/codex_trader_decision_response.json \
  --send
```

Report final status, sent flag, selected lane, target progress, GPT verification status, blockers, and report paths.

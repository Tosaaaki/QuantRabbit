# Trader Prompt Router

## Common Core

- `docs/AGENT_CONTRACT.md` is the single source of truth.
- This router chooses which branch prompt to read next; it does not create trading permission.
- Branch prompts never override the common contract.
- Python route command:

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli trader-prompt-route
```

## State Classification

| State | Branch |
|---|---|
| Required broker / market / intent artifact is missing | `refresh_market_context` |
| Trader-owned position is missing TP or SL | `position_management` |
| Existing open position has a deterministic `tp-rebalance` adjustment | `position_management` |
| Current unconsumed decision receipt exists and still matches current broker / intent artifacts | `verify_execute` |
| Daily target is open and current `LIVE_READY` lanes exist | `entry_decision` |
| Daily target is open but no `LIVE_READY` lane exists | `learning_gap` |
| Target reached or exposure needs review | `position_management` |

## Required Artifact Set

- `data/broker_snapshot.json`
- `data/daily_target_state.json`
- `data/pair_charts.json`
- `data/cross_asset_snapshot.json`
- `data/flow_snapshot.json`
- `data/currency_strength.json`
- `data/levels_snapshot.json`
- `data/economic_calendar.json`
- `data/cot_snapshot.json`
- `data/option_skew_snapshot.json`
- `data/order_intents.json`
- `data/ai_attack_advice.json`
- `data/learning_audit.json`

## Branch Discipline

- Refresh branch may write latest reports and runtime artifacts.
- Decision branches write exactly one receipt: `data/codex_trader_decision_response.json`.
- Verify branch runs `gpt-trader-decision` and then at most one gateway cycle.
- A non-executable accepted WAIT is not final while `trader-prompt-route` reports
  `TP rebalance required`; return to `position_management` and run the TP sidecar.
- Rejected, non-executable, already-consumed, or broker-stale receipts are not stop conditions. Route back to the active decision branch and write one current receipt from refreshed broker truth.
- Learning branch is read-only unless it writes learning / gap artifacts through the CLI.
- No branch may stage or send an order directly.

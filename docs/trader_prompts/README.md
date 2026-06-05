# Trader Prompt Branches

## Purpose

- Keep `docs/SKILL_trader.md` as a small entry prompt.
- Put state-specific trading work in branch prompts.
- Let Python select the branch from current broker truth and runtime artifacts.

## Router

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-prompt-route
```

The command returns:

- `branch` — the active prompt branch.
- `reasons` — state facts that caused the route.
- `read_order` — markdown files to read before acting.

Use `--include-content` only when a caller needs the markdown body in JSON.

## Branch Files

| File | Role |
|---|---|
| `00_router.md` | Entry router and state classification |
| `10_precheck_refresh.md` | Clean-tree precheck and market-context refresh |
| `20_market_packet.md` | What to read from each JSON artifact |
| `30_entry_decision.md` | Flat / layerable entry decision |
| `35_position_management.md` | Protection, tighten, close, and pending-cancel decisions |
| `40_verify_execute.md` | Receipt verification and gateway execution |
| `50_learning_gap.md` | Learning, replay, missed-edge, and gap reports |
| `90_decision_receipt_schema.md` | Shared decision JSON schema |

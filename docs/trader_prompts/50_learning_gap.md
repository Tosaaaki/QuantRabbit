# Learning And Gap Work

## Use When

- Target is open but no current `LIVE_READY` lane exists.
- A trade outcome needs receipt-backed learning.
- The daily target was missed.
- A verifier or gateway rejection exposes a strategy, evidence, or risk bug.

## Evidence Refresh

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories
PYTHONPATH=src python3 -m quant_rabbit.cli promote-receipts
```

## Replay And Certification

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli replay-backtest --start-balance "$(jq -r .start_balance_jpy data/daily_target_state.json)"
PYTHONPATH=src python3 -m quant_rabbit.cli ai-test-bot-backtest --start-balance "$(jq -r .start_balance_jpy data/daily_target_state.json)"
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage
PYTHONPATH=src python3 -m quant_rabbit.cli ai-attack-advice
PYTHONPATH=src python3 -m quant_rabbit.cli learning-audit
PYTHONPATH=src python3 -m quant_rabbit.cli certify-dry-run
```

## Post-Trade Learning

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli learn-post-trade --outcome outcome.json
PYTHONPATH=src python3 -m quant_rabbit.cli learning-audit
```

## Gap Report Content

- Missed setup, blocked trade, market absence, risk rejection, execution cost, timing error, or strategy weakness.
- The exact deterministic issue code or missing artifact.
- Whether the fix belongs in Python risk/strategy code, prompt branch wording, evidence mining, or test coverage.
- A failing regression test for known legacy failure modes before behavior changes.

## Non-Negotiables

- Learning memory is advisory.
- Learning that changes `ai_attack_advice` lane ranking must be audited by
  `learning-audit`; blocked or missing audit makes that learning-influenced
  lane non-executable for live.
- Negative old evidence cannot override current broker-truth risk geometry once a lane is `LIVE_READY`.
- `BLOCK_UNTIL_NEW_EVIDENCE` is never auto-promoted.

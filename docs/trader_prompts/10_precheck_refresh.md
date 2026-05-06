# Precheck And Refresh

## Precheck

- Run before any report-writing command.
- `git status --short` may contain only tracked `docs/*_report.md` runtime drift from the previous cycle.
- Source, config, data, decision, or prompt diffs block the scheduled trader cycle until resolved.
- Confirm exactly one trader scheduled task is enabled.
- Do not run report-writing refresh commands from a dirty development tree.

## Refresh Evidence

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli pair-charts --timeframes M1,M5,M15,M30,H1,H4,D --output data/pair_charts.json
PYTHONPATH=src python3 -m quant_rabbit.cli cross-asset-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli flow-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli currency-strength
PYTHONPATH=src python3 -m quant_rabbit.cli levels-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli economic-calendar
PYTHONPATH=src python3 -m quant_rabbit.cli cot-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli option-skew
```

## Reprice Intents

Context fetches can outlive the quote freshness window. Refresh broker truth again immediately before intent pricing.

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli optimize-coverage
PYTHONPATH=src python3 -m quant_rabbit.cli ai-attack-advice
```

## Refresh Strategy Evidence

Run only when strategy artifacts are missing, stale, or a branch explicitly routes to evidence repair.

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance "$(jq -r .start_balance_jpy data/daily_target_state.json)"
```

## Stop Conditions

- Any `MISSING_*` artifact that is required for the active branch and cannot be refreshed.
- Dirty source/config/data/decision files before report writes.
- More than one trader scheduler enabled.
- Missing OANDA read credentials for broker-truth refresh.

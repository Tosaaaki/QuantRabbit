# Precheck And Refresh

## Precheck

- Run before any report-writing command.
- `git status --short` may contain only tracked `docs/*_report.md` runtime drift from the previous cycle.
- Source, config, data, decision, or prompt diffs block the scheduled trader cycle until resolved.
- Confirm exactly one trader scheduled task is enabled.
- Do not run report-writing refresh commands from a dirty development tree.
- Do not stop because refreshed broker truth disagrees with stale journal, stale local state, or an older decision receipt. Broker truth wins; re-route and rewrite the receipt when needed.
- A locally remembered pending order that is absent from the refreshed OANDA snapshot is stale local memory, not a send blocker.

## Refresh Evidence

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
PYTHONPATH=src python3 -m quant_rabbit.cli pair-charts --timeframes M1,M5,M15,M30,H1,H4,D --output data/pair_charts.json
PYTHONPATH=src python3 -m quant_rabbit.cli cross-asset-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli flow-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli currency-strength
PYTHONPATH=src python3 -m quant_rabbit.cli levels-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli economic-calendar
PYTHONPATH=src python3 -m quant_rabbit.cli cot-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli option-skew
PYTHONPATH=src python3 -m quant_rabbit.cli news-snapshot
```

`news-snapshot` is mandatory — it refreshes `logs/news_digest.md` and
`logs/news_flow_log.md`, the macro-narrative inputs that
`strategy/market_story.py` reads for CPI prints, central-bank tone,
intervention risk, and risk-on/off shifts. Skipping it leaves the cycle
trading against an obsolete narrative. 2026-05-13 incident: the digest
was 7 days stale (last refresh 2026-05-06) while live US CPI
reaccelerated to 3.8% and DXY rallied; the cycle entered EUR/USD LONG
and GBP/USD LONG against the actual session's USD strength.

## Reprice Intents

Context fetches can outlive the quote freshness window. Refresh broker truth again immediately before intent pricing.

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli daily-target-state --snapshot data/broker_snapshot.json --daily-risk-pct 10 --target-trades-per-day 10
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
- Active OANDA broker-truth exposure/risk gates that the current gateway cannot reconcile.

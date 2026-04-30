# QuantRabbit vNext

This is the clean rebuild workspace.

The legacy system was archived before this repository was initialized. Do not copy old runtime behavior forward by default. Pull code back only when it passes the new execution contract.

See `ARCHIVE_POINTER.md` for the legacy snapshot location and `SYSTEM_REBUILD_CHARTER.md` for the rebuild rules.

## Current vNext Commands

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance 222781
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli risk-dry-run --intent intent.json --snapshot snapshot.json
```

`risk-dry-run` reads `data/strategy_profile.json` when present, so mined legacy evidence is enforced alongside current risk geometry. Order intents also carry `market_context` (`regime`, `narrative`, `chart_story`, `method`, `invalidation`) so the system can reject method-vs-regime mismatches before live use.

`plan-campaign` builds the multi-desk daily 10% campaign: trend, range, failure, event-risk, and position-management desks all propose or veto lanes, then the Portfolio Director reports what can become a receipt and what is still missing.

`generate-intents` turns campaign lanes into priced dry-run order intents when a read-only broker snapshot is available. Without a snapshot it reports `NEEDS_BROKER_SNAPSHOT`; this is a hard stop, not a prompt problem.

Live execution is not implemented in this rebuild yet. That is intentional: strategy evidence, market story, and campaign role must become a risk-checked order intent before any OANDA write path exists.

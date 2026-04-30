# QuantRabbit vNext

This is the clean rebuild workspace.

The legacy system was archived before this repository was initialized. Do not copy old runtime behavior forward by default. Pull code back only when it passes the new execution contract.

See `ARCHIVE_POINTER.md` for the legacy snapshot location and `SYSTEM_REBUILD_CHARTER.md` for the rebuild rules.

## Current vNext Commands

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot
PYTHONPATH=src python3 -m quant_rabbit.cli risk-dry-run --intent intent.json --snapshot snapshot.json
```

`risk-dry-run` reads `data/strategy_profile.json` when present, so mined legacy evidence is enforced alongside current risk geometry. Live execution is not implemented in this rebuild yet. That is intentional: strategy evidence must become a risk-checked order intent before any OANDA write path exists.

# QuantRabbit vNext

This is the clean rebuild workspace.

The legacy system was archived before this repository was initialized. Do not copy old runtime behavior forward by default. Pull code back only when it passes the new execution contract.

Previous system location: `/Users/tossaki/App/QuantRabbit_archives/QuantRabbit_legacy_20260430T151527Z`.

See `ARCHIVE_POINTER.md` for the legacy snapshot location and `SYSTEM_REBUILD_CHARTER.md` for the rebuild rules.

## Current vNext Commands

```bash
PYTHONPATH=src python3 -m quant_rabbit.cli import-legacy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-strategy
PYTHONPATH=src python3 -m quant_rabbit.cli mine-market-stories
PYTHONPATH=src python3 -m quant_rabbit.cli plan-campaign --start-balance 222781
PYTHONPATH=src python3 -m quant_rabbit.cli broker-snapshot --output data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --snapshot data/broker_snapshot.json
PYTHONPATH=src python3 -m quant_rabbit.cli promote-receipts
PYTHONPATH=src python3 -m quant_rabbit.cli stage-live-order --lane-id 'failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE'
QR_LIVE_ENABLED=1 PYTHONPATH=src python3 -m quant_rabbit.cli autotrade-cycle --send
PYTHONPATH=src python3 -m quant_rabbit.cli risk-dry-run --intent intent.json --snapshot snapshot.json
```

`risk-dry-run` reads `data/strategy_profile.json` when present, so mined legacy evidence is enforced alongside current risk geometry. Order intents also carry `market_context` (`regime`, `narrative`, `chart_story`, `method`, `invalidation`) so the system can reject method-vs-regime mismatches before live use.

`plan-campaign` builds the multi-desk daily 10% campaign: trend, range, failure, event-risk, and position-management desks all propose or veto lanes, then the Portfolio Director reports what can become a receipt and what is still missing.

`generate-intents` turns campaign lanes into priced dry-run order intents when a read-only broker snapshot is available. Without a snapshot it reports `NEEDS_BROKER_SNAPSHOT`; this is a hard stop, not a prompt problem.

`promote-receipts` feeds successful dry-run receipts back into `data/strategy_profile.json`. It can reopen `RISK_REPAIR_CANDIDATE` only when the current receipt passes risk geometry, and can reopen `MINE_MISSED_EDGE` only when the receipt is a LIMIT or STOP-ENTRY trigger. It never auto-promotes `BLOCK_UNTIL_NEW_EVIDENCE`.

`stage-live-order` turns one live-ready intent into an OANDA order request after fetching fresh broker truth and rerunning live validation. It stages by default. A real send requires `--send --confirm-live`, `QR_LIVE_ENABLED=1`, and an explicit `--lane-id`.

`autotrade-cycle` is the automation entrypoint. It fetches fresh broker truth first; if any position or pending order exists, it manages existing exposure before considering fresh risk and sends no new entry. If a pending order came from a lane the current TraderBrain now vetoes, the cycle can cancel that pending order before waiting for the next cycle. If a position is open, PositionManager writes the TP/SL, remaining risk/reward, same-vs-opposite thesis score, and protection/exit-review action. The protection gateway can then close a contradicted trader-owned trade, create missing TP/SL, or tighten SL to break-even/profit protection; it refuses to widen SL or move existing TP. If flat, it regenerates intents, asks TraderBrain to compare the live-ready lanes against mined history, market story, campaign role, narrative risk, and current broker state, then sends only the selected lane when `QR_LIVE_ENABLED=1` and `--send` are present.

Live execution is guarded behind this gateway: strategy evidence, market story, campaign role, fresh broker truth, risk geometry, and explicit live enablement must all pass before any OANDA write occurs.

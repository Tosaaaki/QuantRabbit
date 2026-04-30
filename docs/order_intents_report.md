# Order Intents Report

- Generated at UTC: `2026-04-30T16:13:13.822961+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `none`
- Results: `12`

## Status Counts

- `NEEDS_BROKER_SNAPSHOT`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`NEEDS_BROKER_SNAPSHOT`
  - note: Run broker-snapshot to a JSON file, then rerun generate-intents with --snapshot.
  - live blocker: broker snapshot is required to price entry/TP/SL

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

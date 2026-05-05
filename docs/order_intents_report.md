# Order Intents Report

- Generated at UTC: `2026-05-05T03:22:14.528542+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `data/broker_snapshot.json`
- Results: `15`

## Status Counts

- `DRY_RUN_BLOCKED`: `3`
- `DRY_RUN_PASSED`: `3`
- `LIVE_READY`: `9`

## Candidates

- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.16852 tp=1.16715 sl=1.169
  - risk metrics: risk=`983.1 JPY` reward=`2805.8 JPY` rr=`2.85` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.169 tp=1.16763 sl=1.16948
  - risk metrics: risk=`983.1 JPY` reward=`2805.8 JPY` rr=`2.85` spread=`0.8pip`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.16852 tp=1.16715 sl=1.169
  - risk metrics: risk=`983.1 JPY` reward=`2805.8 JPY` rr=`2.85` spread=`0.8pip`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.62 tp=112.988 sl=112.524
  - risk metrics: risk=`960.0 JPY` reward=`3680.0 JPY` rr=`3.83` spread=`1.6pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_PASSED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=9000 entry=183.818 tp=183.98 sl=183.71
  - risk metrics: risk=`972.0 JPY` reward=`1458.0 JPY` rr=`1.50` spread=`1.8pip`
  - strategy WARN: STRATEGY_TRIGGER_RECEIPT_REQUIRED EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
  - live blocker: EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.169 tp=1.17008 sl=1.16852
  - risk metrics: risk=`983.1 JPY` reward=`2211.9 JPY` rr=`2.25` spread=`0.8pip`
  - risk BLOCK: OPPOSING_POSITION_EXISTS fresh EUR_USD LONG entry opposes protected EUR_USD SHORT id=470188; use position management instead of a new entry order
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35301 tp=1.35418 sl=1.35223
  - risk metrics: risk=`983.1 JPY` reward=`1474.6 JPY` rr=`1.50` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.54 tp=112.908 sl=112.444
  - risk metrics: risk=`960.0 JPY` reward=`3680.0 JPY` rr=`3.83` spread=`1.6pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_PASSED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=9000 entry=183.728 tp=183.89 sl=183.62
  - risk metrics: risk=`972.0 JPY` reward=`1458.0 JPY` rr=`1.50` spread=`1.8pip`
  - strategy WARN: STRATEGY_TRIGGER_RECEIPT_REQUIRED EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
  - live blocker: EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.16852 tp=1.1696 sl=1.16804
  - risk metrics: risk=`983.1 JPY` reward=`2211.9 JPY` rr=`2.25` spread=`0.8pip`
  - risk BLOCK: OPPOSING_POSITION_EXISTS fresh EUR_USD LONG entry opposes protected EUR_USD SHORT id=470188; use position management instead of a new entry order
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35236 tp=1.35353 sl=1.35158
  - risk metrics: risk=`983.1 JPY` reward=`1474.6 JPY` rr=`1.50` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.62 tp=112.988 sl=112.524
  - risk metrics: risk=`960.0 JPY` reward=`3680.0 JPY` rr=`3.83` spread=`1.6pip`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_PASSED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=9000 entry=183.818 tp=183.98 sl=183.71
  - risk metrics: risk=`972.0 JPY` reward=`1458.0 JPY` rr=`1.50` spread=`1.8pip`
  - strategy WARN: STRATEGY_TRIGGER_RECEIPT_REQUIRED EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
  - live blocker: EUR_JPY LONG requires trigger/pending-entry receipts before live use: missed seats paid more often than captured; build trigger/pending-entry receipts before live execution; every receipt must be risk-resized under the 1051 JPY cap
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.169 tp=1.17008 sl=1.16852
  - risk metrics: risk=`983.1 JPY` reward=`2211.9 JPY` rr=`2.25` spread=`0.8pip`
  - risk BLOCK: OPPOSING_POSITION_EXISTS fresh EUR_USD LONG entry opposes protected EUR_USD SHORT id=470188; use position management instead of a new entry order
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35301 tp=1.35418 sl=1.35223
  - risk metrics: risk=`983.1 JPY` reward=`1474.6 JPY` rr=`1.50` spread=`1.3pip`

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

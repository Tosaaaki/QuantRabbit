# Order Intents Report

- Generated at UTC: `2026-05-04T20:45:54.336190+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `data/broker_snapshot.json`
- Results: `15`

## Status Counts

- `DRY_RUN_BLOCKED`: `15`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=8000 entry=112.716 tp=113.676 sl=112.596
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`2.0pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 40.5s > 20s
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=6000 entry=183.889 tp=184.193 sl=183.733
  - risk metrics: risk=`936.0 JPY` reward=`1824.0 JPY` rr=`1.95` spread=`2.6pip`
  - risk BLOCK: STALE_QUOTE EUR_JPY quote is stale: 39.4s > 20s
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.6pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.16938 tp=1.17166 sl=1.1689
  - risk metrics: risk=`981.1 JPY` reward=`4660.0 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 49.6s > 20s
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 39.8s > 20s
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.1689 tp=1.16602 sl=1.16938
  - risk metrics: risk=`981.1 JPY` reward=`5886.3 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 49.6s > 20s
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 39.8s > 20s
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35347 tp=1.3552 sl=1.35269
  - risk metrics: risk=`981.1 JPY` reward=`2175.9 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: STALE_QUOTE GBP_USD quote is stale: 39.5s > 20s
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 39.8s > 20s
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=8000 entry=112.616 tp=113.576 sl=112.496
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`2.0pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 40.5s > 20s
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=6000 entry=183.759 tp=184.063 sl=183.603
  - risk metrics: risk=`936.0 JPY` reward=`1824.0 JPY` rr=`1.95` spread=`2.6pip`
  - risk BLOCK: STALE_QUOTE EUR_JPY quote is stale: 39.4s > 20s
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.6pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.1689 tp=1.17118 sl=1.16842
  - risk metrics: risk=`981.1 JPY` reward=`4660.0 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 49.6s > 20s
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 39.8s > 20s
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.16938 tp=1.1665 sl=1.16986
  - risk metrics: risk=`981.1 JPY` reward=`5886.3 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 49.6s > 20s
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 39.8s > 20s
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35282 tp=1.35455 sl=1.35204
  - risk metrics: risk=`981.1 JPY` reward=`2175.9 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: STALE_QUOTE GBP_USD quote is stale: 39.5s > 20s
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 39.8s > 20s
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=8000 entry=112.716 tp=113.676 sl=112.596
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`2.0pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 40.5s > 20s
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=6000 entry=183.889 tp=184.193 sl=183.733
  - risk metrics: risk=`936.0 JPY` reward=`1824.0 JPY` rr=`1.95` spread=`2.6pip`
  - risk BLOCK: STALE_QUOTE EUR_JPY quote is stale: 39.4s > 20s
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.6pip exceeds 2.5x normal 0.8pip
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.16938 tp=1.17166 sl=1.1689
  - risk metrics: risk=`981.1 JPY` reward=`4660.0 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 49.6s > 20s
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 39.8s > 20s
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.1689 tp=1.16602 sl=1.16938
  - risk metrics: risk=`981.1 JPY` reward=`5886.3 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 49.6s > 20s
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 39.8s > 20s
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35347 tp=1.3552 sl=1.35269
  - risk metrics: risk=`981.1 JPY` reward=`2175.9 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: STALE_QUOTE GBP_USD quote is stale: 39.5s > 20s
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 39.8s > 20s

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

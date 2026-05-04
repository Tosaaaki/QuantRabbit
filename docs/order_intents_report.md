# Order Intents Report

- Generated at UTC: `2026-05-04T21:46:55.106281+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=2000 entry=112.869 tp=116.517 sl=112.413
  - risk metrics: risk=`912.0 JPY` reward=`7296.0 JPY` rr=`8.00` spread=`7.6pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 7.6pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=2000 entry=183.944 tp=184.74 sl=183.536
  - risk metrics: risk=`816.0 JPY` reward=`1592.0 JPY` rr=`1.95` spread=`6.8pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 6.8pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=11000 entry=1.16928 tp=1.17213 sl=1.16868
  - risk metrics: risk=`1037.6 JPY` reward=`4928.7 JPY` rr=`4.75` spread=`1.0pip`
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 24.8s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 1.7pip exceeds 2.5x normal 0.4pip
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=11000 entry=1.16878 tp=1.16519 sl=1.16938
  - risk metrics: risk=`1037.6 JPY` reward=`6208.4 JPY` rr=`5.98` spread=`1.0pip`
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 24.8s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 1.7pip exceeds 2.5x normal 0.4pip
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=2000 entry=1.35404 tp=1.36083 sl=1.35098
  - risk metrics: risk=`962.1 JPY` reward=`2135.0 JPY` rr=`2.22` spread=`5.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE GBP_USD spread 5.1pip exceeds 2.5x normal 0.9pip
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 24.8s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 1.7pip exceeds 2.5x normal 0.4pip
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=2000 entry=112.489 tp=116.137 sl=112.033
  - risk metrics: risk=`912.0 JPY` reward=`7296.0 JPY` rr=`8.00` spread=`7.6pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 7.6pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=2000 entry=183.604 tp=184.4 sl=183.196
  - risk metrics: risk=`816.0 JPY` reward=`1592.0 JPY` rr=`1.95` spread=`6.8pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 6.8pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=11000 entry=1.16878 tp=1.17163 sl=1.16818
  - risk metrics: risk=`1037.6 JPY` reward=`4928.7 JPY` rr=`4.75` spread=`1.0pip`
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 24.8s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 1.7pip exceeds 2.5x normal 0.4pip
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=11000 entry=1.16928 tp=1.16569 sl=1.16988
  - risk metrics: risk=`1037.6 JPY` reward=`6208.4 JPY` rr=`5.98` spread=`1.0pip`
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 24.8s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 1.7pip exceeds 2.5x normal 0.4pip
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=2000 entry=1.35149 tp=1.35828 sl=1.34843
  - risk metrics: risk=`962.1 JPY` reward=`2135.0 JPY` rr=`2.22` spread=`5.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE GBP_USD spread 5.1pip exceeds 2.5x normal 0.9pip
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 24.8s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 1.7pip exceeds 2.5x normal 0.4pip
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=2000 entry=112.869 tp=116.517 sl=112.413
  - risk metrics: risk=`912.0 JPY` reward=`7296.0 JPY` rr=`8.00` spread=`7.6pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 7.6pip exceeds 2.5x normal 0.8pip
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=2000 entry=183.944 tp=184.74 sl=183.536
  - risk metrics: risk=`816.0 JPY` reward=`1592.0 JPY` rr=`1.95` spread=`6.8pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 6.8pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

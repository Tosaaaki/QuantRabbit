# Order Intents Report

- Generated at UTC: `2026-05-04T21:21:39.535831+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=1000 entry=112.917 tp=117.285 sl=112.371
  - risk metrics: risk=`546.0 JPY` reward=`4368.0 JPY` rr=`8.00` spread=`9.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 9.1pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=1000 entry=184.122 tp=185.631 sl=183.348
  - risk metrics: risk=`774.0 JPY` reward=`1509.0 JPY` rr=`1.95` spread=`12.9pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 12.9pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=2000 entry=1.17042 tp=1.18439 sl=1.16748
  - risk metrics: risk=`924.6 JPY` reward=`4393.2 JPY` rr=`4.75` spread=`4.9pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 4.9pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.1pip exceeds 2.5x normal 0.4pip
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=2000 entry=1.16797 tp=1.15036 sl=1.17091
  - risk metrics: risk=`924.6 JPY` reward=`5537.9 JPY` rr=`5.99` spread=`4.9pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 4.9pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.1pip exceeds 2.5x normal 0.4pip
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=1000 entry=1.35542 tp=1.36967 sl=1.349
  - risk metrics: risk=`1009.5 JPY` reward=`2240.6 JPY` rr=`2.22` spread=`10.7pip`
  - risk BLOCK: SPREAD_TOO_WIDE GBP_USD spread 10.7pip exceeds 2.5x normal 0.9pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.1pip exceeds 2.5x normal 0.4pip
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=1000 entry=112.462 tp=116.83 sl=111.916
  - risk metrics: risk=`546.0 JPY` reward=`4368.0 JPY` rr=`8.00` spread=`9.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 9.1pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=1000 entry=183.477 tp=184.986 sl=182.703
  - risk metrics: risk=`774.0 JPY` reward=`1509.0 JPY` rr=`1.95` spread=`12.9pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 12.9pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=2000 entry=1.16797 tp=1.18194 sl=1.16503
  - risk metrics: risk=`924.6 JPY` reward=`4393.2 JPY` rr=`4.75` spread=`4.9pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 4.9pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.1pip exceeds 2.5x normal 0.4pip
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=2000 entry=1.17042 tp=1.15281 sl=1.17336
  - risk metrics: risk=`924.6 JPY` reward=`5537.9 JPY` rr=`5.99` spread=`4.9pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 4.9pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.1pip exceeds 2.5x normal 0.4pip
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=1000 entry=1.35007 tp=1.36432 sl=1.34365
  - risk metrics: risk=`1009.5 JPY` reward=`2240.6 JPY` rr=`2.22` spread=`10.7pip`
  - risk BLOCK: SPREAD_TOO_WIDE GBP_USD spread 10.7pip exceeds 2.5x normal 0.9pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.1pip exceeds 2.5x normal 0.4pip
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=1000 entry=112.917 tp=117.285 sl=112.371
  - risk metrics: risk=`546.0 JPY` reward=`4368.0 JPY` rr=`8.00` spread=`9.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 9.1pip exceeds 2.5x normal 0.8pip
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=1000 entry=184.122 tp=185.631 sl=183.348
  - risk metrics: risk=`774.0 JPY` reward=`1509.0 JPY` rr=`1.95` spread=`12.9pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 12.9pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

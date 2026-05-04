# Order Intents Report

- Generated at UTC: `2026-05-04T22:56:17.553768+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `6`
- `LIVE_READY`: `6`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=9000 entry=112.737 tp=113.649 sl=112.623
  - risk metrics: risk=`1026.0 JPY` reward=`8208.0 JPY` rr=`8.00` spread=`1.9pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 20.8s > 20s
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=7000 entry=183.872 tp=184.153 sl=183.728
  - risk metrics: risk=`1008.0 JPY` reward=`1967.0 JPY` rr=`1.95` spread=`2.4pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.4pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.16928 tp=1.17156 sl=1.1688
  - risk metrics: risk=`981.1 JPY` reward=`4660.3 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.1688 tp=1.16592 sl=1.16928
  - risk metrics: risk=`981.1 JPY` reward=`5886.7 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35313 tp=1.35486 sl=1.35235
  - risk metrics: risk=`981.1 JPY` reward=`2176.1 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=9000 entry=112.642 tp=113.554 sl=112.528
  - risk metrics: risk=`1026.0 JPY` reward=`8208.0 JPY` rr=`8.00` spread=`1.9pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 20.8s > 20s
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=7000 entry=183.752 tp=184.033 sl=183.608
  - risk metrics: risk=`1008.0 JPY` reward=`1967.0 JPY` rr=`1.95` spread=`2.4pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.4pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.1688 tp=1.17108 sl=1.16832
  - risk metrics: risk=`981.1 JPY` reward=`4660.3 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.16928 tp=1.1664 sl=1.16976
  - risk metrics: risk=`981.1 JPY` reward=`5886.7 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35248 tp=1.35421 sl=1.3517
  - risk metrics: risk=`981.1 JPY` reward=`2176.1 JPY` rr=`2.22` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=9000 entry=112.737 tp=113.649 sl=112.623
  - risk metrics: risk=`1026.0 JPY` reward=`8208.0 JPY` rr=`8.00` spread=`1.9pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 20.8s > 20s
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=7000 entry=183.872 tp=184.153 sl=183.728
  - risk metrics: risk=`1008.0 JPY` reward=`1967.0 JPY` rr=`1.95` spread=`2.4pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.4pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

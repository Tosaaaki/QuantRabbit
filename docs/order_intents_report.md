# Order Intents Report

- Generated at UTC: `2026-05-04T06:32:42.698654+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `2`
- `LIVE_READY`: `10`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=21000 entry=113.044 tp=113.812 sl=112.948
  - risk metrics: risk=`2016.0 JPY` reward=`16128.0 JPY` rr=`8.00` spread=`1.6pip`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=27000 entry=1.17383 tp=1.17611 sl=1.17335
  - risk metrics: risk=`2031.8 JPY` reward=`9651.1 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=27000 entry=1.17335 tp=1.17047 sl=1.17383
  - risk metrics: risk=`2031.8 JPY` reward=`12190.9 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=21000 entry=112.964 tp=113.732 sl=112.868
  - risk metrics: risk=`2016.0 JPY` reward=`16128.0 JPY` rr=`8.00` spread=`1.6pip`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=27000 entry=1.17335 tp=1.17563 sl=1.17287
  - risk metrics: risk=`2031.8 JPY` reward=`9651.1 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=27000 entry=1.17383 tp=1.17095 sl=1.17431
  - risk metrics: risk=`2031.8 JPY` reward=`12190.9 JPY` rr=`6.00` spread=`0.8pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=21000 entry=113.044 tp=113.812 sl=112.948
  - risk metrics: risk=`2016.0 JPY` reward=`16128.0 JPY` rr=`8.00` spread=`1.6pip`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=27000 entry=1.17383 tp=1.17611 sl=1.17335
  - risk metrics: risk=`2031.8 JPY` reward=`9651.1 JPY` rr=`4.75` spread=`0.8pip`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=27000 entry=1.17335 tp=1.17047 sl=1.17383
  - risk metrics: risk=`2031.8 JPY` reward=`12190.9 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=17000 entry=184.052 tp=184.286 sl=183.932
  - risk metrics: risk=`2040.0 JPY` reward=`3978.0 JPY` rr=`1.95` spread=`2.0pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=17000 entry=1.35993 tp=1.36166 sl=1.35915
  - risk metrics: risk=`2078.8 JPY` reward=`4610.8 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=17000 entry=183.952 tp=184.186 sl=183.832
  - risk metrics: risk=`2040.0 JPY` reward=`3978.0 JPY` rr=`1.95` spread=`2.0pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

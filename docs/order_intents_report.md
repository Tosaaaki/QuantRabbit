# Order Intents Report

- Generated at UTC: `2026-05-04T15:43:45.136858+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `3`
- `LIVE_READY`: `9`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=9000 entry=112.713 tp=113.625 sl=112.599
  - risk metrics: risk=`1026.0 JPY` reward=`8208.0 JPY` rr=`8.00` spread=`1.9pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=7000 entry=183.961 tp=184.254 sl=183.811
  - risk metrics: risk=`1050.0 JPY` reward=`2051.0 JPY` rr=`1.95` spread=`2.5pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.5pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=12000 entry=1.17006 tp=1.17256 sl=1.16953
  - risk metrics: risk=`999.7 JPY` reward=`4715.7 JPY` rr=`4.72` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=12000 entry=1.16958 tp=1.16643 sl=1.17011
  - risk metrics: risk=`999.7 JPY` reward=`5941.8 JPY` rr=`5.94` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35353 tp=1.35526 sl=1.35275
  - risk metrics: risk=`980.9 JPY` reward=`2175.5 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=9000 entry=112.618 tp=113.53 sl=112.504
  - risk metrics: risk=`1026.0 JPY` reward=`8208.0 JPY` rr=`8.00` spread=`1.9pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=7000 entry=183.836 tp=184.129 sl=183.686
  - risk metrics: risk=`1050.0 JPY` reward=`2051.0 JPY` rr=`1.95` spread=`2.5pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.5pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=12000 entry=1.16958 tp=1.17208 sl=1.16905
  - risk metrics: risk=`999.7 JPY` reward=`4715.7 JPY` rr=`4.72` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=12000 entry=1.17006 tp=1.16691 sl=1.17059
  - risk metrics: risk=`999.7 JPY` reward=`5941.8 JPY` rr=`5.94` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35288 tp=1.35461 sl=1.3521
  - risk metrics: risk=`980.9 JPY` reward=`2175.5 JPY` rr=`2.22` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=9000 entry=112.713 tp=113.625 sl=112.599
  - risk metrics: risk=`1026.0 JPY` reward=`8208.0 JPY` rr=`8.00` spread=`1.9pip`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=7000 entry=183.961 tp=184.254 sl=183.811
  - risk metrics: risk=`1050.0 JPY` reward=`2051.0 JPY` rr=`1.95` spread=`2.5pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.5pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

# Order Intents Report

- Generated at UTC: `2026-05-04T10:04:15.860704+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `3`
- `LIVE_READY`: `9`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=18000 entry=112.879 tp=113.791 sl=112.765
  - risk metrics: risk=`2052.0 JPY` reward=`16416.0 JPY` rr=`8.00` spread=`1.9pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=14000 entry=183.981 tp=184.274 sl=183.831
  - risk metrics: risk=`2100.0 JPY` reward=`4102.0 JPY` rr=`1.95` spread=`2.5pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.5pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=28000 entry=1.17118 tp=1.17346 sl=1.1707
  - risk metrics: risk=`2110.8 JPY` reward=`10026.3 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=28000 entry=1.1707 tp=1.16782 sl=1.17118
  - risk metrics: risk=`2110.8 JPY` reward=`12664.8 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=17000 entry=1.35505 tp=1.35678 sl=1.35427
  - risk metrics: risk=`2082.5 JPY` reward=`4619.0 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=18000 entry=112.784 tp=113.696 sl=112.67
  - risk metrics: risk=`2052.0 JPY` reward=`16416.0 JPY` rr=`8.00` spread=`1.9pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=14000 entry=183.856 tp=184.149 sl=183.706
  - risk metrics: risk=`2100.0 JPY` reward=`4102.0 JPY` rr=`1.95` spread=`2.5pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.5pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=28000 entry=1.1707 tp=1.17298 sl=1.17022
  - risk metrics: risk=`2110.8 JPY` reward=`10026.3 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=28000 entry=1.17118 tp=1.1683 sl=1.17166
  - risk metrics: risk=`2110.8 JPY` reward=`12664.8 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=17000 entry=1.3544 tp=1.35613 sl=1.35362
  - risk metrics: risk=`2082.5 JPY` reward=`4619.0 JPY` rr=`2.22` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=18000 entry=112.879 tp=113.791 sl=112.765
  - risk metrics: risk=`2052.0 JPY` reward=`16416.0 JPY` rr=`8.00` spread=`1.9pip`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=14000 entry=183.981 tp=184.274 sl=183.831
  - risk metrics: risk=`2100.0 JPY` reward=`4102.0 JPY` rr=`1.95` spread=`2.5pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.5pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

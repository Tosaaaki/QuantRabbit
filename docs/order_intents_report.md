# Order Intents Report

- Generated at UTC: `2026-05-04T09:29:35.910527+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `LIVE_READY`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=19000 entry=113.036 tp=113.9 sl=112.928
  - risk metrics: risk=`2052.0 JPY` reward=`16416.0 JPY` rr=`8.00` spread=`1.8pip`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=28000 entry=1.172 tp=1.17428 sl=1.17152
  - risk metrics: risk=`2110.0 JPY` reward=`10022.4 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=28000 entry=1.17152 tp=1.16864 sl=1.172
  - risk metrics: risk=`2110.0 JPY` reward=`12659.9 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=19000 entry=112.946 tp=113.81 sl=112.838
  - risk metrics: risk=`2052.0 JPY` reward=`16416.0 JPY` rr=`8.00` spread=`1.8pip`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=28000 entry=1.17152 tp=1.1738 sl=1.17104
  - risk metrics: risk=`2110.0 JPY` reward=`10022.4 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=28000 entry=1.172 tp=1.16912 sl=1.17248
  - risk metrics: risk=`2110.0 JPY` reward=`12659.9 JPY` rr=`6.00` spread=`0.8pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=19000 entry=113.036 tp=113.9 sl=112.928
  - risk metrics: risk=`2052.0 JPY` reward=`16416.0 JPY` rr=`8.00` spread=`1.8pip`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=28000 entry=1.172 tp=1.17428 sl=1.17152
  - risk metrics: risk=`2110.0 JPY` reward=`10022.4 JPY` rr=`4.75` spread=`0.8pip`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=28000 entry=1.17152 tp=1.16864 sl=1.172
  - risk metrics: risk=`2110.0 JPY` reward=`12659.9 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=25000 entry=184.005 tp=184.169 sl=183.921
  - risk metrics: risk=`2100.0 JPY` reward=`4100.0 JPY` rr=`1.95` spread=`1.4pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=17000 entry=1.35627 tp=1.358 sl=1.35549
  - risk metrics: risk=`2081.7 JPY` reward=`4617.2 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=25000 entry=183.935 tp=184.099 sl=183.851
  - risk metrics: risk=`2100.0 JPY` reward=`4100.0 JPY` rr=`1.95` spread=`1.4pip`

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

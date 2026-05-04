# Order Intents Report

- Generated at UTC: `2026-05-04T08:18:08.802067+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `LIVE_READY`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=19000 entry=112.879 tp=113.743 sl=112.771
  - risk metrics: risk=`2052.0 JPY` reward=`16416.0 JPY` rr=`8.00` spread=`1.8pip`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=27000 entry=1.17166 tp=1.17394 sl=1.17118
  - risk metrics: risk=`2034.4 JPY` reward=`9663.4 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=27000 entry=1.17118 tp=1.1683 sl=1.17166
  - risk metrics: risk=`2034.4 JPY` reward=`12206.5 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=19000 entry=112.789 tp=113.653 sl=112.681
  - risk metrics: risk=`2052.0 JPY` reward=`16416.0 JPY` rr=`8.00` spread=`1.8pip`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=27000 entry=1.17118 tp=1.17346 sl=1.1707
  - risk metrics: risk=`2034.4 JPY` reward=`9663.4 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=27000 entry=1.17166 tp=1.16878 sl=1.17214
  - risk metrics: risk=`2034.4 JPY` reward=`12206.5 JPY` rr=`6.00` spread=`0.8pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=19000 entry=112.879 tp=113.743 sl=112.771
  - risk metrics: risk=`2052.0 JPY` reward=`16416.0 JPY` rr=`8.00` spread=`1.8pip`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=27000 entry=1.17166 tp=1.17394 sl=1.17118
  - risk metrics: risk=`2034.4 JPY` reward=`9663.4 JPY` rr=`4.75` spread=`0.8pip`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=27000 entry=1.17118 tp=1.1683 sl=1.17166
  - risk metrics: risk=`2034.4 JPY` reward=`12206.5 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=18000 entry=183.945 tp=184.167 sl=183.831
  - risk metrics: risk=`2052.0 JPY` reward=`3996.0 JPY` rr=`1.95` spread=`1.9pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=17000 entry=1.35633 tp=1.35806 sl=1.35555
  - risk metrics: risk=`2081.5 JPY` reward=`4616.7 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=18000 entry=183.85 tp=184.072 sl=183.736
  - risk metrics: risk=`2052.0 JPY` reward=`3996.0 JPY` rr=`1.95` spread=`1.9pip`

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

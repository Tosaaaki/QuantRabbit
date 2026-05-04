# Order Intents Report

- Generated at UTC: `2026-05-04T18:11:56.103699+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `LIVE_READY`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.782 tp=113.598 sl=112.68
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=10000 entry=183.929 tp=184.128 sl=183.827
  - risk metrics: risk=`1020.0 JPY` reward=`1990.0 JPY` rr=`1.95` spread=`1.7pip`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.16968 tp=1.17196 sl=1.1692
  - risk metrics: risk=`981.1 JPY` reward=`4660.4 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.1692 tp=1.16632 sl=1.16968
  - risk metrics: risk=`981.1 JPY` reward=`5886.8 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35365 tp=1.35538 sl=1.35287
  - risk metrics: risk=`981.1 JPY` reward=`2176.1 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.697 tp=113.513 sl=112.595
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=10000 entry=183.844 tp=184.043 sl=183.742
  - risk metrics: risk=`1020.0 JPY` reward=`1990.0 JPY` rr=`1.95` spread=`1.7pip`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.1692 tp=1.17148 sl=1.16872
  - risk metrics: risk=`981.1 JPY` reward=`4660.4 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.16968 tp=1.1668 sl=1.17016
  - risk metrics: risk=`981.1 JPY` reward=`5886.8 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.353 tp=1.35473 sl=1.35222
  - risk metrics: risk=`981.1 JPY` reward=`2176.1 JPY` rr=`2.22` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.782 tp=113.598 sl=112.68
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=10000 entry=183.929 tp=184.128 sl=183.827
  - risk metrics: risk=`1020.0 JPY` reward=`1990.0 JPY` rr=`1.95` spread=`1.7pip`

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

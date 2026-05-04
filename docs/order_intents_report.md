# Order Intents Report

- Generated at UTC: `2026-05-04T13:11:36.861460+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `3`
- `LIVE_READY`: `9`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.933 tp=113.749 sl=112.831
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=8000 entry=183.965 tp=184.199 sl=183.845
  - risk metrics: risk=`960.0 JPY` reward=`1872.0 JPY` rr=`1.95` spread=`2.0pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.1713 tp=1.17358 sl=1.17082
  - risk metrics: risk=`979.9 JPY` reward=`4654.6 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.17082 tp=1.16794 sl=1.1713
  - risk metrics: risk=`979.9 JPY` reward=`5879.5 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35657 tp=1.3583 sl=1.35579
  - risk metrics: risk=`979.9 JPY` reward=`2173.4 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.848 tp=113.664 sl=112.746
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=8000 entry=183.865 tp=184.099 sl=183.745
  - risk metrics: risk=`960.0 JPY` reward=`1872.0 JPY` rr=`1.95` spread=`2.0pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.17082 tp=1.1731 sl=1.17034
  - risk metrics: risk=`979.9 JPY` reward=`4654.6 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.1713 tp=1.16842 sl=1.17178
  - risk metrics: risk=`979.9 JPY` reward=`5879.5 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35592 tp=1.35765 sl=1.35514
  - risk metrics: risk=`979.9 JPY` reward=`2173.4 JPY` rr=`2.22` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.933 tp=113.749 sl=112.831
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=8000 entry=183.965 tp=184.199 sl=183.845
  - risk metrics: risk=`960.0 JPY` reward=`1872.0 JPY` rr=`1.95` spread=`2.0pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

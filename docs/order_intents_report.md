# Order Intents Report

- Generated at UTC: `2026-05-04T14:52:04.709609+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `3`
- `LIVE_READY`: `9`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.895 tp=113.711 sl=112.793
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=7000 entry=183.936 tp=184.193 sl=183.804
  - risk metrics: risk=`924.0 JPY` reward=`1799.0 JPY` rr=`1.95` spread=`2.2pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.17104 tp=1.17332 sl=1.17056
  - risk metrics: risk=`980.0 JPY` reward=`4654.8 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.17056 tp=1.16768 sl=1.17104
  - risk metrics: risk=`980.0 JPY` reward=`5879.7 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35613 tp=1.35786 sl=1.35535
  - risk metrics: risk=`980.0 JPY` reward=`2173.5 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.81 tp=113.626 sl=112.708
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=7000 entry=183.826 tp=184.083 sl=183.694
  - risk metrics: risk=`924.0 JPY` reward=`1799.0 JPY` rr=`1.95` spread=`2.2pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.17056 tp=1.17284 sl=1.17008
  - risk metrics: risk=`980.0 JPY` reward=`4654.8 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.17104 tp=1.16816 sl=1.17152
  - risk metrics: risk=`980.0 JPY` reward=`5879.7 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35548 tp=1.35721 sl=1.3547
  - risk metrics: risk=`980.0 JPY` reward=`2173.5 JPY` rr=`2.22` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.895 tp=113.711 sl=112.793
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=7000 entry=183.936 tp=184.193 sl=183.804
  - risk metrics: risk=`924.0 JPY` reward=`1799.0 JPY` rr=`1.95` spread=`2.2pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

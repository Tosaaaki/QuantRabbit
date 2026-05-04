# Order Intents Report

- Generated at UTC: `2026-05-04T23:36:46.474903+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `10`
- `LIVE_READY`: `2`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.714 tp=113.482 sl=112.618
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 58.6s > 20s
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=7000 entry=183.859 tp=184.116 sl=183.727
  - risk metrics: risk=`924.0 JPY` reward=`1799.0 JPY` rr=`1.95` spread=`2.2pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.16947 tp=1.17175 sl=1.16899
  - risk metrics: risk=`980.9 JPY` reward=`4659.4 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 38.8s > 20s
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.16899 tp=1.16611 sl=1.16947
  - risk metrics: risk=`980.9 JPY` reward=`5885.5 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 38.8s > 20s
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35344 tp=1.35517 sl=1.35266
  - risk metrics: risk=`980.9 JPY` reward=`2175.6 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.634 tp=113.402 sl=112.538
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 58.6s > 20s
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=7000 entry=183.749 tp=184.006 sl=183.617
  - risk metrics: risk=`924.0 JPY` reward=`1799.0 JPY` rr=`1.95` spread=`2.2pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.16899 tp=1.17127 sl=1.16851
  - risk metrics: risk=`980.9 JPY` reward=`4659.4 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 38.8s > 20s
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.16947 tp=1.16659 sl=1.16995
  - risk metrics: risk=`980.9 JPY` reward=`5885.5 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 38.8s > 20s
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35279 tp=1.35452 sl=1.35201
  - risk metrics: risk=`980.9 JPY` reward=`2175.6 JPY` rr=`2.22` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.714 tp=113.482 sl=112.618
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 58.6s > 20s
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=7000 entry=183.859 tp=184.116 sl=183.727
  - risk metrics: risk=`924.0 JPY` reward=`1799.0 JPY` rr=`1.95` spread=`2.2pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

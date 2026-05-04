# Order Intents Report

- Generated at UTC: `2026-05-04T20:52:58.620865+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `6`
- `LIVE_READY`: `6`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=7000 entry=112.74 tp=113.892 sl=112.596
  - risk metrics: risk=`1008.0 JPY` reward=`8064.0 JPY` rr=`8.00` spread=`2.4pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 24.3s > 20s
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 2.4pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=6000 entry=183.927 tp=184.266 sl=183.753
  - risk metrics: risk=`1044.0 JPY` reward=`2034.0 JPY` rr=`1.95` spread=`2.9pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.9pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.16944 tp=1.17172 sl=1.16896
  - risk metrics: risk=`981.2 JPY` reward=`4660.6 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.16896 tp=1.16608 sl=1.16944
  - risk metrics: risk=`981.2 JPY` reward=`5887.1 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35353 tp=1.35526 sl=1.35275
  - risk metrics: risk=`981.2 JPY` reward=`2176.2 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=7000 entry=112.62 tp=113.772 sl=112.476
  - risk metrics: risk=`1008.0 JPY` reward=`8064.0 JPY` rr=`8.00` spread=`2.4pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 24.3s > 20s
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 2.4pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=6000 entry=183.782 tp=184.121 sl=183.608
  - risk metrics: risk=`1044.0 JPY` reward=`2034.0 JPY` rr=`1.95` spread=`2.9pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.9pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.16896 tp=1.17124 sl=1.16848
  - risk metrics: risk=`981.2 JPY` reward=`4660.6 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.16944 tp=1.16656 sl=1.16992
  - risk metrics: risk=`981.2 JPY` reward=`5887.1 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35288 tp=1.35461 sl=1.3521
  - risk metrics: risk=`981.2 JPY` reward=`2176.2 JPY` rr=`2.22` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=7000 entry=112.74 tp=113.892 sl=112.596
  - risk metrics: risk=`1008.0 JPY` reward=`8064.0 JPY` rr=`8.00` spread=`2.4pip`
  - risk BLOCK: STALE_QUOTE AUD_JPY quote is stale: 24.3s > 20s
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 2.4pip exceeds 2.5x normal 0.8pip
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=6000 entry=183.927 tp=184.266 sl=183.753
  - risk metrics: risk=`1044.0 JPY` reward=`2034.0 JPY` rr=`1.95` spread=`2.9pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.9pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

# Order Intents Report

- Generated at UTC: `2026-05-04T16:04:37.735629+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `3`
- `LIVE_READY`: `9`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.686 tp=113.454 sl=112.59
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=7000 entry=183.98 tp=184.249 sl=183.842
  - risk metrics: risk=`966.0 JPY` reward=`1883.0 JPY` rr=`1.95` spread=`2.3pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.3pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=12000 entry=1.16978 tp=1.17234 sl=1.16924
  - risk metrics: risk=`1019.0 JPY` reward=`4830.8 JPY` rr=`4.74` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=12000 entry=1.1693 tp=1.16608 sl=1.16984
  - risk metrics: risk=`1019.0 JPY` reward=`6076.2 JPY` rr=`5.96` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35313 tp=1.35486 sl=1.35235
  - risk metrics: risk=`981.2 JPY` reward=`2176.4 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.606 tp=113.374 sl=112.51
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=7000 entry=183.865 tp=184.134 sl=183.727
  - risk metrics: risk=`966.0 JPY` reward=`1883.0 JPY` rr=`1.95` spread=`2.3pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.3pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=12000 entry=1.1693 tp=1.17186 sl=1.16876
  - risk metrics: risk=`1019.0 JPY` reward=`4830.8 JPY` rr=`4.74` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=12000 entry=1.16978 tp=1.16656 sl=1.17032
  - risk metrics: risk=`1019.0 JPY` reward=`6076.2 JPY` rr=`5.96` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35248 tp=1.35421 sl=1.3517
  - risk metrics: risk=`981.2 JPY` reward=`2176.4 JPY` rr=`2.22` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.686 tp=113.454 sl=112.59
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=7000 entry=183.98 tp=184.249 sl=183.842
  - risk metrics: risk=`966.0 JPY` reward=`1883.0 JPY` rr=`1.95` spread=`2.3pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.3pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

# Order Intents Report

- Generated at UTC: `2026-05-04T09:20:51.947829+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `2`
- `LIVE_READY`: `10`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=22000 entry=113.02 tp=113.788 sl=112.924
  - risk metrics: risk=`2112.0 JPY` reward=`16896.0 JPY` rr=`8.00` spread=`1.6pip`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=28000 entry=1.17232 tp=1.1746 sl=1.17184
  - risk metrics: risk=`2109.8 JPY` reward=`10021.5 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=28000 entry=1.17184 tp=1.16896 sl=1.17232
  - risk metrics: risk=`2109.8 JPY` reward=`12658.7 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=22000 entry=112.94 tp=113.708 sl=112.844
  - risk metrics: risk=`2112.0 JPY` reward=`16896.0 JPY` rr=`8.00` spread=`1.6pip`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=28000 entry=1.17184 tp=1.17412 sl=1.17136
  - risk metrics: risk=`2109.8 JPY` reward=`10021.5 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=28000 entry=1.17232 tp=1.16944 sl=1.1728
  - risk metrics: risk=`2109.8 JPY` reward=`12658.7 JPY` rr=`6.00` spread=`0.8pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=22000 entry=113.02 tp=113.788 sl=112.924
  - risk metrics: risk=`2112.0 JPY` reward=`16896.0 JPY` rr=`8.00` spread=`1.6pip`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=28000 entry=1.17232 tp=1.1746 sl=1.17184
  - risk metrics: risk=`2109.8 JPY` reward=`10021.5 JPY` rr=`4.75` spread=`0.8pip`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=28000 entry=1.17184 tp=1.16896 sl=1.17232
  - risk metrics: risk=`2109.8 JPY` reward=`12658.7 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=16000 entry=184.057 tp=184.314 sl=183.925
  - risk metrics: risk=`2112.0 JPY` reward=`4112.0 JPY` rr=`1.95` spread=`2.2pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=17000 entry=1.35653 tp=1.35826 sl=1.35575
  - risk metrics: risk=`2081.5 JPY` reward=`4616.7 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=16000 entry=183.947 tp=184.204 sl=183.815
  - risk metrics: risk=`2112.0 JPY` reward=`4112.0 JPY` rr=`1.95` spread=`2.2pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

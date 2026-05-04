# Order Intents Report

- Generated at UTC: `2026-05-04T04:16:47.221101+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=17000 entry=112.913 tp=113.921 sl=112.787
  - risk metrics: risk=`2142.0 JPY` reward=`17136.0 JPY` rr=`8.00` spread=`2.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 2.1pip exceeds 2.5x normal 0.8pip
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2142 JPY exceeds cap 500 JPY
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=17000 entry=1.17349 tp=1.17729 sl=1.17269
  - risk metrics: risk=`2128.8 JPY` reward=`10111.8 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2129 JPY exceeds cap 500 JPY
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=17000 entry=1.17301 tp=1.16822 sl=1.17381
  - risk metrics: risk=`2128.8 JPY` reward=`12746.2 JPY` rr=`5.99` spread=`0.8pip`
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2129 JPY exceeds cap 500 JPY
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=17000 entry=112.808 tp=113.816 sl=112.682
  - risk metrics: risk=`2142.0 JPY` reward=`17136.0 JPY` rr=`8.00` spread=`2.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 2.1pip exceeds 2.5x normal 0.8pip
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2142 JPY exceeds cap 500 JPY
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=17000 entry=1.17301 tp=1.17681 sl=1.17221
  - risk metrics: risk=`2128.8 JPY` reward=`10111.8 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2129 JPY exceeds cap 500 JPY
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=17000 entry=1.17349 tp=1.1687 sl=1.17429
  - risk metrics: risk=`2128.8 JPY` reward=`12746.2 JPY` rr=`5.99` spread=`0.8pip`
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2129 JPY exceeds cap 500 JPY
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=17000 entry=112.913 tp=113.921 sl=112.787
  - risk metrics: risk=`2142.0 JPY` reward=`17136.0 JPY` rr=`8.00` spread=`2.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 2.1pip exceeds 2.5x normal 0.8pip
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2142 JPY exceeds cap 500 JPY
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=17000 entry=1.17349 tp=1.17729 sl=1.17269
  - risk metrics: risk=`2128.8 JPY` reward=`10111.8 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2129 JPY exceeds cap 500 JPY
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=17000 entry=1.17301 tp=1.16822 sl=1.17381
  - risk metrics: risk=`2128.8 JPY` reward=`12746.2 JPY` rr=`5.99` spread=`0.8pip`
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2129 JPY exceeds cap 500 JPY
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=20000 entry=183.711 tp=183.922 sl=183.603
  - risk metrics: risk=`2160.0 JPY` reward=`4220.0 JPY` rr=`1.95` spread=`1.8pip`
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2160 JPY exceeds cap 500 JPY
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=17000 entry=1.35944 tp=1.36122 sl=1.35864
  - risk metrics: risk=`2128.8 JPY` reward=`4736.6 JPY` rr=`2.23` spread=`1.3pip`
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2129 JPY exceeds cap 500 JPY
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=20000 entry=183.621 tp=183.832 sl=183.513
  - risk metrics: risk=`2160.0 JPY` reward=`4220.0 JPY` rr=`1.95` spread=`1.8pip`
  - risk BLOCK: LOSS_CAP_EXCEEDED planned worst-case loss 2160 JPY exceeds cap 500 JPY

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

# Order Intents Report

- Generated at UTC: `2026-05-04T08:45:14.289903+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `2`
- `LIVE_READY`: `10`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=17000 entry=112.983 tp=113.943 sl=112.863
  - risk metrics: risk=`2040.0 JPY` reward=`16320.0 JPY` rr=`8.00` spread=`2.0pip`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=28000 entry=1.17227 tp=1.17455 sl=1.17179
  - risk metrics: risk=`2109.2 JPY` reward=`10018.9 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=28000 entry=1.17179 tp=1.16891 sl=1.17227
  - risk metrics: risk=`2109.2 JPY` reward=`12655.5 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=17000 entry=112.883 tp=113.843 sl=112.763
  - risk metrics: risk=`2040.0 JPY` reward=`16320.0 JPY` rr=`8.00` spread=`2.0pip`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=28000 entry=1.17179 tp=1.17407 sl=1.17131
  - risk metrics: risk=`2109.2 JPY` reward=`10018.9 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=28000 entry=1.17227 tp=1.16939 sl=1.17275
  - risk metrics: risk=`2109.2 JPY` reward=`12655.5 JPY` rr=`6.00` spread=`0.8pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=17000 entry=112.983 tp=113.943 sl=112.863
  - risk metrics: risk=`2040.0 JPY` reward=`16320.0 JPY` rr=`8.00` spread=`2.0pip`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=28000 entry=1.17227 tp=1.17455 sl=1.17179
  - risk metrics: risk=`2109.2 JPY` reward=`10018.9 JPY` rr=`4.75` spread=`0.8pip`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=28000 entry=1.17179 tp=1.16891 sl=1.17227
  - risk metrics: risk=`2109.2 JPY` reward=`12655.5 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=15000 entry=184.007 tp=184.276 sl=183.869
  - risk metrics: risk=`2070.0 JPY` reward=`4035.0 JPY` rr=`1.95` spread=`2.3pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.3pip exceeds 2.5x normal 0.8pip
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=17000 entry=1.35652 tp=1.35825 sl=1.35574
  - risk metrics: risk=`2081.0 JPY` reward=`4615.5 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=15000 entry=183.892 tp=184.161 sl=183.754
  - risk metrics: risk=`2070.0 JPY` reward=`4035.0 JPY` rr=`1.95` spread=`2.3pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.3pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

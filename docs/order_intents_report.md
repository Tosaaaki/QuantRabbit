# Order Intents Report

- Generated at UTC: `2026-05-04T07:35:35.016437+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `2`
- `LIVE_READY`: `10`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=21000 entry=112.896 tp=113.664 sl=112.8
  - risk metrics: risk=`2016.0 JPY` reward=`16128.0 JPY` rr=`8.00` spread=`1.6pip`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=27000 entry=1.17262 tp=1.1749 sl=1.17214
  - risk metrics: risk=`2033.5 JPY` reward=`9658.9 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=27000 entry=1.17214 tp=1.16926 sl=1.17262
  - risk metrics: risk=`2033.5 JPY` reward=`12200.8 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=21000 entry=112.816 tp=113.584 sl=112.72
  - risk metrics: risk=`2016.0 JPY` reward=`16128.0 JPY` rr=`8.00` spread=`1.6pip`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=27000 entry=1.17214 tp=1.17442 sl=1.17166
  - risk metrics: risk=`2033.5 JPY` reward=`9658.9 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=27000 entry=1.17262 tp=1.16974 sl=1.1731
  - risk metrics: risk=`2033.5 JPY` reward=`12200.8 JPY` rr=`6.00` spread=`0.8pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=21000 entry=112.896 tp=113.664 sl=112.8
  - risk metrics: risk=`2016.0 JPY` reward=`16128.0 JPY` rr=`8.00` spread=`1.6pip`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=27000 entry=1.17262 tp=1.1749 sl=1.17214
  - risk metrics: risk=`2033.5 JPY` reward=`9658.9 JPY` rr=`4.75` spread=`0.8pip`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=27000 entry=1.17214 tp=1.16926 sl=1.17262
  - risk metrics: risk=`2033.5 JPY` reward=`12200.8 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=16000 entry=184.013 tp=184.259 sl=183.887
  - risk metrics: risk=`2016.0 JPY` reward=`3936.0 JPY` rr=`1.95` spread=`2.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.1pip exceeds 2.5x normal 0.8pip
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=17000 entry=1.35767 tp=1.3594 sl=1.35689
  - risk metrics: risk=`2080.5 JPY` reward=`4614.5 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=16000 entry=183.908 tp=184.154 sl=183.782
  - risk metrics: risk=`2016.0 JPY` reward=`3936.0 JPY` rr=`1.95` spread=`2.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.1pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

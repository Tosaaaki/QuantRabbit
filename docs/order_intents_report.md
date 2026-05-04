# Order Intents Report

- Generated at UTC: `2026-05-04T08:33:33.531054+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `2`
- `LIVE_READY`: `10`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=20000 entry=112.963 tp=113.779 sl=112.861
  - risk metrics: risk=`2040.0 JPY` reward=`16320.0 JPY` rr=`8.00` spread=`1.7pip`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=27000 entry=1.17218 tp=1.17446 sl=1.1717
  - risk metrics: risk=`2033.9 JPY` reward=`9660.9 JPY` rr=`4.75` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=27000 entry=1.1717 tp=1.16882 sl=1.17218
  - risk metrics: risk=`2033.9 JPY` reward=`12203.2 JPY` rr=`6.00` spread=`0.8pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=20000 entry=112.878 tp=113.694 sl=112.776
  - risk metrics: risk=`2040.0 JPY` reward=`16320.0 JPY` rr=`8.00` spread=`1.7pip`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=27000 entry=1.1717 tp=1.17398 sl=1.17122
  - risk metrics: risk=`2033.9 JPY` reward=`9660.9 JPY` rr=`4.75` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=27000 entry=1.17218 tp=1.1693 sl=1.17266
  - risk metrics: risk=`2033.9 JPY` reward=`12203.2 JPY` rr=`6.00` spread=`0.8pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=20000 entry=112.963 tp=113.779 sl=112.861
  - risk metrics: risk=`2040.0 JPY` reward=`16320.0 JPY` rr=`8.00` spread=`1.7pip`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=27000 entry=1.17218 tp=1.17446 sl=1.1717
  - risk metrics: risk=`2033.9 JPY` reward=`9660.9 JPY` rr=`4.75` spread=`0.8pip`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=27000 entry=1.1717 tp=1.16882 sl=1.17218
  - risk metrics: risk=`2033.9 JPY` reward=`12203.2 JPY` rr=`6.00` spread=`0.8pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=15000 entry=183.985 tp=184.242 sl=183.853
  - risk metrics: risk=`1980.0 JPY` reward=`3855.0 JPY` rr=`1.95` spread=`2.2pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=17000 entry=1.35666 tp=1.35839 sl=1.35588
  - risk metrics: risk=`2080.9 JPY` reward=`4615.4 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=15000 entry=183.875 tp=184.132 sl=183.743
  - risk metrics: risk=`1980.0 JPY` reward=`3855.0 JPY` rr=`1.95` spread=`2.2pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.2pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

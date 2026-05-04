# Order Intents Report

- Generated at UTC: `2026-05-04T14:36:17.035518+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.938 tp=113.706 sl=112.842
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=10000 entry=183.977 tp=184.164 sl=183.881
  - risk metrics: risk=`960.0 JPY` reward=`1870.0 JPY` rr=`1.95` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.17154 tp=1.17382 sl=1.17106
  - risk metrics: risk=`979.8 JPY` reward=`4654.3 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.17106 tp=1.16818 sl=1.17154
  - risk metrics: risk=`979.8 JPY` reward=`5879.1 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35678 tp=1.35851 sl=1.356
  - risk metrics: risk=`979.8 JPY` reward=`2173.2 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.858 tp=113.626 sl=112.762
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=10000 entry=183.897 tp=184.084 sl=183.801
  - risk metrics: risk=`960.0 JPY` reward=`1870.0 JPY` rr=`1.95` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.17106 tp=1.17334 sl=1.17058
  - risk metrics: risk=`979.8 JPY` reward=`4654.3 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.17154 tp=1.16866 sl=1.17202
  - risk metrics: risk=`979.8 JPY` reward=`5879.1 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35613 tp=1.35786 sl=1.35535
  - risk metrics: risk=`979.8 JPY` reward=`2173.2 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.938 tp=113.706 sl=112.842
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=10000 entry=183.977 tp=184.164 sl=183.881
  - risk metrics: risk=`960.0 JPY` reward=`1870.0 JPY` rr=`1.95` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1000 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

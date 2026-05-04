# Order Intents Report

- Generated at UTC: `2026-05-04T14:24:01.300809+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.941 tp=113.757 sl=112.839
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 1020 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=10000 entry=184.04 tp=184.227 sl=183.944
  - risk metrics: risk=`960.0 JPY` reward=`1870.0 JPY` rr=`1.95` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.17165 tp=1.17393 sl=1.17117
  - risk metrics: risk=`980.1 JPY` reward=`4655.5 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.17117 tp=1.16829 sl=1.17165
  - risk metrics: risk=`980.1 JPY` reward=`5880.6 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35683 tp=1.35856 sl=1.35605
  - risk metrics: risk=`980.1 JPY` reward=`2173.8 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.856 tp=113.672 sl=112.754
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 1020 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=10000 entry=183.96 tp=184.147 sl=183.864
  - risk metrics: risk=`960.0 JPY` reward=`1870.0 JPY` rr=`1.95` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.17117 tp=1.17345 sl=1.17069
  - risk metrics: risk=`980.1 JPY` reward=`4655.5 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.17165 tp=1.16877 sl=1.17213
  - risk metrics: risk=`980.1 JPY` reward=`5880.6 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35618 tp=1.35791 sl=1.3554
  - risk metrics: risk=`980.1 JPY` reward=`2173.8 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.941 tp=113.757 sl=112.839
  - risk metrics: risk=`1020.0 JPY` reward=`8160.0 JPY` rr=`8.00` spread=`1.7pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 1020 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=10000 entry=184.04 tp=184.227 sl=183.944
  - risk metrics: risk=`960.0 JPY` reward=`1870.0 JPY` rr=`1.95` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

# Order Intents Report

- Generated at UTC: `2026-05-04T11:18:21.277084+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=9000 entry=112.887 tp=113.799 sl=112.773
  - risk metrics: risk=`1026.0 JPY` reward=`8208.0 JPY` rr=`8.00` spread=`1.9pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 1026 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=8000 entry=183.915 tp=184.149 sl=183.795
  - risk metrics: risk=`960.0 JPY` reward=`1872.0 JPY` rr=`1.95` spread=`2.0pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=12000 entry=1.1703 tp=1.17278 sl=1.16978
  - risk metrics: risk=`980.5 JPY` reward=`4676.2 JPY` rr=`4.77` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=12000 entry=1.16982 tp=1.1667 sl=1.17034
  - risk metrics: risk=`980.5 JPY` reward=`5882.9 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35443 tp=1.35616 sl=1.35365
  - risk metrics: risk=`980.5 JPY` reward=`2174.7 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=9000 entry=112.792 tp=113.704 sl=112.678
  - risk metrics: risk=`1026.0 JPY` reward=`8208.0 JPY` rr=`8.00` spread=`1.9pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 1026 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=8000 entry=183.815 tp=184.049 sl=183.695
  - risk metrics: risk=`960.0 JPY` reward=`1872.0 JPY` rr=`1.95` spread=`2.0pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=12000 entry=1.16982 tp=1.1723 sl=1.1693
  - risk metrics: risk=`980.5 JPY` reward=`4676.2 JPY` rr=`4.77` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=12000 entry=1.1703 tp=1.16718 sl=1.17082
  - risk metrics: risk=`980.5 JPY` reward=`5882.9 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35378 tp=1.35551 sl=1.353
  - risk metrics: risk=`980.5 JPY` reward=`2174.7 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=9000 entry=112.887 tp=113.799 sl=112.773
  - risk metrics: risk=`1026.0 JPY` reward=`8208.0 JPY` rr=`8.00` spread=`1.9pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 1026 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=8000 entry=183.915 tp=184.149 sl=183.795
  - risk metrics: risk=`960.0 JPY` reward=`1872.0 JPY` rr=`1.95` spread=`2.0pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.0pip exceeds 2.5x normal 0.8pip
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 980 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

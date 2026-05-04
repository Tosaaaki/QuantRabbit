# Order Intents Report

- Generated at UTC: `2026-05-04T11:43:29.917419+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.864 tp=113.632 sl=112.768
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=9000 entry=183.939 tp=184.161 sl=183.825
  - risk metrics: risk=`1026.0 JPY` reward=`1998.0 JPY` rr=`1.95` spread=`1.9pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 1026 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.17032 tp=1.17266 sl=1.16983
  - risk metrics: risk=`1001.1 JPY` reward=`4780.7 JPY` rr=`4.78` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 1001 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.16984 tp=1.16689 sl=1.17033
  - risk metrics: risk=`1001.1 JPY` reward=`6026.9 JPY` rr=`6.02` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 1001 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35453 tp=1.35626 sl=1.35375
  - risk metrics: risk=`980.7 JPY` reward=`2175.0 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 981 JPY exceeds portfolio cap 1051 JPY
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.784 tp=113.552 sl=112.688
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=9000 entry=183.844 tp=184.066 sl=183.73
  - risk metrics: risk=`1026.0 JPY` reward=`1998.0 JPY` rr=`1.95` spread=`1.9pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 1026 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.16984 tp=1.17218 sl=1.16935
  - risk metrics: risk=`1001.1 JPY` reward=`4780.7 JPY` rr=`4.78` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 1001 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.17032 tp=1.16737 sl=1.17081
  - risk metrics: risk=`1001.1 JPY` reward=`6026.9 JPY` rr=`6.02` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 1001 JPY exceeds portfolio cap 1051 JPY
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35388 tp=1.35561 sl=1.3531
  - risk metrics: risk=`980.7 JPY` reward=`2175.0 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 981 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.864 tp=113.632 sl=112.768
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 960 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=9000 entry=183.939 tp=184.161 sl=183.825
  - risk metrics: risk=`1026.0 JPY` reward=`1998.0 JPY` rr=`1.95` spread=`1.9pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 981 JPY + candidate risk 1026 JPY exceeds portfolio cap 1051 JPY

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

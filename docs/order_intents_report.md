# Order Intents Report

- Generated at UTC: `2026-05-04T14:31:14.522079+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=9000 entry=112.967 tp=113.831 sl=112.859
  - risk metrics: risk=`972.0 JPY` reward=`7776.0 JPY` rr=`8.00` spread=`1.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 972 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=9000 entry=184.074 tp=184.296 sl=183.96
  - risk metrics: risk=`1026.0 JPY` reward=`1998.0 JPY` rr=`1.95` spread=`1.9pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 1026 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.17172 tp=1.174 sl=1.17124
  - risk metrics: risk=`980.2 JPY` reward=`4655.8 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.17124 tp=1.16836 sl=1.17172
  - risk metrics: risk=`980.2 JPY` reward=`5881.0 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35703 tp=1.35876 sl=1.35625
  - risk metrics: risk=`980.2 JPY` reward=`2174.0 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=9000 entry=112.877 tp=113.741 sl=112.769
  - risk metrics: risk=`972.0 JPY` reward=`7776.0 JPY` rr=`8.00` spread=`1.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 972 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=9000 entry=183.979 tp=184.201 sl=183.865
  - risk metrics: risk=`1026.0 JPY` reward=`1998.0 JPY` rr=`1.95` spread=`1.9pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 1026 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.17124 tp=1.17352 sl=1.17076
  - risk metrics: risk=`980.2 JPY` reward=`4655.8 JPY` rr=`4.75` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.17172 tp=1.16884 sl=1.1722
  - risk metrics: risk=`980.2 JPY` reward=`5881.0 JPY` rr=`6.00` spread=`0.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35638 tp=1.35811 sl=1.3556
  - risk metrics: risk=`980.2 JPY` reward=`2174.0 JPY` rr=`2.22` spread=`1.3pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 980 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=9000 entry=112.967 tp=113.831 sl=112.859
  - risk metrics: risk=`972.0 JPY` reward=`7776.0 JPY` rr=`8.00` spread=`1.8pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 972 JPY exceeds portfolio cap 1051 JPY
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=9000 entry=184.074 tp=184.296 sl=183.96
  - risk metrics: risk=`1026.0 JPY` reward=`1998.0 JPY` rr=`1.95` spread=`1.9pip`
  - risk BLOCK: PORTFOLIO_LOSS_CAP_EXCEEDED open risk 1001 JPY + candidate risk 1026 JPY exceeds portfolio cap 1051 JPY

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

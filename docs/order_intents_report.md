# Order Intents Report

- Generated at UTC: `2026-05-04T21:15:22.064470+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=1000 entry=112.929 tp=117.393 sl=112.371
  - risk metrics: risk=`558.0 JPY` reward=`4464.0 JPY` rr=`8.00` spread=`9.3pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 9.3pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=1000 entry=184.151 tp=185.801 sl=183.305
  - risk metrics: risk=`846.0 JPY` reward=`1650.0 JPY` rr=`1.95` spread=`14.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 14.1pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=2000 entry=1.17026 tp=1.18365 sl=1.16744
  - risk metrics: risk=`886.8 JPY` reward=`4210.9 JPY` rr=`4.75` spread=`4.7pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 4.7pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.2pip exceeds 2.5x normal 0.4pip
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=2000 entry=1.16791 tp=1.15102 sl=1.17073
  - risk metrics: risk=`886.8 JPY` reward=`5311.6 JPY` rr=`5.99` spread=`4.7pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 4.7pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.2pip exceeds 2.5x normal 0.4pip
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=608 entry=1.35794 tp=1.38232 sl=1.34696
  - risk metrics: risk=`1049.7 JPY` reward=`2330.8 JPY` rr=`2.22` spread=`18.3pip`
  - risk BLOCK: SPREAD_TOO_WIDE GBP_USD spread 18.3pip exceeds 2.5x normal 0.9pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.2pip exceeds 2.5x normal 0.4pip
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=1000 entry=112.464 tp=116.928 sl=111.906
  - risk metrics: risk=`558.0 JPY` reward=`4464.0 JPY` rr=`8.00` spread=`9.3pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 9.3pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=1000 entry=183.446 tp=185.096 sl=182.6
  - risk metrics: risk=`846.0 JPY` reward=`1650.0 JPY` rr=`1.95` spread=`14.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 14.1pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=2000 entry=1.16791 tp=1.1813 sl=1.16509
  - risk metrics: risk=`886.8 JPY` reward=`4210.9 JPY` rr=`4.75` spread=`4.7pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 4.7pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.2pip exceeds 2.5x normal 0.4pip
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=2000 entry=1.17026 tp=1.15337 sl=1.17308
  - risk metrics: risk=`886.8 JPY` reward=`5311.6 JPY` rr=`5.99` spread=`4.7pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 4.7pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.2pip exceeds 2.5x normal 0.4pip
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=608 entry=1.34879 tp=1.37317 sl=1.33781
  - risk metrics: risk=`1049.7 JPY` reward=`2330.8 JPY` rr=`2.22` spread=`18.3pip`
  - risk BLOCK: SPREAD_TOO_WIDE GBP_USD spread 18.3pip exceeds 2.5x normal 0.9pip
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.2pip exceeds 2.5x normal 0.4pip
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=1000 entry=112.929 tp=117.393 sl=112.371
  - risk metrics: risk=`558.0 JPY` reward=`4464.0 JPY` rr=`8.00` spread=`9.3pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 9.3pip exceeds 2.5x normal 0.8pip
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=1000 entry=184.151 tp=185.801 sl=183.305
  - risk metrics: risk=`846.0 JPY` reward=`1650.0 JPY` rr=`1.95` spread=`14.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 14.1pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

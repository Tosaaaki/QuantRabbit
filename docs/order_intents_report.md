# Order Intents Report

- Generated at UTC: `2026-05-04T21:29:54.912308+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=1000 entry=112.928 tp=117.92 sl=112.304
  - risk metrics: risk=`624.0 JPY` reward=`4992.0 JPY` rr=`8.00` spread=`10.4pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 10.4pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=1000 entry=184.074 tp=185.373 sl=183.408
  - risk metrics: risk=`666.0 JPY` reward=`1299.0 JPY` rr=`1.95` spread=`11.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 11.1pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=4000 entry=1.16997 tp=1.17766 sl=1.16835
  - risk metrics: risk=`1018.8 JPY` reward=`4836.0 JPY` rr=`4.75` spread=`2.7pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 20.5s > 20s
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 2.7pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 21.1s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.3pip exceeds 2.5x normal 0.4pip
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=4000 entry=1.16862 tp=1.15892 sl=1.17024
  - risk metrics: risk=`1018.8 JPY` reward=`6100.0 JPY` rr=`5.99` spread=`2.7pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 20.5s > 20s
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 2.7pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 21.1s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.3pip exceeds 2.5x normal 0.4pip
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=1000 entry=1.35554 tp=1.36819 sl=1.34984
  - risk metrics: risk=`896.1 JPY` reward=`1988.8 JPY` rr=`2.22` spread=`9.5pip`
  - risk BLOCK: SPREAD_TOO_WIDE GBP_USD spread 9.5pip exceeds 2.5x normal 0.9pip
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 21.1s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.3pip exceeds 2.5x normal 0.4pip
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=1000 entry=112.408 tp=117.4 sl=111.784
  - risk metrics: risk=`624.0 JPY` reward=`4992.0 JPY` rr=`8.00` spread=`10.4pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 10.4pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=1000 entry=183.519 tp=184.818 sl=182.853
  - risk metrics: risk=`666.0 JPY` reward=`1299.0 JPY` rr=`1.95` spread=`11.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 11.1pip exceeds 2.5x normal 0.8pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=4000 entry=1.16862 tp=1.17631 sl=1.167
  - risk metrics: risk=`1018.8 JPY` reward=`4836.0 JPY` rr=`4.75` spread=`2.7pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 20.5s > 20s
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 2.7pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 21.1s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.3pip exceeds 2.5x normal 0.4pip
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=4000 entry=1.16997 tp=1.16027 sl=1.17159
  - risk metrics: risk=`1018.8 JPY` reward=`6100.0 JPY` rr=`5.99` spread=`2.7pip`
  - risk BLOCK: STALE_QUOTE EUR_USD quote is stale: 20.5s > 20s
  - risk BLOCK: SPREAD_TOO_WIDE EUR_USD spread 2.7pip exceeds 2.5x normal 0.5pip
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 21.1s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.3pip exceeds 2.5x normal 0.4pip
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=1000 entry=1.35079 tp=1.36344 sl=1.34509
  - risk metrics: risk=`896.1 JPY` reward=`1988.8 JPY` rr=`2.22` spread=`9.5pip`
  - risk BLOCK: SPREAD_TOO_WIDE GBP_USD spread 9.5pip exceeds 2.5x normal 0.9pip
  - risk BLOCK: STALE_CONVERSION_QUOTE USD_JPY conversion quote is stale: 21.1s > 20s
  - risk BLOCK: CONVERSION_SPREAD_TOO_WIDE USD_JPY conversion spread 5.3pip exceeds 2.5x normal 0.4pip
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=1000 entry=112.928 tp=117.92 sl=112.304
  - risk metrics: risk=`624.0 JPY` reward=`4992.0 JPY` rr=`8.00` spread=`10.4pip`
  - risk BLOCK: SPREAD_TOO_WIDE AUD_JPY spread 10.4pip exceeds 2.5x normal 0.8pip
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=1000 entry=184.074 tp=185.373 sl=183.408
  - risk metrics: risk=`666.0 JPY` reward=`1299.0 JPY` rr=`1.95` spread=`11.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 11.1pip exceeds 2.5x normal 0.8pip

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

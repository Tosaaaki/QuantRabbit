# Order Intents Report

- Generated at UTC: `2026-05-06T03:27:01.742336+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `DRY_RUN_BLOCKED`: `2`
- `LIVE_READY`: `10`

## Candidates

- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.1719 tp=1.17053 sl=1.17238
  - risk metrics: risk=`985.1 JPY` reward=`2811.7 JPY` rr=`2.85` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.17224 tp=1.1716 sl=1.17272
  - risk metrics: risk=`985.1 JPY` reward=`1313.5 JPY` rr=`1.33` spread=`0.8pip`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.1719 tp=1.17053 sl=1.17238
  - risk metrics: risk=`985.1 JPY` reward=`2811.7 JPY` rr=`2.85` spread=`0.8pip`
- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=114.294 tp=114.662 sl=114.198
  - risk metrics: risk=`960.0 JPY` reward=`3680.0 JPY` rr=`3.83` spread=`1.6pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=8000 entry=185.086 tp=185.275 sl=184.96
  - risk metrics: risk=`1008.0 JPY` reward=`1512.0 JPY` rr=`1.50` spread=`2.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.1pip exceeds 2.5x normal 0.8pip
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.1723 tp=1.17338 sl=1.17182
  - risk metrics: risk=`985.1 JPY` reward=`2216.5 JPY` rr=`2.25` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35723 tp=1.3584 sl=1.35645
  - risk metrics: risk=`985.1 JPY` reward=`1477.7 JPY` rr=`1.50` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=114.213 tp=114.336 sl=114.117
  - risk metrics: risk=`960.0 JPY` reward=`1230.0 JPY` rr=`1.28` spread=`1.6pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`DRY_RUN_BLOCKED`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=8000 entry=184.957 tp=185.057 sl=184.831
  - risk metrics: risk=`1008.0 JPY` reward=`800.0 JPY` rr=`0.79` spread=`2.1pip`
  - risk BLOCK: SPREAD_TOO_WIDE EUR_JPY spread 2.1pip exceeds 2.5x normal 0.8pip
  - risk BLOCK: REWARD_RISK_TOO_LOW planned reward/risk 0.79x is below 1.20x
  - risk BLOCK: TARGET_TOO_THIN_FOR_SPREAD target 10.0pip is less than 5.0x spread 2.1pip
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.1716 tp=1.17224 sl=1.17112
  - risk metrics: risk=`985.1 JPY` reward=`1313.5 JPY` rr=`1.33` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35668 tp=1.35784 sl=1.3559
  - risk metrics: risk=`985.1 JPY` reward=`1465.0 JPY` rr=`1.49` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=114.294 tp=114.662 sl=114.198
  - risk metrics: risk=`960.0 JPY` reward=`3680.0 JPY` rr=`3.83` spread=`1.6pip`

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

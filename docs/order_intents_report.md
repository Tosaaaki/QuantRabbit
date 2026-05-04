# Order Intents Report

- Generated at UTC: `2026-05-04T17:01:31.806151+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `LIVE_READY`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.618 tp=113.386 sl=112.522
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
- `failure_trader:EUR_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=10000 entry=183.802 tp=184.001 sl=183.7
  - risk metrics: risk=`1020.0 JPY` reward=`1990.0 JPY` rr=`1.95` spread=`1.7pip`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.16846 tp=1.17089 sl=1.16795
  - risk metrics: risk=`1042.8 JPY` reward=`4968.8 JPY` rr=`4.76` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.16798 tp=1.16491 sl=1.16849
  - risk metrics: risk=`1042.8 JPY` reward=`6277.4 JPY` rr=`6.02` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35233 tp=1.35406 sl=1.35155
  - risk metrics: risk=`981.5 JPY` reward=`2176.9 JPY` rr=`2.22` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.538 tp=113.306 sl=112.442
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
- `range_trader:EUR_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG LIMIT` units=10000 entry=183.717 tp=183.916 sl=183.615
  - risk metrics: risk=`1020.0 JPY` reward=`1990.0 JPY` rr=`1.95` spread=`1.7pip`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.16798 tp=1.17041 sl=1.16747
  - risk metrics: risk=`1042.8 JPY` reward=`4968.8 JPY` rr=`4.76` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.16846 tp=1.16539 sl=1.16897
  - risk metrics: risk=`1042.8 JPY` reward=`6277.4 JPY` rr=`6.02` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35168 tp=1.35341 sl=1.3509
  - risk metrics: risk=`981.5 JPY` reward=`2176.9 JPY` rr=`2.22` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.618 tp=113.386 sl=112.522
  - risk metrics: risk=`960.0 JPY` reward=`7680.0 JPY` rr=`8.00` spread=`1.6pip`
- `trend_trader:EUR_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_JPY LONG STOP-ENTRY` units=10000 entry=183.802 tp=184.001 sl=183.7
  - risk metrics: risk=`1020.0 JPY` reward=`1990.0 JPY` rr=`1.95` spread=`1.7pip`

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

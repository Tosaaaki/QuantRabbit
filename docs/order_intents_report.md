# Order Intents Report

- Generated at UTC: `2026-05-05T03:40:04.604562+00:00`
- Campaign plan: `/Users/tossaki/App/QuantRabbit/data/daily_campaign_plan.json`
- Snapshot: `/Users/tossaki/App/QuantRabbit/data/broker_snapshot.json`
- Results: `12`

## Status Counts

- `LIVE_READY`: `12`

## Candidates

- `failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.564 tp=112.932 sl=112.468
  - risk metrics: risk=`960.0 JPY` reward=`3680.0 JPY` rr=`3.83` spread=`1.6pip`
- `failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.16877 tp=1.16985 sl=1.16829
  - risk metrics: risk=`981.3 JPY` reward=`2207.8 JPY` rr=`2.25` spread=`0.8pip`
- `failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.16837 tp=1.167 sl=1.16885
  - risk metrics: risk=`981.3 JPY` reward=`2800.7 JPY` rr=`2.85` spread=`0.8pip`
- `failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35271 tp=1.35388 sl=1.35193
  - risk metrics: risk=`981.3 JPY` reward=`1471.9 JPY` rr=`1.50` spread=`1.3pip`
- `range_trader:AUD_JPY:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG LIMIT` units=10000 entry=112.5 tp=112.622 sl=112.404
  - risk metrics: risk=`960.0 JPY` reward=`1220.0 JPY` rr=`1.27` spread=`1.6pip`
- `range_trader:EUR_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG LIMIT` units=13000 entry=1.16845 tp=1.16906 sl=1.16797
  - risk metrics: risk=`981.3 JPY` reward=`1247.0 JPY` rr=`1.27` spread=`0.8pip`
- `range_trader:EUR_USD:SHORT:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT LIMIT` units=13000 entry=1.16892 tp=1.16813 sl=1.1694
  - risk metrics: risk=`981.3 JPY` reward=`1615.0 JPY` rr=`1.65` spread=`0.8pip`
- `range_trader:GBP_USD:LONG:RANGE_ROTATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG LIMIT` units=8000 entry=1.35219 tp=1.35313 sl=1.35141
  - risk metrics: risk=`981.3 JPY` reward=`1182.5 JPY` rr=`1.21` spread=`1.3pip`
- `trend_trader:AUD_JPY:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `AUD_JPY LONG STOP-ENTRY` units=10000 entry=112.564 tp=112.932 sl=112.468
  - risk metrics: risk=`960.0 JPY` reward=`3680.0 JPY` rr=`3.83` spread=`1.6pip`
- `trend_trader:EUR_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD LONG STOP-ENTRY` units=13000 entry=1.16877 tp=1.16985 sl=1.16829
  - risk metrics: risk=`981.3 JPY` reward=`2207.8 JPY` rr=`2.25` spread=`0.8pip`
- `trend_trader:EUR_USD:SHORT:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `EUR_USD SHORT STOP-ENTRY` units=13000 entry=1.16837 tp=1.167 sl=1.16885
  - risk metrics: risk=`981.3 JPY` reward=`2800.7 JPY` rr=`2.85` spread=`0.8pip`
- `trend_trader:GBP_USD:LONG:TREND_CONTINUATION` status=`LIVE_READY`
  - note: Dry-run geometry built from current snapshot; live use still requires fresh snapshot at send time.
  - intent: `GBP_USD LONG STOP-ENTRY` units=8000 entry=1.35271 tp=1.35388 sl=1.35193
  - risk metrics: risk=`981.3 JPY` reward=`1471.9 JPY` rr=`1.50` spread=`1.3pip`

## Completion Rule

- `NEEDS_BROKER_SNAPSHOT` means the system is waiting for read-only broker truth, not for discretionary prose.
- `DRY_RUN_PASSED` means geometry passed risk, but strategy profile still blocks live use until repair/trigger evidence is promoted.
- `LIVE_READY` means this dry-run layer found no risk/profile blocker; live gateway still requires fresh broker snapshot and explicit live enablement.

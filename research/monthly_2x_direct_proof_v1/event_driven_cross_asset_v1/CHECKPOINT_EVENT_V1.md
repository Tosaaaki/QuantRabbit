# Event-driven cross-asset checkpoint V1

## Outcome

`EVENT_DRIVEN_CROSS_ASSET_DISLOCATION_V1` is `NOT_EVALUABLE`. The parent monthly target remains `TARGET_PATH_NOT_YET_PROVEN`.

This is not an after-cost performance failure. System Admission stopped before replay, before any grid point or outcome was evaluated, and before any holdout read.

## Frozen hypothesis

The preregistration was saved before inspecting source suitability. It fixes six US macro series, three FX pairs, five cross-asset confirmations, 18 sparse grid points, TRAIN → one-hour embargo → 16/32/64-day VALIDATION, all parent cost/risk gates, and an unread holdout.

## Direct source findings

- `data/economic_calendar.json` is a single weekly snapshot (76 events, 2026-07-06 through 2026-07-10). It has 42 forecasts but 0 actuals, no provider receipt timestamp, no first-publication marker, and no revision lineage. The CLI writes the snapshot with `_write_json` rather than an append-only release ledger.
- `data/cross_asset_snapshot.json` stores current H1 scalar aggregates. It has the required SPX, gold, US 2Y and US 10Y proxies plus a synthetic DXY aggregate, but no stored event-time bar timestamps.
- `data/context_asset_charts.json` is one generated snapshot. Each required proxy exposes seven timeframes with only 30 recent OHLC candles per timeframe; none carries bid and ask sides on every candle.
- The inherited decision-time evidence remains 0/251 for slippage/fee/financing, 0/251 for margin/exposure/concurrency, and 0/251 for exit/unwind. No missing value was converted to zero or sufficient.

## Mechanical decision

Six mandatory admission gates fail:

1. historical first-published actuals with provider receipt times;
2. prerelease consensus with observation times;
3. revision-safe append-only release lineage;
4. synchronized cross-asset event history;
5. executable side evidence for the cross-asset reaction path;
6. strict decision-time cost, margin and unwind coverage.

The replay was therefore not started. Rolling-30-day multiple, TRAIN LCB, validation paired LCB, PF, DD and margin are all `null`, not zero.

## Dominant blocker and next action

The dominant blocker is the absence of a causally timestamped, revision-safe macro release ledger bound to synchronized executable cross-asset and FX evidence. `ACQUISITION_CONTRACT_V1.json` freezes the exact missing schemas, coverage, hashes and prohibitions. Its execution requires a separately authorized acquisition phase; this checkpoint performed no network fetch, broker connection, live/Paper/order/deploy action, or holdout read.

## Reproduction

```bash
python3 research/monthly_2x_direct_proof_v1/event_driven_cross_asset_v1/build_system_admission.py --check research/monthly_2x_direct_proof_v1/event_driven_cross_asset_v1/SYSTEM_ADMISSION_V1.json
python3 research/monthly_2x_direct_proof_v1/event_driven_cross_asset_v1/verify_system_admission.py
python3 -m unittest research/monthly_2x_direct_proof_v1/event_driven_cross_asset_v1/test_system_admission.py
```

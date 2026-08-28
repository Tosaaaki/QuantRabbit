# OANDA LIVE Historical M5 BID/ASK Capture V1

This directory contains a bounded, read-only historical input collector. It is
separate from the real-time shadow ledger and from all forward PnL evidence.

## Frozen contract

- Provider/host: OANDA v20 LIVE candles at `https://api-fxtrade.oanda.com`.
- Universe: `EUR_USD`, `USD_JPY`, `AUD_USD` only.
- Window: exactly 730 days ending on a completed UTC M5 boundary.
- Data: completed `M5`, `price=BA`, complete BID and ASK OHLC.
- Requests: `from`/`to` windows, at most 5,000 candles per GET, with 0.6 seconds between attempts.
- Credentials: the existing approved `oanda_live_feed.load_approved_live_credentials` loader is called once by `capture`; values are never printed or persisted.
- Authority: historical market-data read only, no fallback provider, no broker mutation surface, zero external orders.
- Evidence: `historical_input_only=true`, `forward_pnl_included=false`, and `profit_evidence=false` are sealed in every manifest.

The OANDA `volume` field is stored as a price-count proxy, not as centralized
traded FX volume.

## Owner runbook

Run from the repository root. `plan` neither reads credentials nor connects:

```sh
python3 research/oanda_history_capture/2026-08-28-v1/oanda_history_capture.py plan \
  --output-root research/oanda_history_capture/2026-08-28-v1/runs
```

Capture uses the plan's completed M5 boundary. If interrupted, rerunning the
same command resumes the single validated `.partial` run. Supplying the exact
`--end-utc` printed by `plan` is recommended for an explicit frozen boundary:

```sh
python3 research/oanda_history_capture/2026-08-28-v1/oanda_history_capture.py capture \
  --output-root research/oanda_history_capture/2026-08-28-v1/runs \
  --end-utc '<to_utc from plan>'
```

After publication, verify without reading credentials or connecting:

```sh
python3 research/oanda_history_capture/2026-08-28-v1/oanda_history_capture.py verify \
  --output-root research/oanda_history_capture/2026-08-28-v1/runs \
  --run-id '<run_id from capture>'
```

## Output and recovery

Each run is content-addressed by its frozen plan. Window data and receipts are
written under `<run_id>.partial`; every completed window is independently
validated and hash-receipted, so a retry does not re-download it. Publication
builds canonical, UTC-sorted, deduplicated JSONL, then atomically renames the
validated `publish` directory to `<run_id>` and makes that tree read-only.

The final run contains:

- `data/<instrument>_M5_BA.jsonl`: canonical uncompressed rows and SHA-256.
- `window_receipts.jsonl`: chained per-window response/content receipts.
- `gap_report.json`: weekend closure, known holiday, and unexplained weekday gaps kept separate; no missing price is synthesized.
- `manifest.json`: complete file hashes, canonical dataset hash, request count, authority/evidence boundaries, and gap status.

`runs/run_receipts.jsonl` is a separately locked append-only run receipt chain.
An already published run is verified and returned idempotently; its bytes are
never rewritten. A failed `.partial` run remains available for bounded resume.

## Admission boundary

The output can be used only as historical training/development input. It does
not prove profitability, does not count as future shadow evidence, cannot open
a holdout by itself, and cannot authorize any external order.

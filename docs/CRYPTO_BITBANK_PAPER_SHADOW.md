# QuantRabbit Crypto｜bitbank Paper Shadow Operations

## Runtime boundary

Spot and Margin run as separate launchd processes and use separate ledgers:

- Spot: `data/crypto/paper-shadow/spot/`
- Margin: `data/crypto/paper-shadow/margin/`
- Reporter: `data/crypto/paper-shadow/reporting_state.json`

Every process starts with:

- `NO_EXECUTE=true`
- `CRYPTO_LIVE_READY=false`
- `WITHDRAWAL_ENABLED=false`
- `CRYPTO_ORDER_AUTHORITY=NONE`

The services use bitbank Public REST/Public Stream only. They have no order,
cancel, settlement, withdrawal, or API-permission mutation method.

Install and start:

```bash
scripts/install-crypto-paper-shadow.sh --check
scripts/install-crypto-paper-shadow.sh install
scripts/install-crypto-paper-shadow.sh status
```

Stop without deleting runtime evidence:

```bash
scripts/install-crypto-paper-shadow.sh stop
```

## Process isolation

Each trading process holds a non-blocking OS file lock for its mode. A second
process for the same mode exits with `PAPER_SHADOW_ALREADY_RUNNING`.

The service writes an atomic `state.json` heartbeat every five seconds. It
contains PID, start time, run ID, event/decision/fill counts, Guardian state,
metrics, outbox state, safety flags, and stop conditions.

## One-trade-one-row outbox

`PaperEngine` records a completed round trip in the hash-chained SQLite ledger,
then places the minimal trade row onto an in-memory queue. A dedicated thread
appends and fsyncs JSONL outside the market-decision path. On restart it
recovers any missing outbox row from `PAPER_TRADE_CLOSED` ledger events.

Rows are deduplicated by a deterministic operation ID and include:

`trade_id`, `run_id`, mode, pair, side, open/close timestamps, entry/exit,
quantity, notional, gross PnL, fees, spread/adverse costs, funding/interest,
net PnL, holding time, exit reason, strategy, regime, Guardian, and ledger
hashes.

## External reporting

The reporter is a separate launchd process. It never runs in the trading
process and never posts one Slack message per trade.

- Sheets trade ledger: one completed trade per row
- Sheets summary ledger: completed-hour and completed-day aggregates
- Slack: one completed-hour summary and one completed-day summary only
- Append/readback uses operation IDs for retry and duplicate prevention

When a Sheets or Slack connector is unavailable, delivery remains pending in
the local outbox. The trading services continue without waiting for it.

## Decision-latency load check

On 2026-07-28, three 50,000-decision trials were compared while the background
outbox thread appended and fsynced 200 completed-trade rows:

- Baseline median p95: `25.834 μs`
- Loaded median p95: `18.792 μs`
- Delta: `-7.042 μs`
- Outbox rows flushed: `200 / 200`

The test found no decision-latency regression. This is a local implementation
benchmark, not evidence of trading profitability or exchange-side latency.

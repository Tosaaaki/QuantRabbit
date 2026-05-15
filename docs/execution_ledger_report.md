# Execution Ledger Report

- Generated at UTC: `2026-05-14T06:34:17.169350+00:00`
- Status: `SYNCED`
- DB: `/Users/tossaki/App/QuantRabbit/data/execution_ledger.db`
- Transactions seen: `876`
- Transactions inserted: `876`
- Events inserted: `876`
- Gateway receipts inserted: `0`
- Baseline transaction id: `None`
- Last transaction id: `471126`

## Contract

- OANDA transactions are broker truth; the database is an append-only local audit index.
- Unknown transaction types are stored raw and recorded as generic `OANDA_TRANSACTION` events.
- First run without `--since-transaction-id` baselines at the current broker `lastTransactionID`; use an explicit since id for historical backfill.

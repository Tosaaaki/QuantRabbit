# Execution Ledger Report

- Generated at UTC: `2026-07-06T01:57:35.648394+00:00`
- Status: `SYNCED`
- DB: `/Users/tossaki/App/QuantRabbit/data/execution_ledger.db`
- Transactions seen: `0`
- Transactions inserted: `0`
- Events inserted: `0`
- Gateway receipts inserted: `0`
- Reconciled gateway events inserted: `0`
- Baseline transaction id: `None`
- Last transaction id: `472994`

## Contract

- OANDA transactions are broker truth; the database is an append-only local audit index.
- Unknown transaction types are stored raw and recorded as generic `OANDA_TRANSACTION` events.
- First run without `--since-transaction-id` baselines at the current broker `lastTransactionID`; use an explicit since id for historical backfill.

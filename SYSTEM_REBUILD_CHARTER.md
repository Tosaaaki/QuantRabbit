# System Rebuild Charter

## Goal

Build a profitable discretionary FX execution system from a small, testable core. The new system must make it harder to lose money by accident than to place a trade.

## Non-Negotiable Execution Contract

1. Broker access has one gateway. No script places, modifies, closes, or syncs OANDA risk outside that gateway.
2. Default mode is read-only. Live execution requires an explicit `LIVE_ENABLED` switch and a current broker-state receipt.
3. Every live order must know entry, TP, SL, worst-case JPY loss, reward/risk, spread, and owner before it reaches OANDA.
4. Helper bypass is impossible. Manual/tagless or broker-synced exposure is treated as external risk and blocks new entries until adopted or closed.
5. Runtime automation cannot create strategy. It can only schedule observation, reconcile broker truth, and call the execution gateway.
6. Learning memory is advisory. It cannot silently force entries, suppress exits, or resize trades.
7. Every order intent must explain the current market story: regime, narrative, chart story, selected method, and invalidation.
8. The daily 10% campaign is mandatory as a target ledger, but it is not permission to force weak trades or exceed the risk gateway.
9. Multiple trader desks can disagree; Portfolio Director must record which desk wins and why.
10. A rebuild component is accepted only with a failing test, a passing test, and a live-risk failure mode documented.

## vNext Shape

- `broker/`: OANDA read/write gateway and risk model.
- `state/`: append-only broker truth snapshots and position lifecycle records.
- `analysis/`: read-only market, news, and technical evidence.
- `strategy/`: trade thesis generation and scoring, with no broker credentials.
- `execution/`: order intent validation and gateway calls.
- `ops/`: schedulers, health checks, and reporting.

## First Build Target

Build the broker gateway first:

- read account, open trades, pending orders, and prices;
- compute JPY risk for every open and proposed trade;
- reject unknown owner, missing protection, excessive loss, poor reward/risk, and stale prices;
- expose a dry-run order-intent API before any live send path exists.

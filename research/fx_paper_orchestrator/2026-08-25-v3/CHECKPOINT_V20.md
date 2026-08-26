# Safe checkpoint after V20

- Local authority remains paper-only. Broker, account, credentials, order endpoints, live orders, commit, push, and deploy were not used.
- V4 through V20 are replayed and content-addressed. V20 was executed by the restart-safe orchestrator and rejected.
- V19 is the strongest aggregate adverse result in the current family: 1.0019739965 over the two-month walk-forward, but May is 0.9996684947 and the monthly 2x target is not met.
- V20's tuning-only median positive network-alignment gate removed the gross edge: RAW 0.9995199594, base 0.9951162289, adverse 0.9907541180.
- The V18/V19/V20 currency-network entry family stops here. No walk-forward threshold retuning is authorized.
- Old V7-V11 `_001` boundary-noncomparable results are preserved and explicitly superseded by `_002` in `EVIDENCE_INVALIDATION_V7_V11_001.json`.
- The next allowed research change is a new signal family, not leverage, execution assumptions, or an opened-period threshold change.
- Future untouched holdout and resident app heartbeat remain unresolved and are not represented as complete.

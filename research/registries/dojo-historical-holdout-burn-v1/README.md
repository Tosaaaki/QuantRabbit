# DOJO historical holdout burn registry v1

`events/` is a first-write, hash-chained local registry for historical outcome
windows that were already exposed or are reserved relative to one declared
selection lineage.

The initial migration conservatively burns the broad 2020–2026 M5 study, the
exact S5 study, and the W54/W55 AI packet windows.  A `MARKET_PRICE_PATH` burn
also conflicts with any derived entry or exit outcome for the same instrument
and overlapping half-open interval; changing granularity, corpus, prompt, or
candidate name cannot revive it.

This local registry is discipline, not external proof.  Every event keeps
proof, promotion, live permission, and broker mutation disabled.  Use
`scripts/run-dojo-holdout-registry.py status` to verify and derive state.

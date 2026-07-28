# DOJO v2 fast replay preflight

Decision: `NO_GO_CURRENT_GENERATION_REQUIRES_NEW_SPARSE_SAFE_SHARED_TRANSCRIPT`.

No yearly replay or AI review was started. The preflight stopped before source
execution, result creation, or model calls. This is a time-budget and evidence
contract NO-GO, not a profitability result.

## Measured blocker

The immutable r13 2025-01 M5 OHLC Bot-only job completed 12 coordinates in
approximately 7h46m and produced 3,556,930,982 bytes of economic transcript.
Its 6,336 source rows became 25,344 quote batches. A single old-format month
already exceeds the complete v2 budget of four hours, so running a year through
that path would knowingly violate the pilot stop rule.

The source is an availability-preserving sparse slice. The existing compact
transcript cannot encode that sparse availability and refuses it. In addition,
the immutable r13 tuned runtime seal differs from the current
generation/config/source bytes. The attempted compact preflight exited 2 with
no result or evidence artifact:

`tuned runtime seal differs from its generation/config/source denominator`

This prevents old evidence from being silently reinterpreted by new code.

## Change made

`scripts/run-dojo-long-horizon-economic-job.py` now exposes the existing
`V3_COMPACT_SEGMENTS` research mode as an explicit CLI choice while retaining
`V1_JSONL` as the default official evidence format. Runtime/config/source
validation failures now return a bounded exit-code-2 error rather than an
uncaught traceback.

This does not make the current sparse source compatible with compact evidence
and does not authorize a yearly run. A new sealed generation must first store
the sparse quote stream once and bind only per-account decisions and economics
to that source chain. It must benchmark one month, project the full year under
four hours, and preserve deterministic replay before the one-time yearly pilot
may start.

## Preserved state

- `qr-dojo-fresh-model-executor-v1`: PAUSED.
- accepted cells: 8/84, unchanged.
- cell 9: undecided.
- model calls for this preflight: 0.
- Paper hourly supervisor: unchanged and independent.
- live permission: false.
- broker mutation: false.
- order authority: NONE.

# Autonomous Shadow Nervous System

## Outcome and boundary

The system advances a shadow trading episode without human approval in the
normal path. A human is an `ASSIST` source: they may add evidence and may engage
the emergency kill switch, but they cannot approve a transition or grant live
permission.

This component has no broker adapter. Every event fixes
`execution_authority=NONE`, `shadow_only=true`, `broker_mutation_allowed=false`,
and `manual_tagless_policy=NO_TOUCH`. It cannot express order side, entry, exit,
TP/SL, size, capital allocation, or a live permission.

## Worker joints and state flow

```text
IDLE / prior terminal
  -> perception -> SIGNAL
  -> hypothesis -> HYPOTHESIS
  -> critic -> CHALLENGED
  -> admission -> ADMITTED
  -> fill_truth -> FILLED -> lifecycle -> OPEN -> exit -> EXITED
                \-> UNFILLED
  -> learning -> LEARNED
```

Each worker owns one transition. `ADVANCE`, `WAIT`, `BLOCK`, `EXPIRE`, and the
fill-truth-only `NO_FILL` are the only verdicts. `NO_FILL` routes directly to
the learning joint without inventing a virtual fill or open lifecycle. `WAIT`
leaves the current state unchanged, `BLOCK` engages a fail-closed terminal
state, and `EXPIRE` closes a candidate rejected by the deterministic guard. The
critic and admission joints cannot advance unless counterevidence was reviewed.

The arbiter also converts an `ADVANCE` into `WAIT` when
`confidence - uncertainty` is below the configured floor. That makes
uncertainty a first-class stop condition instead of a comment.

## Synapse ledger

Every transition is a content-addressed, hash-chained JSONL event. The ledger is
held under an exclusive file lock from read through append, flushed, and
`fsync`-ed. On every cycle the full chain, state continuity, worker ownership,
authority invariants, and human-assist invariants are revalidated before a new
event is accepted.

Worker receipts carry stable `decision_id` values. Replaying the same receipt
is a no-op; reusing its identity with changed content fails closed. A completed
cycle replay is also a no-op. A new cycle may begin after `LEARNED` or
`EXPIRED`, but not while another cycle is active or after the kill switch has
placed the ledger in `BLOCKED`.

The resident adapter uses one ledger per immutable `signal_id`. This permits
many signals to wait independently for S5 maturity and learning evidence; a
slow episode cannot block newer signals. An aggregate state and report expose
the counts in each state without merging or rewriting episode history.

## Resident data flow

```text
fast_bot_shadow_ledger.jsonl
  -> perception + hypothesis
fast_bot_shock_guard_decision_ledger.jsonl
  -> critic + admission
fast_bot_outcome_ledger.jsonl
  -> fill_truth + lifecycle + exit
fast_bot_learning_episode_ledger.jsonl
  -> learning
  -> per-signal hash-chained synapse ledger
  -> aggregate autonomous-shadow state/report
```

`tools/owner_forward_shadow_runtime.py` invokes
`tools/run_autonomous_shadow_nervous_system.py` immediately after exact-S5
outcome resolution and before costlier evaluation jobs. Versioned learning
created later in a pass is consumed on the next pass. A slow downstream
knowledge aggregation therefore cannot starve perception through exit. The
adapter only reads source ledgers and writes its own cohort-local state. It
imports no broker client, performs no HTTP operation, and invokes no gateway.

Source rows are validated before any episode append: content seals, unique
identities, signal/outcome/learning bindings, guard dimensions, S5 truth
coverage, lifecycle consistency, and zero-authority fields all fail closed.
The resident source bundle includes the kernel, adapter, and runner so a
released process cannot silently run different nervous-system bytes.

## Input contract

`qr-vnext autonomous-shadow-cycle` reads one JSON object with:

- `cycle_id`: stable episode identity.
- `decisions`: zero or one receipt per worker. Missing next-worker evidence is a
  normal wait and does not require human action.
- `human_assist`: optional notes and evidence references; evidence-only.
- `kill_switch`: optional boolean, default `false`.

Each decision contains `decision_id`, worker, verdict, reason, observation and
expiry clocks, supporting and contradicting evidence references,
`counterevidence_reviewed`, confidence, and uncertainty. Unknown fields fail
closed, preventing order-shaped payloads from being smuggled through this
coordination channel.

## Reliability and operating tradeoffs

- The append-only ledger favors auditability and deterministic recovery over
  in-place correction. Corrupt history stops the cycle before append.
- File locking serializes writers on one host. Multi-host operation would need
  a transactional shared ledger with the same compare-and-append semantics.
- A `BLOCKED` ledger does not auto-reset. Emergency recovery needs a separately
  reviewed recovery contract; routine cycles do not.
- The resident adapter advances at most 128 unfinished signals per pass. A new
  cohort normally has no backlog; a historical import drains deterministically
  over multiple 30-second passes and reports `DRAINING_BACKLOG` until complete.
- Missing guard, outcome, or learning evidence is a normal autonomous wait. It
  never becomes a request for human approval.

## Revisit points

Revisit the 128-signal work bound or single-host file locking only if measured
resident latency/backlog requires it. Live execution remains a separate
authority and promotion decision; it must not be inferred from a completed
shadow cycle.

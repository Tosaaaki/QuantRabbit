# DOJO paired model executor

## Purpose

This executor is a research-only Codex decision handoff for the immutable r13
paired inventory study. It must never connect to a broker, live runtime, order
gateway, or model API from repository code.

The queue exposes exactly one unseen content hash at a time. A fresh Codex task
may read only:

- `docs/AGENT_CONTRACT.md`
- this executor contract
- the one JSON returned by `show-ready`
- the static action allowlist embedded in that packet

It must not open the source paired results, later packet inventory, terminal
result, future quote, answer key, prior model response, or conversation history
containing outcomes.

## Queue

Canonical research queue:

`research/training/dojo-r13-paired-model-queue-20260727`

Use the combined preflight. It checks the quota sentinel before opening queue
state and returns either zero work or the one current causal packet:

```bash
PYTHONPATH=src python3 scripts/run-dojo-paired-model-queue.py preflight \
  --queue-dir research/training/dojo-r13-paired-model-queue-20260727
```

If `zero_work=true`, return `DONT_NOTIFY` without invoking a model decision.
The packet, causal-data boundary, action allowlist, hash, authority, and
future/terminal-data prohibitions are unchanged from the original
`status`/`show-ready` flow.

Choose exactly one allowlisted inventory-supervision action using only fields
inside that packet. Never invent an entry, pair, side, units, order type, TP,
SL, price, capital allocation, live permission, or broker mutation.

Seal/reuse the exact response bytes, submit, and verify in one command:

```bash
PYTHONPATH=src python3 scripts/run-dojo-paired-model-queue.py complete-cell \
  --queue-dir research/training/dojo-r13-paired-model-queue-20260727 \
  --action <ALLOWLISTED_ACTION> \
  --reason-id <BOUNDED_REASON_ID> \
  --provider-model <CURRENT_CODEX_MODEL> \
  --provider-execution-id <FRESH_TASK_EXECUTION_ID> \
  --response <EXCLUSIVE_RESPONSE_PATH>
```

One successful submission atomically appends the accepted event and publishes
the next cell hash. Duplicate response bytes are idempotent. Conflicting bytes,
future/terminal/wall-clock claims, non-allowlisted actions, authority changes,
or a broken event chain fail closed.

The intended model/tool orchestration is therefore two repository calls:
`preflight`, then `complete-cell`. Context-source reads required by the active
operating contract still apply, but should be batched. The model remains
`gpt-5.6-sol` with medium reasoning and every cell remains one fresh task.

## Quota halt and resume

On a reported usage hard limit, 429, rate-limit exhaustion, or inability to
continue a model task, persist the first failure before another decision:

```bash
PYTHONPATH=src python3 scripts/run-dojo-paired-model-queue.py halt-quota \
  --queue-dir <QUEUE_DIR> \
  --reason <BOUNDED_REASON> \
  --observed-at-utc <AWARE_UTC_TIMESTAMP>
```

The immutable, hash-sealed `runtime-quota-halt.json` records
`HALTED_QUOTA`/`PAUSE_REQUESTED`, reason, time, last accepted count, and the
same ready packet. Repeated preflight calls are zero-work and do not open the
packet. Accepted response/events are never changed. A response sealed before a
mid-run halt remains unaccepted and byte-identical; after a submit boundary,
only verification of the accepted bytes is needed.

Only an explicit operator resume may remove the sentinel:

```bash
PYTHONPATH=src python3 scripts/run-dojo-paired-model-queue.py resume-quota \
  --queue-dir <QUEUE_DIR>
```

Resume fails closed if accepted count or ready packet changed while halted.
When the platform cannot start the task at all, no repository command runs and
local queue state naturally remains unchanged. Runtime code must never edit an
Automation TOML/SQLite database; supported Automation tooling may pause the
task, while the sentinel remains the queue-level safety boundary.

## Economic application

After the queue is complete, rerun the immutable transcripts with
`run-dojo-paired-inventory-counterfactual.py run --queue-dir <QUEUE_DIR>`.
The adapter verifies all 84 packet/response hashes and applies the accepted
action only at its matching source decision id/state-packet hash. Terminal
economic reduction then uses
`reduce-dojo-paired-model-economics.py`.

The reducer reports TP gross profit, other exit gross profit, ordinary loss,
forced-margin loss, missed profit, execution cost, financing, AI execution
cost, additional reduction required for a positive objective, and the count of
profitable positions cut by AI. If AI cost is unavailable, profitability stays
`UNDETERMINED_AI_COST_MISSING`. These are worn TRAIN diagnostics only and
never promotion or live permission.

## Evidence tier

This V1 proves the queue/executor handoff only. Its accepted responses are
content-addressed and locally verified, but are explicitly
`SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC`; no approved model-response signing
authority is enrolled. The response is not yet applied to an economic reducer
checkpoint, so it must not change the existing 84-cell results, ranking, or
promotion status.

The first accepted queue response was created inside the already-running owner
task and is only a pipeline smoke test. Because that task had prior project
context, it is not a fresh-task checkpoint and the formal fresh-model count
remains 0/84.

Completion of the full Phase B loop requires both:

1. an approved provider/model-response attestation authority or provider-native
   execution receipt, with local verification; and
2. exact reducer checkpoint resume that applies each accepted action and seals
   the resulting economic cell before advancing its fixed denominator.

Paper/replay only, `live_permission=false`,
`broker_mutation_allowed=false`, and `order_authority=NONE` are invariant.

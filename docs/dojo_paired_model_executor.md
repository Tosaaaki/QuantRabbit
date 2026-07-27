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

Read status:

```bash
PYTHONPATH=src python3 scripts/run-dojo-paired-model-queue.py status \
  --queue-dir research/training/dojo-r13-paired-model-queue-20260727
```

If status is not `WAITING_FOR_MODEL`, return without invoking a model decision.
If the ready hash has already been accepted, return without spending another
model call.

Read the one ready packet:

```bash
PYTHONPATH=src python3 scripts/run-dojo-paired-model-queue.py show-ready \
  --queue-dir research/training/dojo-r13-paired-model-queue-20260727
```

Choose exactly one allowlisted inventory-supervision action using only fields
inside that packet. Never invent an entry, pair, side, units, order type, TP,
SL, price, capital allocation, live permission, or broker mutation.

Seal the response:

```bash
PYTHONPATH=src python3 scripts/run-dojo-paired-model-queue.py seal-response \
  --queue-dir research/training/dojo-r13-paired-model-queue-20260727 \
  --action <ALLOWLISTED_ACTION> \
  --reason-id <BOUNDED_REASON_ID> \
  --provider-model <CURRENT_CODEX_MODEL> \
  --provider-execution-id <FRESH_TASK_EXECUTION_ID> \
  --output <UNUSED_EXCLUSIVE_RESPONSE_PATH>
```

Submit and verify:

```bash
PYTHONPATH=src python3 scripts/run-dojo-paired-model-queue.py submit-response \
  --queue-dir research/training/dojo-r13-paired-model-queue-20260727 \
  --response <EXCLUSIVE_RESPONSE_PATH>

PYTHONPATH=src python3 scripts/run-dojo-paired-model-queue.py verify \
  --queue-dir research/training/dojo-r13-paired-model-queue-20260727
```

One successful submission atomically appends the accepted event and publishes
the next cell hash. Duplicate response bytes are idempotent. Conflicting bytes,
future/terminal/wall-clock claims, non-allowlisted actions, authority changes,
or a broken event chain fail closed.

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

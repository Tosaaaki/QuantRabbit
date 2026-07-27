# DOJO fresh Codex model handoff

## Outcome

`dojo_fresh_model_handoff.py` is the credential-free local boundary between
paper/replay state and a fresh Codex task. It does not call a model, import a
broker client, start a process, or mutate an account.

The old hourly sidecar cannot be treated as this executor. It is a direct
`gpt-4o-mini` Responses API caller whose missing environment file currently
causes a fail-closed exit. A replacement Codex Automation must be created or
updated only through `automation_update`, followed by the active inventory
sync runbook. Hand-editing Automation TOML is not an approved fallback.

## Local deterministic flow

1. A paper/replay compiler emits a source packet with safety authority `NONE`.
2. `compile` removes volatile append-time fields and seals the decision state.
3. Market-closed flat state, open-market flat idle state, and an unchanged
   accepted state produce a content-addressed zero-token skip.
4. An exposed or materially changed state publishes exactly one ready packet.
5. A fresh Codex task reads only:
   - current snapshot;
   - previous decision summary;
   - current bounded rolling story;
   - at most 12 recent causal events;
   - static action and safety contract.
6. The response verifier rejects future/terminal/wall-clock use, non-fresh task
   claims, broker/order actions, malformed story content, and authority drift.
7. Acceptance advances the hash-chained story and clears the ready packet.
8. A changed source state can publish the next packet; identical state skips.

The model response remains `SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC`: content seals
are verified, but no approved cryptographic response authority is configured.
It cannot be ranked, promoted, or applied to a live/paper account as a command.

## Bounded rolling story

The story is a current-state ledger, not conversation history. Each immutable
record contains exactly:

- `current_thesis`
- `macro_regime`
- `micro_regime`
- `evidence_for`
- `evidence_against`
- `inventory_risk`
- `last_action`
- `expected_outcome`
- `invalidation_conditions`
- `next_review`
- `confidence`
- `known_unknowns`

Each list has at most eight short entries. The chain stores the prior story
hash and sequence, while a fresh task receives only the current story.

## Cadence and no-token gate

- normal: 60 minutes;
- deterministic high-risk signal: 15 minutes;
- major event: immediate only when the local compiler is invoked.

Dynamic Automation wake is not claimed. Until a supported dynamic wake surface
exists, use a low-frequency heartbeat and content-hash/idempotency gate.
Immediate hard risk protection remains local deterministic Python and is never
delegated to the model.

`compile-rooms` is the preferred paper compiler. It reads the active rooms'
local `session_contract.json`, `broker_snapshot.json`, and `state.json`
directly. It does not load an OpenAI/OANDA environment file, instantiate a
broker client, or make a network request. It rejects stale room/quote state,
invalid paper authority, crossed quotes, and malformed inventory before a
ready packet can exist.

## CLI

```bash
PYTHONPATH=src:. python3 scripts/run-dojo-fresh-model-handoff.py init \
  --root <handoff-root>

PYTHONPATH=src:. python3 scripts/run-dojo-fresh-model-handoff.py compile \
  --root <handoff-root> \
  --source-packet <paper-packet.json>

PYTHONPATH=src:. python3 scripts/run-dojo-fresh-model-handoff.py compile-rooms \
  --root <handoff-root> \
  --rooms-root <paper-rooms-root>

PYTHONPATH=src:. python3 scripts/run-dojo-fresh-model-handoff.py show-ready \
  --root <handoff-root>

PYTHONPATH=src:. python3 scripts/run-dojo-fresh-model-handoff.py seal-response \
  --root <handoff-root> \
  --action HOLD \
  --reason-id CAUSAL_REASON \
  --next-story <next-story.json> \
  --provider-model <model> \
  --provider-execution-id <fresh-task-id> \
  --output <response.json>

PYTHONPATH=src:. python3 scripts/run-dojo-fresh-model-handoff.py \
  submit-response --root <handoff-root> --response <response.json>

PYTHONPATH=src:. python3 scripts/run-dojo-fresh-model-handoff.py verify \
  --root <handoff-root>
```

## Safe replacement sequence

When `automation_update` is available:

1. create/update the fresh-task Automation through that tool;
2. point it at the ready-packet contract, not the old OpenAI env file;
3. prove one accepted fresh response and next-state transition;
4. pause/remove the failing legacy sidecar without stopping any bot;
5. run Automation inventory sync `dry-run -> apply -> dry-run`;
6. require `residual_changes=0` and sync the Notion operation map.

Rollback is to pause the new Automation through `automation_update`. The local
handoff and story artifacts are append-only evidence and need not be deleted.

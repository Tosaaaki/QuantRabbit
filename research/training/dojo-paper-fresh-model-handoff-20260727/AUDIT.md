# DOJO paper fresh-model handoff audit

Observed at 2026-07-27 11:10 JST by a read-only packet build:

- four paper rooms were fresh;
- two rooms had one open virtual position each;
- no room had a resting virtual order at that observation;
- `paper_only=true`, `shadow_only=true`, `live_permission=false`, and
  `order_authority=NONE`;
- the source build used zero model tokens and no broker mutation.

The handoff published one packet:

- packet SHA-256:
  `cb6850f95f0496323255285e80183263a879848b115a48235b9c9da871ffa6c0`
- decision-state SHA-256:
  `4d69bf39a74930f4dd9440111be536081fca233f9c501a7e8277ea23edada772`
- cadence: `NORMAL_60M`
- rolling story sequence: `0`
- state: `WAITING_FOR_FRESH_TASK`

No model response is accepted. A no-history, read-only `codex exec` attempt
was isolated to this exact packet, but local Codex CLI 0.25 could not use the
current app model/account route:

1. `gpt-5.6-sol` required a newer Codex CLI.
2. `gpt-5-codex` was not supported on the active ChatGPT account route.
3. `gpt-5` was not supported on the active ChatGPT account route.

Every attempt ended at provider HTTP 400 before a model response was returned.
The ready packet, event chain, story sequence, and accepted-decision count
therefore remain unchanged. No response was fabricated from the long-running
owner task.

Current external completion gate:

- run this ready packet from a supported fresh Codex task/Automation surface;
- return a response satisfying `QR_DOJO_FRESH_MODEL_RESPONSE_V1`;
- submit it through the local verifier;
- compile a changed snapshot and prove the next packet, story continuity,
  unchanged-state skip, and idempotent duplicate handling.

Automation truth:

- `automation_update` is not callable in the current execution surface, so no
  Codex Automation TOML was created, edited, paused, or resumed;
- inventory sync was not run because there was no automation mutation;
- dynamic event wake is not claimed. The plan truthfully declares
  low-frequency heartbeat plus content-hash idempotency.

Legacy shadow truth:

- `com.quantrabbit.dojo-paper-ai-shadow-hourly` is separate from the four bots,
  is not currently running, and its latest five market-open attempts exited 2;
- its latest status is `FAIL_CLOSED` because the configured OpenAI credential
  file is missing;
- it directly calls the Responses API with `gpt-4o-mini` and has no accepted
  decision ledger entries;
- it was not stopped, modified, or replaced because the approved Automation
  update/sync surface is unavailable. The four bot processes were not touched.

## Direct local compiler follow-up

At 2026-07-27 11:32 JST, the integrated `compile-rooms` path was verified
against the same active local room root in a temporary handoff store:

- active rooms: 4;
- open virtual positions: 2;
- resting virtual orders: 2;
- market feature source: `LOCAL_PAPER_ROOM_STATE`;
- provider/model credentials used: false;
- network/broker client used: false;
- cadence: `NORMAL_60M`;
- temporary ready packet SHA-256:
  `21f861f6cfd7b7210f18eba2d8385fa7906bad41ae51f0d1e335852195d5057f`;
- accepted fresh-model responses: 0.

This path removes the new executor's dependency on the legacy shadow's missing
OpenAI environment file. The committed evidence queue above intentionally
keeps its already-published ready packet immutable; it was not overwritten by
the later observation.

The handed-off draft `dojo_paper_codex_supervisor.py` remains protected and
unmodified in its separate worktree. Its direct room normalization and risk
banding informed `compile-rooms`, but the full draft was not adopted because
it could skip closed-market inventory, overwrite a pending review with a new
state, omitted the required 12-field story, and did not bind responses to a
fresh Codex execution identity and causal flags.

## Separate paper-shadow candidate reconciliation

A later four-file candidate in
`/Users/tossaki/App/QuantRabbit-worktrees/paper-ai-codex-automation` was
reviewed read-only. Its temporary real-room proof contained:

- four rooms, two positions, and two orders;
- a 10,323-byte canonical packet;
- one `BLOCK_LONG` shadow record followed by an idempotent `DUPLICATE`;
- `paper_only=true`, `live_permission=false`, and `order_authority=NONE`.

That record is valid evidence for the candidate's local validator, but it is
not counted as a formal fresh-model checkpoint because the response schema
does not bind a fresh Codex execution identity or causal flags. The candidate
also skips closed-market inventory unconditionally and advances its review
clock when a packet is published rather than when a response is accepted.
Adopting it wholesale would create a second source of truth.

The useful bounded room-ledger input was instead integrated into the owner
implementation. `compile-rooms` now takes at most four event candidates per
room, publishes only the latest 12 across rooms, and rejects any event later
than the source cutoff. A real-room probe produced:

- ready packet
  `9081e7e45a30bed12d825371a1d4932b5d025afd782bbed49a665c5ae73a0f51`;
- 12 bounded recent events;
- four rooms, two positions, and two orders;
- `NORMAL_60M`;
- zero accepted fresh-model responses.

Focused tests passed 23/23. The full repository suite passed 6,759 tests with
one skipped test and two pre-existing collection warnings.

The legacy `com.quantrabbit.dojo-paper-ai-shadow-hourly` service was disabled
and booted out by the separate migration task. This owner task verified the
disabled flag and absence of a loaded service read-only; it did not mutate
launchd. The plist remains available for forensic rollback.

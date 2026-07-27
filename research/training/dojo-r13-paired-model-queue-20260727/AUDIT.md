# r13 paired Codex model queue audit

Checked at: 2026-07-27 10:42:13 JST

## Before implementation

- Frozen paired replay: automatic for 84/84 coordinate-cadence cells.
- Actual provider/model checkpoint decisions: 0/84.
- Decision packet queue: absent.
- Response verifier and idempotency cache: absent.
- Automatic transition after a model response: absent.
- Codex automation executor: absent.
- `dojo-historical-heartbeat`: active but still bound to the obsolete
  `dojo-parallel-rooms` worktree/branch and r12 paths.
- `com.quantrabbit.dojo-paper-ai-shadow-hourly`: bound to the obsolete
  `episode-s5-outcome` worktree, not running, last exit 2.
- `com.quantrabbit.dojo-historical-supervisor-r13`: canonical r13 worktree,
  not running, last exit 2. It remains a historical generation supervisor,
  not a paired model executor.

No automation, launchd process, broker process, live runtime, or existing r13
artifact was changed during this audit.

## Implemented

- Fixed 84-cell queue plan:
  `cb1138290d11f28fe5680c0b0237b078149703d15ba478e78b528b3aeaf688c2`
- Exactly one ready packet is published at a time.
- Ready packets contain only the causal state already available through the
  decision epoch; source `post_outcome`, future quote, terminal result, and
  append wall-clock are excluded.
- Accepted responses are content-addressed, action-allowlisted, authority
  checked, and idempotent by decision packet SHA-256.
- Submission appends an immutable acceptance event and automatically publishes
  the next cell packet.
- Idle model execution is forbidden by contract and status.

## Owner-task pipeline smoke (not fresh-task causal evidence)

- Cell 1 ready packet:
  `e50347eb8798f5171786f91307332c9236cef88d670fdd632d54d87711d9bde0`
- Visible causal state: no positions/orders, drawdown 7.2705752%, three
  consecutive losses, zero margin use, compatible/unknown regime.
- The already-running visible owner task selected `PAUSE_NEW_ENTRIES`.
- Response SHA-256:
  `81a85f897aa1c1e05647006caf641aee333e15733ab3d95e824ed2722786b905`
- Future information used: false.
- Terminal result used: false.
- Append wall-clock used: false.
- Content seal verified: true.
- Accepted queue responses: 1/84.
- Automatic next cell: ordinal 2, packet
  `be976bba92cf53fe6c3099f63825ba544e3b25ffe3e073c8d725ea9213801415`.
- Queue event tip:
  `f49f7d0538f384896e4265f6ef68793a9b8fe97f6742f79eb92e3c1c7867e90e`.

This response was produced inside the existing long-running owner task after
other project context had been loaded. It proves packet sealing, verification,
idempotency, and automatic next-cell publication, but it does **not** satisfy
the later fresh-task/no-conversation-history requirement. Formal fresh Codex
checkpoint evidence therefore remains 0/84 and this response must not be used
as causal economic or promotion evidence.

## Fail-closed remainder

- Cryptographic provider signature verified: false. No approved
  model-response signing authority exists; no key was generated or repurposed.
- Economic application status:
  `NOT_YET_APPLIED_PIPELINE_PROOF_ONLY`.
- The accepted action has not been applied to a reducer checkpoint and does
  not update the paired economic result.
- Formal fresh-task actual-model checkpoints remain 0/84.
- The existing paired study remains experimental and UNRANKED.
- The Codex `automation_update` capability was not exposed in this session.
  Automation TOML was not hand-edited. A periodic executor cannot be registered
  until that approved tool is available; therefore the queue is waiting at
  cell 2 without idle model calls.

## Authority

- paper/replay only: true
- live permission: false
- broker mutation allowed: false
- order authority: NONE
- promotion eligible: false

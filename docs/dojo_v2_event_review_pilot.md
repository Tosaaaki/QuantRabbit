# DOJO v2 — bounded event-review pilot

## Status and boundary

This is a design for one time-limited pilot. It does not resume or modify
`qr-dojo-fresh-model-executor-v1`, does not create a new Automation, and does
not decide cell 9. The old 8/84 accepted state remains immutable. The pilot is
Paper/replay-only: `live_permission=false`, `broker_mutation_allowed=false`,
and `order_authority=NONE`.

The end-to-end wall-clock budget is four hours; the target is 3–4 hours. A
watchdog writes the last accepted content hashes and stops all new work at
3h55m. A run crossing four hours is an automatic NO-GO, even if its economics
look attractive.

## Architecture

1. **Bot-only yearly replay (target 150–210 min).** Replay one sealed calendar
   year once. Persist a causal event stream, fills, orders, inventory,
   mark-to-market equity, costs, margin, and terminal settlements. This stage
   contains no AI calls.
2. **Event miner (target 10–15 min).** From the completed replay, identify
   losses, TP giveback, margin pressure, forced close, direction mismatch, and
   delayed-resume episodes. Select matched non-loss controls at approximately
   1:1 by strategy, pair, side, regime, session, volatility bucket, inventory
   bucket, and intervention-time exposure.
3. **Packet sealer (target 5 min).** Rebuild each packet from an event-time
   watermark. The AI-visible bytes contain only information at or before the
   intervention epoch. Outcomes are retained in a separate evaluator-only
   store.
4. **Bounded AI reviewer (target 20–35 min).** Review deduplicated packet
   hashes with bounded parallelism and cache. Allowed actions are `HOLD`,
   `REDUCE`, `PAUSE`, `CLOSE`, `DIRECTION_LIMIT`, and `RESUME`. No arbitrary
   code or parameters may be generated.
5. **Counterfactual simulator and reducer (target 15–25 min).** Apply each
   accepted action to an isolated copy of the portfolio and use the same sealed
   future quote stream as Bot-only. Measure prevented loss, sacrificed upside,
   delayed-resume opportunity cost, execution/financing cost, DD, margin, PF,
   expectancy, and false interventions.
6. **Reproducibility rerun (target 10–15 min).** Re-run packet selection and
   the cached counterfactual reduction. Content hashes and all deterministic
   metrics must match.

The 240-minute hard budget allocates 210 minutes to yearly replay, 15 to mining
and sealing, 35 to AI review, 25 to simulation/reduction, and 15 to
reproducibility. Stages overlap only where inputs are already sealed. No stage
may borrow beyond the global deadline.

## Data contracts

`bot_replay_manifest` seals data version, quote hashes by month, strategy and
cost config hashes, code commit, initial virtual capital, event count, and
runtime.

`episode_private_record` contains the intervention watermark plus evaluator
fields such as terminal outcome, realized loss, MFE/MAE, future quotes, and
matching labels. It is never serialized into an AI request.

`causal_packet` contains:

- packet/episode IDs and hashes;
- `input_available_through_epoch`;
- strategy, pair, side, causal regime/session/ATR features;
- open positions and pending orders at the watermark;
- realized history available at the watermark;
- unrealized P&L, margin and inventory at the watermark;
- spread/slippage/latency observations available at the watermark;
- action allowlist and authority `NONE`.

It must not contain selection reason, loss amount, terminal reason/outcome,
post-watermark MFE/MAE, future quote timestamps or values, evaluator labels, or
filenames/path names that encode an outcome.

`model_response` binds one action and reason IDs to the exact packet hash.
`counterfactual_result` binds packet, response, future-quote and simulator
hashes and records both Bot-only and managed economics. `pilot_summary` reports
development and holdout separately; no account-level sum is allowed across
duplicated episodes.

## Leakage guard

- Mine with outcome data in an evaluator-only process, then reconstruct packets
  from the causal event stream in a process that cannot read the private store.
- Use an explicit schema allowlist. Reject unknown fields.
- Scan serialized packet bytes for evaluator-only field names, future
  timestamps, outcome labels, and private IDs.
- Require `max(source_event_epoch) <= input_available_through_epoch`.
- Bind every feature to source event hashes and verify the prefix Merkle root.
- Derive packet filenames from packet hash only.
- Keep model execution credentials/process unable to read future quote paths.
- Add synthetic sentinel outcomes to the private store and fail if any sentinel
  appears in packet bytes, prompts, logs, or cache keys.

Any structural leakage failure is terminal NO-GO. A review-time instruction
asking for outcome data also fails that episode closed.

## Episode selection and bias

Development episodes intentionally oversample adverse outcomes. They are
diagnostic and cannot estimate unconditional production lift. Matched non-loss
controls expose false intervention and sacrificed upside, but do not erase
outcome-conditioned sampling bias.

Use development months only to define event thresholds, matching rules, action
semantics, and fixed reducer logic. Freeze them before reviewing holdout.
Reserve whole, unused months for holdout; do not split adjacent events from the
same market shock across sets. Require multiple regime buckets in holdout.
Weight final lift back to the Bot-only replay's event incidence and also show
unweighted case/control metrics.

Pilot budget:

- up to 48 loss/high-value episodes and 48 matched controls in development;
- up to 32 high-value episodes and 32 controls in holdout;
- maximum 160 unique AI packets, with duplicate hashes served from cache;
- bounded concurrency 4; one retry only for transport failure using identical
  bytes; no retry after a valid response;
- stop AI review if estimated usage exceeds the pre-sealed pilot budget.

## Metrics and GO/NO-GO

Primary holdout metric is `net_lift_after_cost`. Secondary metrics are PF,
expectancy, max DD, prevented loss, sacrificed upside, false intervention rate,
margin pressure, forced loss, and delayed-resume cost.

GO requires all of:

- end-to-end runtime at most four hours;
- holdout net lift after all execution, financing and AI costs is positive;
- holdout PF and expectancy both improve;
- holdout max DD does not worsen;
- the direction repeats in at least two predeclared regimes;
- sacrificed upside and false interventions do not offset the improvement;
- packet selection, cached responses, simulation and reduction reproduce
  exactly from sealed hashes;
- measured economic value is proportionate to AI usage/cost.

NO-GO is triggered by any of:

- runtime over four hours;
- structural inability to prevent future/outcome leakage;
- holdout net lift after cost at or below zero, or worse DD;
- sacrificed upside/false intervention offsetting the lift;
- AI usage/cost not justified by the measured value;
- non-reproducible selection or economics.

On NO-GO, do not resume the old or proposed DOJO Automation. Preserve the
replay, episode, packet, response, counterfactual, summary, timing, and accepted
8-cell artifacts. Propose archiving the Notion operation-map row while keeping
restoration metadata. Do not delete or archive anything without explicit user
approval.

## Migration from the 8-cell study

Import none of the eight responses as holdout outcomes or production policy.
Use their immutable packet/response hashes only as pipeline regression
fixtures. The seven PAUSE actions suggest loss/giveback episodes worth mining;
the HOLD action and its sacrificed-upside proxy motivate matched non-loss
controls. This avoids continuing 76 low-value sequential calls while preserving
the original evidence and idempotency chain.

Paper champion/challenger is a separate operational research loop. It neither
waits for nor consumes a DOJO v2 result, and no DOJO result grants it live
authority.

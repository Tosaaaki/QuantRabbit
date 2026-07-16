# QuantRabbit AI Supervisor Runtime

`qr-trader` is retained only as the compatibility automation id. It is not an
order trader. This playbook defines one AI regime/tuning supervisor for the
deterministic fast bot.

## Load order

1. Read `docs/AGENT_CONTRACT.md` in full. It is authoritative.
2. Read the current artifacts listed below from the same live-runtime snapshot.
3. Perform either the six-hour periodic review or one material-event review.
4. Publish only sealed pair supervision and bounded tuning reviews.

Do not read or execute the legacy branch prompts under `docs/trader_prompts/` as
scheduled AI instructions. They are compatibility material for retired flows.

## Authority boundary

- `AI_ORDER_AUTHORITY=NONE` is invariant.
- Model policy is `gpt-5.5` with `reasoning_effort=high`.
- Normal cadence is once every six hours.
- An additional review is allowed only when current Guardian evidence reports a
  material regime, volatility, spread/cost, technical-state, or measured
  performance change.
- The only market-state output is `QR_AI_REGIME_SUPERVISION_V1` with bounded
  per-pair `GO`, `CAUTION`, or `STOP` rows.
- The only learning output is an observation-bound Guardian tuning review under
  the experiment contract in `docs/AGENT_CONTRACT.md` §15.
- AI must not create or select an order action, cancel, close, add, harvest, or
  reduce exposure.
- AI must not choose or alter direction, strategy method, vehicle, entry, TP,
  SL, geometry, risk, multiplier, allocation, or units.
- AI must not grant live permission, call OANDA, call `guardian-action-cycle`,
  invoke `AutoTradeCycle`, `LiveOrderGateway`, or `PositionProtectionGateway`,
  or set any live/handoff/action flag.
- AI must not write `data/codex_market_read_overlay.json`, apply a legacy market
  read, create a GPT decision receipt, stage an order, or run a live wrapper.

Legacy decision, allocation, verifier, receipt, and gateway artifacts may be
read for audit or counterevidence only. They are never supervisor output or
execution authority.

## Required current evidence

Read these before every review:

- `data/hierarchical_bot_regime.json`
- `data/fast_bot_shadow.json`
- `data/fast_bot_scorecard.json`
- `docs/fast_bot_shadow_report.md`
- `data/guardian_events.json`
- `data/guardian_escalation.json`
- `data/guardian_tuning_work_order.json`
- `data/qr_trader_run_watchdog.json`
- `data/broker_snapshot.json` for observation context only

The regime and scorecard must be current, sealed, and internally valid. A
container timestamp cannot refresh stale candles, quotes, score rows, or an old
Guardian observation. Missing, malformed, stale, future-dated, unsealed, or
digest-mismatched evidence fails closed: do not fabricate a pair row or tuning
conclusion.

## Fast-bot state

The deterministic fast bot runs in the 30-second Guardian process without a
per-signal AI call. Its finite hierarchy is:

- M1: execution observation
- M5/M15/M30: operating state
- H1/H4: structure
- D: anchor

The bot remains strictly shadow-only. `shadow_only=true`,
`live_permission=false`, and `broker_mutation_allowed=false` are invariants.
AI `GO` does not create a signal and does not approve a trade; it only leaves
the deterministic shadow gate unblocked. `CAUTION` adds supervisor caution.
`STOP` blocks new shadow hypotheses for that pair until expiry. Missing or
expired supervision is `UNSUPERVISED`, never an invented approval.

For schema-v2 shadow evidence, inspect all four precommitted passive-entry arms
on the same exact OANDA S5 bid/ask path. Never reconstruct an arm after outcome,
select a winner from the same cohort, or allow an experiment arm to change the
primary scorecard. Collapsed same-tick arms are not independent evidence.

Primary promotion evidence requires at least 100 valid fills across at least 10
filled active days, profit factor at least 1.25, and a strictly positive
one-sided 95% lower bound from filled-day mean returns. Passing those thresholds
still grants no live permission. A separate content-addressed deterministic
live-promotion implementation, risk sizing, ownership/margin/duplicate fences,
and gateway integration must be reviewed before any broker mutation exists.

## Pair supervision workflow

1. Verify the current regime and scorecard contract hashes.
2. Identify why the review is due: six-hour periodic review or exact material
   Guardian observation.
3. For each pair that genuinely needs supervision, choose only `GO`, `CAUTION`,
   or `STOP`; write one bounded evidence-based reason and an expiry no later
   than six hours after `reviewed_at_utc`.
4. Omit pairs whose evidence is unavailable. Do not fill missing coverage with
   a guess.
5. Write an ignored temporary candidate JSON with exactly this shape:

```json
{
  "contract": "QR_AI_REGIME_SUPERVISION_CANDIDATE_V1",
  "schema_version": 1,
  "reviewed_at_utc": "aware current UTC instant",
  "review_reason": "bounded periodic or material-event reason",
  "regime_contract_sha256": "current sealed regime digest",
  "scorecard_contract_sha256": "current sealed scorecard digest",
  "pairs": {
    "EUR_USD": {
      "mode": "CAUTION",
      "reason": "bounded evidence-based reason",
      "expires_at_utc": "aware UTC instant no more than six hours later"
    }
  }
}
```

6. Seal and publish it only through:

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"
PYTHONPATH=src "$QR_PYTHON" tools/ai_regime_supervision.py \
  --candidate tmp/ai_regime_supervision_candidate.json \
  --regime data/hierarchical_bot_regime.json \
  --scorecard data/fast_bot_scorecard.json \
  --output data/ai_regime_supervision.json
```

The writer rejects unknown fields, forbidden order-authority fields, stale
bindings, unsupported pairs, invalid clocks, and expiry beyond six hours. Never
hand-edit `data/ai_regime_supervision.json`; the writer must atomically seal it.

## Bounded tuning review

- Normalize the tuning queue before review. One pending
  `TECHNICAL_STATE_CHANGE` scope represents one configured pair; fingerprints,
  candle watermarks, quote/spread, direction, and clocks are observations.
- Bind every review to the exact current `work_order_id` and
  `latest_observation_id`.
- Persist only `NO_CHANGE_INSUFFICIENT_EVIDENCE` with one exact non-executing
  acquisition step, or `TEST_REQUIRED` with one allowlisted non-risk parameter
  and one falsifiable forward experiment.
- Use `tools/guardian_tuning_review_enrich.py`; never edit
  `data/guardian_tuning_work_order.json` directly.
- For a bounded backlog, prefer one compare-checked manifest so all current
  reviews commit atomically or none do.
- Historical replay is hypothesis context only. Tuning proof uses the exact
  first forward cohort fixed by §15; it cannot reuse the outcomes that selected
  the candidate.
- A review never activates a parameter, frees a queue slot, grants live
  permission, or changes risk, ownership, geometry, allocation, or units.
- Lifecycle transition and activation remain separate deterministic,
  content-addressed procedures. The AI supervisor does not call execution
  gateways during or after tuning.

## Deterministic Guardian and ownership

Guardian market observation, stale-input detection, risk monitoring, and
eligible system-position protection continue frequently without an AI model.
The Guardian dispatcher is supervisor-only with
`QR_GUARDIAN_WAKE_GATEWAY_HANDOFF=0`, `QR_GUARDIAN_ACTION_EXECUTE=0`,
`QR_LIVE_ENABLED=0`, and `AI_ORDER_AUTHORITY=NONE`. Re-enabling old flags does
not restore AI action-cycle authority.

Manual, operator-owned, tagless, external, or ambiguous-owner positions are
`NO_TOUCH`. The AI supervisor and fast bot may observe them for account context
but must not close, reduce, add, cancel, replace, or modify TP/SL/protection.
Deterministic protection may act only on explicitly eligible system-owned
exposure under its separate gateway and ownership contract.

## Failure handling

- No material change and six-hour review not due: publish nothing.
- Current evidence missing or stale: preserve the previous sealed artifact; do
  not refresh its timestamp or claim a completed review.
- A previous `STOP` may remain active for the code-fixed 15-minute scheduler
  handoff grace; publish the replacement promptly and never treat the grace as
  permission to extend a stale `GO` or `CAUTION`.
- Candidate validation failure: fix the candidate or record the exact blocker;
  do not bypass the sealing tool.
- Queue conflict or stale observation: reread once and defer to the next review
  if identity changed; never overwrite concurrent state.
- Model/runtime/quota failure: preserve the exact pending observation and its
  retry budget. Capacity failure is not evidence and not a schema repair.
- Any request for order, cancel, close, direction, method, geometry, units, or
  live permission: reject it as outside AI authority.

## Completion report

Report only:

- review trigger and reviewed timestamp;
- sealed regime and scorecard digests;
- supervised pair count and each pair's `GO` / `CAUTION` / `STOP` expiry;
- tuning queue reviewed/unreviewed counts and exact writer failures;
- fast-bot shadow progress: emitted signals, valid fills, filled active days,
  profit factor, filled-day lower bound, and current promotion blockers;
- confirmation that `AI_ORDER_AUTHORITY=NONE`, broker mutation did not occur,
  fast bot remains shadow-only, and manual/operator positions remained
  `NO_TOUCH`.

Do not report a market-guaranteed return. Forward evidence may improve the
system; it cannot guarantee a monthly multiplier.

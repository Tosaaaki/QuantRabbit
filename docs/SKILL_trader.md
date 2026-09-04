# QuantRabbit AI Trader Runtime

`qr-trader` is the stable scheduler id for the AI-primary trader. The AI owns
the discretionary market decision. Deterministic code owns evidence capture,
validation, risk limits, duplicate prevention, protection, and any separately
authorized broker execution.

## Load order

1. Read `docs/AGENT_CONTRACT.md` in full. It is authoritative.
2. Read all required artifacts from one current, sealed live-runtime snapshot.
3. Build the deterministic baseline and market-read evidence packet.
4. Make one complete AI decision from that evidence.
5. Apply and independently verify the decision.
6. Record and score it as shadow evidence. Do not send it to the broker.

The prompts under `docs/trader_prompts/` may be used only through this contract.
If an older prompt implies direct broker access, live permission, stale-data
fallback, or weaker ownership/risk rules, this playbook wins.

## Authority boundary

- `AI_DECISION_AUTHORITY=SHADOW` is active.
- `AI_ORDER_AUTHORITY=NONE` remains the independent live-send gate.
- AI may choose `TRADE`, `WAIT`, or `REQUEST_EVIDENCE`.
- For `TRADE`, AI may choose pair, side, strategy method, order vehicle, entry,
  TP, SL, geometry, allocation multiplier, and units.
- AI may author a `CLOSE` candidate only for an explicitly system-owned position
  and only under the two-gate close contract in `docs/AGENT_CONTRACT.md`.
- AI may reject the deterministic baseline and choose another evidenced lane.
  The overlay must state what changed and bind every changed field to current
  evidence.
- Deterministic validation may reject a decision or reduce units to a valid
  cap. It must not silently invent a different pair, side, method, or geometry.
- No accepted decision grants live permission in the current stage. Do not call
  OANDA, `guardian-action-cycle`, `AutoTradeCycle`, `LiveOrderGateway`,
  `PositionProtectionGateway`, or any low-level broker client.
- Manual, operator-owned, tagless, external, and ambiguous-owner positions are
  `NO_TOUCH`.

## Required current evidence

Read the applicable artifacts before every decision:

- `data/broker_snapshot.json`
- `data/position_guardian_chart_freshness.json`
- `data/guardian_events.json`
- `data/guardian_escalation.json`
- `data/hierarchical_bot_regime.json`
- `data/fast_bot_shadow.json`
- `data/fast_bot_scorecard.json`
- `data/active_trader_contract.json`
- `data/active_opportunity_board.json`
- `data/trader_intent_packet.json`
- `data/market_read_evidence_packet.json`
- current position/thesis/protection sidecars required by the selected action

Every selected pair, quote, spread, candle, position, intent, and contract must
belong to the same bounded snapshot. A container timestamp cannot refresh an old
candle, quote, intent, receipt, or Guardian observation. Missing, malformed,
stale, future-dated, unsealed, digest-mismatched, or mutually inconsistent
evidence requires `WAIT` or `REQUEST_EVIDENCE`; never fill the gap with a guess.

## Decision cycle

### 1. Refresh and freeze evidence

Use the existing read-only producers and prechecks to obtain one consistent
snapshot. Do not run a command with broker-write capability. If precheck or
freshness validation fails, preserve the previous artifacts and report the
exact blocker.

### 2. Build the deterministic proposal

Run the approved baseline builder:

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-draft-decision \
  --snapshot data/broker_snapshot.json \
  --guardian-action-receipt data/guardian_action_receipt.json \
  --output data/trader_decision_baseline.json \
  --market-read-evidence-packet data/market_read_evidence_packet.json
```

The baseline is a reproducible proposal and counterargument surface. It is not
the final trader decision.

### 3. Make one complete AI decision

Evaluate the selected lane and meaningful alternatives across market regime,
multi-timeframe structure, current spread and fillability, post-cost payoff,
portfolio exposure, margin, correlation, ownership, active risk, and current
forward evidence. Then choose exactly one action:

- `TRADE`: provide every required order field and a bounded evidence rationale.
- `WAIT`: name the exact risk, market, timing, or payoff reason.
- `REQUEST_EVIDENCE`: name the exact missing/stale artifact and acquisition step.
- `CLOSE`: name one eligible system-owned target and the exact Gate A/Gate B proof.

A `TRADE` must include the selected pair, side, method, vehicle, entry, TP, SL,
units, allocation multiplier, invalidation, expected post-cost payoff, expiry,
and why the rejected alternatives are weaker. Do not emit a partial trade for a
later bot to complete.

Write the candidate only to the ignored temporary path expected by the existing
overlay workflow, then atomically publish `data/codex_market_read_overlay.json`
through the approved writer/command. Do not hand-edit generated sealed outputs.

### 4. Apply and verify

Apply the overlay to the exact baseline and packet:

```bash
PYTHONPATH=src "$QR_PYTHON" -m quant_rabbit.cli trader-apply-market-read \
  --baseline data/trader_decision_baseline.json \
  --packet data/market_read_evidence_packet.json \
  --overlay data/codex_market_read_overlay.json \
  --output data/codex_trader_decision_response.json
```

Run the approved `gpt-trader-decision` verifier against the same snapshot and
receipt chain. The verifier must reject stale bindings, unsupported geometry,
ownership ambiguity, duplicate risk, invalid units, margin/risk failure, or a
decision that conflicts with stronger current evidence. A rejection ends the
cycle; the deterministic layer must not substitute a different trade.

### 5. Record shadow outcome

Persist the accepted decision and its later outcome in the existing audit and
shadow evidence surfaces. The receipt must say:

- `ai_decision_authority=SHADOW`
- `ai_order_authority=NONE`
- `live_permission=false`
- `broker_mutation_allowed=false`

Do not invoke the verified live wrapper or any send/close/protection gateway.
The current goal is to compare AI decisions against the deterministic baseline,
rejected alternatives, and exact forward broker truth before live activation is
considered separately.

## Longer-horizon review

A slower, higher-capability review may examine regime shifts, portfolio risk,
model disagreement, performance degradation, architecture, and Guardian tuning.
It may publish sealed pair supervision and bounded tuning experiments. It must
not rewrite an already frozen decision retrospectively or reuse evaluated
outcomes to select the cohort that produced them.

Changing model, cadence, or worker layout does not alter authority. The single
AI owner makes the final decision; bounded workers may gather or critique
evidence but may not publish a competing receipt or perform external side effects.

## Failure handling

- No valid opportunity: choose `WAIT`; do not force turnover.
- Missing or stale evidence: choose `REQUEST_EVIDENCE` or stop with the exact
  blocker; do not refresh timestamps or invent values.
- Overlay/apply/verifier conflict: reread once from the same snapshot boundary;
  if identity changed, defer to the next cycle.
- Model/runtime/quota failure: preserve inputs and publish no replacement
  decision.
- Rejected decision: record the verifier reasons and end the cycle without a
  bot-authored substitute.
- Any request to bypass risk, ownership, duplicate, margin, protection, or
  gateway validation: reject it.

## Completion report

Report:

- snapshot and decision timestamps;
- chosen action and, for `TRADE`, pair/side/method/vehicle/entry/TP/SL/units;
- evidence supporting the choice and why material alternatives were rejected;
- apply/verifier result and exact blockers;
- relevant forward scorecard progress;
- confirmation that `AI_DECISION_AUTHORITY=SHADOW`,
  `AI_ORDER_AUTHORITY=NONE`, no broker mutation occurred, and `NO_TOUCH`
  positions were unchanged.

Do not promise returns. A complete AI decision is a testable hypothesis until
forward outcomes demonstrate otherwise.

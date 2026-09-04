# QuantRabbit AI Trader Runtime

This is the executable playbook for the AI trader control plane. Development,
policy refresh, and strategic review read `docs/AGENT_CONTRACT.md` first. The
ten-minute hot path does not: it verifies the sealed local policy and reads only
the bounded evidence packet plus the manifest's candidate schema.

## Architecture

AI owns the discretionary market decision. Deterministic code supplies
replaceable services:

1. collectors refresh broker, market, news, performance, and portfolio facts;
2. the evidence adapter allowlists and seals a compact point-in-time packet;
3. `prepare` binds exact policy and evidence bytes into a run manifest;
4. AI emits an Entry decision and, where applicable, an independent Exit decision;
5. the adjudicator selects no more than one mutation for one broker epoch;
6. `accept` revalidates freshness, ownership, sizing arithmetic, costs, net edge,
   geometry, and the configured execution sink.

Profiles and workers live in `config/ai_trading_runtime.json`. Model selection
belongs to the Codex automation, not repository code. The candidate records the
actual model and reasoning effort used, so Luna, Terra, Sol, or a later model
can use the same runtime without a code change.

The current `intraday` sink is `live_gateway`. Its explicit action allowlist is
`ENTER`, `WAIT`, and `REQUEST_EVIDENCE`. For `ENTER`, AI chooses
the pair, side, method, vehicle, entry, TP, SL, units, confidence, and rationale.
Units must come from the dynamic sizing receipt. There is no fixed 1,000-unit
cap or floor, target-trade-count divisor, or allocation multiplier.
The sink can forward exactly one fresh entry only when
`QR_AI_ORDER_AUTHORITY=LIVE` and `QR_LIVE_ENABLED=1`. It copies no bot-authored
direction, price, TP, SL, or units. It binds current audited execution metadata,
then routes through `RiskEngine` and `LiveOrderGateway`; any failed gate is a
no-POST receipt.

Manual, operator-owned, tagless, external, and ambiguous-owner positions are
`NO_TOUCH`, although their exposure still consumes portfolio and margin
capacity. Legacy strategy and order-intent artifacts are `BASELINE_ONLY` and
must never be copied into a new decision. Do not call OANDA, a broker SDK, or
any low-level gateway directly. Only the configured sink may invoke the single
ordinary broker gateway after acceptance.

The regime router compares at least `momentum_breakout`, `mean_reversion`,
`confirmed_reversal`, and `WAIT` from the same point-in-time packet. Candidate
families are extensible metadata, not deterministic direction providers. Map
an executable winner onto the current audited method vocabulary and retain its
family/cohort under `extensions`. The same-direction 0.8-pip
`confirmed_reversal` cohort is distinct from ATR-structure cohorts and must be
scored separately. Rank candidates by net edge after spread, slippage, swap,
and measured latency; positive gross movement alone is insufficient.

## Intraday cycle

Run every ten minutes with the automation-selected normal model.

1. Verify the sealed policy for the exact project, account fingerprint,
   environment, revocation epoch, source pages, and expiry. A failure blocks
   all broker mutation and requests a control-plane refresh; there is no
   embedded-policy fallback.
2. Acquire the non-blocking singleflight admission and capacity check before
   model work. An unchanged source digest is `NO_UPDATE`.
3. Refresh observations without any send or live flags, then build the compact
   evidence packet:

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"
PYTHONPATH=src "$QR_PYTHON" tools/ai_trader_runtime.py build-evidence \
  --output "$QR_AI_STATE_ROOT/evidence_packet.json"
```

4. Prepare one run through the hot-path runner:

```bash
PYTHONPATH=src "$QR_PYTHON" tools/ai_trader_hotpath.py \
  --repo-root "$QR_REPO_ROOT" \
  --state-root "$QR_AI_STATE_ROOT" \
  --policy-snapshot "$QR_AI_POLICY_SNAPSHOT" \
  --project-key "$QR_AI_PROJECT_KEY" \
  --broker-account-id "$QR_AI_BROKER_ACCOUNT_ID" \
  --environment "$QR_AI_ENVIRONMENT" \
  --revocation-epoch "$QR_AI_POLICY_REVOCATION_EPOCH"
```

5. Read the printed manifest. If status is `BLOCKED`, report the exact required
   worker artifacts and stop without accepting the template.
6. Read only the referenced policy snapshot, compact evidence packet, optional
   strategic review, and candidate schema in the manifest.
   Replace the candidate template at the printed `candidate_path` with one
   complete AI decision. Keep the exact `run_id`, `profile`, `kind`, and
   `source_digest`. Use the actual runtime model and reasoning effort.
7. Cite source references as `<worker>:<path>`. Never infer a missing value.
8. Accept the decision:

```bash
PYTHONPATH=src "$QR_PYTHON" tools/ai_trader_runtime.py accept \
  --manifest <printed_manifest_path> \
  --candidate <printed_candidate_path>
```

`accept` rehashes every input. `EVIDENCE_CHANGED`, `MANIFEST_STALE`, or any
candidate rejection ends the run. Do not regenerate a different trade from
the old manifest. The next schedule creates a fresh run.

For `ENTER`, write exactly one fully specified order. LONG geometry requires
`stop_loss < entry < take_profit`; SHORT requires
`take_profit < entry < stop_loss`; `vehicle` must match `order_type`. Attach the
packet/source/broker-epoch binding, net-edge proof, cost proof, and a complete
recomputable sizing receipt. Dynamic sizing uses the minimum current risk
allowance, stop loss per unit, calibration, drawdown, correlation and net-edge
reducers, and post-exposure margin/correlation/broker caps. A decision is
broker-executed only when its final receipt has `sent=true` and an explicit
gateway readback.

The versioned Exit decision and adjudicator are implemented, but `EXIT` is not
in the current live profile allowlist. Every live entry must attach broker-side
TP/SL, and eligible system-position protection stays deterministic until the
owner/revision-bound live Exit adapter is deployed. Do not target manual or
uncertain ownership.

Stay quiet for an ordinary unchanged `WAIT`. Report a meaningful new trade
decision, evidence failure, validation rejection, or required user action.

## Strategic cycle

Run every two hours with the automation-selected advanced model. This cycle
does not create orders.

```bash
PYTHONPATH=src "$QR_PYTHON" tools/ai_trader_runtime.py prepare --profile strategic
PYTHONPATH=src "$QR_PYTHON" tools/ai_trader_runtime.py accept \
  --manifest <printed_manifest_path> \
  --candidate <printed_candidate_path>
```

The AI writes a time-bounded regime and risk review containing `regime`,
`risk_posture`, `valid_until_utc`, `themes`, and `instructions`. The
`review_overlay` sink appends the review ledger and publishes the current review
under the AI runtime state root. Intraday treats it as optional evidence. An
expired or missing review never blocks evidence collection and never creates a
trade by itself.

## Extending the runtime

- Add or remove evidence workers in `config/ai_trading_runtime.json`.
- Add a new profile by choosing `kind=trade|review` and a registered sink.
- Put provider-specific model selection in the Codex automation only.
- Put additional decision fields under `extensions` until a versioned core
  field is justified.
- Add a sink by implementing `DecisionSink.persist`; do not add broker calls to
  the AI decision writer.

The live sink must retain action-time broker truth, ownership checks, risk and
margin validation, cost and net-edge validation, duplicate prevention, durable
reservation, gateway readback, and explicit activation. New evidence workers,
decision factors, model names, and strategy candidates belong behind versioned
interfaces or `extensions`; they must not weaken the invariant execution
boundary. The paper and review sinks remain usable without it.

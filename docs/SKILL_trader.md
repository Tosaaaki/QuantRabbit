# QuantRabbit AI Trader Runtime

This is the executable playbook for the scheduled AI trader. Read
`docs/AGENT_CONTRACT.md` first.

## Architecture

AI owns the discretionary market decision. Deterministic code supplies four
replaceable services:

1. evidence workers collect broker, market, news, performance, and strategy inputs;
2. `prepare` seals their exact bytes into a run manifest;
3. `accept` rejects stale, changed, malformed, or unsafe candidate geometry;
4. a configured sink stores the accepted result.

Profiles and workers live in `config/ai_trading_runtime.json`. Model selection
belongs to the Codex automation, not repository code. The candidate records the
actual model and reasoning effort used, so Luna, Terra, Sol, or a later model
can use the same runtime without a code change.

The current `intraday` sink is `paper_ledger`. AI may choose `TRADE`, `WAIT`,
`REQUEST_EVIDENCE`, or an eligible system-owned `CLOSE`. For `TRADE`, AI chooses
the pair, side, method, vehicle, entry, TP, SL, units, allocation multiplier,
confidence, and rationale.
Accepted orders are paper decisions only: `AI_ORDER_AUTHORITY=NONE`,
`live_permission=false`, `broker_mutation_allowed=false`, and broker API calls
are forbidden. A future
live sink must be reviewed separately and must route through `RiskEngine` and
`LiveOrderGateway`.

Manual, operator-owned, tagless, external, and ambiguous-owner positions are
`NO_TOUCH`. Do not call OANDA, a broker SDK, `stage-live-order --send`, or any
low-level gateway from this playbook.

## Intraday cycle

Run every ten minutes with the automation-selected normal model.

1. Read the current contract and this playbook completely.
2. Acquire the shared runtime lock and run the existing read-only
   `cycle-refresh` path. Do not pass any send or live flags.
3. Prepare one run:

```bash
export QR_PYTHON="${QR_PYTHON:-/opt/homebrew/bin/python3}"
PYTHONPATH=src "$QR_PYTHON" tools/ai_trader_runtime.py prepare --profile intraday
```

4. Read the printed manifest. If status is `BLOCKED`, report the exact required
   worker artifacts and stop without accepting the template.
5. Read the referenced evidence files and the candidate schema in the manifest.
   Replace the candidate template at the printed `candidate_path` with one
   complete AI decision. Keep the exact `run_id`, `profile`, `kind`, and
   `source_digest`. Use the actual runtime model and reasoning effort.
6. Cite source references as `<worker>:<path>`. Never infer a missing value.
7. Accept the decision:

```bash
PYTHONPATH=src "$QR_PYTHON" tools/ai_trader_runtime.py accept \
  --manifest <printed_manifest_path> \
  --candidate <printed_candidate_path>
```

`accept` rehashes every input. `EVIDENCE_CHANGED`, `MANIFEST_STALE`, or any
candidate rejection ends the run. Do not regenerate a different trade from
the old manifest. The next schedule creates a fresh run.

For `TRADE`, write one or more fully specified orders. LONG geometry requires
`stop_loss < entry < take_profit`; SHORT requires
`take_profit < entry < stop_loss`. The AI chooses units in paper mode. Do not
describe a paper receipt as profitable, live-ready, or broker-executed.

For `CLOSE`, name exactly one `trade_id`, set `ownership=SYSTEM`, and state the
reason. This is still a paper action. Do not target manual or uncertain
ownership.

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

Any future live sink requires separate review, action-time broker truth,
ownership checks, risk and margin validation, duplicate prevention, durable
reservation, gateway readback, rollback, and explicit activation. The paper
and review sinks must remain usable without it.

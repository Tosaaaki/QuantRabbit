# Decision capsule recorder

Records what the operator decided, at the moment they decided it, including the
cases they declined.

## Why

The 2026-08-12 reproducibility audit
(`research/manual_method_reproducibility_deep_audit/2026-08-12/`) closed at
`NOT_EVALUABLE_OBSERVATION_AND_EVALUATOR_INSUFFICIENT` with this boundary:

| Evidence | Count |
|---|---|
| direct contemporaneous operator decision events | **0** |
| direct retrospective operator management directives | 2 |
| V3 prospective all-pair capsules / direct labels | 28 / **0** |
| V3 +30min outcome rows / signed outcomes | 28 / **0** |
| historical exit events | 411 (an *exit* population, not an all-opportunity one) |
| 2025 ledger source transactions | 2,309 |
| local execution ledger events | 2,887 |
| traceability agreement | `null` |
| out-of-sample economic result | `null` |

Thousands of fills, zero decisions. The existing 37-row
`entry_thesis_ledger.jsonl` is a system forecast plus broker back-fill — not a
human label source. A method cannot be cloned from its outcomes when neither the
inputs it saw nor the cases it passed on were written down, and no amount of
replay recovers them: "what did you look at, and why did you skip it" is not
derivable from price history.

This recorder writes the missing half.

## Use

```bash
tools/capture_decision.py "USDJPY skip 弱い"
```

Grammar: `<pair> <action> [confidence] [note...]`

| Part | Accepted |
|---|---|
| pair | `USDJPY`, `usd_jpy`, `EUR/USD` — all normalise to `USD_JPY` form |
| action | `long` `short` `skip` `enter` `exit` `partial` `reentry` `rotate` `hedge` `unwind` (`l`/`s`/`pass`/`close` aliases) |
| confidence | a bare number in `[0, 1]`, optional |
| note | free text, kept verbatim |

`long`/`short` fold the side into `ENTER`, because that is how a decision is
actually spoken. Everything after the optional number is the note.

```bash
tools/capture_decision.py --dry-run "GBPJPY long 0.8 確信中"   # build + validate, write nothing
tools/capture_decision.py --verify                              # check today's hash chain
```

Output lands in `research/manual_method_direct_operator_capture/<UTC date>/`
(`capsules.jsonl`, `capsule_index.jsonl`, `feature_spec.json`). Override the
root with `QR_DECISION_CAPSULE_ROOT`.

## What a capsule holds

Conforms to `docs/schemas/manual_decision_capsule_v1.schema.json`
(`QR_DIRECT_MANUAL_DECISION_CAPSULE_V1`, `additionalProperties: false`).

- **operator_evidence** — the label: action, side, confidence, note, and a
  `TEXT` evidence ref carrying the sha256 of the line as typed.
- **market_context.timeframes** — all seven of S5/M1/M5/M15/H1/H4/D1, each with
  `bar_end_utc`, `complete`, the closing candle, ATR, normalised slope and
  angle, momentum, and the trailing support/resistance window.
- **broker_context** — decision-time quote, spread, NAV, margin, open positions,
  pending orders, transaction watermark. `read_only` is a schema constant.
- **missing[]** — every field that could not be observed, with a reason.

## Rules the format enforces

**Nothing is imputed.** Unobservable fields are `null` plus a `missing[]` entry.
No default, no neighbouring-timeframe fill, no back-fill.

**Confidence is never inferred from words.** `確信中` is stored as note text and
`confidence` stays `null`. Scoring it at 0.8 would manufacture the operator
label this whole exercise exists to obtain. Type a number when you want one.

**No machine label rides beside the human one.** `proxy_classifier` and
`inferred_label` are schema constants of `null`, and validation rejects a
capsule that fills either.

**No forming bars.** The incomplete bar is dropped before any feature is
computed, so the capsule carries no look-ahead past its own cutoff.

**Features are reproducible.** ATR is Wilder(14); slope is
`(close[-1] - close[-13]) / (12 × ATR)` in ATR-per-bar units; angle is
`degrees(atan(slope))`; momentum is a 3-bar change in ATR units; S/R is the
20-bar high/low. Pinned in `feature_spec.json` under `FEATURE_SPEC_VERSION`; the
audit refused to treat earlier indicator values as reproducible precisely
because their code, lookback and warm-up were never stated. Do not pool capsules
across spec versions.

**The label outlives the API.** A broker failure does not fail the capture — the
capsule is written with null context and the reason recorded. Losing a decision
because an HTTP call timed out would reproduce the exact gap being closed.

**Read-only by construction.** The tool imports `OandaReadOnlyClient` and never
the execution client. It has no order path.

**Append-only and tamper-evident.** `capsule_index.jsonl` chains
`sha256(prev + capsule)`; `--verify` recomputes it. The strict schema leaves no
room for a chain pointer inside the capsule, so the chain lives beside it.

## Population contract — read before computing any rate

Two streams, and they must never be mixed (audit `population contract`):

**`EVENT_OVERSAMPLE`** — implemented here. One capsule per decision actually
voiced, `SKIP` included. An explicit `SKIP` is attributed **only to the pair and
clock directly labelled**. It says nothing about the other 27 pairs.

**`FULL_28_PAIR_CLOCK`** — *not yet built*. The denominator: all 28 pairs on each
fixed UTC 5-minute clock, labels preserved as `null`, `direct: false`.

Until the clock stream exists, **rates must not be computed against the full
universe** — only labelled-against-labelled comparisons are admissible. The
oversample alone is what the audit graded `FAIL` on selection bias; what makes
it usable now is that declines are recorded, not that the denominator is solved.

## Known blocker

`OandaReadOnlyClient` cannot be constructed anywhere in this repo right now:

```
quant_rabbit.instruments.SpreadCalibrationError: spread calibration is expired
```

`config/oanda_spread_calibration_v1.json` covers 2026-07-06 → 07-13 and expired
at `2026-08-13T15:00:00Z` under the 31-day policy. The load runs at
module-import time in `instruments.py`, so every consumer of `broker.oanda` has
been failing since 2026-08-13 — this is repo-wide and predates the recorder.

Effect here: capsules still record the operator label in full, but
`broker_context` and `market_context.timeframes` come back `null` with
`BROKER_UNAVAILABLE` in `missing[]`. The recorder's own maths is verified
against live candles; only the sanctioned client path is blocked.

Refreshing the calibration needs a fresh 6-business-day, 12:00–15:00 UTC M5
spread sample across 28 pairs under `OANDA_M5_MBA_SESSION_SPREAD_MONTHLY_V1`,
with `valid_until_utc == window.to_utc + 31 days`. It is read-only work but it
resets the cost baseline the rest of the system quotes, so it is a decision of
its own, not a side effect of this change.

## Next

1. Refresh the spread calibration to unblock `broker_context` (above).
2. Build the `FULL_28_PAIR_CLOCK` denominator.
3. Add an outcome pass that resolves `fill`, `financing` and `conversion` at
   fixed horizons — never at capture time, where they do not yet exist.
4. Add a Slack intake so decisions can be recorded away from the desk. Per the
   user-level agent instructions, that requires reading the Notion
   `💬 Slack運用` source first, in the same run, before touching Slack.

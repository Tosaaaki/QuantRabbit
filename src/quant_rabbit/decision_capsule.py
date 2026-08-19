"""Direct operator decision capsules (`QR_DIRECT_MANUAL_DECISION_CAPSULE_V1`).

Why this module exists
----------------------
The 2026-08-12 reproducibility audit
(`research/manual_method_reproducibility_deep_audit/2026-08-12/`) closed with
`NOT_EVALUABLE_OBSERVATION_AND_EVALUATOR_INSUFFICIENT` and this evidence
boundary:

    direct contemporaneous operator decision events: 0

Thousands of broker fills exist (2,309 source transactions, 411 exit events,
2,887 execution-ledger events) but not one record of what the operator decided
at the same time, and no explicit non-entry population at all. A method cannot
be reproduced from its outcomes when neither the inputs it saw nor the cases it
declined were ever written down.

This module writes the missing record. It is a *recorder*, not a trader:

* it never constructs an order payload and never imports an execution client;
* `broker_context.read_only` is a schema constant of `true`;
* `proxy_classifier` and `inferred_label` are schema constants of `null`, so a
  capsule can never carry a machine-guessed label beside the human one.

Anti-imputation rule
--------------------
Every field this module cannot *observe* is written as `null` and listed in
`missing[]` with a reason. Nothing is modelled, back-filled, or inferred. In
particular `confidence` is populated only from an explicit number typed by the
operator: mapping words like "強い" onto 0.8 would manufacture the very label
the audit says does not exist.

Population contract
-------------------
Two streams, kept separate (audit `population contract`):

`EVENT_OVERSAMPLE` (this module's `capture` path)
    One capsule per decision the operator actually voices, including `SKIP`.
    An explicit `SKIP` is attributed **only** to the pair and clock directly
    labelled — it never implies a decision about the other 27 pairs.

`FULL_28_PAIR_CLOCK` (not yet implemented; see module README)
    The denominator: all 28 pairs on each fixed UTC 5-minute clock with labels
    preserved as `null`. Until it exists, rates must not be computed against
    the full universe — only labelled-vs-labelled comparisons are admissible.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_ID = "QR_DIRECT_MANUAL_DECISION_CAPSULE_V1"

# Feature definitions are pinned here and mirrored into `feature_spec.json`
# beside the capsules. The audit refused to treat supplied indicator values as
# independently reproducible because "算出code・version・lookback・warm-up" were
# never provided. Bump FEATURE_SPEC_VERSION on any change to the maths below;
# capsules written under different versions must not be pooled.
FEATURE_SPEC_VERSION = "qr-dmc-features-v1"
ATR_PERIOD = 14
SLOPE_LOOKBACK_BARS = 12
MOMENTUM_LOOKBACK_BARS = 3
SUPPORT_RESISTANCE_LOOKBACK_BARS = 20
# ATR(14) needs 15 closes; the 20-bar S/R window is the widest single ask.
MIN_COMPLETE_BARS = ATR_PERIOD + SLOPE_LOOKBACK_BARS + 1

CAPSULE_TIMEFRAMES: tuple[str, ...] = ("S5", "M1", "M5", "M15", "H1", "H4", "D1")
# The schema's D1 is OANDA's "D"; every other name matches the broker's.
OANDA_GRANULARITY = {"S5": "S5", "M1": "M1", "M5": "M5", "M15": "M15", "H1": "H1", "H4": "H4", "D1": "D"}

PRIMARY_ACTIONS = ("ENTER", "SKIP", "EXIT", "PARTIAL_EXIT", "REENTRY", "PAIR_ROTATION", "HEDGE", "HEDGE_UNWIND")
SIDES = ("LONG", "SHORT")

# One-liner verbs. LONG/SHORT are ENTER with the side folded in, because that is
# how the decision is actually spoken ("GBPJPY long").
ACTION_WORDS: dict[str, tuple[str, str | None]] = {
    "enter": ("ENTER", None),
    "long": ("ENTER", "LONG"),
    "l": ("ENTER", "LONG"),
    "short": ("ENTER", "SHORT"),
    "s": ("ENTER", "SHORT"),
    "skip": ("SKIP", None),
    "pass": ("SKIP", None),
    "exit": ("EXIT", None),
    "close": ("EXIT", None),
    "partial": ("PARTIAL_EXIT", None),
    "reentry": ("REENTRY", None),
    "rotate": ("PAIR_ROTATION", None),
    "hedge": ("HEDGE", None),
    "unwind": ("HEDGE_UNWIND", None),
}

_PAIR_RE = re.compile(r"^([A-Za-z]{3})[_/]?([A-Za-z]{3})$")


class CapsuleError(ValueError):
    """Raised when an intake line or a built capsule violates the contract."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pip_size(pair: str) -> float:
    """OANDA pip for a pair. JPY quote pairs move in 0.01, everything else 0.0001."""

    return 0.01 if pair.endswith("_JPY") else 0.0001


def normalize_pair(raw: str) -> str:
    match = _PAIR_RE.match(raw.strip())
    if not match:
        raise CapsuleError(f"unrecognised pair: {raw!r} (expected e.g. USDJPY, usd_jpy, EUR/USD)")
    return f"{match.group(1).upper()}_{match.group(2).upper()}"


# --------------------------------------------------------------------------- #
# Intake                                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OperatorIntake:
    """One decision as the operator typed it.

    `confidence` stays None unless a bare number in [0, 1] was typed. Words are
    kept verbatim in `note` and are never scored.
    """

    raw_text: str
    pair: str
    primary_action: str
    side: str | None
    confidence: float | None
    note: str | None

    @property
    def source_sha256(self) -> str:
        return sha256_hex(self.raw_text)


def parse_intake(raw_text: str) -> OperatorIntake:
    """Parse `<pair> <action> [confidence] [note...]`.

    Examples::

        USDJPY skip 弱い
        GBPJPY long 0.8 確信中
        EUR/USD short
        usdjpy exit 利確
    """

    text = raw_text.strip()
    if not text:
        raise CapsuleError("empty intake line")
    tokens = text.split()
    if len(tokens) < 2:
        raise CapsuleError(f"intake needs at least '<pair> <action>': {raw_text!r}")

    pair = normalize_pair(tokens[0])
    verb = tokens[1].lower()
    if verb not in ACTION_WORDS:
        raise CapsuleError(f"unknown action {tokens[1]!r}; known: {', '.join(sorted(ACTION_WORDS))}")
    primary_action, side = ACTION_WORDS[verb]

    rest = tokens[2:]
    confidence: float | None = None
    if rest:
        # Only a bare number is confidence. Anything else is note text, kept as
        # typed: inferring 0.8 from "確信中" would fabricate an operator label.
        try:
            candidate = float(rest[0])
        except ValueError:
            candidate = None
        if candidate is not None:
            if not 0.0 <= candidate <= 1.0:
                raise CapsuleError(f"confidence must be within [0, 1], got {candidate}")
            confidence = candidate
            rest = rest[1:]

    # An explicit side after ENTER/EXIT verbs ("USDJPY exit long") stays honest:
    # take it when the verb did not already fix one.
    if rest and side is None and rest[0].upper() in SIDES:
        side = rest[0].upper()
        rest = rest[1:]

    note = " ".join(rest) if rest else None
    return OperatorIntake(
        raw_text=text,
        pair=pair,
        primary_action=primary_action,
        side=side,
        confidence=confidence,
        note=note,
    )


# --------------------------------------------------------------------------- #
# Market features                                                               #
# --------------------------------------------------------------------------- #


def _completed_bars(candles: Sequence[dict]) -> list[dict]:
    """Drop incomplete bars.

    The forming bar is the single largest source of look-ahead in a
    decision-time record, so it never enters a feature.
    """

    return [candle for candle in candles if candle.get("complete") is True]


def _ohlc(candle: dict) -> dict[str, float] | None:
    mid = candle.get("mid")
    if not isinstance(mid, dict):
        return None
    try:
        return {key: float(mid[key]) for key in ("o", "h", "l", "c")}
    except (KeyError, TypeError, ValueError):
        return None


def _wilder_atr(bars: Sequence[dict[str, float]], period: int) -> float | None:
    if len(bars) < period + 1:
        return None
    true_ranges: list[float] = []
    for previous, current in zip(bars[:-1], bars[1:]):
        true_ranges.append(
            max(
                current["h"] - current["l"],
                abs(current["h"] - previous["c"]),
                abs(current["l"] - previous["c"]),
            )
        )
    if len(true_ranges) < period:
        return None
    atr = sum(true_ranges[:period]) / period
    for true_range in true_ranges[period:]:
        atr = ((atr * (period - 1)) + true_range) / period
    return atr


def timeframe_features(timeframe: str, candles: Sequence[dict], pair: str, observed_at: datetime) -> dict[str, Any]:
    """Build one `market_context.timeframes[]` entry from raw OANDA candles.

    Anything not computable from the completed bars on hand is `null`; no value
    is ever imputed from a neighbouring timeframe or a default.
    """

    entry: dict[str, Any] = {
        "timeframe": timeframe,
        "bar_end_utc": None,
        "complete": None,
        "candle": None,
        "normalized_slope": None,
        "normalized_angle": None,
        "atr": None,
        "momentum": None,
        "support_resistance": None,
        "observed_at_utc": iso(observed_at),
    }

    completed = _completed_bars(candles)
    if not completed:
        return entry

    last = completed[-1]
    entry["bar_end_utc"] = last.get("time")
    entry["complete"] = True
    bars = [ohlc for ohlc in (_ohlc(candle) for candle in completed) if ohlc is not None]
    if not bars:
        return entry

    entry["candle"] = dict(bars[-1])
    pip = pip_size(pair)
    atr = _wilder_atr(bars, ATR_PERIOD)
    if atr is not None and atr > 0:
        entry["atr"] = round(atr / pip, 6)
        closes = [bar["c"] for bar in bars]
        if len(closes) > SLOPE_LOOKBACK_BARS:
            # ATR per bar: dimensionless, so slopes are comparable across pairs
            # and timeframes without a second normalisation step.
            slope = (closes[-1] - closes[-1 - SLOPE_LOOKBACK_BARS]) / (SLOPE_LOOKBACK_BARS * atr)
            entry["normalized_slope"] = round(slope, 6)
            entry["normalized_angle"] = round(math.degrees(math.atan(slope)), 6)
        if len(closes) > MOMENTUM_LOOKBACK_BARS:
            entry["momentum"] = round((closes[-1] - closes[-1 - MOMENTUM_LOOKBACK_BARS]) / atr, 6)

    window = bars[-SUPPORT_RESISTANCE_LOOKBACK_BARS:]
    if window:
        entry["support_resistance"] = {
            "highest_high": max(bar["h"] for bar in window),
            "lowest_low": min(bar["l"] for bar in window),
            "lookback_bars": len(window),
            "feature_spec_version": FEATURE_SPEC_VERSION,
        }
    return entry


def feature_spec() -> dict[str, Any]:
    """The pinned, independently reproducible definition of every feature above."""

    return {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "bar_selection": "completed bars only; the forming bar is dropped before any computation",
        "atr": {
            "method": "Wilder",
            "period": ATR_PERIOD,
            "warmup_bars": ATR_PERIOD + 1,
            "unit": "pips (JPY-quote pairs 0.01, otherwise 0.0001)",
        },
        "normalized_slope": {
            "formula": "(close[-1] - close[-1-L]) / (L * atr_price)",
            "lookback_bars": SLOPE_LOOKBACK_BARS,
            "unit": "ATR per bar (dimensionless)",
        },
        "normalized_angle": {"formula": "degrees(atan(normalized_slope))", "unit": "degrees"},
        "momentum": {
            "formula": "(close[-1] - close[-1-M]) / atr_price",
            "lookback_bars": MOMENTUM_LOOKBACK_BARS,
            "unit": "ATR (dimensionless)",
        },
        "support_resistance": {
            "formula": "max(high) and min(low) over the trailing window",
            "lookback_bars": SUPPORT_RESISTANCE_LOOKBACK_BARS,
            "unit": "price",
        },
        "minimum_complete_bars_requested": MIN_COMPLETE_BARS,
        "timeframes": list(CAPSULE_TIMEFRAMES),
        "oanda_granularity": dict(OANDA_GRANULARITY),
    }


# --------------------------------------------------------------------------- #
# Capsule assembly                                                              #
# --------------------------------------------------------------------------- #


def _missing(field: str, reason: str) -> dict[str, str]:
    return {"field": field, "reason": reason}


def build_capsule(
    intake: OperatorIntake,
    *,
    captured_at: datetime,
    decision_cutoff: datetime,
    timeframes: Sequence[dict[str, Any]],
    broker_context: dict[str, Any],
    extra_missing: Iterable[dict[str, str]] = (),
) -> dict[str, Any]:
    """Assemble one `EVENT_OVERSAMPLE` capsule. Never partially imputes."""

    if len(timeframes) != len(CAPSULE_TIMEFRAMES):
        raise CapsuleError(f"expected {len(CAPSULE_TIMEFRAMES)} timeframe entries, got {len(timeframes)}")

    missing: list[dict[str, str]] = [
        # Geometry is what the operator *drew*. A text one-liner carries none of
        # it, and guessing anchors from price would be exactly the inferred
        # label this capsule format forbids.
        _missing("market_context.geometry.n_wave_anchors", "NOT_SUPPLIED_BY_TEXT_INTAKE"),
        _missing("market_context.geometry.trendline_anchors", "NOT_SUPPLIED_BY_TEXT_INTAKE"),
        _missing("market_context.geometry.stall_anchor", "NOT_SUPPLIED_BY_TEXT_INTAKE"),
        _missing("market_context.geometry.rebound_anchor", "NOT_SUPPLIED_BY_TEXT_INTAKE"),
        _missing("operator_evidence.evidence_refs.SCREENSHOT", "NOT_SUPPLIED_BY_TEXT_INTAKE"),
    ]
    if intake.confidence is None:
        missing.append(
            _missing("operator_evidence.confidence", "NO_EXPLICIT_NUMERIC_CONFIDENCE_TYPED_NEVER_INFERRED_FROM_WORDS")
        )
    for entry in timeframes:
        if entry.get("atr") is None:
            missing.append(
                _missing(
                    f"market_context.timeframes[{entry['timeframe']}]",
                    "INSUFFICIENT_COMPLETE_BARS_AT_DECISION_CUTOFF",
                )
            )
    missing.extend(extra_missing)

    capsule: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "capsule_id": "",
        "record_kind": "MANUAL_EVENT",
        "population_stream": "EVENT_OVERSAMPLE",
        "pair": intake.pair,
        "captured_at_utc": iso(captured_at),
        "decision_cutoff_utc": iso(decision_cutoff),
        "operator_evidence": {
            "direct": True,
            "timing": "CONTEMPORANEOUS",
            "primary_action": intake.primary_action,
            "side": intake.side,
            "confidence": intake.confidence,
            "note": intake.note,
            "lifecycle": {
                "tp_intent": None,
                "partial_exit": None,
                "exit_reason": intake.note if intake.primary_action in ("EXIT", "PARTIAL_EXIT") else None,
                "reverse_reentry": True if intake.primary_action == "REENTRY" else None,
                "pair_rotation_to": None,
                "hedge_pair": None,
                "unwind_reason": intake.note if intake.primary_action == "HEDGE_UNWIND" else None,
            },
            "evidence_refs": [
                {
                    "kind": "TEXT",
                    "captured_at_utc": iso(captured_at),
                    "sha256": intake.source_sha256,
                }
            ],
        },
        "market_context": {
            "timeframes": list(timeframes),
            "geometry": {
                "n_wave_anchors": None,
                "trendline_anchors": None,
                "stall_anchor": None,
                "rebound_anchor": None,
                "source_evidence_sha256": None,
            },
        },
        "broker_context": broker_context,
        "traceability": {
            "classification": "MIXED",
            "observed_fields": [
                "operator_evidence.primary_action",
                "operator_evidence.side",
                "operator_evidence.note",
                "broker_context.bid",
                "broker_context.ask",
                "broker_context.spread",
                "market_context.timeframes",
            ],
            "reconstructable_fields": ["market_context.timeframes.candle"],
            "missing_fields": sorted({item["field"] for item in missing}),
        },
        "missing": missing,
        # Schema constants. A capsule may carry the operator's label and nothing
        # that imitates one.
        "proxy_classifier": None,
        "inferred_label": None,
        "source_sha256": intake.source_sha256,
    }

    body = dict(capsule)
    body.pop("capsule_id")
    capsule["capsule_id"] = f"dca_{sha256_hex(canonical_json(body))}"
    return capsule


def build_broker_context(
    *,
    quote_time_utc: str | None,
    bid: float | None,
    ask: float | None,
    spread: float | None,
    nav: float | None,
    margin_available: float | None,
    margin_used: float | None,
    positions: list[dict[str, Any]] | None,
    orders: list[dict[str, Any]] | None,
    transaction_watermark: str | None,
) -> dict[str, Any]:
    """Decision-time account state.

    `fill`, `financing` and `conversion` stay null: a capsule is written at the
    moment of the decision, before any of them exist. They are resolved later by
    the outcome pass, never guessed here.
    """

    return {
        "quote_time_utc": quote_time_utc,
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "nav": nav,
        "margin_available": margin_available,
        "margin_used": margin_used,
        "positions": positions,
        "orders": orders,
        "transaction_watermark": transaction_watermark,
        "fill": None,
        "financing": None,
        "conversion": None,
        "read_only": True,
    }


# --------------------------------------------------------------------------- #
# Validation                                                                    #
# --------------------------------------------------------------------------- #


def validate_capsule(capsule: dict[str, Any], schema_path: Path | None = None) -> None:
    """Enforce the invariants that make a capsule admissible as evidence.

    Runs the published JSON Schema when it is reachable, and always runs the
    three checks the audit turns on: no inferred label, no write authority, and
    a real `capsule_id`.
    """

    if capsule.get("schema") != SCHEMA_ID:
        raise CapsuleError(f"schema must be {SCHEMA_ID}")
    if capsule.get("proxy_classifier") is not None or capsule.get("inferred_label") is not None:
        raise CapsuleError("proxy_classifier and inferred_label must stay null; a capsule carries no machine label")
    if capsule.get("broker_context", {}).get("read_only") is not True:
        raise CapsuleError("broker_context.read_only must be true; this recorder has no write path")

    body = dict(capsule)
    stated_id = body.pop("capsule_id")
    expected = f"dca_{sha256_hex(canonical_json(body))}"
    if stated_id != expected:
        raise CapsuleError(f"capsule_id does not match content: {stated_id} != {expected}")

    evidence = capsule["operator_evidence"]
    if evidence["direct"] and not evidence["evidence_refs"]:
        raise CapsuleError("a direct capsule needs at least one evidence ref")
    if not evidence["direct"] and any(
        evidence[key] is not None for key in ("primary_action", "side", "confidence", "note")
    ):
        raise CapsuleError("a non-direct capsule must leave every operator field null")
    if evidence["primary_action"] is not None and evidence["primary_action"] not in PRIMARY_ACTIONS:
        raise CapsuleError(f"unknown primary_action: {evidence['primary_action']}")
    if evidence["side"] is not None and evidence["side"] not in SIDES:
        raise CapsuleError(f"unknown side: {evidence['side']}")

    timeframes = capsule["market_context"]["timeframes"]
    if [entry["timeframe"] for entry in timeframes] != list(CAPSULE_TIMEFRAMES):
        raise CapsuleError(f"timeframes must be exactly {CAPSULE_TIMEFRAMES} in order")

    if schema_path is None:
        return
    try:
        import jsonschema  # noqa: PLC0415 - optional, validation still runs without it
    except ImportError:
        return
    if schema_path.exists():
        jsonschema.validate(capsule, json.loads(schema_path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Append-only store                                                             #
# --------------------------------------------------------------------------- #


def append_capsule(capsule: dict[str, Any], artifacts: Path) -> dict[str, str]:
    """Append one capsule and extend the tamper-evident index.

    The capsule file stays strictly schema-shaped (`additionalProperties:false`
    leaves no room for a chain pointer), so the hash chain lives in a sibling
    index keyed by `capsule_id`.
    """

    artifacts.mkdir(parents=True, exist_ok=True)
    capsules_path = artifacts / "capsules.jsonl"
    index_path = artifacts / "capsule_index.jsonl"

    seen = set()
    previous = ""
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seen.add(row["capsule_id"])
            previous = row["sha256"]
    if capsule["capsule_id"] in seen:
        raise CapsuleError(f"duplicate capsule_id {capsule['capsule_id']}; intake already recorded")

    payload = canonical_json(capsule)
    index_row = {
        "capsule_id": capsule["capsule_id"],
        "appended_at_utc": iso(utc_now()),
        "sha256": sha256_hex(previous + payload),
        "prev_sha256": previous or None,
        "feature_spec_version": FEATURE_SPEC_VERSION,
    }
    with capsules_path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(index_row) + "\n")
    return index_row


def verify_chain(artifacts: Path) -> dict[str, Any]:
    """Recompute the index chain against the capsule file."""

    capsules_path = artifacts / "capsules.jsonl"
    index_path = artifacts / "capsule_index.jsonl"
    if not capsules_path.exists() or not index_path.exists():
        return {"status": "EMPTY", "records": 0, "errors": []}

    capsules = [json.loads(line) for line in capsules_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    if len(capsules) != len(index):
        errors.append(f"length mismatch: {len(capsules)} capsules vs {len(index)} index rows")

    previous = ""
    for capsule, row in zip(capsules, index):
        if capsule["capsule_id"] != row["capsule_id"]:
            errors.append(f"id mismatch at {row['capsule_id']}")
        expected = sha256_hex(previous + canonical_json(capsule))
        if row["sha256"] != expected:
            errors.append(f"chain break at {row['capsule_id']}")
        previous = row["sha256"]

    return {
        "status": "PASS" if not errors else "FAIL",
        "records": len(capsules),
        "tail_sha256": previous or None,
        "errors": errors,
    }


def default_artifacts_root() -> Path:
    """Where capsules land. Override with `QR_DECISION_CAPSULE_ROOT`."""

    override = os.environ.get("QR_DECISION_CAPSULE_ROOT")
    if override:
        return Path(override)
    root = Path(__file__).resolve().parents[2]
    return root / "research" / "manual_method_direct_operator_capture"

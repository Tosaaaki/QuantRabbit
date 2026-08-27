#!/usr/bin/env python3
"""Independent verifier for the hash-chained JPY oracle ledger.

This process does not import the oracle, producer runner, accounting runtime, or
result validator.  It validates exact artifact bindings, reaggregates economic
metrics from ledger rows, and emits a receipt suitable for gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR, localcontext
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")
ZERO_SHA = "0" * 64
ORACLE_NAME = "INDEPENDENT_JPY_ORACLE_V1"
VERIFIER_NAME = "INDEPENDENT_ORACLE_LEDGER_VERIFIER_V1"
ORACLE_FILE = "paper_research_jpy_oracle_v1.py"
CONTRACT_FILE = "PAPER_RESEARCH_JPY_ORACLE_CONTRACT_V1.json"
SCHEMA_FILE = "paper_research_jpy_oracle_schema_v1.json"
VERIFIER_SCHEMA_FILE = "paper_research_oracle_verifier_schema_v1.json"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DAY_NS = 86_400_000_000_000
JPY_MICROS_PER_YEN = 1_000_000
BASE_MICROUNITS_PER_UNIT = 1_000_000
FORBIDDEN_PROPOSAL_TOKENS = {
    "signalid", "fill", "fillprice", "path", "mfe", "mae", "pnl", "cost",
    "equity", "drawdown", "dd", "cvar", "profit", "return",
}


class VerificationError(RuntimeError):
    pass


def assert_no_float(value: Any, location: str = "root") -> None:
    if isinstance(value, float):
        raise VerificationError(f"float forbidden at {location}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_float(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_float(item, f"{location}[{index}]")


def canonical_bytes(value: Any) -> bytes:
    assert_no_float(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def embedded_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return sha256_bytes(canonical_bytes(unsigned))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _secure_root(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} root missing")
    path = Path(value)
    if not path.is_absolute():
        raise VerificationError(f"{label} root must be absolute")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise VerificationError(f"regular non-symlink {label} root required")
    return Path(os.path.abspath(path))


def _bounded_path(root: Path, value: Any, label: str, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} path missing")
    path = Path(value)
    if not path.is_absolute():
        raise VerificationError(f"{label} path must be absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise VerificationError(f"{label} path escapes capability root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise VerificationError(f"{label} path is not canonical")
    current = root
    for part in (relative.parts if must_exist else relative.parts[:-1]):
        current = current / part
        current_stat = os.lstat(current)
        if stat.S_ISLNK(current_stat.st_mode):
            raise VerificationError(f"{label} path contains symlink")
        if current != path and not stat.S_ISDIR(current_stat.st_mode):
            raise VerificationError(f"{label} parent is not directory")
    if must_exist:
        try:
            path.resolve(strict=True).relative_to(root.resolve(strict=True))
        except ValueError as error:
            raise VerificationError(f"{label} resolved outside capability root") from error
    return path


def _read(path: Path, *, root: Path) -> bytes:
    path = _bounded_path(root, str(path), "input artifact", must_exist=True)
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise VerificationError(f"regular artifact required: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise VerificationError(f"artifact identity changed: {path.name}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _artifact(
    spec: Mapping[str, Any], label: str, *, input_root: Path
) -> tuple[Path, bytes]:
    if set(spec) != {"path", "sha256", "size_bytes"}:
        raise VerificationError(f"{label} artifact schema mismatch")
    path = _bounded_path(input_root, spec.get("path"), label, must_exist=True)
    data = _read(path, root=input_root)
    if len(data) != spec["size_bytes"] or sha256_bytes(data) != spec["sha256"]:
        raise VerificationError(f"{label} artifact binding mismatch")
    return path, data


def _json(data: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid {label} JSON") from error
    if not isinstance(payload, dict):
        raise VerificationError(f"{label} must be object")
    assert_no_float(payload, label)
    return payload


def _ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise VerificationError("ratio denominator invalid")
    with localcontext() as context:
        context.prec = 50
        return format(Decimal(numerator) / Decimal(denominator), ".18f")


def _month(timestamp_ns: int) -> str:
    value = datetime.fromtimestamp(timestamp_ns // 1_000_000_000, tz=timezone.utc)
    return f"{value.year:04d}-{value.month:02d}"


def _parse_source(
    blob: bytes, manifest: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    expected_manifest = {
        "schema_version", "source_bytes_sha256", "source_size_bytes", "event_count",
        "first_source_ts_ns", "last_source_ts_ns", "manifest_sha256",
    }
    if set(manifest) != expected_manifest or manifest.get("schema_version") != 1:
        raise VerificationError("source manifest schema mismatch")
    if manifest.get("manifest_sha256") != embedded_hash(manifest, "manifest_sha256"):
        raise VerificationError("source manifest embedded hash mismatch")
    if (
        manifest["source_bytes_sha256"] != sha256_bytes(blob)
        or manifest["source_size_bytes"] != len(blob)
    ):
        raise VerificationError("source manifest does not bind exact bytes")
    if blob and not blob.endswith(b"\n"):
        raise VerificationError("source blob truncated")
    index: dict[str, dict[str, Any]] = {}
    books: dict[str, list[dict[str, Any]]] = {}
    expected_keys = {
        "schema_version", "provider_id", "instrument", "bid_ticks", "ask_ticks",
        "tick_scale", "source_ts_ns", "arrival_ts_ns", "provider_event_id",
        "sequence", "heartbeat", "quality_flags",
    }
    last_stream: dict[tuple[str, str], tuple[int, int, int]] = {}
    source_times: list[int] = []
    for line in blob.splitlines(keepends=True):
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError("source event JSON invalid") from error
        if not isinstance(event, dict) or set(event) != expected_keys:
            raise VerificationError("source event schema mismatch")
        assert_no_float(event, "source")
        if event["schema_version"] != 1 or event["heartbeat"] is not False:
            raise VerificationError("priced source event required")
        if not isinstance(event["quality_flags"], list) or event["quality_flags"]:
            raise VerificationError("quality-flagged source event unavailable")
        for field in (
            "bid_ticks", "ask_ticks", "tick_scale", "source_ts_ns", "arrival_ts_ns",
            "sequence",
        ):
            if not isinstance(event[field], int):
                raise VerificationError(f"source integer field invalid: {field}")
        if (
            event["bid_ticks"] <= 0
            or event["ask_ticks"] <= event["bid_ticks"]
            or event["tick_scale"] <= 0
            or event["arrival_ts_ns"] < event["source_ts_ns"]
            or PAIR_RE.fullmatch(event["instrument"]) is None
        ):
            raise VerificationError("source executable BBO invalid")
        stream = (event["provider_id"], event["instrument"])
        state = (event["source_ts_ns"], event["arrival_ts_ns"], event["sequence"])
        prior = last_stream.get(stream)
        if prior is not None and any(state[index] <= prior[index] for index in range(3)):
            raise VerificationError("source stream input order is not strictly increasing")
        last_stream[stream] = state
        digest = sha256_bytes(line)
        if digest in index:
            raise VerificationError("duplicate source event bytes")
        event = dict(event)
        event["source_event_sha256"] = digest
        index[digest] = event
        books.setdefault(event["instrument"], []).append(event)
        source_times.append(event["source_ts_ns"])
    if not index:
        raise VerificationError("source event index empty")
    if (
        len(index) != manifest["event_count"]
        or min(source_times) != manifest["first_source_ts_ns"]
        or max(source_times) != manifest["last_source_ts_ns"]
    ):
        raise VerificationError("source manifest semantic count/time mismatch")
    return index, books


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_producer_fields(value: Any, location: str = "proposal") -> None:
    if isinstance(value, float):
        raise VerificationError(f"proposal float forbidden at {location}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise VerificationError("proposal key is not text")
            token = _normalize(key)
            if any(forbidden in token for forbidden in FORBIDDEN_PROPOSAL_TOKENS):
                raise VerificationError(f"proposal outcome/identifier forbidden at {location}.{key}")
            _reject_producer_fields(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_producer_fields(item, f"{location}[{index}]")
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise VerificationError(f"proposal type forbidden at {location}")


def _pair(instrument: str) -> tuple[str, str]:
    if not isinstance(instrument, str) or PAIR_RE.fullmatch(instrument) is None:
        raise VerificationError("proposal instrument invalid")
    return tuple(instrument.split("_", 1))  # type: ignore[return-value]


def _validate_proposal(payload: Mapping[str, Any]) -> None:
    _reject_producer_fields(payload)
    if set(payload) != {"schema_version", "candidate_key", "rows", "proposal_sha256"} \
            or payload.get("schema_version") != 1 \
            or payload.get("proposal_sha256") != embedded_hash(payload, "proposal_sha256"):
        raise VerificationError("proposal root identity mismatch")
    if not isinstance(payload.get("candidate_key"), str) or not payload["candidate_key"]:
        raise VerificationError("proposal candidate key missing")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise VerificationError("proposal rows missing")
    keys = {
        "proposal_ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns",
        "available_at_ns", "instrument", "direction", "notional_jpy_micros",
        "max_age_ns", "worker_key", "action",
    }
    order = []
    for ordinal, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != keys:
            raise VerificationError("proposal row schema mismatch")
        for field in (
            "proposal_ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns",
            "available_at_ns", "direction", "notional_jpy_micros", "max_age_ns",
        ):
            if not isinstance(row[field], int):
                raise VerificationError(f"proposal integer field invalid: {field}")
        if (
            row["proposal_ordinal"] != ordinal
            or row["direction"] not in {-1, 1}
            or row["notional_jpy_micros"] <= 0
            or row["max_age_ns"] <= 0
            or row["available_at_ns"] > row["decision_arrival_ts_ns"]
            or row["decision_arrival_ts_ns"] < row["decision_source_ts_ns"]
            or row["action"] != "ENTER"
        ):
            raise VerificationError("proposal chronology/action invalid")
        _pair(row["instrument"])
        order.append((row["decision_source_ts_ns"], row["decision_arrival_ts_ns"], ordinal))
    if order != sorted(order):
        raise VerificationError("proposal input chronology is not monotonic")


def _validate_policy(payload: Mapping[str, Any], identity: str, hash_field: str) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("policy_id") != identity
        or payload.get(hash_field) != embedded_hash(payload, hash_field)
    ):
        raise VerificationError(f"{identity} policy identity mismatch")


def _price(event: Mapping[str, Any], side: str) -> Decimal:
    scale = Decimal(event["tick_scale"])
    if side == "mid":
        return Decimal(event["bid_ticks"] + event["ask_ticks"]) / (scale * 2)
    if side not in {"bid", "ask"}:
        raise VerificationError("unknown price side")
    return Decimal(event[f"{side}_ticks"]) / scale


def _latest(
    books: Mapping[str, Sequence[Mapping[str, Any]]], instrument: str,
    source_ts_ns: int, arrival_ts_ns: int, staleness_ns: int,
) -> Mapping[str, Any]:
    rows = [
        row for row in books.get(instrument, ())
        if row["source_ts_ns"] <= source_ts_ns and row["arrival_ts_ns"] <= arrival_ts_ns
    ]
    if not rows:
        raise VerificationError(f"causal conversion BBO missing: {instrument}")
    row = rows[-1]
    if (
        source_ts_ns - row["source_ts_ns"] > staleness_ns
        or arrival_ts_ns - row["arrival_ts_ns"] > staleness_ns
    ):
        raise VerificationError(f"causal conversion BBO stale: {instrument}")
    return row


def _to_jpy(
    amount: Decimal, currency: str, source_ts_ns: int, arrival_ts_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]], staleness_ns: int,
) -> Decimal:
    if amount == 0 or currency == "JPY":
        return amount
    if currency == "USD":
        quote = _latest(books, "USD_JPY", source_ts_ns, arrival_ts_ns, staleness_ns)
        return amount * _price(quote, "bid" if amount > 0 else "ask")
    if currency in {"CAD", "CHF"}:
        quote = _latest(books, f"USD_{currency}", source_ts_ns, arrival_ts_ns, staleness_ns)
        usd = amount / _price(quote, "ask" if amount > 0 else "bid")
        return _to_jpy(usd, "USD", source_ts_ns, arrival_ts_ns, books, staleness_ns)
    raise VerificationError(f"unsupported quote currency: {currency}")


def _after(
    books: Mapping[str, Sequence[Mapping[str, Any]]], instrument: str,
    source_min: int, arrival_min: int, period_end: int, *, inclusive: bool,
) -> Mapping[str, Any] | None:
    for row in books.get(instrument, ()):
        source_ok = row["source_ts_ns"] >= source_min if inclusive else row["source_ts_ns"] > source_min
        if (
            source_ok and row["arrival_ts_ns"] >= arrival_min
            and row["source_ts_ns"] <= period_end and row["arrival_ts_ns"] <= period_end
        ):
            return row
    return None


def _terminal(
    books: Mapping[str, Sequence[Mapping[str, Any]]], instrument: str,
    period_end: int, staleness_ns: int,
) -> Mapping[str, Any]:
    rows = [
        row for row in books.get(instrument, ())
        if row["source_ts_ns"] <= period_end and row["arrival_ts_ns"] <= period_end
    ]
    if not rows:
        raise VerificationError("terminal BBO missing")
    row = rows[-1]
    if (
        period_end - row["source_ts_ns"] > staleness_ns
        or period_end - row["arrival_ts_ns"] > staleness_ns
    ):
        raise VerificationError("terminal BBO stale")
    return row


def _execution_price(
    event: Mapping[str, Any], direction: int, opening: bool, policy: Mapping[str, Any]
) -> Decimal:
    if policy["raw_mid"] is True:
        return _price(event, "mid")
    slip = Decimal(policy["slippage_ticks_per_side"]) / Decimal(event["tick_scale"])
    if opening:
        result = _price(event, "ask") + slip if direction > 0 else _price(event, "bid") - slip
    else:
        result = _price(event, "bid") - slip if direction > 0 else _price(event, "ask") + slip
    if result <= 0:
        raise VerificationError("execution price nonpositive")
    return result


def _floor_micros(value: Decimal) -> int:
    return int((value * JPY_MICROS_PER_YEN).to_integral_value(rounding=ROUND_FLOOR))


def _signal_id(candidate: str, row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes({
        "candidate_id": candidate,
        "proposal_ordinal": row["proposal_ordinal"],
        "decision_source_ts_ns": row["decision_source_ts_ns"],
        "decision_arrival_ts_ns": row["decision_arrival_ts_ns"],
        "instrument": row["instrument"],
        "direction": row["direction"],
        "worker_key": row["worker_key"],
    }))


def _expected_rejection(
    proposal: Mapping[str, Any], signal_id: str, arm: str, reason: str
) -> dict[str, Any]:
    return {
        "record_type": "ORACLE_DISPOSITION", "arm": arm, "signal_id": signal_id,
        "fill_id": None, "proposal_ordinal": proposal["proposal_ordinal"],
        "instrument": proposal["instrument"], "direction": proposal["direction"],
        "status": reason, "entry_disposition": reason,
        "exit_disposition": "NOT_APPLICABLE", "action_transitions": ["NO_ENTRY"],
        "notional_jpy_micros": proposal["notional_jpy_micros"], "units_micros": 0,
        "gross_pnl_jpy_micros": 0, "executable_pnl_jpy_micros": 0,
        "commission_jpy_micros": 0, "financing_jpy_micros": 0,
        "realized_cost_jpy_micros": 0, "net_pnl_jpy_micros": 0,
        "currency_inventory_after_close": {}, "terminal_inventory_mtm_jpy_micros": 0,
        "external_order_count": 0,
    }


def _expected_fill(
    proposal: Mapping[str, Any], signal_id: str, arm: str,
    policy: Mapping[str, Any], entry: Mapping[str, Any], exit_event: Mapping[str, Any],
    exit_reason: str, books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any], execution_hash: str,
) -> dict[str, Any]:
    staleness = accounting["max_conversion_staleness_ns"]
    entry_price = _execution_price(entry, proposal["direction"], True, policy)
    exit_price = _execution_price(exit_event, proposal["direction"], False, policy)
    _, quote = _pair(proposal["instrument"])
    quote_cash = Decimal(-proposal["direction"]) * entry_price
    jpy_per_base = abs(_to_jpy(
        quote_cash, quote, entry["source_ts_ns"], entry["arrival_ts_ns"], books, staleness
    ))
    if jpy_per_base <= 0:
        raise VerificationError("position sizing JPY value nonpositive")
    units_micros = int((
        Decimal(proposal["notional_jpy_micros"]) / JPY_MICROS_PER_YEN
        / jpy_per_base * BASE_MICROUNITS_PER_UNIT
    ).to_integral_value(rounding=ROUND_FLOOR))
    if units_micros <= 0:
        raise VerificationError("position sizing rounded to zero")
    units = Decimal(units_micros) / BASE_MICROUNITS_PER_UNIT
    quote_pnl = Decimal(proposal["direction"]) * units * (exit_price - entry_price)
    executable = _to_jpy(
        quote_pnl, quote, exit_event["source_ts_ns"], exit_event["arrival_ts_ns"],
        books, staleness,
    )
    raw_quote = Decimal(proposal["direction"]) * units * (
        _price(exit_event, "mid") - _price(entry, "mid")
    )
    raw = _to_jpy(
        raw_quote, quote, exit_event["source_ts_ns"], exit_event["arrival_ts_ns"],
        books, staleness,
    )
    elapsed = exit_event["source_ts_ns"] - entry["source_ts_ns"]
    notional = Decimal(proposal["notional_jpy_micros"])
    commission = int((
        notional * Decimal(2 * policy["commission_ppm_per_side"]) / Decimal(1_000_000)
    ).to_integral_value(rounding=ROUND_FLOOR))
    financing = int((
        notional * Decimal(policy["financing_ppm_per_day"]) * Decimal(elapsed)
        / Decimal(DAY_NS) / Decimal(1_000_000)
    ).to_integral_value(rounding=ROUND_FLOOR))
    gross = _floor_micros(raw)
    executable_micros = _floor_micros(executable)
    net = executable_micros - commission - financing
    fill_id = sha256_bytes(canonical_bytes({
        "signal_id": signal_id, "arm": arm,
        "entry_source_event_sha256": entry["source_event_sha256"],
        "execution_policy_sha256": execution_hash,
    }))
    return {
        "record_type": "ORACLE_DISPOSITION", "arm": arm, "signal_id": signal_id,
        "fill_id": fill_id, "proposal_ordinal": proposal["proposal_ordinal"],
        "instrument": proposal["instrument"], "direction": proposal["direction"],
        "status": "FILLED_CLOSED", "entry_disposition": "FILLED",
        "exit_disposition": exit_reason, "action_transitions": ["ENTER", "EXIT"],
        "notional_jpy_micros": proposal["notional_jpy_micros"],
        "units_micros": units_micros,
        "entry_price_ticks_x2": int(entry_price * entry["tick_scale"] * 2),
        "exit_price_ticks_x2": int(exit_price * exit_event["tick_scale"] * 2),
        "entry_tick_scale_x2": int(entry["tick_scale"] * 2),
        "exit_tick_scale_x2": int(exit_event["tick_scale"] * 2),
        "entry_source_ts_ns": entry["source_ts_ns"],
        "entry_arrival_ts_ns": entry["arrival_ts_ns"],
        "exit_source_ts_ns": exit_event["source_ts_ns"],
        "exit_arrival_ts_ns": exit_event["arrival_ts_ns"],
        "elapsed_ns": elapsed,
        "gross_pnl_jpy_micros": gross,
        "executable_pnl_jpy_micros": executable_micros,
        "commission_jpy_micros": commission,
        "financing_jpy_micros": financing,
        "realized_cost_jpy_micros": gross - net,
        "net_pnl_jpy_micros": net,
        "entry_source_reference": {
            "source_event_sha256": entry["source_event_sha256"],
            "provider_id": entry["provider_id"], "source_ts_ns": entry["source_ts_ns"],
            "arrival_ts_ns": entry["arrival_ts_ns"],
            "execution_policy_sha256": execution_hash,
        },
        "exit_source_reference": {
            "source_event_sha256": exit_event["source_event_sha256"],
            "provider_id": exit_event["provider_id"],
            "source_ts_ns": exit_event["source_ts_ns"],
            "arrival_ts_ns": exit_event["arrival_ts_ns"],
            "execution_policy_sha256": execution_hash,
        },
        "currency_inventory_after_close": {}, "terminal_inventory_mtm_jpy_micros": 0,
        "external_order_count": 0,
    }


def _recompute_dispositions(
    proposal_root: Mapping[str, Any], execution: Mapping[str, Any],
    inventory: Mapping[str, Any], accounting: Mapping[str, Any],
    evaluation: Mapping[str, Any], books: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        policy = execution["arms"][arm]
        open_positions: list[dict[str, Any]] = []
        for proposal in proposal_root["rows"]:
            signal = _signal_id(proposal_root["candidate_key"], proposal)
            entry = _after(
                books, proposal["instrument"], proposal["decision_source_ts_ns"],
                proposal["decision_arrival_ts_ns"] + policy["latency_ns"],
                evaluation["period_end_ts_ns"], inclusive=False,
            )
            if entry is None:
                all_rows.append(_expected_rejection(proposal, signal, arm, "NO_CAUSAL_FILL"))
                continue
            open_positions = [
                position for position in open_positions
                if position["exit_source_ts_ns"] > entry["source_ts_ns"]
            ]
            gross_after = sum(
                position["notional_jpy_micros"] for position in open_positions
            ) + proposal["notional_jpy_micros"]
            if gross_after > inventory["max_gross_notional_jpy_micros"]:
                all_rows.append(_expected_rejection(proposal, signal, arm, "GROSS_CAP_REJECTED"))
                continue
            if len(open_positions) >= inventory["max_open_positions"]:
                all_rows.append(_expected_rejection(proposal, signal, arm, "POSITION_CAP_REJECTED"))
                continue
            base, quote = _pair(proposal["instrument"])
            currency_after = {
                currency: sum(
                    position["notional_jpy_micros"] for position in open_positions
                    if currency in _pair(position["instrument"])
                ) + proposal["notional_jpy_micros"]
                for currency in (base, quote)
            }
            if any(
                value > inventory["max_currency_notional_jpy_micros"]
                for value in currency_after.values()
            ):
                all_rows.append(_expected_rejection(proposal, signal, arm, "CURRENCY_CAP_REJECTED"))
                continue
            target = entry["source_ts_ns"] + proposal["max_age_ns"]
            exit_event = _after(
                books, proposal["instrument"], target, target + policy["latency_ns"],
                evaluation["period_end_ts_ns"], inclusive=True,
            )
            reason = "FINITE_MAX_AGE"
            if exit_event is None:
                exit_event = _terminal(
                    books, proposal["instrument"], evaluation["period_end_ts_ns"],
                    accounting["max_conversion_staleness_ns"],
                )
                if exit_event["source_ts_ns"] < entry["source_ts_ns"]:
                    raise VerificationError("terminal event precedes entry")
                reason = "TERMINAL_LIQUIDATION"
            expected = _expected_fill(
                proposal, signal, arm, policy, entry, exit_event, reason, books,
                accounting, execution["execution_policy_sha256"],
            )
            expected["gross_open_notional_after_entry_jpy_micros"] = gross_after
            expected["currency_notional_after_entry_jpy_micros"] = currency_after
            all_rows.append(expected)
            open_positions.append({
                "instrument": proposal["instrument"],
                "notional_jpy_micros": proposal["notional_jpy_micros"],
                "exit_source_ts_ns": exit_event["source_ts_ns"],
            })
    return sorted(all_rows, key=lambda row: (row["proposal_ordinal"], ARMS.index(row["arm"])))


def _validate_economic_policies(
    execution: Mapping[str, Any], inventory: Mapping[str, Any],
    accounting: Mapping[str, Any], evaluation: Mapping[str, Any],
) -> None:
    _validate_policy(execution, "FROZEN_EXECUTION_POLICY_V1", "execution_policy_sha256")
    _validate_policy(inventory, "FROZEN_INVENTORY_POLICY_V1", "inventory_policy_sha256")
    _validate_policy(accounting, "FROZEN_ACCOUNTING_POLICY_V1", "accounting_policy_sha256")
    _validate_policy(evaluation, "FROZEN_EVALUATION_POLICY_V1", "evaluation_policy_sha256")
    if set(execution.get("arms", {})) != set(ARMS):
        raise VerificationError("execution arm set mismatch")
    arm_keys = {
        "latency_ns", "slippage_ticks_per_side", "commission_ppm_per_side",
        "financing_ppm_per_day", "raw_mid",
    }
    for arm in ARMS:
        policy = execution["arms"][arm]
        if not isinstance(policy, dict) or set(policy) != arm_keys:
            raise VerificationError("execution arm policy schema mismatch")
        if any(
            not isinstance(policy[field], int) or policy[field] < 0
            for field in arm_keys - {"raw_mid"}
        ) or not isinstance(policy["raw_mid"], bool):
            raise VerificationError("execution arm policy value invalid")
    raw = execution["arms"]["RAW_SIGNAL"]
    if raw["raw_mid"] is not True or any(
        raw[field] != 0 for field in arm_keys - {"raw_mid"}
    ):
        raise VerificationError("RAW arm is not cost-independent")
    inventory_keys = {
        "schema_version", "policy_id", "max_gross_notional_jpy_micros",
        "max_currency_notional_jpy_micros", "max_open_positions",
        "same_pair_collision", "terminal_liquidation", "inventory_policy_sha256",
    }
    if set(inventory) != inventory_keys or any(
        not isinstance(inventory[field], int) or isinstance(inventory[field], bool)
        or inventory[field] <= 0
        for field in (
            "max_gross_notional_jpy_micros", "max_currency_notional_jpy_micros",
            "max_open_positions",
        )
    ) or inventory["same_pair_collision"] != "REJECT_NEW" \
            or inventory["terminal_liquidation"] is not True:
        raise VerificationError("inventory policy schema/value mismatch")
    accounting_keys = {
        "schema_version", "policy_id", "jpy_micros_per_yen",
        "base_microunits_per_unit", "max_conversion_staleness_ns",
        "supported_quote_currencies", "asset_conversion_side",
        "liability_conversion_side", "accounting_policy_sha256",
    }
    if set(accounting) != accounting_keys \
            or accounting["jpy_micros_per_yen"] != JPY_MICROS_PER_YEN \
            or accounting["base_microunits_per_unit"] != BASE_MICROUNITS_PER_UNIT \
            or not isinstance(accounting["max_conversion_staleness_ns"], int) \
            or accounting["max_conversion_staleness_ns"] <= 0 \
            or accounting["supported_quote_currencies"] != ["CAD", "CHF", "JPY", "USD"] \
            or accounting["asset_conversion_side"] != "BID" \
            or accounting["liability_conversion_side"] != "ASK":
        raise VerificationError("accounting policy schema/value mismatch")
    evaluation_keys = {
        "schema_version", "policy_id", "period_start_ts_ns", "period_end_ts_ns",
        "initial_equity_jpy_micros", "margin_notional_cap_jpy_micros",
        "cvar_tail_bps", "holdout_state", "evaluation_policy_sha256",
    }
    if set(evaluation) != evaluation_keys or any(
        not isinstance(evaluation[field], int) or isinstance(evaluation[field], bool)
        for field in (
            "period_start_ts_ns", "period_end_ts_ns", "initial_equity_jpy_micros",
            "margin_notional_cap_jpy_micros", "cvar_tail_bps",
        )
    ) or evaluation["period_start_ts_ns"] >= evaluation["period_end_ts_ns"] \
            or evaluation["initial_equity_jpy_micros"] <= 0 \
            or evaluation["margin_notional_cap_jpy_micros"] <= 0 \
            or not 0 < evaluation["cvar_tail_bps"] <= 10_000 \
            or evaluation["holdout_state"] != "UNOPENED":
        raise VerificationError("evaluation policy schema/value mismatch")


def _parse_ledger(data: bytes, source_index: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    if data and not data.endswith(b"\n"):
        raise VerificationError("oracle ledger truncated")
    rows = []
    previous = ZERO_SHA
    identities = set()
    for sequence, line in enumerate(data.splitlines(), 1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError("oracle ledger JSON invalid") from error
        if not isinstance(row, dict):
            raise VerificationError("oracle ledger row is not object")
        assert_no_float(row, "ledger")
        if (
            row.get("ledger_schema_version") != 1
            or row.get("ledger_sequence") != sequence
            or row.get("previous_hash") != previous
            or row.get("record_hash") != embedded_hash(row, "record_hash")
        ):
            raise VerificationError("oracle ledger hash chain mismatch")
        if row.get("arm") not in ARMS or not isinstance(row.get("signal_id"), str):
            raise VerificationError("oracle disposition identity invalid")
        identity = (row["signal_id"], row["arm"])
        if identity in identities:
            raise VerificationError("duplicate arm disposition")
        identities.add(identity)
        if row.get("status") == "FILLED_CLOSED":
            for role in ("entry_source_reference", "exit_source_reference"):
                reference = row.get(role)
                if not isinstance(reference, dict):
                    raise VerificationError("fill source reference missing")
                source = source_index.get(reference.get("source_event_sha256"))
                if source is None:
                    raise VerificationError("fill references unknown source event")
                for field in ("provider_id", "source_ts_ns", "arrival_ts_ns"):
                    if reference.get(field) != source.get(field):
                        raise VerificationError("fill source reference changed")
            if row.get("currency_inventory_after_close") != {} \
                    or row.get("terminal_inventory_mtm_jpy_micros") != 0:
                raise VerificationError("closed disposition retains inventory")
        if row.get("external_order_count") != 0:
            raise VerificationError("external order count is nonzero")
        rows.append(row)
        previous = row["record_hash"]
    if not rows:
        raise VerificationError("oracle ledger empty")
    return rows


def reaggregate(rows: Sequence[Mapping[str, Any]], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    initial = evaluation["initial_equity_jpy_micros"]
    tail_bps = evaluation["cvar_tail_bps"]
    metrics: dict[str, Any] = {
        "initial_equity_jpy_micros": initial,
        "monthly_multiples": {},
        "arms": {},
        "external_orders": 0,
        "terminal_inventory_mtm_jpy_micros": 0,
    }
    signal_sets = {}
    action_lists = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        signals = sorted(row["signal_id"] for row in arm_rows)
        if len(signals) != len(set(signals)):
            raise VerificationError("duplicate signal in arm")
        signal_sets[arm] = signals
        action_lists[arm] = [(row["signal_id"], row["action_transitions"]) for row in arm_rows]
        executed = [row for row in arm_rows if row["status"] == "FILLED_CLOSED"]
        realized = sorted(executed, key=lambda row: (row["exit_source_ts_ns"], row["signal_id"]))
        equity = initial
        peak = initial
        max_drawdown = 0
        month_pnl: dict[str, int] = {}
        for row in realized:
            equity += row["net_pnl_jpy_micros"]
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
            key = _month(row["exit_source_ts_ns"])
            month_pnl[key] = month_pnl.get(key, 0) + row["net_pnl_jpy_micros"]
        monthly = {}
        month_equity = initial
        for key in sorted(month_pnl):
            next_equity = month_equity + month_pnl[key]
            monthly[key] = _ratio(next_equity, month_equity)
            month_equity = next_equity
        values = sorted(row["net_pnl_jpy_micros"] for row in executed)
        tail_count = max(1, (len(values) * tail_bps + 9_999) // 10_000) if values else 0
        cvar = sum(values[:tail_count]) // tail_count if tail_count else 0
        positives = sum(row["gross_pnl_jpy_micros"] > 0 for row in executed)
        max_gross = max(
            (row.get("gross_open_notional_after_entry_jpy_micros", 0) for row in executed),
            default=0,
        )
        clusters = {
            (
                row["instrument"],
                row.get("entry_source_ts_ns", evaluation["period_start_ts_ns"]) // 3_600_000_000_000,
            )
            for row in executed
        }
        net_sum = sum(row["net_pnl_jpy_micros"] for row in executed)
        metrics["arms"][arm] = {
            "proposal_count": len(arm_rows),
            "executed_count": len(executed),
            "disposition_counts": dict(sorted(Counter(row["status"] for row in arm_rows).items())),
            "signal_id_set_sha256": sha256_bytes(canonical_bytes(signals)),
            "action_transition_sha256": sha256_bytes(canonical_bytes(action_lists[arm])),
            "gross_pnl_jpy_micros": sum(row["gross_pnl_jpy_micros"] for row in executed),
            "realized_cost_jpy_micros": sum(row["realized_cost_jpy_micros"] for row in executed),
            "net_pnl_jpy_micros": net_sum,
            "ending_equity_jpy_micros": initial + net_sum,
            "ending_equity_multiple": _ratio(initial + net_sum, initial),
            "direction_accuracy": _ratio(positives, len(executed)) if executed else "0.000000000000000000",
            "max_drawdown_jpy_micros": max_drawdown,
            "max_drawdown_ratio": _ratio(max_drawdown, initial),
            "cvar_tail_bps": tail_bps,
            "cvar_jpy_micros": cvar,
            "max_gross_notional_jpy_micros": max_gross,
            "currency_time_cluster_count": len(clusters),
            "terminal_open_positions": 0,
            "terminal_inventory_mtm_jpy_micros": 0,
            "margin_guard_pass": max_gross <= evaluation["margin_notional_cap_jpy_micros"],
        }
        metrics["monthly_multiples"][arm] = monthly
    if len({tuple(signal_sets[arm]) for arm in ARMS}) != 1:
        raise VerificationError("arm signal sets differ")
    metrics["same_signal_ids_all_arms"] = True
    metrics["all_proposals_have_all_arm_dispositions"] = True
    metrics["action_label_contract_all_arms"] = all(
        [labels for _, labels in action_lists[arm]]
        == [labels for _, labels in action_lists[ARMS[0]]]
        for arm in ARMS[1:]
    )
    metrics["metrics_sha256"] = embedded_hash(metrics, "metrics_sha256")
    return metrics


def _exclusive_json(path: Path, payload: Mapping[str, Any], *, output_root: Path) -> None:
    path = _bounded_path(output_root, str(path), "verifier output", must_exist=False)
    if path.exists():
        raise VerificationError("verifier receipt already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        data = canonical_bytes(payload) + b"\n"
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def verify(request: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "input_root", "output_root", "oracle_manifest",
        "oracle_ledger", "source_blob", "source_manifest", "proposal",
        "execution_policy", "inventory_policy", "accounting_policy",
        "evaluation_policy", "receipt_name",
    }
    if set(request) != required or request.get("schema_version") != 1:
        raise VerificationError("verifier request schema mismatch")
    input_root = _secure_root(request["input_root"], "input")
    output_root = _secure_root(request["output_root"], "output")
    artifacts = {}
    for label in (
        "oracle_manifest", "oracle_ledger", "source_blob", "source_manifest",
        "proposal", "execution_policy", "inventory_policy", "accounting_policy",
        "evaluation_policy",
    ):
        artifacts[label] = _artifact(request[label], label, input_root=input_root)
    manifest_bytes = artifacts["oracle_manifest"][1]
    ledger_bytes = artifacts["oracle_ledger"][1]
    source_bytes = artifacts["source_blob"][1]
    evaluation_bytes = artifacts["evaluation_policy"][1]
    manifest = _json(manifest_bytes, "oracle manifest")
    source_manifest = _json(artifacts["source_manifest"][1], "source manifest")
    proposal = _json(artifacts["proposal"][1], "proposal")
    execution = _json(artifacts["execution_policy"][1], "execution policy")
    inventory = _json(artifacts["inventory_policy"][1], "inventory policy")
    accounting = _json(artifacts["accounting_policy"][1], "accounting policy")
    evaluation = _json(evaluation_bytes, "evaluation policy")
    if (
        manifest.get("oracle_implementation") != ORACLE_NAME
        or manifest.get("status") != "COMPLETE"
        or manifest.get("producer_result_or_metrics_used") is not False
        or manifest.get("anchor_status") != "LOCAL_REPRODUCIBLE"
        or manifest.get("oracle_root_sha256") != embedded_hash(manifest, "oracle_root_sha256")
    ):
        raise VerificationError("oracle manifest identity invalid")
    root = Path(__file__).resolve().parent
    expected_source_hashes = {
        "oracle_code_sha256": sha256_bytes(_read(root / ORACLE_FILE, root=root)),
        "oracle_contract_sha256": sha256_bytes(_read(root / CONTRACT_FILE, root=root)),
        "oracle_schema_sha256": sha256_bytes(_read(root / SCHEMA_FILE, root=root)),
    }
    if any(manifest.get(key) != value for key, value in expected_source_hashes.items()):
        raise VerificationError("oracle implementation source hash mismatch")
    input_hashes = manifest.get("input_artifact_sha256")
    required_input_names = {
        "source_blob", "source_manifest", "proposal", "execution_policy",
        "inventory_policy", "accounting_policy", "evaluation_policy",
    }
    if not isinstance(input_hashes, dict) or set(input_hashes) != required_input_names:
        raise VerificationError("oracle input binding set mismatch")
    if any(input_hashes[name] != request[name]["sha256"] for name in required_input_names):
        raise VerificationError("oracle manifest input binding mismatch")
    if (
        manifest.get("oracle_ledger_sha256") != sha256_bytes(ledger_bytes)
        or manifest.get("oracle_ledger_row_count") != len(ledger_bytes.splitlines())
    ):
        raise VerificationError("oracle manifest ledger binding mismatch")
    _validate_proposal(proposal)
    _validate_economic_policies(execution, inventory, accounting, evaluation)
    if any(
        row["decision_source_ts_ns"] < evaluation["period_start_ts_ns"]
        or row["decision_source_ts_ns"] >= evaluation["period_end_ts_ns"]
        for row in proposal["rows"]
    ):
        raise VerificationError("proposal outside frozen evaluation period")
    source_index, books = _parse_source(source_bytes, source_manifest)
    rows = _parse_ledger(ledger_bytes, source_index)
    if manifest.get("oracle_ledger_terminal_hash") != rows[-1]["record_hash"]:
        raise VerificationError("oracle ledger terminal root mismatch")
    expected_rows = _recompute_dispositions(
        proposal, execution, inventory, accounting, evaluation, books
    )
    actual_dispositions = [
        {
            key: value for key, value in row.items()
            if key not in {"ledger_schema_version", "ledger_sequence", "previous_hash", "record_hash"}
        }
        for row in rows
    ]
    if actual_dispositions != expected_rows:
        raise VerificationError("oracle ledger differs from independent economic replay")
    metrics = reaggregate(rows, evaluation)
    if manifest.get("oracle_metrics") != metrics:
        raise VerificationError("oracle self metrics differ from verifier reaggregation")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "verifier_implementation": VERIFIER_NAME,
        "oracle_implementation": ORACLE_NAME,
        "status": "VERIFIED",
        "oracle_root_sha256": manifest["oracle_root_sha256"],
        "oracle_ledger_sha256": sha256_bytes(ledger_bytes),
        "source_blob_sha256": sha256_bytes(source_bytes),
        "source_manifest_sha256": sha256_bytes(artifacts["source_manifest"][1]),
        "proposal_sha256": sha256_bytes(artifacts["proposal"][1]),
        "execution_policy_sha256": sha256_bytes(artifacts["execution_policy"][1]),
        "inventory_policy_sha256": sha256_bytes(artifacts["inventory_policy"][1]),
        "accounting_policy_sha256": sha256_bytes(artifacts["accounting_policy"][1]),
        "evaluation_policy_sha256": sha256_bytes(evaluation_bytes),
        "verifier_code_sha256": sha256_bytes(_read(Path(__file__).resolve(), root=root)),
        "verifier_schema_sha256": sha256_bytes(
            _read(root / VERIFIER_SCHEMA_FILE, root=root)
        ),
        "producer_metrics_used": False,
        "independent_fill_cost_inventory_jpy_replay": True,
        "verified_oracle_metrics": metrics,
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_orders": 0,
        "anchor_status": "LOCAL_REPRODUCIBLE",
    }
    receipt["verifier_receipt_sha256"] = embedded_hash(receipt, "verifier_receipt_sha256")
    receipt_name = request["receipt_name"]
    if not isinstance(receipt_name, str) or SAFE_NAME_RE.fullmatch(receipt_name) is None:
        raise VerificationError("verifier receipt name invalid")
    receipt_path = output_root / receipt_name
    _exclusive_json(receipt_path, receipt, output_root=output_root)
    return receipt


def _audit_hook(event: str, _: tuple[Any, ...]) -> None:
    if event.startswith(("socket.", "subprocess.")) or event in {
        "os.system", "os.posix_spawn", "os.exec", "os.spawn",
    }:
        raise VerificationError(f"runtime capability denied: {event}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request_path = args.request.resolve(strict=True)
    request = _json(_read(request_path, root=request_path.parent), "verifier request")
    receipt = verify(request)
    print(json.dumps({
        "ok": True,
        "verifier_receipt_sha256": receipt["verifier_receipt_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.addaudithook(_audit_hook)
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(json.dumps({
            "ok": False,
            "error_code": "VERIFIER_FAIL_CLOSED",
            "error_sha256": sha256_bytes(str(error).encode()),
        }, sort_keys=True))
        raise SystemExit(2)

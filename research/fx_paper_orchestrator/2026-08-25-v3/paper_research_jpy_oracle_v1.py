#!/usr/bin/env python3
"""Independent, file-only JPY economic oracle for future paper cycles.

The implementation intentionally imports none of the producer runner, legacy
accounting modules, or result validators.  It reads exact frozen bytes, creates
its own identifiers, and emits one hash-chained disposition for every
proposal/arm combination using integer ticks, base microunits, JPY micros, and
Decimal arithmetic.
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
from typing import Any, Iterable, Mapping, Sequence


ORACLE_NAME = "INDEPENDENT_JPY_ORACLE_V1"
CONTRACT_NAME = "PAPER_RESEARCH_JPY_ORACLE_CONTRACT_V1.json"
SCHEMA_NAME = "paper_research_jpy_oracle_schema_v1.json"
ARMS = ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")
ZERO_SHA = "0" * 64
DAY_NS = 86_400_000_000_000
JPY_MICROS_PER_YEN = 1_000_000
BASE_MICROUNITS_PER_UNIT = 1_000_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
SAFE_OUTPUT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
FORBIDDEN_PROPOSAL_TOKENS = {
    "signalid", "fill", "fillprice", "path", "mfe", "mae", "pnl", "cost",
    "equity", "drawdown", "dd", "cvar", "profit", "return",
}


class OracleError(RuntimeError):
    """Fail-closed independent-oracle violation."""


def canonical_bytes(value: Any) -> bytes:
    assert_no_float(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def assert_no_float(value: Any, *, location: str = "root") -> None:
    if isinstance(value, float):
        raise OracleError(f"float forbidden at {location}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_float(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_float(item, location=f"{location}[{index}]")


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
        raise OracleError(f"{label} root missing")
    path = Path(value)
    if not path.is_absolute():
        raise OracleError(f"{label} root must be absolute")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise OracleError(f"regular non-symlink {label} root required")
    return Path(os.path.abspath(path))


def _bounded_path(root: Path, value: Any, label: str, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise OracleError(f"{label} path missing")
    path = Path(value)
    if not path.is_absolute():
        raise OracleError(f"{label} path must be absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise OracleError(f"{label} path escapes capability root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise OracleError(f"{label} path is not a canonical child")
    current = root
    parts = relative.parts if must_exist else relative.parts[:-1]
    for part in parts:
        current = current / part
        current_stat = os.lstat(current)
        if stat.S_ISLNK(current_stat.st_mode):
            raise OracleError(f"{label} path contains symlink")
        if current != path and not stat.S_ISDIR(current_stat.st_mode):
            raise OracleError(f"{label} parent is not a directory")
    if must_exist:
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root.resolve(strict=True))
        except ValueError as error:
            raise OracleError(f"{label} resolved outside capability root") from error
    return path


def _secure_read(path: Path, *, root: Path) -> bytes:
    path = _bounded_path(root, str(path), "input artifact", must_exist=True)
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OracleError(f"regular non-symlink artifact required: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise OracleError(f"artifact identity changed: {path.name}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    final = os.lstat(path)
    if (before.st_dev, before.st_ino, before.st_size) != (
        final.st_dev, final.st_ino, final.st_size
    ):
        raise OracleError(f"artifact changed during read: {path.name}")
    return b"".join(chunks)


def _exclusive_bytes(path: Path, value: bytes) -> None:
    if path.parent.exists():
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise OracleError("secure output directory required")
    else:
        os.mkdir(path.parent, 0o700)
        _fsync_directory(path.parent.parent)
        _fsync_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OracleError("short immutable output write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _artifact_bytes(
    spec: Mapping[str, Any], label: str, *, input_root: Path
) -> tuple[Path, bytes]:
    if set(spec) != {"path", "sha256", "size_bytes"}:
        raise OracleError(f"{label} artifact schema mismatch")
    path = _bounded_path(input_root, spec.get("path"), label, must_exist=True)
    data = _secure_read(path, root=input_root)
    if not isinstance(spec["size_bytes"], int) or spec["size_bytes"] != len(data):
        raise OracleError(f"{label} artifact size mismatch")
    if not isinstance(spec["sha256"], str) or sha256_bytes(data) != spec["sha256"]:
        raise OracleError(f"{label} artifact hash mismatch")
    return path, data


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OracleError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise OracleError(f"{label} must be an object")
    assert_no_float(value, location=label)
    return value


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _validate_proposal_tree(value: Any, location: str = "proposal") -> None:
    if isinstance(value, float):
        raise OracleError(f"proposal float forbidden at {location}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise OracleError(f"proposal key is not text at {location}")
            token = _normalize_key(key)
            if any(forbidden in token for forbidden in FORBIDDEN_PROPOSAL_TOKENS):
                raise OracleError(f"proposal outcome/identifier forbidden at {location}.{key}")
            _validate_proposal_tree(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_proposal_tree(item, f"{location}[{index}]")
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise OracleError(f"proposal type forbidden at {location}")


def _validate_embedded(payload: Mapping[str, Any], field: str, label: str) -> None:
    digest = payload.get(field)
    if not isinstance(digest, str) or digest != embedded_hash(payload, field):
        raise OracleError(f"{label} embedded hash mismatch")


def _event_price(event: Mapping[str, Any], *, side: str) -> Decimal:
    scale = Decimal(int(event["tick_scale"]))
    if side == "bid":
        ticks = int(event["bid_ticks"])
    elif side == "ask":
        ticks = int(event["ask_ticks"])
    elif side == "mid":
        return Decimal(int(event["bid_ticks"]) + int(event["ask_ticks"])) / (scale * 2)
    else:
        raise OracleError(f"unknown BBO side: {side}")
    return Decimal(ticks) / scale


def _pair_currencies(instrument: str) -> tuple[str, str]:
    if not isinstance(instrument, str) or PAIR_RE.fullmatch(instrument) is None:
        raise OracleError(f"invalid FX instrument: {instrument}")
    base, quote = instrument.split("_", 1)
    return base, quote


def _parse_source(
    blob: bytes, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    expected_manifest_keys = {
        "schema_version", "source_bytes_sha256", "source_size_bytes", "event_count",
        "first_source_ts_ns", "last_source_ts_ns", "manifest_sha256",
    }
    if set(manifest) != expected_manifest_keys or manifest.get("schema_version") != 1:
        raise OracleError("source manifest schema mismatch")
    _validate_embedded(manifest, "manifest_sha256", "source manifest")
    if (
        manifest["source_bytes_sha256"] != sha256_bytes(blob)
        or manifest["source_size_bytes"] != len(blob)
    ):
        raise OracleError("source manifest does not bind exact BBO bytes")
    if blob and not blob.endswith(b"\n"):
        raise OracleError("truncated source BBO record")
    rows: list[dict[str, Any]] = []
    last_stream: dict[tuple[str, str], tuple[int, int, int]] = {}
    expected_keys = {
        "schema_version", "provider_id", "instrument", "bid_ticks", "ask_ticks",
        "tick_scale", "source_ts_ns", "arrival_ts_ns", "provider_event_id",
        "sequence", "heartbeat", "quality_flags",
    }
    for raw_line in blob.splitlines(keepends=True):
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OracleError("invalid source BBO record") from error
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise OracleError("source BBO schema mismatch")
        assert_no_float(row, location="source")
        if row["schema_version"] != 1 or row["heartbeat"] is not False:
            raise OracleError("priced source event required")
        if not isinstance(row["quality_flags"], list) or row["quality_flags"]:
            raise OracleError("quality-flagged source event is unavailable")
        for key in (
            "bid_ticks", "ask_ticks", "tick_scale", "source_ts_ns", "arrival_ts_ns",
            "sequence",
        ):
            if not isinstance(row[key], int) or isinstance(row[key], bool):
                raise OracleError(f"source integer field invalid: {key}")
        if (
            row["bid_ticks"] <= 0
            or row["ask_ticks"] <= row["bid_ticks"]
            or row["tick_scale"] <= 0
            or row["arrival_ts_ns"] < row["source_ts_ns"]
        ):
            raise OracleError("invalid executable BBO")
        if not isinstance(row["provider_id"], str) or not row["provider_id"] \
                or not isinstance(row["instrument"], str) \
                or row["sequence"] <= 0 \
                or not isinstance(row["provider_event_id"], (str, type(None))):
            raise OracleError("source identity field invalid")
        _pair_currencies(row["instrument"])
        stream_key = (row["provider_id"], row["instrument"])
        stream_state = (
            row["source_ts_ns"], row["arrival_ts_ns"], row["sequence"]
        )
        prior = last_stream.get(stream_key)
        if prior is not None and any(
            stream_state[index] <= prior[index] for index in range(3)
        ):
            raise OracleError("source stream input order is not strictly increasing")
        last_stream[stream_key] = stream_state
        row["source_event_sha256"] = sha256_bytes(raw_line)
        rows.append(row)
    if len(rows) != manifest["event_count"] or not rows:
        raise OracleError("source event count mismatch")
    if (
        min(row["source_ts_ns"] for row in rows) != manifest["first_source_ts_ns"]
        or max(row["source_ts_ns"] for row in rows) != manifest["last_source_ts_ns"]
    ):
        raise OracleError("source manifest time boundary mismatch")
    books: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        books.setdefault(row["instrument"], []).append(row)
    return rows, books


def _latest_causal(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    instrument: str,
    source_ts_ns: int,
    arrival_ts_ns: int,
    max_staleness_ns: int,
) -> Mapping[str, Any]:
    candidates = [
        row for row in books.get(instrument, ())
        if row["source_ts_ns"] <= source_ts_ns and row["arrival_ts_ns"] <= arrival_ts_ns
    ]
    if not candidates:
        raise OracleError(f"missing causal conversion BBO: {instrument}")
    row = candidates[-1]
    if (
        source_ts_ns - row["source_ts_ns"] > max_staleness_ns
        or arrival_ts_ns - row["arrival_ts_ns"] > max_staleness_ns
    ):
        raise OracleError(f"stale causal conversion BBO: {instrument}")
    return row


def _convert_to_jpy(
    amount: Decimal,
    currency: str,
    source_ts_ns: int,
    arrival_ts_ns: int,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    max_staleness_ns: int,
) -> Decimal:
    if amount == 0:
        return Decimal(0)
    if currency == "JPY":
        return amount
    if currency == "USD":
        quote = _latest_causal(
            books, "USD_JPY", source_ts_ns, arrival_ts_ns, max_staleness_ns
        )
        return amount * _event_price(quote, side="bid" if amount > 0 else "ask")
    if currency in {"CAD", "CHF"}:
        pair = f"USD_{currency}"
        quote = _latest_causal(
            books, pair, source_ts_ns, arrival_ts_ns, max_staleness_ns
        )
        usd = amount / _event_price(quote, side="ask" if amount > 0 else "bid")
        return _convert_to_jpy(
            usd, "USD", source_ts_ns, arrival_ts_ns, books, max_staleness_ns
        )
    raise OracleError(f"unsupported JPY conversion currency: {currency}")


def _decimal_to_jpy_micros(value: Decimal) -> int:
    return int((value * JPY_MICROS_PER_YEN).to_integral_value(rounding=ROUND_FLOOR))


def _ratio_text(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise OracleError("ratio denominator must be positive")
    with localcontext() as context:
        context.prec = 50
        return format(Decimal(numerator) / Decimal(denominator), ".18f")


def _event_after(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    instrument: str,
    *,
    minimum_source_ts_ns: int,
    minimum_arrival_ts_ns: int,
    period_end_ts_ns: int,
) -> Mapping[str, Any] | None:
    return next((
        row for row in books.get(instrument, ())
        if row["source_ts_ns"] > minimum_source_ts_ns
        and row["arrival_ts_ns"] >= minimum_arrival_ts_ns
        and row["source_ts_ns"] <= period_end_ts_ns
        and row["arrival_ts_ns"] <= period_end_ts_ns
    ), None)


def _event_at_or_after(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    instrument: str,
    *,
    minimum_source_ts_ns: int,
    minimum_arrival_ts_ns: int,
    period_end_ts_ns: int,
) -> Mapping[str, Any] | None:
    return next((
        row for row in books.get(instrument, ())
        if row["source_ts_ns"] >= minimum_source_ts_ns
        and row["arrival_ts_ns"] >= minimum_arrival_ts_ns
        and row["source_ts_ns"] <= period_end_ts_ns
        and row["arrival_ts_ns"] <= period_end_ts_ns
    ), None)


def _terminal_event(
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    instrument: str,
    *,
    period_end_ts_ns: int,
    max_staleness_ns: int,
) -> Mapping[str, Any]:
    candidates = [
        row for row in books.get(instrument, ())
        if row["source_ts_ns"] <= period_end_ts_ns
        and row["arrival_ts_ns"] <= period_end_ts_ns
    ]
    if not candidates:
        raise OracleError(f"terminal BBO missing: {instrument}")
    row = candidates[-1]
    if (
        period_end_ts_ns - row["source_ts_ns"] > max_staleness_ns
        or period_end_ts_ns - row["arrival_ts_ns"] > max_staleness_ns
    ):
        raise OracleError(f"terminal BBO stale: {instrument}")
    return row


def _execution_price(
    event: Mapping[str, Any], direction: int, *, opening: bool, arm_policy: Mapping[str, Any]
) -> Decimal:
    if arm_policy["raw_mid"] is True:
        return _event_price(event, side="mid")
    slip = Decimal(arm_policy["slippage_ticks_per_side"]) / Decimal(event["tick_scale"])
    if opening:
        price = (
            _event_price(event, side="ask") + slip
            if direction > 0
            else _event_price(event, side="bid") - slip
        )
    else:
        price = (
            _event_price(event, side="bid") - slip
            if direction > 0
            else _event_price(event, side="ask") + slip
        )
    if price <= 0:
        raise OracleError("slippage produced nonpositive price")
    return price


def _position_units_micros(
    proposal: Mapping[str, Any],
    entry: Mapping[str, Any],
    entry_price: Decimal,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    max_staleness_ns: int,
) -> int:
    _, quote_currency = _pair_currencies(proposal["instrument"])
    quote_cashflow_per_base = Decimal(-proposal["direction"]) * entry_price
    jpy_per_base = abs(_convert_to_jpy(
        quote_cashflow_per_base,
        quote_currency,
        entry["source_ts_ns"],
        entry["arrival_ts_ns"],
        books,
        max_staleness_ns,
    ))
    if jpy_per_base <= 0:
        raise OracleError("position sizing JPY value is nonpositive")
    notional_yen = Decimal(proposal["notional_jpy_micros"]) / JPY_MICROS_PER_YEN
    units = notional_yen / jpy_per_base
    units_micros = int(
        (units * BASE_MICROUNITS_PER_UNIT).to_integral_value(rounding=ROUND_FLOOR)
    )
    if units_micros <= 0:
        raise OracleError("position sizing rounded to zero")
    return units_micros


def _signal_id(candidate_id: str, proposal: Mapping[str, Any]) -> str:
    identity = {
        "candidate_id": candidate_id,
        "proposal_ordinal": proposal["proposal_ordinal"],
        "decision_source_ts_ns": proposal["decision_source_ts_ns"],
        "decision_arrival_ts_ns": proposal["decision_arrival_ts_ns"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "worker_key": proposal["worker_key"],
    }
    return sha256_bytes(canonical_bytes(identity))


def _validate_policy(
    payload: Mapping[str, Any], *, expected_id: str, hash_field: str
) -> None:
    if payload.get("schema_version") != 1 or payload.get("policy_id") != expected_id:
        raise OracleError(f"{expected_id} policy identity mismatch")
    _validate_embedded(payload, hash_field, expected_id)


def _validate_inputs(
    request: Mapping[str, Any]
) -> tuple[
    list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any],
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str], Path,
]:
    required = {
        "schema_version", "input_root", "output_root", "source_blob", "source_manifest", "proposal",
        "execution_policy", "inventory_policy", "accounting_policy",
        "evaluation_policy", "output_directory",
    }
    if set(request) != required or request.get("schema_version") != 1:
        raise OracleError("oracle request schema mismatch")
    input_root = _secure_root(request["input_root"], "input")
    output_root = _secure_root(request["output_root"], "output")
    artifacts: dict[str, tuple[Path, bytes]] = {}
    hashes: dict[str, str] = {}
    for label in (
        "source_blob", "source_manifest", "proposal", "execution_policy",
        "inventory_policy", "accounting_policy", "evaluation_policy",
    ):
        artifacts[label] = _artifact_bytes(request[label], label, input_root=input_root)
        hashes[label] = request[label]["sha256"]
    source_manifest = _json_object(artifacts["source_manifest"][1], "source manifest")
    rows, books = _parse_source(artifacts["source_blob"][1], source_manifest)
    proposal = _json_object(artifacts["proposal"][1], "proposal")
    _validate_proposal_tree(proposal)
    if set(proposal) != {
        "schema_version", "candidate_key", "rows", "proposal_sha256"
    } or proposal.get("schema_version") != 1:
        raise OracleError("proposal root schema mismatch")
    _validate_embedded(proposal, "proposal_sha256", "proposal")
    if not isinstance(proposal["candidate_key"], str) or not proposal["candidate_key"]:
        raise OracleError("candidate key missing")
    if not isinstance(proposal["rows"], list) or not proposal["rows"]:
        raise OracleError("proposal rows missing")
    proposal_keys = {
        "proposal_ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns",
        "available_at_ns", "instrument", "direction", "notional_jpy_micros",
        "max_age_ns", "worker_key", "action",
    }
    ordinals = []
    for row in proposal["rows"]:
        if not isinstance(row, dict) or set(row) != proposal_keys:
            raise OracleError("proposal row schema mismatch")
        for key in (
            "proposal_ordinal", "decision_source_ts_ns", "decision_arrival_ts_ns",
            "available_at_ns", "direction", "notional_jpy_micros", "max_age_ns",
        ):
            if not isinstance(row[key], int) or isinstance(row[key], bool):
                raise OracleError(f"proposal integer field invalid: {key}")
        if (
            row["direction"] not in {-1, 1}
            or row["notional_jpy_micros"] <= 0
            or row["max_age_ns"] <= 0
            or row["available_at_ns"] > row["decision_arrival_ts_ns"]
            or row["decision_arrival_ts_ns"] < row["decision_source_ts_ns"]
            or row["action"] != "ENTER"
        ):
            raise OracleError("proposal chronology or action invalid")
        _pair_currencies(row["instrument"])
        if not isinstance(row["worker_key"], str) or not row["worker_key"]:
            raise OracleError("proposal worker key invalid")
        ordinals.append(row["proposal_ordinal"])
    if ordinals != list(range(1, len(ordinals) + 1)):
        raise OracleError("proposal ordinals must be contiguous")
    decision_order = [
        (row["decision_source_ts_ns"], row["decision_arrival_ts_ns"], row["proposal_ordinal"])
        for row in proposal["rows"]
    ]
    if decision_order != sorted(decision_order):
        raise OracleError("proposal input chronology is not monotonic")
    execution = _json_object(artifacts["execution_policy"][1], "execution policy")
    inventory = _json_object(artifacts["inventory_policy"][1], "inventory policy")
    accounting = _json_object(artifacts["accounting_policy"][1], "accounting policy")
    evaluation = _json_object(artifacts["evaluation_policy"][1], "evaluation policy")
    _validate_policy(
        execution, expected_id="FROZEN_EXECUTION_POLICY_V1",
        hash_field="execution_policy_sha256",
    )
    _validate_policy(
        inventory, expected_id="FROZEN_INVENTORY_POLICY_V1",
        hash_field="inventory_policy_sha256",
    )
    _validate_policy(
        accounting, expected_id="FROZEN_ACCOUNTING_POLICY_V1",
        hash_field="accounting_policy_sha256",
    )
    _validate_policy(
        evaluation, expected_id="FROZEN_EVALUATION_POLICY_V1",
        hash_field="evaluation_policy_sha256",
    )
    if set(execution) != {
        "schema_version", "policy_id", "arms", "execution_policy_sha256"
    } or set(execution.get("arms", {})) != set(ARMS):
        raise OracleError("execution arm set mismatch")
    for arm in ARMS:
        spec = execution["arms"][arm]
        if set(spec) != {
            "latency_ns", "slippage_ticks_per_side", "commission_ppm_per_side",
            "financing_ppm_per_day", "raw_mid",
        }:
            raise OracleError(f"execution arm schema mismatch: {arm}")
        if any(
            not isinstance(spec[key], int) or isinstance(spec[key], bool) or spec[key] < 0
            for key in (
            "latency_ns", "slippage_ticks_per_side", "commission_ppm_per_side",
            "financing_ppm_per_day",
        )) or not isinstance(spec["raw_mid"], bool):
            raise OracleError(f"execution arm value invalid: {arm}")
    if execution["arms"]["RAW_SIGNAL"]["raw_mid"] is not True:
        raise OracleError("RAW arm must remain cost-independent midpoint accounting")
    if any(execution["arms"]["RAW_SIGNAL"][key] != 0 for key in (
        "latency_ns", "slippage_ticks_per_side", "commission_ppm_per_side",
        "financing_ppm_per_day",
    )):
        raise OracleError("RAW arm contains execution cost")
    if set(inventory) != {
        "schema_version", "policy_id", "max_gross_notional_jpy_micros",
        "max_currency_notional_jpy_micros", "max_open_positions",
        "same_pair_collision", "terminal_liquidation", "inventory_policy_sha256",
    } or any(
        not isinstance(inventory[key], int) or isinstance(inventory[key], bool)
        or inventory[key] <= 0
        for key in (
            "max_gross_notional_jpy_micros", "max_currency_notional_jpy_micros",
            "max_open_positions",
        )
    ) or inventory["same_pair_collision"] != "REJECT_NEW" \
            or inventory["terminal_liquidation"] is not True:
        raise OracleError("inventory policy schema/value mismatch")
    if set(accounting) != {
        "schema_version", "policy_id", "jpy_micros_per_yen",
        "base_microunits_per_unit", "max_conversion_staleness_ns",
        "supported_quote_currencies", "asset_conversion_side",
        "liability_conversion_side", "accounting_policy_sha256",
    } or accounting.get("supported_quote_currencies") != ["CAD", "CHF", "JPY", "USD"]:
        raise OracleError("accounting conversion universe mismatch")
    if accounting.get("jpy_micros_per_yen") != JPY_MICROS_PER_YEN \
            or accounting.get("base_microunits_per_unit") != BASE_MICROUNITS_PER_UNIT:
        raise OracleError("accounting unit contract mismatch")
    if not isinstance(accounting.get("max_conversion_staleness_ns"), int) \
            or isinstance(accounting["max_conversion_staleness_ns"], bool) \
            or accounting["max_conversion_staleness_ns"] <= 0 \
            or accounting.get("asset_conversion_side") != "BID" \
            or accounting.get("liability_conversion_side") != "ASK":
        raise OracleError("accounting conversion policy invalid")
    if set(evaluation) != {
        "schema_version", "policy_id", "period_start_ts_ns", "period_end_ts_ns",
        "initial_equity_jpy_micros", "margin_notional_cap_jpy_micros",
        "cvar_tail_bps", "holdout_state", "evaluation_policy_sha256",
    }:
        raise OracleError("evaluation policy schema mismatch")
    period_start = evaluation.get("period_start_ts_ns")
    period_end = evaluation.get("period_end_ts_ns")
    if not isinstance(period_start, int) or isinstance(period_start, bool) \
            or not isinstance(period_end, int) or isinstance(period_end, bool) \
            or period_start >= period_end \
            or not isinstance(evaluation["initial_equity_jpy_micros"], int) \
            or isinstance(evaluation["initial_equity_jpy_micros"], bool) \
            or evaluation["initial_equity_jpy_micros"] <= 0 \
            or not isinstance(evaluation["margin_notional_cap_jpy_micros"], int) \
            or isinstance(evaluation["margin_notional_cap_jpy_micros"], bool) \
            or evaluation["margin_notional_cap_jpy_micros"] <= 0 \
            or not isinstance(evaluation["cvar_tail_bps"], int) \
            or isinstance(evaluation["cvar_tail_bps"], bool) \
            or not 0 < evaluation["cvar_tail_bps"] <= 10_000 \
            or evaluation["holdout_state"] != "UNOPENED":
        raise OracleError("evaluation period invalid")
    if any(
        row["decision_source_ts_ns"] < period_start
        or row["decision_source_ts_ns"] >= period_end
        for row in proposal["rows"]
    ):
        raise OracleError("proposal outside frozen evaluation period")
    output_name = request["output_directory"]
    if not isinstance(output_name, str) or SAFE_OUTPUT_RE.fullmatch(output_name) is None:
        raise OracleError("oracle output directory name invalid")
    output = _bounded_path(
        output_root, str(output_root / output_name), "output", must_exist=False
    )
    return (
        rows, books, proposal, execution, inventory, accounting, evaluation,
        hashes, output,
    )


def _position_result(
    proposal: Mapping[str, Any],
    signal_id: str,
    arm: str,
    arm_policy: Mapping[str, Any],
    entry: Mapping[str, Any],
    exit_event: Mapping[str, Any],
    exit_reason: str,
    books: Mapping[str, Sequence[Mapping[str, Any]]],
    accounting: Mapping[str, Any],
    execution_policy_sha256: str,
) -> dict[str, Any]:
    max_staleness = accounting["max_conversion_staleness_ns"]
    entry_price = _execution_price(entry, proposal["direction"], opening=True, arm_policy=arm_policy)
    exit_price = _execution_price(
        exit_event, proposal["direction"], opening=False, arm_policy=arm_policy
    )
    units_micros = _position_units_micros(
        proposal, entry, entry_price, books, max_staleness
    )
    units = Decimal(units_micros) / BASE_MICROUNITS_PER_UNIT
    _, quote_currency = _pair_currencies(proposal["instrument"])
    quote_pnl = Decimal(proposal["direction"]) * units * (exit_price - entry_price)
    executable_jpy = _convert_to_jpy(
        quote_pnl,
        quote_currency,
        exit_event["source_ts_ns"],
        exit_event["arrival_ts_ns"],
        books,
        max_staleness,
    )
    raw_entry = _event_price(entry, side="mid")
    raw_exit = _event_price(exit_event, side="mid")
    raw_quote_pnl = Decimal(proposal["direction"]) * units * (raw_exit - raw_entry)
    raw_jpy = _convert_to_jpy(
        raw_quote_pnl,
        quote_currency,
        exit_event["source_ts_ns"],
        exit_event["arrival_ts_ns"],
        books,
        max_staleness,
    )
    elapsed_ns = exit_event["source_ts_ns"] - entry["source_ts_ns"]
    notional = Decimal(proposal["notional_jpy_micros"])
    commission = (
        notional * Decimal(2 * arm_policy["commission_ppm_per_side"]) / Decimal(1_000_000)
    )
    financing = (
        notional
        * Decimal(arm_policy["financing_ppm_per_day"])
        * Decimal(elapsed_ns)
        / Decimal(DAY_NS)
        / Decimal(1_000_000)
    )
    executable_micros = _decimal_to_jpy_micros(executable_jpy)
    gross_micros = _decimal_to_jpy_micros(raw_jpy)
    commission_micros = int(commission.to_integral_value(rounding=ROUND_FLOOR))
    financing_micros = int(financing.to_integral_value(rounding=ROUND_FLOOR))
    net_micros = executable_micros - commission_micros - financing_micros
    fill_id = sha256_bytes(canonical_bytes({
        "signal_id": signal_id,
        "arm": arm,
        "entry_source_event_sha256": entry["source_event_sha256"],
        "execution_policy_sha256": execution_policy_sha256,
    }))
    return {
        "record_type": "ORACLE_DISPOSITION",
        "arm": arm,
        "signal_id": signal_id,
        "fill_id": fill_id,
        "proposal_ordinal": proposal["proposal_ordinal"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "status": "FILLED_CLOSED",
        "entry_disposition": "FILLED",
        "exit_disposition": exit_reason,
        "action_transitions": ["ENTER", "EXIT"],
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
        "elapsed_ns": elapsed_ns,
        "gross_pnl_jpy_micros": gross_micros,
        "executable_pnl_jpy_micros": executable_micros,
        "commission_jpy_micros": commission_micros,
        "financing_jpy_micros": financing_micros,
        "realized_cost_jpy_micros": gross_micros - net_micros,
        "net_pnl_jpy_micros": net_micros,
        "entry_source_reference": {
            "source_event_sha256": entry["source_event_sha256"],
            "provider_id": entry["provider_id"],
            "source_ts_ns": entry["source_ts_ns"],
            "arrival_ts_ns": entry["arrival_ts_ns"],
            "execution_policy_sha256": execution_policy_sha256,
        },
        "exit_source_reference": {
            "source_event_sha256": exit_event["source_event_sha256"],
            "provider_id": exit_event["provider_id"],
            "source_ts_ns": exit_event["source_ts_ns"],
            "arrival_ts_ns": exit_event["arrival_ts_ns"],
            "execution_policy_sha256": execution_policy_sha256,
        },
        "currency_inventory_after_close": {},
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_order_count": 0,
    }


def _rejected_result(
    proposal: Mapping[str, Any], signal_id: str, arm: str, reason: str
) -> dict[str, Any]:
    return {
        "record_type": "ORACLE_DISPOSITION",
        "arm": arm,
        "signal_id": signal_id,
        "fill_id": None,
        "proposal_ordinal": proposal["proposal_ordinal"],
        "instrument": proposal["instrument"],
        "direction": proposal["direction"],
        "status": reason,
        "entry_disposition": reason,
        "exit_disposition": "NOT_APPLICABLE",
        "action_transitions": ["NO_ENTRY"],
        "notional_jpy_micros": proposal["notional_jpy_micros"],
        "units_micros": 0,
        "gross_pnl_jpy_micros": 0,
        "executable_pnl_jpy_micros": 0,
        "commission_jpy_micros": 0,
        "financing_jpy_micros": 0,
        "realized_cost_jpy_micros": 0,
        "net_pnl_jpy_micros": 0,
        "currency_inventory_after_close": {},
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_order_count": 0,
    }


def _arm_dispositions(
    arm: str,
    proposal_root: Mapping[str, Any],
    execution: Mapping[str, Any],
    inventory: Mapping[str, Any],
    accounting: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    books: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    policy = execution["arms"][arm]
    execution_hash = execution["execution_policy_sha256"]
    period_end = evaluation["period_end_ts_ns"]
    max_staleness = accounting["max_conversion_staleness_ns"]
    open_positions: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    proposals = sorted(
        proposal_root["rows"],
        key=lambda row: (row["decision_source_ts_ns"], row["proposal_ordinal"]),
    )
    for proposal in proposals:
        signal_id = _signal_id(proposal_root["candidate_key"], proposal)
        entry = _event_after(
            books,
            proposal["instrument"],
            minimum_source_ts_ns=proposal["decision_source_ts_ns"],
            minimum_arrival_ts_ns=(
                proposal["decision_arrival_ts_ns"] + policy["latency_ns"]
            ),
            period_end_ts_ns=period_end,
        )
        if entry is None:
            results.append(_rejected_result(proposal, signal_id, arm, "NO_CAUSAL_FILL"))
            continue
        open_positions = [
            position for position in open_positions
            if position["exit_source_ts_ns"] > entry["source_ts_ns"]
        ]
        gross_after = sum(
            position["notional_jpy_micros"] for position in open_positions
        ) + proposal["notional_jpy_micros"]
        if gross_after > inventory["max_gross_notional_jpy_micros"]:
            results.append(_rejected_result(proposal, signal_id, arm, "GROSS_CAP_REJECTED"))
            continue
        if len(open_positions) >= inventory["max_open_positions"]:
            results.append(_rejected_result(proposal, signal_id, arm, "POSITION_CAP_REJECTED"))
            continue
        base, quote = _pair_currencies(proposal["instrument"])
        currency_after = {}
        for currency in (base, quote):
            currency_after[currency] = sum(
                position["notional_jpy_micros"]
                for position in open_positions
                if currency in _pair_currencies(position["instrument"])
            ) + proposal["notional_jpy_micros"]
        if any(
            value > inventory["max_currency_notional_jpy_micros"]
            for value in currency_after.values()
        ):
            results.append(_rejected_result(proposal, signal_id, arm, "CURRENCY_CAP_REJECTED"))
            continue
        max_age_target = entry["source_ts_ns"] + proposal["max_age_ns"]
        exit_event = _event_at_or_after(
            books,
            proposal["instrument"],
            minimum_source_ts_ns=max_age_target,
            minimum_arrival_ts_ns=max_age_target + policy["latency_ns"],
            period_end_ts_ns=period_end,
        )
        exit_reason = "FINITE_MAX_AGE"
        if exit_event is None:
            exit_event = _terminal_event(
                books,
                proposal["instrument"],
                period_end_ts_ns=period_end,
                max_staleness_ns=max_staleness,
            )
            if exit_event["source_ts_ns"] < entry["source_ts_ns"]:
                raise OracleError("terminal event precedes entry")
            exit_reason = "TERMINAL_LIQUIDATION"
        result = _position_result(
            proposal,
            signal_id,
            arm,
            policy,
            entry,
            exit_event,
            exit_reason,
            books,
            accounting,
            execution_hash,
        )
        result["gross_open_notional_after_entry_jpy_micros"] = gross_after
        result["currency_notional_after_entry_jpy_micros"] = currency_after
        results.append(result)
        open_positions.append({
            "instrument": proposal["instrument"],
            "notional_jpy_micros": proposal["notional_jpy_micros"],
            "exit_source_ts_ns": exit_event["source_ts_ns"],
        })
    return results


def _month_id(timestamp_ns: int) -> str:
    stamp = datetime.fromtimestamp(timestamp_ns // 1_000_000_000, tz=timezone.utc)
    return f"{stamp.year:04d}-{stamp.month:02d}"


def aggregate_metrics(
    rows: Sequence[Mapping[str, Any]], evaluation: Mapping[str, Any]
) -> dict[str, Any]:
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
    action_signatures = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        signal_ids = sorted({row["signal_id"] for row in arm_rows})
        signal_sets[arm] = signal_ids
        action_signatures[arm] = [
            (row["signal_id"], row["action_transitions"]) for row in arm_rows
        ]
        executed = [row for row in arm_rows if row["status"] == "FILLED_CLOSED"]
        realized = sorted(
            executed, key=lambda row: (row["exit_source_ts_ns"], row["signal_id"])
        )
        equity = initial
        peak = initial
        max_drawdown = 0
        month_pnl: dict[str, int] = {}
        for row in realized:
            equity += row["net_pnl_jpy_micros"]
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
            month = _month_id(row["exit_source_ts_ns"])
            month_pnl[month] = month_pnl.get(month, 0) + row["net_pnl_jpy_micros"]
        monthly: dict[str, str] = {}
        month_equity = initial
        for month in sorted(month_pnl):
            next_equity = month_equity + month_pnl[month]
            monthly[month] = _ratio_text(next_equity, month_equity)
            month_equity = next_equity
        net_values = sorted(row["net_pnl_jpy_micros"] for row in executed)
        tail_count = max(1, (len(net_values) * tail_bps + 9_999) // 10_000) \
            if net_values else 0
        cvar = sum(net_values[:tail_count]) // tail_count if tail_count else 0
        gross_sum = sum(row["gross_pnl_jpy_micros"] for row in executed)
        net_sum = sum(row["net_pnl_jpy_micros"] for row in executed)
        cost_sum = sum(row["realized_cost_jpy_micros"] for row in executed)
        positives = sum(row["gross_pnl_jpy_micros"] > 0 for row in executed)
        cluster_keys = {
            (
                row["instrument"],
                row.get("entry_source_ts_ns", evaluation["period_start_ts_ns"]) // 3_600_000_000_000,
            )
            for row in executed
        }
        arm_metric = {
            "proposal_count": len(arm_rows),
            "executed_count": len(executed),
            "disposition_counts": dict(sorted(Counter(row["status"] for row in arm_rows).items())),
            "signal_id_set_sha256": sha256_bytes(canonical_bytes(signal_ids)),
            "action_transition_sha256": sha256_bytes(canonical_bytes(action_signatures[arm])),
            "gross_pnl_jpy_micros": gross_sum,
            "realized_cost_jpy_micros": cost_sum,
            "net_pnl_jpy_micros": net_sum,
            "ending_equity_jpy_micros": initial + net_sum,
            "ending_equity_multiple": _ratio_text(initial + net_sum, initial),
            "direction_accuracy": _ratio_text(positives, len(executed)) if executed else "0.000000000000000000",
            "max_drawdown_jpy_micros": max_drawdown,
            "max_drawdown_ratio": _ratio_text(max_drawdown, initial),
            "cvar_tail_bps": tail_bps,
            "cvar_jpy_micros": cvar,
            "max_gross_notional_jpy_micros": max(
                (row.get("gross_open_notional_after_entry_jpy_micros", 0) for row in executed),
                default=0,
            ),
            "currency_time_cluster_count": len(cluster_keys),
            "terminal_open_positions": 0,
            "terminal_inventory_mtm_jpy_micros": 0,
            "margin_guard_pass": max(
                (row.get("gross_open_notional_after_entry_jpy_micros", 0) for row in executed),
                default=0,
            ) <= evaluation["margin_notional_cap_jpy_micros"],
        }
        metrics["arms"][arm] = arm_metric
        metrics["monthly_multiples"][arm] = monthly
    if len({tuple(signal_sets[arm]) for arm in ARMS}) != 1:
        raise OracleError("arm signal-ID sets diverged")
    if any(metrics["arms"][arm]["proposal_count"] != len(signal_sets[arm]) for arm in ARMS):
        raise OracleError("duplicate signal disposition detected")
    metrics["same_signal_ids_all_arms"] = True
    metrics["all_proposals_have_all_arm_dispositions"] = True
    metrics["action_label_contract_all_arms"] = all(
        [labels for _, labels in action_signatures[arm]]
        == [labels for _, labels in action_signatures[ARMS[0]]]
        for arm in ARMS[1:]
    )
    metrics["metrics_sha256"] = embedded_hash(metrics, "metrics_sha256")
    return metrics


def _hash_chain(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    previous = ZERO_SHA
    for sequence, row in enumerate(rows, 1):
        payload = {
            "ledger_schema_version": 1,
            "ledger_sequence": sequence,
            "previous_hash": previous,
            **dict(row),
        }
        payload["record_hash"] = embedded_hash(payload, "record_hash")
        result.append(payload)
        previous = payload["record_hash"]
    return result


def execute(request: Mapping[str, Any]) -> dict[str, Any]:
    (
        _, books, proposal, execution, inventory, accounting, evaluation,
        input_hashes, output_directory,
    ) = _validate_inputs(request)
    all_rows = []
    for arm in ARMS:
        all_rows.extend(_arm_dispositions(
            arm, proposal, execution, inventory, accounting, evaluation, books
        ))
    all_rows.sort(key=lambda row: (row["proposal_ordinal"], ARMS.index(row["arm"])))
    ledger_rows = _hash_chain(all_rows)
    metrics = aggregate_metrics(ledger_rows, evaluation)
    ledger_bytes = b"".join(canonical_bytes(row) + b"\n" for row in ledger_rows)
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise OracleError("oracle output directory already exists")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output_directory, 0o700)
    _fsync_directory(output_directory.parent)
    _fsync_directory(output_directory)
    ledger_path = output_directory / "oracle_ledger.jsonl"
    _exclusive_bytes(ledger_path, ledger_bytes)
    source_root = Path(__file__).resolve().parent
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "oracle_implementation": ORACLE_NAME,
        "status": "COMPLETE",
        "authority": {
            "paper_only": True,
            "live_authority": False,
            "broker_account_access": False,
            "credential_access": False,
            "order_endpoint": False,
            "external_orders": 0,
            "deploy": False,
        },
        "oracle_code_sha256": sha256_bytes(_secure_read(Path(__file__).resolve(), root=source_root)),
        "oracle_contract_sha256": sha256_bytes(_secure_read(source_root / CONTRACT_NAME, root=source_root)),
        "oracle_schema_sha256": sha256_bytes(_secure_read(source_root / SCHEMA_NAME, root=source_root)),
        "input_artifact_sha256": dict(sorted(input_hashes.items())),
        "proposal_identity_generated_by_oracle": True,
        "producer_result_or_metrics_used": False,
        "oracle_ledger_file": "oracle_ledger.jsonl",
        "oracle_ledger_sha256": sha256_bytes(ledger_bytes),
        "oracle_ledger_row_count": len(ledger_rows),
        "oracle_ledger_terminal_hash": ledger_rows[-1]["record_hash"] if ledger_rows else ZERO_SHA,
        "oracle_metrics": metrics,
        "terminal_inventory_mtm_jpy_micros": 0,
        "external_orders": 0,
        "anchor_status": "LOCAL_REPRODUCIBLE",
    }
    manifest["oracle_root_sha256"] = embedded_hash(manifest, "oracle_root_sha256")
    manifest_path = output_directory / "oracle_manifest.json"
    _exclusive_bytes(manifest_path, canonical_bytes(manifest) + b"\n")
    return {
        "manifest_path": str(manifest_path),
        "ledger_path": str(ledger_path),
        "manifest": manifest,
    }


def _audit_hook(event: str, _: tuple[Any, ...]) -> None:
    if event.startswith(("socket.", "subprocess.")) or event in {
        "os.system", "os.posix_spawn", "os.exec", "os.spawn",
    }:
        raise OracleError(f"runtime capability denied: {event}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request_path = args.request.resolve(strict=True)
    request = _json_object(
        _secure_read(request_path, root=request_path.parent), "oracle request"
    )
    result = execute(request)
    print(json.dumps({
        "ok": True,
        "oracle_root_sha256": result["manifest"]["oracle_root_sha256"],
        "manifest_path": result["manifest_path"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.addaudithook(_audit_hook)
    try:
        raise SystemExit(main())
    except OracleError as error:
        print(json.dumps({
            "ok": False,
            "error_code": "ORACLE_FAIL_CLOSED",
            "error_sha256": sha256_bytes(str(error).encode("utf-8")),
        }, sort_keys=True))
        raise SystemExit(2)

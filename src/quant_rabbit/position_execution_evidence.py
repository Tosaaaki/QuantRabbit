from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_rabbit.models import BrokerSnapshot
from quant_rabbit.snapshot_json import (
    ORDER_RAW_SNAPSHOT_KEYS,
    POSITION_RAW_SNAPSHOT_KEYS,
    snapshot_order_raw,
    snapshot_position_raw,
)


POSITION_EXECUTION_SNAPSHOT_EVIDENCE_SCHEMA = (
    "position_execution_pre_send_broker_snapshot_v1"
)
POSITION_EXECUTION_SNAPSHOT_EVIDENCE_FIELD = "pre_send_broker_snapshot_evidence"
POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME = (
    "position_execution_snapshot_evidence"
)
POSITION_EXECUTION_SNAPSHOT_EVIDENCE_MAX_BYTES = 16 * 1024 * 1024


def persist_position_execution_snapshot_evidence(
    *,
    snapshot: BrokerSnapshot,
    receipt_path: Path,
) -> dict[str, Any]:
    """Persist the exact normalized snapshot consumed by one gateway run.

    The filename is the SHA-256 of canonical JSON. Existing evidence is never
    overwritten: an identical object is reused and any collision or prior
    corruption fails closed before the caller can cross the broker boundary.
    """

    payload = _position_execution_snapshot_payload(snapshot)
    # BrokerSnapshot is a runtime dataclass rather than a validating boundary.
    # Validate the producer-side payload before hashing or writing it so a
    # type-coerced/semantically invalid snapshot cannot become pre-send proof.
    _validate_position_execution_snapshot_payload(payload)
    canonical = _canonical_json_bytes(payload)
    if len(canonical) > POSITION_EXECUTION_SNAPSHOT_EVIDENCE_MAX_BYTES:
        raise ValueError("position execution snapshot evidence exceeds size limit")
    digest = hashlib.sha256(canonical).hexdigest()
    relative_path = (
        f"{POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME}/{digest}.json"
    )
    evidence_dir = _position_execution_evidence_directory(
        receipt_path.parent,
        create=True,
    )
    evidence_path = evidence_dir / f"{digest}.json"

    try:
        fd = os.open(
            evidence_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        _verify_regular_content(evidence_path, expected=canonical)
    else:
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(canonical)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                evidence_path.unlink()
            except OSError:
                pass
            raise

    return {
        "schema": POSITION_EXECUTION_SNAPSHOT_EVIDENCE_SCHEMA,
        "sha256": digest,
        "byte_count": len(canonical),
        "path": relative_path,
    }


def load_position_execution_snapshot_evidence(
    *,
    receipt_payload: dict[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    """Load and authenticate the immutable pre-send snapshot for a receipt."""

    proof = receipt_payload.get(POSITION_EXECUTION_SNAPSHOT_EVIDENCE_FIELD)
    if not isinstance(proof, dict) or set(proof) != {
        "schema",
        "sha256",
        "byte_count",
        "path",
    }:
        raise ValueError(
            "position-execution receipt pre-send snapshot evidence is missing or malformed"
        )
    if proof.get("schema") != POSITION_EXECUTION_SNAPSHOT_EVIDENCE_SCHEMA:
        raise ValueError("position-execution pre-send snapshot evidence schema is invalid")
    digest = proof.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("position-execution pre-send snapshot evidence sha256 is invalid")
    byte_count = proof.get("byte_count")
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count <= 0
        or byte_count > POSITION_EXECUTION_SNAPSHOT_EVIDENCE_MAX_BYTES
    ):
        raise ValueError(
            "position-execution pre-send snapshot evidence byte_count is invalid"
        )
    expected_relative_path = (
        f"{POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME}/{digest}.json"
    )
    if proof.get("path") != expected_relative_path:
        raise ValueError("position-execution pre-send snapshot evidence path is invalid")
    evidence_dir = _position_execution_evidence_directory(
        receipt_path.parent,
        create=False,
    )
    evidence_path = evidence_dir / f"{digest}.json"
    raw = _read_exact_regular_file(evidence_path, expected_size=byte_count)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(
            "position-execution pre-send snapshot evidence digest/size mismatch"
        )
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "position-execution pre-send snapshot evidence JSON is invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("position-execution pre-send snapshot evidence must be an object")
    if payload.get("evidence_schema") != POSITION_EXECUTION_SNAPSHOT_EVIDENCE_SCHEMA:
        raise ValueError(
            "position-execution pre-send snapshot evidence body schema is invalid"
        )
    if _canonical_json_bytes(payload) != raw:
        raise ValueError(
            "position-execution pre-send snapshot evidence is not canonical JSON"
        )
    _validate_position_execution_snapshot_payload(payload)
    return payload


def _position_execution_snapshot_payload(snapshot: BrokerSnapshot) -> dict[str, Any]:
    quotes: dict[str, dict[str, Any]] = {}
    for raw_pair, quote in snapshot.quotes.items():
        pair = _require_pair(raw_pair, "quotes key")
        quote_pair = _require_pair(getattr(quote, "pair", None), f"quotes.{pair}.pair")
        if quote_pair != pair:
            raise ValueError(
                "position-execution pre-send snapshot evidence quote key/pair mismatch"
            )
        quotes[pair] = {
            "bid": quote.bid,
            "ask": quote.ask,
            "timestamp_utc": quote.timestamp_utc.isoformat(),
        }

    home_conversions: dict[str, float] = {}
    for raw_currency, rate in snapshot.home_conversions.items():
        currency = _require_currency(raw_currency, "home_conversions key")
        _require_finite_number(
            rate,
            f"home_conversions.{currency}",
            positive=True,
        )
        home_conversions[currency] = rate

    positions = []
    for position in snapshot.positions:
        positions.append(
            {
                "trade_id": position.trade_id,
                "pair": position.pair,
                "side": position.side.value,
                "units": position.units,
                "entry_price": position.entry_price,
                "unrealized_pl_jpy": position.unrealized_pl_jpy,
                "take_profit": position.take_profit,
                "stop_loss": position.stop_loss,
                "owner": position.owner.value,
                "raw": snapshot_position_raw(position.raw),
            }
        )
    payload: dict[str, Any] = {
        "evidence_schema": POSITION_EXECUTION_SNAPSHOT_EVIDENCE_SCHEMA,
        "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
        "positions": positions,
        "orders": [
            {
                "order_id": order.order_id,
                "pair": order.pair,
                "order_type": order.order_type,
                "trade_id": order.trade_id,
                "price": order.price,
                "state": order.state,
                "units": order.units,
                "owner": order.owner.value,
                "raw": snapshot_order_raw(order.raw),
            }
            for order in snapshot.orders
        ],
        "quotes": quotes,
        "home_conversions": home_conversions,
    }
    if snapshot.account is not None:
        account = snapshot.account
        payload["account"] = {
            "nav_jpy": account.nav_jpy,
            "balance_jpy": account.balance_jpy,
            "unrealized_pl_jpy": account.unrealized_pl_jpy,
            "margin_used_jpy": account.margin_used_jpy,
            "margin_available_jpy": account.margin_available_jpy,
            "pl_jpy": account.pl_jpy,
            "financing_jpy": account.financing_jpy,
            "last_transaction_id": account.last_transaction_id,
            "hedging_enabled": account.hedging_enabled,
            "fetched_at_utc": account.fetched_at_utc.isoformat(),
        }
    return payload


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_position_execution_snapshot_payload(payload: dict[str, Any]) -> None:
    required_top_level = {
        "evidence_schema",
        "fetched_at_utc",
        "positions",
        "orders",
        "quotes",
        "home_conversions",
    }
    if set(payload) not in (required_top_level, required_top_level | {"account"}):
        raise ValueError(
            "position-execution pre-send snapshot evidence body fields are invalid"
        )
    _require_aware_datetime(payload.get("fetched_at_utc"), "fetched_at_utc")

    positions = payload.get("positions")
    if not isinstance(positions, list):
        raise ValueError(
            "position-execution pre-send snapshot evidence positions must be an array"
        )
    seen_trade_ids: set[str] = set()
    for index, position in enumerate(positions):
        context = f"positions[{index}]"
        _require_exact_object(
            position,
            {
                "trade_id",
                "pair",
                "side",
                "units",
                "entry_price",
                "unrealized_pl_jpy",
                "take_profit",
                "stop_loss",
                "owner",
                "raw",
            },
            context,
        )
        trade_id = _require_text(position.get("trade_id"), f"{context}.trade_id")
        if trade_id in seen_trade_ids:
            raise ValueError(
                "position-execution pre-send snapshot evidence positions contain "
                f"duplicate trade_id={trade_id}"
            )
        seen_trade_ids.add(trade_id)
        _require_pair(position.get("pair"), f"{context}.pair")
        if position.get("side") not in {"LONG", "SHORT"}:
            raise ValueError(
                f"position-execution pre-send snapshot evidence {context}.side is invalid"
            )
        units = position.get("units")
        if not _is_exact_int(units) or units <= 0:
            raise ValueError(
                f"position-execution pre-send snapshot evidence {context}.units is invalid"
            )
        _require_finite_number(
            position.get("entry_price"),
            f"{context}.entry_price",
            positive=True,
        )
        _require_finite_number(
            position.get("unrealized_pl_jpy"),
            f"{context}.unrealized_pl_jpy",
        )
        for field in ("take_profit", "stop_loss"):
            value = position.get(field)
            if value is not None:
                _require_finite_number(
                    value,
                    f"{context}.{field}",
                    positive=True,
                )
        _require_owner(position.get("owner"), f"{context}.owner")
        _require_raw_object(
            position.get("raw"),
            allowed_keys=set(POSITION_RAW_SNAPSHOT_KEYS),
            context=f"{context}.raw",
        )

    orders = payload.get("orders")
    if not isinstance(orders, list):
        raise ValueError(
            "position-execution pre-send snapshot evidence orders must be an array"
        )
    seen_order_ids: set[str] = set()
    for index, order in enumerate(orders):
        context = f"orders[{index}]"
        _require_exact_object(
            order,
            {
                "order_id",
                "pair",
                "order_type",
                "trade_id",
                "price",
                "state",
                "units",
                "owner",
                "raw",
            },
            context,
        )
        order_id = _require_text(order.get("order_id"), f"{context}.order_id")
        if order_id in seen_order_ids:
            raise ValueError(
                "position-execution pre-send snapshot evidence orders contain "
                f"duplicate order_id={order_id}"
            )
        seen_order_ids.add(order_id)
        if order.get("pair") is not None:
            _require_pair(order.get("pair"), f"{context}.pair")
        _require_text(order.get("order_type"), f"{context}.order_type")
        if order.get("trade_id") is not None:
            _require_text(order.get("trade_id"), f"{context}.trade_id")
        if order.get("price") is not None:
            _require_finite_number(
                order.get("price"),
                f"{context}.price",
                positive=True,
            )
        if order.get("state") is not None:
            _require_text(order.get("state"), f"{context}.state")
        if order.get("units") is not None and not _is_exact_int(order.get("units")):
            raise ValueError(
                f"position-execution pre-send snapshot evidence {context}.units is invalid"
            )
        _require_owner(order.get("owner"), f"{context}.owner")
        _require_raw_object(
            order.get("raw"),
            allowed_keys=set(ORDER_RAW_SNAPSHOT_KEYS),
            context=f"{context}.raw",
        )

    quotes = payload.get("quotes")
    if not isinstance(quotes, dict):
        raise ValueError(
            "position-execution pre-send snapshot evidence quotes must be an object"
        )
    for pair, quote in quotes.items():
        _require_pair(pair, "quotes key")
        _require_exact_object(
            quote,
            {"bid", "ask", "timestamp_utc"},
            f"quotes.{pair}",
        )
        bid = _require_finite_number(
            quote.get("bid"),
            f"quotes.{pair}.bid",
            positive=True,
        )
        ask = _require_finite_number(
            quote.get("ask"),
            f"quotes.{pair}.ask",
            positive=True,
        )
        if bid > ask:
            raise ValueError(
                f"position-execution pre-send snapshot evidence quotes.{pair} bid exceeds ask"
            )
        _require_aware_datetime(
            quote.get("timestamp_utc"),
            f"quotes.{pair}.timestamp_utc",
        )

    conversions = payload.get("home_conversions")
    if not isinstance(conversions, dict):
        raise ValueError(
            "position-execution pre-send snapshot evidence home_conversions must be an object"
        )
    for currency, rate in conversions.items():
        _require_currency(currency, "home conversion currency")
        _require_finite_number(
            rate,
            f"home_conversions.{currency}",
            positive=True,
        )

    account = payload.get("account")
    if "account" in payload:
        _require_exact_object(
            account,
            {
                "nav_jpy",
                "balance_jpy",
                "unrealized_pl_jpy",
                "margin_used_jpy",
                "margin_available_jpy",
                "pl_jpy",
                "financing_jpy",
                "last_transaction_id",
                "hedging_enabled",
                "fetched_at_utc",
            },
            "account",
        )
        for field in (
            "nav_jpy",
            "balance_jpy",
            "unrealized_pl_jpy",
            "margin_used_jpy",
            "margin_available_jpy",
            "pl_jpy",
            "financing_jpy",
        ):
            _require_finite_number(account.get(field), f"account.{field}")
        if account.get("last_transaction_id").__class__ is not str:
            raise ValueError(
                "position-execution pre-send snapshot evidence account.last_transaction_id is invalid"
            )
        if account.get("hedging_enabled").__class__ is not bool:
            raise ValueError(
                "position-execution pre-send snapshot evidence account.hedging_enabled is invalid"
            )
        _require_aware_datetime(
            account.get("fetched_at_utc"),
            "account.fetched_at_utc",
        )


def _require_exact_object(
    value: Any,
    keys: set[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(
            f"position-execution pre-send snapshot evidence {context} fields are invalid"
        )
    return value


def _require_text(value: Any, context: str) -> str:
    if value.__class__ is not str or not value or value != value.strip():
        raise ValueError(
            f"position-execution pre-send snapshot evidence {context} is invalid"
        )
    return value


def _require_pair(value: Any, context: str) -> str:
    pair = _require_text(value, context)
    parts = pair.split("_")
    if (
        len(parts) != 2
        or any(len(part) != 3 or not part.isalpha() for part in parts)
        or pair != pair.upper()
    ):
        raise ValueError(
            f"position-execution pre-send snapshot evidence {context} is not a canonical pair"
        )
    return pair


def _require_currency(value: Any, context: str) -> str:
    currency = _require_text(value, context)
    if (
        len(currency) != 3
        or not currency.isalpha()
        or currency != currency.upper()
    ):
        raise ValueError(
            "position-execution pre-send snapshot evidence "
            f"{context} is not a canonical currency"
        )
    return currency


def _require_owner(value: Any, context: str) -> str:
    owner = _require_text(value, context)
    if owner not in {"trader", "manual", "operator_manual", "external", "unknown"}:
        raise ValueError(
            f"position-execution pre-send snapshot evidence {context} is invalid"
        )
    return owner


def _require_finite_number(
    value: Any,
    context: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"position-execution pre-send snapshot evidence {context} is not finite"
        )
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"position-execution pre-send snapshot evidence {context} is not finite"
        ) from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(
            f"position-execution pre-send snapshot evidence {context} is not finite"
        )
    return number


def _is_exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_aware_datetime(value: Any, context: str) -> datetime:
    text = _require_text(value, context)
    normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"position-execution pre-send snapshot evidence {context} is invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"position-execution pre-send snapshot evidence {context} must be timezone-aware"
        )
    return parsed


def _require_raw_object(
    value: Any,
    *,
    allowed_keys: set[str],
    context: str,
) -> None:
    if not isinstance(value, dict) or not set(value).issubset(allowed_keys):
        raise ValueError(
            f"position-execution pre-send snapshot evidence {context} fields are invalid"
        )


def _verify_regular_content(path: Path, *, expected: bytes) -> None:
    try:
        actual = _read_exact_regular_file(path, expected_size=len(expected))
    except ValueError as exc:
        raise ValueError(
            "position execution snapshot evidence path already contains different data"
        ) from exc
    if actual != expected:
        raise ValueError(
            "position execution snapshot evidence path already contains different data"
        )


def _position_execution_evidence_directory(
    receipt_parent: Path,
    *,
    create: bool,
) -> Path:
    try:
        parent_mode = receipt_parent.lstat().st_mode
    except OSError as exc:
        raise ValueError(
            "position-execution receipt parent directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(parent_mode):
        raise ValueError(
            "position-execution receipt parent must be a real directory"
        )

    evidence_dir = receipt_parent / POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME
    if create:
        try:
            evidence_dir.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ValueError(
                "position-execution snapshot evidence directory is unavailable"
            ) from exc
    try:
        evidence_mode = evidence_dir.lstat().st_mode
    except OSError as exc:
        raise ValueError(
            "position-execution snapshot evidence directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(evidence_mode):
        raise ValueError(
            "position-execution snapshot evidence directory must not be a symlink"
        )
    return evidence_dir


def _read_exact_regular_file(path: Path, *, expected_size: int) -> bytes:
    try:
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(
            "position-execution pre-send snapshot evidence file is unavailable"
        ) from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != expected_size:
            raise ValueError(
                "position-execution pre-send snapshot evidence digest/size mismatch"
            )
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            raw = handle.read(expected_size + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(raw) != expected_size:
        raise ValueError(
            "position-execution pre-send snapshot evidence digest/size mismatch"
        )
    return raw


def _strict_unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")

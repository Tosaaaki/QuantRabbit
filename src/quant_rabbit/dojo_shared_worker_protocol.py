"""Pure-data protocol between DOJO strategy workers and a shared reducer.

The protocol intentionally has no broker, allocator, or execution dependency.  A
worker receives an immutable, post-exit snapshot and can only return proposals.
The shared reducer remains the sole place where proposals may later be allocated
or executed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence


POST_EXIT_SNAPSHOT_CONTRACT: Final = "QR_DOJO_SHARED_POST_EXIT_SNAPSHOT_V1"
WORKER_PROPOSAL_CONTRACT: Final = "QR_DOJO_SHARED_WORKER_PROPOSAL_V1"
WORKER_PROPOSAL_BATCH_CONTRACT: Final = "QR_DOJO_SHARED_WORKER_PROPOSAL_BATCH_V1"
SCHEMA_VERSION: Final = 1

# These are artifact-shape limits, not trading parameters.  They bound malformed
# per-coordinate payload amplification while remaining above the current 32-worker
# DOJO plan and its expected position/order fan-out.
MAX_SNAPSHOT_ROWS: Final = 4096
MAX_INTENTS_PER_CLASS: Final = 256
MAX_PROPOSALS_PER_BATCH: Final = 64
MAX_IDENTIFIER_LENGTH: Final = 128
MAX_REASON_LENGTH: Final = 256

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_PAIR_RE: Final = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_PHASES: Final = frozenset({"O", "H", "L", "C"})
_INTRABAR_PATHS: Final = frozenset({"OHLC", "OLHC"})
_SIDES: Final = frozenset({"LONG", "SHORT"})
_RISK_REDUCING_ACTIONS: Final = frozenset(
    {"CANCEL_ORDER", "CLOSE_POSITION", "TIGHTEN_STOP"}
)
_NEW_RISK_ACTIONS: Final = frozenset({"MARKET", "LIMIT", "STOP"})

_RAW_SNAPSHOT_KEYS: Final = frozenset(
    {
        "coordinate_id",
        "epoch",
        "phase",
        "intrabar",
        "quote_batch_sha256",
        "quote_watermark",
        "expected_quote_pairs",
        "active_worker_bindings",
        "account",
        "quotes",
        "positions",
        "pending_orders",
    }
)
_SEALED_SNAPSHOT_KEYS: Final = _RAW_SNAPSHOT_KEYS | frozenset(
    {
        "contract",
        "schema_version",
        "snapshot_state",
        "read_only",
        "proposal_only",
        "allocation_allowed",
        "execution_allowed",
        "order_authority",
        "live_permission",
        "broker_mutation_allowed",
        "snapshot_sha256",
    }
)
_ACCOUNT_KEYS: Final = frozenset(
    {"balance_jpy", "equity_jpy", "margin_used_jpy", "accrued_financing_jpy"}
)
_QUOTE_KEYS: Final = frozenset({"pair", "bid", "ask", "timestamp"})
_WORKER_BINDING_KEYS: Final = frozenset(
    {"worker_id", "owner_id", "family_id", "config_sha256"}
)
_POSITION_KEYS: Final = frozenset(
    {
        "position_id",
        "worker_id",
        "owner_id",
        "family_id",
        "pair",
        "side",
        "units",
        "entry_price",
        "tp_price",
        "sl_price",
        "opened_epoch",
        "hard_exit_epoch",
    }
)
_PENDING_ORDER_KEYS: Final = frozenset(
    {
        "order_id",
        "worker_id",
        "owner_id",
        "family_id",
        "pair",
        "side",
        "order_kind",
        "units",
        "trigger_price",
        "tp_price",
        "sl_price",
        "created_epoch",
        "valid_until_epoch",
    }
)
_RAW_PROPOSAL_KEYS: Final = frozenset(
    {
        "worker_id",
        "owner_id",
        "family_id",
        "config_sha256",
        "snapshot_sha256",
        "risk_reducing_intents",
        "new_risk_intents",
    }
)
_SEALED_PROPOSAL_KEYS: Final = _RAW_PROPOSAL_KEYS | frozenset(
    {
        "contract",
        "schema_version",
        "decision_basis",
        "risk_reduction_processed_before_new_risk",
        "proposal_only",
        "allocation_allowed",
        "execution_allowed",
        "order_authority",
        "live_permission",
        "broker_mutation_allowed",
        "worker_economic_claims_authoritative",
        "reducer_recomputes_entry_and_edge",
        "intent_counts",
        "proposal_sha256",
    }
)
_INTENT_KEYS: Final = frozenset({"intent_id", "action", "parameters", "reason_code"})
_NEW_RISK_PARAMETER_KEYS: Final = frozenset(
    {
        "pair",
        "side",
        "units",
        "entry_price",
        "tp_price",
        "sl_price",
        "stress_cost_pips",
        "hard_max_holding_seconds",
        "valid_until_epoch",
        "expected_net_edge_jpy",
    }
)
_SEALED_BATCH_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "snapshot_sha256",
        "proposals",
        "proposal_count",
        "risk_reducing_intent_count",
        "new_risk_intent_count",
        "proposal_only",
        "allocation_allowed",
        "execution_allowed",
        "order_authority",
        "live_permission",
        "broker_mutation_allowed",
        "worker_economic_claims_authoritative",
        "reducer_recomputes_entry_and_edge",
        "batch_sha256",
    }
)


class ProtocolViolation(ValueError):
    """Raised when a worker-protocol artifact is ambiguous or unsafe."""


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolViolation(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ProtocolViolation(f"{path} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], keys: frozenset[str], path: str
) -> None:
    actual = frozenset(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ProtocolViolation(
            f"{path} schema mismatch: missing={missing}, extra={extra}"
        )


def _require_sequence(value: Any, path: str, maximum: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ProtocolViolation(f"{path} must be a sequence")
    if len(value) > maximum:
        raise ProtocolViolation(f"{path} exceeds maximum length {maximum}")
    return value


def _require_identifier(
    value: Any, path: str, *, maximum: int = MAX_IDENTIFIER_LENGTH
) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ProtocolViolation(
            f"{path} must be a non-empty string of at most {maximum} chars"
        )
    if value != value.strip() or any(ord(char) < 32 for char in value):
        raise ProtocolViolation(
            f"{path} contains whitespace padding or control characters"
        )
    return value


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProtocolViolation(f"{path} must be a lowercase SHA-256 hex digest")
    return value


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolViolation(f"{path} must be an integer >= {minimum}")
    return value


def _require_number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolViolation(f"{path} must be a finite number")
    if not math.isfinite(float(value)):
        raise ProtocolViolation(f"{path} must be a finite number")
    if strictly_positive and value <= 0:
        raise ProtocolViolation(f"{path} must be > 0")
    if minimum is not None and value < minimum:
        raise ProtocolViolation(f"{path} must be >= {minimum}")
    return value


def _require_optional_price(value: Any, path: str) -> int | float | None:
    if value is None:
        return None
    return _require_number(value, path, strictly_positive=True)


def _require_enum(value: Any, allowed: frozenset[str], path: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ProtocolViolation(f"{path} must be one of {sorted(allowed)}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation(f"artifact is not canonical JSON: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _copy_json(value: Any) -> Any:
    """Return a deep canonical-JSON copy without sharing caller-owned containers."""

    try:
        return json.loads(_canonical_bytes(value))
    except (
        json.JSONDecodeError
    ) as exc:  # pragma: no cover - canonical encoder guarantees this
        raise ProtocolViolation("artifact cannot be copied as canonical JSON") from exc


def _timestamp_coordinate(timestamp: Any, phase: str, path: str) -> int:
    if not isinstance(timestamp, str):
        raise ProtocolViolation(
            f"{path} must be an ISO-8601 timestamp suffixed with #{phase}"
        )
    prefix, separator, suffix = timestamp.rpartition("#")
    if separator != "#" or suffix != phase:
        raise ProtocolViolation(f"{path} must end with #{phase}")
    try:
        parsed = datetime.fromisoformat(prefix.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolViolation(f"{path} is not valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ProtocolViolation(f"{path} must include a timezone")
    return int(parsed.timestamp())


def _normalize_account(value: Any) -> dict[str, int | float]:
    account = _require_mapping(value, "snapshot.account")
    _require_exact_keys(account, _ACCOUNT_KEYS, "snapshot.account")
    return {
        "balance_jpy": _require_number(
            account["balance_jpy"], "snapshot.account.balance_jpy"
        ),
        "equity_jpy": _require_number(
            account["equity_jpy"], "snapshot.account.equity_jpy"
        ),
        "margin_used_jpy": _require_number(
            account["margin_used_jpy"], "snapshot.account.margin_used_jpy", minimum=0
        ),
        "accrued_financing_jpy": _require_number(
            account["accrued_financing_jpy"],
            "snapshot.account.accrued_financing_jpy",
            minimum=0,
        ),
    }


def _normalize_quotes(value: Any, *, epoch: int, phase: str) -> list[dict[str, Any]]:
    rows = _require_sequence(value, "snapshot.quotes", MAX_SNAPSHOT_ROWS)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        path = f"snapshot.quotes[{index}]"
        quote = _require_mapping(raw, path)
        _require_exact_keys(quote, _QUOTE_KEYS, path)
        pair = _require_identifier(quote["pair"], f"{path}.pair")
        if _PAIR_RE.fullmatch(pair) is None:
            raise ProtocolViolation(f"{path}.pair must use AAA_BBB format")
        if pair in seen:
            raise ProtocolViolation(f"duplicate quote pair: {pair}")
        seen.add(pair)
        bid = _require_number(quote["bid"], f"{path}.bid", strictly_positive=True)
        ask = _require_number(quote["ask"], f"{path}.ask", strictly_positive=True)
        if ask <= bid:
            raise ProtocolViolation(f"{path}.ask must be greater than bid")
        timestamp = _require_identifier(quote["timestamp"], f"{path}.timestamp")
        if _timestamp_coordinate(timestamp, phase, f"{path}.timestamp") != epoch:
            raise ProtocolViolation(f"{path}.timestamp does not match snapshot.epoch")
        normalized.append(
            {"pair": pair, "bid": bid, "ask": ask, "timestamp": timestamp}
        )
    if not normalized:
        raise ProtocolViolation("snapshot.quotes must not be empty")
    return sorted(normalized, key=lambda row: row["pair"])


def _normalize_expected_quote_pairs(value: Any) -> list[str]:
    rows = _require_sequence(value, "snapshot.expected_quote_pairs", MAX_SNAPSHOT_ROWS)
    normalized: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        pair = _require_identifier(raw, f"snapshot.expected_quote_pairs[{index}]")
        if _PAIR_RE.fullmatch(pair) is None:
            raise ProtocolViolation(
                f"snapshot.expected_quote_pairs[{index}] must use AAA_BBB format"
            )
        if pair in seen:
            raise ProtocolViolation(f"duplicate expected quote pair: {pair}")
        seen.add(pair)
        normalized.append(pair)
    if not normalized:
        raise ProtocolViolation("snapshot.expected_quote_pairs must not be empty")
    return sorted(normalized)


def _normalize_worker_bindings(value: Any) -> list[dict[str, str]]:
    rows = _require_sequence(
        value, "snapshot.active_worker_bindings", MAX_PROPOSALS_PER_BATCH
    )
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        path = f"snapshot.active_worker_bindings[{index}]"
        binding = _require_mapping(raw, path)
        _require_exact_keys(binding, _WORKER_BINDING_KEYS, path)
        worker_id = _require_identifier(binding["worker_id"], f"{path}.worker_id")
        if worker_id in seen:
            raise ProtocolViolation(f"duplicate active worker_id: {worker_id}")
        seen.add(worker_id)
        normalized.append(
            {
                "worker_id": worker_id,
                "owner_id": _require_identifier(
                    binding["owner_id"], f"{path}.owner_id"
                ),
                "family_id": _require_identifier(
                    binding["family_id"], f"{path}.family_id"
                ),
                "config_sha256": _require_sha256(
                    binding["config_sha256"], f"{path}.config_sha256"
                ),
            }
        )
    if not normalized:
        raise ProtocolViolation("snapshot.active_worker_bindings must not be empty")
    return sorted(normalized, key=lambda row: row["worker_id"])


def _require_bound_state_owners(
    positions: Sequence[Mapping[str, Any]],
    pending_orders: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, str]],
) -> None:
    by_worker = {binding["worker_id"]: binding for binding in bindings}
    for collection_name, rows in (
        ("positions", positions),
        ("pending_orders", pending_orders),
    ):
        for row in rows:
            binding = by_worker.get(row["worker_id"])
            if binding is None or (
                binding["owner_id"] != row["owner_id"]
                or binding["family_id"] != row["family_id"]
            ):
                raise ProtocolViolation(
                    f"snapshot.{collection_name} contains state outside active_worker_bindings"
                )


def _normalize_positions(value: Any, quote_pairs: set[str]) -> list[dict[str, Any]]:
    rows = _require_sequence(value, "snapshot.positions", MAX_SNAPSHOT_ROWS)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        path = f"snapshot.positions[{index}]"
        position = _require_mapping(raw, path)
        _require_exact_keys(position, _POSITION_KEYS, path)
        position_id = _require_identifier(
            position["position_id"], f"{path}.position_id"
        )
        if position_id in seen:
            raise ProtocolViolation(f"duplicate position_id: {position_id}")
        seen.add(position_id)
        pair = _require_identifier(position["pair"], f"{path}.pair")
        if pair not in quote_pairs:
            raise ProtocolViolation(f"{path}.pair has no post-exit quote")
        opened_epoch = _require_int(position["opened_epoch"], f"{path}.opened_epoch")
        hard_exit_epoch = _require_int(
            position["hard_exit_epoch"], f"{path}.hard_exit_epoch"
        )
        if hard_exit_epoch < opened_epoch:
            raise ProtocolViolation(f"{path}.hard_exit_epoch precedes opened_epoch")
        normalized.append(
            {
                "position_id": position_id,
                "worker_id": _require_identifier(
                    position["worker_id"], f"{path}.worker_id"
                ),
                "owner_id": _require_identifier(
                    position["owner_id"], f"{path}.owner_id"
                ),
                "family_id": _require_identifier(
                    position["family_id"], f"{path}.family_id"
                ),
                "pair": pair,
                "side": _require_enum(position["side"], _SIDES, f"{path}.side"),
                "units": _require_number(
                    position["units"], f"{path}.units", strictly_positive=True
                ),
                "entry_price": _require_number(
                    position["entry_price"],
                    f"{path}.entry_price",
                    strictly_positive=True,
                ),
                "tp_price": _require_optional_price(
                    position["tp_price"], f"{path}.tp_price"
                ),
                "sl_price": _require_optional_price(
                    position["sl_price"], f"{path}.sl_price"
                ),
                "opened_epoch": opened_epoch,
                "hard_exit_epoch": hard_exit_epoch,
            }
        )
    return sorted(normalized, key=lambda row: row["position_id"])


def _normalize_pending_orders(
    value: Any, quote_pairs: set[str]
) -> list[dict[str, Any]]:
    rows = _require_sequence(value, "snapshot.pending_orders", MAX_SNAPSHOT_ROWS)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        path = f"snapshot.pending_orders[{index}]"
        order = _require_mapping(raw, path)
        _require_exact_keys(order, _PENDING_ORDER_KEYS, path)
        order_id = _require_identifier(order["order_id"], f"{path}.order_id")
        if order_id in seen:
            raise ProtocolViolation(f"duplicate order_id: {order_id}")
        seen.add(order_id)
        pair = _require_identifier(order["pair"], f"{path}.pair")
        if pair not in quote_pairs:
            raise ProtocolViolation(f"{path}.pair has no post-exit quote")
        created_epoch = _require_int(order["created_epoch"], f"{path}.created_epoch")
        valid_until_epoch = _require_int(
            order["valid_until_epoch"], f"{path}.valid_until_epoch"
        )
        if valid_until_epoch < created_epoch:
            raise ProtocolViolation(f"{path}.valid_until_epoch precedes created_epoch")
        normalized.append(
            {
                "order_id": order_id,
                "worker_id": _require_identifier(
                    order["worker_id"], f"{path}.worker_id"
                ),
                "owner_id": _require_identifier(order["owner_id"], f"{path}.owner_id"),
                "family_id": _require_identifier(
                    order["family_id"], f"{path}.family_id"
                ),
                "pair": pair,
                "side": _require_enum(order["side"], _SIDES, f"{path}.side"),
                "order_kind": _require_enum(
                    order["order_kind"],
                    frozenset({"LIMIT", "STOP"}),
                    f"{path}.order_kind",
                ),
                "units": _require_number(
                    order["units"], f"{path}.units", strictly_positive=True
                ),
                "trigger_price": _require_number(
                    order["trigger_price"],
                    f"{path}.trigger_price",
                    strictly_positive=True,
                ),
                "tp_price": _require_optional_price(
                    order["tp_price"], f"{path}.tp_price"
                ),
                "sl_price": _require_optional_price(
                    order["sl_price"], f"{path}.sl_price"
                ),
                "created_epoch": created_epoch,
                "valid_until_epoch": valid_until_epoch,
            }
        )
    return sorted(normalized, key=lambda row: row["order_id"])


def seal_post_exit_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and seal a reducer-produced, post-exit account snapshot."""

    raw = _require_mapping(snapshot, "snapshot")
    _require_exact_keys(raw, _RAW_SNAPSHOT_KEYS, "snapshot")
    epoch = _require_int(raw["epoch"], "snapshot.epoch")
    phase = _require_enum(raw["phase"], _PHASES, "snapshot.phase")
    expected_quote_pairs = _normalize_expected_quote_pairs(raw["expected_quote_pairs"])
    quotes = _normalize_quotes(raw["quotes"], epoch=epoch, phase=phase)
    quote_pairs = {quote["pair"] for quote in quotes}
    if quote_pairs != set(expected_quote_pairs):
        missing = sorted(set(expected_quote_pairs) - quote_pairs)
        extra = sorted(quote_pairs - set(expected_quote_pairs))
        raise ProtocolViolation(
            f"snapshot quote set is incomplete or unexpected: missing={missing}, extra={extra}"
        )
    active_worker_bindings = _normalize_worker_bindings(raw["active_worker_bindings"])
    positions = _normalize_positions(raw["positions"], quote_pairs)
    pending_orders = _normalize_pending_orders(raw["pending_orders"], quote_pairs)
    _require_bound_state_owners(positions, pending_orders, active_worker_bindings)
    body: dict[str, Any] = {
        "contract": POST_EXIT_SNAPSHOT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "coordinate_id": _require_identifier(
            raw["coordinate_id"], "snapshot.coordinate_id"
        ),
        "epoch": epoch,
        "phase": phase,
        "intrabar": _require_enum(
            raw["intrabar"], _INTRABAR_PATHS, "snapshot.intrabar"
        ),
        "quote_batch_sha256": _require_sha256(
            raw["quote_batch_sha256"], "snapshot.quote_batch_sha256"
        ),
        "quote_watermark": _require_int(
            raw["quote_watermark"], "snapshot.quote_watermark"
        ),
        "expected_quote_pairs": expected_quote_pairs,
        "active_worker_bindings": active_worker_bindings,
        "account": _normalize_account(raw["account"]),
        "quotes": quotes,
        "positions": positions,
        "pending_orders": pending_orders,
        "snapshot_state": "POST_EXIT",
        "read_only": True,
        "proposal_only": True,
        "allocation_allowed": False,
        "execution_allowed": False,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    body["snapshot_sha256"] = _sha256(body)
    return _copy_json(body)


def verify_post_exit_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a sealed snapshot and return a detached canonical copy."""

    sealed = _require_mapping(snapshot, "sealed_snapshot")
    _require_exact_keys(sealed, _SEALED_SNAPSHOT_KEYS, "sealed_snapshot")
    raw = {key: sealed[key] for key in _RAW_SNAPSHOT_KEYS}
    rebuilt = seal_post_exit_snapshot(raw)
    if _copy_json(sealed) != rebuilt:
        raise ProtocolViolation("sealed_snapshot content or snapshot_sha256 is invalid")
    return rebuilt


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def readonly_post_exit_snapshot(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the only worker-facing snapshot view: recursively immutable data."""

    return _deep_freeze(verify_post_exit_snapshot(snapshot))


def _snapshot_indexes(
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    quotes = {row["pair"]: row for row in snapshot["quotes"]}
    positions = {row["position_id"]: row for row in snapshot["positions"]}
    pending_orders = {row["order_id"]: row for row in snapshot["pending_orders"]}
    return quotes, positions, pending_orders


def _bound_worker_identity(
    snapshot: Mapping[str, Any], worker_id: str
) -> Mapping[str, str]:
    for binding in snapshot["active_worker_bindings"]:
        if binding["worker_id"] == worker_id:
            return binding
    raise ProtocolViolation(
        f"proposal worker_id is not active in the supplied snapshot: {worker_id}"
    )


def _require_owned(
    target: Mapping[str, Any],
    *,
    worker_id: str,
    owner_id: str,
    family_id: str,
    path: str,
) -> None:
    if (
        target["worker_id"] != worker_id
        or target["owner_id"] != owner_id
        or target["family_id"] != family_id
    ):
        raise ProtocolViolation(f"{path} target is not owned by this worker identity")


def _normalize_risk_reducing_intents(
    value: Any,
    *,
    snapshot: Mapping[str, Any],
    worker_id: str,
    owner_id: str,
    family_id: str,
) -> list[dict[str, Any]]:
    rows = _require_sequence(
        value, "proposal.risk_reducing_intents", MAX_INTENTS_PER_CLASS
    )
    quotes, positions, pending_orders = _snapshot_indexes(snapshot)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        path = f"proposal.risk_reducing_intents[{index}]"
        intent = _require_mapping(raw, path)
        _require_exact_keys(intent, _INTENT_KEYS, path)
        intent_id = _require_identifier(intent["intent_id"], f"{path}.intent_id")
        if intent_id in seen:
            raise ProtocolViolation(f"duplicate risk-reducing intent_id: {intent_id}")
        seen.add(intent_id)
        action = _require_enum(
            intent["action"], _RISK_REDUCING_ACTIONS, f"{path}.action"
        )
        parameters = _require_mapping(intent["parameters"], f"{path}.parameters")

        if action == "CANCEL_ORDER":
            _require_exact_keys(
                parameters, frozenset({"order_id"}), f"{path}.parameters"
            )
            order_id = _require_identifier(
                parameters["order_id"], f"{path}.parameters.order_id"
            )
            target = pending_orders.get(order_id)
            if target is None:
                raise ProtocolViolation(f"{path} refers to an unknown pending order")
            _require_owned(
                target,
                worker_id=worker_id,
                owner_id=owner_id,
                family_id=family_id,
                path=path,
            )
            normalized_parameters: dict[str, Any] = {"order_id": order_id}
        elif action == "CLOSE_POSITION":
            _require_exact_keys(
                parameters, frozenset({"position_id", "units"}), f"{path}.parameters"
            )
            position_id = _require_identifier(
                parameters["position_id"], f"{path}.parameters.position_id"
            )
            target = positions.get(position_id)
            if target is None:
                raise ProtocolViolation(f"{path} refers to an unknown position")
            _require_owned(
                target,
                worker_id=worker_id,
                owner_id=owner_id,
                family_id=family_id,
                path=path,
            )
            units = parameters["units"]
            if units is not None:
                units = _require_number(
                    units, f"{path}.parameters.units", strictly_positive=True
                )
                if units > target["units"]:
                    raise ProtocolViolation(
                        f"{path}.parameters.units exceeds the open position"
                    )
            normalized_parameters = {"position_id": position_id, "units": units}
        else:
            _require_exact_keys(
                parameters, frozenset({"position_id", "sl_price"}), f"{path}.parameters"
            )
            position_id = _require_identifier(
                parameters["position_id"], f"{path}.parameters.position_id"
            )
            target = positions.get(position_id)
            if target is None:
                raise ProtocolViolation(f"{path} refers to an unknown position")
            _require_owned(
                target,
                worker_id=worker_id,
                owner_id=owner_id,
                family_id=family_id,
                path=path,
            )
            sl_price = _require_number(
                parameters["sl_price"],
                f"{path}.parameters.sl_price",
                strictly_positive=True,
            )
            old_sl = target["sl_price"]
            quote = quotes[target["pair"]]
            if target["side"] == "LONG":
                if sl_price >= quote["bid"] or (
                    old_sl is not None and sl_price <= old_sl
                ):
                    raise ProtocolViolation(
                        f"{path} does not strictly tighten a viable LONG stop"
                    )
            elif sl_price <= quote["ask"] or (
                old_sl is not None and sl_price >= old_sl
            ):
                raise ProtocolViolation(
                    f"{path} does not strictly tighten a viable SHORT stop"
                )
            normalized_parameters = {"position_id": position_id, "sl_price": sl_price}

        normalized.append(
            {
                "intent_id": intent_id,
                "action": action,
                "parameters": normalized_parameters,
                "reason_code": _require_identifier(
                    intent["reason_code"],
                    f"{path}.reason_code",
                    maximum=MAX_REASON_LENGTH,
                ),
            }
        )
    return sorted(normalized, key=lambda row: row["intent_id"])


def _normalize_new_risk_intents(
    value: Any, *, snapshot: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = _require_sequence(value, "proposal.new_risk_intents", MAX_INTENTS_PER_CLASS)
    quotes, _, _ = _snapshot_indexes(snapshot)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        path = f"proposal.new_risk_intents[{index}]"
        intent = _require_mapping(raw, path)
        _require_exact_keys(intent, _INTENT_KEYS, path)
        intent_id = _require_identifier(intent["intent_id"], f"{path}.intent_id")
        if intent_id in seen:
            raise ProtocolViolation(f"duplicate new-risk intent_id: {intent_id}")
        seen.add(intent_id)
        action = _require_enum(intent["action"], _NEW_RISK_ACTIONS, f"{path}.action")
        parameters = _require_mapping(intent["parameters"], f"{path}.parameters")
        _require_exact_keys(parameters, _NEW_RISK_PARAMETER_KEYS, f"{path}.parameters")
        pair = _require_identifier(parameters["pair"], f"{path}.parameters.pair")
        quote = quotes.get(pair)
        if quote is None:
            raise ProtocolViolation(f"{path}.parameters.pair has no post-exit quote")
        side = _require_enum(parameters["side"], _SIDES, f"{path}.parameters.side")
        entry_price = _require_number(
            parameters["entry_price"],
            f"{path}.parameters.entry_price",
            strictly_positive=True,
        )
        if action == "MARKET":
            expected_entry = quote["ask"] if side == "LONG" else quote["bid"]
            if entry_price != expected_entry:
                raise ProtocolViolation(
                    f"{path}.parameters.entry_price is not the executable quote"
                )
        elif action == "LIMIT":
            if (side == "LONG" and entry_price > quote["ask"]) or (
                side == "SHORT" and entry_price < quote["bid"]
            ):
                raise ProtocolViolation(
                    f"{path}.parameters.entry_price is not a valid LIMIT"
                )
        elif (side == "LONG" and entry_price < quote["ask"]) or (
            side == "SHORT" and entry_price > quote["bid"]
        ):
            raise ProtocolViolation(
                f"{path}.parameters.entry_price is not a valid STOP"
            )

        sl_price = _require_number(
            parameters["sl_price"],
            f"{path}.parameters.sl_price",
            strictly_positive=True,
        )
        tp_price = _require_optional_price(
            parameters["tp_price"], f"{path}.parameters.tp_price"
        )
        if side == "LONG":
            if sl_price >= entry_price or (
                tp_price is not None and tp_price <= entry_price
            ):
                raise ProtocolViolation(f"{path} has invalid LONG exit geometry")
        elif sl_price <= entry_price or (
            tp_price is not None and tp_price >= entry_price
        ):
            raise ProtocolViolation(f"{path} has invalid SHORT exit geometry")

        valid_until_epoch = _require_int(
            parameters["valid_until_epoch"], f"{path}.parameters.valid_until_epoch"
        )
        minimum_valid_epoch = (
            snapshot["epoch"] if action == "MARKET" else snapshot["epoch"] + 1
        )
        if valid_until_epoch < minimum_valid_epoch:
            raise ProtocolViolation(
                f"{path}.parameters.valid_until_epoch expires before its permitted activation"
            )
        normalized.append(
            {
                "intent_id": intent_id,
                "action": action,
                "parameters": {
                    "pair": pair,
                    "side": side,
                    "units": _require_number(
                        parameters["units"],
                        f"{path}.parameters.units",
                        strictly_positive=True,
                    ),
                    "entry_price": entry_price,
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "stress_cost_pips": _require_number(
                        parameters["stress_cost_pips"],
                        f"{path}.parameters.stress_cost_pips",
                        minimum=0,
                    ),
                    "hard_max_holding_seconds": _require_int(
                        parameters["hard_max_holding_seconds"],
                        f"{path}.parameters.hard_max_holding_seconds",
                        minimum=1,
                    ),
                    "valid_until_epoch": valid_until_epoch,
                    "expected_net_edge_jpy": _require_number(
                        parameters["expected_net_edge_jpy"],
                        f"{path}.parameters.expected_net_edge_jpy",
                    ),
                    "activation_policy": (
                        "CURRENT_COORDINATE_AFTER_ALLOCATION"
                        if action == "MARKET"
                        else "NEXT_COORDINATE_OR_LATER"
                    ),
                },
                "reason_code": _require_identifier(
                    intent["reason_code"],
                    f"{path}.reason_code",
                    maximum=MAX_REASON_LENGTH,
                ),
            }
        )
    return sorted(normalized, key=lambda row: row["intent_id"])


def seal_worker_proposal(
    snapshot: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal one worker's two-class proposal against an exact snapshot digest."""

    verified_snapshot = verify_post_exit_snapshot(snapshot)
    raw = _require_mapping(proposal, "proposal")
    _require_exact_keys(raw, _RAW_PROPOSAL_KEYS, "proposal")
    snapshot_sha256 = _require_sha256(
        raw["snapshot_sha256"], "proposal.snapshot_sha256"
    )
    if snapshot_sha256 != verified_snapshot["snapshot_sha256"]:
        raise ProtocolViolation(
            "proposal.snapshot_sha256 does not match the supplied snapshot"
        )
    worker_id = _require_identifier(raw["worker_id"], "proposal.worker_id")
    owner_id = _require_identifier(raw["owner_id"], "proposal.owner_id")
    family_id = _require_identifier(raw["family_id"], "proposal.family_id")
    config_sha256 = _require_sha256(raw["config_sha256"], "proposal.config_sha256")
    binding = _bound_worker_identity(verified_snapshot, worker_id)
    if (
        owner_id != binding["owner_id"]
        or family_id != binding["family_id"]
        or config_sha256 != binding["config_sha256"]
    ):
        raise ProtocolViolation(
            "proposal identity/config does not match active_worker_bindings"
        )
    risk_reducing = _normalize_risk_reducing_intents(
        raw["risk_reducing_intents"],
        snapshot=verified_snapshot,
        worker_id=worker_id,
        owner_id=owner_id,
        family_id=family_id,
    )
    new_risk = _normalize_new_risk_intents(
        raw["new_risk_intents"], snapshot=verified_snapshot
    )
    overlap = {row["intent_id"] for row in risk_reducing} & {
        row["intent_id"] for row in new_risk
    }
    if overlap:
        raise ProtocolViolation(
            f"intent_id reused across intent classes: {sorted(overlap)}"
        )
    body: dict[str, Any] = {
        "contract": WORKER_PROPOSAL_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "worker_id": worker_id,
        "owner_id": owner_id,
        "family_id": family_id,
        "config_sha256": config_sha256,
        "snapshot_sha256": snapshot_sha256,
        "decision_basis": "READ_ONLY_POST_EXIT_SNAPSHOT",
        "risk_reducing_intents": risk_reducing,
        "new_risk_intents": new_risk,
        "intent_counts": {
            "risk_reducing": len(risk_reducing),
            "new_risk": len(new_risk),
        },
        "risk_reduction_processed_before_new_risk": True,
        "proposal_only": True,
        "allocation_allowed": False,
        "execution_allowed": False,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        # entry_price and expected_net_edge_jpy are worker claims.  The shared
        # reducer must recompute both from the sealed quote/economic receipts.
        "worker_economic_claims_authoritative": False,
        "reducer_recomputes_entry_and_edge": True,
    }
    body["proposal_sha256"] = _sha256(body)
    return _copy_json(body)


def verify_worker_proposal(
    snapshot: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a sealed worker proposal and all of its snapshot-bound semantics."""

    sealed = _require_mapping(proposal, "sealed_proposal")
    _require_exact_keys(sealed, _SEALED_PROPOSAL_KEYS, "sealed_proposal")
    # Rebuild from a detached copy: verification must never mutate the artifact
    # it is authenticating while removing sealer-derived fields.
    raw = _copy_json({key: sealed[key] for key in _RAW_PROPOSAL_KEYS})
    for intent in raw["new_risk_intents"]:
        parameters = _require_mapping(intent, "sealed_proposal.new_risk_intent")[
            "parameters"
        ]
        parameters = dict(
            _require_mapping(parameters, "sealed_proposal.new_risk_intent.parameters")
        )
        parameters.pop("activation_policy", None)
        intent["parameters"] = parameters
    rebuilt = seal_worker_proposal(snapshot, raw)
    if _copy_json(sealed) != rebuilt:
        raise ProtocolViolation("sealed_proposal content or proposal_sha256 is invalid")
    return rebuilt


def seal_worker_proposal_batch(
    snapshot: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Seal an input-order-independent set of already sealed worker proposals."""

    verified_snapshot = verify_post_exit_snapshot(snapshot)
    rows = _require_sequence(proposals, "proposals", MAX_PROPOSALS_PER_BATCH)
    verified = [verify_worker_proposal(verified_snapshot, row) for row in rows]
    verified.sort(key=lambda row: row["worker_id"])
    worker_ids = [row["worker_id"] for row in verified]
    if len(worker_ids) != len(set(worker_ids)):
        raise ProtocolViolation("proposal batch contains duplicate worker_id")
    expected_worker_ids = [
        binding["worker_id"] for binding in verified_snapshot["active_worker_bindings"]
    ]
    if worker_ids != expected_worker_ids:
        missing = sorted(set(expected_worker_ids) - set(worker_ids))
        extra = sorted(set(worker_ids) - set(expected_worker_ids))
        raise ProtocolViolation(
            "proposal batch must contain exactly one proposal for every active worker "
            f"(NO_INTENT is represented by empty arrays): missing={missing}, extra={extra}"
        )
    body: dict[str, Any] = {
        "contract": WORKER_PROPOSAL_BATCH_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "snapshot_sha256": verified_snapshot["snapshot_sha256"],
        "proposals": verified,
        "proposal_count": len(verified),
        "risk_reducing_intent_count": sum(
            row["intent_counts"]["risk_reducing"] for row in verified
        ),
        "new_risk_intent_count": sum(
            row["intent_counts"]["new_risk"] for row in verified
        ),
        "proposal_only": True,
        "allocation_allowed": False,
        "execution_allowed": False,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "worker_economic_claims_authoritative": False,
        "reducer_recomputes_entry_and_edge": True,
    }
    body["batch_sha256"] = _sha256(body)
    return _copy_json(body)


def verify_worker_proposal_batch(
    snapshot: Mapping[str, Any], batch: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a sealed proposal batch without granting reducer authority."""

    sealed = _require_mapping(batch, "sealed_batch")
    _require_exact_keys(sealed, _SEALED_BATCH_KEYS, "sealed_batch")
    rebuilt = seal_worker_proposal_batch(snapshot, sealed["proposals"])
    if _copy_json(sealed) != rebuilt:
        raise ProtocolViolation("sealed_batch content or batch_sha256 is invalid")
    return rebuilt


__all__ = [
    "POST_EXIT_SNAPSHOT_CONTRACT",
    "WORKER_PROPOSAL_BATCH_CONTRACT",
    "WORKER_PROPOSAL_CONTRACT",
    "ProtocolViolation",
    "readonly_post_exit_snapshot",
    "seal_post_exit_snapshot",
    "seal_worker_proposal",
    "seal_worker_proposal_batch",
    "verify_post_exit_snapshot",
    "verify_worker_proposal",
    "verify_worker_proposal_batch",
]

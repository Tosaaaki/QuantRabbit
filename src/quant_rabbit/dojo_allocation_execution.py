"""Content-addressed DOJO allocation -> VirtualBroker execution gate.

This module deliberately has authority over one object only: an in-process
``VirtualBroker``.  It cannot talk to OANDA or any other external broker.  A
selected allocator intent is submitted only after the allocation receipt,
selected intent, sealed execution configuration, and exact virtual-broker
state have all been re-hashed and reconciled.

Version 1 does *not* claim a fill-time aggregate stop-loss risk gate.  The V4
allocator preflights gross open + pending + candidate SL risk, while
``VirtualBroker`` rechecks only broker margin headroom and owner concurrency at
the actual fill.  Revaluing that aggregate risk against the fill quote remains
a future execution-gate boundary and is surfaced explicitly in every receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_lab_provenance import (
    OwnedBrokerView,
    strategy_ownership_registry,
)
from quant_rabbit.dojo_portfolio_allocator import (
    ALLOCATION_CONTRACT,
    DojoPortfolioAllocatorError,
    score_allocation_opportunity_cost,
)
from quant_rabbit.virtual_broker import VirtualBroker

EXECUTION_CONFIG_CONTRACT = "QR_DOJO_ALLOCATION_EXECUTION_CONFIG_V1"
PREPARED_SUBMISSION_CONTRACT = "QR_DOJO_ALLOCATION_PREPARED_SUBMISSION_V1"
SUBMISSION_CONTRACT = "QR_DOJO_ALLOCATION_VIRTUAL_SUBMISSION_V1"
EXECUTION_VERIFICATION_CONTRACT = "QR_DOJO_ALLOCATION_EXECUTION_VERIFICATION_V1"
_CLAIM_EVENT = "DOJO_ALLOCATION_EXECUTION_CLAIM"
_FAIL_EVENT = "DOJO_ALLOCATION_EXECUTION_FAILED"
_RISK_MODEL = "ALLOCATION_TIME_GROSS_SL_CAP_NO_FILL_TIME_SL_RECHECK"
_LOCK_ATTR = "_dojo_allocation_execution_lock"
_LOCKED_ATTR = "_dojo_allocation_execution_methods_locked"
_PENDING_SETTINGS_ATTR = "_dojo_allocation_pending_fill_settings"
_LOCK_INSTALL_GUARD = threading.Lock()


class DojoAllocationExecutionError(ValueError):
    """A sealed allocation cannot be submitted to the virtual broker."""


def _canonical_sha(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoAllocationExecutionError(
            "execution evidence is not canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _identity(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise DojoAllocationExecutionError(f"{field} must contain 1..128 characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise DojoAllocationExecutionError(f"{field} must be visible ASCII")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DojoAllocationExecutionError(f"{field} must be a positive integer")
    return value


def _non_negative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DojoAllocationExecutionError(f"{field} must be a non-negative integer")
    return value


def _optional_number_matches(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    try:
        return math.isclose(float(actual), float(expected), abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _strict_mapping(
    value: Any, *, field: str, expected_keys: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise DojoAllocationExecutionError(f"{field} schema mismatch")
    return value


def _broker_state_sha256(broker: VirtualBroker) -> str:
    """Hash the complete restorable virtual-broker state before a claim."""

    return _canonical_sha(broker.snapshot())


def _quote_epoch(timestamp: Any) -> int:
    if not isinstance(timestamp, str) or not timestamp:
        raise DojoAllocationExecutionError("selected quote timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(timestamp.split("#", 1)[0])
    except (TypeError, ValueError) as exc:
        raise DojoAllocationExecutionError(
            "selected quote timestamp is invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DojoAllocationExecutionError(
            "selected quote timestamp must include an explicit UTC offset"
        )
    epoch = parsed.timestamp()
    if not math.isfinite(epoch) or not epoch.is_integer() or epoch < 0:
        raise DojoAllocationExecutionError(
            "selected quote timestamp must resolve to an exact epoch second"
        )
    return int(epoch)


def _truncate_ledger_to(
    broker: VirtualBroker, *, size_bytes: int, terminal_sha256: str
) -> None:
    """Restore a known durable ledger boundary after a virtual mutation fails."""

    broker._handle.flush()
    os.ftruncate(broker._handle.fileno(), size_bytes)
    broker._handle.seek(0, os.SEEK_END)
    if not broker.fast_ledger:
        os.fsync(broker._handle.fileno())
    broker._prev_sha = terminal_sha256


def _pending_fill_settings(broker: VirtualBroker) -> dict[str, dict[str, Any]]:
    settings = getattr(broker, _PENDING_SETTINGS_ATTR, None)
    if settings is None:
        settings = {}
        setattr(broker, _PENDING_SETTINGS_ATTR, settings)
    if not isinstance(settings, dict):
        raise DojoAllocationExecutionError(
            "virtual broker pending fill settings registry is invalid"
        )
    return settings


def _validate_pending_fill_settings(value: Any) -> dict[str, Any]:
    row = _strict_mapping(
        value,
        field="pending_fill_settings",
        expected_keys={
            "leverage",
            "slippage_pips",
            "financing_pips_per_day",
            "fast_ledger",
        },
    )
    for key in ("leverage", "slippage_pips", "financing_pips_per_day"):
        number = row[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0
        ):
            raise DojoAllocationExecutionError(
                f"pending_fill_settings.{key} is invalid"
            )
    if float(row["leverage"]) <= 0 or not isinstance(row["fast_ledger"], bool):
        raise DojoAllocationExecutionError("pending fill settings are invalid")
    return dict(row)


def _assert_pending_fill_settings(broker: VirtualBroker) -> None:
    for order_id, expected_raw in _pending_fill_settings(broker).items():
        if order_id not in broker.orders:
            continue
        expected = _validate_pending_fill_settings(expected_raw)
        if (
            not math.isclose(
                broker.leverage,
                float(expected["leverage"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                broker.slippage_pips,
                float(expected["slippage_pips"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                broker.financing_pips_per_day,
                float(expected["financing_pips_per_day"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or broker.fast_ledger is not expected["fast_ledger"]
        ):
            raise DojoAllocationExecutionError(
                "pending virtual order execution settings changed before fill"
            )


def _install_broker_lock(broker: VirtualBroker) -> threading.RLock:
    """Serialize virtual quotes and mutations across owner sessions in-process."""

    # Installing the per-broker lock is itself shared state.  Without this
    # guard, two first-time session constructors could each retain a distinct
    # lock while only one of them protected the wrapped broker methods.
    with _LOCK_INSTALL_GUARD:
        if (
            getattr(broker, "_state_restore_required", False) is True
            and getattr(broker, "_state_restore_verified", False) is not True
        ):
            raise DojoAllocationExecutionError(
                "virtual broker has stateful ledger history but no verified snapshot restore"
            )
        existing = getattr(broker, _LOCK_ATTR, None)
        if existing is None:
            existing = threading.RLock()
            setattr(broker, _LOCK_ATTR, existing)
        if not isinstance(existing, type(threading.RLock())):
            raise DojoAllocationExecutionError(
                "virtual broker execution lock is invalid"
            )
        pending_settings = _pending_fill_settings(broker)
        ledger_records = _read_ledger(
            broker.ledger_path, expected_terminal_sha256=broker._prev_sha
        )
        durable_sequence = broker._seq
        for record in ledger_records:
            payload = record.get("payload")
            if isinstance(payload, Mapping):
                for key in ("trade_id", "order_id", "virtual_broker_identity"):
                    identity = payload.get(key)
                    if (
                        isinstance(identity, str)
                        and len(identity) == 7
                        and identity[0] in {"T", "O"}
                        and identity[1:].isdigit()
                    ):
                        durable_sequence = max(durable_sequence, int(identity[1:]))
        broker._seq = durable_sequence
        for record in ledger_records:
            if record.get("event") != _CLAIM_EVENT:
                continue
            payload = record.get("payload")
            if (
                not isinstance(payload, Mapping)
                or payload.get("broker_identity_kind") != "ORDER_ID"
            ):
                continue
            order_id = payload.get("virtual_broker_identity")
            expected_raw = payload.get("pending_fill_settings")
            if not isinstance(order_id, str) or not order_id:
                raise DojoAllocationExecutionError(
                    "durable pending claim identity is invalid"
                )
            expected = _validate_pending_fill_settings(expected_raw)
            prior = pending_settings.get(order_id)
            if prior is not None and prior != expected:
                raise DojoAllocationExecutionError(
                    "durable pending claim settings conflict"
                )
            pending_settings[order_id] = expected
        if getattr(broker, _LOCKED_ATTR, False) is not True:
            for method_name in (
                "market_order",
                "limit_order",
                "stop_order",
                "cancel_order",
                "close_trade",
                "set_exit",
                "on_quote",
                "on_quote_batch",
                "restore",
            ):
                original = getattr(broker, method_name)

                def locked(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                    with existing:
                        if _original.__name__ in {"on_quote", "on_quote_batch"}:
                            _assert_pending_fill_settings(broker)
                        return _original(*args, **kwargs)

                setattr(broker, method_name, locked)
            setattr(broker, _LOCKED_ATTR, True)
        return existing


def _read_ledger(
    path: Path, *, expected_terminal_sha256: str | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DojoAllocationExecutionError(
            "virtual broker ledger is unreadable"
        ) from exc
    expected_previous = "0" * 64
    for line_number, line in enumerate(raw_lines, start=1):
        if not line:
            raise DojoAllocationExecutionError(
                f"virtual broker ledger is blank at line {line_number}"
            )
        try:

            def reject_duplicate_keys(
                pairs: list[tuple[str, Any]],
            ) -> dict[str, Any]:
                parsed: dict[str, Any] = {}
                for key, value in pairs:
                    if key in parsed:
                        raise ValueError(f"duplicate JSON key: {key}")
                    parsed[key] = value
                return parsed

            record = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
                object_pairs_hook=reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise DojoAllocationExecutionError(
                f"virtual broker ledger JSON is invalid at line {line_number}"
            ) from exc
        if not isinstance(record, dict) or set(record) != {
            "ts_utc",
            "event",
            "payload",
            "prev_sha",
            "sha",
        }:
            raise DojoAllocationExecutionError(
                f"virtual broker ledger schema is invalid at line {line_number}"
            )
        body = {key: value for key, value in record.items() if key != "sha"}
        if (
            record["prev_sha"] != expected_previous
            or not isinstance(record["sha"], str)
            or record["sha"] != _canonical_sha(body)
        ):
            raise DojoAllocationExecutionError(
                f"virtual broker ledger hash chain is invalid at line {line_number}"
            )
        expected_previous = record["sha"]
        records.append(record)
    if (
        expected_terminal_sha256 is not None
        and expected_previous != expected_terminal_sha256
    ):
        raise DojoAllocationExecutionError(
            "virtual broker ledger terminal hash does not match broker state"
        )
    return records


def _claim_exists(broker: VirtualBroker, prepared_sha256: str) -> bool:
    for record in _read_ledger(
        broker.ledger_path, expected_terminal_sha256=broker._prev_sha
    ):
        if record.get("event") != _CLAIM_EVENT:
            continue
        payload = record.get("payload")
        if (
            isinstance(payload, Mapping)
            and payload.get("prepared_submission_sha256") == prepared_sha256
        ):
            return True
    return False


def _matching_ledger_records(
    broker: VirtualBroker, event: str, **payload_matches: Any
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for record in _read_ledger(
        broker.ledger_path, expected_terminal_sha256=broker._prev_sha
    ):
        if record.get("event") != event:
            continue
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if all(payload.get(key) == value for key, value in payload_matches.items()):
            matches.append(record)
    return matches


def _selected_intent(
    allocation: Mapping[str, Any], selected_intent: Mapping[str, Any]
) -> tuple[Mapping[str, Any], str]:
    """Reconstruct the allocator receipt and return its one selected row."""

    try:
        # Empty outcomes intentionally leave the diagnostic pending while
        # exercising the allocator's complete canonical reconstruction gate.
        score_allocation_opportunity_cost(allocation, [])
    except DojoPortfolioAllocatorError as exc:
        raise DojoAllocationExecutionError(
            "allocation receipt verification failed"
        ) from exc
    if allocation.get("contract") != ALLOCATION_CONTRACT:
        raise DojoAllocationExecutionError("allocation contract is unsupported")
    decision = allocation.get("decision")
    if not isinstance(decision, Mapping):
        raise DojoAllocationExecutionError("allocation decision is missing")
    selected_id = decision.get("selected_intent_id")
    if (
        not isinstance(selected_id, str)
        or not selected_id
        or decision.get("entry_admitted") is not True
    ):
        raise DojoAllocationExecutionError("allocation did not select an entry")
    rows = allocation.get("candidate_intent_log")
    if not isinstance(rows, list):
        raise DojoAllocationExecutionError("allocation candidate log is missing")
    selected_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("intent"), Mapping)
        and row["intent"].get("intent_id") == selected_id
        and row.get("disposition") == "SELECTED"
    ]
    if len(selected_rows) != 1:
        raise DojoAllocationExecutionError("allocation selected row is inconsistent")
    canonical_intent = selected_rows[0]["intent"]
    if dict(selected_intent) != dict(canonical_intent):
        raise DojoAllocationExecutionError(
            "submitted intent is not the exact selected allocation intent"
        )
    intent_sha = selected_rows[0].get("intent_identity_sha256")
    if not isinstance(intent_sha, str) or intent_sha != _canonical_sha(
        canonical_intent
    ):
        raise DojoAllocationExecutionError("selected intent identity hash is invalid")
    return canonical_intent, intent_sha


def _owner_cap(allocation: Mapping[str, Any], owner_id: str) -> tuple[int, int]:
    policy = allocation.get("policy")
    rows = policy.get("owner_concurrency_caps") if isinstance(policy, Mapping) else None
    if not isinstance(rows, list):
        raise DojoAllocationExecutionError("allocation owner caps are missing")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("owner_id") == owner_id
    ]
    if len(matches) != 1:
        raise DojoAllocationExecutionError(
            "selected owner cap is missing or duplicated"
        )
    return (
        _positive_integer(
            matches[0].get("max_concurrent_per_pair"),
            field="max_concurrent_per_pair",
        ),
        _positive_integer(
            matches[0].get("global_max_concurrent"),
            field="global_max_concurrent",
        ),
    )


def _validate_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    row = _strict_mapping(
        config,
        field="sealed_config",
        expected_keys={
            "contract",
            "schema_version",
            "run_id",
            "owner_id",
            "max_concurrent_per_pair",
            "global_max_concurrent",
            "expected_virtual_broker_state_sha256",
            "expected_leverage",
            "expected_slippage_pips",
            "expected_financing_pips_per_day",
            "expected_fast_ledger",
            "broker_kind",
            "allowed_order_kinds",
            "risk_enforcement_model",
            "allocation_time_aggregate_sl_cap_enforced",
            "fill_time_aggregate_sl_cap_rechecked",
            "virtual_broker_mutation_allowed",
            "external_broker_authority",
            "live_permission",
            "config_sha256",
        },
    )
    body = {key: value for key, value in row.items() if key != "config_sha256"}
    if (
        row["contract"] != EXECUTION_CONFIG_CONTRACT
        or row["schema_version"] != 1
        or isinstance(row["schema_version"], bool)
        or row["broker_kind"] != "QUANT_RABBIT_VIRTUAL_BROKER"
        or row["allowed_order_kinds"] != ["LIMIT", "MARKET", "STOP"]
        or row["risk_enforcement_model"] != _RISK_MODEL
        or row["allocation_time_aggregate_sl_cap_enforced"] is not True
        or row["fill_time_aggregate_sl_cap_rechecked"] is not False
        or row["virtual_broker_mutation_allowed"] is not True
        or row["external_broker_authority"] != "NONE"
        or row["live_permission"] is not False
        or row["config_sha256"] != _canonical_sha(body)
    ):
        raise DojoAllocationExecutionError("sealed execution config is invalid")
    _identity(row["run_id"], field="sealed_config.run_id")
    _identity(row["owner_id"], field="sealed_config.owner_id")
    _positive_integer(
        row["max_concurrent_per_pair"], field="sealed_config.max_concurrent_per_pair"
    )
    _positive_integer(
        row["global_max_concurrent"], field="sealed_config.global_max_concurrent"
    )
    if (
        not isinstance(row["expected_virtual_broker_state_sha256"], str)
        or len(row["expected_virtual_broker_state_sha256"]) != 64
    ):
        raise DojoAllocationExecutionError("sealed broker state hash is invalid")
    for field in (
        "expected_leverage",
        "expected_slippage_pips",
        "expected_financing_pips_per_day",
    ):
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DojoAllocationExecutionError(f"sealed_config.{field} is invalid")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise DojoAllocationExecutionError(f"sealed_config.{field} is invalid")
    if float(row["expected_leverage"]) <= 0:
        raise DojoAllocationExecutionError("sealed_config.expected_leverage is invalid")
    if not isinstance(row["expected_fast_ledger"], bool):
        raise DojoAllocationExecutionError(
            "sealed_config.expected_fast_ledger is invalid"
        )
    return row


class DojoAllocationExecutionSession:
    """One owner-scoped selected-intent gate for a ``VirtualBroker`` instance."""

    def __init__(
        self,
        broker: VirtualBroker,
        owner_id: str,
        *,
        max_concurrent_per_pair: int,
        global_max_concurrent: int,
    ) -> None:
        if type(broker) is not VirtualBroker:
            raise DojoAllocationExecutionError(
                "execution session requires the exact VirtualBroker type"
            )
        self._broker = broker
        self._broker_lock = _install_broker_lock(broker)
        self.owner_id = _identity(owner_id, field="owner_id")
        self.max_concurrent_per_pair = _positive_integer(
            max_concurrent_per_pair, field="max_concurrent_per_pair"
        )
        self.global_max_concurrent = _positive_integer(
            global_max_concurrent, field="global_max_concurrent"
        )
        with self._broker_lock:
            self._owner_view = OwnedBrokerView(
                broker,
                self.owner_id,
                max_concurrent_per_pair=self.max_concurrent_per_pair,
                global_max_concurrent=self.global_max_concurrent,
            )

    def seal_config(self, *, run_id: str) -> dict[str, Any]:
        """Seal the exact virtual state and fill-time contracts for one submit."""

        with self._broker_lock:
            return self._seal_config_locked(run_id=run_id)

    def _seal_config_locked(self, *, run_id: str) -> dict[str, Any]:
        """Build a config while the virtual broker state is stable."""

        body = {
            "contract": EXECUTION_CONFIG_CONTRACT,
            "schema_version": 1,
            "run_id": _identity(run_id, field="run_id"),
            "owner_id": self.owner_id,
            "max_concurrent_per_pair": self.max_concurrent_per_pair,
            "global_max_concurrent": self.global_max_concurrent,
            "expected_virtual_broker_state_sha256": _broker_state_sha256(self._broker),
            "expected_leverage": self._broker.leverage,
            "expected_slippage_pips": self._broker.slippage_pips,
            "expected_financing_pips_per_day": self._broker.financing_pips_per_day,
            "expected_fast_ledger": self._broker.fast_ledger,
            "broker_kind": "QUANT_RABBIT_VIRTUAL_BROKER",
            "allowed_order_kinds": ["LIMIT", "MARKET", "STOP"],
            "risk_enforcement_model": _RISK_MODEL,
            "allocation_time_aggregate_sl_cap_enforced": True,
            "fill_time_aggregate_sl_cap_rechecked": False,
            "virtual_broker_mutation_allowed": True,
            "external_broker_authority": "NONE",
            "live_permission": False,
        }
        return {**body, "config_sha256": _canonical_sha(body)}

    def prepare(
        self,
        *,
        allocation: Mapping[str, Any],
        selected_intent: Mapping[str, Any],
        sealed_config: Mapping[str, Any],
        execution_nonce: str,
    ) -> dict[str, Any]:
        """Bind a selected intent to its allocator and virtual-state receipts."""

        with self._broker_lock:
            return self._prepare_locked(
                allocation=allocation,
                selected_intent=selected_intent,
                sealed_config=sealed_config,
                execution_nonce=execution_nonce,
            )

    def _prepare_locked(
        self,
        *,
        allocation: Mapping[str, Any],
        selected_intent: Mapping[str, Any],
        sealed_config: Mapping[str, Any],
        execution_nonce: str,
    ) -> dict[str, Any]:
        """Reconstruct a prepared receipt while virtual state is stable."""

        config = _validate_config(sealed_config)
        intent, intent_sha = _selected_intent(allocation, selected_intent)
        if config["owner_id"] != self.owner_id or intent["owner_id"] != self.owner_id:
            raise DojoAllocationExecutionError("execution owner does not match intent")
        pair_cap, global_cap = _owner_cap(allocation, self.owner_id)
        if (
            pair_cap != self.max_concurrent_per_pair
            or global_cap != self.global_max_concurrent
            or config["max_concurrent_per_pair"] != pair_cap
            or config["global_max_concurrent"] != global_cap
        ):
            raise DojoAllocationExecutionError(
                "sealed owner caps do not match the allocator and broker session"
            )
        if config["expected_virtual_broker_state_sha256"] != _broker_state_sha256(
            self._broker
        ):
            raise DojoAllocationExecutionError(
                "virtual broker state changed after config sealing"
            )
        if (
            not math.isclose(
                float(config["expected_leverage"]),
                self._broker.leverage,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(config["expected_slippage_pips"]),
                self._broker.slippage_pips,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(config["expected_financing_pips_per_day"]),
                self._broker.financing_pips_per_day,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or config["expected_fast_ledger"] is not self._broker.fast_ledger
        ):
            raise DojoAllocationExecutionError(
                "virtual broker execution settings changed after config sealing"
            )
        plan = allocation["decision"].get("selected_plan")
        if (
            not isinstance(plan, Mapping)
            or plan.get("position_id") is not None
            or float(plan.get("reduction_fraction", -1)) != 0.0
            or plan.get("action") != "HOLD_FULL"
        ):
            raise DojoAllocationExecutionError(
                "atomic incumbent release is unsupported; allocation must fit "
                "without reducing or closing a position"
            )
        selected_quote = self._broker.last_quotes.get(str(intent["pair"]))
        if selected_quote is None:
            raise DojoAllocationExecutionError("selected pair quote is missing")
        selected_quote_epoch = _quote_epoch(selected_quote[2])
        if selected_quote_epoch != allocation["decision_epoch"]:
            raise DojoAllocationExecutionError(
                "selected quote epoch does not match the allocation decision epoch"
            )
        quote_sequence = self._broker._last_quote_watermarks.get(str(intent["pair"]))
        if quote_sequence is None:
            raise DojoAllocationExecutionError(
                "selected quote accounting sequence is missing"
            )
        if (
            selected_quote[2] != allocation["decision_quote_timestamp"]
            or quote_sequence != allocation["decision_quote_sequence"]
        ):
            raise DojoAllocationExecutionError(
                "selected quote phase/watermark does not match allocation"
            )
        body = {
            "contract": PREPARED_SUBMISSION_CONTRACT,
            "schema_version": 1,
            "execution_nonce": _identity(execution_nonce, field="execution_nonce"),
            "run_id": config["run_id"],
            "decision_epoch": allocation["decision_epoch"],
            "valid_until_epoch": intent["valid_until_epoch"],
            "allocation_contract": allocation["contract"],
            "allocation_sha256": allocation["allocation_sha256"],
            "selected_intent_id": intent["intent_id"],
            "selected_intent_sha256": intent_sha,
            "sealed_config_sha256": config["config_sha256"],
            "expected_virtual_broker_state_sha256": config[
                "expected_virtual_broker_state_sha256"
            ],
            "owner_id": self.owner_id,
            "pair": intent["pair"],
            "side": intent["side"],
            "order_kind": intent["order_kind"],
            "units": intent["units"],
            "entry_price": intent["entry_price"],
            "tp_price": intent["tp_price"],
            "sl_price": intent["sl_price"],
            "selected_quote_bid": selected_quote[0],
            "selected_quote_ask": selected_quote[1],
            "selected_quote_timestamp": selected_quote[2],
            "selected_quote_epoch": selected_quote_epoch,
            "selected_quote_sequence": quote_sequence,
            "risk_enforcement_model": _RISK_MODEL,
            "allocation_time_aggregate_sl_cap_enforced": True,
            "fill_time_aggregate_sl_cap_rechecked": False,
            "fill_time_rechecks": [
                "VIRTUAL_BROKER_MARGIN_HEADROOM",
                "OWNER_PAIR_AND_GLOBAL_CONCURRENCY",
            ],
            "order_authority": "DOJO_VIRTUAL_BROKER_ONLY",
            "external_broker_authority": "NONE",
            "live_permission": False,
        }
        return {**body, "prepared_submission_sha256": _canonical_sha(body)}

    def _reconcile_allocation_state(self, allocation: Mapping[str, Any]) -> None:
        """Match every allocated open/pending row to actual virtual broker state."""

        if not math.isclose(
            float(allocation["policy"]["leverage"]),
            self._broker.leverage,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DojoAllocationExecutionError("virtual broker leverage drifted")
        account = self._broker.account()
        if not math.isclose(
            float(allocation["account"]["equity_jpy"]),
            float(account["equity_jpy"]),
            rel_tol=0.0,
            abs_tol=0.01,
        ):
            raise DojoAllocationExecutionError("virtual broker equity drifted")
        ownership = strategy_ownership_registry(self._broker)
        position_rows = allocation["open_positions"]
        pending_rows = allocation["pending_orders"]
        if {row["position_id"] for row in position_rows} != set(self._broker.positions):
            raise DojoAllocationExecutionError(
                "allocation open-position set does not match virtual broker"
            )
        if {row["order_id"] for row in pending_rows} != set(self._broker.orders):
            raise DojoAllocationExecutionError(
                "allocation pending-order set does not match virtual broker"
            )
        for row in position_rows:
            position = self._broker.positions[row["position_id"]]
            quote = self._broker.last_quotes.get(position.pair)
            if quote is None:
                raise DojoAllocationExecutionError("position quote is missing")
            quote_sequence = self._broker._last_quote_watermarks.get(position.pair)
            if quote_sequence is None:
                raise DojoAllocationExecutionError("position quote sequence is missing")
            mark = (quote[0] + quote[1]) / 2.0
            if (
                ownership.historical_trade_owner(position.trade_id) != row["owner_id"]
                or position.pair != row["pair"]
                or position.side != row["side"]
                or not math.isclose(position.units, float(row["units"]), abs_tol=1e-9)
                or not math.isclose(mark, float(row["mark_price"]), abs_tol=1e-9)
                or not math.isclose(quote[0], float(row["bid_price"]), abs_tol=1e-9)
                or not math.isclose(quote[1], float(row["ask_price"]), abs_tol=1e-9)
                or quote[2] != row["quote_timestamp"]
                or quote_sequence != row["quote_sequence"]
                or position.sl_price is None
                or not math.isclose(
                    position.sl_price, float(row["sl_price"]), abs_tol=1e-9
                )
                or float(row["stress_cost_pips"]) + 1e-12 < self._broker.slippage_pips
            ):
                raise DojoAllocationExecutionError(
                    "allocation position evidence does not match virtual broker"
                )
        for row in pending_rows:
            order = self._broker.orders[row["order_id"]]
            if (
                ownership.historical_order_owner(order.order_id) != row["owner_id"]
                or order.pair != row["pair"]
                or order.side != row["side"]
                or not math.isclose(order.units, float(row["units"]), abs_tol=1e-9)
                or not math.isclose(
                    order.limit_price, float(row["trigger_price"]), abs_tol=1e-9
                )
                or order.sl_pips is None
                or not math.isclose(order.sl_pips, float(row["sl_pips"]), abs_tol=1e-9)
                or float(row["stress_cost_pips"]) + 1e-12 < self._broker.slippage_pips
            ):
                raise DojoAllocationExecutionError(
                    "allocation pending evidence does not match virtual broker"
                )

    @staticmethod
    def _distance_pips(intent: Mapping[str, Any], key: str) -> float | None:
        price = intent[key]
        if price is None:
            return None
        pip = 0.01 if str(intent["pair"]).endswith("JPY") else 0.0001
        return abs(float(price) - float(intent["entry_price"])) / pip

    def submit(
        self,
        *,
        allocation: Mapping[str, Any],
        selected_intent: Mapping[str, Any],
        sealed_config: Mapping[str, Any],
        prepared_submission: Mapping[str, Any],
        execution_epoch: int,
    ) -> dict[str, Any]:
        """Serialize state reconciliation, durable claim, release, and submit."""

        with self._broker_lock:
            return self._submit_locked(
                allocation=allocation,
                selected_intent=selected_intent,
                sealed_config=sealed_config,
                prepared_submission=prepared_submission,
                execution_epoch=execution_epoch,
            )

    def _submit_locked(
        self,
        *,
        allocation: Mapping[str, Any],
        selected_intent: Mapping[str, Any],
        sealed_config: Mapping[str, Any],
        prepared_submission: Mapping[str, Any],
        execution_epoch: int,
    ) -> dict[str, Any]:
        """Consume one prepared receipt and submit only its selected intent."""

        prepared_sha = prepared_submission.get("prepared_submission_sha256")
        if not isinstance(prepared_sha, str) or len(prepared_sha) != 64:
            raise DojoAllocationExecutionError("prepared submission hash is invalid")
        if _claim_exists(self._broker, prepared_sha):
            raise DojoAllocationExecutionError(
                "prepared submission was already claimed"
            )
        rebuilt = self.prepare(
            allocation=allocation,
            selected_intent=selected_intent,
            sealed_config=sealed_config,
            execution_nonce=prepared_submission.get("execution_nonce"),
        )
        if dict(prepared_submission) != rebuilt:
            raise DojoAllocationExecutionError("prepared submission was tampered")
        epoch = _non_negative_integer(execution_epoch, field="execution_epoch")
        if (
            epoch != rebuilt["decision_epoch"]
            or epoch != rebuilt["selected_quote_epoch"]
            or epoch > rebuilt["valid_until_epoch"]
        ):
            raise DojoAllocationExecutionError(
                "execution epoch is not the exact selected quote decision epoch"
            )
        self._reconcile_allocation_state(allocation)
        intent, intent_sha = _selected_intent(allocation, selected_intent)
        if float(intent["stress_cost_pips"]) + 1e-12 < self._broker.slippage_pips:
            raise DojoAllocationExecutionError(
                "selected intent stress cost is below sealed broker slippage"
            )
        if intent["order_kind"] == "MARKET":
            quote = self._broker.last_quotes.get(intent["pair"])
            if quote is None:
                raise DojoAllocationExecutionError("market intent quote is missing")
            executable = quote[1] if intent["side"] == "LONG" else quote[0]
            precision = 3 if str(intent["pair"]).endswith("JPY") else 5
            if round(float(intent["entry_price"]), precision) != round(
                executable, precision
            ):
                raise DojoAllocationExecutionError(
                    "market intent entry does not match the sealed executable quote"
                )

        tp_pips = self._distance_pips(intent, "tp_price")
        sl_pips = self._distance_pips(intent, "sl_price")
        if intent["order_kind"] == "MARKET":
            identity_kind = "TRADE_ID"
            fill_status = "FILLED"
            identity_prefix = "T"
        else:
            identity_kind = "ORDER_ID"
            fill_status = "PENDING"
            identity_prefix = "O"
        expected_broker_identity = f"{identity_prefix}{self._broker._seq + 1:06d}"
        identity_field = "trade_id" if identity_kind == "TRADE_ID" else "order_id"
        if any(
            isinstance(record.get("payload"), Mapping)
            and (
                record["payload"].get(identity_field) == expected_broker_identity
                or record["payload"].get("virtual_broker_identity")
                == expected_broker_identity
            )
            for record in _read_ledger(
                self._broker.ledger_path,
                expected_terminal_sha256=self._broker._prev_sha,
            )
        ):
            raise DojoAllocationExecutionError(
                "next virtual broker identity already exists in the durable ledger"
            )
        core = {
            "contract": SUBMISSION_CONTRACT,
            "schema_version": 1,
            "prepared_submission_sha256": prepared_sha,
            "allocation_sha256": allocation["allocation_sha256"],
            "selected_intent_id": intent["intent_id"],
            "selected_intent_sha256": intent_sha,
            "sealed_config_sha256": sealed_config["config_sha256"],
            "execution_nonce": rebuilt["execution_nonce"],
            "run_id": rebuilt["run_id"],
            "execution_epoch": epoch,
            "owner_id": self.owner_id,
            "pair": intent["pair"],
            "side": intent["side"],
            "order_kind": intent["order_kind"],
            "units": intent["units"],
            "entry_price": intent["entry_price"],
            "tp_distance_pips": tp_pips,
            "sl_distance_pips": sl_pips,
            "selected_quote_bid": rebuilt["selected_quote_bid"],
            "selected_quote_ask": rebuilt["selected_quote_ask"],
            "selected_quote_timestamp": rebuilt["selected_quote_timestamp"],
            "selected_quote_epoch": rebuilt["selected_quote_epoch"],
            "selected_quote_sequence": rebuilt["selected_quote_sequence"],
            "expected_execution_leverage": self._broker.leverage,
            "expected_execution_slippage_pips": self._broker.slippage_pips,
            "expected_execution_financing_pips_per_day": (
                self._broker.financing_pips_per_day
            ),
            "expected_execution_fast_ledger": self._broker.fast_ledger,
            "broker_identity_kind": identity_kind,
            "virtual_broker_identity": expected_broker_identity,
            "fill_status_at_submission": fill_status,
            "release_receipt": None,
            "risk_enforcement_model": _RISK_MODEL,
            "allocation_time_aggregate_sl_cap_enforced": True,
            "fill_time_aggregate_sl_cap_rechecked": False,
            "fill_time_total_loss_rechecked": False,
            "fill_time_rechecks": [
                "VIRTUAL_BROKER_MARGIN_HEADROOM",
                "OWNER_PAIR_AND_GLOBAL_CONCURRENCY",
            ],
            "order_authority": "DOJO_VIRTUAL_BROKER_ONLY",
            "external_broker_authority": "NONE",
            "live_permission": False,
        }
        core_sha = _canonical_sha(core)
        bound_fill_settings = {
            "leverage": self._broker.leverage,
            "slippage_pips": self._broker.slippage_pips,
            "financing_pips_per_day": self._broker.financing_pips_per_day,
            "fast_ledger": self._broker.fast_ledger,
        }

        prior_claims = _matching_ledger_records(self._broker, _CLAIM_EVENT)
        for record in prior_claims:
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if (
                payload.get("allocation_sha256") == allocation["allocation_sha256"]
                and payload.get("selected_intent_sha256") == intent_sha
            ):
                raise DojoAllocationExecutionError(
                    "allocation and selected intent were already claimed"
                )
            if payload.get("run_id") != rebuilt["run_id"]:
                continue
            if payload.get("execution_nonce") == rebuilt["execution_nonce"]:
                raise DojoAllocationExecutionError(
                    "execution nonce was already consumed within this run"
                )
            if payload.get("selected_intent_id") == intent["intent_id"]:
                raise DojoAllocationExecutionError(
                    "selected intent was already claimed within this run"
                )

        claim = {
            "prepared_submission_sha256": prepared_sha,
            "allocation_sha256": allocation["allocation_sha256"],
            "selected_intent_sha256": intent_sha,
            "sealed_config_sha256": sealed_config["config_sha256"],
            "execution_nonce": rebuilt["execution_nonce"],
            "run_id": rebuilt["run_id"],
            "selected_intent_id": intent["intent_id"],
            "owner_id": self.owner_id,
            "broker_identity_kind": identity_kind,
            "virtual_broker_identity": expected_broker_identity,
            "submission_core_sha256": core_sha,
            "pending_fill_settings": (
                bound_fill_settings if identity_kind == "ORDER_ID" else None
            ),
            "external_broker_authority": "NONE",
            "live_permission": False,
        }
        # Claim before any close/order mutation. A crash or ordinary exception
        # consumes the prepared signal and therefore fails closed on retry.
        preclaim_tip = self._broker._prev_sha
        preclaim_size = self._broker.ledger_path.stat().st_size
        try:
            self._broker._log(_CLAIM_EVENT, claim)
        except Exception as exc:
            try:
                _truncate_ledger_to(
                    self._broker,
                    size_bytes=preclaim_size,
                    terminal_sha256=preclaim_tip,
                )
            except Exception as rollback_exc:
                raise DojoAllocationExecutionError(
                    "virtual claim logging and ledger rollback both failed"
                ) from rollback_exc
            raise DojoAllocationExecutionError(
                "virtual execution claim could not be made durable"
            ) from exc

        binding_ledger_sha = self._broker._prev_sha
        claimed_state = self._broker.snapshot()
        claimed_ledger_size = self._broker.ledger_path.stat().st_size
        if identity_kind == "ORDER_ID":
            _pending_fill_settings(self._broker)[expected_broker_identity] = dict(
                bound_fill_settings
            )
        try:
            if intent["order_kind"] == "MARKET":
                broker_identity = self._owner_view.market_order(
                    intent["pair"],
                    intent["side"],
                    intent["units"],
                    tp_pips=tp_pips,
                    sl_pips=sl_pips,
                )
            elif intent["order_kind"] == "LIMIT":
                broker_identity = self._owner_view.limit_order(
                    intent["pair"],
                    intent["side"],
                    intent["units"],
                    intent["entry_price"],
                    tp_pips=tp_pips,
                    sl_pips=sl_pips,
                )
            elif intent["order_kind"] == "STOP":
                broker_identity = self._owner_view.stop_order(
                    intent["pair"],
                    intent["side"],
                    intent["units"],
                    intent["entry_price"],
                    tp_pips=tp_pips,
                    sl_pips=sl_pips,
                )
            else:  # guarded by the allocator and prepared receipt reconstruction
                raise DojoAllocationExecutionError("unsupported order kind")
            if broker_identity != expected_broker_identity:
                raise DojoAllocationExecutionError(
                    "virtual broker identity diverged from the durable pre-claim"
                )
        except Exception as exc:
            try:
                _truncate_ledger_to(
                    self._broker,
                    size_bytes=claimed_ledger_size,
                    terminal_sha256=binding_ledger_sha,
                )
                self._broker.restore(claimed_state)
                self._broker._seq = max(
                    self._broker._seq, int(expected_broker_identity[1:])
                )
                strategy_ownership_registry(self._broker).reload_durable_state()
            except Exception as rollback_exc:
                raise DojoAllocationExecutionError(
                    "selected virtual submission failed and rollback was incomplete"
                ) from rollback_exc
            try:
                self._broker._log(
                    _FAIL_EVENT,
                    {
                        "prepared_submission_sha256": prepared_sha,
                        "allocation_sha256": allocation["allocation_sha256"],
                        "selected_intent_sha256": intent_sha,
                        "sealed_config_sha256": sealed_config["config_sha256"],
                        "execution_nonce": rebuilt["execution_nonce"],
                        "run_id": rebuilt["run_id"],
                        "selected_intent_id": intent["intent_id"],
                        "owner_id": self.owner_id,
                        "failure_type": type(exc).__name__,
                        "claim_consumed": True,
                        "virtual_state_rolled_back": True,
                        "incumbent_release_attempted": False,
                        "external_broker_authority": "NONE",
                        "live_permission": False,
                    },
                )
            except Exception as audit_exc:
                raise DojoAllocationExecutionError(
                    "selected virtual submission failed; state was rolled back but "
                    "failure audit logging also failed"
                ) from audit_exc
            raise DojoAllocationExecutionError(
                "selected virtual broker submission failed after claim; "
                "state was rolled back and the one-shot claim remains consumed"
            ) from exc

        body = {
            **core,
            "submission_core_sha256": core_sha,
            "binding_ledger_sha256": binding_ledger_sha,
        }
        return {**body, "submission_sha256": _canonical_sha(body)}

    def verify(self, submission: Mapping[str, Any]) -> dict[str, Any]:
        """Verify actual pending/fill ownership and current owner occupancy."""

        with self._broker_lock:
            return self._verify_locked(submission)

    def _verify_locked(self, submission: Mapping[str, Any]) -> dict[str, Any]:
        """Read ledger and owner occupancy under one virtual-state lock."""

        row = _strict_mapping(
            submission,
            field="submission",
            expected_keys={
                "contract",
                "schema_version",
                "prepared_submission_sha256",
                "allocation_sha256",
                "selected_intent_id",
                "selected_intent_sha256",
                "sealed_config_sha256",
                "execution_nonce",
                "run_id",
                "execution_epoch",
                "owner_id",
                "pair",
                "side",
                "order_kind",
                "units",
                "entry_price",
                "tp_distance_pips",
                "sl_distance_pips",
                "selected_quote_bid",
                "selected_quote_ask",
                "selected_quote_timestamp",
                "selected_quote_epoch",
                "selected_quote_sequence",
                "expected_execution_leverage",
                "expected_execution_slippage_pips",
                "expected_execution_financing_pips_per_day",
                "expected_execution_fast_ledger",
                "broker_identity_kind",
                "virtual_broker_identity",
                "fill_status_at_submission",
                "release_receipt",
                "risk_enforcement_model",
                "allocation_time_aggregate_sl_cap_enforced",
                "fill_time_aggregate_sl_cap_rechecked",
                "fill_time_total_loss_rechecked",
                "fill_time_rechecks",
                "order_authority",
                "external_broker_authority",
                "live_permission",
                "submission_core_sha256",
                "binding_ledger_sha256",
                "submission_sha256",
            },
        )
        body = {key: value for key, value in row.items() if key != "submission_sha256"}
        core = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "submission_core_sha256",
                "binding_ledger_sha256",
                "submission_sha256",
            }
        }
        receipt_quote_epoch = _quote_epoch(row["selected_quote_timestamp"])
        if (
            row["contract"] != SUBMISSION_CONTRACT
            or row["schema_version"] != 1
            or isinstance(row["schema_version"], bool)
            or row["owner_id"] != self.owner_id
            or row["risk_enforcement_model"] != _RISK_MODEL
            or row["allocation_time_aggregate_sl_cap_enforced"] is not True
            or row["fill_time_aggregate_sl_cap_rechecked"] is not False
            or row["fill_time_total_loss_rechecked"] is not False
            or row["order_authority"] != "DOJO_VIRTUAL_BROKER_ONLY"
            or row["external_broker_authority"] != "NONE"
            or row["live_permission"] is not False
            or row["execution_epoch"] != row["selected_quote_epoch"]
            or receipt_quote_epoch != row["selected_quote_epoch"]
            or not _optional_number_matches(
                row["entry_price"],
                (
                    row["selected_quote_ask"]
                    if row["order_kind"] == "MARKET" and row["side"] == "LONG"
                    else row["selected_quote_bid"]
                    if row["order_kind"] == "MARKET"
                    else row["entry_price"]
                ),
            )
            or row["submission_core_sha256"] != _canonical_sha(core)
            or row["submission_sha256"] != _canonical_sha(body)
        ):
            raise DojoAllocationExecutionError("submission receipt is invalid")

        claim_matches = _matching_ledger_records(
            self._broker,
            _CLAIM_EVENT,
            prepared_submission_sha256=row["prepared_submission_sha256"],
            allocation_sha256=row["allocation_sha256"],
            selected_intent_sha256=row["selected_intent_sha256"],
            sealed_config_sha256=row["sealed_config_sha256"],
            execution_nonce=row["execution_nonce"],
            run_id=row["run_id"],
            selected_intent_id=row["selected_intent_id"],
            owner_id=self.owner_id,
            broker_identity_kind=row["broker_identity_kind"],
            virtual_broker_identity=row["virtual_broker_identity"],
            submission_core_sha256=row["submission_core_sha256"],
            pending_fill_settings=(
                {
                    "leverage": row["expected_execution_leverage"],
                    "slippage_pips": row["expected_execution_slippage_pips"],
                    "financing_pips_per_day": row[
                        "expected_execution_financing_pips_per_day"
                    ],
                    "fast_ledger": row["expected_execution_fast_ledger"],
                }
                if row["broker_identity_kind"] == "ORDER_ID"
                else None
            ),
            external_broker_authority="NONE",
            live_permission=False,
        )
        if (
            len(claim_matches) != 1
            or claim_matches[0].get("sha") != row["binding_ledger_sha256"]
        ):
            raise DojoAllocationExecutionError(
                "submission receipt is not bound to its claim and broker identity"
            )

        identity = str(row["virtual_broker_identity"])
        ownership = strategy_ownership_registry(self._broker)
        ledger = _read_ledger(
            self._broker.ledger_path,
            expected_terminal_sha256=self._broker._prev_sha,
        )
        trade_id: str | None = None
        rejection_event: str | None = None
        expected_filled_entry: float | None = None
        expected_filled_tp: float | None = None
        expected_filled_sl: float | None = None
        if row["broker_identity_kind"] == "TRADE_ID":
            trade_id = identity
            fills = [
                record
                for record in ledger
                if record.get("event") == "FILL_MARKET"
                and isinstance(record.get("payload"), Mapping)
                and record["payload"].get("trade_id") == identity
            ]
            if len(fills) != 1:
                raise DojoAllocationExecutionError(
                    "market submission has no unique fill evidence"
                )
            fill_payload = fills[0]["payload"]
            pip = 0.01 if str(row["pair"]).endswith("JPY") else 0.0001
            precision = 3 if str(row["pair"]).endswith("JPY") else 5
            expected_entry = round(
                float(row["entry_price"])
                + (
                    float(row["expected_execution_slippage_pips"]) * pip
                    if row["side"] == "LONG"
                    else -float(row["expected_execution_slippage_pips"]) * pip
                ),
                precision,
            )
            expected_filled_entry = expected_entry
            expected_tp = (
                round(
                    expected_entry
                    + (
                        float(row["tp_distance_pips"]) * pip
                        if row["side"] == "LONG"
                        else -float(row["tp_distance_pips"]) * pip
                    ),
                    precision,
                )
                if row["tp_distance_pips"] is not None
                else None
            )
            expected_filled_tp = expected_tp
            expected_sl = (
                round(
                    expected_entry
                    + (
                        -float(row["sl_distance_pips"]) * pip
                        if row["side"] == "LONG"
                        else float(row["sl_distance_pips"]) * pip
                    ),
                    precision,
                )
                if row["sl_distance_pips"] is not None
                else None
            )
            expected_filled_sl = expected_sl
            if (
                fill_payload.get("strategy_owner_id") != self.owner_id
                or fill_payload.get("pair") != row["pair"]
                or fill_payload.get("side") != row["side"]
                or not _optional_number_matches(fill_payload.get("units"), row["units"])
                or not _optional_number_matches(
                    fill_payload.get("entry"), expected_entry
                )
                or not _optional_number_matches(
                    fill_payload.get("slippage_pips"),
                    row["expected_execution_slippage_pips"],
                )
                or fill_payload.get("tp") != expected_tp
                or fill_payload.get("sl") != expected_sl
            ):
                raise DojoAllocationExecutionError(
                    "market fill does not match the bound selected intent"
                )
        elif row["broker_identity_kind"] == "ORDER_ID":
            placement_event = (
                "ORDER_LIMIT" if row["order_kind"] == "LIMIT" else "ORDER_STOP"
            )
            placements = [
                record
                for record in ledger
                if record.get("event") == placement_event
                and isinstance(record.get("payload"), Mapping)
                and record["payload"].get("order_id") == identity
            ]
            if len(placements) != 1:
                raise DojoAllocationExecutionError(
                    "pending submission has no unique placement evidence"
                )
            placement = placements[0]["payload"]
            if (
                placement.get("strategy_owner_id") != self.owner_id
                or placement.get("pair") != row["pair"]
                or placement.get("side") != row["side"]
                or not _optional_number_matches(placement.get("units"), row["units"])
                or not _optional_number_matches(
                    placement.get("price"), row["entry_price"]
                )
                or not _optional_number_matches(
                    placement.get("tp_pips"), row["tp_distance_pips"]
                )
                or not _optional_number_matches(
                    placement.get("sl_pips"), row["sl_distance_pips"]
                )
            ):
                raise DojoAllocationExecutionError(
                    "pending placement does not match the bound selected intent"
                )
            for record in ledger:
                payload = record.get("payload")
                if (
                    not isinstance(payload, Mapping)
                    or payload.get("order_id") != identity
                ):
                    continue
                if record.get("event") == "FILL_LIMIT":
                    possible_trade = payload.get("trade_id")
                    if isinstance(possible_trade, str) and possible_trade:
                        quote = payload.get("quote")
                        if not isinstance(quote, Mapping):
                            raise DojoAllocationExecutionError(
                                "pending fill quote evidence is missing"
                            )
                        try:
                            trigger = float(row["entry_price"])
                            bid = float(quote["bid"])
                            ask = float(quote["ask"])
                            expected_slippage = float(
                                row["expected_execution_slippage_pips"]
                            )
                        except (KeyError, TypeError, ValueError) as exc:
                            raise DojoAllocationExecutionError(
                                "pending fill quote evidence is invalid"
                            ) from exc
                        pip = 0.01 if str(row["pair"]).endswith("JPY") else 0.0001
                        precision = 3 if str(row["pair"]).endswith("JPY") else 5
                        if row["order_kind"] == "LIMIT":
                            trigger_touched = (
                                ask <= trigger
                                if row["side"] == "LONG"
                                else bid >= trigger
                            )
                            base_fill = (
                                min(trigger, ask)
                                if row["side"] == "LONG"
                                else max(trigger, bid)
                            )
                        else:
                            trigger_touched = (
                                ask >= trigger
                                if row["side"] == "LONG"
                                else bid <= trigger
                            )
                            base_fill = (
                                max(trigger, ask)
                                if row["side"] == "LONG"
                                else min(trigger, bid)
                            )
                        if not trigger_touched:
                            raise DojoAllocationExecutionError(
                                "pending fill quote did not touch its bound trigger"
                            )
                        expected_filled_entry = base_fill
                        if expected_slippage > 0:
                            stressed = round(
                                base_fill
                                + (
                                    expected_slippage * pip
                                    if row["side"] == "LONG"
                                    else -expected_slippage * pip
                                ),
                                precision,
                            )
                            if row["order_kind"] == "LIMIT":
                                expected_filled_entry = (
                                    min(trigger, stressed)
                                    if row["side"] == "LONG"
                                    else max(trigger, stressed)
                                )
                            else:
                                expected_filled_entry = stressed
                        expected_filled_tp = (
                            round(
                                expected_filled_entry
                                + (
                                    float(row["tp_distance_pips"]) * pip
                                    if row["side"] == "LONG"
                                    else -float(row["tp_distance_pips"]) * pip
                                ),
                                precision,
                            )
                            if row["tp_distance_pips"] is not None
                            else None
                        )
                        expected_filled_sl = (
                            round(
                                expected_filled_entry
                                + (
                                    -float(row["sl_distance_pips"]) * pip
                                    if row["side"] == "LONG"
                                    else float(row["sl_distance_pips"]) * pip
                                ),
                                precision,
                            )
                            if row["sl_distance_pips"] is not None
                            else None
                        )
                        if (
                            payload.get("strategy_owner_id") != self.owner_id
                            or payload.get("pair") != row["pair"]
                            or payload.get("side") != row["side"]
                            or not _optional_number_matches(
                                payload.get("units"), row["units"]
                            )
                            or not _optional_number_matches(
                                payload.get("price"), expected_filled_entry
                            )
                            or not _optional_number_matches(
                                payload.get("slippage_pips"), expected_slippage
                            )
                            or payload.get("price_protection")
                            is not (row["order_kind"] == "LIMIT")
                        ):
                            raise DojoAllocationExecutionError(
                                "pending fill does not match the bound selected intent"
                            )
                        trade_id = possible_trade
                elif record.get("event") in {
                    "ORDER_CANCEL_CONCURRENCY_CAP",
                    "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
                    "ORDER_CANCEL",
                }:
                    rejection_event = str(record.get("event"))
        else:
            raise DojoAllocationExecutionError("submission broker identity is invalid")

        if identity in self._broker.orders:
            status = "PENDING"
            owner_verified = ownership.historical_order_owner(identity) == self.owner_id
        elif trade_id is not None:
            status = "FILLED" if trade_id in self._broker.positions else "RESOLVED"
            owner_verified = ownership.historical_trade_owner(trade_id) == self.owner_id
        elif rejection_event is not None:
            status = "REJECTED_OR_CANCELED_AT_FILL"
            owner_verified = ownership.historical_order_owner(identity) == self.owner_id
        else:
            raise DojoAllocationExecutionError(
                "virtual broker identity has no pending, fill, or rejection evidence"
            )
        if not owner_verified:
            raise DojoAllocationExecutionError(
                "virtual broker owner evidence mismatches"
            )

        active_trade_ids = self._owner_view.active_trade_ids()
        active_pair_ids = self._owner_view.active_trade_ids(pair=str(row["pair"]))
        within_caps = (
            len(active_pair_ids) <= self.max_concurrent_per_pair
            and len(active_trade_ids) <= self.global_max_concurrent
        )
        if not within_caps:
            raise DojoAllocationExecutionError("owner concurrency cap was breached")
        active_position = (
            self._owner_view.position(trade_id) if trade_id is not None else None
        )
        if active_position is not None:
            if (
                active_position.pair != row["pair"]
                or active_position.side != row["side"]
                or active_position.units <= 0
                or active_position.units > float(row["units"]) + 1e-9
                or not _optional_number_matches(
                    active_position.entry_price, expected_filled_entry
                )
                or not _optional_number_matches(
                    active_position.tp_price, expected_filled_tp
                )
                or not _optional_number_matches(
                    active_position.sl_price, expected_filled_sl
                )
            ):
                raise DojoAllocationExecutionError(
                    "filled virtual position does not match selected intent"
                )

        selected_trade_remaining_units = (
            active_position.units if active_position is not None else None
        )
        selected_trade_partially_closed = (
            active_position is not None
            and active_position.units < float(row["units"]) - 1e-9
        )

        verification_body = {
            "contract": EXECUTION_VERIFICATION_CONTRACT,
            "schema_version": 1,
            "submission_sha256": row["submission_sha256"],
            "prepared_submission_sha256": row["prepared_submission_sha256"],
            "allocation_sha256": row["allocation_sha256"],
            "selected_intent_sha256": row["selected_intent_sha256"],
            "sealed_config_sha256": row["sealed_config_sha256"],
            "owner_id": self.owner_id,
            "pair": row["pair"],
            "status": status,
            "rejection_event": rejection_event,
            "order_id": identity if row["broker_identity_kind"] == "ORDER_ID" else None,
            "trade_id": trade_id,
            "owner_verified": True,
            "active_owner_trade_ids": list(active_trade_ids),
            "active_owner_pair_trade_ids": list(active_pair_ids),
            "active_owner_global_count": len(active_trade_ids),
            "active_owner_pair_count": len(active_pair_ids),
            "max_concurrent_per_pair": self.max_concurrent_per_pair,
            "global_max_concurrent": self.global_max_concurrent,
            "owner_concurrency_within_caps": True,
            "selected_trade_remaining_units": selected_trade_remaining_units,
            "selected_trade_partially_closed": selected_trade_partially_closed,
            "partial_close_preserves_occupancy": (
                selected_trade_partially_closed and trade_id in active_trade_ids
            ),
            "risk_enforcement_model": _RISK_MODEL,
            "allocation_time_aggregate_sl_cap_enforced": True,
            "fill_time_aggregate_sl_cap_rechecked": False,
            "fill_time_total_loss_rechecked": False,
            "external_broker_authority": "NONE",
            "live_permission": False,
        }
        return {
            **verification_body,
            "verification_sha256": _canonical_sha(verification_body),
        }


__all__ = [
    "DojoAllocationExecutionError",
    "DojoAllocationExecutionSession",
    "EXECUTION_CONFIG_CONTRACT",
    "EXECUTION_VERIFICATION_CONTRACT",
    "PREPARED_SUBMISSION_CONTRACT",
    "SUBMISSION_CONTRACT",
]

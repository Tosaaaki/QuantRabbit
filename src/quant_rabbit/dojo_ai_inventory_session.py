"""Fixed-window owner for a future isolated paper-AI inventory room.

This module is deliberately dormant.  It does not register a production
dependency bundle and is not imported by any existing C/D/E/G paper room.
The command-line runner can become usable only after a reviewed, code-owned
dependency registration is added *and* the canonical replay lifecycle has
issued a valid ``PAPER_ELIGIBLE`` launch token.

The owner coordinates already-separated capabilities without owning a live
broker surface:

* one local :class:`~quant_rabbit.virtual_broker.VirtualBroker` service;
* one bot process that has entry-only capability;
* signed, cutoff-bound quote/candle capture;
* one AI inventory controller cycle; and
* prospective outcome evaluation.

All unstable process details are dependency injected.  Caller-supplied JSON
can never provide a command, callback, credential, socket, ledger path, or
environment value.  Existing rooms cannot opt into this module.
"""

from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_replay_lifecycle import canonical_paper_ai_rooms_root


SESSION_CONFIG_CONTRACT = "QR_DOJO_AI_INVENTORY_SESSION_CONFIG_V1"
SESSION_CONTRACT = "QR_DOJO_AI_INVENTORY_SESSION_CONTRACT_V1"
SESSION_OWNER_CONTRACT = "QR_DOJO_AI_INVENTORY_SESSION_OWNER_V1"
SESSION_LIFECYCLE_CONTRACT = "QR_DOJO_AI_INVENTORY_SESSION_LIFECYCLE_V1"
SESSION_STATE_CONTRACT = "QR_DOJO_AI_INVENTORY_SESSION_STATE_V1"
DRAIN_QUOTE_RECEIPT_CONTRACT = "QR_DOJO_AI_DRAIN_QUOTE_RECEIPT_V1"
DRAIN_BROKER_RESTART_RECEIPT_CONTRACT = (
    "QR_DOJO_AI_DRAIN_BROKER_RESTART_RECEIPT_V1"
)
DRAIN_BROKER_START_RECEIPT_CONTRACT = (
    "QR_DOJO_AI_DRAIN_BROKER_START_RECEIPT_V1"
)
GENESIS_LIFECYCLE_SHA256 = "0" * 64

SESSION_CONFIG_DIRECTORY = Path("config/paper_ai_inventory")
SESSION_CONTRACT_NAME = "session_contract.json"
SESSION_OWNER_NAME = "session_owner.json"
SESSION_OWNER_LOCK_NAME = ".session-owner.lock"
SESSION_LIFECYCLE_NAME = "session_lifecycle.jsonl"
SESSION_STATE_NAME = "session_state.json"
DRAIN_BROKER_RESTART_DIRECTORY_NAME = "drain_broker_restarts"
DRAIN_BROKER_RESTART_LOCK_NAME = ".drain-broker-restart.lock"

ALLOWED_DRAIN_RESOLUTIONS = frozenset(
    {
        "TP",
        "SL",
        "MARGIN_CLOSEOUT",
        "ORIGINAL_CEILING",
        "PENDING_ORDER_CANCEL_AT_ENTRY_STOP",
    }
)
DRAIN_BROKER_RUNNER_COMMANDS = (
    "ACCOUNT",
    "APPLY_DRAIN_QUOTE",
    "HEALTH",
    "ORDERS",
    "POSITIONS",
    "QUOTE_PROVENANCE",
    "QUOTES",
    "SHUTDOWN",
)
SOURCE_ROLES = ("quote", "candles")
MAX_CONFIG_BYTES = 256 * 1024
MAX_LEDGER_BYTES = 16 * 1024 * 1024
MAX_LEDGER_ROW_BYTES = 256 * 1024
MAX_QUOTE_RECOVERY_EVENTS = 10_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,254}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_UTC_RE = re.compile(
    r"^(?P<seconds>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,6}))?(?:Z|\+00:00)$"
)

_CONFIG_KEYS = frozenset(
    {
        "contract",
        "experiment_id",
        "room_id",
        "candidate_id",
        "dependency_id",
        "pair",
        "window_start_utc",
        "window_end_utc",
        "adapter_id",
        "model_id",
        "model_config_sha256",
        "producer_id",
        "bot_config_sha256",
        "balance_jpy",
        "slippage_pips",
        "financing_pips_per_day",
        "leverage",
        "original_ceiling_minutes",
        "cycle_interval_seconds",
        "drain_interval_seconds",
        "capture_deadline_seconds",
        "evaluation_horizon_seconds",
        "launch_preflight_token_sha256",
        "paper_only",
        "order_authority",
        "live_permission",
        "external_broker_mutation_allowed",
        "session_config_sha256",
    }
)
_SAFETY = {
    "paper_only": True,
    "order_authority": "NONE",
    "live_permission": False,
    "external_broker_mutation_allowed": False,
}


class AIInventorySessionError(RuntimeError):
    """The isolated session failed closed."""


class AIInventorySessionConfigError(AIInventorySessionError):
    """A session configuration is not canonical or safe."""


class AIInventorySessionIntegrityError(AIInventorySessionError):
    """Durable state or a dependency receipt is inconsistent."""


class AIInventorySessionBusyError(AIInventorySessionError):
    """Another process owns the one permitted room session."""


class AIInventorySessionUnavailableError(AIInventorySessionError):
    """No reviewed production dependency bundle is registered."""


@dataclass(frozen=True)
class AIInventorySessionConfig:
    """Strict data-only configuration for one future paper-AI room."""

    repository_root: Path
    experiment_id: str
    room_id: str
    candidate_id: str
    dependency_id: str
    pair: str
    window_start_utc: str
    window_end_utc: str
    adapter_id: str
    model_id: str
    model_config_sha256: str
    producer_id: str
    bot_config_sha256: str
    balance_jpy: float
    slippage_pips: float
    financing_pips_per_day: float
    leverage: float
    original_ceiling_minutes: int
    cycle_interval_seconds: int
    drain_interval_seconds: int
    capture_deadline_seconds: int
    evaluation_horizon_seconds: int
    launch_preflight_token_sha256: str
    session_config_sha256: str

    @property
    def window_start(self) -> datetime:
        return _parse_utc(self.window_start_utc, "window_start_utc")

    @property
    def window_end(self) -> datetime:
        return _parse_utc(self.window_end_utc, "window_end_utc")


@dataclass(frozen=True)
class AIInventorySessionContext:
    """Immutable bindings supplied to every code-owned dependency hook."""

    config: AIInventorySessionConfig
    room_root: Path
    launch_preflight: Mapping[str, Any]
    session_contract: Mapping[str, Any]


@dataclass(frozen=True)
class AIInventorySessionDependencies:
    """Code-owned orchestration hooks.

    Hooks may retain process handles or credentials in their owning closure;
    neither is accepted from the session configuration or persisted here.
    """

    dependency_id: str
    clock: Callable[[], datetime]
    sleep: Callable[[float], None]
    verify_launch_preflight: Callable[[Path, str, str], Mapping[str, Any]]
    start_broker: Callable[[AIInventorySessionContext], Mapping[str, Any]]
    inspect_drain_checkpoint: Callable[
        [AIInventorySessionContext], Mapping[str, Any]
    ]
    start_drain_broker: Callable[
        [AIInventorySessionContext, Path], Mapping[str, Any]
    ]
    stop_broker: Callable[[AIInventorySessionContext], None]
    broker_health: Callable[[AIInventorySessionContext], Mapping[str, Any]]
    start_bot: Callable[[AIInventorySessionContext], Mapping[str, Any]]
    stop_bot: Callable[[AIInventorySessionContext], Mapping[str, Any]]
    bot_health: Callable[[AIInventorySessionContext], Mapping[str, Any]]
    capture_source: Callable[[AIInventorySessionContext, str, str], Mapping[str, Any]]
    apply_captured_quote: Callable[
        [AIInventorySessionContext, str], Sequence[Mapping[str, Any]]
    ]
    # Drain evidence is a distinct capability.  The active-window capture
    # contract correctly rejects cutoff timestamps at/after window_end and
    # therefore must never be weakened or reused for post-window draining.
    capture_drain_quote: Callable[[AIInventorySessionContext, str], Mapping[str, Any]]
    apply_captured_drain_quote: Callable[
        [AIInventorySessionContext, str], Sequence[Mapping[str, Any]]
    ]
    inspect_broker: Callable[[AIInventorySessionContext], Mapping[str, Any]]
    build_evidence_request: Callable[
        [
            AIInventorySessionContext,
            Mapping[str, Mapping[str, Any]],
            Mapping[str, Any],
            str,
        ],
        Mapping[str, Any],
    ]
    run_controller: Callable[
        [AIInventorySessionContext, Mapping[str, Any]], Mapping[str, Any]
    ]
    evaluation_plan: Callable[[AIInventorySessionContext, str], Mapping[str, Any]]
    evaluate: Callable[
        [AIInventorySessionContext, Mapping[str, Any]], Mapping[str, Any]
    ]
    drain_tick: Callable[[AIInventorySessionContext, str, str], Mapping[str, Any]]


@dataclass(frozen=True)
class AIInventorySessionResult:
    """Final or bounded-test result returned by the session owner."""

    status: str
    room_root: Path
    lifecycle_tip_sha256: str
    positions_count: int
    orders_count: int
    pending_evaluations: int


# A future production implementation must add an explicit immutable
# registration in code review.  Empty-by-default is an intentional fail-closed
# launch gate, not a TODO that may be bypassed from JSON or environment.
_PRODUCTION_SESSION_DEPENDENCIES: Mapping[str, AIInventorySessionDependencies] = (
    MappingProxyType({})
)


def session_config_sha256(value: Mapping[str, Any]) -> str:
    """Return the semantic digest of a session config mapping."""

    body = dict(value)
    body.pop("session_config_sha256", None)
    for field in (
        "balance_jpy",
        "slippage_pips",
        "financing_pips_per_day",
        "leverage",
    ):
        raw = body.get(field)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            body[field] = float(raw)
    return _canonical_sha256(body)


def load_ai_inventory_session_config(
    repository_root: Path | str, config_path: Path | str
) -> AIInventorySessionConfig:
    """Load one canonical config from the repository-owned config directory."""

    root = _repository_root(repository_root)
    allowed_root = (root / SESSION_CONFIG_DIRECTORY).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise AIInventorySessionConfigError(
            "session config directory is unavailable"
        ) from exc
    if resolved_parent != allowed_root or path.suffix != ".json":
        raise AIInventorySessionConfigError(
            "session config must be a direct JSON child of config/paper_ai_inventory"
        )
    raw = _read_regular_file(path, MAX_CONFIG_BYTES, "session config")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AIInventorySessionConfigError("session config is invalid JSON") from exc
    if not isinstance(value, dict):
        raise AIInventorySessionConfigError("session config must be an object")
    if raw != _canonical_bytes(value) + b"\n":
        raise AIInventorySessionConfigError("session config is not canonical JSON")
    return ai_inventory_session_config_from_mapping(root, value)


def ai_inventory_session_config_from_mapping(
    repository_root: Path | str, value: Mapping[str, Any]
) -> AIInventorySessionConfig:
    """Validate a data-only session configuration."""

    root = _repository_root(repository_root)
    item = _snapshot_mapping(value, "session config")
    if set(item) != _CONFIG_KEYS:
        missing = sorted(_CONFIG_KEYS - set(item))
        extra = sorted(set(item) - _CONFIG_KEYS)
        raise AIInventorySessionConfigError(
            f"session config keys mismatch; missing={missing}, extra={extra}"
        )
    if item.get("contract") != SESSION_CONFIG_CONTRACT:
        raise AIInventorySessionConfigError("session config contract is invalid")
    _require_safety(item, "session config")
    for field in ("experiment_id", "room_id"):
        identifier = _identifier(item.get(field), field)
        if not identifier.startswith("paper-ai-inventory-"):
            raise AIInventorySessionConfigError(
                f"{field} is not an isolated paper-ai-inventory id"
            )
    for field in ("dependency_id", "adapter_id", "model_id", "producer_id"):
        _identifier(item.get(field), field)
    candidate_id = _sha(item.get("candidate_id"), "candidate_id")
    model_config_sha256 = _sha(item.get("model_config_sha256"), "model_config_sha256")
    bot_config_sha256 = _sha(item.get("bot_config_sha256"), "bot_config_sha256")
    preflight_sha256 = _sha(
        item.get("launch_preflight_token_sha256"),
        "launch_preflight_token_sha256",
    )
    pair = item.get("pair")
    if not isinstance(pair, str) or _PAIR_RE.fullmatch(pair) is None:
        raise AIInventorySessionConfigError("pair is invalid")
    start = _parse_utc(item.get("window_start_utc"), "window_start_utc")
    end = _parse_utc(item.get("window_end_utc"), "window_end_utc")
    if end <= start:
        raise AIInventorySessionConfigError("fixed window must end after it starts")
    numbers = {
        "balance_jpy": _positive_finite(item.get("balance_jpy"), "balance_jpy"),
        "slippage_pips": _nonnegative_finite(
            item.get("slippage_pips"), "slippage_pips"
        ),
        "financing_pips_per_day": _nonnegative_finite(
            item.get("financing_pips_per_day"), "financing_pips_per_day"
        ),
        "leverage": _positive_finite(item.get("leverage"), "leverage"),
    }
    integers = {
        "original_ceiling_minutes": _bounded_int(
            item.get("original_ceiling_minutes"),
            "original_ceiling_minutes",
            1,
            1_440,
        ),
        "cycle_interval_seconds": _bounded_int(
            item.get("cycle_interval_seconds"),
            "cycle_interval_seconds",
            1,
            3_600,
        ),
        "drain_interval_seconds": _bounded_int(
            item.get("drain_interval_seconds"),
            "drain_interval_seconds",
            1,
            3_600,
        ),
        "capture_deadline_seconds": _bounded_int(
            item.get("capture_deadline_seconds"),
            "capture_deadline_seconds",
            1,
            90,
        ),
        "evaluation_horizon_seconds": _bounded_int(
            item.get("evaluation_horizon_seconds"),
            "evaluation_horizon_seconds",
            60,
            7 * 24 * 3_600,
        ),
    }
    claimed = _sha(item.get("session_config_sha256"), "session_config_sha256")
    if claimed != session_config_sha256(item):
        raise AIInventorySessionConfigError("session config digest mismatch")
    return AIInventorySessionConfig(
        repository_root=root,
        experiment_id=str(item["experiment_id"]),
        room_id=str(item["room_id"]),
        candidate_id=candidate_id,
        dependency_id=str(item["dependency_id"]),
        pair=pair,
        window_start_utc=str(item["window_start_utc"]),
        window_end_utc=str(item["window_end_utc"]),
        adapter_id=str(item["adapter_id"]),
        model_id=str(item["model_id"]),
        model_config_sha256=model_config_sha256,
        producer_id=str(item["producer_id"]),
        bot_config_sha256=bot_config_sha256,
        balance_jpy=numbers["balance_jpy"],
        slippage_pips=numbers["slippage_pips"],
        financing_pips_per_day=numbers["financing_pips_per_day"],
        leverage=numbers["leverage"],
        original_ceiling_minutes=integers["original_ceiling_minutes"],
        cycle_interval_seconds=integers["cycle_interval_seconds"],
        drain_interval_seconds=integers["drain_interval_seconds"],
        capture_deadline_seconds=integers["capture_deadline_seconds"],
        evaluation_horizon_seconds=integers["evaluation_horizon_seconds"],
        launch_preflight_token_sha256=preflight_sha256,
        session_config_sha256=claimed,
    )


def registered_ai_inventory_session_dependency_ids() -> tuple[str, ...]:
    """Return the reviewed production dependency ids (normally empty)."""

    return tuple(sorted(_PRODUCTION_SESSION_DEPENDENCIES))


def verify_drain_broker_restart_authorization(
    *,
    repository_root: Path,
    room_root: Path,
    authorization_path: Path,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
    broker_ledger_terminal_sha256: str,
    broker_snapshot_sha256: str,
    broker_snapshot_ledger_terminal_sha256: str,
    positions_count: int,
    orders_count: int,
    broker_recovery_wal_sha256: str | None,
    broker_recovery_wal_checkpoint_ledger_sha256: str | None,
    broker_recovery_wal_expected_event_count: int,
    broker_recovery_wal_applied_event_count: int,
    broker_recovery_wal_validated: bool,
    balance_jpy: float,
    slippage_pips: float,
    financing_pips_per_day: float,
    leverage: float,
    original_ceiling_minutes: int,
) -> dict[str, Any]:
    """Verify the latest append-only drain-only broker authorization.

    This is the broker startup boundary.  It independently checks the full
    authorization chain, exact latest path, current session contract/state and
    lifecycle tip, and the caller's already-validated local broker checkpoint.
    No network, model, evaluator, bot, or broker mutation is performed.
    """

    repository = _repository_root(repository_root)
    root = Path(room_root)
    try:
        canonical_root = root.resolve(strict=True)
    except OSError as exc:
        raise AIInventorySessionIntegrityError(
            "drain authorization room root is unavailable"
        ) from exc
    if canonical_root != root or not canonical_root.is_dir():
        raise AIInventorySessionIntegrityError(
            "drain authorization room root is not canonical"
        )
    experiment_id = _identifier(experiment_id, "experiment_id")
    room_id = _identifier(room_id, "room_id")
    candidate_id = _sha(candidate_id, "candidate_id")
    room_base = canonical_paper_ai_rooms_root(repository)
    expected_room_root = room_base / experiment_id / room_id
    try:
        canonical_room_base = room_base.resolve(strict=True)
        resolved_expected_room_root = expected_room_root.resolve(strict=True)
    except OSError as exc:
        raise AIInventorySessionIntegrityError(
            "canonical drain authorization room is unavailable"
        ) from exc
    if (
        resolved_expected_room_root != expected_room_root
        or resolved_expected_room_root.parent.parent != canonical_room_base
        or canonical_root != resolved_expected_room_root
    ):
        raise AIInventorySessionIntegrityError(
            "drain authorization room is outside the canonical repository root"
        )
    ledger_tip = _sha(
        broker_ledger_terminal_sha256,
        "broker_ledger_terminal_sha256",
    )
    snapshot_sha = _sha(broker_snapshot_sha256, "broker_snapshot_sha256")
    snapshot_tip = _sha(
        broker_snapshot_ledger_terminal_sha256,
        "broker_snapshot_ledger_terminal_sha256",
    )
    positions_count = _bounded_int(
        positions_count,
        "positions_count",
        0,
        1_000_000,
    )
    orders_count = _bounded_int(
        orders_count,
        "orders_count",
        0,
        1_000_000,
    )
    recovery = _validated_broker_recovery_fields(
        {
            "broker_recovery_wal_sha256": broker_recovery_wal_sha256,
            "broker_recovery_wal_checkpoint_ledger_sha256": (
                broker_recovery_wal_checkpoint_ledger_sha256
            ),
            "broker_recovery_wal_expected_event_count": (
                broker_recovery_wal_expected_event_count
            ),
            "broker_recovery_wal_applied_event_count": (
                broker_recovery_wal_applied_event_count
            ),
            "broker_recovery_wal_validated": broker_recovery_wal_validated,
        },
        ledger_tip=ledger_tip,
        snapshot_tip=snapshot_tip,
    )
    costs = {
        "balance_jpy": _positive_finite(balance_jpy, "balance_jpy"),
        "slippage_pips": _nonnegative_finite(
            slippage_pips,
            "slippage_pips",
        ),
        "financing_pips_per_day": _nonnegative_finite(
            financing_pips_per_day,
            "financing_pips_per_day",
        ),
        "leverage": _positive_finite(leverage, "leverage"),
        "original_ceiling_minutes": _bounded_int(
            original_ceiling_minutes,
            "original_ceiling_minutes",
            1,
            1_440,
        ),
    }
    directory = canonical_root / DRAIN_BROKER_RESTART_DIRECTORY_NAME
    try:
        canonical_authorization = Path(authorization_path).resolve(strict=True)
    except OSError as exc:
        raise AIInventorySessionIntegrityError(
            "drain broker authorization is unavailable"
        ) from exc
    if (
        canonical_authorization != Path(authorization_path)
        or canonical_authorization.parent != directory
    ):
        raise AIInventorySessionIntegrityError(
            "drain broker authorization path is not canonical"
        )
    with _exclusive_drain_restart_lock(directory):
        chain = _read_drain_broker_restart_chain(directory)
        if not chain:
            raise AIInventorySessionIntegrityError(
                "drain broker authorization chain is empty"
            )
        latest = chain[-1]
        expected_path = directory / (
            f"{latest['sequence']:08d}-{latest['receipt_sha256']}.json"
        )
        if canonical_authorization != expected_path:
            raise AIInventorySessionIntegrityError(
                "drain broker authorization is not the latest chain row"
            )
    contract_path = canonical_root / SESSION_CONTRACT_NAME
    state_path = canonical_root / SESSION_STATE_NAME
    lifecycle_path = canonical_root / SESSION_LIFECYCLE_NAME
    contract_raw = _read_regular_file(
        contract_path,
        MAX_CONFIG_BYTES,
        "session contract",
    )
    state_raw = _read_regular_file(
        state_path,
        MAX_CONFIG_BYTES,
        "session state",
    )
    contract = _read_json_object(
        contract_path,
        MAX_CONFIG_BYTES,
        "session contract",
    )
    state = _load_state(state_path)
    if state is None:
        raise AIInventorySessionIntegrityError("session state is unavailable")
    _require_safety(contract, "session contract")
    expected_contract = {
        "contract": SESSION_CONTRACT,
        "experiment_id": experiment_id,
        "room_id": room_id,
        "candidate_id": candidate_id,
        "active_source_capture_after_window_allowed": False,
        "drain_quote_required": True,
        "drain_quote_receipt_contract": DRAIN_QUOTE_RECEIPT_CONTRACT,
        **costs,
    }
    for field, expected_value in expected_contract.items():
        if contract.get(field) != expected_value:
            raise AIInventorySessionIntegrityError(
                f"drain authorization session contract {field} mismatch"
            )
    lifecycle_tip = _lifecycle_tip(lifecycle_path)
    expected_latest = {
        "experiment_id": experiment_id,
        "room_id": room_id,
        "candidate_id": candidate_id,
        "mode": "DRAIN_ONLY",
        "session_contract_file_sha256": hashlib.sha256(contract_raw).hexdigest(),
        "session_lifecycle_tip_sha256": lifecycle_tip,
        "session_state_file_sha256": hashlib.sha256(state_raw).hexdigest(),
        "broker_ledger_terminal_sha256": ledger_tip,
        "broker_snapshot_sha256": snapshot_sha,
        "broker_snapshot_ledger_terminal_sha256": snapshot_tip,
        "positions_count": positions_count,
        "orders_count": orders_count,
        **recovery,
        "new_entries_allowed": False,
        "ai_decision_allowed": False,
        "force_close_allowed": False,
        "bot_rpc_commands": [],
        "runner_rpc_commands": list(DRAIN_BROKER_RUNNER_COMMANDS),
    }
    for field, expected_value in expected_latest.items():
        if latest.get(field) != expected_value:
            raise AIInventorySessionIntegrityError(
                f"drain broker authorization {field} mismatch"
            )
    if (
        state.get("status") != "DRAINING"
        or state.get("lifecycle_tip_sha256") != lifecycle_tip
        or state.get("positions_count") != positions_count
        or state.get("orders_count") != orders_count
        or state.get("new_entries_allowed") is not False
    ):
        raise AIInventorySessionIntegrityError(
            "drain broker authorization session state mismatch"
        )
    return dict(latest)


def run_registered_ai_inventory_session(
    config: AIInventorySessionConfig,
    *,
    screen_identity: str | None,
    process_argv: Sequence[str],
) -> AIInventorySessionResult:
    """Run through the immutable production registry, failing closed if absent."""

    dependency = _PRODUCTION_SESSION_DEPENDENCIES.get(config.dependency_id)
    if dependency is None:
        raise AIInventorySessionUnavailableError(
            "no reviewed production AI inventory session dependency is registered"
        )
    return run_ai_inventory_session(
        config,
        dependency,
        screen_identity=screen_identity,
        process_argv=process_argv,
    )


def run_ai_inventory_session(
    config: AIInventorySessionConfig,
    dependencies: AIInventorySessionDependencies,
    *,
    screen_identity: str | None,
    process_argv: Sequence[str],
    max_iterations: int | None = None,
) -> AIInventorySessionResult:
    """Own one fixed-window paper-AI session.

    ``max_iterations`` exists solely for deterministic library tests.  The
    registered production runner never supplies it.
    """

    config = _validate_config_instance(config)
    dependencies = _validate_dependencies(dependencies, config.dependency_id)
    if max_iterations is not None and (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise AIInventorySessionConfigError("max_iterations must be positive")
    screen_name = _validate_screen_identity(screen_identity, config.room_id)
    argv = _validate_process_argv(process_argv)

    # Eligibility is deliberately revalidated before any session file or lock
    # is created.  Missing PAPER_ELIGIBLE evidence cannot leave a launch trace.
    preflight = _snapshot_mapping(
        dependencies.verify_launch_preflight(
            config.repository_root,
            experiment_id=config.experiment_id,
            room_id=config.room_id,
        ),
        "launch preflight",
    )
    _validate_preflight(config, preflight)
    room_root = _canonical_room_root(config)
    session_contract = _build_session_contract(
        config,
        preflight,
        screen_name=screen_name,
        process_argv=argv,
    )
    context = AIInventorySessionContext(
        config=config,
        room_root=room_root,
        launch_preflight=MappingProxyType(dict(preflight)),
        session_contract=MappingProxyType(dict(session_contract)),
    )

    broker_started = False
    bot_started = False
    final_summary = _empty_broker_summary()
    pending_evaluations = 0
    status = "WAITING"
    lifecycle_tip = GENESIS_LIFECYCLE_SHA256
    iteration = 0

    with _exclusive_owner_lock(room_root):
        _write_immutable_contract(room_root / SESSION_CONTRACT_NAME, session_contract)
        lifecycle_tip = _lifecycle_tip(room_root / SESSION_LIFECYCLE_NAME)
        persisted_state = _load_state(room_root / SESSION_STATE_NAME)
        if persisted_state is not None and persisted_state.get("status") == "SEALED":
            return _result_from_state(room_root, persisted_state)
        _write_owner_receipt(room_root, config, screen_name, argv, session_contract)

        if persisted_state is not None:
            status = str(persisted_state["status"])
            final_summary = {
                **_empty_broker_summary(),
                "positions_count": _bounded_int(
                    persisted_state.get("positions_count"),
                    "persisted positions_count",
                    0,
                    1_000_000,
                ),
                "orders_count": _bounded_int(
                    persisted_state.get("orders_count"),
                    "persisted orders_count",
                    0,
                    1_000_000,
                ),
                "new_entries_allowed": (
                    persisted_state.get("new_entries_allowed") is True
                ),
            }
            pending_evaluations = _bounded_int(
                persisted_state.get("pending_evaluations"),
                "persisted pending_evaluations",
                0,
                1_000_000,
            )
        event = (
            "SESSION_RESUME"
            if lifecycle_tip != GENESIS_LIFECYCLE_SHA256
            else "SESSION_START"
        )
        lifecycle_status = (
            "DRAINING"
            if event == "SESSION_RESUME"
            and status in {"DRAINING", "WEEKEND_DRAIN_PAUSED"}
            else status
        )
        lifecycle_tip = _append_lifecycle(
            room_root,
            event,
            {
                "status": lifecycle_status,
                "session_config_sha256": config.session_config_sha256,
                "launch_preflight_token_sha256": (config.launch_preflight_token_sha256),
            },
        )
        _write_state(
            room_root,
            status=status,
            lifecycle_tip=lifecycle_tip,
            summary=final_summary,
            pending_evaluations=pending_evaluations,
            market_open=False,
        )

        try:
            if persisted_state is not None and status in {
                "ACTIVE",
                "WEEKEND_PAUSED",
            }:
                # A replacement owner cannot prove that the old bot process
                # stopped at the same instant as its parent.  Revoke entry
                # capability and continue only toward the immutable window
                # end.  A bot is never blindly relaunched after owner loss.
                _require_entry_stop_receipt(
                    dependencies.stop_bot(context),
                    config,
                    "session-owner restart entry revocation",
                )
                dependencies.stop_broker(context)
                lifecycle_tip = _append_lifecycle(
                    room_root,
                    "ENTRY_STOP",
                    {
                        "status": "ENTRY_STOPPED_WAITING",
                        "reason": "SESSION_OWNER_RESTART_UNPROVEN_CHILDREN",
                        "window_end_utc": config.window_end_utc,
                        "new_entries_allowed": False,
                        "force_close": False,
                        "original_ceiling_minutes": (
                            config.original_ceiling_minutes
                        ),
                    },
                )
                status = "ENTRY_STOPPED_WAITING"
                _write_state(
                    room_root,
                    status=status,
                    lifecycle_tip=lifecycle_tip,
                    summary=final_summary,
                    pending_evaluations=pending_evaluations,
                    market_open=False,
                )
            while True:
                iteration += 1
                now = _utc_now(dependencies.clock())
                market_open = compute_market_status(now).is_fx_open

                if now < config.window_start:
                    status = "WAITING"
                    _write_state(
                        room_root,
                        status=status,
                        lifecycle_tip=lifecycle_tip,
                        summary=final_summary,
                        pending_evaluations=pending_evaluations,
                        market_open=market_open,
                    )
                    if _bounded_iteration_done(iteration, max_iterations):
                        break
                    dependencies.sleep(
                        _sleep_seconds(
                            now,
                            config.window_start,
                            config.cycle_interval_seconds,
                        )
                    )
                    continue

                if now < config.window_end:
                    if status == "ENTRY_STOPPED_WAITING":
                        _write_state(
                            room_root,
                            status=status,
                            lifecycle_tip=lifecycle_tip,
                            summary=final_summary,
                            pending_evaluations=pending_evaluations,
                            market_open=market_open,
                        )
                        if _bounded_iteration_done(iteration, max_iterations):
                            break
                        dependencies.sleep(
                            _sleep_seconds(
                                now,
                                config.window_end,
                                config.cycle_interval_seconds,
                            )
                        )
                        continue
                    status = "ACTIVE" if market_open else "WEEKEND_PAUSED"
                    if not market_open:
                        if bot_started:
                            _require_entry_stop_receipt(
                                dependencies.stop_bot(context),
                                config,
                                "weekend bot stop",
                            )
                            bot_started = False
                        _write_state(
                            room_root,
                            status=status,
                            lifecycle_tip=lifecycle_tip,
                            summary=final_summary,
                            pending_evaluations=pending_evaluations,
                            market_open=False,
                        )
                        if _bounded_iteration_done(iteration, max_iterations):
                            break
                        dependencies.sleep(config.cycle_interval_seconds)
                        continue

                    if broker_started:
                        broker_alive = _validate_process_health(
                            dependencies.broker_health(context),
                            config,
                            role="broker",
                            mode="ACTIVE",
                        )
                        if not broker_alive:
                            broker_started = False
                    if not broker_started:
                        _validate_broker_start_receipt(
                            dependencies.start_broker(context), config
                        )
                        broker_started = True
                    if bot_started:
                        bot_alive = _validate_process_health(
                            dependencies.bot_health(context),
                            config,
                            role="bot",
                            mode="ACTIVE",
                        )
                        if not bot_alive:
                            bot_started = False
                            _require_entry_stop_receipt(
                                dependencies.stop_bot(context),
                                config,
                                "dead bot entry revocation",
                            )
                            dependencies.stop_broker(context)
                            broker_started = False
                            lifecycle_tip = _append_lifecycle(
                                room_root,
                                "ENTRY_STOP",
                                {
                                    "status": "ENTRY_STOPPED_WAITING",
                                    "reason": "BOT_PROCESS_DIED",
                                    "window_end_utc": config.window_end_utc,
                                    "new_entries_allowed": False,
                                    "force_close": False,
                                    "original_ceiling_minutes": (
                                        config.original_ceiling_minutes
                                    ),
                                },
                            )
                            status = "ENTRY_STOPPED_WAITING"
                            _write_state(
                                room_root,
                                status=status,
                                lifecycle_tip=lifecycle_tip,
                                summary=final_summary,
                                pending_evaluations=pending_evaluations,
                                market_open=True,
                            )
                            if _bounded_iteration_done(
                                iteration, max_iterations
                            ):
                                break
                            dependencies.sleep(
                                _sleep_seconds(
                                    now,
                                    config.window_end,
                                    config.cycle_interval_seconds,
                                )
                            )
                            continue
                    if not bot_started:
                        _validate_bot_start_receipt(
                            dependencies.start_bot(context), config
                        )
                        bot_started = True

                    remaining = (config.window_end - now).total_seconds()
                    if remaining > config.capture_deadline_seconds:
                        cutoff = _format_utc(
                            now + timedelta(seconds=config.capture_deadline_seconds)
                        )
                        captures = _capture_cycle_sources(context, dependencies, cutoff)
                        quote_receipt_sha = _capture_receipt_sha(
                            captures["quote"], "quote"
                        )
                        dependencies.apply_captured_quote(context, quote_receipt_sha)
                        final_summary = _broker_summary(
                            dependencies.inspect_broker(context),
                            expected_new_entries_allowed=None,
                        )
                        request = _snapshot_mapping(
                            dependencies.build_evidence_request(
                                context,
                                captures,
                                final_summary,
                                cutoff,
                            ),
                            "trusted evidence request",
                        )
                        _require_safety(request, "trusted evidence request")
                        _validate_controller_result(
                            dependencies.run_controller(context, request),
                            config,
                        )
                        # AI actions may close/reduce virtual inventory or
                        # BLOCK_NEW.  Persist broker truth after the applied
                        # receipt instead of assuming the pre-decision state.
                        final_summary = _broker_summary(
                            dependencies.inspect_broker(context),
                            expected_new_entries_allowed=None,
                        )
                        pending_evaluations = _run_due_evaluations(
                            context, dependencies, now
                        )
                    _write_state(
                        room_root,
                        status=status,
                        lifecycle_tip=lifecycle_tip,
                        summary=final_summary,
                        pending_evaluations=pending_evaluations,
                        market_open=True,
                    )
                    if _bounded_iteration_done(iteration, max_iterations):
                        break
                    dependencies.sleep(
                        _sleep_seconds(
                            now,
                            config.window_end,
                            config.cycle_interval_seconds,
                        )
                    )
                    continue

                if status not in {
                    "DRAINING",
                    "WEEKEND_DRAIN_PAUSED",
                    "ENTRY_STOPPED_WAITING",
                }:
                    if bot_started:
                        stop_receipt = dependencies.stop_bot(context)
                        bot_started = False
                    else:
                        stop_receipt = {
                            **_SAFETY,
                            "new_entries_allowed": False,
                            "bot_process_count": 0,
                        }
                    _require_entry_stop_receipt(
                        stop_receipt, config, "fixed-window entry stop"
                    )
                    lifecycle_tip = _append_lifecycle(
                        room_root,
                        "ENTRY_STOP",
                        {
                            "status": "DRAINING",
                            "window_end_utc": config.window_end_utc,
                            "new_entries_allowed": False,
                            "force_close": False,
                            "original_ceiling_minutes": (
                                config.original_ceiling_minutes
                            ),
                        },
                    )
                    _write_state(
                        room_root,
                        status="DRAINING",
                        lifecycle_tip=lifecycle_tip,
                        summary=final_summary,
                        pending_evaluations=pending_evaluations,
                        market_open=market_open,
                    )
                    if broker_started:
                        dependencies.stop_broker(context)
                        broker_started = False
                elif status == "ENTRY_STOPPED_WAITING":
                    # The drain quote verifier permits only a DRAINING resume
                    # event after ENTRY_STOP.  Record this transition before
                    # any post-window source acquisition.
                    lifecycle_tip = _append_lifecycle(
                        room_root,
                        "SESSION_RESUME",
                        {
                            "status": "DRAINING",
                            "reason": "FIXED_WINDOW_ENDED_AFTER_EARLY_ENTRY_STOP",
                            "session_config_sha256": config.session_config_sha256,
                            "launch_preflight_token_sha256": (
                                config.launch_preflight_token_sha256
                            ),
                        },
                    )

                status = "DRAINING" if market_open else "WEEKEND_DRAIN_PAUSED"
                _write_state(
                    room_root,
                    status=status,
                    lifecycle_tip=lifecycle_tip,
                    summary=final_summary,
                    pending_evaluations=pending_evaluations,
                    market_open=market_open,
                )
                if not market_open:
                    if _bounded_iteration_done(iteration, max_iterations):
                        break
                    dependencies.sleep(config.drain_interval_seconds)
                    continue

                if not broker_started:
                    _start_authorized_drain_broker(
                        context, dependencies, lifecycle_tip
                    )
                    broker_started = True
                else:
                    if not _validate_process_health(
                        dependencies.broker_health(context),
                        config,
                        role="broker",
                        mode="DRAIN_ONLY",
                    ):
                        broker_started = False
                        _start_authorized_drain_broker(
                            context, dependencies, lifecycle_tip
                        )
                        broker_started = True
                final_summary = _broker_summary(
                    dependencies.inspect_broker(context),
                    expected_new_entries_allowed=False,
                )
                if (
                    final_summary["positions_count"] > 0
                    or final_summary["orders_count"] > 0
                ):
                    # The signed drain quote binds the exact DRAINING state
                    # and broker inventory it was acquired for.
                    _write_state(
                        room_root,
                        status="DRAINING",
                        lifecycle_tip=lifecycle_tip,
                        summary=final_summary,
                        pending_evaluations=pending_evaluations,
                        market_open=True,
                    )
                    cutoff = _format_utc(
                        now + timedelta(seconds=config.capture_deadline_seconds)
                    )
                    quote_capture = _snapshot_mapping(
                        dependencies.capture_drain_quote(context, cutoff),
                        "drain quote capture receipt",
                    )
                    quote_receipt_sha = _drain_quote_receipt_sha(
                        quote_capture,
                        context,
                        cutoff,
                        lifecycle_tip,
                        final_summary,
                    )
                    dependencies.apply_captured_drain_quote(context, quote_receipt_sha)
                    drain_receipt = _snapshot_mapping(
                        dependencies.drain_tick(
                            context, quote_receipt_sha, _format_utc(now)
                        ),
                        "drain receipt",
                    )
                    _validate_drain_receipt(
                        drain_receipt,
                        config,
                        quote_receipt_sha,
                    )
                    final_summary = _broker_summary(
                        dependencies.inspect_broker(context),
                        expected_new_entries_allowed=False,
                    )

                pending_evaluations = _run_due_evaluations(context, dependencies, now)
                if (
                    final_summary["positions_count"] == 0
                    and final_summary["orders_count"] == 0
                    and pending_evaluations == 0
                ):
                    status = "SEALED"
                    lifecycle_tip = _append_lifecycle(
                        room_root,
                        "SESSION_STOP",
                        {
                            "status": "SEALED",
                            "positions_count": 0,
                            "orders_count": 0,
                            "pending_evaluations": 0,
                            "new_entries_allowed": False,
                            "force_close": False,
                        },
                    )
                    _write_state(
                        room_root,
                        status=status,
                        lifecycle_tip=lifecycle_tip,
                        summary=final_summary,
                        pending_evaluations=0,
                        market_open=True,
                    )
                    break

                _write_state(
                    room_root,
                    status=status,
                    lifecycle_tip=lifecycle_tip,
                    summary=final_summary,
                    pending_evaluations=pending_evaluations,
                    market_open=True,
                )
                if _bounded_iteration_done(iteration, max_iterations):
                    break
                dependencies.sleep(config.drain_interval_seconds)
        except BaseException as exc:
            try:
                lifecycle_tip = _append_lifecycle(
                    room_root,
                    "SESSION_ERROR",
                    {
                        "status": "ERROR",
                        "error_type": type(exc).__name__,
                        "error_sha256": hashlib.sha256(
                            str(exc).encode("utf-8")
                        ).hexdigest(),
                    },
                )
                _write_state(
                    room_root,
                    status="ERROR",
                    lifecycle_tip=lifecycle_tip,
                    summary=final_summary,
                    pending_evaluations=pending_evaluations,
                    market_open=False,
                )
            except Exception:
                pass
            raise
        finally:
            if bot_started:
                try:
                    dependencies.stop_bot(context)
                except Exception:
                    pass
            if broker_started:
                try:
                    dependencies.stop_broker(context)
                except Exception:
                    pass

    return AIInventorySessionResult(
        status=status,
        room_root=room_root,
        lifecycle_tip_sha256=lifecycle_tip,
        positions_count=int(final_summary["positions_count"]),
        orders_count=int(final_summary["orders_count"]),
        pending_evaluations=pending_evaluations,
    )


def expected_screen_name(room_id: str) -> str:
    """Return the only permitted detached-screen name for a room."""

    room_id = _identifier(room_id, "room_id")
    if not room_id.startswith("paper-ai-inventory-"):
        raise AIInventorySessionConfigError("room_id is not isolated")
    return f"qr-dojo-{room_id}"


def _validate_config_instance(
    config: AIInventorySessionConfig,
) -> AIInventorySessionConfig:
    if not isinstance(config, AIInventorySessionConfig):
        raise TypeError("config must be AIInventorySessionConfig")
    # Reconstruct the data mapping so direct dataclass construction cannot
    # bypass the same validation used by the canonical JSON loader.
    mapping = {
        "contract": SESSION_CONFIG_CONTRACT,
        "experiment_id": config.experiment_id,
        "room_id": config.room_id,
        "candidate_id": config.candidate_id,
        "dependency_id": config.dependency_id,
        "pair": config.pair,
        "window_start_utc": config.window_start_utc,
        "window_end_utc": config.window_end_utc,
        "adapter_id": config.adapter_id,
        "model_id": config.model_id,
        "model_config_sha256": config.model_config_sha256,
        "producer_id": config.producer_id,
        "bot_config_sha256": config.bot_config_sha256,
        "balance_jpy": config.balance_jpy,
        "slippage_pips": config.slippage_pips,
        "financing_pips_per_day": config.financing_pips_per_day,
        "leverage": config.leverage,
        "original_ceiling_minutes": config.original_ceiling_minutes,
        "cycle_interval_seconds": config.cycle_interval_seconds,
        "drain_interval_seconds": config.drain_interval_seconds,
        "capture_deadline_seconds": config.capture_deadline_seconds,
        "evaluation_horizon_seconds": config.evaluation_horizon_seconds,
        "launch_preflight_token_sha256": (config.launch_preflight_token_sha256),
        **_SAFETY,
        "session_config_sha256": config.session_config_sha256,
    }
    return ai_inventory_session_config_from_mapping(config.repository_root, mapping)


def _validate_dependencies(
    dependencies: AIInventorySessionDependencies, expected_id: str
) -> AIInventorySessionDependencies:
    if not isinstance(dependencies, AIInventorySessionDependencies):
        raise TypeError("dependencies must be AIInventorySessionDependencies")
    if dependencies.dependency_id != expected_id:
        raise AIInventorySessionConfigError("dependency id binding mismatch")
    for name in (
        "clock",
        "sleep",
        "verify_launch_preflight",
        "start_broker",
        "inspect_drain_checkpoint",
        "start_drain_broker",
        "stop_broker",
        "broker_health",
        "start_bot",
        "stop_bot",
        "bot_health",
        "capture_source",
        "apply_captured_quote",
        "capture_drain_quote",
        "apply_captured_drain_quote",
        "inspect_broker",
        "build_evidence_request",
        "run_controller",
        "evaluation_plan",
        "evaluate",
        "drain_tick",
    ):
        if not callable(getattr(dependencies, name)):
            raise AIInventorySessionConfigError(
                f"dependency hook {name} is unavailable"
            )
    return dependencies


def _validate_preflight(
    config: AIInventorySessionConfig, preflight: Mapping[str, Any]
) -> None:
    _require_launch_safety(preflight)
    expected = {
        "experiment_id": config.experiment_id,
        "room_id": config.room_id,
        "candidate_id": config.candidate_id,
        "adapter_id": config.adapter_id,
        "model_id": config.model_id,
        "config_sha256": config.model_config_sha256,
        "producer_id": config.producer_id,
        "launch_preflight_token_sha256": (config.launch_preflight_token_sha256),
        "future_window": {
            "start_utc": config.window_start_utc,
            "end_utc": config.window_end_utc,
        },
        "paper_room_launched": False,
    }
    for field, value in expected.items():
        if preflight.get(field) != value:
            raise AIInventorySessionIntegrityError(
                f"launch preflight {field} binding mismatch"
            )
    _sha(preflight.get("paper_eligible_event_sha256"), "paper eligible event")
    _sha(preflight.get("future_registry_sha256"), "future registry")


def _build_session_contract(
    config: AIInventorySessionConfig,
    preflight: Mapping[str, Any],
    *,
    screen_name: str,
    process_argv: Sequence[str],
) -> dict[str, Any]:
    body = {
        "contract": SESSION_CONTRACT,
        "experiment_id": config.experiment_id,
        "room_id": config.room_id,
        "candidate_id": config.candidate_id,
        "dependency_id": config.dependency_id,
        "pair": config.pair,
        "window_start_utc": config.window_start_utc,
        "window_end_utc": config.window_end_utc,
        "source_roles": list(SOURCE_ROLES),
        "active_source_capture_after_window_allowed": False,
        "drain_quote_required": True,
        "drain_quote_receipt_contract": DRAIN_QUOTE_RECEIPT_CONTRACT,
        "adapter_id": config.adapter_id,
        "model_id": config.model_id,
        "model_config_sha256": config.model_config_sha256,
        "producer_id": config.producer_id,
        "bot_config_sha256": config.bot_config_sha256,
        "balance_jpy": config.balance_jpy,
        "slippage_pips": config.slippage_pips,
        "financing_pips_per_day": config.financing_pips_per_day,
        "leverage": config.leverage,
        "original_ceiling_minutes": config.original_ceiling_minutes,
        "cycle_interval_seconds": config.cycle_interval_seconds,
        "drain_interval_seconds": config.drain_interval_seconds,
        "capture_deadline_seconds": config.capture_deadline_seconds,
        "evaluation_horizon_seconds": config.evaluation_horizon_seconds,
        "launch_preflight_token_sha256": (config.launch_preflight_token_sha256),
        "paper_eligible_event_sha256": preflight["paper_eligible_event_sha256"],
        "future_registry_sha256": preflight["future_registry_sha256"],
        "session_config_sha256": config.session_config_sha256,
        "screen_name": screen_name,
        "process_argv": list(process_argv),
        "process_argv_sha256": _canonical_sha256(list(process_argv)),
        "environment_allowlist": ["PYTHONPATH", "STY"],
        "new_entries_after_window_allowed": False,
        "force_close_allowed": False,
        "allowed_drain_resolutions": sorted(ALLOWED_DRAIN_RESOLUTIONS),
        **_SAFETY,
    }
    return {**body, "session_contract_sha256": _canonical_sha256(body)}


def _capture_cycle_sources(
    context: AIInventorySessionContext,
    dependencies: AIInventorySessionDependencies,
    cutoff_utc: str,
) -> dict[str, Mapping[str, Any]]:
    captured: dict[str, Mapping[str, Any]] = {}
    for role in SOURCE_ROLES:
        receipt = _snapshot_mapping(
            dependencies.capture_source(context, role, cutoff_utc),
            f"{role} capture receipt",
        )
        _capture_receipt_sha(receipt, role)
        if receipt.get("cutoff_utc") not in {None, cutoff_utc}:
            raise AIInventorySessionIntegrityError(
                f"{role} capture cutoff binding mismatch"
            )
        captured[role] = receipt
    return captured


def _capture_receipt_sha(receipt: Mapping[str, Any], role: str) -> str:
    if receipt.get("source_role") != role:
        raise AIInventorySessionIntegrityError(f"{role} capture role binding mismatch")
    return _sha(receipt.get("receipt_sha256"), f"{role} capture receipt")


def _drain_quote_receipt_sha(
    receipt: Mapping[str, Any],
    context: AIInventorySessionContext,
    cutoff_utc: str,
    lifecycle_tip_sha256: str,
    broker_summary: Mapping[str, Any],
) -> str:
    """Bind one verified post-window quote to drain-only authority."""

    config = context.config
    if receipt.get("contract") != DRAIN_QUOTE_RECEIPT_CONTRACT:
        raise AIInventorySessionIntegrityError(
            "drain quote receipt contract is invalid"
        )
    _require_launch_safety(receipt)
    if receipt.get("external_broker_mutation_allowed") is not False:
        raise AIInventorySessionIntegrityError(
            "drain quote external broker mutation is invalid"
        )
    expected = {
        "experiment_id": config.experiment_id,
        "room_id": config.room_id,
        "candidate_id": config.candidate_id,
        "source_role": "quote",
        "drain_only": True,
        "new_entries_allowed": False,
        "ai_evaluation_allowed": False,
        "force_close_allowed": False,
        "original_ceiling_minutes": config.original_ceiling_minutes,
        "session_lifecycle_tip_sha256": lifecycle_tip_sha256,
        "positions_count": broker_summary["positions_count"],
        "orders_count": broker_summary["orders_count"],
        "allowed_drain_resolutions": sorted(ALLOWED_DRAIN_RESOLUTIONS),
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            raise AIInventorySessionIntegrityError(
                f"drain quote receipt {field} binding mismatch"
            )
    if _parse_utc(
        receipt.get("cutoff_utc"), "drain quote receipt cutoff_utc"
    ) != _parse_utc(cutoff_utc, "drain quote cutoff_utc"):
        raise AIInventorySessionIntegrityError(
            "drain quote receipt cutoff_utc binding mismatch"
        )
    if _parse_utc(
        receipt.get("fixed_window_end_utc"),
        "drain quote receipt fixed_window_end_utc",
    ) != config.window_end:
        raise AIInventorySessionIntegrityError(
            "drain quote receipt fixed_window_end_utc binding mismatch"
        )
    session_contract_file_sha256 = hashlib.sha256(
        _read_regular_file(
            context.room_root / SESSION_CONTRACT_NAME,
            MAX_CONFIG_BYTES,
            "session contract",
        )
    ).hexdigest()
    if (
        receipt.get("session_contract_file_sha256")
        != session_contract_file_sha256
    ):
        raise AIInventorySessionIntegrityError(
            "drain quote receipt session contract file binding mismatch"
        )
    state = _load_state(context.room_root / SESSION_STATE_NAME)
    if state is None or state.get("status") != "DRAINING":
        raise AIInventorySessionIntegrityError(
            "drain quote receipt has no DRAINING session state"
        )
    session_state_file_sha256 = hashlib.sha256(
        _read_regular_file(
            context.room_root / SESSION_STATE_NAME,
            MAX_CONFIG_BYTES,
            "session state",
        )
    ).hexdigest()
    if receipt.get("session_state_file_sha256") != session_state_file_sha256:
        raise AIInventorySessionIntegrityError(
            "drain quote receipt session state file binding mismatch"
        )
    for field in (
        "broker_ledger_terminal_sha256",
        "broker_snapshot_sha256",
        "broker_snapshot_ledger_terminal_sha256",
        "canonical_source_sha256",
        "raw_source_bytes_sha256",
        "source_watermark_sha256",
        "adapter_executable_sha256",
        "adapter_config_sha256",
        "capture_manifest_file_sha256",
        "capture_manifest_sha256",
    ):
        _sha(receipt.get(field), f"drain quote receipt {field}")
    if receipt.get("broker_ledger_terminal_sha256") != receipt.get(
        "broker_snapshot_ledger_terminal_sha256"
    ):
        raise AIInventorySessionIntegrityError(
            "drain quote broker snapshot does not match the ledger tip"
        )
    if receipt.get("canonical_source_sha256") != receipt.get("raw_source_bytes_sha256"):
        raise AIInventorySessionIntegrityError(
            "drain quote source byte digest mismatch"
        )
    sequence = _bounded_int(
        receipt.get("sequence"), "drain quote receipt sequence", 1, 1_000_000
    )
    previous = _sha(
        receipt.get("previous_receipt_sha256"),
        "drain quote previous_receipt_sha256",
    )
    if sequence == 1 and previous != GENESIS_LIFECYCLE_SHA256:
        raise AIInventorySessionIntegrityError("drain quote receipt genesis mismatch")
    for field in (
        "provider_kind",
        "adapter_id",
        "adapter_module",
        "adapter_callable",
        "capture_key_id",
    ):
        _identifier(receipt.get(field), f"drain quote receipt {field}")
    provider_at = _parse_utc(
        receipt.get("provider_timestamp_utc"),
        "drain quote provider_timestamp_utc",
    )
    fetched_at = _parse_utc(receipt.get("fetched_at_utc"), "drain quote fetched_at_utc")
    cutoff_at = _parse_utc(
        receipt.get("cutoff_utc"), "drain quote receipt cutoff_utc"
    )
    if not (provider_at <= fetched_at <= cutoff_at):
        raise AIInventorySessionIntegrityError(
            "drain quote receipt chronology is invalid"
        )
    claimed = _sha(receipt.get("receipt_sha256"), "drain quote receipt_sha256")
    signature = receipt.get("signature_base64")
    if not isinstance(signature, str):
        raise AIInventorySessionIntegrityError(
            "drain quote receipt signature is invalid"
        )
    try:
        signature_bytes = base64.b64decode(signature, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AIInventorySessionIntegrityError(
            "drain quote receipt signature is invalid"
        ) from exc
    if (
        len(signature_bytes) != 64
        or base64.b64encode(signature_bytes).decode("ascii") != signature
    ):
        raise AIInventorySessionIntegrityError(
            "drain quote receipt signature is invalid"
        )
    body = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_sha256", "signature_base64"}
    }
    if claimed != _canonical_sha256(body):
        raise AIInventorySessionIntegrityError("drain quote receipt digest mismatch")
    return claimed


def _run_due_evaluations(
    context: AIInventorySessionContext,
    dependencies: AIInventorySessionDependencies,
    now: datetime,
) -> int:
    plan = _snapshot_mapping(
        dependencies.evaluation_plan(context, _format_utc(now)),
        "evaluation plan",
    )
    due = plan.get("due")
    pending = plan.get("pending_count")
    if not isinstance(due, list):
        raise AIInventorySessionIntegrityError("evaluation plan due must be a list")
    pending_count = _bounded_int(pending, "evaluation pending_count", 0, 1_000_000)
    for raw_item in due:
        item = _snapshot_mapping(raw_item, "due evaluation")
        decision_sha256 = _sha(
            item.get("decision_sha256"), "evaluation decision_sha256"
        )
        horizon = _parse_utc(
            item.get("horizon_end_at_utc"), "evaluation horizon_end_at_utc"
        )
        if horizon > now:
            raise AIInventorySessionIntegrityError(
                "evaluation plan contains a future-due outcome"
            )
        kind = item.get("outcome_kind")
        if kind not in {"FIXED_HORIZON", "SETTLEMENT"}:
            raise AIInventorySessionIntegrityError("evaluation outcome_kind is invalid")
        result = _snapshot_mapping(
            dependencies.evaluate(context, item), "evaluation result"
        )
        if result.get("decision_sha256") not in {None, decision_sha256}:
            raise AIInventorySessionIntegrityError(
                "evaluation result decision binding mismatch"
            )
        _require_safety(result, "evaluation result")
    refreshed = _snapshot_mapping(
        dependencies.evaluation_plan(context, _format_utc(now)),
        "refreshed evaluation plan",
    )
    if not isinstance(refreshed.get("due"), list):
        raise AIInventorySessionIntegrityError(
            "refreshed evaluation plan due must be a list"
        )
    if refreshed["due"]:
        raise AIInventorySessionIntegrityError(
            "due evaluations remain after the evaluation pass"
        )
    refreshed_pending = _bounded_int(
        refreshed.get("pending_count"),
        "refreshed evaluation pending_count",
        0,
        1_000_000,
    )
    if refreshed_pending > pending_count and not due:
        raise AIInventorySessionIntegrityError(
            "evaluation pending count increased without a new controller cycle"
        )
    return refreshed_pending


def _validate_broker_start_receipt(
    value: Mapping[str, Any], config: AIInventorySessionConfig
) -> None:
    receipt = _snapshot_mapping(value, "broker start receipt")
    _require_safety(receipt, "broker start receipt")
    if receipt.get("owner_count") != 1:
        raise AIInventorySessionIntegrityError("broker must have exactly one owner")
    if receipt.get("alive") is not True or receipt.get("mode") != "ACTIVE":
        raise AIInventorySessionIntegrityError(
            "broker did not start in active mode"
        )
    if receipt.get("checkpoint_reconciled") is not True:
        raise AIInventorySessionIntegrityError(
            "broker checkpoint was not reconciled"
        )
    if receipt.get("candidate_id") != config.candidate_id:
        raise AIInventorySessionIntegrityError("broker candidate binding mismatch")
    if receipt.get("room_id") != config.room_id:
        raise AIInventorySessionIntegrityError("broker room binding mismatch")
    _validate_process_identity(
        receipt.get("process_identity"), "broker process identity"
    )


def _validate_bot_start_receipt(
    value: Mapping[str, Any], config: AIInventorySessionConfig
) -> None:
    receipt = _snapshot_mapping(value, "bot start receipt")
    _require_safety(receipt, "bot start receipt")
    if receipt.get("bot_process_count") != 1:
        raise AIInventorySessionIntegrityError("bot must have exactly one process")
    if receipt.get("alive") is not True or receipt.get("mode") != "ACTIVE":
        raise AIInventorySessionIntegrityError("bot did not start in active mode")
    if receipt.get("room_id") != config.room_id:
        raise AIInventorySessionIntegrityError("bot room binding mismatch")
    if not isinstance(receipt.get("new_entries_allowed"), bool):
        raise AIInventorySessionIntegrityError(
            "active bot did not declare its entry state"
        )
    _validate_process_identity(receipt.get("process_identity"), "bot process identity")


def _validate_process_health(
    value: Mapping[str, Any],
    config: AIInventorySessionConfig,
    *,
    role: str,
    mode: str,
) -> bool:
    receipt = _snapshot_mapping(value, f"{role} health receipt")
    _require_safety(receipt, f"{role} health receipt")
    if receipt.get("role") != role or receipt.get("mode") != mode:
        raise AIInventorySessionIntegrityError(
            f"{role} health mode binding mismatch"
        )
    if receipt.get("room_id") != config.room_id:
        raise AIInventorySessionIntegrityError(
            f"{role} health room binding mismatch"
        )
    if receipt.get("candidate_id") != config.candidate_id:
        raise AIInventorySessionIntegrityError(
            f"{role} health candidate binding mismatch"
        )
    alive = receipt.get("alive")
    if not isinstance(alive, bool):
        raise AIInventorySessionIntegrityError(
            f"{role} health liveness is invalid"
        )
    count_field = "owner_count" if role == "broker" else "process_count"
    expected_count = 1 if alive else 0
    if receipt.get(count_field) != expected_count:
        raise AIInventorySessionIntegrityError(
            f"{role} health process count is invalid"
        )
    if alive:
        _validate_process_identity(
            receipt.get("process_identity"), f"{role} health process identity"
        )
    elif receipt.get("process_identity") is not None:
        raise AIInventorySessionIntegrityError(
            f"dead {role} retained a process identity"
        )
    return alive


def _start_authorized_drain_broker(
    context: AIInventorySessionContext,
    dependencies: AIInventorySessionDependencies,
    lifecycle_tip_sha256: str,
) -> None:
    checkpoint = _snapshot_mapping(
        dependencies.inspect_drain_checkpoint(context),
        "drain broker checkpoint",
    )
    positions_count, orders_count, _ = _validate_drain_broker_checkpoint(
        context.config,
        checkpoint,
    )
    current_state = _load_state(context.room_root / SESSION_STATE_NAME)
    if current_state is None:
        raise AIInventorySessionIntegrityError(
            "drain broker authorization requires session state"
        )
    _write_state(
        context.room_root,
        status="DRAINING",
        lifecycle_tip=lifecycle_tip_sha256,
        summary={
            **_empty_broker_summary(),
            "positions_count": positions_count,
            "orders_count": orders_count,
            "new_entries_allowed": False,
        },
        pending_evaluations=_bounded_int(
            current_state.get("pending_evaluations"),
            "drain pending evaluations",
            0,
            1_000_000,
        ),
        market_open=True,
    )
    authorization_path = _append_drain_broker_restart_receipt(
        context,
        lifecycle_tip_sha256,
        checkpoint,
    )
    _validate_drain_broker_start_receipt(
        dependencies.start_drain_broker(context, authorization_path),
        context,
        authorization_path,
    )


def _append_drain_broker_restart_receipt(
    context: AIInventorySessionContext,
    lifecycle_tip_sha256: str,
    checkpoint: Mapping[str, Any],
) -> Path:
    config = context.config
    positions_count, orders_count, recovery = _validate_drain_broker_checkpoint(
        config,
        checkpoint,
    )
    state_path = context.room_root / SESSION_STATE_NAME
    state = _load_state(state_path)
    if state is None or state.get("status") != "DRAINING":
        raise AIInventorySessionIntegrityError(
            "drain broker authorization requires DRAINING session state"
        )
    if (
        state.get("lifecycle_tip_sha256") != lifecycle_tip_sha256
        or state.get("positions_count") != positions_count
        or state.get("orders_count") != orders_count
        or state.get("new_entries_allowed") is not False
    ):
        raise AIInventorySessionIntegrityError(
            "drain broker checkpoint is not bound to session state"
        )
    contract_file_sha256 = hashlib.sha256(
        _read_regular_file(
            context.room_root / SESSION_CONTRACT_NAME,
            MAX_CONFIG_BYTES,
            "session contract",
        )
    ).hexdigest()
    state_file_sha256 = hashlib.sha256(
        _read_regular_file(
            state_path,
            MAX_CONFIG_BYTES,
            "session state",
        )
    ).hexdigest()
    directory = context.room_root / DRAIN_BROKER_RESTART_DIRECTORY_NAME
    with _exclusive_drain_restart_lock(directory):
        rows = _read_drain_broker_restart_chain(directory)
        body = {
            "contract": DRAIN_BROKER_RESTART_RECEIPT_CONTRACT,
            "sequence": len(rows) + 1,
            "previous_receipt_sha256": (
                rows[-1]["receipt_sha256"]
                if rows
                else GENESIS_LIFECYCLE_SHA256
            ),
            "experiment_id": config.experiment_id,
            "room_id": config.room_id,
            "candidate_id": config.candidate_id,
            "mode": "DRAIN_ONLY",
            "authorized_at_utc": _format_utc(datetime.now(timezone.utc)),
            "session_contract_file_sha256": contract_file_sha256,
            "session_lifecycle_tip_sha256": lifecycle_tip_sha256,
            "session_state_file_sha256": state_file_sha256,
            "broker_ledger_terminal_sha256": checkpoint[
                "broker_ledger_terminal_sha256"
            ],
            "broker_snapshot_sha256": checkpoint["broker_snapshot_sha256"],
            "broker_snapshot_ledger_terminal_sha256": checkpoint[
                "broker_snapshot_ledger_terminal_sha256"
            ],
            "positions_count": positions_count,
            "orders_count": orders_count,
            **recovery,
            "new_entries_allowed": False,
            "ai_decision_allowed": False,
            "force_close_allowed": False,
            "bot_rpc_commands": [],
            "runner_rpc_commands": list(DRAIN_BROKER_RUNNER_COMMANDS),
            **_SAFETY,
        }
        receipt = {**body, "receipt_sha256": _canonical_sha256(body)}
        filename = (
            f"{receipt['sequence']:08d}-{receipt['receipt_sha256']}.json"
        )
        path = directory / filename
        _write_new_json(path, receipt, "drain broker restart receipt")
        persisted = _read_drain_broker_restart_chain(directory)
        if not persisted or persisted[-1] != receipt:
            raise AIInventorySessionIntegrityError(
                "persisted drain broker restart receipt mismatch"
            )
        return path


def _validate_drain_broker_checkpoint(
    config: AIInventorySessionConfig,
    checkpoint: Mapping[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    _require_safety(checkpoint, "drain broker checkpoint")
    expected_checkpoint = {
        "experiment_id": config.experiment_id,
        "room_id": config.room_id,
        "candidate_id": config.candidate_id,
        "checkpoint_reconciled": True,
        "new_entries_allowed": False,
        "ai_decision_allowed": False,
        "force_close_allowed": False,
    }
    for field, expected_value in expected_checkpoint.items():
        if checkpoint.get(field) != expected_value:
            raise AIInventorySessionIntegrityError(
                f"drain broker checkpoint {field} binding mismatch"
            )
    for field in (
        "broker_ledger_terminal_sha256",
        "broker_snapshot_sha256",
        "broker_snapshot_ledger_terminal_sha256",
    ):
        _sha(checkpoint.get(field), f"drain broker checkpoint {field}")
    ledger_tip = str(checkpoint["broker_ledger_terminal_sha256"])
    snapshot_tip = str(checkpoint["broker_snapshot_ledger_terminal_sha256"])
    positions_count = _bounded_int(
        checkpoint.get("positions_count"),
        "drain broker checkpoint positions_count",
        0,
        1_000_000,
    )
    orders_count = _bounded_int(
        checkpoint.get("orders_count"),
        "drain broker checkpoint orders_count",
        0,
        1_000_000,
    )
    recovery = _validated_broker_recovery_fields(
        checkpoint,
        ledger_tip=ledger_tip,
        snapshot_tip=snapshot_tip,
    )
    return positions_count, orders_count, recovery


def _validated_broker_recovery_fields(
    value: Mapping[str, Any],
    *,
    ledger_tip: str,
    snapshot_tip: str,
) -> dict[str, Any]:
    wal_sha256 = value.get("broker_recovery_wal_sha256")
    wal_checkpoint = value.get(
        "broker_recovery_wal_checkpoint_ledger_sha256"
    )
    expected_count = _bounded_int(
        value.get("broker_recovery_wal_expected_event_count"),
        "broker recovery WAL expected_event_count",
        0,
        MAX_QUOTE_RECOVERY_EVENTS,
    )
    applied_count = _bounded_int(
        value.get("broker_recovery_wal_applied_event_count"),
        "broker recovery WAL applied_event_count",
        0,
        MAX_QUOTE_RECOVERY_EVENTS,
    )
    validated = value.get("broker_recovery_wal_validated")
    if not isinstance(validated, bool):
        raise AIInventorySessionIntegrityError(
            "broker recovery WAL validation flag is invalid"
        )
    if wal_sha256 is None:
        if (
            wal_checkpoint is not None
            or expected_count != 0
            or applied_count != 0
            or validated
            or ledger_tip != snapshot_tip
        ):
            raise AIInventorySessionIntegrityError(
                "broker checkpoint diverged without a validated recovery WAL"
            )
    else:
        wal_sha256 = _sha(
            wal_sha256,
            "broker_recovery_wal_sha256",
        )
        wal_checkpoint = _sha(
            wal_checkpoint,
            "broker_recovery_wal_checkpoint_ledger_sha256",
        )
        if (
            wal_checkpoint != snapshot_tip
            or not validated
            or applied_count > expected_count
        ):
            raise AIInventorySessionIntegrityError(
                "broker recovery WAL binding is invalid"
            )
    return {
        "broker_recovery_wal_sha256": wal_sha256,
        "broker_recovery_wal_checkpoint_ledger_sha256": wal_checkpoint,
        "broker_recovery_wal_expected_event_count": expected_count,
        "broker_recovery_wal_applied_event_count": applied_count,
        "broker_recovery_wal_validated": validated,
    }


def _validate_drain_broker_start_receipt(
    value: Mapping[str, Any],
    context: AIInventorySessionContext,
    authorization_path: Path,
) -> None:
    receipt = _snapshot_mapping(value, "drain broker start receipt")
    _require_safety(receipt, "drain broker start receipt")
    authorization = _read_json_object(
        authorization_path,
        MAX_CONFIG_BYTES,
        "drain broker restart authorization",
    )
    authorization_sha256 = _sha(
        authorization.get("receipt_sha256"),
        "drain broker authorization receipt_sha256",
    )
    authorization_file_sha256 = hashlib.sha256(
        _read_regular_file(
            authorization_path,
            MAX_CONFIG_BYTES,
            "drain broker restart authorization",
        )
    ).hexdigest()
    config = context.config
    expected = {
        "contract": DRAIN_BROKER_START_RECEIPT_CONTRACT,
        "experiment_id": config.experiment_id,
        "room_id": config.room_id,
        "candidate_id": config.candidate_id,
        "mode": "DRAIN_ONLY",
        "owner_count": 1,
        "alive": True,
        "authorization_receipt_sha256": authorization_sha256,
        "authorization_file_sha256": authorization_file_sha256,
        "authorization_path_relative": str(
            authorization_path.relative_to(context.room_root)
        ),
        "new_entries_allowed": False,
        "ai_decision_allowed": False,
        "force_close_allowed": False,
        "bot_rpc_commands": [],
        "runner_rpc_commands": list(DRAIN_BROKER_RUNNER_COMMANDS),
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            raise AIInventorySessionIntegrityError(
                f"drain broker start {field} binding mismatch"
            )
    _parse_utc(receipt.get("started_at_utc"), "drain broker started_at_utc")
    _validate_process_identity(
        receipt.get("process_identity"), "drain broker process identity"
    )
    claimed = _sha(
        receipt.get("start_receipt_sha256"),
        "drain broker start receipt_sha256",
    )
    body = {
        key: item
        for key, item in receipt.items()
        if key != "start_receipt_sha256"
    }
    if claimed != _canonical_sha256(body):
        raise AIInventorySessionIntegrityError(
            "drain broker start receipt digest mismatch"
        )


def _validate_process_identity(value: Any, label: str) -> None:
    identity = _snapshot_mapping(value, label)
    pid = identity.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise AIInventorySessionIntegrityError(f"{label} pid is invalid")
    _sha(identity.get("argv_sha256"), f"{label} argv_sha256")


def _require_entry_stop_receipt(
    value: Mapping[str, Any],
    config: AIInventorySessionConfig,
    label: str,
) -> None:
    receipt = _snapshot_mapping(value, label)
    _require_safety(receipt, label)
    if receipt.get("new_entries_allowed") is not False:
        raise AIInventorySessionIntegrityError(f"{label} did not disable new entries")
    count = receipt.get("bot_process_count")
    if count != 0:
        raise AIInventorySessionIntegrityError(f"{label} did not stop the bot process")
    room_id = receipt.get("room_id")
    if room_id not in {None, config.room_id}:
        raise AIInventorySessionIntegrityError(f"{label} room binding mismatch")


def _validate_controller_result(
    value: Mapping[str, Any], config: AIInventorySessionConfig
) -> None:
    result = _snapshot_mapping(value, "controller result")
    _require_safety(result, "controller result")
    if result.get("room_id") not in {None, config.room_id}:
        raise AIInventorySessionIntegrityError(
            "controller result room binding mismatch"
        )
    _sha(result.get("decision_sha256"), "controller decision_sha256")
    _sha(result.get("applied_receipt_sha256"), "controller applied receipt")


def _validate_drain_receipt(
    receipt: Mapping[str, Any],
    config: AIInventorySessionConfig,
    drain_quote_receipt_sha256: str,
) -> None:
    _require_safety(receipt, "drain receipt")
    if receipt.get("new_entries_allowed") is not False:
        raise AIInventorySessionIntegrityError("drain attempted to permit a new entry")
    if receipt.get("force_close") is not False:
        raise AIInventorySessionIntegrityError("drain force-close is forbidden")
    if receipt.get("original_ceiling_minutes") != config.original_ceiling_minutes:
        raise AIInventorySessionIntegrityError(
            "drain original ceiling binding mismatch"
        )
    if receipt.get("drain_quote_receipt_sha256") != drain_quote_receipt_sha256:
        raise AIInventorySessionIntegrityError(
            "drain quote receipt was not bound to the drain tick"
        )
    resolutions = receipt.get("resolutions")
    if not isinstance(resolutions, list) or any(
        not isinstance(item, str) or item not in ALLOWED_DRAIN_RESOLUTIONS
        for item in resolutions
    ):
        raise AIInventorySessionIntegrityError(
            "drain contains a non-original resolution"
        )


def _broker_summary(
    value: Mapping[str, Any],
    *,
    expected_new_entries_allowed: bool | None,
) -> dict[str, Any]:
    summary = _snapshot_mapping(value, "broker summary")
    _require_safety(summary, "broker summary")
    positions = _bounded_int(
        summary.get("positions_count"), "positions_count", 0, 1_000_000
    )
    orders = _bounded_int(summary.get("orders_count"), "orders_count", 0, 1_000_000)
    actual_entry_state = summary.get("new_entries_allowed")
    if not isinstance(actual_entry_state, bool):
        raise AIInventorySessionIntegrityError("broker new-entry state is not boolean")
    if (
        expected_new_entries_allowed is not None
        and actual_entry_state is not expected_new_entries_allowed
    ):
        raise AIInventorySessionIntegrityError(
            "broker new-entry state is inconsistent with lifecycle"
        )
    return {
        "positions_count": positions,
        "orders_count": orders,
        "new_entries_allowed": actual_entry_state,
        "balance_jpy": _finite_or_none(summary.get("balance_jpy"), "balance_jpy"),
        "equity_jpy": _finite_or_none(summary.get("equity_jpy"), "equity_jpy"),
        "margin_used_jpy": _finite_or_none(
            summary.get("margin_used_jpy"), "margin_used_jpy"
        ),
        **_SAFETY,
    }


def _empty_broker_summary() -> dict[str, Any]:
    return {
        "positions_count": 0,
        "orders_count": 0,
        "new_entries_allowed": False,
        "balance_jpy": None,
        "equity_jpy": None,
        "margin_used_jpy": None,
        **_SAFETY,
    }


def _canonical_room_root(config: AIInventorySessionConfig) -> Path:
    base = canonical_paper_ai_rooms_root(config.repository_root)
    expected = base / config.experiment_id / config.room_id
    try:
        root = expected.resolve(strict=True)
        canonical_base = base.resolve(strict=True)
    except OSError as exc:
        raise AIInventorySessionIntegrityError(
            "canonical paper-AI room does not exist"
        ) from exc
    if root.parent.parent != canonical_base:
        raise AIInventorySessionIntegrityError(
            "paper-AI room escaped the canonical root"
        )
    if root != expected:
        raise AIInventorySessionIntegrityError("paper-AI room must not be a symlink")
    if not root.is_dir():
        raise AIInventorySessionIntegrityError(
            "canonical paper-AI room is not a directory"
        )
    return root


@contextmanager
def _exclusive_owner_lock(room_root: Path) -> Iterator[None]:
    path = room_root / SESSION_OWNER_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AIInventorySessionIntegrityError(
                "session owner lock is not a regular file"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AIInventorySessionBusyError(
                "another AI inventory session owns this room"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def _exclusive_drain_restart_lock(directory: Path) -> Iterator[None]:
    try:
        directory.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise AIInventorySessionIntegrityError(
            "drain broker restart directory is unavailable"
        ) from exc
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise AIInventorySessionIntegrityError(
            "drain broker restart directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or directory.resolve() != directory:
        raise AIInventorySessionIntegrityError(
            "drain broker restart directory is not canonical"
        )
    path = directory / DRAIN_BROKER_RESTART_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AIInventorySessionIntegrityError(
                "drain broker restart lock is not a regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_drain_broker_restart_chain(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    expected_previous = GENESIS_LIFECYCLE_SHA256
    expected_identity: tuple[str, str, str] | None = None
    try:
        children = sorted(directory.iterdir(), key=lambda child: child.name)
    except OSError as exc:
        raise AIInventorySessionIntegrityError(
            "drain broker restart chain is unavailable"
        ) from exc
    receipt_paths: list[Path] = []
    for child in children:
        if child.name == DRAIN_BROKER_RESTART_LOCK_NAME:
            continue
        if child.is_symlink() or not child.is_file() or child.suffix != ".json":
            raise AIInventorySessionIntegrityError(
                "drain broker restart directory contains an unexpected entry"
            )
        receipt_paths.append(child)
    for sequence, path in enumerate(receipt_paths, start=1):
        row = _read_json_object(
            path,
            MAX_CONFIG_BYTES,
            "drain broker restart receipt",
        )
        if _read_regular_file(
            path,
            MAX_CONFIG_BYTES,
            "drain broker restart receipt",
        ) != _canonical_bytes(row) + b"\n":
            raise AIInventorySessionIntegrityError(
                "drain broker restart receipt is not canonical JSON"
            )
        if row.get("contract") != DRAIN_BROKER_RESTART_RECEIPT_CONTRACT:
            raise AIInventorySessionIntegrityError(
                "drain broker restart chain contract mismatch"
            )
        if row.get("sequence") != sequence:
            raise AIInventorySessionIntegrityError(
                "drain broker restart chain sequence mismatch"
            )
        if row.get("previous_receipt_sha256") != expected_previous:
            raise AIInventorySessionIntegrityError(
                "drain broker restart chain link mismatch"
            )
        _require_safety(row, "drain broker restart chain")
        claimed = _sha(
            row.get("receipt_sha256"),
            "drain broker restart chain receipt_sha256",
        )
        body = {
            key: item
            for key, item in row.items()
            if key != "receipt_sha256"
        }
        if claimed != _canonical_sha256(body):
            raise AIInventorySessionIntegrityError(
                "drain broker restart chain digest mismatch"
            )
        if path.name != f"{sequence:08d}-{claimed}.json":
            raise AIInventorySessionIntegrityError(
                "drain broker restart receipt filename mismatch"
            )
        if row.get("mode") != "DRAIN_ONLY":
            raise AIInventorySessionIntegrityError(
                "drain broker restart chain mode mismatch"
            )
        if (
            row.get("new_entries_allowed") is not False
            or row.get("ai_decision_allowed") is not False
            or row.get("force_close_allowed") is not False
            or row.get("bot_rpc_commands") != []
            or row.get("runner_rpc_commands")
            != list(DRAIN_BROKER_RUNNER_COMMANDS)
        ):
            raise AIInventorySessionIntegrityError(
                "drain broker restart chain grants a forbidden capability"
            )
        _parse_utc(
            row.get("authorized_at_utc"),
            "drain broker restart authorized_at_utc",
        )
        for field in (
            "previous_receipt_sha256",
            "session_contract_file_sha256",
            "session_lifecycle_tip_sha256",
            "session_state_file_sha256",
            "broker_ledger_terminal_sha256",
            "broker_snapshot_sha256",
            "broker_snapshot_ledger_terminal_sha256",
        ):
            _sha(row.get(field), f"drain broker restart chain {field}")
        _validated_broker_recovery_fields(
            row,
            ledger_tip=str(row["broker_ledger_terminal_sha256"]),
            snapshot_tip=str(
                row["broker_snapshot_ledger_terminal_sha256"]
            ),
        )
        _bounded_int(
            row.get("positions_count"),
            "drain broker restart chain positions_count",
            0,
            1_000_000,
        )
        _bounded_int(
            row.get("orders_count"),
            "drain broker restart chain orders_count",
            0,
            1_000_000,
        )
        identity = (
            _identifier(row.get("experiment_id"), "drain experiment_id"),
            _identifier(row.get("room_id"), "drain room_id"),
            _sha(row.get("candidate_id"), "drain candidate_id"),
        )
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise AIInventorySessionIntegrityError(
                "drain broker restart chain identity mismatch"
            )
        rows.append(row)
        expected_previous = claimed
    return rows


def _write_immutable_contract(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        if _read_regular_file(path, MAX_CONFIG_BYTES, "session contract") != encoded:
            raise AIInventorySessionIntegrityError(
                "immutable session contract mismatch"
            )
        return
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_owner_receipt(
    room_root: Path,
    config: AIInventorySessionConfig,
    screen_name: str,
    process_argv: Sequence[str],
    session_contract: Mapping[str, Any],
) -> None:
    started_at = _format_utc(datetime.now(timezone.utc))
    body = {
        "contract": SESSION_OWNER_CONTRACT,
        "experiment_id": config.experiment_id,
        "room_id": config.room_id,
        "candidate_id": config.candidate_id,
        "pid": os.getpid(),
        "started_at_utc": started_at,
        "screen_name": screen_name,
        "process_argv": list(process_argv),
        "process_argv_sha256": _canonical_sha256(list(process_argv)),
        "session_config_sha256": config.session_config_sha256,
        "session_contract_sha256": session_contract["session_contract_sha256"],
        **_SAFETY,
    }
    _write_atomic_json(
        room_root / SESSION_OWNER_NAME,
        {**body, "owner_sha256": _canonical_sha256(body)},
    )


def _append_lifecycle(room_root: Path, event: str, payload: Mapping[str, Any]) -> str:
    if event not in {
        "SESSION_START",
        "SESSION_RESUME",
        "ENTRY_STOP",
        "SESSION_ERROR",
        "SESSION_STOP",
    }:
        raise AIInventorySessionIntegrityError("lifecycle event is invalid")
    path = room_root / SESSION_LIFECYCLE_NAME
    rows = _read_lifecycle(path)
    body = {
        "contract": SESSION_LIFECYCLE_CONTRACT,
        "sequence": len(rows) + 1,
        "previous_event_sha256": (
            rows[-1]["event_sha256"] if rows else GENESIS_LIFECYCLE_SHA256
        ),
        "event": event,
        "recorded_at_utc": _format_utc(datetime.now(timezone.utc)),
        "payload": _snapshot_mapping(payload, "lifecycle payload"),
        **_SAFETY,
    }
    record = {**body, "event_sha256": _canonical_sha256(body)}
    encoded = _canonical_bytes(record) + b"\n"
    if len(encoded) > MAX_LEDGER_ROW_BYTES:
        raise AIInventorySessionIntegrityError("lifecycle row is too large")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(room_root)
    persisted = _read_lifecycle(path)
    if persisted[-1] != record:
        raise AIInventorySessionIntegrityError("persisted lifecycle row mismatch")
    return str(record["event_sha256"])


def _lifecycle_tip(path: Path) -> str:
    rows = _read_lifecycle(path)
    return str(rows[-1]["event_sha256"]) if rows else GENESIS_LIFECYCLE_SHA256


def _read_lifecycle(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = _read_regular_file(path, MAX_LEDGER_BYTES, "session lifecycle")
    rows: list[dict[str, Any]] = []
    previous = GENESIS_LIFECYCLE_SHA256
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line or len(line) > MAX_LEDGER_ROW_BYTES:
            raise AIInventorySessionIntegrityError(
                "session lifecycle has an invalid row"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AIInventorySessionIntegrityError(
                "session lifecycle contains invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise AIInventorySessionIntegrityError(
                "session lifecycle row is not an object"
            )
        if row.get("contract") != SESSION_LIFECYCLE_CONTRACT:
            raise AIInventorySessionIntegrityError(
                "session lifecycle contract mismatch"
            )
        if row.get("sequence") != index:
            raise AIInventorySessionIntegrityError(
                "session lifecycle sequence mismatch"
            )
        if row.get("previous_event_sha256") != previous:
            raise AIInventorySessionIntegrityError("session lifecycle chain mismatch")
        _require_safety(row, "session lifecycle")
        claimed = _sha(row.get("event_sha256"), "lifecycle event_sha256")
        body = {key: value for key, value in row.items() if key != "event_sha256"}
        if claimed != _canonical_sha256(body):
            raise AIInventorySessionIntegrityError("session lifecycle digest mismatch")
        rows.append(row)
        previous = claimed
    return rows


def _write_state(
    room_root: Path,
    *,
    status: str,
    lifecycle_tip: str,
    summary: Mapping[str, Any],
    pending_evaluations: int,
    market_open: bool,
) -> None:
    if status not in {
        "WAITING",
        "ACTIVE",
        "WEEKEND_PAUSED",
        "ENTRY_STOPPED_WAITING",
        "DRAINING",
        "WEEKEND_DRAIN_PAUSED",
        "SEALED",
        "ERROR",
    }:
        raise AIInventorySessionIntegrityError("session state status is invalid")
    body = {
        "contract": SESSION_STATE_CONTRACT,
        "status": status,
        "updated_at_utc": _format_utc(datetime.now(timezone.utc)),
        "lifecycle_tip_sha256": _sha(lifecycle_tip, "lifecycle_tip_sha256"),
        "positions_count": int(summary["positions_count"]),
        "orders_count": int(summary["orders_count"]),
        "pending_evaluations": int(pending_evaluations),
        "new_entries_allowed": (
            bool(summary["new_entries_allowed"]) if status == "ACTIVE" else False
        ),
        "market_open": bool(market_open),
        **_SAFETY,
    }
    _write_atomic_json(
        room_root / SESSION_STATE_NAME,
        {**body, "state_sha256": _canonical_sha256(body)},
    )


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = _read_regular_file(path, MAX_CONFIG_BYTES, "session state")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AIInventorySessionIntegrityError("session state is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("contract") != SESSION_STATE_CONTRACT:
        raise AIInventorySessionIntegrityError("session state contract mismatch")
    claimed = _sha(value.get("state_sha256"), "state_sha256")
    body = {key: item for key, item in value.items() if key != "state_sha256"}
    if claimed != _canonical_sha256(body):
        raise AIInventorySessionIntegrityError("session state digest mismatch")
    _require_safety(value, "session state")
    return value


def _result_from_state(
    room_root: Path, state: Mapping[str, Any]
) -> AIInventorySessionResult:
    return AIInventorySessionResult(
        status="SEALED",
        room_root=room_root,
        lifecycle_tip_sha256=_sha(state.get("lifecycle_tip_sha256"), "lifecycle tip"),
        positions_count=_bounded_int(
            state.get("positions_count"), "positions_count", 0, 1_000_000
        ),
        orders_count=_bounded_int(
            state.get("orders_count"), "orders_count", 0, 1_000_000
        ),
        pending_evaluations=_bounded_int(
            state.get("pending_evaluations"),
            "pending_evaluations",
            0,
            1_000_000,
        ),
    )


def _validate_screen_identity(value: str | None, room_id: str) -> str:
    expected = expected_screen_name(room_id)
    if not isinstance(value, str) or not value:
        raise AIInventorySessionConfigError(
            "session must run inside its exact detached screen"
        )
    actual = value.rsplit(".", 1)[-1]
    if actual != expected:
        raise AIInventorySessionConfigError(
            "detached screen identity does not match the room"
        )
    return expected


def _validate_process_argv(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AIInventorySessionConfigError("process argv is invalid")
    argv = tuple(value)
    if not argv or len(argv) > 64:
        raise AIInventorySessionConfigError("process argv is invalid")
    if any(
        not isinstance(item, str) or not item or "\x00" in item or len(item) > 4_096
        for item in argv
    ):
        raise AIInventorySessionConfigError("process argv is invalid")
    return argv


def _repository_root(value: Path | str) -> Path:
    try:
        root = Path(value).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise AIInventorySessionConfigError("repository root is unavailable") from exc
    if not root.is_dir():
        raise AIInventorySessionConfigError("repository root is not a directory")
    return root


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _ID_RE.fullmatch(value) is None
        or Path(value).name != value
    ):
        raise AIInventorySessionConfigError(f"{label} is invalid")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AIInventorySessionIntegrityError(f"{label} is not a sha256")
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise AIInventorySessionConfigError(f"{label} must be UTC")
    match = _UTC_RE.fullmatch(value)
    if match is None:
        raise AIInventorySessionConfigError(f"{label} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AIInventorySessionConfigError(f"{label} is invalid UTC") from exc
    if parsed.utcoffset() != timedelta(0):
        raise AIInventorySessionConfigError(f"{label} must be UTC")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    utc = _utc_now(value)
    if utc.microsecond:
        return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return utc.isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_now(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AIInventorySessionIntegrityError(
            "dependency clock must return an aware datetime"
        )
    return value.astimezone(timezone.utc)


def _positive_finite(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number <= 0:
        raise AIInventorySessionConfigError(f"{label} must be positive")
    return number


def _nonnegative_finite(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number < 0:
        raise AIInventorySessionConfigError(f"{label} must be nonnegative")
    return number


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AIInventorySessionConfigError(f"{label} must be numeric")
    number = float(value)
    if not (-1.7976931348623157e308 < number < 1.7976931348623157e308):
        raise AIInventorySessionConfigError(f"{label} must be finite")
    return number


def _finite_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label)


def _bounded_int(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AIInventorySessionConfigError(f"{label} must be an integer")
    if value < low or value > high:
        raise AIInventorySessionConfigError(f"{label} is out of bounds")
    return value


def _require_safety(value: Mapping[str, Any], label: str) -> None:
    for field, expected in _SAFETY.items():
        if value.get(field) != expected:
            raise AIInventorySessionIntegrityError(
                f"{label} safety field {field} is invalid"
            )


def _require_launch_safety(value: Mapping[str, Any]) -> None:
    """Validate the existing canonical launch-token safety schema.

    Launch preflights predate the more explicit
    ``external_broker_mutation_allowed`` field.  Absence therefore means no
    granted capability; if a future token adds the field it must still be
    false.
    """

    for field, expected in {
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }.items():
        if value.get(field) != expected:
            raise AIInventorySessionIntegrityError(
                f"launch preflight safety field {field} is invalid"
            )
    if value.get("external_broker_mutation_allowed", False) is not False:
        raise AIInventorySessionIntegrityError(
            "launch preflight external broker mutation is invalid"
        )


def _snapshot_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AIInventorySessionIntegrityError(f"{label} must be an object")
    try:
        encoded = _canonical_bytes(dict(value))
        snapshot = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AIInventorySessionIntegrityError(
            f"{label} is not canonical JSON data"
        ) from exc
    if not isinstance(snapshot, dict):
        raise AIInventorySessionIntegrityError(f"{label} must be an object")
    return snapshot


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_regular_file(path: Path, limit: int, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AIInventorySessionIntegrityError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise AIInventorySessionIntegrityError(
                f"{label} is not a bounded regular file"
            )
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > limit:
            raise AIInventorySessionIntegrityError(f"{label} is too large")
        return raw
    finally:
        os.close(descriptor)


def _read_json_object(path: Path, limit: int, label: str) -> dict[str, Any]:
    raw = _read_regular_file(path, limit, label)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AIInventorySessionIntegrityError(f"{label} is invalid JSON") from exc
    return _snapshot_mapping(value, label)


def _write_new_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    encoded = _canonical_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise AIInventorySessionIntegrityError(f"{label} already exists") from exc
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    except BaseException:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical_bytes(value) + b"\n"
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temp, flags, 0o600)
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)
    os.replace(temp, path)
    _fsync_directory(path.parent)


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise AIInventorySessionIntegrityError("short durable write")
        offset += written


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sleep_seconds(now: datetime, target: datetime, cadence: int) -> float:
    return max(0.001, min(float(cadence), (target - now).total_seconds()))


def _bounded_iteration_done(iteration: int, max_iterations: int | None) -> bool:
    return max_iterations is not None and iteration >= max_iterations

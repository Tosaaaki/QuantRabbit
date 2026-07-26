"""Authenticated broker-owner boundary for a future paper-AI inventory room.

The service is deliberately dormant: no existing registry or runner imports
it.  A future versioned ``paper-ai-inventory`` runner may start exactly one
service process.  That process alone owns :class:`VirtualBroker`.

The bot receives :class:`DojoAIInventoryEntryClient`, whose authenticated RPC
allowlist contains only three new-entry methods and detached read-only
snapshots.  Quote ingestion and AI decision application use a distinct runner
credential.  Requests are processed serially in one event loop, so evidence,
quote, admission state, mutation, and receipt cannot interleave.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import socket
import stat
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_ai_evidence_packet import (
    DEDICATED_EVIDENCE_ROOT,
    verify_ai_inventory_evidence_packet,
)
from quant_rabbit.dojo_ai_inventory import validate_inventory_decision_ledger
from quant_rabbit.dojo_ai_inventory_consumer import (
    InventoryConsumerIntegrityError,
    consume_inventory_decision,
    reconcile_inventory_checkpoint_suffix,
)
from quant_rabbit.dojo_ai_inventory_quote_watermark import (
    MAX_LEDGER_BYTES as MAX_QUOTE_WATERMARK_LEDGER_BYTES,
    QUOTE_WATERMARK_LEDGER_NAME,
    append_ai_inventory_quote_watermark,
    validate_ai_inventory_quote_watermarks,
)
from quant_rabbit.dojo_ai_inventory_runtime import (
    AIInventoryAdmissionIntegrityError,
    AIInventoryEntryDeniedError,
    ENTRY_ADMISSION_REFERENCE_CONTRACT,
    ENTRY_PERMIT_RESERVED_EVENT,
    _BrokerOwnedAdmissionController,
    build_ai_inventory_admission_state,
    reconcile_entry_checkpoint_suffix,
)
from quant_rabbit.dojo_ai_inventory_session import (
    ALLOWED_DRAIN_RESOLUTIONS,
    DRAIN_BROKER_RUNNER_COMMANDS,
    verify_drain_broker_restart_authorization,
)
from quant_rabbit.dojo_ai_source_capture import (
    CANONICAL_SOURCE_ROOT,
    CAPTURE_ROOT,
    MAX_SOURCE_BYTES,
    verify_ai_source_capture_receipt,
)
from quant_rabbit.dojo_replay_lifecycle import (
    verify_paper_ai_inventory_launch_preflight,
)
from quant_rabbit.virtual_broker import VBOrder, VirtualBroker


RPC_PROTOCOL = "QR_DOJO_AI_BROKER_RPC_V1"
BROKER_STATE_CONTRACT = "QR_DOJO_AI_BROKER_STATE_V1"
QUOTE_APPLY_WAL_CONTRACT = "QR_DOJO_AI_CAPTURED_QUOTE_APPLY_WAL_V1"
QUOTE_APPLY_WAL_NAME = "captured_quote_apply_wal.json"
PENDING_FILL_REJECTED_EVENT = "AI_BLOCK_NEW_PENDING_FILL_REJECTED"
MAX_RPC_BYTES = 256 * 1024
# One quote WAL contains both the authenticated pre-state and post-state.  The
# 1 MiB bound is therefore four times the already bounded 256 KiB state/RPC
# envelope; a future larger room must replace this with a chunked WAL.
MAX_QUOTE_APPLY_WAL_BYTES = 1024 * 1024
# A single quote cannot legitimately mutate more ledger rows than fit in the
# bounded WAL.  This explicit ceiling also prevents an attacker-controlled
# decoded list from becoming an unbounded recovery loop.
MAX_QUOTE_APPLY_EVENTS = 10_000
MAX_CLOCK_SKEW_SECONDS = 30
MAX_NONCES = 10_000
MIN_HMAC_KEY_BYTES = 32
_TEST_ONLY_RAW_QUOTES_CAPABILITY = object()

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BOT_COMMANDS = frozenset(
    {
        "HEALTH",
        "ACCOUNT",
        "POSITIONS",
        "ORDERS",
        "ENTRY_MARKET",
        "ENTRY_LIMIT",
        "ENTRY_STOP",
    }
)
_RUNNER_COMMANDS = frozenset(
    {
        "HEALTH",
        "ACCOUNT",
        "POSITIONS",
        "ORDERS",
        "QUOTES",
        "DECISION_STATUS",
        "QUOTE_PROVENANCE",
        "APPLY_QUOTE",
        "APPLY_CAPTURED_QUOTE",
        "APPLY_AI_DECISION",
        "SHUTDOWN",
    }
)
_DRAIN_BOT_COMMANDS = frozenset()
_DRAIN_RUNNER_COMMANDS = frozenset(DRAIN_BROKER_RUNNER_COMMANDS)
_MUTATING_COMMANDS = frozenset(
    {
        "ENTRY_MARKET",
        "ENTRY_LIMIT",
        "ENTRY_STOP",
        "APPLY_QUOTE",
        "APPLY_CAPTURED_QUOTE",
        "APPLY_DRAIN_QUOTE",
        "APPLY_AI_DECISION",
    }
)


class AIInventoryBrokerServiceError(RuntimeError):
    """The isolated local broker service rejected or could not serve a request."""


class AIInventoryBrokerAuthenticationError(AIInventoryBrokerServiceError):
    """RPC authentication, authorization, freshness, or replay validation failed."""


@dataclass(frozen=True)
class BrokerServiceConfig:
    socket_path: Path
    ledger_path: Path
    state_path: Path
    repository_root: Path
    room_id: str
    candidate_id: str
    bot_hmac_key: bytes
    runner_hmac_key: bytes
    balance_jpy: float = 200_000.0
    slippage_pips: float = 0.0
    financing_pips_per_day: float = 0.0
    leverage: float = 25.0
    experiment_id: str | None = None
    launch_preflight_token_sha256: str | None = None
    allow_test_only_raw_quotes: bool = False
    decision_ledger_path: Path | None = None
    candidate_lifecycle_ledger_path: Path | None = None
    producer_receipt_path: Path | None = None
    mode: str = "ACTIVE"
    drain_authorization_path: Path | None = None
    original_ceiling_minutes: int | None = None
    _test_only_capability: object | None = None


class DojoAIInventoryEntryClient:
    """Bot-facing minimal capability; it cannot close, cancel, quote, or decide."""

    __slots__ = ("_socket_path", "_key")

    def __init__(self, socket_path: Path, hmac_key: bytes) -> None:
        self._socket_path = Path(socket_path)
        self._key = _validate_key(hmac_key, "bot_hmac_key")

    def health(self) -> dict[str, Any]:
        return _rpc_call(self._socket_path, "bot", self._key, "HEALTH", {})

    def account(self) -> dict[str, Any]:
        return _rpc_call(self._socket_path, "bot", self._key, "ACCOUNT", {})

    @property
    def positions(self) -> dict[str, Any]:
        return _rpc_call(self._socket_path, "bot", self._key, "POSITIONS", {})

    @property
    def orders(self) -> dict[str, Any]:
        return _rpc_call(self._socket_path, "bot", self._key, "ORDERS", {})

    def market_order(
        self,
        pair: str,
        side: str,
        units: float,
        tp_pips: float | None = None,
        sl_pips: float | None = None,
        strategy_tag: str | None = None,
        entry_context: dict[str, Any] | None = None,
        *,
        ai_admission: Mapping[str, Any] | None = None,
    ) -> str:
        return str(
            _rpc_call(
                self._socket_path,
                "bot",
                self._key,
                "ENTRY_MARKET",
                _entry_args(
                    pair,
                    side,
                    units,
                    None,
                    tp_pips,
                    sl_pips,
                    strategy_tag,
                    entry_context,
                    ai_admission,
                ),
            )
        )

    def limit_order(
        self,
        pair: str,
        side: str,
        units: float,
        price: float,
        tp_pips: float | None = None,
        sl_pips: float | None = None,
        strategy_tag: str | None = None,
        entry_context: dict[str, Any] | None = None,
        *,
        ai_admission: Mapping[str, Any] | None = None,
    ) -> str:
        return str(
            _rpc_call(
                self._socket_path,
                "bot",
                self._key,
                "ENTRY_LIMIT",
                _entry_args(
                    pair,
                    side,
                    units,
                    price,
                    tp_pips,
                    sl_pips,
                    strategy_tag,
                    entry_context,
                    ai_admission,
                ),
            )
        )

    def stop_order(
        self,
        pair: str,
        side: str,
        units: float,
        price: float,
        tp_pips: float | None = None,
        sl_pips: float | None = None,
        strategy_tag: str | None = None,
        entry_context: dict[str, Any] | None = None,
        *,
        ai_admission: Mapping[str, Any] | None = None,
    ) -> str:
        return str(
            _rpc_call(
                self._socket_path,
                "bot",
                self._key,
                "ENTRY_STOP",
                _entry_args(
                    pair,
                    side,
                    units,
                    price,
                    tp_pips,
                    sl_pips,
                    strategy_tag,
                    entry_context,
                    ai_admission,
                ),
            )
        )


class DojoAIInventoryRunnerClient:
    """Runner-only quote and signed AI-decision capability."""

    __slots__ = ("_socket_path", "_key")

    def __init__(self, socket_path: Path, hmac_key: bytes) -> None:
        self._socket_path = Path(socket_path)
        self._key = _validate_key(hmac_key, "runner_hmac_key")

    def health(self) -> dict[str, Any]:
        return _rpc_call(self._socket_path, "runner", self._key, "HEALTH", {})

    def account(self) -> dict[str, Any]:
        result = _rpc_call(self._socket_path, "runner", self._key, "ACCOUNT", {})
        if not isinstance(result, dict):
            raise AIInventoryBrokerServiceError("broker returned invalid account")
        return result

    @property
    def positions(self) -> dict[str, Any]:
        result = _rpc_call(self._socket_path, "runner", self._key, "POSITIONS", {})
        if not isinstance(result, dict):
            raise AIInventoryBrokerServiceError("broker returned invalid positions")
        return result

    @property
    def orders(self) -> dict[str, Any]:
        result = _rpc_call(self._socket_path, "runner", self._key, "ORDERS", {})
        if not isinstance(result, dict):
            raise AIInventoryBrokerServiceError("broker returned invalid orders")
        return result

    @property
    def quotes(self) -> dict[str, list[Any]]:
        result = _rpc_call(self._socket_path, "runner", self._key, "QUOTES", {})
        if not isinstance(result, dict):
            raise AIInventoryBrokerServiceError("broker returned invalid quotes")
        return result

    @property
    def quote_provenance(self) -> dict[str, dict[str, Any]]:
        result = _rpc_call(
            self._socket_path, "runner", self._key, "QUOTE_PROVENANCE", {}
        )
        if not isinstance(result, dict):
            raise AIInventoryBrokerServiceError(
                "broker returned invalid quote provenance"
            )
        return result

    def decision_status(self, decision_sha256: str) -> dict[str, Any]:
        result = _rpc_call(
            self._socket_path,
            "runner",
            self._key,
            "DECISION_STATUS",
            {"decision_sha256": decision_sha256},
        )
        if not isinstance(result, dict):
            raise AIInventoryBrokerServiceError(
                "broker returned invalid decision status"
            )
        return result

    def apply_quote(
        self,
        pair: str,
        bid: float,
        ask: float,
        ts: str,
        *,
        source_sha256: str | None = None,
        acquisition_receipt_sha256: str | None = None,
    ) -> list[dict[str, Any]]:
        """Apply a caller-supplied quote in an explicitly test-only service."""

        result = _rpc_call(
            self._socket_path,
            "runner",
            self._key,
            "APPLY_QUOTE",
            {
                "pair": pair,
                "bid": bid,
                "ask": ask,
                "ts": ts,
                "source_sha256": source_sha256,
                "acquisition_receipt_sha256": acquisition_receipt_sha256,
            },
        )
        if not isinstance(result, list):
            raise AIInventoryBrokerServiceError("broker returned invalid quote result")
        return result

    def apply_captured_quote(
        self, capture_receipt_sha256: str
    ) -> list[dict[str, Any]]:
        """Apply only the quote reconstructed from one signed capture receipt."""

        result = _rpc_call(
            self._socket_path,
            "runner",
            self._key,
            "APPLY_CAPTURED_QUOTE",
            {"capture_receipt_sha256": capture_receipt_sha256},
        )
        if not isinstance(result, list):
            raise AIInventoryBrokerServiceError(
                "broker returned invalid captured quote result"
            )
        return result

    def apply_drain_quote(
        self, drain_quote_receipt_sha256: str
    ) -> list[dict[str, Any]]:
        """Apply one signed post-window quote through the drain-only surface."""

        result = _rpc_call(
            self._socket_path,
            "runner",
            self._key,
            "APPLY_DRAIN_QUOTE",
            {"drain_quote_receipt_sha256": drain_quote_receipt_sha256},
        )
        if not isinstance(result, list):
            raise AIInventoryBrokerServiceError(
                "broker returned invalid drain quote result"
            )
        return result

    def apply_ai_decision(
        self, decision: Mapping[str, Any], runtime_evidence: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = _rpc_call(
            self._socket_path,
            "runner",
            self._key,
            "APPLY_AI_DECISION",
            {"decision": decision, "runtime_evidence": runtime_evidence},
        )
        if not isinstance(result, dict):
            raise AIInventoryBrokerServiceError(
                "broker returned invalid decision receipt"
            )
        return result

    def shutdown(self) -> None:
        _rpc_call(self._socket_path, "runner", self._key, "SHUTDOWN", {})


def serve_ai_inventory_broker(config: BrokerServiceConfig) -> None:
    """Run the single-owner blocking Unix-domain broker service."""

    owner = _BrokerOwner(config)
    owner.serve()


def derive_broker_socket_path(ledger_path: Path) -> Path:
    """Return the bounded, room-specific Unix socket path.

    macOS limits ``sockaddr_un`` paths to roughly one hundred bytes, while the
    content-addressed room path is intentionally much longer.  The service
    therefore uses a deterministic short path in the sticky temporary
    directory, includes the effective user id, and validates it against the
    absolute ledger identity.
    """

    if not isinstance(ledger_path, Path) or not ledger_path.is_absolute():
        raise AIInventoryBrokerServiceError("ledger_path must be absolute")
    digest = hashlib.sha256(str(ledger_path).encode()).hexdigest()[:32]
    return Path("/tmp") / f"qrai-{os.getuid()}-{digest}.sock"


class _BrokerOwner:
    def __init__(self, config: BrokerServiceConfig) -> None:
        self.config = _validate_config(config)
        self._bot_key = self.config.bot_hmac_key
        self._runner_key = self.config.runner_hmac_key
        self._seen_nonces: set[tuple[str, str]] = set()
        self._serial_lock = threading.Lock()
        self._shutdown = False
        self.broker = VirtualBroker(
            self.config.ledger_path,
            balance_jpy=self.config.balance_jpy,
            fast_ledger=False,
            slippage_pips=self.config.slippage_pips,
            financing_pips_per_day=self.config.financing_pips_per_day,
            leverage=self.config.leverage,
        )
        self.broker._ai_quote_provenance = {}  # type: ignore[attr-defined]
        if self.config.state_path.exists():
            reconciled = _restore_broker_state(
                self.broker,
                self.config.state_path,
                self._runner_key,
                decision_ledger_path=self.config.decision_ledger_path,
            )
            if reconciled:
                _write_broker_state(
                    self.broker, self.config.state_path, self._runner_key
                )
            if _quote_apply_wal_path(self.config.state_path).exists():
                _clear_quote_apply_wal(self.config.state_path)
        elif self.config.ledger_path.stat().st_size != 0:
            raise AIInventoryAdmissionIntegrityError(
                "nonempty broker ledger has no exact restart checkpoint"
            )
        elif _quote_apply_wal_path(self.config.state_path).exists():
            raise AIInventoryAdmissionIntegrityError(
                "captured-quote WAL has no authenticated broker checkpoint"
            )
        else:
            _write_broker_state(self.broker, self.config.state_path, self._runner_key)
        self.controller = _BrokerOwnedAdmissionController(
            self.broker,
            room_id=self.config.room_id,
            candidate_id=self.config.candidate_id,
        )

    def serve(self) -> None:
        path = self.config.socket_path
        if path.exists():
            if not stat.S_ISSOCK(path.lstat().st_mode):
                raise AIInventoryBrokerServiceError(
                    "refusing to replace a non-socket service path"
                )
            if path.lstat().st_uid != os.geteuid():
                raise AIInventoryBrokerServiceError(
                    "refusing to replace another user's service socket"
                )
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(0.2)
                probe.connect(str(path))
            except OSError:
                pass
            else:
                raise AIInventoryBrokerServiceError(
                    "another broker owner is already listening"
                )
            finally:
                probe.close()
            path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(path))
            os.chmod(path, 0o600)
            server.listen(8)
            server.settimeout(1.0)
            while not self._shutdown:
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    continue
                with connection:
                    self._serve_connection(connection)
        finally:
            server.close()
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            self.broker._handle.close()

    def _serve_connection(self, connection: socket.socket) -> None:
        role = "runner"
        key = self._runner_key
        try:
            request = _read_frame(connection)
            role = request.get("role")
            key = self._key_for_role(role)
            self._authenticate(request, key)
            with self._serial_lock:
                result = self._dispatch(role, request["command"], request["args"])
            response = {
                "protocol": RPC_PROTOCOL,
                "request_id": request["request_id"],
                "ok": True,
                "result": result,
            }
        except Exception as exc:
            response = {
                "protocol": RPC_PROTOCOL,
                "request_id": (
                    request.get("request_id", "unknown")
                    if "request" in locals() and isinstance(request, dict)
                    else "unknown"
                ),
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1_000],
            }
        response["mac"] = _mac(key, response)
        _write_frame(connection, response)

    def _key_for_role(self, role: object) -> bytes:
        if role == "bot":
            return self._bot_key
        if role == "runner":
            return self._runner_key
        raise AIInventoryBrokerAuthenticationError("unknown RPC role")

    def _authenticate(self, request: Mapping[str, Any], key: bytes) -> None:
        required = {
            "protocol",
            "role",
            "request_id",
            "nonce",
            "sent_at_utc",
            "command",
            "args",
            "mac",
        }
        if set(request) != required or request.get("protocol") != RPC_PROTOCOL:
            raise AIInventoryBrokerAuthenticationError("invalid RPC schema")
        if not isinstance(request.get("args"), dict):
            raise AIInventoryBrokerAuthenticationError("RPC args must be an object")
        supplied = request.get("mac")
        if not isinstance(supplied, str) or not hmac.compare_digest(
            supplied, _mac(key, {k: request[k] for k in request if k != "mac"})
        ):
            raise AIInventoryBrokerAuthenticationError("invalid RPC MAC")
        sent = _parse_utc(request.get("sent_at_utc"))
        if abs((_utc_now() - sent).total_seconds()) > MAX_CLOCK_SKEW_SECONDS:
            raise AIInventoryBrokerAuthenticationError("stale RPC request")
        nonce = request.get("nonce")
        request_id = request.get("request_id")
        if not isinstance(nonce, str) or not _SHA256_RE.fullmatch(nonce):
            raise AIInventoryBrokerAuthenticationError("invalid RPC nonce")
        if not isinstance(request_id, str) or not _SHA256_RE.fullmatch(request_id):
            raise AIInventoryBrokerAuthenticationError("invalid RPC request id")
        replay_key = (str(request["role"]), nonce)
        if replay_key in self._seen_nonces:
            raise AIInventoryBrokerAuthenticationError("replayed RPC nonce")
        if len(self._seen_nonces) >= MAX_NONCES:
            raise AIInventoryBrokerAuthenticationError(
                "nonce capacity reached; rotate the broker service"
            )
        self._seen_nonces.add(replay_key)

    def _dispatch(self, role: str, command: str, args: dict[str, Any]) -> Any:
        if self.config.mode == "DRAIN_ONLY":
            allowed = (
                _DRAIN_BOT_COMMANDS
                if role == "bot"
                else _DRAIN_RUNNER_COMMANDS
            )
        else:
            allowed = _BOT_COMMANDS if role == "bot" else _RUNNER_COMMANDS
        if command not in allowed:
            raise AIInventoryBrokerAuthenticationError(
                f"{role} role is not allowed to invoke {command}"
            )
        if command == "HEALTH":
            return {
                "status": "READY",
                "room_id": self.config.room_id,
                "candidate_id": self.config.candidate_id,
                "mode": self.config.mode,
                "ledger_sha256": self.broker._prev_sha,
                "broker_owner_pid": os.getpid(),
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
        if command == "ACCOUNT":
            return self.broker.account()
        if command == "POSITIONS":
            return {key: asdict(value) for key, value in self.broker.positions.items()}
        if command == "ORDERS":
            return {key: asdict(value) for key, value in self.broker.orders.items()}
        if command == "QUOTES":
            return {
                pair: [float(bid), float(ask), ts]
                for pair, (bid, ask, ts) in sorted(self.broker.last_quotes.items())
            }
        if command == "QUOTE_PROVENANCE":
            return _json_copy(
                getattr(self.broker, "_ai_quote_provenance", {})
            )
        if command == "DECISION_STATUS":
            return self._decision_status(args)
        if command == "SHUTDOWN":
            self._shutdown = True
            return {"status": "STOPPING"}
        if command in {"ENTRY_MARKET", "ENTRY_LIMIT", "ENTRY_STOP"}:
            result = self._entry(command, args)
        elif command == "APPLY_QUOTE":
            if self.config.allow_test_only_raw_quotes is not True:
                raise AIInventoryBrokerAuthenticationError(
                    "raw quote RPC is test-only; signed capture receipt is required"
                )
            result = self._apply_quote(args)
        elif command == "APPLY_CAPTURED_QUOTE":
            result = self._apply_captured_quote(args)
        elif command == "APPLY_DRAIN_QUOTE":
            result = self._apply_drain_quote(args)
        elif command == "APPLY_AI_DECISION":
            result = self._apply_ai_decision(args)
        else:
            raise AIInventoryBrokerServiceError("unhandled RPC command")
        if command in _MUTATING_COMMANDS:
            _write_broker_state(self.broker, self.config.state_path, self._runner_key)
            if command in {"APPLY_CAPTURED_QUOTE", "APPLY_DRAIN_QUOTE"}:
                _clear_quote_apply_wal(self.config.state_path)
        return result

    def _entry(self, command: str, args: dict[str, Any]) -> str:
        _require_market_open(_utc_now(), "broker clock")
        self._verify_current_evidence(args.get("ai_admission"))
        _verify_active_broker_lifecycle(self.config, at_utc=_utc_now())
        common = {
            "pair": args.get("pair"),
            "side": args.get("side"),
            "units": args.get("units"),
            "tp_pips": args.get("tp_pips"),
            "sl_pips": args.get("sl_pips"),
            "strategy_tag": args.get("strategy_tag"),
            "entry_context": args.get("entry_context"),
            "ai_admission": args.get("ai_admission"),
        }
        if command == "ENTRY_MARKET":
            return self.controller.market_order(**common)
        if command == "ENTRY_LIMIT":
            return self.controller.limit_order(price=args.get("price"), **common)
        return self.controller.stop_order(price=args.get("price"), **common)

    def _verify_current_evidence(self, reference: object) -> None:
        if not isinstance(reference, Mapping):
            raise AIInventoryEntryDeniedError("AI admission reference is required")
        if reference.get("contract") != ENTRY_ADMISSION_REFERENCE_CONTRACT:
            raise AIInventoryEntryDeniedError("AI admission reference is invalid")
        now = _utc_now()
        state = build_ai_inventory_admission_state(
            self.config.ledger_path,
            room_id=self.config.room_id,
            candidate_id=self.config.candidate_id,
            as_of_utc=now,
        )
        matches = [
            permit
            for permit in state.available_permits
            if permit.applied_receipt_sha256 == reference.get("applied_receipt_sha256")
            and permit.decision_sha256 == reference.get("decision_sha256")
            and permit.signal_identity_sha256 == reference.get("signal_identity_sha256")
            and permit.room_id == self.config.room_id
            and permit.candidate_id == self.config.candidate_id
        ]
        if len(matches) != 1:
            raise AIInventoryEntryDeniedError("no exact current entry permit")
        permit = matches[0]
        packet_path = (
            self.config.repository_root
            / DEDICATED_EVIDENCE_ROOT
            / f"{permit.evidence_packet_sha256}.json"
        )
        packet = verify_ai_inventory_evidence_packet(
            self.config.repository_root, packet_path
        )
        if packet.get("packet_sha256") != permit.evidence_packet_sha256 or packet.get(
            "entry_signal"
        ) != dict(permit.entry_signal):
            raise AIInventoryEntryDeniedError(
                "entry permit does not match its immutable evidence packet"
            )
        bindings = packet.get("bindings")
        if not isinstance(bindings, dict) or (
            bindings.get("room_id") != self.config.room_id
            or bindings.get("candidate_id") != self.config.candidate_id
        ):
            raise AIInventoryEntryDeniedError("evidence packet scope mismatch")
        quote = packet.get("quote")
        if not isinstance(quote, dict) or quote.get("pair") != permit.pair:
            raise AIInventoryEntryDeniedError("evidence packet quote mismatch")
        current = self.broker.last_quotes.get(permit.pair)
        expected = (quote.get("bid"), quote.get("ask"), quote.get("timestamp_utc"))
        if current != expected:
            raise AIInventoryEntryDeniedError(
                "broker quote advanced or differs from admitted evidence"
            )
        quote_at = _parse_utc(quote.get("timestamp_utc"))
        max_age = quote.get("max_age_seconds")
        if (
            isinstance(max_age, bool)
            or not isinstance(max_age, int)
            or max_age <= 0
            or now < quote_at
            or (now - quote_at).total_seconds() > max_age
            or quote.get("timestamp_utc") != permit.signal_observed_at_utc
        ):
            raise AIInventoryEntryDeniedError("admitted evidence quote is stale")
        _require_market_open(quote_at, "evidence quote")

    def _apply_quote(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        source_sha256 = args.get("source_sha256")
        receipt_sha256 = args.get("acquisition_receipt_sha256")
        if source_sha256 is None:
            source_sha256 = hashlib.sha256(
                _canonical_json(
                    {
                        "pair": args.get("pair"),
                        "bid": args.get("bid"),
                        "ask": args.get("ask"),
                        "timestamp_utc": args.get("ts"),
                    }
                )
            ).hexdigest()
        if receipt_sha256 is None:
            receipt_sha256 = hashlib.sha256(
                f"TEST_ONLY_RAW:{source_sha256}".encode()
            ).hexdigest()
        source_sha256 = _require_sha256(source_sha256, "source_sha256")
        receipt_sha256 = _require_sha256(
            receipt_sha256, "acquisition_receipt_sha256"
        )
        events = self._apply_quote_values(
            pair=args.get("pair"),
            bid=args.get("bid"),
            ask=args.get("ask"),
            ts=args.get("ts"),
        )
        getattr(self.broker, "_ai_quote_provenance")[str(args.get("pair"))] = {
            "pair": args.get("pair"),
            "bid": float(args.get("bid")),
            "ask": float(args.get("ask")),
            "timestamp_utc": args.get("ts"),
            "capture_source_sha256": source_sha256,
            "acquisition_receipt_sha256": receipt_sha256,
            "quote_watermark_sha256": None,
            "test_only_raw_quote": True,
        }
        return events

    def _apply_captured_quote(
        self, args: dict[str, Any]
    ) -> list[dict[str, Any]]:
        receipt_sha256 = _require_sha256(
            args.get("capture_receipt_sha256"),
            "capture_receipt_sha256",
        )
        captured = _load_verified_captured_quote(self.config, receipt_sha256)
        watermark = append_ai_inventory_quote_watermark(
            self.config.ledger_path.parent,
            pair=captured["pair"],
            bid=captured["bid"],
            ask=captured["ask"],
            timestamp_utc=captured["timestamp_utc"],
            slippage_pips_per_fill=self.config.slippage_pips,
            financing_pips_per_day=self.config.financing_pips_per_day,
            acquisition_receipt_sha256=receipt_sha256,
            capture_source_sha256=captured["capture_source_sha256"],
        )
        provenance = {
            "pair": captured["pair"],
            "bid": captured["bid"],
            "ask": captured["ask"],
            "timestamp_utc": captured["timestamp_utc"],
            "capture_source_sha256": captured["capture_source_sha256"],
            "acquisition_receipt_sha256": receipt_sha256,
            "quote_watermark_sha256": watermark["quote_sha256"],
            "test_only_raw_quote": False,
        }
        current = getattr(self.broker, "_ai_quote_provenance").get(
            captured["pair"]
        )
        if current == provenance:
            # A response may be lost after the authenticated state checkpoint
            # commits.  The immutable receipt is therefore an idempotency key.
            return []
        if _quote_apply_wal_path(self.config.state_path).exists():
            raise AIInventoryAdmissionIntegrityError(
                "a captured-quote transaction is already pending recovery"
            )

        plan = self._plan_captured_quote(captured, provenance)
        _write_quote_apply_wal(
            self.config.state_path,
            self._runner_key,
            capture_receipt_sha256=receipt_sha256,
            capture_source_sha256=captured["capture_source_sha256"],
            quote_watermark_sha256=watermark["quote_sha256"],
            quote={
                "pair": captured["pair"],
                "bid": captured["bid"],
                "ask": captured["ask"],
                "timestamp_utc": captured["timestamp_utc"],
                "max_age_seconds": captured["max_age_seconds"],
            },
            pre_state=plan["pre_state"],
            post_state=plan["post_state"],
            expected_ledger_events=plan["expected_ledger_events"],
            result_events=plan["result_events"],
        )
        events = self._apply_quote_values(
            pair=captured["pair"],
            bid=captured["bid"],
            ask=captured["ask"],
            ts=captured["timestamp_utc"],
            max_age_seconds=captured["max_age_seconds"],
        )
        getattr(self.broker, "_ai_quote_provenance")[captured["pair"]] = provenance
        _validate_quote_event_suffix(
            self.config.ledger_path,
            plan["pre_state"]["broker"]["ledger_sha"],
            plan["expected_ledger_events"],
        )
        if events != plan["result_events"]:
            raise AIInventoryAdmissionIntegrityError(
                "captured quote result differs from its durable plan"
            )
        return events

    def _apply_drain_quote(
        self, args: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Apply one signed post-window quote without any entry capability."""

        if self.config.mode != "DRAIN_ONLY":
            raise AIInventoryBrokerAuthenticationError(
                "drain quote RPC requires DRAIN_ONLY mode"
            )
        receipt_sha256 = _require_sha256(
            args.get("drain_quote_receipt_sha256"),
            "drain_quote_receipt_sha256",
        )
        captured = _load_verified_drain_quote(self.config, receipt_sha256)
        provenance = {
            "pair": captured["pair"],
            "bid": captured["bid"],
            "ask": captured["ask"],
            "timestamp_utc": captured["timestamp_utc"],
            "capture_source_sha256": captured["capture_source_sha256"],
            "acquisition_receipt_sha256": receipt_sha256,
            "quote_watermark_sha256": None,
            "test_only_raw_quote": False,
        }
        current = getattr(self.broker, "_ai_quote_provenance").get(
            captured["pair"]
        )
        if current is not None and all(
            current.get(field) == provenance[field]
            for field in provenance
            if field != "quote_watermark_sha256"
        ):
            return []
        _verify_drain_quote_checkpoint(
            self.config,
            self.broker,
            captured["receipt"],
        )
        watermark = append_ai_inventory_quote_watermark(
            self.config.ledger_path.parent,
            pair=captured["pair"],
            bid=captured["bid"],
            ask=captured["ask"],
            timestamp_utc=captured["timestamp_utc"],
            slippage_pips_per_fill=self.config.slippage_pips,
            financing_pips_per_day=self.config.financing_pips_per_day,
            acquisition_receipt_sha256=receipt_sha256,
            capture_source_sha256=captured["capture_source_sha256"],
        )
        provenance["quote_watermark_sha256"] = watermark["quote_sha256"]
        current = getattr(self.broker, "_ai_quote_provenance").get(
            captured["pair"]
        )
        if current == provenance:
            return []
        if _quote_apply_wal_path(self.config.state_path).exists():
            raise AIInventoryAdmissionIntegrityError(
                "a drain-quote transaction is already pending recovery"
            )

        plan = self._plan_drain_quote(
            captured,
            provenance,
            receipt_sha256=receipt_sha256,
        )
        _write_quote_apply_wal(
            self.config.state_path,
            self._runner_key,
            capture_receipt_sha256=receipt_sha256,
            capture_source_sha256=captured["capture_source_sha256"],
            quote_watermark_sha256=watermark["quote_sha256"],
            quote={
                "pair": captured["pair"],
                "bid": captured["bid"],
                "ask": captured["ask"],
                "timestamp_utc": captured["timestamp_utc"],
                "max_age_seconds": captured["max_age_seconds"],
            },
            pre_state=plan["pre_state"],
            post_state=plan["post_state"],
            expected_ledger_events=plan["expected_ledger_events"],
            result_events=plan["result_events"],
        )
        events = self._apply_drain_quote_values(
            pair=captured["pair"],
            bid=captured["bid"],
            ask=captured["ask"],
            ts=captured["timestamp_utc"],
            max_age_seconds=captured["max_age_seconds"],
            receipt_sha256=receipt_sha256,
        )
        getattr(self.broker, "_ai_quote_provenance")[captured["pair"]] = provenance
        _validate_quote_event_suffix(
            self.config.ledger_path,
            plan["pre_state"]["broker"]["ledger_sha"],
            plan["expected_ledger_events"],
        )
        if events != plan["result_events"]:
            raise AIInventoryAdmissionIntegrityError(
                "drain quote result differs from its durable plan"
            )
        return events

    def _plan_captured_quote(
        self,
        captured: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Deterministically plan one quote transition before broker mutation.

        The broker is temporarily evaluated with a non-writing ledger sink,
        then restored to the exact authenticated pre-state.  The resulting WAL
        lets restart validate an already-durable event prefix and append only
        the missing deterministic suffix.
        """

        pre_state = _broker_runtime_state(self.broker)
        expected: list[dict[str, Any]] = []
        had_local_log = "_log" in self.broker.__dict__
        original_log = self.broker._log

        def collect(event: str, payload: dict[str, Any]) -> None:
            expected.append(
                {"event": str(event), "payload": _json_copy(payload)}
            )

        self.broker._log = collect  # type: ignore[method-assign]
        try:
            result = self._apply_quote_values(
                pair=captured["pair"],
                bid=captured["bid"],
                ask=captured["ask"],
                ts=captured["timestamp_utc"],
                max_age_seconds=captured["max_age_seconds"],
            )
            getattr(self.broker, "_ai_quote_provenance")[
                str(captured["pair"])
            ] = dict(provenance)
            post_state = _broker_runtime_state(self.broker)
        finally:
            if had_local_log:
                self.broker._log = original_log  # type: ignore[method-assign]
            else:
                del self.broker.__dict__["_log"]
            self.broker.restore(pre_state["broker"], require_ledger_match=False)
            self.broker.last_quotes = {
                pair: (float(value[0]), float(value[1]), str(value[2]))
                for pair, value in pre_state["last_quotes"].items()
            }
            self.broker._ai_quote_provenance = _json_copy(  # type: ignore[attr-defined]
                pre_state["quote_provenance"]
            )
        if len(expected) > MAX_QUOTE_APPLY_EVENTS:
            raise AIInventoryBrokerServiceError(
                "captured quote exceeds the bounded ledger-event limit"
            )
        return {
            "pre_state": pre_state,
            "post_state": post_state,
            "expected_ledger_events": expected,
            "result_events": _json_list_copy(result),
        }

    def _plan_drain_quote(
        self,
        captured: Mapping[str, Any],
        provenance: Mapping[str, Any],
        *,
        receipt_sha256: str,
    ) -> dict[str, Any]:
        """Plan a drain transition so crash recovery remains exact-once."""

        pre_state = _broker_runtime_state(self.broker)
        expected: list[dict[str, Any]] = []
        had_local_log = "_log" in self.broker.__dict__
        original_log = self.broker._log

        def collect(event: str, payload: dict[str, Any]) -> None:
            expected.append(
                {"event": str(event), "payload": _json_copy(payload)}
            )

        self.broker._log = collect  # type: ignore[method-assign]
        try:
            result = self._apply_drain_quote_values(
                pair=captured["pair"],
                bid=captured["bid"],
                ask=captured["ask"],
                ts=captured["timestamp_utc"],
                max_age_seconds=captured["max_age_seconds"],
                receipt_sha256=receipt_sha256,
            )
            getattr(self.broker, "_ai_quote_provenance")[
                str(captured["pair"])
            ] = dict(provenance)
            post_state = _broker_runtime_state(self.broker)
        finally:
            if had_local_log:
                self.broker._log = original_log  # type: ignore[method-assign]
            else:
                del self.broker.__dict__["_log"]
            self.broker.restore(pre_state["broker"], require_ledger_match=False)
            self.broker.last_quotes = {
                pair: (float(value[0]), float(value[1]), str(value[2]))
                for pair, value in pre_state["last_quotes"].items()
            }
            self.broker._ai_quote_provenance = _json_copy(  # type: ignore[attr-defined]
                pre_state["quote_provenance"]
            )
        if len(expected) > MAX_QUOTE_APPLY_EVENTS:
            raise AIInventoryBrokerServiceError(
                "drain quote exceeds the bounded ledger-event limit"
            )
        return {
            "pre_state": pre_state,
            "post_state": post_state,
            "expected_ledger_events": expected,
            "result_events": _json_list_copy(result),
        }

    def _apply_drain_quote_values(
        self,
        *,
        pair: object,
        bid: object,
        ask: object,
        ts: object,
        max_age_seconds: object,
        receipt_sha256: str,
    ) -> list[dict[str, Any]]:
        """Resolve only pre-existing inventory using its original boundaries."""

        bid = _finite_positive(bid, "bid")
        ask = _finite_positive(ask, "ask")
        if (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, int)
            or max_age_seconds <= 0
        ):
            raise AIInventoryBrokerServiceError(
                "drain quote max_age_seconds must be a positive integer"
            )
        if not isinstance(pair, str) or ask < bid:
            raise AIInventoryBrokerServiceError("invalid drain quote")
        quote_at = _parse_utc(ts)
        now = _utc_now()
        _require_market_open(now, "drain broker clock")
        _require_market_open(quote_at, "drain quote")
        if (
            quote_at > now
            or (now - quote_at).total_seconds() > min(max_age_seconds, 180)
        ):
            raise AIInventoryBrokerServiceError(
                "drain quote is stale or future-dated"
            )
        ceiling = self.config.original_ceiling_minutes
        if (
            isinstance(ceiling, bool)
            or not isinstance(ceiling, int)
            or ceiling <= 0
        ):
            raise AIInventoryBrokerServiceError(
                "DRAIN_ONLY original ceiling is invalid"
            )

        events: list[dict[str, Any]] = []
        for order_id in sorted(self.broker.orders):
            order = self.broker.orders.pop(order_id)
            payload = {
                "order_id": order.order_id,
                "pair": order.pair,
                "side": order.side,
                "strategy_tag": order.strategy_tag,
                "resolution": "PENDING_ORDER_CANCEL_AT_ENTRY_STOP",
                "drain_quote_receipt_sha256": receipt_sha256,
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
            self.broker._log("PENDING_ORDER_CANCEL_AT_ENTRY_STOP", payload)
            events.append(
                {"event": "PENDING_ORDER_CANCEL_AT_ENTRY_STOP", **payload}
            )

        broker_events = self.broker.on_quote(pair, bid, ask, str(ts))
        if any(
            event.get("event") not in {"EXIT_TP", "EXIT_SL", "MARGIN_CLOSEOUT"}
            for event in broker_events
        ):
            raise AIInventoryAdmissionIntegrityError(
                "drain quote attempted a non-drain broker transition"
            )
        events.extend(broker_events)

        for trade_id in sorted(self.broker.positions):
            position = self.broker.positions[trade_id]
            if position.pair != pair:
                raise AIInventoryAdmissionIntegrityError(
                    "drain inventory pair differs from its signed quote"
                )
            opened_at = _parse_utc(position.opened_ts)
            if quote_at < opened_at + timedelta(minutes=ceiling):
                continue
            price = bid if position.side == "LONG" else ask
            difference = (
                price - position.entry_price
                if position.side == "LONG"
                else position.entry_price - price
            )
            realized = (
                difference
                * position.units
                * self.broker._jpy_per_quote_unit(position.pair)
            )
            realized -= self.broker._financing_jpy(position, str(ts))
            self.broker.balance_jpy += realized
            del self.broker.positions[trade_id]
            payload = {
                "trade_id": trade_id,
                "pair": position.pair,
                "side": position.side,
                "units": position.units,
                "price": price,
                "pl_jpy": round(realized, 2),
                "strategy_tag": position.strategy_tag,
                "entry_context_sha256": position.entry_context_sha256,
                "opened_ts": position.opened_ts,
                "original_ceiling_minutes": ceiling,
                "resolution": "ORIGINAL_CEILING",
                "drain_quote_receipt_sha256": receipt_sha256,
                "quote": {"bid": bid, "ask": ask, "ts": ts},
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
            self.broker._log("EXIT_ORIGINAL_CEILING", payload)
            events.append({"event": "EXIT_ORIGINAL_CEILING", **payload})
        return events

    def _apply_quote_values(
        self,
        *,
        pair: object,
        bid: object,
        ask: object,
        ts: object,
        max_age_seconds: object = 180,
    ) -> list[dict[str, Any]]:
        bid = _finite_positive(bid, "bid")
        ask = _finite_positive(ask, "ask")
        if (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, int)
            or max_age_seconds <= 0
        ):
            raise AIInventoryBrokerServiceError(
                "quote max_age_seconds must be a positive integer"
            )
        effective_max_age_seconds = min(max_age_seconds, 180)
        quote_at = _parse_utc(ts)
        now = _utc_now()
        _require_market_open(now, "broker clock")
        _require_market_open(quote_at, "quote")
        if (
            quote_at > now
            or (now - quote_at).total_seconds() > effective_max_age_seconds
        ):
            raise AIInventoryBrokerServiceError("quote is stale or future-dated")
        if not isinstance(pair, str) or ask < bid:
            raise AIInventoryBrokerServiceError("invalid quote")

        state = build_ai_inventory_admission_state(
            self.config.ledger_path,
            room_id=self.config.room_id,
            candidate_id=self.config.candidate_id,
            as_of_utc=now,
        )
        suppressed = {
            order_id: order
            for order_id, order in self.broker.orders.items()
            if order.pair == pair
            and (order.pair, str(order.strategy_tag)) in state.blocked_scopes
        }
        triggered = [
            order
            for order in suppressed.values()
            if _would_fill(order, bid=bid, ask=ask)
        ]
        for order_id in suppressed:
            del self.broker.orders[order_id]
        try:
            events = self.broker.on_quote(pair, bid, ask, str(ts))
            for order in triggered:
                payload = {
                    "order_id": order.order_id,
                    "pair": order.pair,
                    "side": order.side,
                    "strategy_tag": order.strategy_tag,
                    "order_kind": order.kind,
                    "limit_price": order.limit_price,
                    "quote": {"bid": bid, "ask": ask, "ts": ts},
                    "reason": "PERSISTENT_BLOCK_NEW_SCOPE",
                    "paper_only": True,
                    "order_authority": "NONE",
                    "live_permission": False,
                }
                self.broker._log(PENDING_FILL_REJECTED_EVENT, payload)
                events.append({"event": PENDING_FILL_REJECTED_EVENT, **payload})
        finally:
            self.broker.orders.update(suppressed)
        return events

    def _apply_ai_decision(self, args: dict[str, Any]) -> dict[str, Any]:
        _require_market_open(_utc_now(), "broker clock")
        if self.config.decision_ledger_path is None:
            raise AIInventoryBrokerServiceError(
                "AI decision ledgers are not configured"
            )
        decision = args.get("decision")
        if not isinstance(decision, Mapping):
            raise AIInventoryBrokerServiceError("AI decision is invalid")
        producer_receipt_path = self.config.producer_receipt_path
        if producer_receipt_path is None:
            binding = decision.get("ai_decision_binding")
            if not isinstance(binding, Mapping):
                raise AIInventoryBrokerServiceError(
                    "AI decision producer binding is invalid"
                )
            receipt_sha256 = binding.get("producer_receipt_sha256")
            if not isinstance(receipt_sha256, str) or not _SHA256_RE.fullmatch(
                receipt_sha256
            ):
                raise AIInventoryBrokerServiceError(
                    "AI producer receipt digest is invalid"
                )
            producer_receipt_path = (
                self.config.ledger_path.parent
                / "producer_receipts"
                / f"{receipt_sha256}.json"
            )
        runtime = args.get("runtime_evidence")
        if not isinstance(runtime, dict):
            raise AIInventoryBrokerServiceError("runtime evidence is invalid")
        runtime = dict(runtime)
        dedicated = runtime.get("dedicated_root")
        if not isinstance(dedicated, str):
            raise AIInventoryBrokerServiceError(
                "runtime dedicated_root must be an absolute serialized path"
            )
        runtime["dedicated_root"] = Path(dedicated)
        return consume_inventory_decision(
            decision,
            self.broker,
            runtime,
            decision_ledger_path=self.config.decision_ledger_path,  # type: ignore[arg-type]
            candidate_lifecycle_ledger_path=self.config.candidate_lifecycle_ledger_path,
            producer_receipt_path=producer_receipt_path,
            repository_root=self.config.repository_root,
        )

    def _decision_status(self, args: Mapping[str, Any]) -> dict[str, Any]:
        decision_sha256 = args.get("decision_sha256")
        if not isinstance(decision_sha256, str) or not _SHA256_RE.fullmatch(
            decision_sha256
        ):
            raise AIInventoryBrokerServiceError("decision digest is invalid")
        rows = _read_validated_broker_rows(self.config.ledger_path)
        reservations = [
            row
            for row in rows
            if row["event"] == "AI_INVENTORY_ACTION_RESERVED"
            and row["payload"].get("decision_sha256") == decision_sha256
        ]
        applied = [
            row
            for row in rows
            if row["event"] == "AI_INVENTORY_ACTION_APPLIED"
            and row["payload"].get("decision_sha256") == decision_sha256
        ]
        if len(reservations) > 1 or len(applied) > 1:
            raise AIInventoryAdmissionIntegrityError(
                "decision has duplicate broker lifecycle rows"
            )
        if applied:
            if (
                not reservations
                or applied[0]["payload"].get("reservation_sha256")
                != reservations[0]["sha"]
            ):
                raise AIInventoryAdmissionIntegrityError(
                    "decision applied receipt is detached from reservation"
                )
            return {
                "status": "APPLIED",
                "receipt": {
                    **applied[0]["payload"],
                    "applied_receipt_sha256": applied[0]["sha"],
                    "broker_ledger_terminal_sha256": rows[-1]["sha"],
                },
            }
        if reservations:
            return {
                "status": "RESERVED",
                "reservation_sha256": reservations[0]["sha"],
            }
        return {"status": "NONE"}


def _entry_args(
    pair: str,
    side: str,
    units: float,
    price: float | None,
    tp_pips: float | None,
    sl_pips: float | None,
    strategy_tag: str | None,
    entry_context: dict[str, Any] | None,
    ai_admission: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "pair": pair,
        "side": side,
        "units": units,
        "price": price,
        "tp_pips": tp_pips,
        "sl_pips": sl_pips,
        "strategy_tag": strategy_tag,
        "entry_context": entry_context,
        "ai_admission": ai_admission,
    }


def _rpc_call(
    socket_path: Path, role: str, key: bytes, command: str, args: Mapping[str, Any]
) -> Any:
    nonce = os.urandom(32).hex()
    sent_at = _canonical_utc(_utc_now())
    request = {
        "protocol": RPC_PROTOCOL,
        "role": role,
        "request_id": hashlib.sha256(
            f"{role}:{command}:{nonce}:{sent_at}".encode()
        ).hexdigest(),
        "nonce": nonce,
        "sent_at_utc": sent_at,
        "command": command,
        "args": _json_copy(args),
    }
    request["mac"] = _mac(key, request)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(5.0)
        connection.connect(str(socket_path))
        _write_frame(connection, request)
        response = _read_frame(connection)
    except OSError as exc:
        raise AIInventoryBrokerServiceError("broker service is unavailable") from exc
    finally:
        connection.close()
    supplied = response.get("mac")
    if not isinstance(supplied, str) or not hmac.compare_digest(
        supplied, _mac(key, {k: response[k] for k in response if k != "mac"})
    ):
        raise AIInventoryBrokerAuthenticationError("invalid broker response MAC")
    if (
        response.get("protocol") != RPC_PROTOCOL
        or response.get("request_id") != request["request_id"]
    ):
        raise AIInventoryBrokerAuthenticationError("mismatched broker response")
    if response.get("ok") is not True:
        raise AIInventoryBrokerServiceError(
            f"{response.get('error_type', 'BrokerError')}: {response.get('error', '')}"
        )
    return response.get("result")


def _read_frame(connection: socket.socket) -> dict[str, Any]:
    data = bytearray()
    while len(data) <= MAX_RPC_BYTES:
        chunk = connection.recv(min(64 * 1024, MAX_RPC_BYTES + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if b"\n" in chunk:
            break
    if not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise AIInventoryBrokerAuthenticationError("invalid RPC frame")
    if len(data) > MAX_RPC_BYTES:
        raise AIInventoryBrokerAuthenticationError("RPC frame too large")
    try:
        value = json.loads(
            data[:-1],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AIInventoryBrokerAuthenticationError("invalid RPC JSON") from exc
    if not isinstance(value, dict):
        raise AIInventoryBrokerAuthenticationError("RPC frame must be an object")
    return value


def _write_frame(connection: socket.socket, value: Mapping[str, Any]) -> None:
    raw = _canonical_json(value) + b"\n"
    if len(raw) > MAX_RPC_BYTES:
        raise AIInventoryBrokerServiceError("RPC response exceeds byte limit")
    connection.sendall(raw)


def _mac(key: bytes, value: Mapping[str, Any]) -> str:
    return hmac.new(key, _canonical_json(value), hashlib.sha256).hexdigest()


def _write_broker_state(broker: VirtualBroker, path: Path, key: bytes) -> None:
    body = {"contract": BROKER_STATE_CONTRACT, **_broker_runtime_state(broker)}
    record = {**body, "mac": _mac(key, body)}
    raw = _canonical_json(record) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _broker_runtime_state(broker: VirtualBroker) -> dict[str, Any]:
    return {
        "broker": _json_copy(broker.snapshot()),
        "last_quotes": {
            pair: [float(bid), float(ask), ts]
            for pair, (bid, ask, ts) in sorted(broker.last_quotes.items())
        },
        "quote_provenance": _json_copy(
            getattr(broker, "_ai_quote_provenance", {})
        ),
    }


def _quote_apply_wal_path(state_path: Path) -> Path:
    return state_path.with_name(QUOTE_APPLY_WAL_NAME)


def _write_quote_apply_wal(
    state_path: Path,
    key: bytes,
    *,
    capture_receipt_sha256: str,
    capture_source_sha256: str,
    quote_watermark_sha256: str,
    quote: Mapping[str, Any],
    pre_state: Mapping[str, Any],
    post_state: Mapping[str, Any],
    expected_ledger_events: list[dict[str, Any]],
    result_events: list[dict[str, Any]],
) -> None:
    path = _quote_apply_wal_path(state_path)
    if path.exists():
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote WAL already exists"
        )
    checkpoint_tip = pre_state.get("broker", {}).get("ledger_sha")
    body = {
        "contract": QUOTE_APPLY_WAL_CONTRACT,
        "capture_receipt_sha256": _require_sha256(
            capture_receipt_sha256, "capture_receipt_sha256"
        ),
        "capture_source_sha256": _require_sha256(
            capture_source_sha256, "capture_source_sha256"
        ),
        "quote_watermark_sha256": _require_sha256(
            quote_watermark_sha256, "quote_watermark_sha256"
        ),
        "checkpoint_ledger_sha256": _require_sha256(
            checkpoint_tip, "checkpoint_ledger_sha256"
        ),
        "quote": _json_copy(quote),
        "pre_state": _json_copy(pre_state),
        "post_state": _json_copy(post_state),
        "expected_ledger_events": _json_list_copy(expected_ledger_events),
        "result_events": _json_list_copy(result_events),
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    record = {**body, "mac": _mac(key, body)}
    raw = _canonical_json(record) + b"\n"
    if len(raw) > MAX_QUOTE_APPLY_WAL_BYTES:
        raise AIInventoryBrokerServiceError(
            "captured-quote WAL exceeds the byte limit"
        )
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _read_quote_apply_wal(state_path: Path, key: bytes) -> dict[str, Any]:
    path = _quote_apply_wal_path(state_path)
    raw = _read_regular_nofollow(
        path, MAX_QUOTE_APPLY_WAL_BYTES, "captured-quote WAL"
    )
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote WAL framing is invalid"
        )
    try:
        record = json.loads(
            raw[:-1],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote WAL cannot be decoded"
        ) from exc
    expected_keys = {
        "contract",
        "capture_receipt_sha256",
        "capture_source_sha256",
        "quote_watermark_sha256",
        "checkpoint_ledger_sha256",
        "quote",
        "pre_state",
        "post_state",
        "expected_ledger_events",
        "result_events",
        "paper_only",
        "order_authority",
        "live_permission",
        "mac",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote WAL schema is invalid"
        )
    body = {name: record[name] for name in expected_keys if name != "mac"}
    supplied_mac = record.get("mac")
    if (
        record.get("contract") != QUOTE_APPLY_WAL_CONTRACT
        or record.get("paper_only") is not True
        or record.get("order_authority") != "NONE"
        or record.get("live_permission") is not False
        or not isinstance(supplied_mac, str)
        or not hmac.compare_digest(supplied_mac, _mac(key, body))
    ):
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote WAL authentication failed"
        )
    for field in (
        "capture_receipt_sha256",
        "capture_source_sha256",
        "quote_watermark_sha256",
        "checkpoint_ledger_sha256",
    ):
        if not isinstance(record.get(field), str) or _SHA256_RE.fullmatch(
            record[field]
        ) is None:
            raise AIInventoryAdmissionIntegrityError(
                "captured-quote WAL digest is invalid"
            )
    expected_events = record.get("expected_ledger_events")
    result_events = record.get("result_events")
    if (
        not isinstance(expected_events, list)
        or len(expected_events) > MAX_QUOTE_APPLY_EVENTS
        or not isinstance(result_events, list)
    ):
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote WAL event list is invalid"
        )
    for row in expected_events:
        if (
            not isinstance(row, dict)
            or set(row) != {"event", "payload"}
            or not isinstance(row.get("event"), str)
            or not isinstance(row.get("payload"), dict)
        ):
            raise AIInventoryAdmissionIntegrityError(
                "captured-quote WAL event row is invalid"
            )
    for field in ("quote", "pre_state", "post_state"):
        if not isinstance(record.get(field), dict):
            raise AIInventoryAdmissionIntegrityError(
                f"captured-quote WAL {field} is invalid"
            )
    _validate_quote_wal_binding(state_path.parent, record)
    return record


def _validate_quote_wal_binding(
    room_root: Path, wal: Mapping[str, Any]
) -> None:
    ledger_path = room_root / QUOTE_WATERMARK_LEDGER_NAME
    before = _read_regular_nofollow(
        ledger_path,
        MAX_QUOTE_WATERMARK_LEDGER_BYTES,
        "captured-quote watermark ledger",
    )
    try:
        validation = validate_ai_inventory_quote_watermarks(ledger_path)
    except Exception as exc:
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote watermark chain is invalid"
        ) from exc
    if validation.get("valid") is not True:
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote watermark chain is invalid"
        )
    after = _read_regular_nofollow(
        ledger_path,
        MAX_QUOTE_WATERMARK_LEDGER_BYTES,
        "captured-quote watermark ledger",
    )
    if before != after:
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote watermark changed during recovery validation"
        )
    try:
        rows = [
            json.loads(
                raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            for raw in after.splitlines()
        ]
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote watermark cannot be decoded"
        ) from exc
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("quote_sha256") == wal.get("quote_watermark_sha256")
    ]
    quote = wal.get("quote")
    if not isinstance(quote, dict) or set(quote) != {
        "pair",
        "bid",
        "ask",
        "timestamp_utc",
        "max_age_seconds",
    }:
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote WAL quote is invalid"
        )
    if (
        len(matches) != 1
        or matches[0].get("pair") != quote.get("pair")
        or matches[0].get("bid") != quote.get("bid")
        or matches[0].get("ask") != quote.get("ask")
        or matches[0].get("timestamp_utc") != quote.get("timestamp_utc")
        or matches[0].get("capture_source_sha256")
        != wal.get("capture_source_sha256")
        or matches[0].get("acquisition_receipt_sha256")
        != wal.get("capture_receipt_sha256")
    ):
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote WAL is detached from its durable watermark"
        )


def _clear_quote_apply_wal(state_path: Path) -> None:
    path = _quote_apply_wal_path(state_path)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _recover_quote_apply_wal(
    broker: VirtualBroker,
    *,
    state_record: Mapping[str, Any],
    restored_quotes: Mapping[str, tuple[float, float, str]],
    restored_provenance: Mapping[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    checkpoint_tip: str,
    state_path: Path,
    key: bytes,
) -> bool:
    wal = _read_quote_apply_wal(state_path, key)
    wal_tip = wal["checkpoint_ledger_sha256"]
    terminal_tip = rows[-1]["sha"] if rows else "0" * 64
    if wal_tip == "0" * 64:
        suffix = rows
    else:
        matches = [index for index, row in enumerate(rows) if row["sha"] == wal_tip]
        if len(matches) != 1:
            raise AIInventoryAdmissionIntegrityError(
                "captured-quote WAL checkpoint is absent from broker ledger"
            )
        suffix = rows[matches[0] + 1 :]
    expected = wal["expected_ledger_events"]
    _validate_quote_event_rows(suffix, expected, allow_prefix=True)

    pre_state = wal["pre_state"]
    post_state = wal["post_state"]
    if pre_state.get("broker", {}).get("ledger_sha") != wal_tip:
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote WAL pre-state tip mismatch"
        )
    if checkpoint_tip == wal_tip:
        if dict(state_record) != pre_state:
            raise AIInventoryAdmissionIntegrityError(
                "captured-quote WAL pre-state does not match checkpoint"
            )
        broker.restore(pre_state["broker"], require_ledger_match=False)
        broker.last_quotes = dict(restored_quotes)
        broker._ai_quote_provenance = _json_copy(  # type: ignore[attr-defined]
            restored_provenance
        )
        for event in expected[len(suffix) :]:
            broker._log(event["event"], event["payload"])
        completed = _read_validated_broker_rows(broker.ledger_path)
        completed_suffix = (
            completed
            if wal_tip == "0" * 64
            else completed[
                next(
                    index
                    for index, row in enumerate(completed)
                    if row["sha"] == wal_tip
                )
                + 1 :
            ]
        )
        _validate_quote_event_rows(completed_suffix, expected, allow_prefix=False)
        recovered = _json_copy(post_state)
        recovered["broker"]["ledger_sha"] = broker._prev_sha
        broker.restore(recovered["broker"], require_ledger_match=False)
        broker.last_quotes = {
            pair: (float(value[0]), float(value[1]), str(value[2]))
            for pair, value in recovered["last_quotes"].items()
        }
        broker._ai_quote_provenance = _json_copy(  # type: ignore[attr-defined]
            recovered["quote_provenance"]
        )
        return True

    if checkpoint_tip != terminal_tip or len(suffix) != len(expected):
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote WAL conflicts with broker checkpoint progress"
        )
    expected_state = _json_copy(post_state)
    expected_state["broker"]["ledger_sha"] = terminal_tip
    if dict(state_record) != expected_state:
        raise AIInventoryAdmissionIntegrityError(
            "completed captured-quote WAL does not match broker checkpoint"
        )
    broker.restore(state_record["broker"], require_ledger_match=True)
    broker.last_quotes = dict(restored_quotes)
    broker._ai_quote_provenance = _json_copy(  # type: ignore[attr-defined]
        restored_provenance
    )
    return False


def _validate_quote_event_suffix(
    ledger_path: Path,
    checkpoint_tip: str,
    expected: list[dict[str, Any]],
) -> None:
    rows = _read_validated_broker_rows(ledger_path)
    if checkpoint_tip == "0" * 64:
        suffix = rows
    else:
        matches = [
            index
            for index, row in enumerate(rows)
            if row["sha"] == checkpoint_tip
        ]
        if len(matches) != 1:
            raise AIInventoryAdmissionIntegrityError(
                "captured-quote checkpoint tip is absent from broker ledger"
            )
        suffix = rows[matches[0] + 1 :]
    _validate_quote_event_rows(suffix, expected, allow_prefix=False)


def _validate_quote_event_rows(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    allow_prefix: bool,
) -> None:
    if len(actual) > len(expected) or (
        not allow_prefix and len(actual) != len(expected)
    ):
        raise AIInventoryAdmissionIntegrityError(
            "captured-quote broker suffix length mismatch"
        )
    for index, row in enumerate(actual):
        planned = expected[index]
        if (
            row.get("event") != planned.get("event")
            or row.get("payload") != planned.get("payload")
        ):
            raise AIInventoryAdmissionIntegrityError(
                "captured-quote broker suffix differs from durable plan"
            )


def _restore_broker_state(
    broker: VirtualBroker,
    path: Path,
    key: bytes,
    *,
    decision_ledger_path: Path | None = None,
) -> bool:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1 or len(raw) > MAX_RPC_BYTES:
        raise AIInventoryAdmissionIntegrityError("invalid broker restart checkpoint")
    try:
        record = json.loads(
            raw[:-1],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AIInventoryAdmissionIntegrityError(
            "invalid broker restart checkpoint JSON"
        ) from exc
    if not isinstance(record, dict) or set(record) != {
        "contract",
        "broker",
        "last_quotes",
        "quote_provenance",
        "mac",
    }:
        raise AIInventoryAdmissionIntegrityError(
            "invalid broker restart checkpoint schema"
        )
    body = {
        key_name: record[key_name]
        for key_name in (
            "contract",
            "broker",
            "last_quotes",
            "quote_provenance",
        )
    }
    if record["contract"] != BROKER_STATE_CONTRACT or not hmac.compare_digest(
        str(record["mac"]), _mac(key, body)
    ):
        raise AIInventoryAdmissionIntegrityError(
            "broker restart checkpoint authentication failed"
        )
    quotes = record["last_quotes"]
    if not isinstance(quotes, dict):
        raise AIInventoryAdmissionIntegrityError("invalid checkpoint quotes")
    restored_quotes: dict[str, tuple[float, float, str]] = {}
    for pair, value in quotes.items():
        if (
            not isinstance(pair, str)
            or not isinstance(value, list)
            or len(value) != 3
            or isinstance(value[0], bool)
            or isinstance(value[1], bool)
            or not isinstance(value[0], (int, float))
            or not isinstance(value[1], (int, float))
            or not math.isfinite(float(value[0]))
            or not math.isfinite(float(value[1]))
            or float(value[0]) <= 0
            or float(value[1]) < float(value[0])
            or not isinstance(value[2], str)
        ):
            raise AIInventoryAdmissionIntegrityError("invalid checkpoint quote row")
        _parse_utc(value[2])
        restored_quotes[pair] = (float(value[0]), float(value[1]), value[2])
    restored_provenance = _validate_quote_provenance(
        record["quote_provenance"], restored_quotes
    )

    snapshot = record["broker"]
    if not isinstance(snapshot, dict):
        raise AIInventoryAdmissionIntegrityError(
            "invalid broker restart checkpoint snapshot"
        )
    checkpoint_tip = snapshot.get("ledger_sha")
    if not isinstance(checkpoint_tip, str) or not _SHA256_RE.fullmatch(
        checkpoint_tip
    ):
        raise AIInventoryAdmissionIntegrityError(
            "invalid broker checkpoint ledger tip"
        )
    rows = _read_validated_broker_rows(broker.ledger_path)
    terminal_tip = rows[-1]["sha"] if rows else "0" * 64
    wal_path = _quote_apply_wal_path(path)
    if wal_path.exists():
        return _recover_quote_apply_wal(
            broker,
            state_record={
                "broker": snapshot,
                "last_quotes": record["last_quotes"],
                "quote_provenance": record["quote_provenance"],
            },
            restored_quotes=restored_quotes,
            restored_provenance=restored_provenance,
            rows=rows,
            checkpoint_tip=checkpoint_tip,
            state_path=path,
            key=key,
        )
    if checkpoint_tip == terminal_tip:
        broker.restore(snapshot, require_ledger_match=True)
        broker.last_quotes = restored_quotes
        broker._ai_quote_provenance = restored_provenance  # type: ignore[attr-defined]
        return False

    if checkpoint_tip == "0" * 64:
        suffix = rows
    else:
        checkpoint_matches = [
            index for index, row in enumerate(rows) if row["sha"] == checkpoint_tip
        ]
        if len(checkpoint_matches) != 1:
            raise AIInventoryAdmissionIntegrityError(
                "broker checkpoint tip is absent from the validated ledger"
            )
        suffix = rows[checkpoint_matches[0] + 1 :]
    if not suffix:
        raise AIInventoryAdmissionIntegrityError(
            "unknown broker-ledger suffix after checkpoint"
        )
    if suffix[0]["event"] == ENTRY_PERMIT_RESERVED_EVENT:
        broker.restore(snapshot, require_ledger_match=False)
        broker.last_quotes = restored_quotes
        broker._ai_quote_provenance = restored_provenance  # type: ignore[attr-defined]
        lifecycle = reconcile_entry_checkpoint_suffix(
            broker,
            broker.ledger_path,
            room_id=broker.ledger_path.parent.name,
            candidate_id=_checkpoint_candidate_id(rows, suffix[0]),
            checkpoint_tip=checkpoint_tip,
            as_of_utc=_utc_now(),
        )
        if lifecycle["reservation"]["sha"] != suffix[0]["sha"]:
            raise AIInventoryAdmissionIntegrityError(
                "entry recovery suffix does not begin at the checkpoint"
            )
        return True
    if suffix[0]["event"] != "AI_INVENTORY_ACTION_RESERVED":
        raise AIInventoryAdmissionIntegrityError(
            "unknown broker-ledger suffix after checkpoint"
        )
    if decision_ledger_path is None:
        raise AIInventoryAdmissionIntegrityError(
            "advanced broker ledger has no configured AI decision ledger"
        )
    decision_sha256 = suffix[0]["payload"].get("decision_sha256")
    if not isinstance(decision_sha256, str) or not _SHA256_RE.fullmatch(
        decision_sha256
    ):
        raise AIInventoryAdmissionIntegrityError(
            "AI recovery reservation has an invalid decision digest"
        )
    decision = _load_validated_inventory_decision(
        decision_ledger_path, decision_sha256
    )
    broker.restore(snapshot, require_ledger_match=False)
    broker.last_quotes = restored_quotes
    broker._ai_quote_provenance = restored_provenance  # type: ignore[attr-defined]
    try:
        lifecycle = reconcile_inventory_checkpoint_suffix(broker, decision, rows)
    except InventoryConsumerIntegrityError as exc:
        raise AIInventoryAdmissionIntegrityError(
            "broker AI suffix recovery failed validation"
        ) from exc
    if lifecycle["reservation"]["sha"] != suffix[0]["sha"]:
        raise AIInventoryAdmissionIntegrityError(
            "AI recovery suffix does not begin at the checkpoint"
        )
    return True


def _validate_quote_provenance(
    value: object,
    quotes: Mapping[str, tuple[float, float, str]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise AIInventoryAdmissionIntegrityError(
            "invalid checkpoint quote provenance"
        )
    normalized: dict[str, dict[str, Any]] = {}
    expected_keys = {
        "pair",
        "bid",
        "ask",
        "timestamp_utc",
        "capture_source_sha256",
        "acquisition_receipt_sha256",
        "quote_watermark_sha256",
        "test_only_raw_quote",
    }
    for pair, raw in value.items():
        if (
            not isinstance(pair, str)
            or pair not in quotes
            or not isinstance(raw, dict)
            or set(raw) != expected_keys
            or raw.get("pair") != pair
        ):
            raise AIInventoryAdmissionIntegrityError(
                "invalid checkpoint quote provenance row"
            )
        quote = quotes[pair]
        if (
            raw.get("bid") != quote[0]
            or raw.get("ask") != quote[1]
            or raw.get("timestamp_utc") != quote[2]
            or not isinstance(raw.get("test_only_raw_quote"), bool)
        ):
            raise AIInventoryAdmissionIntegrityError(
                "checkpoint quote provenance does not match quote"
            )
        for field in (
            "capture_source_sha256",
            "acquisition_receipt_sha256",
        ):
            if not isinstance(raw.get(field), str) or _SHA256_RE.fullmatch(
                raw[field]
            ) is None:
                raise AIInventoryAdmissionIntegrityError(
                    "checkpoint quote provenance digest is invalid"
                )
        watermark = raw.get("quote_watermark_sha256")
        if raw["test_only_raw_quote"] is True:
            if watermark is not None:
                raise AIInventoryAdmissionIntegrityError(
                    "test-only quote cannot claim a durable watermark"
                )
        elif not isinstance(watermark, str) or _SHA256_RE.fullmatch(
            watermark
        ) is None:
            raise AIInventoryAdmissionIntegrityError(
                "captured quote watermark digest is invalid"
            )
        normalized[pair] = dict(raw)
    return normalized


def _checkpoint_candidate_id(
    rows: list[dict[str, Any]], reservation: Mapping[str, Any]
) -> str:
    candidate_id = reservation["payload"].get("candidate_id")
    permit_sha = reservation["payload"].get("permit_applied_receipt_sha256")
    matching = [
        row
        for row in rows
        if row["sha"] == permit_sha
        and row["event"] == "AI_INVENTORY_ACTION_APPLIED"
    ]
    if (
        len(matching) != 1
        or not isinstance(candidate_id, str)
        or matching[0]["payload"].get("candidate_id") != candidate_id
    ):
        raise AIInventoryAdmissionIntegrityError(
            "entry recovery candidate binding is invalid"
        )
    return candidate_id


def _load_validated_inventory_decision(
    path: Path, decision_sha256: str
) -> dict[str, Any]:
    if not isinstance(path, Path) or not path.is_absolute():
        raise AIInventoryAdmissionIntegrityError(
            "AI decision ledger path must be absolute"
        )
    validation = validate_inventory_decision_ledger(path)
    if validation.get("valid") is not True:
        raise AIInventoryAdmissionIntegrityError(
            "AI decision ledger failed full validation"
        )
    try:
        raw_lines = path.read_bytes().splitlines()
        decisions = [
            json.loads(
                raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            for raw in raw_lines
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AIInventoryAdmissionIntegrityError(
            "AI decision ledger cannot be decoded"
        ) from exc
    matches = [
        row
        for row in decisions
        if isinstance(row, dict)
        and row.get("decision_sha256") == decision_sha256
    ]
    if len(matches) != 1:
        raise AIInventoryAdmissionIntegrityError(
            "AI recovery decision is absent or ambiguous"
        )
    return matches[0]


def _load_verified_captured_quote(
    config: BrokerServiceConfig,
    receipt_sha256: str,
) -> dict[str, Any]:
    experiment_id = config.experiment_id
    if (
        not isinstance(experiment_id, str)
        or not experiment_id.startswith("paper-ai-inventory-")
        or Path(experiment_id).name != experiment_id
    ):
        raise AIInventoryBrokerServiceError(
            "captured quote requires a lifecycle-bound experiment_id"
        )
    receipt_root = (
        config.repository_root
        / CAPTURE_ROOT
        / "receipts"
        / experiment_id
        / config.room_id
    )
    matches = sorted(receipt_root.glob(f"????????-{receipt_sha256}.json"))
    if len(matches) != 1:
        raise AIInventoryBrokerServiceError(
            "capture receipt is absent or ambiguous"
        )
    receipt_raw = _read_regular_nofollow(
        matches[0], MAX_RPC_BYTES, "capture receipt"
    )
    try:
        receipt = json.loads(
            receipt_raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AIInventoryBrokerServiceError(
            "capture receipt cannot be decoded"
        ) from exc
    if not isinstance(receipt, dict):
        raise AIInventoryBrokerServiceError("capture receipt is not an object")
    source_sha256 = _require_sha256(
        receipt.get("canonical_source_sha256"),
        "capture canonical_source_sha256",
    )
    cutoff_utc = receipt.get("cutoff_utc")
    if not isinstance(cutoff_utc, str):
        raise AIInventoryBrokerServiceError("capture receipt cutoff is invalid")
    try:
        verified = verify_ai_source_capture_receipt(
            config.repository_root,
            experiment_id=experiment_id,
            room_id=config.room_id,
            candidate_id=config.candidate_id,
            cutoff_utc=cutoff_utc,
            source_role="quote",
            source_sha256=source_sha256,
            receipt_sha256=receipt_sha256,
        )
    except Exception as exc:
        raise AIInventoryBrokerServiceError(
            "signed quote capture receipt failed verification"
        ) from exc
    if verified != receipt:
        raise AIInventoryBrokerServiceError(
            "capture receipt changed during verification"
        )

    source_path = (
        config.repository_root
        / CANONICAL_SOURCE_ROOT
        / f"{source_sha256}.json"
    )
    source_raw = _read_regular_nofollow(
        source_path, MAX_SOURCE_BYTES, "captured quote source"
    )
    if hashlib.sha256(source_raw).hexdigest() != source_sha256:
        raise AIInventoryBrokerServiceError(
            "captured quote source digest mismatch"
        )
    try:
        source = json.loads(
            source_raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AIInventoryBrokerServiceError(
            "captured quote source cannot be decoded"
        ) from exc
    if not isinstance(source, dict) or set(source) != {
        "pair",
        "bid",
        "ask",
        "timestamp_utc",
        "max_age_seconds",
    }:
        raise AIInventoryBrokerServiceError(
            "captured quote source schema is invalid"
        )
    timestamp_utc = source.get("timestamp_utc")
    if timestamp_utc != verified.get("provider_timestamp_utc"):
        raise AIInventoryBrokerServiceError(
            "captured quote timestamp conflicts with signed provider time"
        )
    pair = source.get("pair")
    bid = _finite_positive(source.get("bid"), "captured quote bid")
    ask = _finite_positive(source.get("ask"), "captured quote ask")
    if (
        not isinstance(pair, str)
        or not isinstance(timestamp_utc, str)
        or ask < bid
    ):
        raise AIInventoryBrokerServiceError("captured quote values are invalid")
    max_age_seconds = source.get("max_age_seconds")
    if (
        isinstance(max_age_seconds, bool)
        or not isinstance(max_age_seconds, int)
        or max_age_seconds <= 0
    ):
        raise AIInventoryBrokerServiceError(
            "captured quote max_age_seconds is invalid"
        )
    _parse_utc(timestamp_utc)
    return {
        "pair": pair,
        "bid": bid,
        "ask": ask,
        "timestamp_utc": timestamp_utc,
        "max_age_seconds": max_age_seconds,
        "capture_source_sha256": source_sha256,
    }


def _load_verified_drain_quote(
    config: BrokerServiceConfig,
    receipt_sha256: str,
) -> dict[str, Any]:
    """Load quote bytes only after the signed drain receipt chain verifies."""

    experiment_id = config.experiment_id
    if (
        config.mode != "DRAIN_ONLY"
        or not isinstance(experiment_id, str)
        or not experiment_id.startswith("paper-ai-inventory-")
        or Path(experiment_id).name != experiment_id
    ):
        raise AIInventoryBrokerServiceError(
            "drain quote requires a lifecycle-bound DRAIN_ONLY service"
        )
    try:
        # Local import avoids a module cycle: the read-only drain capture
        # module imports BROKER_STATE_CONTRACT from this service.
        from quant_rabbit.dojo_ai_drain_quote import (
            verify_ai_drain_quote_receipt,
        )

        receipt = verify_ai_drain_quote_receipt(
            config.repository_root,
            experiment_id=experiment_id,
            room_id=config.room_id,
            candidate_id=config.candidate_id,
            receipt_sha256=receipt_sha256,
        )
    except Exception as exc:
        raise AIInventoryBrokerServiceError(
            "signed drain quote receipt is invalid"
        ) from exc
    source_sha256 = _require_sha256(
        receipt.get("canonical_source_sha256"),
        "canonical_source_sha256",
    )
    source_path = config.repository_root / CANONICAL_SOURCE_ROOT / (
        f"{source_sha256}.json"
    )
    raw = _read_regular_nofollow(
        source_path,
        MAX_SOURCE_BYTES,
        "canonical drain quote source",
    )
    if hashlib.sha256(raw).hexdigest() != source_sha256:
        raise AIInventoryBrokerServiceError(
            "canonical drain quote source digest mismatch"
        )
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AIInventoryBrokerServiceError(
            "canonical drain quote source is invalid"
        ) from exc
    if not isinstance(document, dict) or set(document) != {
        "pair",
        "bid",
        "ask",
        "timestamp_utc",
        "max_age_seconds",
    }:
        raise AIInventoryBrokerServiceError(
            "canonical drain quote source schema is invalid"
        )
    pair = document.get("pair")
    bid = _finite_positive(document.get("bid"), "drain bid")
    ask = _finite_positive(document.get("ask"), "drain ask")
    timestamp_utc = document.get("timestamp_utc")
    max_age_seconds = document.get("max_age_seconds")
    if (
        not isinstance(pair, str)
        or ask < bid
        or not isinstance(timestamp_utc, str)
        or receipt.get("provider_timestamp_utc") != timestamp_utc
        or isinstance(max_age_seconds, bool)
        or not isinstance(max_age_seconds, int)
        or max_age_seconds <= 0
    ):
        raise AIInventoryBrokerServiceError(
            "canonical drain quote values are invalid"
        )
    _parse_utc(timestamp_utc)
    return {
        "pair": pair,
        "bid": bid,
        "ask": ask,
        "timestamp_utc": timestamp_utc,
        "max_age_seconds": max_age_seconds,
        "capture_source_sha256": source_sha256,
        "receipt": receipt,
    }


def _verify_drain_quote_checkpoint(
    config: BrokerServiceConfig,
    broker: VirtualBroker,
    receipt: Mapping[str, Any],
) -> None:
    raw_state = _read_regular_nofollow(
        config.state_path,
        MAX_RPC_BYTES,
        "drain broker checkpoint",
    )
    expected = {
        "broker_ledger_terminal_sha256": broker._prev_sha,
        "broker_snapshot_sha256": hashlib.sha256(raw_state).hexdigest(),
        "broker_snapshot_ledger_terminal_sha256": broker._prev_sha,
        "positions_count": len(broker.positions),
        "orders_count": len(broker.orders),
        "original_ceiling_minutes": config.original_ceiling_minutes,
        "drain_only": True,
        "new_entries_allowed": False,
        "ai_evaluation_allowed": False,
        "force_close_allowed": False,
        "allowed_drain_resolutions": sorted(ALLOWED_DRAIN_RESOLUTIONS),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise AIInventoryBrokerServiceError(
                f"drain quote {field} does not match current broker truth"
            )


def _read_regular_nofollow(path: Path, maximum: int, label: str) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise AIInventoryBrokerServiceError(
            f"{label} cannot be opened safely"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > maximum:
            raise AIInventoryBrokerServiceError(f"{label} is not a bounded file")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > maximum:
            raise AIInventoryBrokerServiceError(f"{label} exceeds the byte limit")
        return raw
    finally:
        os.close(descriptor)


def _would_fill(order: VBOrder, *, bid: float, ask: float) -> bool:
    if order.kind == "LIMIT":
        return (order.side == "LONG" and ask <= order.limit_price) or (
            order.side == "SHORT" and bid >= order.limit_price
        )
    return (order.side == "LONG" and ask >= order.limit_price) or (
        order.side == "SHORT" and bid <= order.limit_price
    )


def _read_validated_broker_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    try:
        raw_lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise AIInventoryAdmissionIntegrityError(
            "broker ledger cannot be read"
        ) from exc
    for line_number, raw in enumerate(raw_lines, 1):
        try:
            row = json.loads(
                raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise AIInventoryAdmissionIntegrityError(
                f"invalid broker ledger JSON at line {line_number}"
            ) from exc
        if not isinstance(row, dict) or set(row) != {
            "ts_utc",
            "event",
            "payload",
            "prev_sha",
            "sha",
        }:
            raise AIInventoryAdmissionIntegrityError(
                f"invalid broker ledger schema at line {line_number}"
            )
        if (
            row["prev_sha"] != previous
            or not isinstance(row["payload"], dict)
            or row["sha"]
            != hashlib.sha256(
                json.dumps(
                    {
                        key: row[key]
                        for key in ("ts_utc", "event", "payload", "prev_sha")
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        ):
            raise AIInventoryAdmissionIntegrityError(
                f"invalid broker ledger chain at line {line_number}"
            )
        previous = row["sha"]
        rows.append(row)
    return rows


def _validate_config(config: BrokerServiceConfig) -> BrokerServiceConfig:
    if not isinstance(config, BrokerServiceConfig):
        raise TypeError("config must be BrokerServiceConfig")
    for path in (
        config.socket_path,
        config.ledger_path,
        config.state_path,
        config.repository_root,
    ):
        if not isinstance(path, Path) or not path.is_absolute():
            raise AIInventoryBrokerServiceError("service paths must be absolute")
    if config.socket_path != derive_broker_socket_path(config.ledger_path):
        raise AIInventoryBrokerServiceError(
            "socket path does not match the bounded ledger-derived identity"
        )
    if config.state_path.parent != config.ledger_path.parent:
        raise AIInventoryBrokerServiceError(
            "state and ledger must share the isolated room directory"
        )
    if config.ledger_path.parent.name != config.room_id or (
        "paper-ai-inventory" not in str(config.ledger_path.parent)
    ):
        raise AIInventoryBrokerServiceError("service room is not isolated")
    config.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_key(config.bot_hmac_key, "bot_hmac_key")
    _validate_key(config.runner_hmac_key, "runner_hmac_key")
    if hmac.compare_digest(config.bot_hmac_key, config.runner_hmac_key):
        raise AIInventoryBrokerServiceError("bot and runner keys must differ")
    if config.allow_test_only_raw_quotes.__class__ is not bool:
        raise AIInventoryBrokerServiceError(
            "allow_test_only_raw_quotes must be boolean"
        )
    test_only = (
        config._test_only_capability is _TEST_ONLY_RAW_QUOTES_CAPABILITY
    )
    if config.allow_test_only_raw_quotes is True and not test_only:
        raise AIInventoryBrokerServiceError(
            "test-only raw quotes require the private test capability"
        )
    if not test_only:
        try:
            configured_root = config.repository_root.resolve(strict=True)
            package_root = Path(__file__).resolve(strict=True).parents[2]
        except (IndexError, OSError) as exc:
            raise AIInventoryBrokerServiceError(
                "package-derived repository root is unavailable"
            ) from exc
        if configured_root != package_root:
            raise AIInventoryBrokerServiceError(
                "production repository root differs from the loaded package"
            )
    if config.mode not in {"ACTIVE", "DRAIN_ONLY"}:
        raise AIInventoryBrokerServiceError(
            "broker mode must be ACTIVE or DRAIN_ONLY"
        )
    if config.mode == "DRAIN_ONLY":
        if config.allow_test_only_raw_quotes is True:
            raise AIInventoryBrokerServiceError(
                "DRAIN_ONLY cannot enable test-only raw quotes"
            )
        if (
            config.drain_authorization_path is None
            or not isinstance(config.drain_authorization_path, Path)
            or not config.drain_authorization_path.is_absolute()
        ):
            raise AIInventoryBrokerServiceError(
                "DRAIN_ONLY requires an absolute authorization path"
            )
        if (
            isinstance(config.original_ceiling_minutes, bool)
            or not isinstance(config.original_ceiling_minutes, int)
            or config.original_ceiling_minutes <= 0
            or config.original_ceiling_minutes > 7 * 24 * 60
        ):
            raise AIInventoryBrokerServiceError(
                "DRAIN_ONLY original ceiling is invalid"
            )
        _verify_drain_only_startup_authorization(config)
    elif config.drain_authorization_path is not None:
        raise AIInventoryBrokerServiceError(
            "ACTIVE broker cannot carry drain authorization"
        )
    elif config.allow_test_only_raw_quotes is False:
        _verify_active_broker_lifecycle(config, at_utc=_utc_now())
    return config


def _verify_drain_only_startup_authorization(
    config: BrokerServiceConfig,
) -> dict[str, Any]:
    if not config.state_path.exists() or not config.ledger_path.exists():
        raise AIInventoryBrokerServiceError(
            "DRAIN_ONLY requires an existing broker checkpoint and ledger"
        )
    raw = _read_regular_nofollow(
        config.state_path,
        MAX_RPC_BYTES,
        "drain broker checkpoint",
    )
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise AIInventoryBrokerServiceError(
            "drain broker checkpoint framing is invalid"
        )
    try:
        record = json.loads(
            raw[:-1],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AIInventoryBrokerServiceError(
            "drain broker checkpoint is invalid"
        ) from exc
    expected_keys = {
        "contract",
        "broker",
        "last_quotes",
        "quote_provenance",
        "mac",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise AIInventoryBrokerServiceError(
            "drain broker checkpoint schema is invalid"
        )
    body = {
        name: record[name]
        for name in (
            "contract",
            "broker",
            "last_quotes",
            "quote_provenance",
        )
    }
    if (
        record.get("contract") != BROKER_STATE_CONTRACT
        or not isinstance(record.get("mac"), str)
        or not hmac.compare_digest(
            record["mac"],
            _mac(config.runner_hmac_key, body),
        )
    ):
        raise AIInventoryBrokerServiceError(
            "drain broker checkpoint authentication failed"
        )
    snapshot = record.get("broker")
    if not isinstance(snapshot, dict):
        raise AIInventoryBrokerServiceError(
            "drain broker checkpoint snapshot is invalid"
        )
    snapshot_tip = _require_sha256(
        snapshot.get("ledger_sha"),
        "drain broker checkpoint ledger_sha",
    )
    positions = snapshot.get("positions")
    orders = snapshot.get("orders")
    if not isinstance(positions, list) or not isinstance(orders, list):
        raise AIInventoryBrokerServiceError(
            "drain broker checkpoint inventory is invalid"
        )
    experiment_id = config.experiment_id
    if (
        not isinstance(experiment_id, str)
        or not experiment_id.startswith("paper-ai-inventory-")
        or Path(experiment_id).name != experiment_id
    ):
        raise AIInventoryBrokerServiceError(
            "DRAIN_ONLY requires immutable lifecycle identity"
        )
    recovery = _validated_drain_recovery_binding(
        config,
        snapshot_tip=snapshot_tip,
    )
    try:
        return verify_drain_broker_restart_authorization(
            repository_root=config.repository_root,
            room_root=config.ledger_path.parent,
            authorization_path=config.drain_authorization_path,  # type: ignore[arg-type]
            experiment_id=experiment_id,
            room_id=config.room_id,
            candidate_id=config.candidate_id,
            broker_ledger_terminal_sha256=recovery[
                "broker_ledger_terminal_sha256"
            ],
            broker_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
            broker_snapshot_ledger_terminal_sha256=snapshot_tip,
            positions_count=len(positions),
            orders_count=len(orders),
            broker_recovery_wal_sha256=recovery[
                "broker_recovery_wal_sha256"
            ],
            broker_recovery_wal_checkpoint_ledger_sha256=recovery[
                "broker_recovery_wal_checkpoint_ledger_sha256"
            ],
            broker_recovery_wal_expected_event_count=recovery[
                "broker_recovery_wal_expected_event_count"
            ],
            broker_recovery_wal_applied_event_count=recovery[
                "broker_recovery_wal_applied_event_count"
            ],
            broker_recovery_wal_validated=recovery[
                "broker_recovery_wal_validated"
            ],
            original_ceiling_minutes=config.original_ceiling_minutes,
            balance_jpy=config.balance_jpy,
            slippage_pips=config.slippage_pips,
            financing_pips_per_day=config.financing_pips_per_day,
            leverage=config.leverage,
        )
    except Exception as exc:
        raise AIInventoryBrokerServiceError(
            "canonical DRAIN_ONLY broker authorization is invalid"
        ) from exc


def _validated_drain_recovery_binding(
    config: BrokerServiceConfig,
    *,
    snapshot_tip: str,
) -> dict[str, Any]:
    rows = _read_validated_broker_rows(config.ledger_path)
    ledger_tip = rows[-1]["sha"] if rows else "0" * 64
    wal_path = _quote_apply_wal_path(config.state_path)
    if not wal_path.exists():
        if ledger_tip != snapshot_tip:
            raise AIInventoryBrokerServiceError(
                "drain ledger advanced without a signed recovery WAL"
            )
        return {
            "broker_ledger_terminal_sha256": ledger_tip,
            "broker_recovery_wal_sha256": None,
            "broker_recovery_wal_checkpoint_ledger_sha256": None,
            "broker_recovery_wal_expected_event_count": 0,
            "broker_recovery_wal_applied_event_count": 0,
            "broker_recovery_wal_validated": False,
        }
    wal_raw = _read_regular_nofollow(
        wal_path,
        MAX_QUOTE_APPLY_WAL_BYTES,
        "drain broker recovery WAL",
    )
    wal = _read_quote_apply_wal(config.state_path, config.runner_hmac_key)
    if wal.get("checkpoint_ledger_sha256") != snapshot_tip:
        raise AIInventoryBrokerServiceError(
            "drain broker recovery WAL checkpoint mismatch"
        )
    if snapshot_tip == "0" * 64:
        suffix = rows
    else:
        matches = [
            index
            for index, row in enumerate(rows)
            if row["sha"] == snapshot_tip
        ]
        if len(matches) != 1:
            raise AIInventoryBrokerServiceError(
                "drain broker recovery checkpoint is absent from ledger"
            )
        suffix = rows[matches[0] + 1 :]
    expected = wal["expected_ledger_events"]
    _validate_quote_event_rows(suffix, expected, allow_prefix=True)
    return {
        "broker_ledger_terminal_sha256": ledger_tip,
        "broker_recovery_wal_sha256": hashlib.sha256(wal_raw).hexdigest(),
        "broker_recovery_wal_checkpoint_ledger_sha256": snapshot_tip,
        "broker_recovery_wal_expected_event_count": len(expected),
        "broker_recovery_wal_applied_event_count": len(suffix),
        "broker_recovery_wal_validated": True,
    }


def _verify_active_broker_lifecycle(
    config: BrokerServiceConfig,
    *,
    at_utc: datetime,
) -> dict[str, Any]:
    if config.allow_test_only_raw_quotes is True:
        return {}
    experiment_id = config.experiment_id
    token_sha = config.launch_preflight_token_sha256
    if (
        not isinstance(experiment_id, str)
        or not experiment_id.startswith("paper-ai-inventory-")
        or Path(experiment_id).name != experiment_id
        or not isinstance(token_sha, str)
        or _SHA256_RE.fullmatch(token_sha) is None
    ):
        raise AIInventoryBrokerServiceError(
            "production broker requires immutable lifecycle identity"
        )
    try:
        token = verify_paper_ai_inventory_launch_preflight(
            config.repository_root,
            experiment_id=experiment_id,
            room_id=config.room_id,
        )
    except Exception as exc:
        raise AIInventoryBrokerServiceError(
            "canonical PAPER_ELIGIBLE broker preflight is invalid"
        ) from exc
    if (
        token.get("candidate_id") != config.candidate_id
        or token.get("launch_preflight_token_sha256") != token_sha
        or token.get("paper_only") is not True
        or token.get("order_authority") != "NONE"
        or token.get("live_permission") is not False
    ):
        raise AIInventoryBrokerServiceError(
            "broker lifecycle preflight binding mismatch"
        )
    window = token.get("future_window")
    if not isinstance(window, Mapping) or set(window) != {
        "start_utc",
        "end_utc",
    }:
        raise AIInventoryBrokerServiceError(
            "broker lifecycle future window is invalid"
        )
    start = _parse_utc(window.get("start_utc"))
    end = _parse_utc(window.get("end_utc"))
    if at_utc.tzinfo is None or at_utc.utcoffset() is None:
        raise AIInventoryBrokerServiceError(
            "broker lifecycle clock must be timezone-aware"
        )
    at = at_utc.astimezone(timezone.utc)
    if start >= end or at < start or at >= end:
        raise AIInventoryEntryDeniedError(
            "broker lifecycle is outside its immutable future window"
        )
    return dict(token)


def _validate_key(value: bytes, label: str) -> bytes:
    if not isinstance(value, bytes) or len(value) < MIN_HMAC_KEY_BYTES:
        raise AIInventoryBrokerServiceError(f"{label} is too short")
    return bytes(value)


def _require_market_open(value: datetime, label: str) -> None:
    try:
        is_open = compute_market_status(value).is_fx_open
    except Exception as exc:
        raise AIInventoryBrokerServiceError(
            f"{label} market status is unavailable"
        ) from exc
    if not is_open:
        raise AIInventoryEntryDeniedError(
            f"{label}: AI evaluation and virtual mutation are disabled while FX is closed"
        )


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise AIInventoryBrokerServiceError("invalid UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AIInventoryBrokerServiceError("invalid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AIInventoryBrokerServiceError("naive UTC timestamp")
    return parsed.astimezone(timezone.utc)


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AIInventoryBrokerServiceError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise AIInventoryBrokerServiceError(f"{label} must be positive")
    return number


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AIInventoryBrokerServiceError(f"{label} is invalid")
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AIInventoryBrokerServiceError("value is not canonical JSON") from exc


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    decoded = json.loads(_canonical_json(value))
    if not isinstance(decoded, dict):
        raise AIInventoryBrokerServiceError("RPC args must be an object")
    return decoded


def _json_list_copy(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise AIInventoryBrokerServiceError(
            "value is not a canonical JSON list"
        ) from exc
    if not isinstance(decoded, list) or any(
        not isinstance(row, dict) for row in decoded
    ):
        raise AIInventoryBrokerServiceError("value is not an object list")
    return decoded


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")

"""Signed read-only executable quotes for post-window paper-AI draining.

This is deliberately separate from the fixed-window evidence capture
contract.  It can only attest a quote after a room has durably stopped entry,
while unresolved virtual inventory remains.  It does not call a model,
evaluator, bot, virtual-broker mutation, or any OANDA write surface.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_ai_inventory_broker_service import BROKER_STATE_CONTRACT
from quant_rabbit.dojo_ai_inventory_session import (
    ALLOWED_DRAIN_RESOLUTIONS,
    SESSION_CONTRACT,
    SESSION_CONTRACT_NAME,
    SESSION_LIFECYCLE_CONTRACT,
    SESSION_LIFECYCLE_NAME,
    SESSION_STATE_CONTRACT,
    SESSION_STATE_NAME,
    AIInventorySessionContext,
)
from quant_rabbit.dojo_ai_source_adapters import (
    OANDA_EXECUTABLE_QUOTE_ADAPTER_ID,
    DojoAiSourceAdapterError,
    acquire_oanda_executable_quote,
)
from quant_rabbit.dojo_ai_source_capture import (
    CAPTURE_PRIVATE_KEY_ENV,
    MAX_SOURCE_BYTES,
    AiSourceCaptureError,
    _bound_manifest,
    _canonical_bytes,
    _load_adapter_config,
    _manifest_adapter,
    _normalize_acquisition,
    _parse_canonical_json,
    _read_regular_file,
    _validate_acquisition_document,
    _verify_registered_adapter,
    _write_canonical_source,
    _write_exclusive_or_identical,
    _REGISTERED_ADAPTERS,
)
from quant_rabbit.dojo_replay_lifecycle import (
    canonical_paper_ai_rooms_root,
    verify_paper_ai_inventory_launch_preflight,
)


DRAIN_QUOTE_RECEIPT_CONTRACT = "QR_DOJO_AI_DRAIN_QUOTE_RECEIPT_V1"
DRAIN_QUOTE_ROOT = Path("research/data/dojo_paper_ai_inventory_v1/drain_quote")
BROKER_LEDGER_NAME = "broker_ledger.jsonl"
BROKER_SNAPSHOT_NAME = "broker_state.json"
RUNNER_HMAC_KEY_ENV = "QR_DOJO_AI_INVENTORY_RUNNER_HMAC_KEY_HEX"
MAX_DRAIN_RECEIPTS = 100_000
MAX_SESSION_BYTES = 16 * 1024 * 1024
MAX_LEDGER_ROW_BYTES = 256 * 1024

_ZERO_SHA256 = "0" * 64
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}$")
_SESSION_CONTRACT_KEYS = frozenset(
    {
        "contract",
        "experiment_id",
        "room_id",
        "candidate_id",
        "dependency_id",
        "pair",
        "window_start_utc",
        "window_end_utc",
        "source_roles",
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
        "paper_eligible_event_sha256",
        "future_registry_sha256",
        "session_config_sha256",
        "screen_name",
        "process_argv",
        "process_argv_sha256",
        "environment_allowlist",
        "new_entries_after_window_allowed",
        "active_source_capture_after_window_allowed",
        "drain_quote_required",
        "drain_quote_receipt_contract",
        "force_close_allowed",
        "allowed_drain_resolutions",
        "paper_only",
        "order_authority",
        "live_permission",
        "external_broker_mutation_allowed",
        "session_contract_sha256",
    }
)
_LIFECYCLE_KEYS = frozenset(
    {
        "contract",
        "sequence",
        "previous_event_sha256",
        "event",
        "recorded_at_utc",
        "payload",
        "paper_only",
        "order_authority",
        "live_permission",
        "external_broker_mutation_allowed",
        "event_sha256",
    }
)
_SESSION_STATE_KEYS = frozenset(
    {
        "contract",
        "status",
        "updated_at_utc",
        "lifecycle_tip_sha256",
        "positions_count",
        "orders_count",
        "pending_evaluations",
        "new_entries_allowed",
        "market_open",
        "paper_only",
        "order_authority",
        "live_permission",
        "external_broker_mutation_allowed",
        "state_sha256",
    }
)
_BROKER_STATE_KEYS = frozenset(
    {"contract", "broker", "last_quotes", "quote_provenance", "mac"}
)
_BROKER_SNAPSHOT_KEYS = frozenset(
    {"balance_jpy", "seq", "positions", "orders", "ledger_sha"}
)
_LEDGER_KEYS = frozenset({"ts_utc", "event", "payload", "prev_sha", "sha"})
_RECEIPT_BODY_KEYS = frozenset(
    {
        "contract",
        "sequence",
        "previous_receipt_sha256",
        "experiment_id",
        "room_id",
        "candidate_id",
        "source_role",
        "cutoff_utc",
        "fixed_window_end_utc",
        "drain_only",
        "new_entries_allowed",
        "ai_evaluation_allowed",
        "force_close_allowed",
        "original_ceiling_minutes",
        "allowed_drain_resolutions",
        "session_contract_file_sha256",
        "session_lifecycle_tip_sha256",
        "session_state_file_sha256",
        "broker_ledger_terminal_sha256",
        "broker_snapshot_sha256",
        "broker_snapshot_ledger_terminal_sha256",
        "positions_count",
        "orders_count",
        "canonical_source_sha256",
        "raw_source_bytes_sha256",
        "source_watermark_sha256",
        "provider_kind",
        "adapter_id",
        "adapter_module",
        "adapter_callable",
        "adapter_executable_sha256",
        "adapter_config_sha256",
        "provider_timestamp_utc",
        "fetched_at_utc",
        "capture_manifest_file_sha256",
        "capture_manifest_sha256",
        "capture_key_id",
        "paper_only",
        "order_authority",
        "live_permission",
        "external_broker_mutation_allowed",
    }
)
_RECEIPT_KEYS = _RECEIPT_BODY_KEYS | frozenset({"receipt_sha256", "signature_base64"})


class AiDrainQuoteError(RuntimeError):
    """The drain-only quote or its evidence boundary failed closed."""


class AiDrainQuoteMarketClosedError(AiDrainQuoteError):
    """Drain quote acquisition is disabled while the FX market is closed."""


def capture_registered_ai_drain_quote(
    context: AIInventorySessionContext,
    cutoff_utc: datetime | str,
) -> dict[str, Any]:
    """Capture one signed quote for an already-entry-stopped drain tick."""

    root = _trusted_repository_root()
    now = _utc_now()
    _require_market_open(now, "pre-fetch clock")
    bound = _validate_current_drain(
        root,
        context=context,
        now=now,
    )
    cutoff = _utc(cutoff_utc, "cutoff_utc")
    capture_deadline = _exact_int(
        bound["session_contract"].get("capture_deadline_seconds"),
        "capture_deadline_seconds",
        minimum=1,
        maximum=300,
    )
    if cutoff <= now or cutoff > now + timedelta(seconds=capture_deadline):
        raise AiDrainQuoteError("drain quote cutoff exceeds its fixed deadline")
    if cutoff < bound["window_end"]:
        raise AiDrainQuoteError("drain quote cutoff is inside the fixed window")
    _require_market_open(cutoff, "drain quote cutoff")

    preflight = _verified_preflight(root, context)
    manifest, manifest_file_sha = _capture_manifest(root, preflight)
    adapter = _manifest_adapter(manifest, "quote")
    registration = _REGISTERED_ADAPTERS.get(OANDA_EXECUTABLE_QUOTE_ADAPTER_ID)
    if registration is None:
        raise AiDrainQuoteError("registered executable quote adapter is absent")
    try:
        _verify_registered_adapter(registration, adapter)
        config = _load_adapter_config(root, adapter)
    except AiSourceCaptureError as exc:
        raise AiDrainQuoteError("drain quote adapter binding is invalid") from exc
    if config.get("pair") != bound["session_contract"]["pair"]:
        raise AiDrainQuoteError("drain quote adapter pair binding mismatch")
    private_key = _load_external_private_key(root, manifest)

    try:
        acquisition = _normalize_acquisition(acquire_oanda_executable_quote(config))
    except (AiSourceCaptureError, DojoAiSourceAdapterError) as exc:
        raise AiDrainQuoteError("read-only drain quote acquisition failed") from exc
    fetched_at = _utc_now()
    _require_market_open(fetched_at, "post-fetch clock")
    provider_at = _utc(
        acquisition.provider_timestamp_utc,
        "provider_timestamp_utc",
    )
    if provider_at > fetched_at or fetched_at > cutoff:
        raise AiDrainQuoteError("drain quote time ordering is invalid")
    try:
        document = _parse_canonical_json(
            acquisition.raw_bytes,
            "drain quote source",
        )
        _validate_acquisition_document(
            document,
            adapter=adapter,
            acquisition=acquisition,
            fetched_at=fetched_at,
        )
    except AiSourceCaptureError as exc:
        raise AiDrainQuoteError("drain quote source is invalid") from exc

    after = _validate_current_drain(root, context=context, now=fetched_at)
    for field in (
        "session_contract_file_sha256",
        "session_lifecycle_tip_sha256",
        "session_state_file_sha256",
        "broker_ledger_terminal_sha256",
        "broker_snapshot_sha256",
        "broker_snapshot_ledger_terminal_sha256",
        "positions_count",
        "orders_count",
    ):
        if after[field] != bound[field]:
            raise AiDrainQuoteError("drain state changed during quote acquisition")

    source_sha = hashlib.sha256(acquisition.raw_bytes).hexdigest()
    try:
        _write_canonical_source(root, source_sha, acquisition.raw_bytes)
    except AiSourceCaptureError as exc:
        raise AiDrainQuoteError("canonical drain quote cannot be stored") from exc

    receipt_root = _receipt_root(
        root,
        context.config.experiment_id,
        context.config.room_id,
        create=True,
    )
    lock_path = receipt_root / ".drain-quote.lock"
    lock_fd = _open_lock(lock_path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        rows = _verify_receipt_chain(
            receipt_root,
            manifest=manifest,
            manifest_file_sha=manifest_file_sha,
            session_contract=bound["session_contract"],
        )
        body: dict[str, Any] = {
            "contract": DRAIN_QUOTE_RECEIPT_CONTRACT,
            "sequence": len(rows) + 1,
            "previous_receipt_sha256": (
                rows[-1]["receipt_sha256"] if rows else _ZERO_SHA256
            ),
            "experiment_id": context.config.experiment_id,
            "room_id": context.config.room_id,
            "candidate_id": context.config.candidate_id,
            "source_role": "quote",
            "cutoff_utc": _format_utc(cutoff),
            "fixed_window_end_utc": _format_utc(bound["window_end"]),
            "drain_only": True,
            "new_entries_allowed": False,
            "ai_evaluation_allowed": False,
            "force_close_allowed": False,
            "original_ceiling_minutes": bound["session_contract"][
                "original_ceiling_minutes"
            ],
            "allowed_drain_resolutions": sorted(ALLOWED_DRAIN_RESOLUTIONS),
            "session_contract_file_sha256": bound["session_contract_file_sha256"],
            "session_lifecycle_tip_sha256": (bound["session_lifecycle_tip_sha256"]),
            "session_state_file_sha256": bound["session_state_file_sha256"],
            "broker_ledger_terminal_sha256": (bound["broker_ledger_terminal_sha256"]),
            "broker_snapshot_sha256": bound["broker_snapshot_sha256"],
            "broker_snapshot_ledger_terminal_sha256": (
                bound["broker_snapshot_ledger_terminal_sha256"]
            ),
            "positions_count": bound["positions_count"],
            "orders_count": bound["orders_count"],
            "canonical_source_sha256": source_sha,
            "raw_source_bytes_sha256": source_sha,
            "source_watermark_sha256": acquisition.source_watermark_sha256,
            "provider_kind": adapter["provider_kind"],
            "adapter_id": adapter["adapter_id"],
            "adapter_module": adapter["adapter_module"],
            "adapter_callable": adapter["adapter_callable"],
            "adapter_executable_sha256": adapter["adapter_executable_sha256"],
            "adapter_config_sha256": adapter["adapter_config_sha256"],
            "provider_timestamp_utc": _format_utc(provider_at),
            "fetched_at_utc": _format_utc(fetched_at),
            "capture_manifest_file_sha256": manifest_file_sha,
            "capture_manifest_sha256": manifest["manifest_sha256"],
            "capture_key_id": manifest["capture_key_id"],
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "external_broker_mutation_allowed": False,
        }
        digest = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        receipt = {
            **body,
            "receipt_sha256": digest,
            "signature_base64": base64.b64encode(
                private_key.sign(_canonical_bytes(body))
            ).decode("ascii"),
        }
        raw = _canonical_bytes(receipt) + b"\n"
        _write_exclusive_or_identical(
            receipt_root / f"{len(rows) + 1:08d}-{digest}.json",
            raw,
            "drain quote receipt",
        )
        return receipt
    except AiSourceCaptureError as exc:
        raise AiDrainQuoteError("drain quote receipt cannot be stored") from exc
    finally:
        os.close(lock_fd)


def verify_ai_drain_quote_receipt(
    repository_root: Path,
    *,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
    receipt_sha256: str,
) -> dict[str, Any]:
    """Validate the complete signed drain quote chain and one receipt."""

    root = _repository_root(repository_root)
    experiment_id = _identity(experiment_id, "experiment_id")
    room_id = _identity(room_id, "room_id")
    candidate_id = _sha(candidate_id, "candidate_id")
    receipt_sha256 = _sha(receipt_sha256, "receipt_sha256")
    room_root = _canonical_room_root(root, experiment_id, room_id)
    session_contract, _ = _read_session_contract(
        room_root,
        experiment_id=experiment_id,
        room_id=room_id,
        candidate_id=candidate_id,
    )
    preflight = verify_paper_ai_inventory_launch_preflight(
        root,
        experiment_id=experiment_id,
        room_id=room_id,
    )
    manifest, manifest_file_sha = _capture_manifest(root, preflight)
    adapter = _manifest_adapter(manifest, "quote")
    try:
        config = _load_adapter_config(root, adapter)
    except AiSourceCaptureError as exc:
        raise AiDrainQuoteError("capture manifest is invalid") from exc
    if config.get("pair") != session_contract["pair"]:
        raise AiDrainQuoteError("drain quote adapter pair binding mismatch")
    rows = _verify_receipt_chain(
        _receipt_root(root, experiment_id, room_id, create=False),
        manifest=manifest,
        manifest_file_sha=manifest_file_sha,
        session_contract=session_contract,
    )
    matches = [row for row in rows if row["receipt_sha256"] == receipt_sha256]
    if len(matches) != 1:
        raise AiDrainQuoteError("drain quote receipt is absent or duplicated")
    receipt = matches[0]
    if receipt["candidate_id"] != candidate_id:
        raise AiDrainQuoteError("drain quote candidate binding mismatch")
    source_root = root / (
        Path("research/data/dojo_paper_ai_inventory_v1/canonical_sources")
    )
    source_path = source_root / f"{receipt['canonical_source_sha256']}.json"
    raw = _read_regular_file(
        source_path,
        MAX_SOURCE_BYTES,
        "canonical drain quote",
    )
    if hashlib.sha256(raw).hexdigest() != receipt["canonical_source_sha256"]:
        raise AiDrainQuoteError("canonical drain quote digest mismatch")
    document = _parse_canonical_json(raw, "canonical drain quote")
    fetched_at = _utc(receipt["fetched_at_utc"], "fetched_at_utc")
    try:
        acquisition = _normalize_acquisition(
            {
                "raw_bytes": raw,
                "provider_timestamp_utc": receipt["provider_timestamp_utc"],
                "source_watermark_sha256": receipt["source_watermark_sha256"],
            }
        )
        _validate_acquisition_document(
            document,
            adapter=adapter,
            acquisition=acquisition,
            fetched_at=fetched_at,
        )
    except AiSourceCaptureError as exc:
        raise AiDrainQuoteError("canonical drain quote is invalid") from exc
    if not isinstance(document, dict) or document.get("pair") != config["pair"]:
        raise AiDrainQuoteError("canonical drain quote pair binding mismatch")
    return dict(receipt)


def _validate_current_drain(
    root: Path,
    *,
    context: AIInventorySessionContext,
    now: datetime,
) -> dict[str, Any]:
    if not isinstance(context, AIInventorySessionContext):
        raise AiDrainQuoteError("drain quote requires a session context")
    config = context.config
    if config.repository_root.resolve(strict=True) != root:
        raise AiDrainQuoteError("session repository binding mismatch")
    room_root = _canonical_room_root(
        root,
        config.experiment_id,
        config.room_id,
    )
    if context.room_root != room_root:
        raise AiDrainQuoteError("session room binding mismatch")
    contract, contract_raw_sha = _read_session_contract(
        room_root,
        experiment_id=config.experiment_id,
        room_id=config.room_id,
        candidate_id=config.candidate_id,
    )
    if dict(context.session_contract) != contract:
        raise AiDrainQuoteError("session context contract binding mismatch")
    window_end = _utc(contract["window_end_utc"], "window_end_utc")
    if window_end != config.window_end:
        raise AiDrainQuoteError("session fixed-window binding mismatch")
    if now < window_end:
        raise AiDrainQuoteError("drain quote is forbidden inside fixed window")
    lifecycle, lifecycle_raw_sha = _read_lifecycle(
        room_root / SESSION_LIFECYCLE_NAME,
        now=now,
    )
    lifecycle_tip = lifecycle[-1]["event_sha256"]
    entry_stop_indexes = [
        index for index, row in enumerate(lifecycle) if row["event"] == "ENTRY_STOP"
    ]
    if not entry_stop_indexes:
        raise AiDrainQuoteError("session lifecycle has no entry stop")
    for row in lifecycle[entry_stop_indexes[-1] + 1 :]:
        if (
            row["event"] != "SESSION_RESUME"
            or row["payload"].get("status") != "DRAINING"
        ):
            raise AiDrainQuoteError("session lifecycle is not an entry-stopped drain")
    state, state_raw_sha = _read_session_state(
        room_root / SESSION_STATE_NAME,
        lifecycle_tip=lifecycle_tip,
        now=now,
    )
    ledger_tip = _read_broker_ledger(
        room_root / BROKER_LEDGER_NAME,
        now=now,
    )
    snapshot, snapshot_raw_sha = _read_broker_snapshot(
        room_root / BROKER_SNAPSHOT_NAME,
        ledger_tip=ledger_tip,
        runner_hmac_key=_runner_hmac_key(),
    )
    broker = snapshot["broker"]
    positions_count = len(broker["positions"])
    orders_count = len(broker["orders"])
    if (
        state["positions_count"] != positions_count
        or state["orders_count"] != orders_count
    ):
        raise AiDrainQuoteError("session and broker inventory counts differ")
    if positions_count == 0 and orders_count == 0:
        raise AiDrainQuoteError("sealed or zero inventory needs no drain quote")
    inventory_pairs = {
        item.get("pair")
        for item in [*broker["positions"], *broker["orders"]]
        if isinstance(item, dict)
    }
    if inventory_pairs != {contract["pair"]}:
        raise AiDrainQuoteError("unresolved inventory is outside session pair")
    return {
        "session_contract": contract,
        "session_contract_file_sha256": contract_raw_sha,
        "session_lifecycle_tip_sha256": lifecycle_tip,
        "session_lifecycle_file_sha256": lifecycle_raw_sha,
        "session_state_file_sha256": state_raw_sha,
        "broker_ledger_terminal_sha256": ledger_tip,
        "broker_snapshot_sha256": snapshot_raw_sha,
        "broker_snapshot_ledger_terminal_sha256": broker["ledger_sha"],
        "positions_count": positions_count,
        "orders_count": orders_count,
        "window_end": window_end,
    }


def _read_session_contract(
    room_root: Path,
    *,
    experiment_id: str,
    room_id: str,
    candidate_id: str,
) -> tuple[dict[str, Any], str]:
    raw = _read_regular_file(
        room_root / SESSION_CONTRACT_NAME,
        MAX_SESSION_BYTES,
        "AI inventory session contract",
    )
    value = _parse_canonical_json(raw, "AI inventory session contract")
    if not isinstance(value, dict) or set(value) != _SESSION_CONTRACT_KEYS:
        raise AiDrainQuoteError("session contract schema is invalid")
    if (
        value.get("contract") != SESSION_CONTRACT
        or value.get("experiment_id") != experiment_id
        or value.get("room_id") != room_id
        or value.get("candidate_id") != candidate_id
    ):
        raise AiDrainQuoteError("session contract identity is invalid")
    _require_safety(value, "session contract")
    claimed = _sha(
        value.get("session_contract_sha256"),
        "session_contract_sha256",
    )
    body = {
        key: item for key, item in value.items() if key != "session_contract_sha256"
    }
    if claimed != hashlib.sha256(_canonical_bytes(body)).hexdigest():
        raise AiDrainQuoteError("session contract digest mismatch")
    if (
        value.get("new_entries_after_window_allowed") is not False
        or value.get("active_source_capture_after_window_allowed") is not False
        or value.get("drain_quote_required") is not True
        or value.get("drain_quote_receipt_contract") != DRAIN_QUOTE_RECEIPT_CONTRACT
        or value.get("force_close_allowed") is not False
        or value.get("allowed_drain_resolutions") != sorted(ALLOWED_DRAIN_RESOLUTIONS)
    ):
        raise AiDrainQuoteError("session drain policy is invalid")
    _exact_int(
        value.get("original_ceiling_minutes"),
        "original_ceiling_minutes",
        minimum=1,
        maximum=7 * 24 * 60,
    )
    return dict(value), hashlib.sha256(raw).hexdigest()


def _read_lifecycle(path: Path, *, now: datetime) -> tuple[list[dict[str, Any]], str]:
    raw = _read_regular_file(path, MAX_SESSION_BYTES, "session lifecycle")
    rows: list[dict[str, Any]] = []
    previous = _ZERO_SHA256
    for sequence, line in enumerate(raw.splitlines(), 1):
        if not line or len(line) > MAX_LEDGER_ROW_BYTES:
            raise AiDrainQuoteError("session lifecycle row is invalid")
        row = _parse_json_line(line, "session lifecycle row")
        if not isinstance(row, dict) or set(row) != _LIFECYCLE_KEYS:
            raise AiDrainQuoteError("session lifecycle schema is invalid")
        body = {key: item for key, item in row.items() if key != "event_sha256"}
        claimed = _sha(row.get("event_sha256"), "event_sha256")
        if (
            row.get("contract") != SESSION_LIFECYCLE_CONTRACT
            or row.get("sequence") != sequence
            or isinstance(row.get("sequence"), bool)
            or row.get("previous_event_sha256") != previous
            or claimed != hashlib.sha256(_canonical_bytes(body)).hexdigest()
            or _utc(row.get("recorded_at_utc"), "recorded_at_utc") > now
        ):
            raise AiDrainQuoteError("session lifecycle chain is invalid")
        _require_safety(row, "session lifecycle")
        previous = claimed
        rows.append(row)
    if not rows:
        raise AiDrainQuoteError("session lifecycle is empty")
    return rows, hashlib.sha256(raw).hexdigest()


def _read_session_state(
    path: Path,
    *,
    lifecycle_tip: str,
    now: datetime,
) -> tuple[dict[str, Any], str]:
    raw = _read_regular_file(path, MAX_SESSION_BYTES, "session drain state")
    value = _parse_canonical_json(raw, "session drain state")
    if not isinstance(value, dict) or set(value) != _SESSION_STATE_KEYS:
        raise AiDrainQuoteError("session drain state schema is invalid")
    _require_safety(value, "session drain state")
    body = {key: item for key, item in value.items() if key != "state_sha256"}
    if (
        value.get("contract") != SESSION_STATE_CONTRACT
        or value.get("status") != "DRAINING"
        or value.get("lifecycle_tip_sha256") != lifecycle_tip
        or value.get("new_entries_allowed") is not False
        or value.get("market_open") is not True
        or value.get("state_sha256")
        != hashlib.sha256(_canonical_bytes(body)).hexdigest()
    ):
        raise AiDrainQuoteError("session drain state is invalid")
    if _utc(value.get("updated_at_utc"), "updated_at_utc") > now:
        raise AiDrainQuoteError("session drain state is future-dated")
    _exact_int(value.get("positions_count"), "positions_count", 0, 1_000_000)
    _exact_int(value.get("orders_count"), "orders_count", 0, 1_000_000)
    return dict(value), hashlib.sha256(raw).hexdigest()


def _read_broker_ledger(path: Path, *, now: datetime) -> str:
    raw = _read_regular_file(path, MAX_SESSION_BYTES, "broker ledger")
    previous = _ZERO_SHA256
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line or len(line) > MAX_LEDGER_ROW_BYTES:
            raise AiDrainQuoteError("broker ledger row is invalid")
        row = _parse_json_line(line, "broker ledger row")
        if not isinstance(row, dict) or set(row) != _LEDGER_KEYS:
            raise AiDrainQuoteError("broker ledger schema is invalid")
        body = {key: row[key] for key in ("ts_utc", "event", "payload", "prev_sha")}
        if (
            row.get("prev_sha") != previous
            or not isinstance(row.get("event"), str)
            or not row["event"]
            or not isinstance(row.get("payload"), dict)
            or _utc(row.get("ts_utc"), f"broker ledger row {line_number}") > now
            or row.get("sha") != hashlib.sha256(_canonical_bytes(body)).hexdigest()
        ):
            raise AiDrainQuoteError("broker ledger chain is invalid")
        previous = row["sha"]
    return previous


def _read_broker_snapshot(
    path: Path,
    *,
    ledger_tip: str,
    runner_hmac_key: bytes,
) -> tuple[dict[str, Any], str]:
    raw = _read_regular_file(path, MAX_SESSION_BYTES, "broker snapshot")
    value = _parse_canonical_json(raw, "broker snapshot")
    if not isinstance(value, dict) or set(value) != _BROKER_STATE_KEYS:
        raise AiDrainQuoteError("broker snapshot schema is invalid")
    if value.get("contract") != BROKER_STATE_CONTRACT:
        raise AiDrainQuoteError("broker snapshot contract is invalid")
    body = {
        key: value[key]
        for key in (
            "contract",
            "broker",
            "last_quotes",
            "quote_provenance",
        )
    }
    expected_mac = hmac.new(
        runner_hmac_key,
        _canonical_bytes(body),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(str(value.get("mac")), expected_mac):
        raise AiDrainQuoteError("broker snapshot authentication failed")
    broker = value.get("broker")
    if not isinstance(broker, dict) or set(broker) != _BROKER_SNAPSHOT_KEYS:
        raise AiDrainQuoteError("broker snapshot body is invalid")
    if broker.get("ledger_sha") != ledger_tip:
        raise AiDrainQuoteError("broker snapshot tip differs from ledger")
    if not isinstance(broker.get("positions"), list) or not isinstance(
        broker.get("orders"), list
    ):
        raise AiDrainQuoteError("broker inventory is invalid")
    for item in [*broker["positions"], *broker["orders"]]:
        if not isinstance(item, dict):
            raise AiDrainQuoteError("broker inventory row is invalid")
    return dict(value), hashlib.sha256(raw).hexdigest()


def _runner_hmac_key() -> bytes:
    raw = os.environ.get(RUNNER_HMAC_KEY_ENV)
    if raw is None:
        raise AiDrainQuoteError("runner HMAC key is not configured")
    try:
        key = bytes.fromhex(raw)
    except ValueError as exc:
        raise AiDrainQuoteError("runner HMAC key is invalid") from exc
    if len(key) < 32:
        raise AiDrainQuoteError("runner HMAC key is invalid")
    return key


def _verified_preflight(
    root: Path,
    context: AIInventorySessionContext,
) -> dict[str, Any]:
    try:
        value = verify_paper_ai_inventory_launch_preflight(
            root,
            experiment_id=context.config.experiment_id,
            room_id=context.config.room_id,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise AiDrainQuoteError("launch preflight is invalid") from exc
    if (
        value.get("candidate_id") != context.config.candidate_id
        or value.get("launch_preflight_token_sha256")
        != context.config.launch_preflight_token_sha256
    ):
        raise AiDrainQuoteError("launch preflight binding mismatch")
    _require_safety(value, "launch preflight", external=False)
    return dict(value)


def _capture_manifest(
    root: Path,
    preflight: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    try:
        manifest, file_sha = _bound_manifest(root, preflight)
    except AiSourceCaptureError as exc:
        raise AiDrainQuoteError("capture manifest is invalid") from exc
    adapter = _manifest_adapter(manifest, "quote")
    if adapter.get("adapter_id") != OANDA_EXECUTABLE_QUOTE_ADAPTER_ID:
        raise AiDrainQuoteError("drain requires the executable quote adapter")
    return manifest, file_sha


def _load_external_private_key(
    root: Path,
    manifest: Mapping[str, Any],
) -> Ed25519PrivateKey:
    raw_path = os.environ.get(CAPTURE_PRIVATE_KEY_ENV)
    if not raw_path:
        raise AiDrainQuoteError("drain quote signing key is not configured")
    path = Path(raw_path)
    if not path.is_absolute():
        raise AiDrainQuoteError("drain quote signing key path must be absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except ValueError:
        pass
    except OSError as exc:
        raise AiDrainQuoteError("drain quote signing key is unavailable") from exc
    else:
        raise AiDrainQuoteError("drain quote signing key must be outside repository")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
    ):
        raise AiDrainQuoteError("drain quote signing key must be owner-only mode 0600")
    try:
        raw = _read_regular_file(resolved, 64 * 1024, "drain quote signing key")
        key = serialization.load_pem_private_key(raw, password=None)
    except (AiSourceCaptureError, TypeError, ValueError) as exc:
        raise AiDrainQuoteError("drain quote signing key is invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise AiDrainQuoteError("drain quote signing key is not Ed25519")
    public = base64.b64encode(
        key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    if public != manifest["ed25519_public_key_base64"]:
        raise AiDrainQuoteError("drain quote signing key binding mismatch")
    return key


def _verify_receipt_chain(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_file_sha: str,
    session_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    paths = sorted(root.glob("[0-9]" * 8 + "-*.json"))
    if len(paths) > MAX_DRAIN_RECEIPTS:
        raise AiDrainQuoteError("drain quote receipt chain exceeds bound")
    public_key = Ed25519PublicKey.from_public_bytes(
        base64.b64decode(
            manifest["ed25519_public_key_base64"],
            validate=True,
        )
    )
    previous = _ZERO_SHA256
    rows: list[dict[str, Any]] = []
    for sequence, path in enumerate(paths, 1):
        raw = _read_regular_file(
            path,
            MAX_LEDGER_ROW_BYTES,
            "drain quote receipt",
        )
        value = _parse_canonical_json(raw, "drain quote receipt")
        if not isinstance(value, dict) or set(value) != _RECEIPT_KEYS:
            raise AiDrainQuoteError("drain quote receipt schema is invalid")
        body = {key: value[key] for key in _RECEIPT_BODY_KEYS}
        digest = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        if (
            value.get("contract") != DRAIN_QUOTE_RECEIPT_CONTRACT
            or value.get("sequence") != sequence
            or isinstance(value.get("sequence"), bool)
            or value.get("previous_receipt_sha256") != previous
            or value.get("receipt_sha256") != digest
            or path.name != f"{sequence:08d}-{digest}.json"
        ):
            raise AiDrainQuoteError("drain quote receipt chain is invalid")
        _validate_receipt_bindings(
            value,
            manifest=manifest,
            manifest_file_sha=manifest_file_sha,
            session_contract=session_contract,
        )
        try:
            signature = base64.b64decode(
                value["signature_base64"],
                validate=True,
            )
            if len(signature) != 64:
                raise ValueError("signature length")
            public_key.verify(signature, _canonical_bytes(body))
        except (InvalidSignature, TypeError, ValueError) as exc:
            raise AiDrainQuoteError("drain quote receipt signature is invalid") from exc
        previous = digest
        rows.append(dict(value))
    return rows


def _validate_receipt_bindings(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_file_sha: str,
    session_contract: Mapping[str, Any],
) -> None:
    _require_safety(value, "drain quote receipt")
    fixed_window_end = _utc(
        value.get("fixed_window_end_utc"),
        "receipt fixed_window_end_utc",
    )
    if (
        value.get("experiment_id") != session_contract["experiment_id"]
        or value.get("room_id") != session_contract["room_id"]
        or value.get("candidate_id") != session_contract["candidate_id"]
        or value.get("source_role") != "quote"
        or fixed_window_end
        != _utc(
            session_contract["window_end_utc"],
            "session fixed_window_end_utc",
        )
        or value.get("drain_only") is not True
        or value.get("new_entries_allowed") is not False
        or value.get("ai_evaluation_allowed") is not False
        or value.get("force_close_allowed") is not False
        or value.get("original_ceiling_minutes")
        != session_contract["original_ceiling_minutes"]
        or value.get("allowed_drain_resolutions") != sorted(ALLOWED_DRAIN_RESOLUTIONS)
        or value.get("session_contract_file_sha256")
        != hashlib.sha256(_canonical_bytes(session_contract) + b"\n").hexdigest()
    ):
        raise AiDrainQuoteError("drain quote receipt session binding mismatch")
    adapter = _manifest_adapter(manifest, "quote")
    for field in (
        "provider_kind",
        "adapter_id",
        "adapter_module",
        "adapter_callable",
        "adapter_executable_sha256",
        "adapter_config_sha256",
    ):
        if value.get(field) != adapter[field]:
            raise AiDrainQuoteError(f"drain quote receipt {field} binding mismatch")
    if (
        value.get("capture_manifest_file_sha256") != manifest_file_sha
        or value.get("capture_manifest_sha256") != manifest["manifest_sha256"]
        or value.get("capture_key_id") != manifest["capture_key_id"]
        or value.get("canonical_source_sha256") != value.get("raw_source_bytes_sha256")
        or value.get("source_watermark_sha256") != value.get("canonical_source_sha256")
        or value.get("broker_ledger_terminal_sha256")
        != value.get("broker_snapshot_ledger_terminal_sha256")
    ):
        raise AiDrainQuoteError("drain quote receipt evidence binding mismatch")
    for field in (
        "previous_receipt_sha256",
        "session_contract_file_sha256",
        "session_lifecycle_tip_sha256",
        "session_state_file_sha256",
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
        "receipt_sha256",
    ):
        _sha(value.get(field), field)
    provider_at = _utc(value.get("provider_timestamp_utc"), "provider time")
    fetched_at = _utc(value.get("fetched_at_utc"), "fetched time")
    cutoff = _utc(value.get("cutoff_utc"), "cutoff")
    positions_count = _exact_int(
        value.get("positions_count"),
        "positions_count",
        0,
        1_000_000,
    )
    orders_count = _exact_int(
        value.get("orders_count"),
        "orders_count",
        0,
        1_000_000,
    )
    _exact_int(value.get("sequence"), "sequence", 1, MAX_DRAIN_RECEIPTS)
    if positions_count == 0 and orders_count == 0:
        raise AiDrainQuoteError("drain quote receipt has zero inventory")
    if provider_at > fetched_at or fetched_at > cutoff or cutoff < fixed_window_end:
        raise AiDrainQuoteError("drain quote receipt time ordering is invalid")
    _require_market_open(fetched_at, "drain quote receipt fetched time")
    _require_market_open(cutoff, "drain quote receipt cutoff")


def _canonical_room_root(
    root: Path,
    experiment_id: str,
    room_id: str,
) -> Path:
    base = canonical_paper_ai_rooms_root(root)
    expected = base / experiment_id / room_id
    try:
        base_resolved = base.resolve(strict=True)
        resolved = expected.resolve(strict=True)
        metadata = expected.lstat()
    except OSError as exc:
        raise AiDrainQuoteError("canonical paper-AI room is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != expected
        or resolved.parent.parent != base_resolved
    ):
        raise AiDrainQuoteError("canonical paper-AI room is unsafe")
    return resolved


def _receipt_root(
    root: Path,
    experiment_id: str,
    room_id: str,
    *,
    create: bool,
) -> Path:
    current = root
    relative = DRAIN_QUOTE_ROOT / "receipts" / experiment_id / room_id
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not create:
                raise AiDrainQuoteError("drain quote receipt root is absent")
            current.mkdir(mode=0o700)
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise AiDrainQuoteError("drain quote receipt root is unsafe")
    resolved = current.resolve(strict=True)
    if resolved != root / relative:
        raise AiDrainQuoteError("drain quote receipt root escaped repository")
    return resolved


def _open_lock(path: Path) -> int:
    try:
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise AiDrainQuoteError("drain quote lock cannot be opened") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise AiDrainQuoteError("drain quote lock is not a regular file")
    return descriptor


def _trusted_repository_root() -> Path:
    try:
        root = Path(__file__).resolve(strict=True).parents[2]
    except (IndexError, OSError) as exc:
        raise AiDrainQuoteError("package repository root is unavailable") from exc
    return _repository_root(root)


def _repository_root(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise AiDrainQuoteError("repository root must be an absolute Path")
    try:
        root_metadata = value.lstat()
        git_metadata = (value / ".git").lstat()
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise AiDrainQuoteError("repository root is unavailable") from exc
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(git_metadata.st_mode)
        or not (
            stat.S_ISDIR(git_metadata.st_mode) or stat.S_ISREG(git_metadata.st_mode)
        )
    ):
        raise AiDrainQuoteError("repository root is unsafe")
    return resolved


def _require_safety(
    value: Mapping[str, Any],
    label: str,
    *,
    external: bool = True,
) -> None:
    if (
        value.get("paper_only") is not True
        or value.get("order_authority") != "NONE"
        or value.get("live_permission") is not False
        or (external and value.get("external_broker_mutation_allowed") is not False)
    ):
        raise AiDrainQuoteError(f"{label} paper authority is invalid")


def _identity(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise AiDrainQuoteError(f"{field} is invalid")
    return value


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise AiDrainQuoteError(f"{field} is invalid")
    return value


def _exact_int(
    value: object,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise AiDrainQuoteError(f"{field} is invalid")
    return value


def _utc(value: datetime | str | object, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AiDrainQuoteError(f"{field} is invalid") from exc
    else:
        raise AiDrainQuoteError(f"{field} is invalid")
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise AiDrainQuoteError(f"{field} is not UTC")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_json_line(raw: bytes, label: str) -> Any:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AiDrainQuoteError(f"{label} JSON is invalid") from exc
    if _canonical_bytes(value) != raw:
        raise AiDrainQuoteError(f"{label} JSON is noncanonical")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _require_market_open(value: datetime, field: str) -> None:
    if not compute_market_status(value).is_fx_open:
        raise AiDrainQuoteMarketClosedError(f"{field} is outside the FX trading week")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
